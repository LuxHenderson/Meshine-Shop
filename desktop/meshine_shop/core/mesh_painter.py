"""
Mesh painting backend — BVH ray casting, PIL texture buffer, region flood fill.

This module owns the mesh geometry and the mutable albedo texture. It provides
the bridge between 3D viewport interactions (mouse clicks in screen space) and
2D texture edits (pixel modifications in the albedo PNG). No Qt, no OpenGL —
this is pure CPU-side geometry and image processing. The ViewportWidget calls
into this module and then uploads the modified texture data to the GPU.

High-level flow:
    1. Load meshed_uv.obj via trimesh (preserves UV seams)
    2. Build BVH with trimesh's built-in RayMeshIntersector
    3. Load albedo.png into a mutable PIL RGBA buffer
    4. Precompute face adjacency for BFS region fill
    5. Expose paint_brush(), fill_region(), ray_cast() to the viewport
    6. Track a "dirty rect" so the viewport only reuploads changed pixels
    7. save_albedo() writes the edited buffer back to disk
"""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)


class MeshPainter:
    """
    Owns the UV-unwrapped mesh, its BVH, and the mutable albedo texture.

    This class is the sole owner of the PIL texture buffer. The ViewportWidget
    calls ray_cast() to find where the user clicked on the mesh, then calls
    paint_brush() or fill_region() to modify the texture. After painting, the
    viewport reads get_dirty_patch() and uploads only the changed rectangle to
    the GPU via glTexSubImage2D, keeping the texture refresh fast even for
    large (4K) textures.

    Parameters
    ----------
    mesh_path : Path
        Path to the UV-unwrapped OBJ file (meshed_uv.obj). Must have UV coords.
    textures_dir : Path
        Directory containing albedo.png (and optionally normal/roughness/etc.).
    """

    def __init__(self, mesh_path: Path, textures_dir: Path) -> None:
        # ------------------------------------------------------------------ #
        # Load mesh — process=False preserves UV seams exactly as authored     #
        # (trimesh's default merging can weld UV-seam vertices together, which  #
        # breaks per-face UV lookups when we need to paint into UV space)       #
        # ------------------------------------------------------------------ #
        log.info("MeshPainter: loading %s", mesh_path)
        self._mesh: trimesh.Trimesh = trimesh.load(str(mesh_path), process=False)

        if not isinstance(self._mesh, trimesh.Trimesh):
            # trimesh.load can return a Scene if the file has groups — unwrap it
            if hasattr(self._mesh, "dump"):
                meshes = self._mesh.dump()
                self._mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError(f"Could not load a single Trimesh from {mesh_path}")

        log.info(
            "MeshPainter: mesh has %d vertices, %d faces",
            len(self._mesh.vertices),
            len(self._mesh.faces),
        )

        # ------------------------------------------------------------------ #
        # Extract UV coordinates                                               #
        # trimesh stores per-face-vertex UVs in the visual material.           #
        # If the OBJ has a MTL/texture, UVs live in mesh.visual.uv.           #
        # We store them as (n_vertices, 2) float32, indexed by face vertex.    #
        # ------------------------------------------------------------------ #
        self._uvs: np.ndarray | None = self._extract_uvs()

        # ------------------------------------------------------------------ #
        # Build BVH for fast ray–triangle intersection                         #
        # trimesh.RayMeshIntersector uses a CPU BVH (Embree if installed,      #
        # otherwise pure Python). We cache it once on load.                   #
        # ------------------------------------------------------------------ #
        self._ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            self._mesh
        ) if trimesh.ray.has_embree else trimesh.ray.ray_triangle.RayMeshIntersector(
            self._mesh
        )

        log.info(
            "MeshPainter: BVH built (Embree=%s)", trimesh.ray.has_embree
        )

        # ------------------------------------------------------------------ #
        # Load albedo texture into a mutable PIL RGBA buffer                   #
        # We always work in RGBA so the brush stroke alpha blending works      #
        # correctly. On save we convert back to whatever format the file uses. #
        # ------------------------------------------------------------------ #
        albedo_path = textures_dir / "albedo.png"
        if albedo_path.exists():
            self._albedo_img: Image.Image = Image.open(albedo_path).convert("RGBA")
            log.info(
                "MeshPainter: albedo loaded %dx%d", *self._albedo_img.size
            )
        else:
            # No baked texture — create a blank 1024×1024 gray canvas so the
            # painter still works (user can paint from scratch)
            log.warning(
                "MeshPainter: albedo.png not found at %s — creating blank 1024×1024",
                albedo_path,
            )
            self._albedo_img = Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))

        self._albedo_path: Path = albedo_path
        self._tex_w, self._tex_h = self._albedo_img.size

        # ------------------------------------------------------------------ #
        # Dirty-rect tracking — accumulate modified region per paint call      #
        # The viewport polls get_dirty_rect() each frame and uploads the       #
        # minimum rectangle to the GPU via glTexSubImage2D.                   #
        # ------------------------------------------------------------------ #
        self._dirty: tuple[int, int, int, int] | None = None  # (x, y, x2, y2)

        # ------------------------------------------------------------------ #
        # Precompute face adjacency for BFS flood fill                         #
        # Two faces are adjacent if they share an edge (two vertices).         #
        # We build this once as a dict {face_idx: [neighbor_face_idx, ...]}   #
        # so fill_region() can do BFS without recomputing topology each call. #
        # ------------------------------------------------------------------ #
        self._face_adj: dict[int, list[int]] = self._build_face_adjacency()
        log.info("MeshPainter: face adjacency built")

        # Precompute vertex neighbor lists for sculpt Smooth brush.
        # List of numpy arrays: _vertex_neighbors[i] = array of vertex indices
        # adjacent to vertex i via shared edges.
        self._vertex_neighbors: list = self._build_vertex_neighbors()

        # Precompute UV-seam partner groups. OBJ UV seams duplicate vertices:
        # same 3D position, different UV index. Sculpting one side without the
        # other tears the mesh open. We group coincident vertices so sculpt()
        # can propagate every displacement to all partners at the same position.
        self._seam_partners: list = self._build_seam_partners()

        # Set True by sculpt() when vertex positions have changed.
        # Viewport polls is_sculpt_dirty() to decide whether to re-upload the VBO.
        self._sculpt_dirty: bool = False

    # ------------------------------------------------------------------ #
    # UV extraction                                                        #
    # ------------------------------------------------------------------ #

    def _extract_uvs(self) -> np.ndarray | None:
        """
        Pull per-face-vertex UV coordinates out of the trimesh visual.

        Returns (n_vertices, 2) float32 array indexed by vertex index, or None
        if the mesh has no UV data. When UVs are stored per-face-vertex (the
        common OBJ case), we return them in trimesh's flattened vertex order.
        """
        visual = self._mesh.visual
        # TextureVisuals carries UVs; ColorVisuals does not
        if not hasattr(visual, "uv") or visual.uv is None:
            log.warning("MeshPainter: mesh has no UV coordinates — painting disabled")
            return None
        uvs = np.asarray(visual.uv, dtype=np.float32)
        # UV origin: OBJ stores V=0 at bottom; PIL has Y=0 at top → flip V
        uvs[:, 1] = 1.0 - uvs[:, 1]
        return uvs

    # ------------------------------------------------------------------ #
    # Face adjacency                                                       #
    # ------------------------------------------------------------------ #

    def _build_face_adjacency(self) -> dict[int, list[int]]:
        """
        Build a face→[neighbor faces] adjacency dict using edge sharing.

        Two faces share an edge when they both contain two of the same
        vertex indices. trimesh.graph.face_adjacency gives us this as an
        (N, 2) array of adjacent face-index pairs.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(len(self._mesh.faces))}
        try:
            # trimesh.graph.face_adjacency returns (N, 2) pairs of adjacent faces
            pairs = trimesh.graph.face_adjacency(self._mesh.faces)
            for a, b in pairs:
                adj[int(a)].append(int(b))
                adj[int(b)].append(int(a))
        except Exception:
            log.warning("MeshPainter: could not build face adjacency — region fill disabled")
        return adj

    def _build_vertex_neighbors(self) -> list:
        """
        Build per-vertex neighbor vertex index lists from face topology.

        Returns a list where index i is a numpy array of vertex indices
        adjacent to vertex i via a shared face edge. Used by the Smooth
        sculpt brush to compute Laplacian averages.

        Vectorized construction: stacks all directed edge pairs, sorts by
        source vertex using searchsorted boundaries, then deduplicates each
        vertex's neighbor slice with np.unique.
        """
        faces = self._mesh.faces  # (F, 3) int
        n_verts = len(self._mesh.vertices)

        # Build all directed edges from all three edge pairs per face:
        # (a→b, b→a), (a→c, c→a), (b→c, c→b) — gives all adjacencies
        edges = np.vstack([
            faces[:, [0, 1]], faces[:, [1, 0]],
            faces[:, [0, 2]], faces[:, [2, 0]],
            faces[:, [1, 2]], faces[:, [2, 1]],
        ])  # (6F, 2)

        # Sort edges by source vertex so we can slice per-vertex with searchsorted
        src = edges[:, 0]
        dst = edges[:, 1]
        order = np.argsort(src, kind="stable")
        src_sorted = src[order]
        dst_sorted = dst[order]

        # Compute slice boundaries: boundaries[v] = first edge index with src == v
        boundaries = np.searchsorted(src_sorted, np.arange(n_verts + 1))

        neighbors = []
        for v in range(n_verts):
            s, e = int(boundaries[v]), int(boundaries[v + 1])
            nbrs = np.unique(dst_sorted[s:e])
            neighbors.append(nbrs)

        return neighbors

    def _build_seam_partners(self) -> list:
        """
        Group vertices that share the same 3D position (UV-seam duplicates).

        OBJ files split vertices at UV seams, creating duplicate vertices: same
        XYZ position but different UV coordinate and vertex index. If sculpt moves
        one side of a seam without the other, the mesh tears open along the seam.

        Returns a list of lists: seam_partners[i] contains the vertex indices of
        every other vertex that occupies the same 3D position as vertex i.
        Most vertices have an empty partner list (they are interior, not on a seam).

        Uses integer-rounded positions (6 decimal places) as hash keys, which is
        robust against floating-point loading noise while tight enough that truly
        distinct vertices are never accidentally grouped together.
        """
        from collections import defaultdict

        verts = np.asarray(self._mesh.vertices, dtype=np.float64)
        n = len(verts)

        # Round each coordinate to 6 dp so float-load noise doesn't prevent grouping
        rounded = np.round(verts, decimals=6)
        groups: dict = defaultdict(list)
        for i in range(n):
            key = (rounded[i, 0], rounded[i, 1], rounded[i, 2])
            groups[key].append(i)

        partners: list = [[] for _ in range(n)]
        seam_verts = 0
        for indices in groups.values():
            if len(indices) > 1:
                seam_verts += len(indices)
                for i in indices:
                    partners[i] = [j for j in indices if j != i]

        log.info("MeshPainter: %d seam-duplicate vertices found", seam_verts)
        return partners

    # ------------------------------------------------------------------ #
    # Ray casting                                                          #
    # ------------------------------------------------------------------ #

    def ray_cast(
        self, ray_origin: np.ndarray, ray_dir: np.ndarray
    ) -> tuple[int, np.ndarray] | None:
        """
        Cast a ray into the mesh and return the closest hit.

        Parameters
        ----------
        ray_origin : (3,) float array
            World-space origin of the ray (camera position).
        ray_dir : (3,) float array
            World-space unit direction of the ray.

        Returns
        -------
        (face_index, uv_coord) or None if no intersection.
        uv_coord is a (2,) float array in [0,1]² (already V-flipped for PIL).
        """
        if self._uvs is None:
            return None

        origins = ray_origin[np.newaxis, :]  # (1, 3)
        dirs = ray_dir[np.newaxis, :]        # (1, 3)

        # intersects_location returns (locations, index_ray, index_tri).
        # multiple_hits=True is required to guarantee we can select the
        # CLOSEST hit — trimesh's BVH does not guarantee ordering with
        # multiple_hits=False and may return a back face instead of the
        # nearest front face.
        locations, index_ray, index_tri = self._ray_intersector.intersects_location(
            origins, dirs, multiple_hits=True
        )

        if len(locations) == 0:
            return None  # Ray missed the mesh

        # Select the hit closest to the ray origin (front-facing surface)
        distances = np.linalg.norm(locations - ray_origin, axis=1)
        closest = int(np.argmin(distances))
        hit_pos = locations[closest]
        face_idx = int(index_tri[closest])

        # Compute UV via barycentric interpolation over the face's UV triangle
        uv_coord = self._barycentric_uv(face_idx, hit_pos)
        return face_idx, uv_coord

    def _barycentric_uv(self, face_idx: int, hit_pos: np.ndarray) -> np.ndarray:
        """
        Interpolate UV coordinates at a hit point using barycentric weights.

        The OBJ vertex UVs are stored per-vertex in trimesh. For a face with
        vertex indices [i0, i1, i2], the UV at the hit is the barycentric
        blend of uvs[i0], uvs[i1], uvs[i2].
        """
        face = self._mesh.faces[face_idx]  # [i0, i1, i2]
        v0 = self._mesh.vertices[face[0]]
        v1 = self._mesh.vertices[face[1]]
        v2 = self._mesh.vertices[face[2]]

        # Compute barycentric coordinates with trimesh utility
        bary = trimesh.triangles.points_to_barycentric(
            np.array([[v0, v1, v2]]),
            np.array([hit_pos]),
        )[0]  # (3,)

        uv0 = self._uvs[face[0]]
        uv1 = self._uvs[face[1]]
        uv2 = self._uvs[face[2]]

        uv = bary[0] * uv0 + bary[1] * uv1 + bary[2] * uv2
        return uv.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Brush painting                                                       #
    # ------------------------------------------------------------------ #

    def paint_brush(
        self,
        uv: np.ndarray,
        color: tuple[int, int, int],
        radius_px: int,
        opacity: float,
    ) -> None:
        """
        Paint a circular brush stroke at the given UV coordinate.

        Parameters
        ----------
        uv : (2,) float array in [0,1]²
            Texture-space hit point (V already flipped to PIL convention).
        color : (R, G, B) ints in 0–255
        radius_px : int
            Brush radius in pixels.
        opacity : float in 0.0–1.0
            Stroke alpha — 1.0 is fully opaque.
        """
        # UV [0,1] → pixel coords in texture image
        px = int(uv[0] * self._tex_w)
        py = int(uv[1] * self._tex_h)

        # Draw the brush ellipse onto a temporary RGBA layer so we can
        # alpha-composite it over the existing texture
        stroke_layer = Image.new("RGBA", self._albedo_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(stroke_layer)

        alpha = int(opacity * 255)
        fill_color = (color[0], color[1], color[2], alpha)

        x0, y0 = px - radius_px, py - radius_px
        x1, y1 = px + radius_px, py + radius_px
        draw.ellipse([x0, y0, x1, y1], fill=fill_color)

        # Alpha-composite the stroke over the existing buffer
        self._albedo_img = Image.alpha_composite(self._albedo_img, stroke_layer)

        # Expand dirty rect to cover this stroke
        self._expand_dirty(
            max(0, x0), max(0, y0),
            min(self._tex_w, x1 + 1), min(self._tex_h, y1 + 1)
        )

    # ------------------------------------------------------------------ #
    # Region flood fill                                                    #
    # ------------------------------------------------------------------ #

    def fill_region(
        self, seed_face: int, color: tuple[int, int, int]
    ) -> None:
        """
        Flood-fill a contiguous mesh region with a solid color.

        Starting from seed_face, BFS expands through the precomputed face
        adjacency graph to find all connected faces. Each face's UV triangle
        is then rasterized into the PIL texture buffer with ImageDraw.polygon().

        Parameters
        ----------
        seed_face : int
            Starting face index (from ray_cast hit).
        color : (R, G, B) ints in 0–255
        """
        if self._uvs is None:
            return

        # BFS across face adjacency to collect the connected region
        visited: set[int] = set()
        queue: deque[int] = deque([seed_face])
        visited.add(seed_face)

        while queue:
            fi = queue.popleft()
            for nb in self._face_adj.get(fi, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        # Rasterize each face's UV triangle into the albedo buffer
        fill_color_rgba = (color[0], color[1], color[2], 255)
        draw = ImageDraw.Draw(self._albedo_img)

        dirty_x0, dirty_y0 = self._tex_w, self._tex_h
        dirty_x1, dirty_y1 = 0, 0

        for fi in visited:
            face = self._mesh.faces[fi]  # [i0, i1, i2]
            # Convert UV coords to pixel coords for the three vertices
            pts = []
            for vi in face:
                u, v = self._uvs[vi]
                x = int(u * self._tex_w)
                y = int(v * self._tex_h)
                pts.append((x, y))
                dirty_x0 = min(dirty_x0, x)
                dirty_y0 = min(dirty_y0, y)
                dirty_x1 = max(dirty_x1, x + 1)
                dirty_y1 = max(dirty_y1, y + 1)

            draw.polygon(pts, fill=fill_color_rgba)

        # Clamp dirty rect to texture bounds and expand the tracker
        dirty_x0 = max(0, dirty_x0)
        dirty_y0 = max(0, dirty_y0)
        dirty_x1 = min(self._tex_w, dirty_x1)
        dirty_y1 = min(self._tex_h, dirty_y1)
        if dirty_x1 > dirty_x0 and dirty_y1 > dirty_y0:
            self._expand_dirty(dirty_x0, dirty_y0, dirty_x1, dirty_y1)

    def sculpt(
        self,
        hit_point: np.ndarray,
        mode: str,
        radius: float,
        strength: float,
    ) -> None:
        """
        Apply a sculpt brush at a 3D hit point on the mesh surface.

        Finds all vertices within `radius` world units of `hit_point`, applies
        a cosine-squared falloff weight (1 at center, 0 at edge, C1 continuous),
        then deforms them according to `mode`.

        Parameters
        ----------
        hit_point : (3,) float array
            Model-space position of the ray–mesh intersection (from ray_cast_world_pos).
        mode : str
            One of "inflate", "deflate", "smooth", "flatten".
        radius : float
            World-space brush radius in mesh units.
        strength : float
            Maximum displacement per call (scaled by falloff weight).
        """
        # Work in float64 for precision; trimesh stores vertices as float64
        verts = np.asarray(self._mesh.vertices, dtype=np.float64)

        # --- Vertex selection: all vertices within radius ---
        diffs = verts - hit_point.astype(np.float64)
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists < radius
        if not mask.any():
            return

        idxs = np.where(mask)[0]
        d_norm = np.clip(dists[idxs] / radius, 0.0, 1.0)

        # Cosine-squared falloff: smooth curve, 1 at center → 0 at edge,
        # C1 continuous so there's no sharp boundary artefact on the mesh.
        weights = np.cos(d_norm * (math.pi / 2.0)) ** 2  # (K,)

        # --- Copy vertices so we can write back cleanly ---
        new_verts = verts.copy()

        if mode in ("inflate", "deflate"):
            # Push / pull vertices along their vertex normals.
            # vertex_normals is a cached trimesh property; we read it fresh each
            # call so it reflects any deformation from the previous sculpt stroke.
            sign = 1.0 if mode == "inflate" else -1.0
            norms = np.asarray(self._mesh.vertex_normals, dtype=np.float64)
            # Scale each normal by its falloff weight and strength, apply in-place
            new_verts[idxs] += norms[idxs] * (weights * strength * sign)[:, np.newaxis]

        elif mode == "smooth":
            # Laplacian smooth: move each affected vertex toward the average
            # position of its immediate neighbors, weighted by falloff.
            for i, v_idx in enumerate(idxs):
                nbrs = self._vertex_neighbors[v_idx]
                if len(nbrs) == 0:
                    continue
                avg = np.mean(new_verts[nbrs], axis=0)
                # Blend toward avg — strength controls how far it moves per call
                new_verts[v_idx] += (avg - new_verts[v_idx]) * weights[i] * strength

        elif mode == "flatten":
            # Project affected vertices toward their shared best-fit plane.
            # PCA on the affected vertex cloud: the eigenvector corresponding
            # to the smallest eigenvalue is the plane normal (least variance
            # direction = direction perpendicular to the flat plane).
            w_sum = weights.sum()
            if w_sum < 1e-9:
                return

            # Weighted centroid of affected vertices
            centroid = np.average(verts[idxs], axis=0, weights=weights)

            # Weighted covariance matrix
            diff = verts[idxs] - centroid                          # (K, 3)
            cov = (diff * weights[:, np.newaxis]).T @ diff / w_sum # (3, 3)

            # Eigen-decompose; eigh returns ascending eigenvalues
            _, eigenvectors = np.linalg.eigh(cov)
            plane_normal = eigenvectors[:, 0]  # smallest eigenvalue = plane normal

            # Signed distance of each affected vertex from the plane
            dots = diff @ plane_normal  # (K,) scalar signed distances

            # Move each vertex toward the plane proportional to falloff × strength
            new_verts[idxs] -= (
                plane_normal[np.newaxis, :] *
                (dots * weights * strength)[:, np.newaxis]
            )

        # --- Seam reconciliation (prevents mesh splitting) ---
        #
        # UV seams duplicate vertices: same 3D position, different UV index.
        # Two problems arise without this step:
        #
        #   1. A seam partner OUTSIDE the brush radius receives no displacement,
        #      so its side of the seam stays put while the other moves → tear.
        #
        #   2. Both seam partners are INSIDE the brush (they always are — they sit
        #      at the same position so they share the same distance to hit_point).
        #      Each gets an independent displacement (different vertex normals for
        #      inflate/deflate, different neighbor sets for smooth) → divergence.
        #
        # Fix: after all per-mode displacements are computed, build the full
        # (N,3) delta array, then for every seam group touching the brush:
        #   • Average the deltas of all group members (inside AND outside brush).
        #   • Write that averaged delta back to every member.
        # This locks all coincident vertices to an identical displacement,
        # keeping the mesh watertight regardless of brush size or stroke magnitude.
        delta = new_verts - verts   # (N, 3), zero for vertices outside the brush

        seen_groups: set = set()
        for v_idx in idxs:
            partners = self._seam_partners[v_idx]
            if not partners:
                continue
            # Use the smallest index in the group as a dedup key
            group = [v_idx] + partners
            group_key = min(group)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)

            # Average all members' deltas (partners outside brush have delta=0,
            # which is correct — they contribute zero and pull the average
            # slightly inward, but the result stays watertight).
            avg_delta = np.mean([delta[g] for g in group], axis=0)
            for g in group:
                delta[g] = avg_delta

        # Assign back through trimesh's setter to invalidate cached normals,
        # bounds, and other derived properties that depend on vertex positions.
        self._mesh.vertices = verts + delta
        self._sculpt_dirty = True

    def is_sculpt_dirty(self) -> bool:
        """Return True if sculpt() has modified geometry since the last call."""
        dirty = self._sculpt_dirty
        self._sculpt_dirty = False
        return dirty

    # ------------------------------------------------------------------ #
    # Dirty rect tracking                                                  #
    # ------------------------------------------------------------------ #

    def _expand_dirty(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Union the given rectangle into the current dirty rect."""
        if self._dirty is None:
            self._dirty = (x0, y0, x1, y1)
        else:
            dx0, dy0, dx1, dy1 = self._dirty
            self._dirty = (
                min(dx0, x0), min(dy0, y0),
                max(dx1, x1), max(dy1, y1),
            )

    # ------------------------------------------------------------------ #
    # GPU data accessors                                                   #
    # ------------------------------------------------------------------ #

    def get_texture_rgba(self) -> bytes:
        """
        Return the full albedo texture as raw RGBA bytes.

        Called once on initial mesh load to upload the texture to the GPU
        via glTexImage2D. After that, only dirty patches are reuploaded.
        """
        return self._albedo_img.tobytes("raw", "RGBA")

    def get_texture_size(self) -> tuple[int, int]:
        """Return (width, height) of the albedo texture."""
        return self._tex_w, self._tex_h

    def get_dirty_rect(self) -> tuple[int, int, int, int] | None:
        """
        Return and clear the accumulated dirty rect.

        Returns (x, y, x2, y2) pixel coords of the changed rectangle since the
        last call, or None if nothing changed. Clears the dirty flag so
        subsequent calls return None until another paint operation occurs.
        """
        rect = self._dirty
        self._dirty = None
        return rect

    def get_dirty_patch(self, x: int, y: int, x2: int, y2: int) -> bytes:
        """
        Return raw RGBA bytes for the cropped dirty rectangle.

        Used by the viewport to upload only the changed region to the GPU
        via glTexSubImage2D, avoiding a full texture reupload each frame.

        Parameters
        ----------
        x, y : top-left pixel (inclusive)
        x2, y2 : bottom-right pixel (exclusive)
        """
        patch = self._albedo_img.crop((x, y, x2, y2))
        return patch.tobytes("raw", "RGBA")

    # ------------------------------------------------------------------ #
    # Mesh geometry accessors                                              #
    # ------------------------------------------------------------------ #

    def get_render_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return per-vertex arrays for GPU upload.

        Returns
        -------
        vertices : (N, 3) float32 — XYZ positions
        normals  : (N, 3) float32 — vertex normals
        uvs      : (N, 2) float32 — UV coordinates (V-flipped for OpenGL)
        indices  : (M, 3) uint32  — triangle face indices
        """
        verts = np.asarray(self._mesh.vertices, dtype=np.float32)
        # trimesh vertex normals are computed lazily and cached
        norms = np.asarray(self._mesh.vertex_normals, dtype=np.float32)
        uvs = self._uvs if self._uvs is not None else np.zeros(
            (len(verts), 2), dtype=np.float32
        )
        faces = np.asarray(self._mesh.faces, dtype=np.uint32)
        return verts, norms, uvs, faces

    def ray_cast_world_pos(
        self, ray_origin: np.ndarray, ray_dir: np.ndarray
    ) -> np.ndarray | None:
        """
        Cast a ray and return the world-space hit position, or None on miss.

        Unlike ray_cast(), this skips UV computation and only returns the 3D
        intersection point. Used by the viewport to determine the orbit pivot
        when the user clicks on the mesh surface.
        """
        if self._uvs is None:
            return None
        origins = ray_origin[np.newaxis, :]
        dirs    = ray_dir[np.newaxis, :]
        # Use multiple_hits=True so we can pick the closest hit explicitly —
        # multiple_hits=False does not guarantee the nearest surface is returned.
        locations, _, _ = self._ray_intersector.intersects_location(
            origins, dirs, multiple_hits=True
        )
        if len(locations) == 0:
            return None
        distances = np.linalg.norm(locations - ray_origin, axis=1)
        return locations[int(np.argmin(distances))].astype(np.float64)

    def get_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (bbox_min, bbox_max) as (3,) float arrays.

        Used by ViewportCamera.frame_mesh() to position the camera so the
        entire mesh is visible when the viewport first loads.
        """
        bbox_min = self._mesh.bounds[0].astype(np.float32)
        bbox_max = self._mesh.bounds[1].astype(np.float32)
        return bbox_min, bbox_max

    # ------------------------------------------------------------------ #
    # Undo / Redo snapshot support                                        #
    # ------------------------------------------------------------------ #

    def get_snapshot_data(
        self, geometry: bool = True
    ) -> tuple[
        "np.ndarray | None",
        "np.ndarray | None",
        "np.ndarray | None",
        "Image.Image",
    ]:
        """
        Return deep copies of the current editor state for a history snapshot.

        Called by EditHistory.push_snapshot() before each destructive operation.

        Parameters
        ----------
        geometry : bool
            True  → copy vertices, faces, and normals (needed for sculpt/mesh ops)
            False → skip geometry copies (paint-only operations don't change it)

        Returns
        -------
        (vertices_copy, faces_copy, normals_copy, albedo_copy)
        Geometry arrays are None when geometry=False.
        """
        if geometry:
            # Deep copy all geometry arrays so the snapshot is fully independent
            # of any subsequent in-place modifications to the mesh.
            verts = np.array(self._mesh.vertices, dtype=np.float32, copy=True)
            faces = np.array(self._mesh.faces, dtype=np.uint32, copy=True)
            # vertex_normals is a lazily-computed trimesh property — force
            # evaluation before copying so we capture the current normals.
            norms = np.array(self._mesh.vertex_normals, dtype=np.float32, copy=True)
        else:
            verts = faces = norms = None

        # Always copy the PIL image — it is mutable and paint strokes modify it
        # in place, so a reference copy would be invalidated immediately.
        albedo = self._albedo_img.copy()

        return verts, faces, norms, albedo

    def restore_snapshot(
        self,
        vertices: "np.ndarray | None",
        faces: "np.ndarray | None",
        normals: "np.ndarray | None",
        albedo: "Image.Image",
    ) -> None:
        """
        Overwrite the current editor state with a previously captured snapshot.

        Called by ViewportWidget._restore_from_snapshot() after popping an
        undo or redo entry. After this call the caller must re-upload the VBO
        and texture to the GPU to reflect the restored state visually.

        Parameters
        ----------
        vertices, faces, normals : arrays or None
            When not None, the mesh geometry is replaced and the BVH + face
            adjacency are rebuilt from scratch to match the new topology.
        albedo : PIL Image
            The texture buffer to restore. Always provided.
        """
        # Restore geometry if this was a geometry-including snapshot
        if vertices is not None and faces is not None:
            # Rebuild the trimesh from the snapshot arrays so all internal
            # caches (normals, bounds, etc.) reflect the restored geometry.
            self._mesh = trimesh.Trimesh(
                vertices=vertices.astype(np.float64),
                faces=faces,
                process=False,
            )
            # Re-extract UV coords from the new mesh. The reconstructed Trimesh
            # has no TextureVisuals (we only snapshot positions+faces, not the
            # visual), so _extract_uvs() will return None. In that case, keep
            # the existing _uvs — sculpt never changes UV layout, so the stored
            # array is still correctly aligned to the (unchanged) face topology.
            new_uvs = self._extract_uvs()
            if new_uvs is not None:
                self._uvs = new_uvs
            # else: retain self._uvs as-is (valid for sculpt-only restores)

            # Rebuild BVH so ray_cast() works correctly on restored geometry
            self._rebuild_bvh()

            # Rebuild face adjacency for BFS region fill
            self._face_adj = self._build_face_adjacency()

            # Rebuild vertex neighbors for the Smooth sculpt brush
            self._vertex_neighbors = self._build_vertex_neighbors()

            # Rebuild seam partner groups (topology is identical for sculpt
            # restores, but rebuild anyway to stay correct after mesh ops)
            self._seam_partners = self._build_seam_partners()

        # Restore texture — replace the PIL buffer with the snapshot copy
        self._albedo_img = albedo.copy()
        self._tex_w, self._tex_h = self._albedo_img.size

        # Mark texture as fully dirty so the full texture is reuploaded to the
        # GPU (the caller reads get_dirty_rect() or calls get_texture_rgba()).
        self._dirty = (0, 0, self._tex_w, self._tex_h)

    def _rebuild_bvh(self) -> None:
        """
        Rebuild the ray–triangle BVH after geometry changes.

        Called after restore_snapshot() and after any sculpt or mesh operation
        that modifies vertex positions or topology. The old BVH is invalidated
        whenever self._mesh changes.
        """
        self._ray_intersector = (
            trimesh.ray.ray_pyembree.RayMeshIntersector(self._mesh)
            if trimesh.ray.has_embree
            else trimesh.ray.ray_triangle.RayMeshIntersector(self._mesh)
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_albedo(self) -> None:
        """
        Write the modified PIL buffer back to albedo.png on disk.

        Called when the user navigates away from the Viewport or explicitly
        saves. The file is written as RGBA PNG to preserve any alpha channel
        that downstream tools (Blender, Unreal, Unity) might use.
        """
        self._albedo_img.save(str(self._albedo_path), format="PNG")
        log.info("MeshPainter: albedo saved to %s", self._albedo_path)
