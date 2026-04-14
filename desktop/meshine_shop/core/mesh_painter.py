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
from PIL import Image, ImageDraw, ImageFilter

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

        # Keep a pristine copy of the original baked texture so the erase tool
        # can restore original pixels rather than just painting over them.
        self._albedo_orig: Image.Image = self._albedo_img.copy()

        # ------------------------------------------------------------------ #
        # Layer texture compositing                                            #
        # _albedo_base = original baked texture + brush/fill/erase strokes.   #
        # _albedo_img  = composite display = _albedo_base + visible layer      #
        #                textures on top. Rebuilt by _rebuild_composite()      #
        #                whenever a layer texture changes or is toggled.       #
        # Keeping them separate lets eye-toggle show/hide a layer's texture    #
        # without losing brush work painted before or after the projection.    #
        # ------------------------------------------------------------------ #
        self._albedo_base: Image.Image = self._albedo_img.copy()

        # Per-layer projected textures: layer_id → full-size RGBA Image.
        # Populated by project_texture(); cleared by remove_layer_texture().
        self._layer_textures: dict[int, Image.Image] = {}

        # Per-layer visibility for the projected texture (mirrors the eye toggle).
        # When False the layer's texture is excluded from _rebuild_composite().
        self._layer_visible: dict[int, bool] = {}

        # Per-layer planar projection parameters for shader-based display.
        # Populated by setup_layer_projection(); read by ViewportWidget to
        # set GLSL uniforms. Does NOT modify _albedo_img — the shader renders
        # the projection entirely on the GPU without touching the PIL buffer.
        self._layer_projections: dict[int, dict] = {}

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
        # alpha-composite it over the base texture buffer (not the composite
        # display — brush strokes live in _albedo_base, not _albedo_img).
        stroke_layer = Image.new("RGBA", self._albedo_base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(stroke_layer)

        alpha = int(opacity * 255)
        fill_color = (color[0], color[1], color[2], alpha)

        x0, y0 = px - radius_px, py - radius_px
        x1, y1 = px + radius_px, py + radius_px
        draw.ellipse([x0, y0, x1, y1], fill=fill_color)

        # Write stroke into _albedo_base (layer textures excluded from base)
        self._albedo_base = Image.alpha_composite(self._albedo_base, stroke_layer)

        if self._layer_textures:
            # Has projected layer textures: rebuild full composite so layers
            # remain correctly composited on top of the updated base.
            self._rebuild_composite()
        else:
            # Fast path — no layer textures, mirror stroke directly into display
            # buffer and use fine-grained dirty-rect tracking for GPU efficiency.
            self._albedo_img = Image.alpha_composite(self._albedo_img, stroke_layer)
            self._expand_dirty(
                max(0, x0), max(0, y0),
                min(self._tex_w, x1 + 1), min(self._tex_h, y1 + 1)
            )

    def erase_brush(
        self,
        uv: np.ndarray,
        radius_px: int,
        opacity: float,
    ) -> None:
        """
        Restore original baked texture pixels within a circular brush area.

        Rather than painting a new color, this samples from _albedo_orig (the
        pristine texture loaded at startup) and alpha-composites it back over
        the current working buffer. At opacity=1.0 the original texture is
        fully restored; lower opacity blends gradually.

        Parameters
        ----------
        uv : (2,) float array in [0,1]²
            Texture-space hit point.
        radius_px : int
            Brush radius in pixels.
        opacity : float in 0.0–1.0
            Erase strength — 1.0 fully restores original pixels.
        """
        px = int(uv[0] * self._tex_w)
        py = int(uv[1] * self._tex_h)

        x0 = max(0, px - radius_px)
        y0 = max(0, py - radius_px)
        x1 = min(self._tex_w, px + radius_px + 1)
        y1 = min(self._tex_h, py + radius_px + 1)

        if x1 <= x0 or y1 <= y0:
            return

        # Build a circular mask for the brush area
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        draw = ImageDraw.Draw(mask)
        # Circle is drawn relative to the crop region
        cx, cy = px - x0, py - y0
        draw.ellipse(
            [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
            fill=int(opacity * 255),
        )

        # Restore original pixels into _albedo_base (the brush-stroke buffer).
        # Layer textures composited on top are NOT affected by erase — hiding a
        # layer first lets the user see the erased base pixels underneath.
        orig_patch = self._albedo_orig.crop((x0, y0, x1, y1))
        curr_patch = self._albedo_base.crop((x0, y0, x1, y1))
        curr_patch.paste(orig_patch, mask=mask)
        self._albedo_base.paste(curr_patch, (x0, y0))

        if self._layer_textures:
            self._rebuild_composite()
        else:
            # Fast path: mirror directly to display buffer
            self._albedo_img.paste(curr_patch, (x0, y0))
            self._expand_dirty(x0, y0, x1, y1)

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

        # Rasterize each face's UV triangle into the base buffer.
        # Layer textures remain composited on top regardless.
        fill_color_rgba = (color[0], color[1], color[2], 255)
        draw = ImageDraw.Draw(self._albedo_base)

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

        # Clamp dirty rect to texture bounds
        dirty_x0 = max(0, dirty_x0)
        dirty_y0 = max(0, dirty_y0)
        dirty_x1 = min(self._tex_w, dirty_x1)
        dirty_y1 = min(self._tex_h, dirty_y1)

        if dirty_x1 > dirty_x0 and dirty_y1 > dirty_y0:
            if self._layer_textures:
                # Rebuild composite so layers stay on top of the filled base
                self._rebuild_composite()
            else:
                # Fast path: mirror fill directly to display buffer
                patch = self._albedo_base.crop((dirty_x0, dirty_y0, dirty_x1, dirty_y1))
                self._albedo_img.paste(patch, (dirty_x0, dirty_y0))
                self._expand_dirty(dirty_x0, dirty_y0, dirty_x1, dirty_y1)

    # ------------------------------------------------------------------ #
    # Texture projection                                                   #
    # ------------------------------------------------------------------ #

    def setup_layer_projection(
        self,
        layer_id: int,
        face_set: set,
        texture_img: "Image.Image",
        rotate_deg: float = 0.0,
        scale: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        opacity: float = 1.0,
    ) -> dict | None:
        """
        Compute and store planar projection parameters for a layer.

        Unlike the old UV-atlas baking approach, this method does NOT modify
        _albedo_img or _albedo_base. All display happens in the OpenGL
        projection shader pass in ViewportWidget.paintGL(). The UV atlas is
        only used at export time (future bake-on-save path).

        Algorithm:
            1. Average face normal → orthonormal frame (R=right, U=up, N=normal)
            2. Project selected vertices onto (R, U) plane → normalization bounds
            3. Pre-compute cos/sin for rotation uniform
            4. Store all shader uniforms in _layer_projections[layer_id]
            5. Store source PIL image in _layer_textures[layer_id] for undo tracking

        Returns the params dict so ViewportWidget can immediately build GPU resources.
        Returns None if face_set is empty or UVs not loaded.

        Parameters
        ----------
        layer_id    : The layer this projection belongs to.
        face_set    : Mesh face indices that define the selection.
        texture_img : Source texture to project (any PIL mode — converted to RGBA).
        rotate_deg  : Counter-clockwise rotation in degrees.
        scale       : Uniform texture scale (1.0 = fill selection).
        offset_x    : Horizontal shift (-1.0 to 1.0) in normalized planar space.
        offset_y    : Vertical shift (-1.0 to 1.0) in normalized planar space.
        opacity     : Overall blend opacity (0.0–1.0).
        """
        if self._uvs is None or not face_set:
            return None

        verts = np.asarray(self._mesh.vertices, dtype=np.float64)
        faces = np.asarray(self._mesh.faces,    dtype=np.int32)

        # ---------------------------------------------------------------- #
        # Step 1: Average face normal → build orthonormal projection frame. #
        # ---------------------------------------------------------------- #
        normals_acc = np.zeros(3, dtype=np.float64)
        for fi in face_set:
            f  = faces[fi]
            e1 = verts[f[1]] - verts[f[0]]
            e2 = verts[f[2]] - verts[f[0]]
            n  = np.cross(e1, e2)
            nn = np.linalg.norm(n)
            if nn > 1e-10:
                normals_acc += n / nn

        avg_n = normals_acc / (np.linalg.norm(normals_acc) + 1e-10)

        # Build stable right / up vectors perpendicular to N
        up_hint = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(avg_n, up_hint))) > 0.98:
            up_hint = np.array([1.0, 0.0, 0.0])
        R_vec = np.cross(avg_n, up_hint);  R_vec /= np.linalg.norm(R_vec)
        U_vec = np.cross(R_vec, avg_n);    U_vec /= np.linalg.norm(U_vec)
        # Negate U so "up" in world maps to "up" (low row index) in image
        U_vec = -U_vec

        # ---------------------------------------------------------------- #
        # Step 2: Project selected vertices onto (R, U) plane.             #
        # Normalization bounds ensure the texture fills the selection at   #
        # scale=1.0 — same convention the fragment shader uses.            #
        # ---------------------------------------------------------------- #
        sel_verts: set[int] = set()
        for fi in face_set:
            sel_verts.update(int(v) for v in faces[fi])

        r_coords = [float(np.dot(verts[vi], R_vec)) for vi in sel_verts]
        u_coords = [float(np.dot(verts[vi], U_vec)) for vi in sel_verts]

        r_min = min(r_coords);  r_max = max(r_coords)
        u_min = min(u_coords);  u_max = max(u_coords)
        r_range = max(r_max - r_min, 1e-9)
        u_range = max(u_max - u_min, 1e-9)

        # Centroid of selected vertices — used with avg_n to define the projection
        # half-space plane. Fragments behind this plane are discarded in the shader
        # so the texture never wraps to the back side of the mesh.
        sel_verts_arr = np.array([verts[vi] for vi in sel_verts], dtype=np.float64)
        centroid      = sel_verts_arr.mean(axis=0)
        plane_d       = float(np.dot(centroid, avg_n))

        # ---------------------------------------------------------------- #
        # Step 3: Pre-compute rotation trig and pack all shader uniforms.  #
        # ---------------------------------------------------------------- #
        cos_a = float(math.cos(math.radians(rotate_deg)))
        sin_a = float(math.sin(math.radians(rotate_deg)))

        params: dict = {
            "right":     R_vec.astype(np.float32),   # vec3 — horizontal axis
            "up":        U_vec.astype(np.float32),   # vec3 — vertical axis (negated)
            "normal":    avg_n.astype(np.float32),   # vec3 — projection plane normal (may need flip)
            "plane_d":   plane_d,                    # dot(centroid, normal) — half-space constant
            "centroid":  centroid.astype(np.float32),# world-space centroid — for normal orientation in viewport
            "r_min":     float(r_min),               # normalization: left edge
            "r_range":   float(r_range),             # normalization: width
            "u_min":     float(u_min),               # normalization: top edge
            "u_range":   float(u_range),             # normalization: height
            "scale":     max(float(scale), 1e-9),    # uniform scale
            "cos":       cos_a,                      # rotation cos (pre-computed)
            "sin":       sin_a,                      # rotation sin
            "offset_x":  float(offset_x),            # horizontal shift
            "offset_y":  float(offset_y),            # vertical shift
            "opacity":   float(opacity),             # blend opacity
            "face_set":  face_set,                   # kept so viewport rebuilds VAO on mesh reload
            "rotate_deg": rotate_deg,                # kept for re-projection on slider change
            "texture_img": texture_img.convert("RGBA"),  # source RGBA for GPU upload
        }

        # ---------------------------------------------------------------- #
        # Step 4: Store params and source texture.                         #
        # _layer_textures[layer_id] = source PIL image (for undo tracking  #
        # and future export-bake). _layer_projections[layer_id] = uniforms #
        # consumed by ViewportWidget to set shader state each frame.       #
        # ---------------------------------------------------------------- #
        self._layer_projections[layer_id] = params
        self._layer_visible[layer_id]     = True
        # Store source image in _layer_textures for snapshot undo tracking.
        # This is NOT a baked atlas image — it is just the raw source texture.
        self._layer_textures[layer_id] = texture_img.convert("RGBA")

        log.debug(
            "MeshPainter: setup_layer_projection layer=%d faces=%d",
            layer_id, len(face_set),
        )
        return params

    def _rebuild_composite(self) -> None:
        """
        Rebuild _albedo_img from _albedo_base.

        With shader-based projection, layer textures are rendered on the GPU and
        do NOT modify _albedo_img. This method is kept for brush/fill/erase ops
        that still write into _albedo_base and need the dirty rect refreshed.

        _layer_textures now stores raw source PIL images (not atlas-baked images),
        so compositing them would produce garbage — we only copy the base.
        """
        # Start from the brush-stroke base (no layer textures composited in)
        self._albedo_img = self._albedo_base.copy()
        # Mark full texture dirty so viewport reuploads on next frame
        self._dirty = (0, 0, self._tex_w, self._tex_h)

    def set_layer_texture_visible(self, layer_id: int, visible: bool) -> None:
        """
        Show or hide a layer's projected texture in the composite display.

        Called by ViewportWidget when the layers panel eye button is toggled.
        Rebuilds the composite immediately so the viewport reflects the change
        on the next frame without requiring another Apply.

        Parameters
        ----------
        layer_id : int
            The layer whose texture visibility is being toggled.
        visible : bool
            True = include in composite; False = exclude from composite.
        """
        self._layer_visible[layer_id] = visible
        # Update shader projection params visibility so the viewport render
        # pass skips invisible layers without needing a GPU resource teardown.
        if layer_id in self._layer_projections:
            self._layer_projections[layer_id]["visible"] = visible
        # No composite rebuild needed — shader handles projection display.
        log.debug("MeshPainter: layer %d visibility → %s", layer_id, visible)

    def remove_layer_texture(self, layer_id: int) -> None:
        """
        Remove a layer's projection — both source texture and shader params.

        Called by ViewportWidget.delete_layer() when the user deletes a layer.
        Clears both the source texture (used for undo tracking) and the shader
        projection parameters. The viewport render pass will no longer include
        this layer on the next frame.

        Parameters
        ----------
        layer_id : int
            The layer being deleted.
        """
        self._layer_textures.pop(layer_id, None)
        self._layer_projections.pop(layer_id, None)
        self._layer_visible.pop(layer_id, None)
        # No composite rebuild needed — shader handles display.
        log.debug("MeshPainter: removed projection for layer %d", layer_id)

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
            #
            # The strength slider is calibrated for inflate/deflate (0.001–0.05
            # world-unit displacement per call). For smooth, the effective
            # per-call movement is `strength × weight × |avg − pos|`. Because
            # |avg − pos| is near-zero on any mesh that isn't completely jagged,
            # the raw strength value produces microscopic displacements that look
            # like nothing is happening on drag. Scaling by ×20 maps the slider's
            # midpoint (~0.025) to a 50% blend per call — immediately visible and
            # still controllable from subtle (low end) to aggressive (high end).
            blend = min(strength * 20.0, 1.0)
            for i, v_idx in enumerate(idxs):
                nbrs = self._vertex_neighbors[v_idx]
                if len(nbrs) == 0:
                    continue
                avg = np.mean(new_verts[nbrs], axis=0)
                new_verts[v_idx] += (avg - new_verts[v_idx]) * weights[i] * blend

        elif mode == "flatten":
            # Project affected vertices toward their shared best-fit plane.
            # PCA on the affected vertex cloud: the eigenvector corresponding
            # to the smallest eigenvalue is the plane normal (least variance
            # direction = direction perpendicular to the flat plane).
            #
            # Same scaling issue as smooth: `dots` (distance from plane) is
            # tiny on any near-flat surface, so raw strength × dots approaches
            # zero and produces invisible drag results. Scale ×20 for the same
            # reason — maps slider midpoint to a 50% projection per call.
            w_sum = weights.sum()
            if w_sum < 1e-9:
                return

            blend = min(strength * 20.0, 1.0)

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

            # Move each vertex toward the plane proportional to falloff × blend
            new_verts[idxs] -= (
                plane_normal[np.newaxis, :] *
                (dots * weights * blend)[:, np.newaxis]
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
    ) -> tuple:
        """
        Return deep copies of the current editor state for a history snapshot.

        Called by EditHistory._make_snapshot() before each destructive operation.

        Parameters
        ----------
        geometry : bool
            True  → copy vertices, faces, normals, uvs + texture (sculpt / mesh ops)
            False → skip all geometry/UV copies (paint-only operations don't change them)

        Returns
        -------
        (vertices_copy, faces_copy, normals_copy, uvs_copy, albedo_copy)
        Geometry arrays (including uvs) are None when geometry=False.
        """
        if geometry:
            # Deep copy all geometry arrays so the snapshot is fully independent
            # of any subsequent in-place modifications to the mesh.
            verts = np.array(self._mesh.vertices, dtype=np.float32, copy=True)
            faces = np.array(self._mesh.faces, dtype=np.uint32, copy=True)
            # vertex_normals is a lazily-computed trimesh property — force
            # evaluation before copying so we capture the current normals.
            norms = np.array(self._mesh.vertex_normals, dtype=np.float32, copy=True)
            # Snapshot UVs: topology-changing mesh ops (decimate, fill
            # holes) update _uvs to match the new vertex layout. Without snapshotting
            # them here, undo would restore the old geometry with the new UV array,
            # causing texture coordinates to map to the wrong albedo regions.
            uvs = self._uvs.copy() if self._uvs is not None else None
        else:
            verts = faces = norms = uvs = None

        # Snapshot the BASE buffer (brush strokes only, no layer textures).
        # Undo restores this base, then rebuilds the composite from the
        # simultaneously-snapshotted layer_textures state.
        albedo = self._albedo_base.copy()

        # Copy per-layer texture and visibility state so undoing a texture
        # projection also removes it from the composite (not just the base).
        layer_textures = {lid: img.copy() for lid, img in self._layer_textures.items()}
        layer_visible  = dict(self._layer_visible)

        return verts, faces, norms, uvs, albedo, layer_textures, layer_visible

    def restore_snapshot(
        self,
        vertices: "np.ndarray | None",
        faces: "np.ndarray | None",
        normals: "np.ndarray | None",
        uvs: "np.ndarray | None",
        albedo: "Image.Image",
        layer_textures: "dict | None" = None,
        layer_visible:  "dict | None" = None,
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
        uvs : array or None
            UV coordinate array to restore alongside geometry. Must match the
            vertex layout of the restored mesh. None for paint-only snapshots.
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

            # Restore the UV array from the snapshot. The reconstructed Trimesh
            # has no TextureVisuals so _extract_uvs() always returns None here.
            # Using the snapshotted uvs guarantees the UV layout matches the
            # restored topology exactly — critical after topology-changing ops.
            if uvs is not None:
                self._uvs = uvs.copy()
            # If uvs is None (old snapshot without uv field), fall back to
            # current _uvs (valid for sculpt-only restores where topology is same)

            # Rebuild all topology caches for the restored mesh
            self._rebuild_topology()

        # Restore the BASE buffer (brush strokes, no layer textures)
        self._albedo_base = albedo.copy()
        self._tex_w, self._tex_h = self._albedo_base.size

        # Restore per-layer texture state if the snapshot includes it.
        # If not provided (old snapshot), keep existing layer state so the
        # composite still reflects any layers the user has active.
        if layer_textures is not None:
            self._layer_textures = {
                lid: img.copy() for lid, img in layer_textures.items()
            }
        if layer_visible is not None:
            self._layer_visible = dict(layer_visible)

        # Sync _layer_projections: remove any projections for layers that
        # are absent from the restored snapshot. This ensures that undoing
        # a projection call correctly removes it from the shader render pass.
        if layer_textures is not None:
            stale = [lid for lid in list(self._layer_projections.keys())
                     if lid not in layer_textures]
            for lid in stale:
                self._layer_projections.pop(lid, None)
                log.debug("MeshPainter: restore_snapshot cleared projection for layer %d", lid)

        # Rebuild the composite from the restored base.
        # With shader-based projection, _rebuild_composite() just copies _albedo_base
        # and marks the texture dirty for GPU reupload.
        self._rebuild_composite()

    def _rebuild_topology(self) -> None:
        """
        Rebuild all topology-derived caches after any mesh change.

        Called after restore_snapshot() and after every mesh operation that
        modifies vertex positions, face count, or connectivity. Rebuilds:
          - BVH (ray–triangle intersector)
          - Face adjacency dict (for BFS region fill)
          - Vertex neighbor lists (for Smooth sculpt brush)
          - Seam partner groups (for seam-safe sculpt deformation)
        """
        self._rebuild_bvh()
        self._face_adj = self._build_face_adjacency()
        self._vertex_neighbors = self._build_vertex_neighbors()
        self._seam_partners = self._build_seam_partners()

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
    # Mesh operations                                                      #
    # ------------------------------------------------------------------ #

    def mesh_smooth(self, iterations: int = 5) -> None:
        """
        Apply Taubin smoothing to the entire mesh.

        Taubin smoothing alternates between a positive and a negative Laplacian
        pass, which reduces noise without the shrinkage that plain Laplacian
        smoothing causes. Vertex count and face topology are unchanged, so UV
        coordinates remain valid and no UV transfer is needed.

        Parameters
        ----------
        iterations : int
            Number of Taubin iteration pairs (each pair = one shrink + one inflate).
            Typical useful range: 1–20.
        """
        import trimesh.smoothing as _smooth

        # filter_taubin modifies mesh.vertices in-place and invalidates trimesh cache.
        # It processes every vertex independently without knowing about UV-seam
        # duplicates, so it will move the two sides of each seam by slightly
        # different amounts (different neighbour sets on each side), tearing the
        # mesh open exactly like the sculpt brush did before the seam fix.
        _smooth.filter_taubin(self._mesh, lamb=0.5, nu=-0.53, iterations=iterations)

        # Seam reconciliation — same logic as the sculpt seam fix:
        # For every UV-seam group, replace every member's position with the
        # group average so both sides of the seam end up at the same point.
        verts = np.asarray(self._mesh.vertices, dtype=np.float64)
        seen_groups: set = set()
        for i, partners in enumerate(self._seam_partners):
            if not partners:
                continue
            group = [i] + partners
            group_key = min(group)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            # Average the post-smooth positions of all seam-duplicate vertices
            avg_pos = np.mean([verts[g] for g in group], axis=0)
            for g in group:
                verts[g] = avg_pos

        # Write back through the trimesh setter to invalidate cached normals/bounds
        self._mesh.vertices = verts

        log.info("MeshPainter: Taubin smooth (%d iterations)", iterations)

        # Topology unchanged — only BVH needs rebuilding (vertex positions changed)
        self._rebuild_bvh()
        self._seam_partners = self._build_seam_partners()

    def mesh_remove_floaters(self, min_faces: int = 100) -> None:
        """
        Remove disconnected mesh components smaller than a face-count threshold.

        Photogrammetry often produces small disconnected fragments (floaters) that
        are noise artefacts. This splits the mesh into connected components and
        discards any whose face count is below min_faces, keeping the main body
        of the object intact.

        UV coordinates for the removed vertices are simply discarded; the remaining
        vertices keep their original UV indices.

        Parameters
        ----------
        min_faces : int
            Any component with fewer triangles than this is removed.
            If every component is below the threshold, only the largest is kept
            to avoid accidentally deleting the entire mesh.
        """
        components = self._mesh.split(only_watertight=False)
        if not components:
            return

        # Keep components at or above the threshold
        keep = [c for c in components if len(c.faces) >= min_faces]

        # Safety net: if the threshold would remove everything, keep the largest
        if not keep:
            keep = [max(components, key=lambda m: len(m.faces))]
            log.warning(
                "MeshPainter: remove_floaters min_faces=%d too high — "
                "keeping largest component (%d faces)",
                min_faces, len(keep[0].faces),
            )

        if len(keep) == len(components):
            log.info("MeshPainter: remove_floaters — nothing to remove")
            return

        # Concatenate retained components into one mesh
        new_mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]

        # Transfer UVs: the new mesh is a subset of the original vertices.
        # Use KD-tree nearest-vertex lookup to map new vertices back to their
        # original UV coordinates (exact match for unmodified vertices).
        self._uvs = self._transfer_uvs_by_position(
            new_mesh,
            source_verts=np.asarray(self._mesh.vertices, dtype=np.float64),
        )
        self._mesh = new_mesh

        removed = sum(len(c.faces) for c in components) - len(new_mesh.faces)
        log.info(
            "MeshPainter: removed %d floater triangles, kept %d",
            removed, len(new_mesh.faces),
        )
        self._rebuild_topology()

    def mesh_fill_holes(self) -> None:
        """
        Fill open boundary loops in the mesh.

        Photogrammetry meshes frequently have holes where the scanner had no
        coverage (e.g. the underside of an object). trimesh.repair.fill_holes
        identifies open boundary edges and adds triangles to close them.

        New vertices introduced to fill holes are appended to the end of the
        vertex array and are assigned UV (0, 0) — the filled patch won't have
        correct texture, but the mesh becomes watertight for export.
        """
        n_verts_before = len(self._mesh.vertices)
        trimesh.repair.fill_holes(self._mesh)
        n_new = len(self._mesh.vertices) - n_verts_before

        # Extend UV array with (0, 0) entries for new hole-fill vertices
        if n_new > 0 and self._uvs is not None:
            new_uvs = np.zeros((n_new, 2), dtype=np.float32)
            self._uvs = np.vstack([self._uvs, new_uvs])
            log.info("MeshPainter: fill_holes added %d vertices (UV set to 0,0)", n_new)
        else:
            log.info("MeshPainter: fill_holes — no new vertices added")

        self._rebuild_topology()

    def mesh_decimate(self, target_faces: int) -> None:
        """
        Reduce polygon count via quadric error decimation.

        Uses trimesh's simplify_quadric_decimation which is a fast CPU-side
        quadric error metric decimator. After decimation the vertex layout is
        completely new, so UVs are transferred via KD-tree nearest-original-
        vertex lookup — each new vertex inherits the UV of the closest original
        vertex. This is accurate for moderate decimation where new vertices
        remain geometrically close to their originals.

        Parameters
        ----------
        target_faces : int
            Desired triangle count in the output mesh. Clamped to at least 4
            (minimum valid mesh) and at most the current face count.
        """
        current_faces = len(self._mesh.faces)
        target_faces = max(4, min(target_faces, current_faces))

        if target_faces >= current_faces:
            log.info("MeshPainter: decimate skipped — target >= current (%d)", current_faces)
            return

        # Snapshot source positions for UV transfer before the mesh changes
        source_verts = np.asarray(self._mesh.vertices, dtype=np.float64).copy()

        decimated = self._mesh.simplify_quadric_decimation(target_faces)
        log.info(
            "MeshPainter: decimated %d → %d faces",
            current_faces, len(decimated.faces),
        )

        # Transfer UVs from original vertex positions to decimated vertex positions
        self._uvs = self._transfer_uvs_by_position(decimated, source_verts=source_verts)
        self._mesh = decimated
        self._rebuild_topology()

    def _transfer_uvs_by_position(
        self,
        new_mesh: "trimesh.Trimesh",
        source_verts: "np.ndarray",
    ) -> "np.ndarray | None":
        """
        Build a UV array for new_mesh by mapping each vertex to the nearest
        vertex in source_verts via KD-tree, then copying that vertex's UV.

        Used after topology-changing operations (decimate, remove
        floaters) where the new mesh shares approximate 3D positions with the
        original but has a different vertex index layout.

        Parameters
        ----------
        new_mesh : trimesh.Trimesh
            The mesh whose vertices need UV coordinates.
        source_verts : (N, 3) float64
            Original vertex positions to build the KD-tree from.

        Returns
        -------
        (M, 2) float32 UV array aligned to new_mesh.vertices, or None if the
        source mesh has no UV data.
        """
        if self._uvs is None:
            return None

        from scipy.spatial import cKDTree

        # Build KD-tree on original vertex positions for fast nearest-neighbour query
        tree = cKDTree(source_verts)

        new_verts = np.asarray(new_mesh.vertices, dtype=np.float64)
        # k=1 → single nearest neighbour; returns (distances, indices)
        _, idx = tree.query(new_verts, k=1, workers=-1)

        # Map each new vertex to its nearest source vertex's UV
        return self._uvs[idx].astype(np.float32)

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

    def bake_projections_to_atlas(self) -> "Image.Image":
        """
        Bake all visible shader-projection layers onto the UV atlas.

        Implements the same planar-projection math as PROJ_FRAG_SHADER (in
        viewport.py) but runs entirely on the CPU using numpy so the result
        can be embedded in an exported mesh file without any OpenGL dependency.

        Algorithm (per projection layer, per selected face):
            1. Rasterize the face's UV-atlas triangle to find all covered texels.
            2. Compute barycentric weights → interpolate 3D world position.
            3. Project 3D position onto the orthonormal (R, U) plane.
            4. Normalise → apply material transform (scale, rotate, offset).
            5. Bilinear-sample the source texture.
            6. Alpha-composite onto the result buffer.

        Returns
        -------
        Image.Image
            RGBA copy of the albedo with all visible projection layers baked in.
            The original _albedo_img is NOT modified — callers receive an
            independent image suitable for saving or passing to the exporter.
        """
        # Supersample at 2× resolution then downsample at the end.
        # Each UV triangle covers 4× more pixels during rasterization, giving
        # smoother edges and better colour gradients before the final LANCZOS
        # downsample brings it back to the atlas resolution.
        SUPER = 2
        W0, H0 = self._albedo_img.size                    # target (atlas) size
        W,  H  = W0 * SUPER, H0 * SUPER                  # working size

        # Start from a 2× upscaled copy so we never mutate the live buffer.
        result_arr = np.array(
            self._albedo_img.resize((W, H), Image.LANCZOS), dtype=np.float32
        )  # (H, W, 4)

        if not self._layer_projections or self._uvs is None:
            baked = Image.fromarray(result_arr.clip(0, 255).astype(np.uint8), "RGBA")
            return baked.resize((W0, H0), Image.LANCZOS) if SUPER > 1 else baked

        verts = np.asarray(self._mesh.vertices, dtype=np.float64)
        faces = np.asarray(self._mesh.faces,    dtype=np.int32)
        uvs   = np.asarray(self._uvs,           dtype=np.float32)

        for layer_id, params in self._layer_projections.items():
            # Respect eye-toggle visibility — hidden layers are skipped.
            if not self._layer_visible.get(layer_id, True):
                continue

            face_set  = params["face_set"]
            R_vec     = params["right"].astype(np.float64)
            U_vec     = params["up"].astype(np.float64)
            r_min     = float(params["r_min"])
            r_range   = float(params["r_range"])
            u_min     = float(params["u_min"])
            u_range   = float(params["u_range"])
            scale     = float(params["scale"])
            cos_a     = float(params["cos"])
            sin_a     = float(params["sin"])
            offset_x  = float(params["offset_x"])
            offset_y  = float(params["offset_y"])
            opacity   = float(params["opacity"])
            tex_img   = params["texture_img"]           # PIL RGBA from setup_layer_projection
            tex_arr   = np.array(tex_img, dtype=np.float32)  # (th, tw, 4)
            th, tw    = tex_arr.shape[:2]

            for fi in face_set:
                f          = faces[fi]
                v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])

                # UV-atlas pixel positions of the three triangle vertices.
                # Multiply by the supersampled dimensions so the rasterization
                # covers SUPER× more pixels per triangle than the atlas native size.
                uv0 = uvs[v0];  uv1 = uvs[v1];  uv2 = uvs[v2]
                px0 = float(uv0[0]) * (W - 1)
                py0 = float(uv0[1]) * (H - 1)
                px1 = float(uv1[0]) * (W - 1)
                py1 = float(uv1[1]) * (H - 1)
                px2 = float(uv2[0]) * (W - 1)
                py2 = float(uv2[1]) * (H - 1)

                # Bounding box of this UV triangle in texel space.
                x_min = max(0,     int(np.floor(min(px0, px1, px2))))
                x_max = min(W - 1, int(np.ceil( max(px0, px1, px2))))
                y_min = max(0,     int(np.floor(min(py0, py1, py2))))
                y_max = min(H - 1, int(np.ceil( max(py0, py1, py2))))
                if x_min >= x_max or y_min >= y_max:
                    continue

                # Candidate pixel grid inside the bounding box.
                xs = np.arange(x_min, x_max + 1, dtype=np.float32)
                ys = np.arange(y_min, y_max + 1, dtype=np.float32)
                gx, gy = np.meshgrid(xs, ys)
                gx = gx.ravel();  gy = gy.ravel()

                # Barycentric coordinates in UV-atlas space.
                denom = (py1 - py2) * (px0 - px2) + (px2 - px1) * (py0 - py2)
                if abs(denom) < 1e-10:
                    continue
                w0 = ((py1 - py2) * (gx - px2) + (px2 - px1) * (gy - py2)) / denom
                w1 = ((py2 - py0) * (gx - px2) + (px0 - px2) * (gy - py2)) / denom
                w2 = 1.0 - w0 - w1

                inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if not np.any(inside):
                    continue

                gx_in = gx[inside];  gy_in = gy[inside]
                w0_in = w0[inside][:, None]
                w1_in = w1[inside][:, None]
                w2_in = w2[inside][:, None]

                # Interpolate 3D world position for each interior texel.
                p0 = verts[v0];  p1 = verts[v1];  p2 = verts[v2]
                pos = (w0_in * p0 + w1_in * p1 + w2_in * p2)  # (N, 3)

                # Planar UV projection — mirrors the GLSL shader exactly.
                r_coord = pos @ R_vec
                u_coord = pos @ U_vec
                pu = (r_coord - r_min) / r_range
                pv = (u_coord - u_min) / u_range

                # Material transform: centre, scale, rotate, re-wrap.
                pu_c = pu - 0.5 - offset_x
                pv_c = pv - 0.5 - offset_y
                pu_c /= scale
                pv_c /= scale
                pu_r = pu_c * cos_a - pv_c * sin_a
                pv_r = pu_c * sin_a + pv_c * cos_a

                tex_u = np.mod(pu_r + 0.5, 1.0)
                tex_v = np.mod(pv_r + 0.5, 1.0)

                # Bilinear sample the source texture.
                fx   = tex_u * (tw - 1)
                fy   = tex_v * (th - 1)
                x0i  = np.floor(fx).astype(np.int32).clip(0, tw - 2)
                y0i  = np.floor(fy).astype(np.int32).clip(0, th - 2)
                x1i  = x0i + 1;  y1i = y0i + 1
                wx   = (fx - x0i)[:, None]
                wy   = (fy - y0i)[:, None]
                c00  = tex_arr[y0i, x0i]
                c01  = tex_arr[y0i, x1i]
                c10  = tex_arr[y1i, x0i]
                c11  = tex_arr[y1i, x1i]
                samp = (c00 * (1 - wx) * (1 - wy) + c01 * wx * (1 - wy)
                        + c10 * (1 - wx) * wy      + c11 * wx * wy)

                # Scale alpha by opacity.
                samp[:, 3] *= opacity

                # Alpha-composite sampled colour over existing result.
                ix    = gx_in.astype(np.int32).clip(0, W - 1)
                iy    = gy_in.astype(np.int32).clip(0, H - 1)
                src_a = samp[:, 3:4] / 255.0
                dst   = result_arr[iy, ix]
                result_arr[iy, ix] = samp * src_a + dst * (1.0 - src_a)

            log.debug(
                "MeshPainter: baked projection layer %d (%d faces)",
                layer_id, len(face_set),
            )

        # Downsample from the supersampled working size back to the atlas
        # resolution using LANCZOS (high-quality anti-aliasing filter).
        baked = Image.fromarray(result_arr.clip(0, 255).astype(np.uint8), "RGBA")
        if SUPER > 1:
            baked = baked.resize((W0, H0), Image.LANCZOS)
        return baked
