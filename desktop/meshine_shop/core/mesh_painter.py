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
