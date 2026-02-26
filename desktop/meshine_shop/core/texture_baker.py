"""
Texture baking module for Meshine Shop (Phase 2c, 2d, 2g).

Generates five PBR texture maps from the UV-mapped decimated mesh:

    albedo.png    — Diffuse/base color projected from the reconstruction source.
                    Apple path: texel-space projection from USDZ texture atlas —
                    every output texel samples the original full-res USDZ texture
                    directly, recovering full photographic detail.
                    COLMAP path: colored dense/sparse point cloud nearest-neighbor.
                    Resolution: preset-scaled (Mobile 1024, PC 2048, Cinematic 4096).

    normal.png    — Tangent-space normal map encoding surface orientation.
                    Apple path: texel-space rebake from USDZ *_norm0.png map.
                    COLMAP path: computed from mesh vertex normals.
                    Resolution: preset-scaled (Mobile 1024, PC 2048, Cinematic 4096).

    ao.png        — Ambient occlusion map.
                    Apple path: texel-space rebake from USDZ *_ao0.png map.
                    COLMAP path: hemisphere ray casting via Open3D C++ BVH.
                    Bright (white) = fully lit, dark (black) = occluded.
                    Resolution: preset-scaled (Mobile 1024, PC 2048, Cinematic 4096).

    roughness.png — Surface roughness.
                    Apple path: texel-space rebake from USDZ *_roughness0.png.
                    COLMAP path: estimated from albedo HSV + AO blend.
                    Resolution: preset-scaled (Mobile 1024, PC 2048, Cinematic 4096).

    metallic.png  — Metallic/non-metallic map.
                    Apple path: texel-space rebake from USDZ *_metalness0.png.
                    COLMAP path: estimated from albedo saturation.
                    Resolution: preset-scaled (Mobile 1024, PC 2048, Cinematic 4096).

Core technique — texel-space UV projection with barycentric interpolation (Phase 2g+):
    The previous vertex-space approach sampled the original texture at each of
    the ~25K–200K decimated mesh vertices, then interpolated. This lost up to
    99% of the original texture detail between vertices.

    The Phase 2g texel-space approach works per output pixel:
    1. For each texel in the output atlas, find which UV triangle it falls inside
       and compute its 3D world-space position via barycentric interpolation of
       the triangle's 3D vertices.
    2. BVH surface proximity query: for each texel's 3D position, find the nearest
       SURFACE POINT on the original USDZ mesh (not just nearest vertex) using
       trimesh's ProximityQuery. Returns closest point + triangle face ID.
    3. Cramér's rule barycentric coordinates: compute the texel's weights within
       its USDZ triangle, then interpolate the per-face-vertex USDZ UV at those
       weights → one unique, smooth UV per texel.
    4. Sample the USDZ texture at the interpolated UV.
    Result: photographic-quality baking with no Voronoi polygon artifacts, limited
    only by the original Object Capture texture resolution.

    The earlier nearest-vertex KD-tree approach caused a "Voronoi mosaic" artifact:
    every texel nearest to the same USDZ vertex got the same UV → same source pixel
    → flat colour per Voronoi cell → blocky polygon patches visible at any zoom level.
    Barycentric surface interpolation eliminates this completely.

Texture dilation:
    After rasterizing any UV atlas, pixels just outside UV island boundaries
    remain unfilled (zero). Game engine mipmapping blends these black pixels
    into visible dark seams at UV island borders. Dilation expands each island
    outward by DILATION_PIXELS (4px) using scipy's distance_transform_edt,
    filling border pixels with the color of their nearest island neighbor.

Public API:
    bake_usdz_maps_texelspace(mesh, usdz_extract_dir, output_dir, on_progress)
        → dict[str, Path]  — Texel-space rebake of all 5 USDZ PBR maps.
    bake_albedo_from_usdz(mesh, usdz_extract_dir, on_progress)
        → np.ndarray | None  — Vertex-space albedo fallback (COLMAP-style).
    bake_albedo_from_pointcloud(mesh, pcd_path, on_progress)
        → np.ndarray | None  — COLMAP colored point cloud vertex colors.
    vertex_colors_to_texture(uvs, faces, vertex_colors, image_size)
        → PIL Image  — Rasterize per-vertex colors with dilation.
    bake_normal_map(mesh, image_size) → PIL Image
    bake_ao(mesh, num_rays, image_size, on_progress) → PIL Image
    bake_roughness_map(albedo_path, ao_path, output_path, image_size)
        → PIL Image
    bake_metallic_map(albedo_path, output_path, image_size) → PIL Image
"""

import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default texture resolution for the PC quality preset.
# Actual resolution is set per-run by the engine factory based on quality preset:
#   Mobile  (15K triangles)  → 1024×1024  — fast baking, adequate for small assets
#   PC      (65K triangles)  → 2048×2048  — standard game asset quality (this default)
#   Cinematic (200K triangles)→ 4096×4096 — 4× more texels for fine skin/surface detail
DEFAULT_TEXTURE_SIZE = 2048

# Epsilon for barycentric coordinate inside-triangle test. A small negative
# value allows pixels exactly on triangle edges to be included, preventing
# hairline gaps between adjacent UV islands.
BARY_EPSILON = -1e-5

# Ray origin offset to prevent self-intersection during AO ray casting.
# Rays start slightly above the surface along the vertex normal.
AO_RAY_OFFSET = 1e-3

# Texture dilation radius in pixels. After UV rasterization, pixels near
# UV island boundaries are unfilled. Mipmapping in game engines blends these
# black gaps into visible dark seams. Dilation fills the 4-pixel border zone
# with the nearest island color, eliminating the artifact.
DILATION_PIXELS = 4


# ---------------------------------------------------------------------------
# UV-space triangle rasterization
# ---------------------------------------------------------------------------

def _rasterize_triangles(uvs, faces, vertex_data, image_size):
    """
    Rasterize UV-mapped mesh triangles into a texture image.

    For each triangle in UV space:
    1. Compute its pixel-space bounding box
    2. For each pixel in the bounding box, compute barycentric coordinates
    3. If inside the triangle, interpolate vertex_data at that pixel
    4. Write the interpolated value to the output image

    This is standard software UV rasterization — the same algorithm used
    inside Blender's bake system and Substance Painter.

    Args:
        uvs:         (N, 2) float32 UV coordinates in [0, 1]² space.
        faces:       (F, 3) int32 triangle face indices into uvs.
        vertex_data: (N, C) float32 per-vertex data to interpolate
                     (RGB colors, normal vectors, AO values, etc.).
        image_size:  int — output texture resolution (square).

    Returns:
        numpy array of shape (image_size, image_size, C) with float32 values.
        Unfilled pixels (outside all UV islands) remain as 0.
    """
    n_channels = vertex_data.shape[1]
    img = np.zeros((image_size, image_size, n_channels), dtype=np.float32)

    # Convert UV coordinates to pixel coordinates.
    # U (horizontal) maps directly to X. V (vertical) is flipped because UV
    # has origin at bottom-left but image arrays have origin at top-left.
    px = uvs[:, 0] * (image_size - 1)                   # U → pixel X
    py = (1.0 - uvs[:, 1]) * (image_size - 1)           # V → pixel Y (flipped)

    for face in faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

        # Triangle vertex positions in pixel space.
        x0, y0 = px[i0], py[i0]
        x1, y1 = px[i1], py[i1]
        x2, y2 = px[i2], py[i2]

        # Per-vertex data to be interpolated across this triangle.
        d0 = vertex_data[i0]
        d1 = vertex_data[i1]
        d2 = vertex_data[i2]

        # Axis-aligned bounding box of this triangle in pixel space.
        # Clamp to image bounds to avoid out-of-range writes.
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(image_size - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(image_size - 1, int(np.ceil(max(y0, y1, y2))))

        if xmin > xmax or ymin > ymax:
            continue  # Triangle outside image bounds

        # Denominator of the barycentric coordinate formula.
        # Uses the signed area of the triangle (cross product magnitude).
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue  # Degenerate / zero-area triangle

        # Generate pixel coordinates for the entire bounding box at once.
        # np.meshgrid creates 2D arrays of (x, y) for every pixel in the bbox.
        xs = np.arange(xmin, xmax + 1, dtype=np.float32)
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        # Compute barycentric coordinates for all bbox pixels simultaneously.
        # w0, w1, w2 are the weights of each vertex's contribution at each pixel.
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1

        # Pixels inside the triangle have all weights >= 0.
        # BARY_EPSILON allows edge pixels to be included (prevents UV seam gaps).
        inside = (w0 >= BARY_EPSILON) & (w1 >= BARY_EPSILON) & (w2 >= BARY_EPSILON)
        if not inside.any():
            continue

        # Get the local (row, col) indices of inside pixels.
        iy, ix = np.where(inside)

        # Interpolate vertex data at each inside pixel using barycentric weights.
        # Expand weights to (P, 1) so they broadcast correctly against (C,) data.
        w0_i = w0[iy, ix][:, np.newaxis]
        w1_i = w1[iy, ix][:, np.newaxis]
        w2_i = w2[iy, ix][:, np.newaxis]
        interpolated = w0_i * d0 + w1_i * d1 + w2_i * d2

        # Write to the full image at the correct pixel coordinates.
        # ymin + iy converts local bbox row indices to image row indices.
        img[ymin + iy, xmin + ix] = interpolated

    return img


# ---------------------------------------------------------------------------
# Texture dilation — prevents dark seam artifacts at UV island borders
# ---------------------------------------------------------------------------

def _dilate_texture(img_data, filled_mask, iterations=DILATION_PIXELS):
    """
    Expand UV island borders to prevent dark seam artifacts in game engines.

    After UV rasterization, pixels just outside UV island boundaries remain
    unfilled (zero / black). When a game engine generates mipmaps, it blends
    these black pixels into the adjacent island pixels, producing visible dark
    seams at UV island borders. This is most noticeable on lower mip levels.

    This function fills a DILATION_PIXELS-wide border zone around each island
    with the color of the nearest filled pixel. The result: seam-adjacent texels
    have proper color values and mipmapping produces clean island borders.

    Uses scipy.ndimage.distance_transform_edt for a single vectorized pass
    (no per-pixel Python loop). Falls back to an iterative numpy expansion
    if scipy is not available.

    Args:
        img_data:    numpy array (H, W, C) or (H, W) — texture data to dilate.
        filled_mask: (H, W) bool array — True where UV islands were rasterized.
        iterations:  int — dilation radius in pixels (default: DILATION_PIXELS).

    Returns:
        numpy array of same shape as img_data with border pixels filled.
    """
    result = img_data.copy()

    try:
        from scipy.ndimage import distance_transform_edt

        # distance_transform_edt(~filled_mask) gives each unfilled pixel its
        # distance to the nearest filled pixel, and (with return_indices=True)
        # the row/col of that nearest pixel. We copy from that nearest pixel.
        dist, nearest_indices = distance_transform_edt(
            ~filled_mask, return_indices=True
        )

        # Only fill pixels within `iterations` pixels of a UV island.
        # Pixels further away keep their initial value (zero / background).
        dilation_mask = (dist > 0) & (dist <= iterations)

        if dilation_mask.any():
            nearest_r = nearest_indices[0][dilation_mask]
            nearest_c = nearest_indices[1][dilation_mask]
            result[dilation_mask] = img_data[nearest_r, nearest_c]

        return result

    except ImportError:
        # scipy not available — iterative numpy expansion (4× slower but correct).
        current_filled = filled_mask.copy()

        for _ in range(iterations):
            new_filled = current_filled.copy()
            new_data = result.copy()

            # Expand in 8 directions using numpy roll + edge masking.
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                s_filled = np.roll(current_filled, (dr, dc), axis=(0, 1))
                s_data = np.roll(result, (dr, dc), axis=(0, 1))

                # Zero out pixels that wrapped around the image boundary.
                if dr < 0:
                    s_filled[dr:] = False
                elif dr > 0:
                    s_filled[:dr] = False
                if dc < 0:
                    s_filled[:, dc:] = False
                elif dc > 0:
                    s_filled[:, :dc] = False

                can_fill = ~current_filled & s_filled & ~new_filled
                if can_fill.any():
                    new_data[can_fill] = s_data[can_fill]
                    new_filled[can_fill] = True

            if not np.any(new_filled & ~current_filled):
                break  # No new pixels reached — done early

            result = new_data
            current_filled = new_filled

        return result


# ---------------------------------------------------------------------------
# Texel-space USDZ PBR map extraction (Phase 2g — Apple Object Capture path)
# ---------------------------------------------------------------------------

def bake_usdz_maps_texelspace(mesh, usdz_extract_dir, output_dir, on_progress,
                               image_size=DEFAULT_TEXTURE_SIZE):
    """
    Extract all available PBR maps from a USDZ archive and rebake them onto
    the UV-mapped decimated mesh using texel-space projection.

    This is the highest-quality texture baking method available for the Apple
    Object Capture path. Instead of sampling the original texture at each of
    the ~25K–200K decimated vertex positions (vertex-space), this function
    samples at every output texel (~4M pixels), recovering the full resolution
    and detail of the original Object Capture PBR maps.

    The projection is done once and reused for all 5 PBR maps:
    1. Rasterize the decimated mesh's UV atlas → (H, W, 3) 3D world positions
       per output texel. (Re-uses _rasterize_triangles with vertex_data=positions.)
    2. Gather all M filled texels → (M, 3) world positions.
    3. Batch KD-tree query: for each texel's 3D position, find the nearest
       vertex in the original USDZ mesh (via scipy KDTree, parallel with all CPUs).
    4. Read each original vertex's UV in the USDZ atlas → (M, 2) source UVs.
    5. For each available USDZ PBR map: sample source texture at those UVs,
       write to output atlas, apply dilation, save to output_dir/.

    Apple Object Capture USDZ naming convention:
        *_tex0.png       → albedo (diffuse base color)
        *_norm0.png      → tangent-space normal map
        *_ao0.png        → ambient occlusion
        *_roughness0.png → surface roughness
        *_metalness0.png → metallic mask

    Args:
        mesh:             trimesh.Trimesh — the UV-mapped decimated mesh.
        usdz_extract_dir: Path — workspace/mesh/usdz_extracted/.
        output_dir:       Path — workspace/textures/ for output PNGs.
        on_progress:      Callback(str) for status messages.
        image_size:       int — output texture resolution (default 2048).

    Returns:
        dict mapping texture name → saved Path for successfully baked maps.
        Example: {"albedo": Path(".../albedo.png"), "normal": Path(".../normal.png")}
        Empty dict if USDZ geometry cannot be loaded.
    """
    usdz_extract_dir = Path(usdz_extract_dir)
    output_dir = Path(output_dir)

    # --- Load the original USDZ mesh + UV coordinates ---
    # The .usdc geometry file was extracted by mesh_reconstruct(). We need:
    #   - original vertex positions (for KD-tree construction)
    #   - face-varying UV primvar (for sampling the original texture atlas)
    usdc_files = list(usdz_extract_dir.rglob("*.usdc"))
    if not usdc_files:
        on_progress("No .usdc geometry in USDZ extraction — skipping texel-space baking")
        return {}

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(usdc_files[0]))

        # Find the first mesh prim in the stage hierarchy.
        mesh_prim = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = UsdGeom.Mesh(prim)
                break

        if mesh_prim is None:
            on_progress("No mesh prim found in USDZ — skipping texel-space baking")
            return {}

        # Extract vertex positions and face-vertex indices.
        orig_points = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        orig_face_indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get())

        # Find the UV primvar. USD stores UVs as faceVarying primvars named
        # "st", "st_0", or "uv" depending on the exporter.
        uv_primvar = None
        for attr_name in ["st", "st_0", "uv"]:
            pv = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar(attr_name)
            if pv and pv.IsDefined():
                uv_primvar = pv
                break

        if uv_primvar is None:
            on_progress("No UV primvar in USDZ — skipping texel-space baking")
            return {}

        # Decode face-varying UVs into per-face arrays (shape: F×3×2).
        #
        # The old approach averaged UVs per vertex — fundamentally wrong at seam
        # vertices, which have MULTIPLE UV positions (one per UV island they border).
        # Averaging gives a UV that doesn't exist in the texture, and more critically
        # causes the Voronoi mosaic artifact: every texel near a vertex gets the same
        # averaged UV → same source pixel → flat-color polygon patches.
        #
        # Per-face storage lets us barycentric-interpolate the UV at any sub-triangle
        # point on the USDZ surface, giving smooth, photographic-quality results.
        uv_values = np.array(uv_primvar.Get(), dtype=np.float32)  # (N_uv, 2)
        uv_indices = uv_primvar.GetIndices()
        face_vertex_counts_raw = mesh_prim.GetFaceVertexCountsAttr().Get()

        # Fast path: all-triangle mesh (Apple Object Capture always outputs triangles).
        all_tris = all(int(c) == 3 for c in face_vertex_counts_raw)
        if all_tris:
            orig_triangles = orig_face_indices.reshape(-1, 3).astype(np.int32)
            if uv_indices is not None and len(uv_indices) > 0:
                # Indexed face-varying: uv_values[uv_indices[fv_idx]] = UV at that corner.
                uv_idx_grid = np.array(uv_indices, dtype=np.int32).reshape(-1, 3)  # (F, 3)
                face_uvs = uv_values[uv_idx_grid]                                  # (F, 3, 2)
            else:
                # Direct face-varying: uv_values already in face-vertex order.
                face_uvs = uv_values.reshape(-1, 3, 2)                            # (F, 3, 2)
        else:
            # Slow path: fan-triangulate n-gons (rare for Object Capture output).
            uv_idx_arr = (np.array(uv_indices, dtype=np.int32)
                          if (uv_indices is not None and len(uv_indices) > 0)
                          else None)
            tri_verts_list, tri_uvs_list = [], []
            fv_cursor = 0
            for fvc in face_vertex_counts_raw:
                n = int(fvc)
                face_vi = orig_face_indices[fv_cursor:fv_cursor + n]
                if uv_idx_arr is not None:
                    uv_v = uv_values[uv_idx_arr[fv_cursor:fv_cursor + n]]  # (n, 2)
                else:
                    uv_v = uv_values[fv_cursor:fv_cursor + n]              # (n, 2)
                for k in range(n - 2):
                    tri_verts_list.append([int(face_vi[0]), int(face_vi[k + 1]),
                                           int(face_vi[k + 2])])
                    tri_uvs_list.append([uv_v[0].tolist(), uv_v[k + 1].tolist(),
                                         uv_v[k + 2].tolist()])
                fv_cursor += n
            orig_triangles = np.array(tri_verts_list, dtype=np.int32)   # (F, 3)
            face_uvs = np.array(tri_uvs_list, dtype=np.float32)          # (F, 3, 2)

        on_progress(
            f"USDZ original mesh: {len(orig_points):,} vertices, "
            f"{len(orig_triangles):,} triangles"
        )

    except Exception as e:
        on_progress(f"Failed to load USDZ geometry: {e} — skipping texel-space baking")
        return {}

    # --- Step 1: Rasterize output UV atlas → 3D world positions per texel ---
    # Re-uses the existing rasterizer with vertex positions as "vertex_data".
    # Output: (image_size, image_size, 3) where each texel stores its 3D
    # world position, or (0, 0, 0) for pixels outside all UV islands.
    on_progress("Rasterizing output UV atlas to 3D world positions...")

    vertices = np.array(mesh.vertices, dtype=np.float32)
    uvs = np.array(mesh.visual.uv, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    pos_image = _rasterize_triangles(uvs, faces, vertices, image_size)
    # (image_size, image_size, 3) — zero for unfilled pixels

    # Identify filled pixels: at least one 3D position component is non-zero.
    # Vertices at the world origin (0,0,0) would be incorrectly excluded —
    # extremely rare in practice and acceptable as an edge case.
    filled_mask = np.any(pos_image != 0, axis=2)  # (H, W) bool
    filled_ys, filled_xs = np.where(filled_mask)

    if len(filled_ys) == 0:
        on_progress("No filled texels found — UV map may be empty")
        return {}

    texel_pos_3d = pos_image[filled_ys, filled_xs]  # (M, 3)
    on_progress(
        f"Projecting {len(texel_pos_3d):,} texels onto USDZ mesh "
        f"({len(orig_triangles):,} triangles) via BVH surface proximity..."
    )

    # --- Step 2: BVH surface proximity + barycentric UV interpolation ---
    #
    # Root cause of the Voronoi mosaic artifact in the old code:
    #   Old approach → nearest VERTEX (KD-tree) → that vertex's averaged UV.
    #   Every texel whose closest vertex is V gets the same UV → same source
    #   pixel → flat color filling the Voronoi cell → blocky polygon patches.
    #
    # Fix:
    #   1. Find the nearest SURFACE POINT on any triangle face (not just vertices)
    #      using trimesh's BVH-accelerated ProximityQuery.
    #   2. Compute Cramér's-rule barycentric coordinates of that surface point
    #      within its USDZ triangle.
    #   3. Interpolate the USDZ UV at the three triangle corners using the
    #      barycentric weights → unique, smooth UV per texel.
    #   Result: photographic-quality texture sampling with no polygon boundary
    #   artifacts, regardless of texture resolution.
    import trimesh as _trimesh

    # Build a trimesh from the original USDZ geometry for BVH proximity queries.
    usdz_trimesh = _trimesh.Trimesh(
        vertices=orig_points, faces=orig_triangles, process=False
    )
    prox = _trimesh.proximity.ProximityQuery(usdz_trimesh)

    # Query in chunks of 100K to keep peak memory constant across all presets
    # (1024² → ~1M texels; 4096² → ~16M texels — chunking prevents OOM).
    chunk_size = 100_000
    face_ids    = np.zeros(len(texel_pos_3d), dtype=np.int64)
    closest_pts = np.zeros_like(texel_pos_3d)

    for chunk_start in range(0, len(texel_pos_3d), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(texel_pos_3d))
        c_pts, _, c_fids = prox.on_surface(texel_pos_3d[chunk_start:chunk_end])
        closest_pts[chunk_start:chunk_end] = c_pts
        face_ids[chunk_start:chunk_end]    = c_fids
        if chunk_start % 500_000 == 0 and chunk_start > 0:
            on_progress(
                f"Surface proximity: {chunk_start:,}/{len(texel_pos_3d):,} texels..."
            )

    on_progress("Surface proximity complete — computing barycentric UV interpolation...")

    # Cramér's rule barycentric coordinates (vectorised):
    # For point P in triangle (V0, V1, V2) with edges e1=V1−V0, e2=V2−V0:
    #   bary_v = (d11·d20 − d01·d21) / (d00·d11 − d01²)
    #   bary_w = (d00·d21 − d01·d20) / (d00·d11 − d01²)
    #   bary_u = 1 − bary_v − bary_w
    tri_v = orig_points[orig_triangles[face_ids]]      # (M, 3, 3) triangle corners
    e1    = tri_v[:, 1] - tri_v[:, 0]                 # (M, 3) first edge
    e2    = tri_v[:, 2] - tri_v[:, 0]                 # (M, 3) second edge
    vp    = closest_pts - tri_v[:, 0]                 # (M, 3) vector to surface point

    d00  = (e1 * e1).sum(axis=1)
    d01  = (e1 * e2).sum(axis=1)
    d11  = (e2 * e2).sum(axis=1)
    d20  = (vp * e1).sum(axis=1)
    d21  = (vp * e2).sum(axis=1)

    denom      = d00 * d11 - d01 * d01
    safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)

    bary_v = (d11 * d20 - d01 * d21) / safe_denom    # weight for corner 1
    bary_w = (d00 * d21 - d01 * d20) / safe_denom    # weight for corner 2
    bary_u = 1.0 - bary_v - bary_w                   # weight for corner 0
    bary   = np.clip(np.stack([bary_u, bary_v, bary_w], axis=1), 0.0, 1.0)

    # Interpolate USDZ UV at each texel's barycentric position within its triangle.
    # face_uvs[face_ids]: (M, 3, 2) — UV at each corner for each texel's triangle.
    # Einstein sum:  UV[m] = Σ_i  bary[m,i] · face_uvs[face_ids[m], i]
    orig_uvs_at_texels = np.einsum(
        'mi,mid->md', bary, face_uvs[face_ids]
    )  # (M, 2)

    on_progress("Barycentric UV interpolation complete")

    # Helper to find a USDZ PBR texture by name patterns in the extraction dir.
    def _find_usdz_map(include_patterns, exclude_patterns=None):
        """Return the first image file matching any include pattern, excluding others."""
        for pattern in include_patterns:
            matches = list(usdz_extract_dir.rglob(f"*{pattern}*"))
            image_matches = [
                p for p in matches
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
            if exclude_patterns:
                image_matches = [
                    p for p in image_matches
                    if not any(ex in p.stem.lower() for ex in exclude_patterns)
                ]
            if image_matches:
                return image_matches[0]
        return None

    # --- Step 4: Sample each USDZ PBR map and write to output atlas ---
    # PBR map definitions:
    #   (output_name, search_patterns, exclude_patterns, channels, fill_value)
    #
    # fill_value: what to put in pixels that are > DILATION_PIXELS away from
    # any UV island (after dilation). Neutral values by convention:
    #   albedo:    mid-grey (0.5, 0.5, 0.5)
    #   normal:    flat normal pointing out (0.5, 0.5, 1.0) in RGB encoding
    #   ao:        fully lit (1.0)
    #   roughness: rough (0.8) — photogrammetry subjects are rarely mirrors
    #   metallic:  non-metallic (0.0)
    usdz_map_defs = [
        (
            "albedo",
            ["_tex0", "_tex", "diffuse", "albedo", "baseColor", "basecolor", "color"],
            ["_ao", "_norm", "_rough", "_metal", "_disp", "_emissive"],
            "RGB",
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        ),
        (
            "normal",
            ["_norm0", "_norm", "normal", "Normal"],
            ["_ao", "_rough", "_metal", "_tex"],
            "RGB",
            np.array([0.5, 0.5, 1.0], dtype=np.float32),
        ),
        (
            "ao",
            ["_ao0", "_ao", "ambient", "occlusion"],
            ["_norm", "_rough", "_metal", "_tex"],
            "L",
            np.float32(1.0),
        ),
        (
            "roughness",
            ["_roughness0", "_roughness", "roughness", "Roughness"],
            ["_norm", "_ao", "_metal", "_tex"],
            "L",
            np.float32(0.8),
        ),
        (
            "metallic",
            ["_metalness0", "_metalness", "_metallic", "metallic", "Metallic"],
            ["_norm", "_ao", "_rough", "_tex"],
            "L",
            np.float32(0.0),
        ),
    ]

    saved_maps = {}

    for map_name, search_patterns, exclude_patterns, channels, fill_value in usdz_map_defs:

        source_path = _find_usdz_map(search_patterns, exclude_patterns)
        if source_path is None:
            on_progress(f"USDZ {map_name} map not found — will use fallback")
            continue

        try:
            # Load source USDZ texture as float32.
            src_img = Image.open(source_path).convert(channels)
            src_np = np.array(src_img, dtype=np.float32) / 255.0
            src_h = src_np.shape[0]
            src_w = src_np.shape[1]

            # Compute integer sample coordinates in source texture space.
            u_orig = orig_uvs_at_texels[:, 0]
            v_orig = orig_uvs_at_texels[:, 1]
            # V is flipped: UV origin is bottom-left, image origin is top-left.
            tex_x = np.clip((u_orig * (src_w - 1)).astype(np.int32), 0, src_w - 1)
            tex_y = np.clip(
                ((1.0 - v_orig) * (src_h - 1)).astype(np.int32), 0, src_h - 1
            )

            # Build output texture, initialised to the neutral fill value.
            if channels == "RGB":
                output = np.broadcast_to(fill_value, (image_size, image_size, 3)).copy()
                texel_colors = src_np[tex_y, tex_x]  # (M, 3)
                output[filled_ys, filled_xs] = texel_colors

                # Dilate island borders to prevent dark seam artifacts.
                output = _dilate_texture(output, filled_mask)

                img_uint8 = (np.clip(output, 0.0, 1.0) * 255).astype(np.uint8)
                result_img = Image.fromarray(img_uint8, "RGB")

            else:
                # Greyscale: src_np is (H, W) after PIL convert("L")
                output = np.full((image_size, image_size), float(fill_value), dtype=np.float32)
                texel_values = src_np[tex_y, tex_x]  # (M,)
                output[filled_ys, filled_xs] = texel_values

                # Expand to (H, W, 1) for the generic dilation function, then squeeze.
                output_3d = output[:, :, np.newaxis]
                output_3d = _dilate_texture(output_3d, filled_mask)
                output = output_3d[:, :, 0]

                img_uint8 = (np.clip(output, 0.0, 1.0) * 255).astype(np.uint8)
                result_img = Image.fromarray(img_uint8, "L")

            out_path = output_dir / f"{map_name}.png"
            result_img.save(str(out_path))
            saved_maps[map_name] = out_path
            on_progress(
                f"Baked {map_name} from USDZ ({source_path.name}) "
                f"→ {map_name}.png"
            )

        except Exception as e:
            on_progress(f"Failed to bake {map_name}: {e}")

    return saved_maps


# ---------------------------------------------------------------------------
# Nearest-neighbour color lookup (shared helper)
# ---------------------------------------------------------------------------

def _nearest_neighbor_colors(query_verts, source_verts, source_colors):
    """
    Assign a color to each query vertex by finding its nearest source point.

    Used to transfer color data from a point cloud (with RGB) to a mesh
    (with no color). Open3D's KDTreeFlann performs the nearest-neighbour
    search efficiently in C++.

    Args:
        query_verts:   (N, 3) float64 mesh vertex positions.
        source_verts:  (M, 3) float64 point cloud positions.
        source_colors: (M, 3) float32 point cloud RGB colors in [0, 1].

    Returns:
        (N, 3) float32 array of RGB colors in [0, 1] for each query vertex.
    """
    import open3d as o3d

    # Build a KD-tree from the source point cloud for fast nearest-neighbour queries.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source_verts.astype(np.float64))
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    result_colors = np.zeros((len(query_verts), 3), dtype=np.float32)

    for i, v in enumerate(query_verts):
        # search_knn_vector_3d returns (k, [indices], [sq_distances]).
        # k=1 finds the single nearest point.
        [_, idx, _] = kd_tree.search_knn_vector_3d(v.astype(np.float64), 1)
        result_colors[i] = source_colors[idx[0]]

    return result_colors


# ---------------------------------------------------------------------------
# Albedo baking — Apple Object Capture (USDZ) vertex-space fallback
# ---------------------------------------------------------------------------

def bake_albedo_from_usdz(mesh, usdz_extract_dir, on_progress):
    """
    Extract the diffuse texture from a USDZ extraction directory and transfer
    its colors to the UV-mapped decimated mesh (vertex-space approach).

    NOTE: This is the legacy vertex-space fallback. The new texel-space method
    bake_usdz_maps_texelspace() produces significantly higher quality by
    sampling at every output texel instead of every vertex. This function is
    retained as a fallback for use when the full texel-space projection fails.

    Apple Object Capture stores its results as a USDZ archive containing:
    - A .usdc geometry file with the original full-resolution mesh + UV coords
    - PNG texture images (diffuse, normal, roughness, metallic, AO)

    Transfer approach:
    1. Find the diffuse texture image in the extracted directory
    2. Load the original USD mesh vertices and face-varying UV coordinates
    3. For each vertex in the decimated mesh, find the nearest original vertex
    4. Sample the diffuse texture at that vertex's approximate UV position
    5. Return per-vertex colors for rasterization

    Args:
        mesh:             trimesh.Trimesh — the UV-mapped decimated mesh.
        usdz_extract_dir: Path — workspace/mesh/usdz_extracted/.
        on_progress:      Callback for status messages.

    Returns:
        (N, 3) float32 per-vertex RGB colors in [0, 1], or None if unavailable.
    """
    import open3d as o3d

    usdz_extract_dir = Path(usdz_extract_dir)

    # --- Find the diffuse texture image ---
    # Object Capture uses consistent naming conventions for PBR maps.
    # Apple stores textures in a subdirectory so we use rglob (recursive).
    diffuse_names = ["diffuse", "albedo", "baseColor", "basecolor", "color",
                     "Diffuse", "_tex"]

    texture_path = None
    for candidate_name in diffuse_names:
        matches = list(usdz_extract_dir.rglob(f"*{candidate_name}*"))
        image_matches = [p for p in matches if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        non_diffuse = {"_ao", "_norm", "_rough", "_metal", "_disp", "_emissive"}
        image_matches = [
            p for p in image_matches
            if not any(tag in p.stem.lower() for tag in non_diffuse)
        ]
        if image_matches:
            texture_path = image_matches[0]
            break

    if texture_path is None:
        # Fallback: pick the largest PNG that isn't a known non-diffuse map.
        png_files = (
            list(usdz_extract_dir.rglob("*.png"))
            + list(usdz_extract_dir.rglob("*.jpg"))
        )
        non_diffuse = {"_ao", "_norm", "_rough", "_metal", "_disp", "_emissive"}
        png_files = [
            p for p in png_files
            if not any(tag in p.stem.lower() for tag in non_diffuse)
        ]
        if png_files:
            texture_path = max(png_files, key=lambda p: p.stat().st_size)

    if texture_path is None:
        on_progress("No diffuse texture found in USDZ — using fallback gray")
        return None

    on_progress(f"Found diffuse texture: {texture_path.name}")

    # --- Load the original USD mesh vertices and UVs ---
    usdc_files = list(usdz_extract_dir.rglob("*.usdc"))
    if not usdc_files:
        on_progress("No .usdc geometry file found — using fallback gray")
        return None

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(usdc_files[0]))

        mesh_prim = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = UsdGeom.Mesh(prim)
                break

        if mesh_prim is None:
            on_progress("No mesh prim in USD — using fallback gray")
            return None

        orig_points = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        orig_face_indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get())

        uv_primvar = None
        for attr_name in ["st", "st_0", "uv"]:
            pv = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar(attr_name)
            if pv and pv.IsDefined():
                uv_primvar = pv
                break

        if uv_primvar is None:
            on_progress("No UV primvar in USD — using fallback gray")
            return None

        uv_values = np.array(uv_primvar.Get(), dtype=np.float32)
        uv_indices = uv_primvar.GetIndices()

        per_vertex_uvs = np.zeros((len(orig_points), 2), dtype=np.float32)
        uv_count = np.zeros(len(orig_points), dtype=np.int32)

        if uv_indices is not None and len(uv_indices) > 0:
            uv_indices_arr = np.array(uv_indices)
            for vi, uv_i in zip(orig_face_indices, uv_indices_arr):
                per_vertex_uvs[vi] += uv_values[uv_i]
                uv_count[vi] += 1
        else:
            for face_vi, vi in enumerate(orig_face_indices):
                if face_vi < len(uv_values):
                    per_vertex_uvs[vi] += uv_values[face_vi]
                    uv_count[vi] += 1

        mask = uv_count > 0
        per_vertex_uvs[mask] /= uv_count[mask, np.newaxis]

    except Exception as e:
        on_progress(f"USD UV extraction failed: {e} — using fallback gray")
        return None

    # --- Load the diffuse texture ---
    try:
        diffuse_img = Image.open(texture_path).convert("RGB")
        diffuse_np = np.array(diffuse_img, dtype=np.float32) / 255.0
        tex_h, tex_w = diffuse_np.shape[:2]
    except Exception as e:
        on_progress(f"Failed to load diffuse texture: {e} — using fallback gray")
        return None

    on_progress("Transferring USDZ colors to decimated mesh vertices...")

    # --- KD-tree: find nearest original vertex for each decimated mesh vertex ---
    decimated_verts = np.array(mesh.vertices, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(orig_points.astype(np.float64))
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    vertex_colors = np.zeros((len(decimated_verts), 3), dtype=np.float32)

    for i, v in enumerate(decimated_verts):
        [_, idx, _] = kd_tree.search_knn_vector_3d(v.astype(np.float64), 1)
        orig_idx = idx[0]

        u, v_uv = per_vertex_uvs[orig_idx]
        tex_x = int(np.clip(u * (tex_w - 1), 0, tex_w - 1))
        tex_y = int(np.clip((1.0 - v_uv) * (tex_h - 1), 0, tex_h - 1))

        vertex_colors[i] = diffuse_np[tex_y, tex_x]

    on_progress(
        f"Color transfer complete: {len(decimated_verts):,} vertices "
        f"sampled from {texture_path.name}"
    )
    return vertex_colors


# ---------------------------------------------------------------------------
# Albedo baking — COLMAP colored point cloud path
# ---------------------------------------------------------------------------

def bake_albedo_from_pointcloud(mesh, pcd_path, on_progress):
    """
    Assign albedo colors to mesh vertices by finding the nearest colored
    point in a COLMAP-reconstructed point cloud.

    COLMAP stores RGB color at every reconstructed 3D point (from the
    source photo colors used during feature matching). Both the dense
    fused.ply (from patch-match stereo, if CUDA ran) and the sparse
    points3D.ply (always present) contain this color data.

    Args:
        mesh:        trimesh.Trimesh — the UV-mapped decimated mesh.
        pcd_path:    Path to the PLY point cloud file.
        on_progress: Callback for status messages.

    Returns:
        (N, 3) float32 per-vertex RGB colors in [0, 1], or None if unavailable.
    """
    import open3d as o3d

    if not Path(pcd_path).exists():
        on_progress(f"Point cloud not found: {pcd_path}")
        return None

    on_progress(f"Loading colored point cloud: {Path(pcd_path).name}...")

    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
    except Exception as e:
        on_progress(f"Failed to load point cloud: {e}")
        return None

    if len(pcd.points) == 0:
        on_progress("Point cloud is empty")
        return None

    if not pcd.has_colors():
        on_progress("Point cloud has no color data")
        return None

    source_verts = np.asarray(pcd.points, dtype=np.float32)
    source_colors = np.asarray(pcd.colors, dtype=np.float32)  # Already [0, 1]

    on_progress(
        f"Point cloud loaded: {len(source_verts):,} points. "
        "Assigning colors to mesh vertices..."
    )

    query_verts = np.array(mesh.vertices, dtype=np.float32)
    vertex_colors = _nearest_neighbor_colors(query_verts, source_verts, source_colors)

    on_progress(f"Color assignment complete: {len(query_verts):,} vertices colored")
    return vertex_colors


# ---------------------------------------------------------------------------
# Vertex colors → texture image
# ---------------------------------------------------------------------------

def vertex_colors_to_texture(uvs, faces, vertex_colors, image_size=DEFAULT_TEXTURE_SIZE):
    """
    Rasterize per-vertex RGB colors into a 2D texture image with dilation.

    This is the final step of the albedo baking pipeline for the COLMAP path:
    takes per-vertex colors (assigned from point cloud) and paints them into
    UV texture space using barycentric interpolation. Texture dilation (4px)
    is applied after rasterization to prevent dark seam artifacts.

    Args:
        uvs:           (N, 2) float32 UV coordinates in [0, 1]² space.
        faces:         (F, 3) int32 face indices.
        vertex_colors: (N, 3) float32 RGB colors in [0, 1].
        image_size:    int — output texture resolution (default 2048).

    Returns:
        PIL Image (RGB, image_size×image_size).
    """
    # Rasterize the colored UV mesh into a float image.
    img_data = _rasterize_triangles(uvs, faces, vertex_colors, image_size)

    # Track which pixels were filled by the rasterizer (inside UV islands).
    # np.any(> 0) is a proxy: truly black RGB texels could be missed, but
    # this is an acceptable edge case for point cloud data.
    island_mask = np.any(img_data > 0, axis=2)

    # Apply texture dilation: fill DILATION_PIXELS-wide border around islands.
    img_data = _dilate_texture(img_data, island_mask)

    # Clamp to [0, 1] and convert to uint8.
    img_uint8 = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(img_uint8, "RGB")


# ---------------------------------------------------------------------------
# Normal map baking
# ---------------------------------------------------------------------------

def _compute_tangent(normal):
    """
    Compute a tangent vector perpendicular to the given normal.

    Used to build the TBN (Tangent-Bitangent-Normal) frame for each vertex.
    Handles the degenerate case where the normal is nearly parallel to the
    default up vector by falling back to a different reference axis.

    Args:
        normal: (3,) float32 unit normal vector.

    Returns:
        (3,) float32 unit tangent vector.
    """
    if abs(normal[2]) < 0.99:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    tangent = np.cross(ref, normal)
    norm = np.linalg.norm(tangent)
    if norm < 1e-8:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        tangent /= norm
    return tangent


def bake_normal_map(mesh, image_size=DEFAULT_TEXTURE_SIZE):
    """
    Generate a tangent-space normal map from the mesh's own vertex normals.

    The normal map encodes each vertex's surface orientation relative to its
    local tangent frame. For a mesh with smooth normals, this captures surface
    curvature at the polygon level. Game engines use this for per-pixel shading.

    Note: For the Apple Object Capture path, bake_usdz_maps_texelspace() bakes
    the normal map directly from the USDZ *_norm0.png (which captures the
    original high-poly surface detail). This function is used as a fallback
    when USDZ normal extraction is unavailable (COLMAP path, or extraction fail).

    Encoding convention (OpenGL standard):
        R = (Nx + 1) / 2  → [0, 255], neutral = 128 (X=0)
        G = (Ny + 1) / 2  → [0, 255], neutral = 128 (Y=0)
        B = (Nz + 1) / 2  → [128, 255], neutral = 255 (Z=1, pointing out)

    Args:
        mesh:       trimesh.Trimesh with UV coordinates at mesh.visual.uv.
        image_size: int — output texture resolution.

    Returns:
        PIL Image (RGB, image_size×image_size).
    """
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    uvs = np.array(mesh.visual.uv, dtype=np.float32)
    vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

    n = len(vertices)

    # --- Accumulate tangent and bitangent vectors per vertex from UV gradients ---
    tangents = np.zeros((n, 3), dtype=np.float32)
    bitangents = np.zeros((n, 3), dtype=np.float32)

    for face in faces:
        i0, i1, i2 = face[0], face[1], face[2]

        dv1 = vertices[i1] - vertices[i0]
        dv2 = vertices[i2] - vertices[i0]

        duv1 = uvs[i1] - uvs[i0]
        duv2 = uvs[i2] - uvs[i0]

        denom = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        if abs(denom) < 1e-10:
            continue

        r = 1.0 / denom
        tangent = (duv2[1] * dv1 - duv1[1] * dv2) * r
        bitangent = (duv1[0] * dv2 - duv2[0] * dv1) * r

        for idx in [i0, i1, i2]:
            tangents[idx] += tangent
            bitangents[idx] += bitangent

    # --- Compute tangent-space normals per vertex ---
    tangent_space_normals = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        normal = vertex_normals[i]
        tangent = tangents[i]

        # Gram-Schmidt orthogonalization.
        tangent = tangent - np.dot(tangent, normal) * normal
        t_len = np.linalg.norm(tangent)

        if t_len < 1e-8:
            tangent = _compute_tangent(normal)
        else:
            tangent /= t_len

        bitangent = np.cross(normal, tangent)

        # TBN^T transforms world-space normal to tangent space.
        tbn_T = np.stack([tangent, bitangent, normal], axis=0)
        ts_normal = tbn_T @ normal
        tangent_space_normals[i] = ts_normal

    # --- Encode tangent-space normals as RGB ---
    encoded = (tangent_space_normals + 1.0) / 2.0  # (N, 3) in [0, 1]

    # Rasterize into texture image.
    img_data = _rasterize_triangles(uvs, faces, encoded, image_size)

    # Track which pixels are inside UV islands (before filling background).
    # The normal map is never pure black inside a valid triangle, so any
    # non-zero pixel indicates an island pixel.
    island_mask = np.any(img_data > 0, axis=2)

    # Apply texture dilation: expand island borders by DILATION_PIXELS.
    img_data = _dilate_texture(img_data, island_mask)

    # Fill remaining unfilled pixels (further than DILATION_PIXELS from any island)
    # with the flat normal direction (pointing straight out of the surface).
    still_unfilled = (
        (img_data[:, :, 0] == 0)
        & (img_data[:, :, 1] == 0)
        & (img_data[:, :, 2] == 0)
    )
    img_data[still_unfilled] = [0.5, 0.5, 1.0]

    img_uint8 = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(img_uint8, "RGB")


# ---------------------------------------------------------------------------
# AO baking via hemisphere ray casting
# ---------------------------------------------------------------------------

def _generate_hemisphere_samples(num_samples):
    """
    Generate uniformly distributed ray directions in a Z-up unit hemisphere.

    Uses cosine-weighted sampling: rays are more concentrated near the pole
    (normal direction) and sparser near the horizon. This gives better AO
    quality than uniform sampling for the same ray count because near-horizon
    rays contribute less to diffuse illumination (Lambert's cosine law).

    Args:
        num_samples: int — number of ray directions to generate.

    Returns:
        (num_samples, 3) float32 unit vectors in the Z-up hemisphere.
    """
    # Cosine-weighted hemisphere sampling via Malley's method.
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    u1 = rng.uniform(0, 1, num_samples)
    u2 = rng.uniform(0, 1, num_samples)

    theta = np.arccos(np.sqrt(u1))
    phi = 2.0 * np.pi * u2

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def bake_ao(mesh, num_rays=64, image_size=DEFAULT_TEXTURE_SIZE, on_progress=None):
    """
    Compute per-vertex ambient occlusion by hemisphere ray casting.

    For each vertex, casts num_rays rays distributed over a hemisphere oriented
    along the vertex normal. AO = 1 - (fraction of rays that hit the mesh).
    Unoccluded vertices (open areas) get bright AO values; occluded vertices
    (corners, crevices) get dark values.

    Note: For the Apple Object Capture path, bake_usdz_maps_texelspace() bakes
    AO directly from the USDZ *_ao0.png (which is RealityKit's own computed AO).
    This function is used as a fallback for COLMAP or when USDZ AO is unavailable.

    Ray casting is performed by Open3D's RaycastingScene (C++ BVH). All rays
    are submitted in a single batch for maximum performance.

    Args:
        mesh:        trimesh.Trimesh with UV coordinates at mesh.visual.uv.
        num_rays:    int — rays per vertex (more = less noise, slower).
        image_size:  int — output texture resolution.
        on_progress: Callback for status messages.

    Returns:
        PIL Image (greyscale 'L', image_size×image_size).
    """
    import open3d as o3d

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    uvs = np.array(mesh.visual.uv, dtype=np.float32)
    vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

    n = len(vertices)

    if on_progress:
        on_progress(
            f"Building AO raycasting scene ({n:,} vertices, "
            f"{num_rays} rays each)..."
        )

    # --- Build Open3D raycasting scene from the mesh ---
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)

    # --- Precompute hemisphere ray directions ---
    hemisphere_local = _generate_hemisphere_samples(num_rays)

    if on_progress:
        on_progress(f"Casting {n * num_rays:,} AO rays...")

    # --- Build all rays in a single batch ---
    all_origins = np.zeros((n * num_rays, 3), dtype=np.float32)
    all_dirs = np.zeros((n * num_rays, 3), dtype=np.float32)

    for i in range(n):
        normal = vertex_normals[i]

        tangent = _compute_tangent(normal)
        bitangent = np.cross(normal, tangent)

        R = np.stack([tangent, bitangent, normal], axis=1)
        dirs = hemisphere_local @ R.T

        origin = vertices[i] + normal * AO_RAY_OFFSET

        start = i * num_rays
        end = start + num_rays
        all_origins[start:end] = origin
        all_dirs[start:end] = dirs

    # --- Cast all rays in one Open3D call ---
    rays = np.concatenate([all_origins, all_dirs], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    hit_counts = scene.count_intersections(rays_tensor).numpy()

    # --- Compute per-vertex AO values ---
    hit_matrix = (hit_counts > 0).reshape(n, num_rays)
    ao_values = 1.0 - hit_matrix.mean(axis=1).astype(np.float32)

    if on_progress:
        avg_ao = ao_values.mean()
        on_progress(
            f"AO complete: avg {avg_ao:.2f} (1=lit, 0=occluded). "
            "Rasterizing to texture..."
        )

    # --- Rasterize per-vertex AO values into texture image ---
    ao_2d = ao_values.reshape(-1, 1)
    img_data = _rasterize_triangles(uvs, faces, ao_2d, image_size)
    # img_data shape: (H, W, 1)

    # Track island coverage before any gap-fill.
    island_mask = img_data[:, :, 0] != 0

    # Apply texture dilation: fill DILATION_PIXELS border with island AO values.
    img_data = _dilate_texture(img_data, island_mask)

    # Fill remaining unfilled pixels with fully lit (1.0) — safe default for
    # areas far from geometry (background / UV padding).
    still_unfilled = img_data[:, :, 0] == 0
    img_data[still_unfilled, 0] = 1.0

    # Convert to greyscale PIL Image.
    img_uint8 = (np.clip(img_data[:, :, 0], 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(img_uint8, "L")


# ---------------------------------------------------------------------------
# Roughness map estimation (Phase 2d) — COLMAP / fallback path
# ---------------------------------------------------------------------------

def bake_roughness_map(albedo_path, ao_path, output_path,
                       image_size=DEFAULT_TEXTURE_SIZE):
    """
    Estimate a roughness map from the baked albedo and AO textures.

    Works entirely in image space — no UV rasterization pass needed since
    the albedo is already a fully-resolved 2D texture.

    Note: For the Apple Object Capture path, bake_usdz_maps_texelspace() bakes
    roughness directly from the USDZ *_roughness0.png. This function is used
    as a fallback for COLMAP or when USDZ roughness extraction fails.

    Algorithm:
    1. Convert albedo RGB to HSV (vectorised numpy — no cv2 dependency).
    2. Specularity proxy = Value × (1 − Saturation).
       - Bright + desaturated pixel → high specularity → low roughness.
       - Dark or saturated pixel → low specularity → high roughness.
    3. roughness = 1.0 − specularity.
    4. Blend with AO: crevices (low AO) get a +15% roughness boost.
    5. Clamp to [0.2, 1.0] — photogrammetry never produces perfect mirrors.

    Args:
        albedo_path:  Path to the baked albedo.png.
        ao_path:      Path to the baked ao.png (optional AO blend).
        output_path:  Destination path for roughness.png.
        image_size:   int — output texture resolution (default 2048).

    Returns:
        PIL Image (greyscale 'L', image_size×image_size).
    """
    albedo_img = Image.open(albedo_path).convert("RGB")

    if albedo_img.size != (image_size, image_size):
        albedo_img = albedo_img.resize((image_size, image_size), Image.LANCZOS)

    albedo = np.array(albedo_img, dtype=np.float32) / 255.0
    r, g, b = albedo[..., 0], albedo[..., 1], albedo[..., 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Safe divide: replace zero denominators with 1.0 so numpy doesn't raise a
    # RuntimeWarning on pure-black pixels. np.where evaluates BOTH branches
    # before selecting, so the division happens for all pixels. The mask in the
    # outer np.where then zeroes out the result wherever max_c was 0.
    safe_max_c = np.where(max_c > 1e-6, max_c, 1.0)
    saturation = np.where(max_c > 1e-6, delta / safe_max_c, 0.0).astype(np.float32)
    value = max_c

    specularity = value * (1.0 - saturation)
    roughness = 1.0 - specularity

    if ao_path is not None and Path(ao_path).exists():
        ao_img = Image.open(ao_path).convert("L")
        if ao_img.size != (image_size, image_size):
            ao_img = ao_img.resize((image_size, image_size), Image.LANCZOS)
        ao = np.array(ao_img, dtype=np.float32) / 255.0
        roughness = roughness + (1.0 - ao) * 0.15

    roughness = np.clip(roughness, 0.2, 1.0)

    roughness_uint8 = (roughness * 255).astype(np.uint8)
    img = Image.fromarray(roughness_uint8, "L")
    img.save(str(output_path))
    return img


# ---------------------------------------------------------------------------
# Metallic map estimation (Phase 2d) — COLMAP / fallback path
# ---------------------------------------------------------------------------

def bake_metallic_map(albedo_path, output_path,
                      image_size=DEFAULT_TEXTURE_SIZE):
    """
    Estimate a metallic map from the baked albedo texture.

    Works entirely in image space — no UV rasterization pass needed.

    Note: For the Apple Object Capture path, bake_usdz_maps_texelspace() bakes
    metallic directly from the USDZ *_metalness0.png. This function is used
    as a fallback for COLMAP or when USDZ metallic extraction fails.

    Algorithm:
    1. Convert albedo RGB to HSV.
    2. Metallic score = (1 − Saturation) × Value.
    3. Soft threshold at 0.7–0.9: below → non-metallic, above → metallic.
    4. Conservative threshold prevents false positives on white/overexposed surfaces.

    Args:
        albedo_path: Path to the baked albedo.png.
        output_path: Destination path for metallic.png.
        image_size:  int — output texture resolution (default 2048).

    Returns:
        PIL Image (greyscale 'L', image_size×image_size).
    """
    albedo_img = Image.open(albedo_path).convert("RGB")
    if albedo_img.size != (image_size, image_size):
        albedo_img = albedo_img.resize((image_size, image_size), Image.LANCZOS)

    albedo = np.array(albedo_img, dtype=np.float32) / 255.0
    r, g, b = albedo[..., 0], albedo[..., 1], albedo[..., 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    # Safe divide — same pattern as bake_roughness_map. np.where computes both
    # branches before masking, so a direct delta/max_c fires on black pixels.
    # Replacing zero denominators with 1.0 before dividing silences the warning;
    # the outer np.where then correctly yields 0.0 for those pixels anyway.
    safe_max_c = np.where(max_c > 1e-6, max_c, 1.0)
    saturation = np.where(max_c > 1e-6, delta / safe_max_c, 0.0).astype(np.float32)
    value = max_c

    metallic_score = (1.0 - saturation) * value
    metallic = np.clip((metallic_score - 0.7) / 0.2, 0.0, 1.0).astype(np.float32)

    metallic_uint8 = (metallic * 255).astype(np.uint8)
    img = Image.fromarray(metallic_uint8, "L")
    img.save(str(output_path))
    return img
