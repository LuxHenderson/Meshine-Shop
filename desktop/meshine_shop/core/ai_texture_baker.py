"""
AI-enhanced texture synthesis for Meshine Shop (Phase 3).

Provides two complementary algorithms that produce cleaner PBR albedo textures
from photogrammetry reconstructions by operating on the source photo data
rather than the already-baked output:

    apply_retinex_delighting() — Multi-Scale Retinex Color Restoration (MSRCR).
        Removes baked directional lighting from photogrammetry albedo using
        Edwin Land's Retinex theory in log-domain:
            - Illumination field L estimated at 3 scales via Gaussian blur
            - Surface reflectance R = log(I) - log(L)  (log-domain subtraction)
            - Rahman color restoration factor re-introduces hue balance
        Applied as a configurable blend with the original albedo to avoid
        destroying genuine dark surfaces.
        No new dependencies — uses existing numpy and scipy.ndimage.

    bake_multiview_albedo_colmap() — Multi-view photo projection.
        Loads COLMAP camera poses from the sparse reconstruction, projects each
        source photo onto the UV-mapped mesh vertices from every registered
        camera angle, and accumulates cosine-angle-weighted colors:
            w = max(0, vertex_normal · unit_vector_to_camera)
        Averaging colors across many viewpoints from different directions
        cancels out directional lighting that appears in any single photo.
        No new dependencies — uses existing pycolmap, numpy, PIL, and the
        vertex_colors_to_texture() rasterizer from texture_baker.py.

Both functions are designed to be called from bake_textures() in the engine
implementations and are non-fatal: exceptions are caught by the caller so the
pipeline continues with the best albedo available.

Phase 3 integration:
    Apple engine: apply_retinex_delighting() after enhance_albedo_clarity()
    COLMAP engine: bake_multiview_albedo_colmap() before point-cloud fallback,
                   then apply_retinex_delighting() as a final pass on albedo
"""

import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Multi-Scale Retinex Color Restoration (MSRCR)
# ---------------------------------------------------------------------------

def apply_retinex_delighting(textures_dir, on_progress=None, blend=0.7):
    """
    Apply Multi-Scale Retinex Color Restoration to albedo.png.

    Removes baked directional lighting from photogrammetry albedo using Edwin
    Land's Retinex theory. In log-domain, the image I = L × R (illumination
    times reflectance). Gaussian blur estimates L; the residual log(I) - log(L)
    isolates R (true surface color, free of lighting direction).

    Three blur scales (σ = 15, 80, 250 px) target fine shadow edges, mid-scale
    object shading, and large global illumination gradients respectively.
    Rahman's color restoration factor (N=125) re-introduces the hue balance
    that flat Retinex tends to wash out.

    The Retinex result is blended with the original image so that genuine dark
    surfaces (dark leather, black paint) are not over-brightened:
        blend = 0.0  →  pure original, no effect
        blend = 0.7  →  default, softens shadows while preserving dark zones
        blend = 1.0  →  full Retinex, maximum de-lighting

    Args:
        textures_dir: Path to workspace/textures/ containing albedo.png.
        on_progress:  Optional callback(str) for status messages shown in UI.
        blend:        Weight of the Retinex result vs. original (0.0–1.0).

    Side-effects:
        Overwrites textures_dir/albedo.png in-place with the de-lit image.
        No-ops silently if albedo.png does not exist.
    """
    from scipy.ndimage import gaussian_filter

    def _log(msg):
        if on_progress:
            on_progress(msg)

    albedo_path = Path(textures_dir) / "albedo.png"
    if not albedo_path.exists():
        _log("Retinex: albedo.png not found — skipping")
        return

    _log("Applying Multi-Scale Retinex de-lighting to albedo...")

    # Load as float32 in [0, 1] — keep the original for blending later.
    img = Image.open(albedo_path).convert("RGB")
    original = np.array(img, dtype=np.float32) / 255.0

    # Scale to [0, 255] for Land's original log-domain formulation.
    arr = original * 255.0

    # ---- Multi-Scale Retinex at three spatial scales ----
    # σ =  15 px → fine-scale (texel-level shadow edges, small surface features)
    # σ =  80 px → mid-scale  (object-level shading, cast shadows)
    # σ = 250 px → large-scale (global illumination gradient across the image)
    sigmas = [15, 80, 250]
    log_img = np.log1p(arr)   # log(1 + I), shape (H, W, 3) — avoids log(0)
    msr = np.zeros_like(log_img)

    for sigma in sigmas:
        # Estimate illumination at this scale via per-channel Gaussian blur.
        # sigma=[sigma, sigma, 0] blurs spatially but not across channels.
        log_blur = np.log1p(gaussian_filter(arr, sigma=[sigma, sigma, 0]))
        msr += log_img - log_blur   # Retinex response = log(I) - log(L)

    msr /= len(sigmas)   # Average response across all three scales.

    # ---- Rahman color restoration factor ----
    # Without this step, Retinex tends to flatten colors toward neutral gray
    # because it independently normalises each channel. The restoration factor
    # re-weights each channel by the ratio of that channel to the total
    # luminance, preserving the original hue balance.
    # N=125 is Rahman's standard constant (Jobson, Rahman & Woodell, 1997).
    channel_sum = np.sum(arr, axis=2, keepdims=True) + 1e-6
    N = 125.0
    color_factor = np.log1p(N * arr / channel_sum)   # (H, W, 3) per-channel
    msrcr = msr * color_factor                        # final MSRCR response

    # ---- Per-channel percentile normalisation ----
    # Remap each channel independently from [1st, 99th] percentile → [0, 1].
    # Clipping before remapping prevents outlier texels from compressing the
    # visible range. A black patch or bright specular shouldn't dominate.
    retinex = np.zeros_like(msrcr, dtype=np.float32)
    for c in range(3):
        lo = np.percentile(msrcr[:, :, c], 1)
        hi = np.percentile(msrcr[:, :, c], 99)
        if hi > lo:
            retinex[:, :, c] = np.clip(
                (msrcr[:, :, c] - lo) / (hi - lo), 0.0, 1.0
            ).astype(np.float32)
        else:
            # Degenerate channel (constant value) — set to neutral 0.5.
            retinex[:, :, c] = 0.5

    # ---- Blend Retinex result with original ----
    # blend=0.7 keeps 70% de-lit + 30% original, preventing over-correction
    # of surfaces that are supposed to be dark (dark leather, black paint).
    result = blend * retinex + (1.0 - blend) * original
    result = np.clip(result, 0.0, 1.0)

    # ---- Save in-place ----
    out = Image.fromarray((result * 255).astype(np.uint8))
    out.save(str(albedo_path))

    _log("Retinex de-lighting complete — albedo.png updated")


# ---------------------------------------------------------------------------
# Multi-view photo projection (COLMAP path only)
# ---------------------------------------------------------------------------

def bake_multiview_albedo_colmap(mesh, workspace, on_progress=None, image_size=2048):
    """
    Bake albedo by projecting source photos onto mesh vertices from all cameras.

    Loads the COLMAP sparse reconstruction to get registered camera poses, then
    for each registered source photo:

        1. Transform mesh vertices to camera space: X_cam = R @ X_world + t
        2. Depth-cull: discard vertices behind the camera (z ≤ 0)
        3. Project to image plane using camera intrinsics: (u, v) = K @ X_cam / z
        4. Bounds-cull: discard vertices that project outside the image
        5. Compute cosine weight: w = max(0, vertex_normal · unit_to_camera)
           (upweights surface patches facing the camera, downweights grazing)
        6. Bilinearly sample the source photo at (u, v)
        7. Accumulate: color_sum[v] += w × color,  weight_sum[v] += w

    After all cameras, normalise: albedo[v] = color_sum[v] / weight_sum[v]

    Averaging colors from many viewpoints in different directions cancels out
    the baked directional lighting that appears in any single photo — the key
    insight is that the object's surface color is constant across all photos
    while the lighting contribution changes direction with each camera.

    Args:
        mesh:        trimesh.Trimesh with UV coordinates (meshed_uv.obj).
        workspace:   WorkspacePaths with .sparse, .images, .textures attributes.
        on_progress: Optional callback(str) for status messages shown in UI.
        image_size:  Output texture resolution in pixels (e.g. 2048 for PC).

    Returns:
        True  — multi-view projection succeeded; albedo.png written.
        False — fewer than 3 cameras registered, or vertex coverage < 30%;
                caller should fall back to point-cloud nearest-neighbour bake.

    Side-effects:
        Writes workspace.textures/albedo.png if successful.
        No-ops and returns False if sparse reconstruction directory is missing.
    """
    import pycolmap

    def _log(msg):
        if on_progress:
            on_progress(msg)

    # ---- Locate the sparse reconstruction directory ----
    # COLMAP writes numbered sub-directories (0, 1, ...) inside workspace/sparse/.
    # We use "0" (the primary reconstruction) and fall back to whichever exists.
    sparse_dir = workspace.sparse / "0"
    if not sparse_dir.exists():
        candidates = (
            sorted(workspace.sparse.iterdir())
            if workspace.sparse.exists()
            else []
        )
        if not candidates:
            _log("Multi-view albedo: no sparse reconstruction found — falling back")
            return False
        sparse_dir = candidates[0]

    # ---- Load the COLMAP reconstruction ----
    try:
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(str(sparse_dir))
    except Exception as exc:
        _log(f"Multi-view albedo: could not load reconstruction ({exc}) — falling back")
        return False

    images  = reconstruction.images   # dict {image_id: Image}
    cameras = reconstruction.cameras  # dict {camera_id: Camera}

    if len(images) < 3:
        _log(
            f"Multi-view albedo: only {len(images)} camera(s) registered "
            "(need ≥ 3) — falling back"
        )
        return False

    _log(f"Multi-view albedo: projecting source photos from {len(images)} cameras...")

    # ---- Mesh data ----
    vertices       = np.array(mesh.vertices,       dtype=np.float64)  # (N, 3)
    vertex_normals = np.array(mesh.vertex_normals, dtype=np.float64)  # (N, 3)
    n_verts = len(vertices)

    # Accumulate weighted colors and weights per vertex across all cameras.
    color_acc  = np.zeros((n_verts, 3), dtype=np.float64)
    weight_acc = np.zeros(n_verts,      dtype=np.float64)

    # ---- Project each registered image onto the mesh ----
    for img_id, colmap_image in images.items():

        # Locate the source photo in the workspace images directory.
        # ingest_images() converts everything to JPEG (stem.jpg), so we try
        # the registered name first, then the stem with .jpg extension.
        img_name = Path(colmap_image.name)
        img_path = workspace.images / img_name.name
        if not img_path.exists():
            img_path = workspace.images / (img_name.stem + ".jpg")
        if not img_path.exists():
            continue

        try:
            photo = (
                np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
                / 255.0
            )
        except Exception:
            continue

        photo_h, photo_w = photo.shape[:2]

        # ---- Camera extrinsics: world → camera ----
        # The world-to-camera transform is: X_cam = R @ X_world + t
        try:
            R = colmap_image.rotation_matrix()   # 3×3 numpy array
        except AttributeError:
            # Older pycolmap: derive R from the stored quaternion.
            R = pycolmap.qvec2rotmat(colmap_image.qvec)
        t = np.array(colmap_image.tvec, dtype=np.float64)   # (3,)

        # Camera centre in world space: C = -R.T @ t
        # Used to compute the to-camera direction for cosine weighting.
        cam_center = -R.T @ t   # (3,)

        # ---- Camera intrinsics ----
        camera = cameras[colmap_image.camera_id]
        fx, fy, cx, cy = _get_camera_intrinsics(camera)

        # ---- Transform all mesh vertices to camera space (vectorised) ----
        # (R @ vertices.T) has shape (3, N); adding t broadcasts over N verts.
        X_cam = (R @ vertices.T).T + t   # (N, 3)

        # ---- Depth test: discard vertices behind the camera (z ≤ 0.01) ----
        valid = X_cam[:, 2] > 0.01

        # ---- Project to image plane ----
        z  = np.maximum(X_cam[:, 2], 1e-8)       # prevent division by zero
        u_px = fx * X_cam[:, 0] / z + cx          # pixel column (float)
        v_px = fy * X_cam[:, 1] / z + cy          # pixel row    (float)

        # Bounds check: keep only vertices that project inside the image.
        in_bounds = (
            (u_px >= 0) & (u_px < photo_w - 1) &
            (v_px >= 0) & (v_px < photo_h - 1)
        )
        valid = valid & in_bounds

        if not valid.any():
            continue

        # ---- Cosine angle weight ----
        # w = max(0, vertex_normal · (camera_centre − vertex) / |(…)|)
        # This upweights surface patches that face the camera squarely and
        # downweights grazing-angle projections, which are noisier.
        to_cam      = cam_center[None, :] - vertices               # (N, 3)
        to_cam_len  = np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-8
        to_cam_unit = to_cam / to_cam_len                          # (N, 3) unit
        cos_theta   = np.einsum('ni,ni->n', vertex_normals, to_cam_unit)  # (N,)
        weights     = np.maximum(0.0, cos_theta)                   # (N,) ≥ 0
        weights[~valid] = 0.0   # Zero out vertices that failed depth/bounds tests.

        # ---- Bilinear colour sampling for valid vertices ----
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue

        u_f = u_px[idx]   # float pixel-x for each valid vertex
        v_f = v_px[idx]   # float pixel-y

        # Integer corners of the 2×2 bilinear kernel.
        u0 = np.floor(u_f).astype(int)
        v0 = np.floor(v_f).astype(int)
        u1 = np.minimum(u0 + 1, photo_w - 1)
        v1 = np.minimum(v0 + 1, photo_h - 1)

        # Sub-pixel fractional offsets for bilinear interpolation.
        uf = (u_f - u0)[:, None]   # (M, 1) — fraction along x
        vf = (v_f - v0)[:, None]   # (M, 1) — fraction along y

        # Bilinearly interpolated colour: (M, 3)
        sampled = (
            (1 - uf) * (1 - vf) * photo[v0, u0] +
            uf       * (1 - vf) * photo[v0, u1] +
            (1 - uf) * vf       * photo[v1, u0] +
            uf       * vf       * photo[v1, u1]
        )

        # ---- Accumulate into buffers ----
        w = weights[idx, None]       # (M, 1) — broadcast over colour channels
        color_acc[idx]  += w * sampled
        weight_acc[idx] += weights[idx]

    # ---- Normalise accumulated colours ----
    has_weight = weight_acc > 0
    coverage   = has_weight.sum() / n_verts
    _log(
        f"Multi-view albedo: {coverage:.1%} vertex coverage "
        f"({has_weight.sum():,}/{n_verts:,} vertices)"
    )

    # Require at least 30% coverage — below this the projection is too sparse
    # and the point-cloud fallback will give better spatial coverage.
    if coverage < 0.30:
        _log("Multi-view albedo: coverage too low — falling back to point cloud")
        return False

    vertex_colors = np.zeros((n_verts, 3), dtype=np.float64)
    vertex_colors[has_weight] = (
        color_acc[has_weight] / weight_acc[has_weight, None]
    )

    # ---- Fill uncovered vertices with mean colour ----
    # Uncovered vertices are typically back-facing or hidden by other geometry.
    # Fill with the mean of covered vertices so the rasteriser doesn't produce
    # black patches in the final texture atlas.
    if not has_weight.all():
        mean_color = vertex_colors[has_weight].mean(axis=0)
        vertex_colors[~has_weight] = mean_color

    # ---- Rasterise vertex colours to texture ----
    # Reuse the existing UV rasteriser from texture_baker.py to convert per-
    # vertex colours into a texel-resolution atlas image.
    from meshine_shop.core.texture_baker import vertex_colors_to_texture

    uvs   = np.array(mesh.visual.uv, dtype=np.float32)
    faces = np.array(mesh.faces,     dtype=np.int32)

    _log("Rasterising multi-view albedo to texture atlas...")
    albedo_img = vertex_colors_to_texture(
        uvs, faces, vertex_colors, image_size=image_size
    )
    albedo_img.save(str(Path(workspace.textures) / "albedo.png"))
    _log("Multi-view albedo saved: albedo.png")

    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_camera_intrinsics(camera):
    """
    Extract (fx, fy, cx, cy) from a pycolmap Camera object.

    COLMAP camera models order their params differently depending on whether
    they use a single or dual focal length:

        Single-focal models (SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, …):
            params = [f, cx, cy, …]        → fx = fy = f
        Dual-focal models (PINHOLE, OPENCV, FOV, THIN_PRISM_FISHEYE, …):
            params = [fx, fy, cx, cy, …]   → fx ≠ fy allowed

    The default model used by pycolmap's extract_features() is SIMPLE_RADIAL
    (single focal length). PINHOLE and OPENCV are also common when explicitly
    chosen in the extraction options.

    Returns:
        (fx, fy, cx, cy) as float64 values.
    """
    params = np.array(camera.params, dtype=np.float64)

    # Retrieve the model name for type identification.
    # pycolmap >= 0.6 stores model as a CameraModelId enum with a .name attr.
    try:
        model_name = camera.model.name.upper()
    except AttributeError:
        model_name = str(camera.model).upper()

    # Dual-focal-length models — params layout: [fx, fy, cx, cy, …]
    # Includes: PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, THIN_PRISM.
    # Note: SIMPLE_PINHOLE is a *different* model from PINHOLE (single focal).
    dual_focal = (
        ('PINHOLE'    in model_name and 'SIMPLE_PINHOLE' not in model_name) or
        'OPENCV'      in model_name or
        'FOV'         in model_name or
        'THIN_PRISM'  in model_name
    )

    if dual_focal:
        # params = [fx, fy, cx, cy, …]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        # Single focal length — params = [f, cx, cy, …]
        # Covers: SIMPLE_PINHOLE, SIMPLE_RADIAL (default), RADIAL,
        #         SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE.
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f

    return fx, fy, cx, cy
