"""
AI texture generation via Stability AI — Phase 2 of the Meshine Shop pipeline.

This module replaces the old photo-color baking stage with AI-generated PBR textures
driven by a user-supplied material description. The workflow is:

    1. Load the UV-mapped mesh (meshed_uv.obj) from the workspace
    2. Generate 6 synthetic camera viewpoints around the mesh's bounding sphere
       (front, back, left, right, top, diagonal — Open3D Embree raycasting avoids
       the BVH hang that trimesh's pure-Python intersector had on diagonal angles)
    3. Render a depth map from each viewpoint using trimesh raycasting (no GPU needed)
    4. Send each depth render + the user's material prompt to the Stability AI
       structure control endpoint — the model generates a plausible textured view
       that follows the mesh's silhouette and surface features
    5. Back-project all generated views onto the UV layout using cosine-angle weighted
       blending (same projection math as the multi-view albedo projector)
    6. Compute an object-space normal map from the mesh's vertex normals
    7. Derive roughness and metallic values from material keywords in the prompt
    8. Write albedo.png, normal.png, roughness.png, metallic.png to workspace/textures/

The API key is read from the STABILITY_API_KEY environment variable, or from
~/.meshine_shop/config.json ({"stability_api_key": "sk-..."}). If neither is present,
the stage logs a warning and returns without writing textures — the pipeline continues
with an untextured mesh (same output as Phase 0).

Stability AI endpoint used:
    POST https://api.stability.ai/v2beta/stable-image/control/structure
    Auth: Authorization: Bearer {api_key}
    Body: multipart/form-data — image (depth PNG), prompt, control_strength, output_format
"""

import io
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
import requests
import trimesh
from PIL import Image

# ---------------------------------------------------------------------------
# Material preset library — keyword → (roughness, metallic) defaults
# ---------------------------------------------------------------------------
# Maps lowercase keywords that might appear in the user's prompt to physically
# correct PBR roughness and metallic values. First keyword match wins.
# Roughness: 0.0 = mirror-smooth, 1.0 = fully diffuse scatter
# Metallic:  0.0 = dielectric (plastic, wood, stone), 1.0 = conductor (metal)

MATERIAL_PRESETS = {
    # Metals — conductors with low roughness for reflective finishes
    "chrome":       (0.05, 1.0),
    "polished metal": (0.10, 1.0),
    "brushed metal": (0.40, 1.0),
    "metal":        (0.25, 1.0),
    "steel":        (0.30, 1.0),
    "iron":         (0.45, 1.0),
    "copper":       (0.30, 1.0),
    "gold":         (0.20, 1.0),
    "aluminum":     (0.15, 1.0),
    # Dielectrics — non-conducting materials with varying surface roughness
    "glass":        (0.05, 0.0),
    "ceramic":      (0.20, 0.0),
    "porcelain":    (0.15, 0.0),
    "plastic":      (0.40, 0.0),
    "rubber":       (0.90, 0.0),
    "leather":      (0.85, 0.0),
    "suede":        (0.95, 0.0),
    "fabric":       (0.95, 0.0),
    "cloth":        (0.95, 0.0),
    "cotton":       (0.95, 0.0),
    "denim":        (0.95, 0.0),
    "silk":         (0.55, 0.0),
    "wood":         (0.80, 0.0),
    "bark":         (0.90, 0.0),
    "stone":        (0.90, 0.0),
    "rock":         (0.90, 0.0),
    "concrete":     (0.90, 0.0),
    "brick":        (0.90, 0.0),
    "sand":         (0.95, 0.0),
    "dirt":         (0.95, 0.0),
    "mud":          (0.95, 0.0),
    "clay":         (0.90, 0.0),
    "paper":        (0.90, 0.0),
    "cardboard":    (0.95, 0.0),
    "foam":         (0.98, 0.0),
    "wax":          (0.35, 0.0),
    "skin":         (0.65, 0.0),
}

# Default PBR values when no keyword from the preset table matches the prompt.
_DEFAULT_ROUGHNESS = 0.70
_DEFAULT_METALLIC  = 0.00

# Stability AI structure control endpoint — v2beta is the current stable API.
_STABILITY_URL = "https://api.stability.ai/v2beta/stable-image/control/structure"

# Number of synthetic camera viewpoints used for depth rendering + API calls.
# 6 covers front/back/left/right/top and one diagonal — good hemisphere coverage.
# Open3D RaycastingScene (Embree-based) handles diagonal angles without hanging,
# unlike the trimesh pure-Python BVH which stalls on edge-on triangle traversal.
_N_VIEWS = 6

# Depth render resolution fed to the API. Smaller = faster + cheaper API calls.
# 512×512 is sufficient for structure control to capture mesh silhouette detail.
_RENDER_SIZE = 512

# Output texture atlas resolution in pixels.
_TEXTURE_SIZE = 2048


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    """
    Return the Stability AI API key, or None if not configured.

    Priority:
        1. STABILITY_API_KEY environment variable
        2. ~/.meshine_shop/config.json → {"stability_api_key": "sk-..."}
    """
    # Check env var first — preferred for CI/CD or advanced users.
    key = os.environ.get("STABILITY_API_KEY", "").strip()
    if key:
        return key

    # Fall back to the local config file stored outside the git repo.
    config_path = Path.home() / ".meshine_shop" / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            key = str(data.get("stability_api_key", "")).strip()
            if key:
                return key
        except Exception:
            pass  # Malformed JSON — fall through to return None

    return None


# ---------------------------------------------------------------------------
# Material PBR parameter extraction
# ---------------------------------------------------------------------------

def _material_pbr_params(prompt: str) -> tuple[float, float]:
    """
    Scan the prompt for material keywords and return (roughness, metallic).

    The keyword that appears EARLIEST in the prompt string wins, so the
    primary material described first takes precedence over incidentally
    mentioned materials (e.g. "leather costume with metal buckles" → leather,
    not metal). If no keyword matches, returns the defaults.

    Args:
        prompt: The user's material description string.

    Returns:
        Tuple of (roughness, metallic), each in the range [0.0, 1.0].
    """
    lower = prompt.lower()

    # Find the position of each keyword in the prompt; keep whichever appears first.
    best_pos = len(lower) + 1  # sentinel — beyond end of string
    best_params = (_DEFAULT_ROUGHNESS, _DEFAULT_METALLIC)

    for keyword, params in MATERIAL_PRESETS.items():
        pos = lower.find(keyword)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_params = params

    return best_params


# ---------------------------------------------------------------------------
# Synthetic camera generation
# ---------------------------------------------------------------------------

def _synthetic_cameras(bounds: np.ndarray) -> list[dict]:
    """
    Generate N_VIEWS synthetic camera poses orbiting the mesh bounding sphere.

    Places cameras at 6 standard positions (front, back, left, right, top, and
    a 45° front-right diagonal). Each camera looks toward the mesh centroid from
    a distance of 2.5× the bounding sphere radius, so the entire mesh is visible
    in every depth render regardless of its shape.

    Args:
        bounds: (2, 3) array — [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    Returns:
        List of dicts with keys:
            R       — (3, 3) rotation matrix (world → camera)
            t       — (3,) translation vector (camera origin in world space)
            fov_deg — horizontal field of view in degrees (constant 60°)
    """
    # Compute mesh centroid and bounding sphere radius.
    centroid = (bounds[0] + bounds[1]) / 2.0
    radius = np.linalg.norm(bounds[1] - bounds[0]) / 2.0
    dist = 2.5 * max(radius, 1e-6)  # Guard against degenerate (near-zero) meshes

    # 6 viewpoint directions in world space — each points FROM the camera TO the centroid.
    # Diagonal is included now that depth rendering uses Open3D's Embree-based
    # RaycastingScene, which handles any ray angle without hanging.
    directions = [
        np.array([0.0,  0.0, -1.0]),        # Front         (+Z → looking -Z)
        np.array([0.0,  0.0,  1.0]),        # Back          (-Z → looking +Z)
        np.array([-1.0, 0.0,  0.0]),        # Left          (+X → looking -X)
        np.array([1.0,  0.0,  0.0]),        # Right         (-X → looking +X)
        np.array([0.0, -1.0,  0.0]),        # Top           (+Y → looking -Y)
        np.array([-0.707, 0.0, -0.707]),    # Front-left diagonal
    ]

    cameras = []
    for look_dir in directions:
        look_dir = look_dir / np.linalg.norm(look_dir)  # Normalize

        # Camera origin: centroid + look_dir * dist (camera is behind the object
        # from this view's perspective — it looks inward toward the centroid).
        cam_pos = centroid + look_dir * dist

        # Build a look-at rotation matrix.
        # forward = direction from camera to centroid (into the scene).
        forward = -look_dir  # Already normalized

        # Choose world up. When looking mostly along Y, use Z as up instead
        # to avoid gimbal lock when the camera is above/below the object.
        if abs(forward[1]) > 0.9:
            world_up = np.array([0.0, 0.0, -1.0])
        else:
            world_up = np.array([0.0, 1.0, 0.0])

        # right = forward × up (camera X axis)
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-8:
            # Degenerate case — pick an arbitrary perpendicular
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= right_len

        # up = right × forward (camera Y axis, corrected)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # Build 3×3 rotation matrix: rows are the camera's X, Y, Z axes in world space.
        # R transforms world-space vectors into camera space.
        R = np.stack([right, up, -forward], axis=0)  # (3, 3)

        # Translation: t = -R @ cam_pos  (standard camera extrinsic convention)
        t = -R @ cam_pos  # (3,)

        cameras.append({"R": R, "t": t, "fov_deg": 60.0, "pos": cam_pos})

    return cameras


# ---------------------------------------------------------------------------
# Depth rendering via raycasting (Open3D / Embree)
# ---------------------------------------------------------------------------

def _build_raycasting_scene(mesh: trimesh.Trimesh) -> "o3d.t.geometry.RaycastingScene":
    """
    Build an Open3D RaycastingScene from a trimesh mesh.

    Open3D's RaycastingScene uses Intel Embree internally — a robust, optimized
    ray tracing library that handles any ray direction without degenerate traversal.
    This replaces the trimesh pure-Python BVH which hung indefinitely on diagonal
    views (45° angle exposes an edge-on triangle traversal pathology in that impl).

    The scene object is built once and reused for all N_VIEWS depth renders.

    Args:
        mesh: The UV-mapped trimesh.Trimesh object.

    Returns:
        Initialized Open3D RaycastingScene ready for cast_rays() calls.
    """
    vertices = np.array(mesh.vertices, dtype=np.float32)   # (V, 3)
    faces    = np.array(mesh.faces,    dtype=np.uint32)    # (F, 3)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces,    dtype=o3d.core.Dtype.UInt32),
    )
    return scene


def _render_depth_view(
    mesh: trimesh.Trimesh,
    cam: dict,
    size: int = _RENDER_SIZE,
    scene: "o3d.t.geometry.RaycastingScene | None" = None,
) -> Image.Image:
    """
    Render a grayscale depth map of the mesh from the given camera viewpoint.

    Uses Open3D's Embree-backed RaycastingScene — headless, CPU-only, no GPU or
    display server required, and robust against diagonal/edge-on ray angles that
    caused the trimesh pure-Python BVH to hang.

    Args:
        mesh:  The UV-mapped trimesh.Trimesh object (used only to build scene on demand).
        cam:   Camera dict from _synthetic_cameras() with R, t, fov_deg, pos keys.
        size:  Output image size in pixels (square).
        scene: Pre-built RaycastingScene. If None, a new scene is built from mesh
               (slower; prefer pre-building once and passing it in).

    Returns:
        PIL Image in mode "L" (grayscale). Near hits → white, far hits → dark gray,
        background (no hit) → black.
    """
    R       = cam["R"]        # (3, 3) world → camera rotation
    fov_deg = cam["fov_deg"]
    cam_pos = cam["pos"]      # (3,)   camera origin in world space

    # ---- Generate pixel ray directions in world space ----
    # Build a grid of pixel centers in normalized image coordinates [-1, 1].
    half = np.tan(np.deg2rad(fov_deg / 2.0))
    xs = np.linspace(-half, half, size, dtype=np.float32)
    ys = np.linspace(-half, half, size, dtype=np.float32)

    xx, yy = np.meshgrid(xs, ys)
    # Camera-space directions: (x, y, -1) pointing into the scene — N = size×size rays.
    rays_cam = np.stack([xx.ravel(), yy.ravel(), -np.ones(size * size, dtype=np.float32)], axis=-1)

    # Transform to world space (R is orthonormal so R.T = R^-1).
    rays_world = rays_cam @ R.astype(np.float32)  # (N, 3)
    norms = np.linalg.norm(rays_world, axis=1, keepdims=True)
    rays_world /= np.maximum(norms, 1e-10)

    # ---- Build or reuse the Open3D raycasting scene ----
    if scene is None:
        scene = _build_raycasting_scene(mesh)

    # Open3D cast_rays() takes an (N, 6) tensor: [ox, oy, oz, dx, dy, dz].
    origins = np.tile(cam_pos.astype(np.float32), (len(rays_world), 1))  # (N, 3)
    rays_o3d = np.concatenate([origins, rays_world], axis=1)              # (N, 6)
    rays_tensor = o3d.core.Tensor(rays_o3d, dtype=o3d.core.Dtype.Float32)

    result = scene.cast_rays(rays_tensor)
    # t_hit is the ray-parameter distance to the first intersection. inf = no hit.
    t_hit = result["t_hit"].numpy().reshape(size, size)  # (size, size)

    # ---- Normalize depth to [0, 255] ----
    hit_mask = np.isfinite(t_hit)
    depth_map = t_hit.copy()
    depth_map[~hit_mask] = 0.0  # Background → black

    if hit_mask.any():
        d_min = depth_map[hit_mask].min()
        d_max = depth_map[hit_mask].max()
        if d_max > d_min:
            # Invert so near = bright, far = dark (standard depth visualization).
            depth_map[hit_mask] = 255.0 * (1.0 - (depth_map[hit_mask] - d_min) / (d_max - d_min))
        else:
            depth_map[hit_mask] = 200.0  # Flat / equidistant surface → mid-gray

    return Image.fromarray(depth_map.astype(np.uint8), mode="L")


# ---------------------------------------------------------------------------
# Stability AI API call
# ---------------------------------------------------------------------------

def _call_stability_structure(
    depth_img: Image.Image,
    prompt: str,
    api_key: str,
    on_progress: Callable[[str], None],
) -> Image.Image | None:
    """
    Call the Stability AI structure control endpoint and return the generated image.

    Uses the depth render as a structural reference — the model generates a
    plausible textured view that follows the mesh silhouette while applying
    the material described in the prompt.

    Args:
        depth_img:   Grayscale depth render (PIL Image, mode "L").
        prompt:      User's material description text.
        api_key:     Stability AI API key.
        on_progress: Progress callback for status messages.

    Returns:
        PIL Image (RGB) from the API response, or None if the call failed.
        None is non-fatal — the calling code skips this view.
    """
    # Encode the depth image to PNG bytes for the multipart upload.
    buf = io.BytesIO()
    depth_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    # Build the enriched prompt — appends PBR-specific guidance so the model
    # generates albedo-appropriate colors (no baked shadows, no specular highlights).
    full_prompt = (
        f"{prompt}, photorealistic PBR albedo texture, flat studio lighting, "
        "no shadows, no specular highlights, diffuse color only, game-ready asset"
    )

    try:
        response = requests.post(
            _STABILITY_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*",
            },
            files={
                # Stability API requires the image as a file upload in multipart.
                "image": ("depth.png", buf, "image/png"),
            },
            data={
                "prompt": full_prompt,
                # control_strength=0.85 means the structure reference has strong
                # influence — the generated image closely follows the mesh's
                # silhouette while the prompt drives the material/color.
                "control_strength": "0.85",
                "output_format": "png",
            },
            timeout=60,  # Stability API can be slow — give it a full minute
        )

        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return img
        elif response.status_code == 402:
            # Insufficient credits — raise so the caller can abort all remaining
            # views immediately rather than retrying 5 more times.
            raise RuntimeError(
                "Stability AI: insufficient credits. "
                "Purchase credits at https://platform.stability.ai/account/credits"
            )
        else:
            on_progress(
                f"Stability AI returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
            return None

    except Exception as e:
        on_progress(f"Stability AI call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# UV projection — back-project generated views onto the UV atlas
# ---------------------------------------------------------------------------

def _project_views_to_uv(
    mesh: trimesh.Trimesh,
    generated_views: list[Image.Image | None],
    cameras: list[dict],
    image_size: int,
    on_progress: Callable[[str], None],
) -> Image.Image:
    """
    Back-project AI-generated views onto the UV atlas via cosine-weighted blending.

    For each generated view + camera pair:
        1. Transform mesh vertices to camera space (X_cam = R @ X_world + t)
        2. Project to image plane using the camera's intrinsics
        3. Compute cosine weight: angle between vertex normal and camera direction
        4. Bilinearly sample the generated image at the projected pixel
        5. Winner-takes-all: each vertex keeps only the color from the view
           where it has the highest cosine weight (most directly facing the camera)

    Winner-takes-all eliminates cross-view color blending, which produced a muddy
    smeared result when views of very different hues averaged together. Each face
    is textured from its single best view, giving clean color boundaries at
    silhouette edges rather than a gradient smear across the whole surface.

    Args:
        mesh:            UV-mapped trimesh.Trimesh (loaded from meshed_uv.obj).
        generated_views: List of PIL Images (or None for failed API calls).
        cameras:         Matching list of camera dicts from _synthetic_cameras().
        image_size:      Output atlas resolution in pixels.
        on_progress:     Progress callback.

    Returns:
        PIL Image (RGB) of the UV atlas at image_size × image_size.
    """
    from meshine_shop.core.texture_baker import vertex_colors_to_texture

    vertices = np.array(mesh.vertices, dtype=np.float64)        # (V, 3)
    normals  = np.array(mesh.vertex_normals, dtype=np.float64)  # (V, 3)
    n_verts  = len(vertices)

    # Winner-takes-all state: track the best cosine weight and the corresponding
    # color seen so far for each vertex. No blending — each vertex ends up with
    # exactly one color from whichever view faced it most directly.
    best_weight   = np.zeros(n_verts,       dtype=np.float64)
    vertex_colors = np.zeros((n_verts, 3),  dtype=np.float32)

    fov_deg = cameras[0]["fov_deg"]
    # f = (size / 2) / tan(fov/2) — focal length in pixels
    f_px = (_RENDER_SIZE / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx = cy = _RENDER_SIZE / 2.0  # Principal point at image center

    successful_views = 0
    for img, cam in zip(generated_views, cameras):
        if img is None:
            continue

        R = cam["R"]  # (3, 3)
        t = cam["t"]  # (3,)

        # Resize generated image to render size for consistent pixel sampling.
        view_arr = np.array(img.resize((_RENDER_SIZE, _RENDER_SIZE)), dtype=np.float32) / 255.0

        # Transform all vertices to camera space: X_cam = R @ X_world + t
        X_cam = (R @ vertices.T).T + t  # (V, 3)

        # _synthetic_cameras builds R = [right, up, -forward] (OpenGL convention).
        # Objects in front of the camera have NEGATIVE z. Depth = -z (flipped positive).
        in_front = X_cam[:, 2] < -0.01
        depth = np.where(in_front, -X_cam[:, 2], 1.0)

        # Project to image plane.
        u = np.where(in_front, f_px * X_cam[:, 0] / depth + cx, -1.0)
        v = np.where(in_front, f_px * X_cam[:, 1] / depth + cy, -1.0)

        in_bounds = in_front & (u >= 0) & (u < _RENDER_SIZE - 1) & (v >= 0) & (v < _RENDER_SIZE - 1)

        # Cosine weight: 1.0 for vertex perfectly facing this camera, 0 for edge-on / back-facing.
        cam_dirs = cam["pos"] - vertices
        cam_dirs /= np.maximum(np.linalg.norm(cam_dirs, axis=1, keepdims=True), 1e-10)
        weights = np.maximum(np.sum(normals * cam_dirs, axis=1), 0.0)  # (V,)

        # Vectorized bilinear sample of the generated image at every in-bounds vertex.
        ui = np.clip(u.astype(int), 0, _RENDER_SIZE - 2)
        vi = np.clip(v.astype(int), 0, _RENDER_SIZE - 2)
        uf = u - ui
        vf = v - vi

        c00 = view_arr[vi, ui]          # (V, 3)
        c01 = view_arr[vi, ui + 1]
        c10 = view_arr[vi + 1, ui]
        c11 = view_arr[vi + 1, ui + 1]
        colors_all = ((1 - vf[:, None]) * (1 - uf[:, None]) * c00 +
                      (1 - vf[:, None]) *      uf[:, None]  * c01 +
                           vf[:, None]  * (1 - uf[:, None]) * c10 +
                           vf[:, None]  *      uf[:, None]  * c11)  # (V, 3)

        # Winner-takes-all update: overwrite a vertex's color only if this view
        # beats its current best weight. No blending → no cross-view smearing.
        update = in_bounds & (weights > best_weight)
        vertex_colors[update] = colors_all[update].astype(np.float32)
        best_weight[update]   = weights[update]

        successful_views += 1

    on_progress(f"Projected {successful_views} views onto UV layout (winner-takes-all)")

    # has_color: any vertex that received a color from at least one view.
    has_color = best_weight > 1e-10

    # Fill vertices with no coverage using the overall mean color.
    if has_color.any():
        mean_color = vertex_colors[has_color].mean(axis=0)
        vertex_colors[~has_color] = mean_color
    else:
        # Total failure — fallback to neutral gray.
        vertex_colors[:] = 0.5

    # vertex_colors is already float32 [0, 1] from the winner-takes-all assignment.
    vertex_colors_f = np.clip(vertex_colors, 0.0, 1.0)

    # Extract UV coordinates and face indices that vertex_colors_to_texture needs.
    uvs   = np.array(mesh.visual.uv, dtype=np.float32)  # (V, 2) in [0, 1]² space
    faces = np.array(mesh.faces,     dtype=np.int32)     # (F, 3) vertex indices

    # Rasterize per-vertex colors to a UV atlas image.
    albedo_img = vertex_colors_to_texture(uvs, faces, vertex_colors_f, image_size)
    return albedo_img


# ---------------------------------------------------------------------------
# Segment-aware UV projection (Phase 3 — mesh segmentation)
# ---------------------------------------------------------------------------

def _project_segment_to_uv(
    mesh: trimesh.Trimesh,
    vertex_mask: np.ndarray,
    generated_views: list[Image.Image | None],
    cameras: list[dict],
    image_size: int,
    on_progress: Callable[[str], None],
) -> np.ndarray:
    """
    Back-project AI-generated views onto the UV layout for one mesh segment only.

    Identical projection math to _project_views_to_uv but with two changes:
      1. The winner-takes-all update is restricted to vertices where vertex_mask
         is True — cross-segment contamination is impossible.
      2. Uncovered vertices within the segment are filled with the intra-segment
         mean color (not the global mesh mean), keeping color seams between
         segments as clean as possible.

    The function does NOT call vertex_colors_to_texture. It returns a raw
    (V, 3) float32 array so the caller can assemble all segments into a single
    vertex_colors buffer before rasterizing once.

    Non-segment vertex slots (where vertex_mask is False) are left as 0.0 —
    the caller must ignore those positions and fill them from the corresponding
    segment's projection.

    Args:
        mesh:            UV-mapped trimesh.Trimesh.
        vertex_mask:     (V,) bool — True for vertices belonging to this segment.
        generated_views: Per-camera AI-generated images (or None for failed calls).
        cameras:         Camera dicts from _synthetic_cameras().
        image_size:      Output atlas size (pixels) — used only for annotation.
        on_progress:     Progress callback.

    Returns:
        (V, 3) float32 np.ndarray in [0, 1]. Only positions where vertex_mask
        is True contain projected colors. All other positions are 0.0.
    """
    vertices = np.array(mesh.vertices, dtype=np.float64)        # (V, 3)
    normals  = np.array(mesh.vertex_normals, dtype=np.float64)  # (V, 3)
    n_verts  = len(vertices)

    # Per-vertex winner-takes-all state — scoped to this segment.
    best_weight   = np.zeros(n_verts,      dtype=np.float64)
    vertex_colors = np.zeros((n_verts, 3), dtype=np.float32)

    fov_deg = cameras[0]["fov_deg"]
    f_px = (_RENDER_SIZE / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx = cy = _RENDER_SIZE / 2.0

    for img, cam in zip(generated_views, cameras):
        if img is None:
            continue

        R = cam["R"]
        t = cam["t"]

        view_arr = np.array(
            img.resize((_RENDER_SIZE, _RENDER_SIZE)), dtype=np.float32
        ) / 255.0

        # Transform vertices to camera space.
        X_cam    = (R @ vertices.T).T + t    # (V, 3)
        in_front = X_cam[:, 2] < -0.01      # OpenGL: front objects have negative Z
        depth    = np.where(in_front, -X_cam[:, 2], 1.0)

        # Project to image plane.
        u = np.where(in_front, f_px * X_cam[:, 0] / depth + cx, -1.0)
        v = np.where(in_front, f_px * X_cam[:, 1] / depth + cy, -1.0)

        in_bounds = (
            in_front
            & (u >= 0) & (u < _RENDER_SIZE - 1)
            & (v >= 0) & (v < _RENDER_SIZE - 1)
        )

        # Cosine weight: 1.0 for vertex perfectly facing this camera.
        cam_dirs = cam["pos"] - vertices
        cam_dirs /= np.maximum(np.linalg.norm(cam_dirs, axis=1, keepdims=True), 1e-10)
        weights  = np.maximum(np.sum(normals * cam_dirs, axis=1), 0.0)  # (V,)

        # Vectorized bilinear sample.
        ui = np.clip(u.astype(int), 0, _RENDER_SIZE - 2)
        vi = np.clip(v.astype(int), 0, _RENDER_SIZE - 2)
        uf = u - ui
        vf = v - vi

        c00 = view_arr[vi, ui]
        c01 = view_arr[vi, ui + 1]
        c10 = view_arr[vi + 1, ui]
        c11 = view_arr[vi + 1, ui + 1]
        colors_all = (
            (1 - vf[:, None]) * (1 - uf[:, None]) * c00 +
            (1 - vf[:, None]) *      uf[:, None]  * c01 +
                 vf[:, None]  * (1 - uf[:, None]) * c10 +
                 vf[:, None]  *      uf[:, None]  * c11
        )  # (V, 3)

        # Winner-takes-all — restricted to this segment's vertices only.
        # Vertices in other segments are never touched here.
        update = in_bounds & (weights > best_weight) & vertex_mask
        vertex_colors[update] = colors_all[update].astype(np.float32)
        best_weight[update]   = weights[update]

    # Fill uncovered segment vertices with the intra-segment mean.
    # "Uncovered" = in this segment but never received a view (occluded in all
    # 6 views). Using the segment mean (not global mesh mean) keeps color seams
    # between adjacent segments tighter.
    has_color_in_seg = vertex_mask & (best_weight > 1e-10)
    if has_color_in_seg.any():
        seg_mean = vertex_colors[has_color_in_seg].mean(axis=0)
        fill_mask = vertex_mask & ~has_color_in_seg
        vertex_colors[fill_mask] = seg_mean
    else:
        # No view covered any vertex in this segment — use neutral gray.
        vertex_colors[vertex_mask] = 0.5

    return np.clip(vertex_colors, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-vertex scalar to UV atlas (replaces _solid_map for PBR maps)
# ---------------------------------------------------------------------------

def _vertex_scalar_to_texture(
    mesh: trimesh.Trimesh,
    vertex_scalars: np.ndarray,
    image_size: int,
) -> Image.Image:
    """
    Rasterize per-vertex scalar values to a single-channel UV atlas texture.

    Used for roughness and metallic maps when segmentation produces per-region
    PBR values (i.e. roughness varies across the mesh). Works by expanding the
    scalar to a 3-channel float array and then calling vertex_colors_to_texture,
    then stripping back to single-channel.

    Args:
        mesh:            UV-mapped trimesh.Trimesh.
        vertex_scalars:  (V,) float32 in [0, 1] — one value per vertex.
        image_size:      Output atlas resolution in pixels.

    Returns:
        PIL Image in mode 'L' (single channel, 8-bit grayscale).
    """
    from meshine_shop.core.texture_baker import vertex_colors_to_texture

    # Expand (V,) → (V, 3) — three identical channels.
    scalar_rgb = np.stack([vertex_scalars] * 3, axis=1).astype(np.float32)

    uvs   = np.array(mesh.visual.uv, dtype=np.float32)
    faces = np.array(mesh.faces,     dtype=np.int32)

    # vertex_colors_to_texture handles dilation and UV rasterization.
    rgb_img = vertex_colors_to_texture(uvs, faces, scalar_rgb, image_size)

    # Return only the R channel (identical to G and B) as a grayscale image.
    return rgb_img.split()[0]


# ---------------------------------------------------------------------------
# Normal map from mesh geometry
# ---------------------------------------------------------------------------

def _mesh_normal_map(mesh: trimesh.Trimesh, image_size: int) -> Image.Image:
    """
    Generate an object-space normal map from mesh vertex normals.

    Maps the vertex normals from [-1, 1] world-space range to [0, 255] RGB:
        R = (nx + 1) / 2 × 255
        G = (ny + 1) / 2 × 255
        B = (nz + 1) / 2 × 255

    This is object-space (not tangent-space), which most game engines support
    directly or can convert. Rasterized to a UV atlas at image_size × image_size.

    Args:
        mesh:       UV-mapped trimesh.Trimesh with computed vertex normals.
        image_size: Output atlas resolution in pixels.

    Returns:
        PIL Image (RGB) of the normal map.
    """
    from meshine_shop.core.texture_baker import vertex_colors_to_texture

    normals = np.array(mesh.vertex_normals, dtype=np.float32)  # (V, 3) in [-1, 1]
    # Map from [-1, 1] to [0, 1] float — vertex_colors_to_texture expects this range.
    normals_f = np.clip(normals * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)

    # Extract UV layout and face topology.
    uvs   = np.array(mesh.visual.uv, dtype=np.float32)  # (V, 2) in [0, 1]² space
    faces = np.array(mesh.faces,     dtype=np.int32)     # (F, 3) vertex indices

    normal_img = vertex_colors_to_texture(uvs, faces, normals_f, image_size)
    return normal_img


# ---------------------------------------------------------------------------
# Solid map helper
# ---------------------------------------------------------------------------

def _solid_map(value_0_to_1: float, size: int) -> Image.Image:
    """
    Create a solid-color grayscale texture at the given normalized value.

    Used for roughness and metallic maps derived from prompt keywords.
    A solid roughness map means the entire surface has uniform PBR roughness —
    appropriate for Phase 2 where we don't yet generate per-texel roughness detail.

    Args:
        value_0_to_1: Normalized pixel value in [0.0, 1.0].
        size:         Output image size in pixels (square).

    Returns:
        PIL Image (L mode, single channel) of size × size.
    """
    pixel_val = int(np.clip(value_0_to_1 * 255, 0, 255))
    return Image.new("L", (size, size), pixel_val)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_ai_textures(workspace, on_progress: Callable[[str], None]) -> None:
    """
    Generate PBR texture maps via Stability AI and write them to workspace/textures/.

    Called by both AppleObjectCaptureEngine and ColmapEngine — the process is
    engine-agnostic because it only operates on the UV-mapped mesh file
    (meshed_uv.obj) that both engines produce.

    Non-fatal: any error is caught and logged, and the stage exits without
    raising an exception. The pipeline continues with an untextured mesh.

    Output files (all at 2048×2048):
        workspace/textures/albedo.png    — AI-generated diffuse color
        workspace/textures/normal.png    — object-space normal from mesh geometry
        workspace/textures/roughness.png — solid map from material keyword in prompt
        workspace/textures/metallic.png  — solid map from material keyword in prompt

    Args:
        workspace:   WorkspacePaths dataclass with .root, .mesh, .textures attributes.
        on_progress: Callback(str) — forwards status messages to the UI queue.
    """
    try:
        _generate_ai_textures_impl(workspace, on_progress)
    except Exception as e:
        # Top-level guard — ensures the stage never crashes the pipeline.
        on_progress(f"AI texture generation failed unexpectedly: {e}")


def _generate_ai_textures_impl(workspace, on_progress: Callable[[str], None]) -> None:
    """
    Inner implementation — called by generate_ai_textures() inside a try/except.

    Phase 3 changes vs Phase 2:
    - Mesh is automatically segmented into up to 5 discrete regions.
    - Each segment gets its own material prompt derived from keyword-position
      parsing of the user's combined prompt.
    - Depth views are still rendered once for the full mesh (6 views total).
    - Stability AI is called once per (segment × view): N_segs × 6 API calls.
    - UV projection is restricted per-segment via a boolean vertex mask.
    - Roughness and metallic maps are now per-vertex (not solid fills).
    """
    # ---- 1. Read user prompt ----
    prompt_path = workspace.root / "ai_prompt.txt"
    if prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    else:
        prompt = ""
    if not prompt:
        # Sensible default when the user left the field blank.
        prompt = "photorealistic surface material, game-ready PBR"
    on_progress(f"Material prompt: \"{prompt}\"")

    # ---- 2. Load API key ----
    api_key = _load_api_key()
    if not api_key:
        on_progress(
            "AI Texture Generation skipped — no Stability AI API key found. "
            "Set the STABILITY_API_KEY environment variable or add your key to "
            "~/.meshine_shop/config.json to enable this stage."
        )
        return

    # ---- 3. Load UV-mapped mesh ----
    mesh_path = workspace.mesh / "meshed_uv.obj"
    if not mesh_path.exists():
        on_progress(f"AI Texture Generation skipped — mesh not found: {mesh_path}")
        return

    on_progress("Loading UV-mapped mesh...")
    mesh = trimesh.load(str(mesh_path), process=False)

    # trimesh.load may return a Scene for multi-part meshes — extract the geometry.
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            on_progress("AI Texture Generation skipped — mesh file contains no geometry.")
            return
        mesh = trimesh.util.concatenate(geoms)

    on_progress(f"Mesh loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Force vertex normal computation (trimesh is lazy about this).
    _ = mesh.vertex_normals

    # ---- 4. Segment mesh + assign per-segment prompts and PBR params ----
    from meshine_shop.core.mesh_segmentation import segment_mesh, assign_segment_prompts

    on_progress("Segmenting mesh into material regions...")
    segments = segment_mesh(mesh, max_segments=5)  # list[np.ndarray] of vertex indices
    n_segments = len(segments)

    # Warn the user upfront about API credit cost — N_segs × 6 views.
    total_api_calls = n_segments * _N_VIEWS
    on_progress(
        f"{n_segments} segment(s) detected — up to {total_api_calls} "
        f"Stability AI call(s) this run"
    )

    # Each segment gets a prompt derived from the user's text.
    segment_prompts = assign_segment_prompts(segments, prompt)

    # PBR roughness/metallic per segment — each segment prompt starts with the
    # segment's primary material keyword, so _material_pbr_params picks it up.
    segment_pbr = [_material_pbr_params(sp) for sp in segment_prompts]

    for i, (sp, (r, m)) in enumerate(zip(segment_prompts, segment_pbr)):
        # Truncate long prompts in the log to keep the queue readable.
        preview = sp if len(sp) <= 70 else sp[:67] + "..."
        on_progress(
            f"  Segment {i + 1}/{n_segments}: \"{preview}\" "
            f"[roughness={r:.2f}, metallic={m:.2f}]"
        )

    # ---- 5. Generate synthetic cameras ----
    bounds = mesh.bounds  # (2, 3)
    cameras = _synthetic_cameras(bounds)
    on_progress(f"Generated {len(cameras)} synthetic camera viewpoints")

    # Pre-build the Open3D Embree raycasting scene once and reuse it for all
    # views. Open3D handles any ray angle robustly; the previous trimesh BVH
    # stalled indefinitely on diagonal rays against high-poly meshes.
    on_progress("Building mesh raycasting scene (Open3D / Embree)...")
    o3d_scene = _build_raycasting_scene(mesh)

    # ---- 6. Render depth views (full mesh, shared by all segments) ----
    # Rendering the full mesh gives Stability AI context about surrounding
    # geometry — each segment's depth view shows where it sits within the whole
    # object. Per-segment isolation happens in the UV projection step, not here.
    depth_views: list[Image.Image] = []
    _VIEW_LABELS = ["front", "back", "left", "right", "top", "diagonal"]
    for i, cam in enumerate(cameras):
        view_label = _VIEW_LABELS[i]
        on_progress(f"Rendering depth view {i + 1}/{len(cameras)} ({view_label})...")
        depth_img = _render_depth_view(mesh, cam, size=_RENDER_SIZE, scene=o3d_scene)
        depth_views.append(depth_img)

    # ---- 7. Per-segment Stability AI calls + UV projection ----
    # Assemble final per-vertex buffers. Each segment writes only to its own
    # vertex indices; the full array is combined and rasterized once at the end.
    n_verts = len(mesh.vertices)
    final_vertex_colors    = np.zeros((n_verts, 3), dtype=np.float32)
    final_vertex_roughness = np.zeros(n_verts,      dtype=np.float32)
    final_vertex_metallic  = np.zeros(n_verts,      dtype=np.float32)

    any_albedo_written = False

    for seg_idx, (seg_verts, seg_prompt, (seg_r, seg_m)) in enumerate(
        zip(segments, segment_prompts, segment_pbr)
    ):
        on_progress(
            f"Processing segment {seg_idx + 1}/{n_segments} "
            f"({len(seg_verts):,} vertices)..."
        )

        # Boolean mask: True only for this segment's vertices.
        vertex_mask = np.zeros(n_verts, dtype=bool)
        vertex_mask[seg_verts] = True

        # Call Stability AI once per depth view, using this segment's prompt.
        seg_generated_views: list[Image.Image | None] = []
        credits_exhausted = False

        for view_idx, (depth_img, cam) in enumerate(zip(depth_views, cameras)):
            view_label = _VIEW_LABELS[view_idx]
            on_progress(
                f"  Seg {seg_idx + 1}/{n_segments}, "
                f"view {view_idx + 1}/{len(cameras)} ({view_label})..."
            )
            try:
                result = _call_stability_structure(
                    depth_img, seg_prompt, api_key, on_progress
                )
            except RuntimeError as e:
                # 402 Insufficient credits — abort the entire run immediately.
                on_progress(str(e))
                credits_exhausted = True
                break
            seg_generated_views.append(result)

        if credits_exhausted:
            # No point continuing to other segments — credits are gone.
            break

        seg_successful = sum(1 for v in seg_generated_views if v is not None)
        on_progress(
            f"  Segment {seg_idx + 1}: {seg_successful}/{len(cameras)} views OK"
        )

        if seg_successful == 0:
            # All API calls for this segment failed — use neutral gray.
            on_progress(
                f"  Segment {seg_idx + 1}: all views failed — neutral gray applied"
            )
            final_vertex_colors[vertex_mask] = 0.5
        else:
            # Project this segment's generated views onto its vertices only.
            seg_colors = _project_segment_to_uv(
                mesh, vertex_mask, seg_generated_views, cameras,
                _TEXTURE_SIZE, on_progress
            )
            # Write only the segment's slice into the global buffer.
            final_vertex_colors[vertex_mask] = seg_colors[vertex_mask]
            any_albedo_written = True

        # Write per-vertex PBR scalars for this segment.
        final_vertex_roughness[seg_verts] = seg_r
        final_vertex_metallic[seg_verts]  = seg_m

    # ---- 8. Rasterize assembled vertex colors → albedo.png ----
    from meshine_shop.core.texture_baker import vertex_colors_to_texture

    uvs   = np.array(mesh.visual.uv, dtype=np.float32)  # (V, 2) in [0,1]²
    faces = np.array(mesh.faces,     dtype=np.int32)     # (F, 3) vertex indices

    if any_albedo_written:
        on_progress("Rasterizing assembled albedo to UV atlas...")
        albedo_img  = vertex_colors_to_texture(
            uvs, faces, final_vertex_colors, _TEXTURE_SIZE
        )
        albedo_path = workspace.textures / "albedo.png"
        albedo_img.save(str(albedo_path))
        on_progress(f"Albedo map saved: {albedo_path.name} ({_TEXTURE_SIZE}×{_TEXTURE_SIZE})")
    else:
        on_progress(
            "AI Texture Generation: all API calls failed. "
            "Check your Stability AI account balance and API key."
        )

    # ---- 9. Normal map from mesh geometry (unchanged) ----
    on_progress("Computing normal map from mesh geometry...")
    normal_img  = _mesh_normal_map(mesh, _TEXTURE_SIZE)
    normal_path = workspace.textures / "normal.png"
    normal_img.save(str(normal_path))
    on_progress(f"Normal map saved: {normal_path.name}")

    # ---- 10. Per-segment roughness and metallic maps ----
    # _vertex_scalar_to_texture rasterizes per-vertex scalar values to a UV
    # atlas, producing spatial variation across segment boundaries rather than
    # a flat solid fill. Each segment's vertices carry their own PBR values.
    roughness_img  = _vertex_scalar_to_texture(mesh, final_vertex_roughness, _TEXTURE_SIZE)
    roughness_path = workspace.textures / "roughness.png"
    roughness_img.save(str(roughness_path))
    on_progress(f"Roughness map saved: {roughness_path.name} (per-segment values)")

    metallic_img  = _vertex_scalar_to_texture(mesh, final_vertex_metallic, _TEXTURE_SIZE)
    metallic_path = workspace.textures / "metallic.png"
    metallic_img.save(str(metallic_path))
    on_progress(f"Metallic map saved: {metallic_path.name} (per-segment values)")

    on_progress(
        "AI Texture Generation complete. "
        "albedo.png, normal.png, roughness.png, metallic.png saved to workspace/textures/"
    )
