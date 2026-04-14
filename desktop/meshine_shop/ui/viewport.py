"""
3D viewport widget — OpenGL rendering, camera navigation, and texture painting.

This is the heart of the Phase 5 Viewport feature. ViewportWidget is a
QOpenGLWidget subclass that uses moderngl to render the UV-unwrapped,
photo-textured mesh in real-time. It handles all mouse and keyboard input for
UE-style camera navigation and routes paint interactions to MeshPainter.

Rendering pipeline (per frame):
    1. Clear depth; draw gradient background quad (fullscreen, z=0.9999)
    2. Draw ground grid (alpha-blended, distance-faded, anti-aliased via fwidth)
    3. Bind albedo texture; compute MVP; draw mesh triangles via VAO
    4. After paint strokes: upload dirty rect to GPU via glTexSubImage2D

Camera navigation:
    - LMB drag                 → orbit around focal point
    - Scroll wheel             → dolly (zoom toward cursor)
    - MMB drag                 → pan (shift camera + focal point)
    - F key                    → frame mesh (auto-position camera)

Texture painting:
    - Brush tool: click/drag on mesh → ray cast → UV paint → GPU upload
    - Region tool: click on mesh → BFS flood fill → GPU upload
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import moderngl
    _HAS_MODERNGL = True
except ImportError:
    _HAS_MODERNGL = False

from PySide6.QtCore import Qt, QPoint, QTimer, Signal
from PySide6.QtGui import (
    QColor, QCursor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygon, QWheelEvent,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from meshine_shop.core.edit_history import EditHistory
from meshine_shop.core.mesh_painter import MeshPainter
from meshine_shop.core.viewport_camera import ViewportCamera

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# macOS native cursor warp
# ---------------------------------------------------------------------------
# Qt's QCursor.setPos() is ignored by macOS during an active mouse drag. We
# call CGWarpMouseCursorPosition directly via ctypes to bypass that restriction.
# On non-macOS platforms we fall back to QCursor.setPos().

if sys.platform == "darwin":
    try:
        _cg_lib = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("CoreGraphics") or "CoreGraphics"
        )

        class _CGPoint(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

        _cg_lib.CGWarpMouseCursorPosition.restype = ctypes.c_int
        _cg_lib.CGWarpMouseCursorPosition.argtypes = [_CGPoint]

        def _warp_cursor_native(x: float, y: float) -> None:
            """Move cursor to (x, y) in global screen logical coordinates."""
            _cg_lib.CGWarpMouseCursorPosition(_CGPoint(x, y))

    except Exception:
        _warp_cursor_native = None  # type: ignore[assignment]
else:
    _warp_cursor_native = None  # type: ignore[assignment]


def _warp_cursor(pos: QPoint) -> None:
    """
    Warp the system cursor to a global screen position.

    Uses CGWarpMouseCursorPosition on macOS (works during active drags).
    Falls back to QCursor.setPos on other platforms.
    """
    if _warp_cursor_native is not None:
        _warp_cursor_native(float(pos.x()), float(pos.y()))
    else:
        QCursor.setPos(pos)

# --------------------------------------------------------------------------- #
# GLSL shaders                                                                 #
# --------------------------------------------------------------------------- #

# Vertex shader: transform position by MVP, pass UV and view-space normal
# through to the fragment shader for lighting.
VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 MVP;   // projection * view * model (model = identity)
uniform mat4 MV;    // view * model — used to transform normals to view space

out vec2 frag_uv;
out vec3 frag_normal_vs;  // normal in view space

void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    frag_uv = in_uv;
    // Transform normal to view space (model is identity so MV == V * I == V)
    // Use transpose(inverse(MV)) for correct normal transformation
    frag_normal_vs = mat3(transpose(inverse(MV))) * in_normal;
}
"""

# Fragment shader: sample albedo texture + simple diffuse + ambient lighting.
# Light direction is fixed in view space (z+ = toward camera) for a
# "headlight" effect that keeps the mesh always readable.
FRAG_SHADER = """
#version 330 core

uniform sampler2D albedo;

in vec2 frag_uv;
in vec3 frag_normal_vs;

out vec4 out_color;

void main() {
    vec3 tex_color = texture(albedo, frag_uv).rgb;
    vec3 N = normalize(frag_normal_vs);

    // Directional "headlight" from camera direction (0, 0, 1) in view space
    float diffuse  = max(dot(N, vec3(0.0, 0.0, 1.0)), 0.0) * 0.65;
    float ambient  = 0.35;
    float lighting = diffuse + ambient;

    out_color = vec4(tex_color * lighting, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Environment shaders — background gradient + ground grid
# ---------------------------------------------------------------------------

# Background gradient: a fullscreen NDC quad placed at maximum depth (z=0.9999)
# so every subsequent draw call naturally renders on top without needing special
# ordering tricks. Depth testing is disabled while drawing this quad.
BG_VERT_SHADER = """
#version 330 core

in vec2 in_pos;     // NDC position [-1, 1]
out float frag_y;   // 0 = screen bottom, 1 = screen top

void main() {
    // z=0.9999 → sits just in front of the far plane, behind all scene geometry
    gl_Position = vec4(in_pos, 0.9999, 1.0);
    frag_y = (in_pos.y + 1.0) * 0.5;
}
"""

BG_FRAG_SHADER = """
#version 330 core

in float frag_y;
out vec4 out_color;

void main() {
    // Near-black at the bottom, subtle cool blue-gray at the top —
    // a quiet gradient that reads as "sky" without dominating the mesh.
    vec3 bottom = vec3(0.13, 0.13, 0.14);
    vec3 top    = vec3(0.20, 0.21, 0.25);
    out_color = vec4(mix(bottom, top, frag_y), 1.0);
}
"""

# Ground grid: a large world-space XZ plane with anti-aliased minor (1-unit)
# and major (5-unit) grid lines. fwidth() gives per-pixel derivatives so line
# thickness is exactly 1 screen pixel regardless of camera zoom. Alpha fades
# with distance so the grid doesn't extend to a harsh visible edge.
GRID_VERT_SHADER = """
#version 330 core

in vec2 in_xz;         // XZ position in the ground plane
uniform mat4 MVP;      // projection × view (model = identity, grid is world-space)
uniform float grid_y;  // world Y of the ground plane (set to mesh bbox floor)

out vec3 world_pos;    // passed to fragment for grid line computation

void main() {
    vec3 pos    = vec3(in_xz.x, grid_y, in_xz.y);
    world_pos   = pos;
    gl_Position = MVP * vec4(pos, 1.0);
}
"""

GRID_FRAG_SHADER = """
#version 330 core

in vec3 world_pos;
uniform vec3 cam_pos;   // camera world position, used for distance fade

out vec4 out_color;

void main() {
    vec2 xz = world_pos.xz;

    // --- Minor grid: 1-unit spacing ---
    // fwidth gives the screen-space derivative → line thickness stays 1 px
    vec2 d_minor = fwidth(xz);
    vec2 minor   = abs(fract(xz - 0.5) - 0.5) / d_minor;
    float minor_a = 1.0 - min(min(minor.x, minor.y), 1.0);

    // --- Major grid: 5-unit spacing ---
    vec2 xz5     = xz / 5.0;
    vec2 d_major = fwidth(xz5);
    vec2 major   = abs(fract(xz5 - 0.5) - 0.5) / d_major;
    float major_a = 1.0 - min(min(major.x, major.y), 1.0);

    // --- Distance fade: full opacity nearby, transparent beyond 80 units ---
    float dist = length(xz - cam_pos.xz);
    float fade = 1.0 - smoothstep(45.0, 85.0, dist);

    // --- Compose: major lines brighter and more opaque than minor ---
    vec3  minor_col = vec3(0.27, 0.27, 0.28);
    vec3  major_col = vec3(0.40, 0.40, 0.42);
    float alpha     = (minor_a * 0.55 + major_a * 1.0) * fade;
    vec3  color     = mix(minor_col, major_col, step(0.01, major_a));

    out_color = vec4(color, clamp(alpha, 0.0, 1.0));
}
"""

# Rotation gizmo: three colored rings (X/Y/Z) centered at the mesh centroid.
# Rendered in world space using VP only — gizmo rings don't follow model rotation,
# they represent fixed world axes that the user drags to rotate the mesh around.

GIZMO_VERT_SHADER = """
#version 330 core

in vec3 in_position;
uniform mat4 VP;   // proj * view — gizmo lives in world space, no model matrix

void main() {
    gl_Position = VP * vec4(in_position, 1.0);
}
"""

GIZMO_FRAG_SHADER = """
#version 330 core

uniform vec4 ring_color;   // RGBA (alpha for hover highlight)
out vec4 out_color;

void main() {
    out_color = ring_color;
}
"""

# Flat-color highlight shader — used for polygon selection highlights
# (pending selection teal overlay + saved layer colored overlays).
# Simple MVP vertex pass-through with a user-supplied RGBA color uniform.
HIGHLIGHT_VERT_SHADER = """
#version 330 core

in vec3 in_position;
uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    // Push the highlight slightly toward the camera (subtract 0.002 in NDC
    // depth).  This ensures every highlight fragment has a strictly smaller
    // depth value than the mesh surface it sits on, so GL_LESS always passes
    // for front-facing selected faces and fails for back faces (which are much
    // further from the camera).  Without this push, the mesh and highlight are
    // at the exact same depth and z-fighting / GL_LEQUAL accumulation causes
    // shimmering and visible triangle-edge seams inside the selection.
    gl_Position.z -= 0.002 * gl_Position.w;
}
"""

HIGHLIGHT_FRAG_SHADER = """
#version 330 core

uniform vec4 highlight_color;
out vec4 out_color;

void main() {
    out_color = highlight_color;
}
"""

# UV-space mask shaders for pixel-accurate selection DISPLAY.
#
# After the FBO selection, the polygon is rasterised into a 1024×1024
# UV-space texture (uv_mask).  The full mesh is then rendered through
# these shaders; only fragments whose UV coordinate lands on a painted
# texel are drawn with the highlight colour.  Because the mask lives in
# UV space (not screen space) the selection looks correct from any camera
# angle, exactly as in Blender's texture-paint mask overlay.
UVMASK_VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec2 in_uv;

out vec2 v_uv;

uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_uv = in_uv;
}
"""

UVMASK_FRAG_SHADER = """
#version 330 core

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D uv_mask;
uniform vec4      highlight_color;

void main() {
    float m = texture(uv_mask, v_uv).r;
    // Discard fully-transparent fragments; scale alpha by mask value so the
    // blurred/dilated edges fade out smoothly — gives a soft, feathered
    // selection boundary instead of hard triangle edges.
    if (m < 0.04) discard;
    out_color = vec4(highlight_color.rgb, highlight_color.a * m);
}
"""

# Face-ID shaders for pixel-accurate polygon selection.
#
# Instead of testing face centroids (which gives jagged triangle-boundary
# selections), we render the mesh into an offscreen FBO where every fragment
# is colored with the unique ID of its face, then read which IDs appear
# under the pixels of the user's drawn polygon.  Only faces that have at
# least one visible pixel inside the polygon are selected — this matches
# how Blender and Maya implement lasso/boundary select.
#
# Encoding: face_index+1 packed into 24-bit RGB (background stays (0,0,0,0)).
# 'flat' interpolation on v_face_id guarantees every fragment in a primitive
# gets the provoking-vertex value — no float interpolation across the triangle.
FACEID_VERT_SHADER = """
#version 330 core

in vec3 in_position;
in float in_face_id;

flat out float v_face_id;

uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_face_id = in_face_id;
}
"""

FACEID_FRAG_SHADER = """
#version 330 core

flat in float v_face_id;
out vec4 out_color;

void main() {
    // Store face_index+1 as 24-bit RGB so background (cleared to 0) is
    // unambiguously different from face 0.
    int id = int(v_face_id) + 1;
    float r = float(id & 0xFF)        / 255.0;
    float g = float((id >> 8)  & 0xFF) / 255.0;
    float b = float((id >> 16) & 0xFF) / 255.0;
    out_color = vec4(r, g, b, 1.0);
}
"""

# Fullscreen quad vertex shader — used for scene copy and overlay passes.
# Converts NDC quad positions to UV coordinates for texture sampling.
QUAD_VERT_SHADER = """
#version 330 core

in vec2 in_pos;   // NDC position (-1..1)
out vec2 v_uv;    // UV coordinates (0..1)

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    // NDC (-1..1) → UV (0..1): origin is bottom-left in both spaces
    v_uv = in_pos * 0.5 + 0.5;
}
"""

# Scene copy shader — blits the offscreen FBO color texture straight to
# the Qt framebuffer with no modification.  Used as the first pass of the
# composite pipeline before overlays are applied on top.
COPY_FRAG_SHADER = """
#version 330 core

uniform sampler2D scene_color;
in vec2 v_uv;
out vec4 out_color;

void main() {
    out_color = texture(scene_color, v_uv);
}
"""

# Screen-space polygon overlay shader.
#
# Used for both the pending selection (teal) and committed layer highlights.
# Per-pixel accuracy: each fragment tests:
#   (a) Is this pixel inside the polygon mask (a rasterized polygon image)?
#   (b) Is there mesh geometry here? (depth < background threshold)
# If both — output the highlight color, blended over the existing scene.
#
# The mask texture is in PIL/Qt coordinate space (origin top-left), while
# the scene depth texture is in OpenGL space (origin bottom-left). The
# v_uv.y is therefore flipped when sampling the mask.
#
# Depth threshold: the background gradient is drawn at NDC z=0.9999 with
# w=1.0, which maps to depth-buffer value ~0.99995. Mesh geometry at
# typical viewing distances has depth < 0.9995, so that is the threshold.
SCREEN_OVERLAY_FRAG_SHADER = """
#version 330 core

uniform sampler2D scene_depth;    // depth texture from offscreen FBO
uniform sampler2D sel_mask;       // screen-space polygon mask (R8, white=selected)
uniform vec4      overlay_color;  // RGBA highlight color
// Projected NDC depth [0,1] of the polygon centroid. Used to occlude the
// overlay when the polygon is behind mesh geometry (e.g. viewing back side).
// Pass 0.0 to disable occlusion (pending selections always visible).
uniform float     poly_depth;

in vec2 v_uv;
out vec4 out_color;

void main() {
    float depth = texture(scene_depth, v_uv).r;

    // Flip Y to convert from OpenGL UV space to PIL/Qt screen space before
    // sampling the mask (mask was rasterized in PIL top-left-origin space).
    float mask  = texture(sel_mask, vec2(v_uv.x, 1.0 - v_uv.y)).r;

    // Skip background pixels (depth buffer cleared to exactly 1.0 — the
    // background gradient quad renders with depth test disabled so it never
    // writes depth). With near=0.01 / far=1000, mesh pixels reach ~0.9999
    // at ~100 units from camera, so 0.9999 safely keeps all mesh geometry
    // while still discarding true background (depth = 1.0).
    if (depth >= 0.9999 || mask < 0.04) discard;

    // Depth occlusion: if scene geometry at this pixel is closer to the camera
    // than the polygon (e.g. back-face mesh occluding a front-face layer),
    // discard so the highlight disappears behind the model.
    // Small epsilon (0.005) prevents z-fighting on the layer's own surface.
    if (depth < poly_depth - 0.005) discard;

    // Alpha = requested overlay alpha multiplied by mask coverage.
    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending on the destination stacks
    // multiple layer overlays cleanly without washing out base scene color.
    float a = overlay_color.a * mask;
    out_color = vec4(overlay_color.rgb, a);
}
"""

# Variant of SCREEN_OVERLAY_FRAG_SHADER without the PIL Y-flip.
# Used for committed layers where the mask is rendered by OpenGL into an FBO
# each frame — OpenGL textures are already in bottom-left-origin convention
# so no coordinate flip is needed when sampling sel_mask.
SCREEN_OVERLAY_FRAG_SHADER_NOFLIP = """
#version 330 core

uniform sampler2D scene_depth;
uniform sampler2D sel_mask;
uniform vec4      overlay_color;

in vec2 v_uv;
out vec4 out_color;

void main() {
    float depth = texture(scene_depth, v_uv).r;
    // No Y flip — mask was rendered by OpenGL and shares its UV convention.
    float mask  = texture(sel_mask, v_uv).r;
    // Same threshold as the Y-flip variant — see comment above.
    if (depth >= 0.9999 || mask < 0.04) discard;
    float a = overlay_color.a * mask;
    out_color = vec4(overlay_color.rgb, a);
}
"""

# ---------------------------------------------------------------------------
# Shader-based planar texture projection
# ---------------------------------------------------------------------------
#
# These shaders implement view-independent planar texture projection directly
# on the GPU. Rather than baking the projection into the UV atlas (which
# creates jagged seam artifacts), we render only the selected faces in a
# second pass using world-space position to compute planar UVs per fragment.
#
# Advantages over UV-atlas baking:
#   - Shape exactly matches the face selection — no UV fragmentation
#   - GPU bilinear + trilinear mipmap filtering — no aliasing grain
#   - Sliders update as live uniforms — no rebaking per adjustment
#   - Export bake is deferred to save time, done only once at export

PROJ_VERT_SHADER = """
#version 330 core

in vec3 in_position;

uniform mat4 MVP;

// World-space position passed to fragment shader for planar UV computation
// and winding-independent normal estimation via dFdx/dFdy.
out vec3 v_world_pos;

void main() {
    v_world_pos = in_position;
    gl_Position = MVP * vec4(in_position, 1.0);
}
"""

PROJ_FRAG_SHADER = """
#version 330 core

// Planar projection orthonormal frame (world space)
uniform vec3  proj_right;     // R vector — horizontal axis of projection plane
uniform vec3  proj_up;        // U vector — vertical axis (Y-down image convention)

// Normalization bounds — map projected world coords to [0, 1] at scale=1.0
uniform float proj_r_min;     // leftmost projected R coordinate (world units)
uniform float proj_r_range;   // total width of projection extent (world units)
uniform float proj_u_min;     // topmost projected U coordinate (world units)
uniform float proj_u_range;   // total height of projection extent (world units)

// Material transform — matches the layers panel slider values
uniform float proj_scale;     // uniform scale (1.0 = texture fills selection)
uniform float proj_cos;       // cos(rotate_deg) — pre-computed on CPU each frame
uniform float proj_sin;       // sin(rotate_deg)
uniform float proj_offset_x;  // horizontal shift in normalized [0,1] space
uniform float proj_offset_y;  // vertical shift in normalized [0,1] space
uniform float proj_opacity;   // overall blend opacity (0.0 – 1.0)

// Projection source texture (bound to texture unit 1)
uniform sampler2D proj_tex;

// Screen-space polygon mask (bound to texture unit 2).
// Rasterized each frame from the layer's poly_pts_3d so the projected
// texture is clipped to the exact shape the user drew — not just the
// rough union of selected face triangles.
uniform sampler2D layer_mask;

// Viewport dimensions — used to convert gl_FragCoord to [0,1] UV for mask lookup.
// gl_FragCoord uses OpenGL bottom-left origin; PIL mask uses top-left, so Y is flipped.
uniform float viewport_w;
uniform float viewport_h;

in  vec3 v_world_pos;
out vec4 f_color;

void main() {
    // Step 1: Project world-space fragment position onto the (R, U) plane.
    // dot(p, R) is the scalar projection along the right axis;
    // dot(p, U) is the scalar projection along the up axis.
    float r = dot(v_world_pos, proj_right);
    float u = dot(v_world_pos, proj_up);

    // Step 2: Normalize to [0, 1] within the selection bounding box.
    // At scale=1.0 the texture fills the entire selection with no waste.
    float pu = (r - proj_r_min) / proj_r_range;
    float pv = (u - proj_u_min) / proj_u_range;

    // Step 3: Apply material transform.
    // Center the UV around (0.5, 0.5), apply scale, rotate, re-center.
    // Matches the CPU math in mesh_painter.py setup_layer_projection().
    float pu_c = pu - 0.5 - proj_offset_x;
    float pv_c = pv - 0.5 - proj_offset_y;
    pu_c /= proj_scale;
    pv_c /= proj_scale;
    float pu_r = pu_c * proj_cos - pv_c * proj_sin;
    float pv_r = pu_c * proj_sin + pv_c * proj_cos;

    // Wrap to [0, 1] for seamless tiling
    float tex_u = fract(pu_r + 0.5);
    float tex_v = fract(pv_r + 0.5);

    // Step 4: Sample source texture. GPU handles bilinear + trilinear mipmap
    // filtering automatically — no aliasing grain, no "Doom look".
    vec4 tex_color = texture(proj_tex, vec2(tex_u, tex_v));

    // Step 5: Headlight shading — same as base mesh shader.
    // Face culling is handled at IBO build time on the CPU (only front-facing
    // faces are included in the projection VAO), so no per-fragment discard
    // is needed here. dFdx/dFdy gives a winding-independent geometric normal
    // for correct diffuse shading regardless of mesh winding convention.
    vec3 geo_n = normalize(cross(dFdx(v_world_pos), dFdy(v_world_pos)));
    vec3 N = geo_n;
    float diffuse  = max(dot(N, vec3(0.0, 0.0, 1.0)), 0.0) * 0.65;
    float ambient  = 0.35;
    tex_color.rgb *= (diffuse + ambient);

    // Step 7: Clip to the exact drawn polygon via the screen-space mask.
    // Y is flipped because gl_FragCoord is bottom-left but PIL mask is top-left.
    vec2 screen_uv = vec2(gl_FragCoord.x / viewport_w, 1.0 - gl_FragCoord.y / viewport_h);
    float mask = texture(layer_mask, screen_uv).r;

    tex_color.a *= mask * proj_opacity;

    // Discard fully transparent fragments so depth buffer is not dirtied
    // and the albedo texture shows through around the projection edges.
    if (tex_color.a < 0.004) discard;

    f_color = tex_color;
}
"""


class _PlaceholderLabel(QWidget):
    """
    Shown in the viewport area before any mesh is loaded.

    A centered label with muted text, matching the app's dark aesthetic.
    Replaced entirely by the OpenGL surface once a mesh is ready.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Run the pipeline to load a mesh into the viewport.")
        label.setObjectName("viewport_placeholder")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)


class _DotOverlay(QWidget):
    """
    Tiny white dot drawn as a child widget overlay over the OpenGL viewport.

    Positioned at the LMB click point on the mesh surface and visible while
    the button is held. Being a separate QWidget means QPainter never touches
    the OpenGL context — no ghosting, no framebuffer interference.
    """

    RADIUS: int = 5  # dot radius in pixels

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Fixed square bounding box large enough for the circle + 1px padding
        size = self.RADIUS * 2 + 2
        self.setFixedSize(size, size)
        # Pass all mouse/keyboard events through to the viewport below
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def place(self, x: int, y: int) -> None:
        """Center the dot on (x, y) in parent-widget pixel coordinates."""
        self.move(x - self.RADIUS - 1, y - self.RADIUS - 1)
        self.raise_()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        # Slightly translucent white so the dot doesn't look pasted on
        painter.setBrush(QColor(255, 255, 255, 210))
        painter.drawEllipse(1, 1, self.RADIUS * 2, self.RADIUS * 2)


class _PolyOverlay(QWidget):
    """
    Semi-transparent QPainter overlay that draws the polygon selection in progress.

    Positioned as a full-size child widget over the QOpenGLWidget so it covers
    the entire render area. Transparent to mouse events — all clicks pass through
    to the viewport underneath.

    Visual elements:
    - Solid cyan lines between placed anchor points
    - Dashed rubber-band line from the last anchor to the current cursor
    - Small filled dots at each anchor point
    - Larger circle around the first anchor (shows the "close here" zone when ≥3 pts)
    - Semi-transparent filled polygon preview when ≥3 points are placed
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Pass all mouse/keyboard events through to the viewport below
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # Transparent background so the GL content shows through
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()
        self._points: list[tuple[int, int]] = []
        self._cursor: tuple[int, int] = (0, 0)

    def set_points(
        self, points: list[tuple[int, int]], cursor: tuple[int, int]
    ) -> None:
        """Update the anchor-point list and rubber-band cursor position."""
        self._points = list(points)
        self._cursor = cursor
        if points:
            # Always fill the full parent area so lines reach every corner
            if self.parent():
                self.resize(self.parent().size())
            self.show()
            self.raise_()
        else:
            self.hide()
        # Trigger immediate repaint so callers that update points from outside
        # a normal mouse-move event (e.g. camera reprojection in paintGL) are
        # reflected on-screen without waiting for the next spontaneous repaint.
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        if not self._points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pts = [QPoint(px, py) for px, py in self._points]
        cyan = QColor(0, 220, 255)

        # --- Solid lines between consecutive anchor points ---
        line_pen = QPen(cyan, 1.5)
        line_pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(line_pen)
        for i in range(1, len(pts)):
            painter.drawLine(pts[i - 1], pts[i])

        # --- Dashed rubber-band line from last anchor to cursor ---
        if pts:
            rubber_pen = QPen(QColor(0, 220, 255, 160), 1)
            rubber_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(rubber_pen)
            painter.drawLine(pts[-1], QPoint(*self._cursor))

        # --- Closing-zone circle around the first anchor when ≥ 3 points ---
        # Gives the user a visible target to click to close the polygon loop.
        if len(pts) >= 3:
            painter.setBrush(QColor(0, 220, 255, 50))
            painter.setPen(QPen(cyan, 1))
            painter.drawEllipse(pts[0], 12, 12)

        # --- Small filled dots at each anchor point ---
        painter.setBrush(cyan)
        painter.setPen(Qt.PenStyle.NoPen)
        for p in pts:
            painter.drawEllipse(p, 4, 4)


class ViewportWidget(QOpenGLWidget):
    """
    OpenGL 3D viewport — renders the mesh and handles navigation + painting.

    This widget is placed in the center of the Viewport page (QStackedWidget
    index 2). It is intentionally lazy: OpenGL resources are only allocated
    inside initializeGL() (called by Qt the first time the widget is shown)
    and whenever load_mesh() is called.

    Public API (called by ViewportView / ViewportToolsPanel):
        load_mesh(mesh_path, textures_dir) — load mesh + texture
        set_tool(tool)                     — "brush" | "region"
        set_paint_color(color)             — (R, G, B) tuple
        set_brush_size(px)                 — brush radius in pixels
        set_brush_opacity(opacity)         — 0.0–1.0
        save_textures()                    — write edited albedo back to disk
        reset()                            — clear all state, stop timers
    """

    # Emitted whenever the undo/redo stack depth changes so the tools panel
    # can enable/disable the Undo and Redo buttons. Args: (can_undo, can_redo).
    history_changed = Signal(bool, bool)

    # Emitted when a polygon selection is finalized. Arg = number of selected
    # faces. The layers panel listens to enable the "Save as Layer" button.
    selection_ready = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ------------------------------------------------------------------ #
        # moderngl context + GL resources (set in initializeGL / load_mesh)   #
        # ------------------------------------------------------------------ #
        self._ctx: moderngl.Context | None = None

        # Mesh rendering resources (released and rebuilt on each mesh load)
        self._prog: moderngl.Program | None = None

        # Shader-based planar projection pass (second render pass per frame).
        # Renders only the selected faces with GPU texture projection — no
        # UV atlas baking, no PIL compositing. Shape = exact face selection.
        self._proj_prog: "moderngl.Program | None" = None
        # Per-layer GPU resources: {layer_id: {gpu_tex, ibo, vao, params, visible}}
        self._proj_layers: dict[int, dict] = {}
        # Pending projection setups deferred to paintGL for safe GPU resource creation.
        # Keyed by layer_id; each value is the dict returned by project_texture_to_layer
        # before GPU objects exist.  paintGL picks these up and builds VAO/tex/IBO.
        self._pending_proj_setups: dict[int, dict] = {}

        self._vao: moderngl.VertexArray | None = None
        self._vbo: moderngl.Buffer | None = None
        self._ibo: moderngl.Buffer | None = None
        self._albedo_tex: moderngl.Texture | None = None

        # Environment resources (built once in initializeGL, persist across mesh loads)
        self._bg_prog:   "moderngl.Program | None"     = None
        self._bg_vao:    "moderngl.VertexArray | None" = None
        self._bg_vbo:    "moderngl.Buffer | None"      = None
        self._grid_prog: "moderngl.Program | None"     = None
        self._grid_vao:  "moderngl.VertexArray | None" = None
        self._grid_vbo:  "moderngl.Buffer | None"      = None
        # World Y of the grid ground plane; updated to bbox_min.y on mesh load
        self._grid_y: float = 0.0

        # ------------------------------------------------------------------ #
        # Domain objects                                                       #
        # ------------------------------------------------------------------ #
        self._painter: MeshPainter | None = None
        self._camera: ViewportCamera | None = None
        # Edit history — reset on each new mesh load, shared with the tools panel
        self._history: EditHistory = EditHistory()
        # Set True by load_mesh(); cleared in paintGL after GPU upload.
        # Ensures GPU upload happens inside the guaranteed-current GL context.
        self._needs_upload: bool = False

        # ------------------------------------------------------------------ #
        # Paint tool state                                                     #
        # ------------------------------------------------------------------ #
        self._tool: str = ""                  # "" | "brush" | "region" | "rotate"

        # ------------------------------------------------------------------ #
        # Sculpt tool state                                                    #
        # ------------------------------------------------------------------ #
        # World-space brush radius for sculpt operations.
        # 0.05 is a reasonable default for a hand-sized real-world scan.
        self._sculpt_radius: float = 0.05
        # Displacement strength per sculpt call. Scaled by the falloff weight
        # so vertices at the brush center move by exactly this amount.
        self._sculpt_strength: float = 0.005
        # True if sculpt() modified geometry this frame — triggers VBO re-upload
        # in paintGL without reuploading the texture.
        self._needs_mesh_upload: bool = False

        self._paint_color: tuple = (220, 100, 60)  # default rust-orange
        self._brush_size: int = 12            # pixels
        self._brush_opacity: float = 1.0

        # ------------------------------------------------------------------ #
        # Model rotation (rotate tool)                                         #
        # ------------------------------------------------------------------ #
        # 3×3 orthogonal rotation matrix — accumulated as the user drags the
        # gizmo rings. Applied as a model matrix: T(centroid) @ R4x4 @ T(-centroid).
        # Kept separate from the camera so painting rays can be inverse-transformed.
        self._model_rot: np.ndarray = np.eye(3, dtype=np.float32)
        # World-space centroid of the mesh — rotation pivot point.
        self._mesh_centroid: np.ndarray = np.zeros(3, dtype=np.float32)

        # Gizmo GPU resources — built in _build_gizmo_rings() after mesh load
        self._gizmo_prog: "moderngl.Program | None" = None
        self._gizmo_vbos: dict = {}        # axis_name → moderngl.Buffer (ring band)
        self._gizmo_vaos: dict = {}        # axis_name → moderngl.VertexArray (ring band)
        self._gizmo_diamond_vbos: dict = {}  # axis_name → moderngl.Buffer (diamond handle)
        self._gizmo_diamond_vaos: dict = {}  # axis_name → moderngl.VertexArray
        # World-space midline points kept for screen-space hit testing
        self._gizmo_ring_pts: dict = {}    # axis_name → np.ndarray

        # Gizmo interaction state
        self._gizmo_axis: "str | None" = None   # ring being dragged ("x"/"y"/"z")
        self._gizmo_hover: "str | None" = None  # ring under cursor (for highlight)

        # ------------------------------------------------------------------ #
        # Mouse / keyboard navigation state                                    #
        # ------------------------------------------------------------------ #
        self._alt_held: bool = False          # True while Alt is held
        self._mmb_held: bool = False          # True while MMB is held
        self._lmb_held: bool = False          # True while LMB is held (for paint drag)
        # Set on LMB miss (cursor warp) — absorbs the first move event so the
        # warp delta (potentially hundreds of pixels) never triggers an orbit.
        self._orbit_skip_one: bool = False
        # Global screen position (QPoint) where LMB was pressed — cursor is
        # warped back here on release so it reappears exactly where the user
        # clicked. Storing global avoids any mapToGlobal conversion issues on
        # Retina displays.
        self._lmb_cursor_origin_global: "QPoint | None" = None
        # White dot overlay shown at the click position while LMB is held
        self._dot_overlay = _DotOverlay(self)

        # ------------------------------------------------------------------ #
        # Polygon selection tool state                                         #
        # ------------------------------------------------------------------ #
        # Transparent overlay widget drawn on top of the GL surface while
        # the user is building a polygon selection (shows anchor points,
        # connecting lines, rubber-band, and semi-transparent fill preview).
        self._poly_overlay = _PolyOverlay(self)

        # Anchor points placed so far in widget pixel coords.
        self._poly_points: list[tuple[int, int]] = []
        # Parallel world-space positions for each anchor in _poly_points.
        # Stored at click time via raycasting so that when the camera orbits
        # the anchors can be reprojected through the live MVP each frame,
        # keeping the overlay glued to the mesh surface instead of drifting.
        self._poly_points_3d: list[list[float]] = []
        # Current cursor position for the rubber-band line preview.
        self._poly_cursor: tuple[int, int] = (0, 0)
        # Deferred anchor placement state — anchor is not committed until LMB
        # release so that a drag (orbit) never plants an accidental point.
        # _poly_pending_press: the screen position of the current LMB press.
        # _poly_orbit_dragged: True if the user dragged the mouse after press,
        #   meaning the press was an orbit gesture, not a click to place anchor.
        self._poly_pending_press: "tuple[int, int] | None" = None
        self._poly_orbit_dragged: bool = False
        # Flag set by orbit/pan handlers to trigger one reprojection pass in
        # paintGL — ensures in-progress anchors follow the mesh during camera
        # navigation without running the MVP round-trip every single frame.
        self._camera_moved: bool = False

        # Pending polygon selection — set of face indices inside the last
        # finalized polygon. Shown as a teal highlight until saved as a layer.
        # Cleared when the user saves to a layer or switches tools.
        self._pending_faces: "set[int] | None" = None
        # True → rebuild the pending highlight VAO on the next paintGL call.
        self._pending_rebuild: bool = False

        # GL resources for the pending selection highlight (teal overlay).
        # These are kept for backward compatibility but are no longer used for
        # the pending display — screen mask textures replaced them.
        self._pending_vao: "moderngl.VertexArray | None" = None
        self._pending_vbo: "moderngl.Buffer | None" = None

        # Screen-space polygon mask texture for the pending (unsaved) selection.
        # A single-channel R8 texture at viewport resolution — white pixels mark
        # the interior of the drawn polygon. Created in _finalize_poly_selection()
        # and released when the selection is saved or cancelled. Replaces the old
        # face-VAO highlight, giving pixel-perfect selection boundaries.
        self._pending_screen_mask_tex: "moderngl.Texture | None" = None

        # Off-screen scene FBO — all scene elements (background, grid, mesh,
        # gizmo) render here first. The resulting color + depth textures are then
        # composited to Qt's framebuffer with selection overlays applied on top.
        self._scene_fbo:       "moderngl.Framebuffer | None" = None
        self._scene_color_tex: "moderngl.Texture | None"     = None
        self._scene_depth_tex: "moderngl.Texture | None"     = None

        # Two-pass composite programs:
        #   _copy_prog                  — blits scene_color_tex to qt_fbo unchanged
        #   _screen_overlay_prog        — overlay for PIL masks (pending selection, Y-flipped)
        #   _screen_overlay_prog_noflip — overlay for OpenGL-rendered masks (committed layers)
        self._copy_prog:                   "moderngl.Program | None"     = None
        self._screen_overlay_prog:         "moderngl.Program | None"     = None
        self._screen_overlay_prog_noflip:  "moderngl.Program | None"     = None

        # Fullscreen quad resources — shared VBO, three VAOs (one per program)
        self._quad_vbo:            "moderngl.Buffer | None"      = None
        self._quad_vao_copy:       "moderngl.VertexArray | None" = None
        self._quad_vao_overlay:    "moderngl.VertexArray | None" = None
        self._quad_vao_overlay_nf: "moderngl.VertexArray | None" = None  # no-flip variant

        # Reusable 1-channel texture for committed layer mask rendering.
        # Each frame the polygon anchor points are reprojected through the current
        # camera, PIL-rasterized, and written into this texture for the overlay pass.
        self._layer_mask_fbo: "moderngl.Framebuffer | None" = None  # unused, kept for compat
        self._layer_mask_tex: "moderngl.Texture | None"     = None

        # 3D world-space positions of the polygon anchor points, computed at
        # finalize time by projecting each screen-space anchor onto the mesh
        # surface via barycentric interpolation. Transferred to the layer dict
        # on save so the layer highlight can reproject correctly each frame.
        self._pending_poly_pts_3d: "list | None" = None

        # Average world-space face normal of the pending selection, used to
        # determine which side of the model the selection is on. Committed
        # layers use this to cull the overlay when the camera is on the back
        # side, so the highlight naturally disappears behind the geometry.
        self._pending_poly_avg_normal: "np.ndarray | None" = None

        # 2D screen-space polygon and camera matrices captured at selection time.
        # Used by the cookie-cutter deletion to clip boundary triangles exactly
        # along the polygon outline rather than deleting whole triangles.
        self._pending_poly_pts_2d: "list | None" = None
        self._pending_view_mat:    "np.ndarray | None" = None
        self._pending_proj_mat:    "np.ndarray | None" = None
        self._pending_model_mat:   "np.ndarray | None" = None

        # Saved layers — keyed by integer layer_id assigned at save time.
        # Each entry: {"name", "faces", "visible", "color", "vao", "vbo"}
        self._layers: dict[int, dict] = {}
        self._next_layer_id: int = 0
        # ID of the layer currently selected in the layers panel — rendered
        # brighter so the user can see which selection is active.
        self._active_layer_id: int | None = None

        # Flat-color shader for all selection highlights (pending + layers)
        self._hl_prog: "moderngl.Program | None" = None

        # UV-space mask shader — renders the full mesh; only fragments whose
        # UV coordinate lands on a painted texel in uv_mask are drawn.
        # Gives pixel-accurate selection display from any camera angle.
        self._uvmask_prog: "moderngl.Program | None" = None

        # Full-mesh VAO bound to _uvmask_prog (pos + uv, reuses _vbo/_ibo).
        # Created after _vbo upload; None until then.
        self._hl_mesh_vao: "moderngl.VertexArray | None" = None

        # UV mask texture for the pending (not yet saved) selection.
        self._pending_uv_mask_tex: "moderngl.Texture | None" = None

        # Face-ID shader — used in _finalize_poly_selection() to render an
        # offscreen FBO where each fragment carries its triangle's unique ID.
        # Pixel readback + polygon mask gives Blender-quality pixel-accurate selection.
        self._faceid_prog: "moderngl.Program | None" = None

        # World-space 3D position of the LMB hit point (or miss-anchor point) —
        # re-projected every frame so the dot tracks the model surface as it orbits.
        self._dot_world_pos: "np.ndarray | None" = None
        # True when _dot_world_pos was set for a miss (off-model click). Release
        # handling skips the dot-based cursor restore in this case so the cursor
        # returns to the original click position instead.
        self._dot_is_miss: bool = False
        # Set of held button name strings ("left", "right", "middle") — used to
        # match multi-button chord bindings (e.g. "left+right" for pan)
        self._mouse_btns: set[str] = set()
        self._keys_held: set[Qt.Key] = set()
        self._last_mouse_pos: tuple[int, int] | None = None

        # Smoothed WASD velocity components (world-space, range −1..+1).
        self._vel_fwd:   float = 0.0
        self._vel_right: float = 0.0
        self._vel_up:    float = 0.0

        # Movement tick timer — fires at ~60fps so WASD translation works
        # whenever the viewport has focus.
        self._move_timer = QTimer(self)
        self._move_timer.setInterval(16)
        self._move_timer.timeout.connect(self._move_tick)
        self._move_timer.start()
        self._last_tick_time: float = time.monotonic()

        # ------------------------------------------------------------------ #
        # Focus policy — needed to receive key events                          #
        # ------------------------------------------------------------------ #
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_mesh(self, mesh_path: Path, textures_dir: Path) -> None:
        """
        Load a UV-unwrapped OBJ + its textures and begin rendering.

        This stores the mesh and camera state, then schedules a repaint.
        GPU upload is deferred to the first paintGL() call where the OpenGL
        context is guaranteed to be current. This avoids a common failure
        mode where makeCurrent() silently fails on a hidden QStackedWidget
        page, leaving _vao as None.
        """
        log.info("ViewportWidget: loading mesh %s", mesh_path)

        # Release any prior GPU resources so a new job replaces the old one
        if self._ctx is not None:
            self.makeCurrent()
            self._release_gpu_resources()
            self.doneCurrent()

        # Clear undo/redo history — a new mesh load starts a fresh edit session
        self._history.clear()
        self.history_changed.emit(False, False)

        # Invalidate all polygon selections and layers — face indices from the
        # previous mesh are meaningless after loading a new one.
        self._poly_points.clear()
        self._poly_points_3d.clear()
        self._poly_pending_press = None
        self._poly_orbit_dragged = False
        self._poly_overlay.set_points([], (0, 0))
        self._pending_faces = None
        self._pending_rebuild = False
        self._pending_screen_mask_tex = None
        self._pending_poly_pts_3d = None
        self._pending_poly_avg_normal = None
        self._pending_poly_pts_2d  = None
        self._pending_view_mat     = None
        self._pending_proj_mat     = None
        self._pending_model_mat    = None
        self._layers.clear()
        self._next_layer_id = 0

        # Create the painter (loads mesh + PIL texture, builds BVH)
        self._painter = MeshPainter(mesh_path, textures_dir)

        # Create/reset camera
        self._camera = ViewportCamera()
        self._camera.load_settings()

        # Position camera to see the full mesh
        bbox_min, bbox_max = self._painter.get_bbox()
        self._camera.frame_mesh(bbox_min, bbox_max)

        # Set grid floor Y to the bottom of the mesh bounding box
        self._grid_y = float(bbox_min[1])

        # Compute mesh centroid (midpoint of bounding box) — rotation pivot
        self._mesh_centroid = (
            (bbox_min.astype(np.float32) + bbox_max.astype(np.float32)) * 0.5
        )
        # Reset model rotation whenever a new mesh is loaded
        self._model_rot = np.eye(3, dtype=np.float32)
        self._gizmo_hover = None
        self._gizmo_axis  = None

        # Flag that GPU resources need to be uploaded on the next paintGL call.
        # Gizmo rings are also built there (see paintGL _needs_upload block)
        # because paintGL guarantees the GL context is current.
        # Deferring to paintGL() guarantees we're inside Qt's OpenGL context
        # (makeCurrent() has already been called by Qt for us).
        self._needs_upload = True

        self.update()

    def set_tool(self, tool: str) -> None:
        """
        Set the active paint tool.

        tool must be "brush", "region", "rotate", or "" (no tool — navigation mode).
        When a tool is active LMB paints on the mesh; when empty LMB orbits.
        Calls update() so the gizmo shows/hides immediately without waiting for
        the next user interaction.
        """
        self._tool = tool
        # Cancel any in-progress polygon selection when switching away from the tool
        if tool != "poly_select":
            self._cancel_poly_selection()
        self._gizmo_axis = None   # clear any in-progress drag when tool switches
        self._gizmo_hover = None
        self.update()             # repaint now — gizmo appears/disappears instantly

    def set_paint_color(self, color: tuple) -> None:
        """Set the brush/fill color as an (R, G, B) tuple (0–255 each)."""
        self._paint_color = color

    def set_brush_size(self, px: int) -> None:
        """Set the brush radius in texture pixels."""
        self._brush_size = px

    def set_brush_opacity(self, opacity: float) -> None:
        """Set brush opacity. 1.0 = fully opaque, 0.0 = transparent."""
        self._brush_opacity = opacity

    def save_textures(self) -> None:
        """Write the edited albedo buffer back to albedo.png."""
        if self._painter:
            self._painter.save_albedo()

    def set_sculpt_radius(self, radius: float) -> None:
        """Set the world-space radius for sculpt brushes."""
        self._sculpt_radius = float(radius)

    def set_sculpt_strength(self, strength: float) -> None:
        """Set the per-call displacement strength for sculpt brushes."""
        self._sculpt_strength = float(strength)

    # ------------------------------------------------------------------ #
    # Undo / Redo — called by tool panel buttons and Cmd+Z shortcuts       #
    # ------------------------------------------------------------------ #

    def undo(self) -> None:
        """
        Restore the previous edit state from the undo stack.

        Special case: while the polygon select tool is active and anchor points
        are being placed, Undo pops the last anchor point instead of touching
        the geometry history. This lets the user fix accidental clicks during
        lasso drawing without disrupting the undo stack.

        Normal case: captures the current state onto the redo stack first so
        it can be re-applied by redo(), then pops the undo snapshot, restores
        the painter, re-uploads GPU resources, and emits history_changed.
        """
        # While drawing a polygon, undo removes the last-placed anchor point.
        # Both the Undo button and Cmd+Z route through here, so both work.
        if self._tool == "poly_select" and self._poly_points:
            self._poly_points.pop()
            if self._poly_points_3d:
                self._poly_points_3d.pop()
            self._poly_overlay.set_points(self._poly_points, self._poly_cursor)
            self._poly_overlay.update()
            return

        if not self._history.can_undo or self._painter is None:
            return

        # Save current state to redo stack before overwriting.
        # Always include geometry so redo can re-apply sculpt deformations.
        self._history.push_redo(self._painter, geometry=True)

        snap = self._history.undo()
        if snap:
            self._restore_from_snapshot(snap)

        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

    def redo(self) -> None:
        """
        Re-apply an undone edit from the redo stack.

        Captures the current state back onto the undo stack first so the
        user can undo again after redoing. Then pops the redo snapshot,
        restores the painter, re-uploads GPU resources, and emits history_changed.
        """
        if not self._history.can_redo or self._painter is None:
            return

        # Save current state back to undo stack WITHOUT clearing redo.
        # push_snapshot() would wipe _redos before we can pop from it.
        # Always include geometry so undo-after-redo restores sculpt correctly.
        self._history.push_undo_only(self._painter, geometry=True)

        snap = self._history.redo()
        if snap:
            self._restore_from_snapshot(snap)

        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

    def _restore_from_snapshot(self, snap) -> None:
        """
        Apply a snapshot to the painter and re-upload GPU resources.

        For geometry snapshots: rebuilds the mesh VBO/VAO (new vertex data).
        For texture-only snapshots: marks the full texture dirty so paintGL
        calls _refresh_dirty_texture on the next frame, uploading via
        glTexSubImage2D.
        """
        if self._painter is None:
            return

        # Restore painter state (geometry + texture or texture only).
        # layer_textures / layer_visible are included in the snapshot so undoing
        # a texture projection correctly removes it from the composite display.
        self._painter.restore_snapshot(
            snap.vertices, snap.faces, snap.normals, snap.uvs, snap.albedo,
            layer_textures=getattr(snap, "layer_textures", None),
            layer_visible=getattr(snap, "layer_visible", None),
        )

        # Sync _proj_layers with painter state after undo/redo.
        # If the restored snapshot predates a projection, painter._layer_projections
        # will no longer contain that layer_id — release the stale GPU resources.
        self.makeCurrent()
        for lid in list(self._proj_layers.keys()):
            if lid not in self._painter._layer_projections:
                pdata = self._proj_layers.pop(lid, {})
                for key in ("gpu_tex", "ibo", "vao"):
                    res = pdata.get(key)
                    if res is not None:
                        try:
                            res.release()
                        except Exception:
                            pass
        self.doneCurrent()

        if snap.geometry_included:
            # Geometry changed — need a full VBO/VAO rebuild.
            # Set _needs_upload so paintGL triggers the full GPU re-upload.
            self._needs_upload = True
        else:
            # Texture-only change — painter.restore_snapshot already set the
            # full dirty rect, so _refresh_dirty_texture will handle it.
            pass

        self.update()

    def reset_model_rotation(self) -> None:
        """
        Snap the mesh back to its original orientation (identity rotation).

        Called by the Reset button in the tools panel. Clears _model_rot to
        the identity matrix so the MVP shader receives no model transform.
        """
        self._model_rot = np.eye(3, dtype=np.float32)
        self._gizmo_axis = None
        self.update()

    def normalize_scale(self) -> None:
        """
        Rescale the mesh so its longest bounding-box axis equals 1.0 unit.

        This is a destructive geometry edit (vertices are modified in place)
        and is undoable via the normal undo stack. It ensures that meshes
        exported to platforms without import-time scale controls (web viewers,
        AR/VR runtimes, etc.) arrive at a predictable neutral size.

        The centroid stays at the same world position — only the scale changes.
        """
        if self._painter is None:
            return

        # Snapshot current state for undo before modifying geometry
        self._history.push_snapshot(self._painter, geometry=True)
        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

        # Compute the longest extent of the current bounding box
        bbox_min, bbox_max = self._painter.get_bbox()
        extents = bbox_max - bbox_min                         # (3,) per-axis lengths
        longest = float(np.max(extents))
        if longest < 1e-9:
            log.warning("normalize_scale: degenerate mesh (zero extent) — skipped")
            return

        scale_factor = 1.0 / longest

        # Scale vertices around their centroid so the mesh stays centered
        centroid = (bbox_min + bbox_max) * 0.5
        verts = np.asarray(self._painter._mesh.vertices, dtype=np.float64)
        verts = (verts - centroid) * scale_factor + centroid
        self._painter._mesh.vertices = verts

        # Rebuild BVH so ray-cast / sculpt stay accurate after the scale change
        self._painter._rebuild_bvh()

        # Update centroid and grid floor to reflect the new bounding box
        new_min, new_max = self._painter.get_bbox()
        self._mesh_centroid = ((new_min + new_max) * 0.5).astype(np.float32)
        self._grid_y = float(new_min[1])

        # Flag GPU buffers for rebuild on next paintGL call
        self._needs_upload = True
        self.update()

    def reset(self) -> None:
        """Clear all state. Returns widget to idle."""
        self._move_timer.stop()
        self._painter = None
        self._camera = None
        self._needs_upload = False
        self._dot_world_pos = None
        self._dot_is_miss = False
        self._vel_fwd = self._vel_right = self._vel_up = 0.0
        self._keys_held.clear()
        self._last_mouse_pos = None
        # Reset model rotation and gizmo interaction state
        self._model_rot = np.eye(3, dtype=np.float32)
        self._gizmo_axis = None
        self._gizmo_hover = None

        # Clear polygon selection state (pending and layers)
        self._poly_points.clear()
        self._poly_points_3d.clear()
        self._poly_pending_press = None
        self._poly_orbit_dragged = False
        self._poly_overlay.set_points([], (0, 0))
        self._pending_faces = None
        self._pending_rebuild = False
        self._pending_screen_mask_tex = None
        self._pending_poly_pts_3d = None
        self._pending_poly_pts_2d  = None
        self._pending_view_mat     = None
        self._pending_proj_mat     = None
        self._pending_model_mat    = None
        self._layers.clear()
        self._next_layer_id = 0

        # Release mesh-specific GPU resources only.  Background gradient and
        # ground grid (_build_environment) are mesh-independent and only built
        # once in initializeGL — destroying them here would leave the viewport
        # permanently black after the first reset.
        if self._ctx is not None:
            self.makeCurrent()
            self._release_gpu_resources()
            self._release_gizmo()
            self.doneCurrent()

        self.update()

    # ------------------------------------------------------------------ #
    # Qt OpenGL lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def initializeGL(self) -> None:
        """Called once by Qt when the OpenGL context is first created."""
        if not _HAS_MODERNGL:
            log.error("ViewportWidget: moderngl not installed — viewport disabled")
            return

        # Create moderngl context that wraps Qt's underlying OpenGL context
        self._ctx = moderngl.create_context()
        log.info(
            "ViewportWidget: moderngl context created (GL %s)",
            self._ctx.version_code,
        )

        # Compile shaders
        self._prog = self._ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        # Enable depth testing so the mesh renders correctly.
        # CULL_FACE is intentionally disabled: Apple Object Capture meshes
        # may use clockwise winding, which OpenGL would cull as back-faces.
        # We render both sides and rely on depth testing for correctness.
        self._ctx.enable(moderngl.DEPTH_TEST)

        # Build background quad + ground grid — one-time setup, mesh-independent
        self._build_environment()

        # Compile gizmo shader (ring geometry is built per-mesh in load_mesh)
        self._gizmo_prog = self._ctx.program(
            vertex_shader=GIZMO_VERT_SHADER,
            fragment_shader=GIZMO_FRAG_SHADER,
        )

        # Compile flat-color highlight shader (polygon selection overlays)
        self._hl_prog = self._ctx.program(
            vertex_shader=HIGHLIGHT_VERT_SHADER,
            fragment_shader=HIGHLIGHT_FRAG_SHADER,
        )

        # Compile UV-space mask highlight shader.  Renders the full mesh and
        # discards any fragment whose UV coordinate is not painted in uv_mask.
        self._uvmask_prog = self._ctx.program(
            vertex_shader=UVMASK_VERT_SHADER,
            fragment_shader=UVMASK_FRAG_SHADER,
        )

        # Compile face-ID shader for pixel-accurate polygon selection (FBO-based).
        # Each fragment is written with its triangle's 24-bit RGB-encoded index
        # so readback + polygon mask gives exact Blender-quality lasso behavior.
        self._faceid_prog = self._ctx.program(
            vertex_shader=FACEID_VERT_SHADER,
            fragment_shader=FACEID_FRAG_SHADER,
        )

        # Compile fullscreen quad programs for the composite pipeline:
        #   1. Copy pass     — blit offscreen scene color → Qt framebuffer
        #   2. Overlay pass  — blend PIL polygon mask (Y-flipped, pending selection)
        #   3. Overlay noflip — blend OpenGL-rendered face mask (committed layers)
        self._copy_prog = self._ctx.program(
            vertex_shader=QUAD_VERT_SHADER,
            fragment_shader=COPY_FRAG_SHADER,
        )
        self._screen_overlay_prog = self._ctx.program(
            vertex_shader=QUAD_VERT_SHADER,
            fragment_shader=SCREEN_OVERLAY_FRAG_SHADER,
        )
        self._screen_overlay_prog_noflip = self._ctx.program(
            vertex_shader=QUAD_VERT_SHADER,
            fragment_shader=SCREEN_OVERLAY_FRAG_SHADER_NOFLIP,
        )

        # Compile shader-based planar projection program.
        # Second render pass draws only selected faces with world-space planar UVs.
        self._proj_prog = self._ctx.program(
            vertex_shader=PROJ_VERT_SHADER,
            fragment_shader=PROJ_FRAG_SHADER,
        )

        # Fullscreen NDC quad (two CCW triangles covering [-1,1]²)
        _quad_verts = np.array([
            -1.0, -1.0,   1.0, -1.0,   1.0,  1.0,
            -1.0, -1.0,   1.0,  1.0,  -1.0,  1.0,
        ], dtype=np.float32)
        self._quad_vbo = self._ctx.buffer(_quad_verts.tobytes())
        self._quad_vao_copy = self._ctx.vertex_array(
            self._copy_prog, [(self._quad_vbo, "2f", "in_pos")]
        )
        self._quad_vao_overlay = self._ctx.vertex_array(
            self._screen_overlay_prog, [(self._quad_vbo, "2f", "in_pos")]
        )
        self._quad_vao_overlay_nf = self._ctx.vertex_array(
            self._screen_overlay_prog_noflip, [(self._quad_vbo, "2f", "in_pos")]
        )

        # GPU upload is handled in paintGL() via _needs_upload — no upload here.
        # initializeGL() is sometimes called for hidden widgets at startup,
        # before any mesh has loaded, so we defer upload to the first paintGL
        # where Qt guarantees the context is current.

    def paintGL(self) -> None:
        """Called by Qt each time the widget needs to be redrawn."""
        if self._ctx is None or self._prog is None:
            return

        # QOpenGLWidget renders into its own custom FBO, not the default
        # framebuffer 0. moderngl's ctx.clear() always targets FBO 0, so
        # we must explicitly bind Qt's FBO before every clear and draw call.
        # Without this, all rendering goes to the wrong framebuffer and the
        # widget surface stays empty (shows Qt's widget background instead).
        qt_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())

        # Use the offscreen scene FBO as the render target when available.
        # This lets us sample its depth texture in the overlay pass so the
        # selection highlight is depth-clipped to mesh pixels only.
        # Falls back to qt_fbo if _scene_fbo is not yet built (e.g. first frame
        # before resizeGL has been called).
        has_overlay = (
            self._scene_fbo is not None
            and self._scene_color_tex is not None
            and self._scene_depth_tex is not None
            and self._copy_prog is not None
            and self._screen_overlay_prog is not None
            and self._quad_vao_copy is not None
            and self._quad_vao_overlay is not None
        )
        scene_target = self._scene_fbo if has_overlay else qt_fbo
        scene_target.use()

        # Local helper: copy the offscreen scene to qt_fbo.
        # Called before any return from paintGL and before the overlay pass.
        def _flush_scene() -> None:
            if not has_overlay:
                return   # already rendering to qt_fbo — nothing to flush
            qt_fbo.use()
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._scene_color_tex.use(0)
            self._copy_prog["scene_color"].value = 0
            self._quad_vao_copy.render()
            self._ctx.enable(moderngl.DEPTH_TEST)

        # Deferred GPU upload: load_mesh() sets _needs_upload=True and calls
        # update(). Qt then calls paintGL() with the context guaranteed current.
        if self._needs_upload and self._painter is not None:
            self._needs_upload = False
            try:
                self._upload_mesh()
                self._upload_texture_full()
                # Build gizmo rings here — after _upload_mesh so the GL context
                # is guaranteed current (paintGL is always called with context active).
                # Building in load_mesh() was unreliable because _upload_mesh()
                # calls _release_gpu_resources() which previously wiped the rings.
                self._build_gizmo_rings()
                log.info("ViewportWidget: GPU upload complete — vao=%s tex=%s",
                         self._vao, self._albedo_tex)
            except Exception:
                log.exception("ViewportWidget: GPU upload failed")

        # Sculpt VBO re-upload: vertex positions changed but texture is intact.
        # _upload_mesh_only() rebuilds VBO/VAO/IBO without touching _albedo_tex.
        if self._needs_mesh_upload and self._painter is not None:
            self._needs_mesh_upload = False
            try:
                self._upload_mesh_only()
            except Exception:
                log.exception("ViewportWidget: sculpt VBO re-upload failed")

        # Deferred projection GPU setup: build VAO/IBO/texture for any pending
        # projection layers.  Done here (inside Qt's managed GL frame) so the
        # context is guaranteed current — avoids the makeCurrent/doneCurrent
        # lifecycle issue that caused projections to silently render nothing.
        if self._pending_proj_setups and self._proj_prog is not None and self._vbo is not None:
            for _lid, _setup in list(self._pending_proj_setups.items()):
                try:
                    _tex_img  = _setup["texture_img"]
                    _face_set = _setup["face_set"]
                    _params   = _setup["params"]

                    # Upload texture to GPU.
                    _tw, _th  = _tex_img.size
                    _gpu_tex  = self._ctx.texture((_tw, _th), 4, data=_tex_img.tobytes())
                    _gpu_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                    _gpu_tex.build_mipmaps()

                    # Build IBO from all selected faces (no front-face filtering needed —
                    # the polygon mask clips the projection to the drawn shape each frame).
                    _face_arr    = np.asarray(self._painter._mesh.faces, dtype=np.int32)
                    _sel_idx     = np.array(sorted(_face_set), dtype=np.int32)
                    _ibo_data    = _face_arr[_sel_idx].flatten().astype(np.uint32)
                    _proj_ibo    = self._ctx.buffer(_ibo_data.tobytes())

                    # VAO: position-only binding from the interleaved VBO.
                    # Layout: [pos(3f) | normal(3f) | uv(2f)] = 32 bytes per vertex.
                    # "3f 20x" = read 3 floats (pos), skip 20 bytes (normal+uv).
                    _proj_vao = self._ctx.vertex_array(
                        self._proj_prog,
                        [(self._vbo, "3f 20x", "in_position")],
                        _proj_ibo,
                    )

                    # Release stale GPU resources if re-projecting the same layer.
                    _old = self._proj_layers.get(_lid)
                    if _old is not None:
                        for _k in ("gpu_tex", "ibo", "vao"):
                            _r = _old.get(_k)
                            if _r is not None:
                                try:
                                    _r.release()
                                except Exception:
                                    pass

                    self._proj_layers[_lid] = {
                        "gpu_tex": _gpu_tex,
                        "ibo":     _proj_ibo,
                        "vao":     _proj_vao,
                        "params":  _params,
                        "visible": _setup.get("visible", True),
                    }
                    log.info("paintGL: proj GPU setup complete layer=%d faces=%d tex=%dx%d",
                             _lid, len(_sel_idx), _tw, _th)
                except Exception:
                    log.exception("paintGL: proj GPU setup FAILED layer=%d", _lid)
                finally:
                    self._pending_proj_setups.pop(_lid, None)

        # Reproject in-progress polygon anchors when the camera actually moved.
        # _camera_moved is set to True by orbit/pan handlers and cleared here
        # each frame — reprojection only fires on genuine camera navigation, not
        # every frame, to prevent floating-point drift from accumulating.
        if self._camera_moved:
            if self._poly_points_3d:
                self._reproject_poly_overlay(self._poly_cursor)
            self._camera_moved = False

        # Clear depth to 1.0 and color to black; the background gradient quad
        # will immediately overwrite every color pixel, so the clear color
        # doesn't matter visually — we just need a clean depth buffer.
        scene_target.clear(0.0, 0.0, 0.0)

        # --- 1. Background gradient (fullscreen NDC quad) ---
        # Depth test and depth write are disabled so the quad paints every
        # pixel without interfering with scene depth values.
        if self._bg_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._bg_vao.render()
            self._ctx.enable(moderngl.DEPTH_TEST)

        # If no camera is ready, flush whatever was drawn (background) and stop.
        if self._camera is None:
            _flush_scene()
            return

        # Compute camera matrices shared across grid, mesh, and gizmo
        aspect = self.width() / max(self.height(), 1)
        view   = self._camera.get_view_matrix()
        proj   = self._camera.get_projection_matrix(aspect)
        # VP = proj @ view — used for the grid and gizmo (world-space geometry)
        vp     = (proj @ view).astype("f4")
        # MVP = VP @ model — used for the mesh (model matrix includes rotation about centroid)
        model  = self._model_mat_4x4()
        mvp    = (proj @ view @ model.astype(np.float64)).astype("f4")

        # --- 2. Mesh (opaque, depth-tested) ---
        if self._painter is None or self._vao is None:
            # No mesh yet — flush background so it shows on screen
            _flush_scene()
            return

        # Upload any texture pixels painted since the last frame
        self._refresh_dirty_texture()

        # MV = view @ model — used to transform normals to view space
        mv = (view @ model.astype(np.float64)).astype("f4")

        # Upload uniforms — OpenGL expects column-major (Fortran order).
        # numpy stores matrices row-major, so .T.tobytes() transposes before
        # serialising, giving OpenGL the correct column-major layout.
        self._prog["MVP"].write(mvp.T.tobytes())
        self._prog["MV"].write(mv.T.tobytes())
        self._prog["albedo"].value = 0   # texture unit 0

        # Bind texture and draw
        self._albedo_tex.use(0)
        self._vao.render()

        # --- 2b. Shader projection layers (second pass, selected faces only) ---
        # Each active projection layer is rendered over the albedo using planar
        # UV projection computed in world space. This gives a perfect shape
        # (exactly the face selection) with full GPU bilinear/mipmap quality.
        if (self._proj_layers and self._proj_prog is not None
                and self._layer_mask_tex is not None):
            from PIL import Image as _PilProj, ImageDraw as _PilDProj  # noqa: PLC0415
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            # LEQUAL so the projection pass (same faces as base mesh) passes depth.
            self._ctx.depth_func = "<="
            w_vp = float(self.width())
            h_vp = float(self.height())
            self._proj_prog["MVP"].write(mvp.T.tobytes())
            self._proj_prog["proj_tex"].value   = 1   # source texture on unit 1
            self._proj_prog["layer_mask"].value = 2   # polygon mask on unit 2
            self._proj_prog["viewport_w"].value = w_vp
            self._proj_prog["viewport_h"].value = h_vp

            for lid, pdata in self._proj_layers.items():
                if not pdata.get("visible", True):
                    continue

                try:
                    # --- Rasterize the polygon mask for this layer ---
                    # Reproject poly_pts_3d through the live MVP so the mask always
                    # matches the current camera angle, then PIL-fill into _layer_mask_tex.
                    # NOTE: No per-frame back-face normal check here — Apple OC meshes have
                    # CW-wound faces whose raw cross-product normals point inward, so a dot-
                    # product test would incorrectly cull front-facing projections.  Back-face
                    # geometry is already excluded by the signed-area IBO filter applied when
                    # the projection was set up; we just trust that and always render.
                    layer       = self._layers.get(lid, {})
                    poly_pts_3d = layer.get("poly_pts_3d")
                    mask_w = self._layer_mask_tex.width
                    mask_h = self._layer_mask_tex.height
                    if poly_pts_3d and len(poly_pts_3d) >= 3:
                        from PIL import ImageFilter as _IF
                        pts_w = np.array(poly_pts_3d, dtype=np.float32)
                        pts4  = np.hstack([pts_w, np.ones((len(pts_w), 1), np.float32)])
                        lproj = (mvp @ pts4.T).T
                        lwc   = np.where(np.abs(lproj[:, 3]) < 1e-8, 1e-8, lproj[:, 3])
                        l_sx  = ((lproj[:, 0] / lwc + 1.0) * 0.5 * mask_w).astype(int)
                        l_sy  = ((1.0 - (lproj[:, 1] / lwc + 1.0) * 0.5) * mask_h).astype(int)
                        # Supersample 4× for clean polygon edge anti-aliasing.
                        # PIL Draw.polygon() has no native AA — rasterizing at 4× screen
                        # resolution then Lanczos-downsampling gives sub-pixel smooth edges
                        # with no staircase artifacts. Minimal blur (radius=1) feathers the
                        # last sub-pixel fringe without introducing softness.
                        _SS = 4
                        lmask_hi = _PilProj.new("L", (mask_w * _SS, mask_h * _SS), 0)
                        _PilDProj.Draw(lmask_hi).polygon(
                            [(int(l_sx[k]) * _SS, int(l_sy[k]) * _SS)
                             for k in range(len(l_sx))],
                            fill=255,
                        )
                        lmask = lmask_hi.resize(
                            (mask_w, mask_h), _PilProj.Resampling.LANCZOS
                        )
                        lmask = lmask.filter(_IF.GaussianBlur(radius=1))
                        self._layer_mask_tex.write(np.array(lmask, dtype=np.uint8).tobytes())
                    else:
                        # No polygon data — fill mask with white (no clipping)
                        self._layer_mask_tex.write(
                            bytes([255] * (mask_w * mask_h))
                        )

                    pr  = pdata["params"]
                    p   = self._proj_prog
                    p["proj_right"].write(pr["right"].tobytes())
                    p["proj_up"].write(pr["up"].tobytes())
                    p["proj_r_min"].value    = pr["r_min"]
                    p["proj_r_range"].value  = pr["r_range"]
                    p["proj_u_min"].value    = pr["u_min"]
                    p["proj_u_range"].value  = pr["u_range"]
                    p["proj_scale"].value    = pr["scale"]
                    p["proj_cos"].value      = pr["cos"]
                    p["proj_sin"].value      = pr["sin"]
                    p["proj_offset_x"].value = pr["offset_x"]
                    p["proj_offset_y"].value = pr["offset_y"]
                    p["proj_opacity"].value  = pr["opacity"]
                    pdata["gpu_tex"].use(1)
                    self._layer_mask_tex.use(2)
                    pdata["vao"].render()
                except Exception:
                    log.exception("paintGL: projection render EXCEPTION lid=%d", lid)

            # Restore depth func and blend state
            self._ctx.depth_func = "<"
            self._ctx.disable(moderngl.BLEND)

        # --- 3. Ground grid (alpha-blended, depth-tested, drawn AFTER mesh) ---
        # Rendering the grid after the mesh lets the depth buffer handle both
        # viewing angles correctly automatically:
        #   • Camera above grid → mesh is closer to camera than grid → mesh wrote
        #     smaller depth values first → grid fragments behind the mesh FAIL the
        #     depth test → mesh occludes grid lines as expected.
        #   • Camera below grid → grid is closer to camera than the mesh → grid
        #     fragments have smaller depth values than the mesh → they PASS the
        #     depth test and draw on top → grid lines remain visible through mesh.
        if self._grid_vao is not None:
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            # Grid is pure world-space — use VP (no model transform)
            self._grid_prog["MVP"].write(vp.T.tobytes())
            self._grid_prog["grid_y"].value = self._grid_y
            # Pass camera world position so the fragment shader can compute fade
            self._grid_prog["cam_pos"].write(
                self._camera.position.astype("f4").tobytes()
            )
            self._grid_vao.render()
            self._ctx.disable(moderngl.BLEND)

        # --- 4. Rotation gizmo rings (only when rotate tool is active) ---
        if self._tool == "rotate" and self._gizmo_prog is not None and self._gizmo_vaos:
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._gizmo_prog["VP"].write(vp.T.tobytes())

            # Ring colors: idle slightly translucent, active fully opaque + bright
            axis_colors = {
                "x": (0.95, 0.22, 0.22, 0.85),  # red
                "y": (0.20, 0.90, 0.25, 0.85),  # green
                "z": (0.20, 0.45, 1.00, 0.85),  # blue
            }
            active = self._gizmo_axis or self._gizmo_hover

            for ax in ("x", "y", "z"):
                ring_vao    = self._gizmo_vaos.get(ax)
                diamond_vao = self._gizmo_diamond_vaos.get(ax)
                if ring_vao is None:
                    continue

                r_, g_, b_, a_ = axis_colors[ax]
                if ax == active:
                    # Highlight: brighter and fully opaque on hover/drag
                    r_ = min(r_ * 1.25 + 0.15, 1.0)
                    g_ = min(g_ * 1.25 + 0.15, 1.0)
                    b_ = min(b_ * 1.25 + 0.15, 1.0)
                    a_ = 1.0

                self._gizmo_prog["ring_color"].write(
                    np.array([r_, g_, b_, a_], dtype=np.float32).tobytes()
                )
                # Draw ring band (thin TRIANGLE_STRIP annulus)
                ring_vao.render(moderngl.TRIANGLE_STRIP)
                # Draw diamond handle (two filled triangles at angle 0)
                if diamond_vao is not None:
                    diamond_vao.render(moderngl.TRIANGLES)

            self._ctx.disable(moderngl.BLEND)

        # --- 5. Flush scene to Qt framebuffer + screen-space overlay pass ---
        #
        # First: copy the offscreen scene (background + grid + mesh + gizmo)
        # from _scene_fbo to qt_fbo via a simple fullscreen blit.
        #
        # Then: for each visible selection (pending polygon or saved layer),
        # run a second fullscreen pass that composites a per-pixel polygon mask
        # on top of the scene, clipped to mesh pixels via the depth texture.
        #
        # This approach gives pixel-perfect selection boundaries that exactly
        # match the drawn polygon shape — no triangle-edge jagging, no UV atlas
        # issues.  Each mask is a grayscale R8 texture rasterized from the
        # user's polygon at viewport resolution by _finalize_poly_selection().
        _flush_scene()
        qt_fbo.use()

        # Run overlay passes only when the composite infrastructure is ready
        # and there is at least one visible selection to show.
        _pending_has_overlay  = self._pending_screen_mask_tex is not None
        _layers_have_overlays = any(
            layer.get("visible", True)
            and layer.get("poly_pts_3d") is not None
            # Layers with an active shader projection skip the color highlight
            # overlay — the projected texture IS the visual for these layers.
            and lid not in self._proj_layers
            for lid, layer in self._layers.items()
        )
        if has_overlay and (_pending_has_overlay or _layers_have_overlays):
            self._ctx.disable(moderngl.DEPTH_TEST)

            # --- Pending selection: teal/cyan overlay (screen-space PIL mask) ---
            # Reproject 3D anchor points each frame so the overlay tracks the
            # mesh during orbit.  Done inline here (not in _camera_moved) so we
            # are guaranteed to be inside an active GL context with qt_fbo bound,
            # using .write() on the existing texture (safe mid-frame pattern,
            # same as committed layers) instead of allocating a new texture object.
            if _pending_has_overlay and self._pending_poly_pts_3d is not None:
                from PIL import Image as _PilPM, ImageDraw as _PilPMD  # noqa: PLC0415
                _pm_pts = [p for p in self._pending_poly_pts_3d if p is not None]
                if len(_pm_pts) >= 3:
                    _pm_vw = float(self.width())
                    _pm_vh = float(self.height())
                    if _pm_vw > 0 and _pm_vh > 0:
                        _pm_w    = np.array(_pm_pts, dtype=np.float32)
                        _pm_view  = self._camera.get_view_matrix()
                        _pm_proj  = self._camera.get_projection_matrix(_pm_vw / _pm_vh)
                        _pm_model = self._model_mat_4x4()
                        _pm_mvp  = (_pm_proj @ _pm_view
                                    @ _pm_model.astype(np.float64)).astype(np.float32)
                        _pm_pts4 = np.hstack([_pm_w, np.ones((len(_pm_w), 1), np.float32)])
                        _pm_clip = (_pm_mvp @ _pm_pts4.T).T
                        _pm_wc   = np.where(np.abs(_pm_clip[:, 3]) < 1e-8,
                                            1e-8, _pm_clip[:, 3])
                        _pm_sx   = ((_pm_clip[:, 0] / _pm_wc + 1.0) * 0.5 * _pm_vw).astype(int)
                        _pm_sy   = ((1.0 - (_pm_clip[:, 1] / _pm_wc + 1.0) * 0.5) * _pm_vh).astype(int)
                        _pm_img  = _PilPM.new("L", (int(_pm_vw), int(_pm_vh)), 0)
                        _PilPMD.Draw(_pm_img).polygon(
                            [(_pm_sx[k], _pm_sy[k]) for k in range(len(_pm_sx))], fill=255
                        )
                        self._pending_screen_mask_tex.write(
                            np.array(_pm_img, dtype=np.uint8).tobytes()
                        )
                        # Keep cookie-cutter matrices in sync with current camera.
                        self._pending_poly_pts_2d = [
                            (float(_pm_sx[k]), float(_pm_sy[k])) for k in range(len(_pm_sx))
                        ]
                        self._pending_view_mat  = _pm_view.astype(np.float64).copy()
                        self._pending_proj_mat  = _pm_proj.astype(np.float64).copy()
                        self._pending_model_mat = _pm_model.astype(np.float64).copy()

            if _pending_has_overlay:
                self._ctx.enable(moderngl.BLEND)
                self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self._scene_depth_tex.use(0)
                self._screen_overlay_prog["scene_depth"].value = 0
                self._screen_overlay_prog["sel_mask"].value    = 1
                self._screen_overlay_prog["overlay_color"].write(
                    np.array([0.1, 0.85, 0.9, 0.55], dtype=np.float32).tobytes()
                )
                # Pending selection is always shown — no depth occlusion needed.
                self._screen_overlay_prog["poly_depth"].value = 0.0
                self._pending_screen_mask_tex.use(1)
                self._quad_vao_overlay.render()
                self._ctx.disable(moderngl.BLEND)

            # --- Committed layer highlights: reproject 3D anchor polygon ---
            #
            # Each frame, the polygon's saved 3D world-space anchor points are
            # projected through the current camera MVP to new screen positions.
            # PIL rasterizes the reprojected polygon into a fresh mask texture
            # which is then composited via the depth-clipped overlay shader.
            #
            # Result: the drawn polygon shape stays glued to the model surface
            # from any orbit angle, with pixel-accurate boundaries.
            w = self.width()
            h = self.height()
            if _layers_have_overlays and self._layer_mask_tex is not None:
                from PIL import Image as _PilL, ImageDraw as _PilDL  # noqa: PLC0415
                for lid, layer in self._layers.items():
                    if not layer.get("visible", True):
                        continue
                    # Skip the colored highlight overlay when a shader projection
                    # exists for this layer — the projected texture IS the visual
                    # and the opaque highlight would cover it completely.
                    if lid in self._proj_layers:
                        continue
                    poly_pts_3d = layer.get("poly_pts_3d")
                    if not poly_pts_3d or len(poly_pts_3d) < 3:
                        continue

                    # Project 3D anchors → current screen positions
                    # NOTE: No back-face cull here — Apple OC meshes use CW winding so
                    # the raw avg_normal points inward; a dot-product test would hide the
                    # highlight whenever the camera faces the selection front.
                    pts_w = np.array(poly_pts_3d, dtype=np.float32)       # (K, 3)
                    pts4  = np.hstack([pts_w, np.ones((len(pts_w), 1), np.float32)])
                    lproj = (mvp @ pts4.T).T                               # (K, 4)
                    lwc   = np.where(np.abs(lproj[:, 3]) < 1e-8, 1e-8, lproj[:, 3])
                    l_sx  = ((lproj[:, 0] / lwc + 1.0) * 0.5 * w).astype(int)
                    l_sy  = ((1.0 - (lproj[:, 1] / lwc + 1.0) * 0.5) * h).astype(int)

                    # PIL-rasterize the reprojected polygon (top-left origin)
                    lmask_pil = _PilL.new("L", (w, h), 0)
                    _PilDL.Draw(lmask_pil).polygon(
                        [(int(l_sx[k]), int(l_sy[k])) for k in range(len(l_sx))],
                        fill=255,
                    )
                    # Write directly into the reusable layer mask texture
                    self._layer_mask_tex.write(
                        np.array(lmask_pil, dtype=np.uint8).tobytes()
                    )

                    # Composite onto qt_fbo — Y-flip variant (PIL origin)
                    qt_fbo.use()
                    r_, g_, b_, a_ = layer["color"]
                    if lid == self._active_layer_id:
                        a_ = min(1.0, a_ * 2.0)
                    self._ctx.enable(moderngl.BLEND)
                    self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    self._scene_depth_tex.use(0)
                    self._screen_overlay_prog["scene_depth"].value = 0
                    self._screen_overlay_prog["sel_mask"].value    = 1
                    self._screen_overlay_prog["overlay_color"].write(
                        np.array([r_, g_, b_, a_], dtype=np.float32).tobytes()
                    )
                    # Depth occlusion handled by the back-face cull above;
                    # pass 0.0 here to disable the shader-side depth test.
                    self._screen_overlay_prog["poly_depth"].value = 0.0
                    self._layer_mask_tex.use(1)
                    self._quad_vao_overlay.render()
                    self._ctx.disable(moderngl.BLEND)

            self._ctx.enable(moderngl.DEPTH_TEST)

        # Re-project the 3D dot anchor to screen every frame so the white dot
        # tracks the mesh surface point as the model orbits.
        if self._dot_world_pos is not None and self._dot_overlay.isVisible():
            screen = self._world_to_screen(self._dot_world_pos)
            if screen is not None:
                self._dot_overlay.place(*screen)
            else:
                # Hit point behind camera — hide dot until it's visible again
                self._dot_overlay.hide()

    def resizeGL(self, w: int, h: int) -> None:
        """Called by Qt when the widget is resized."""
        if self._ctx is not None:
            self._ctx.viewport = (0, 0, w, h)
            # Rebuild the offscreen scene FBO at the new size so the depth
            # texture matches the viewport — needed for the overlay depth test.
            self._rebuild_scene_fbo(w, h)
        # Keep the poly overlay sized to match the viewport
        self._poly_overlay.resize(self.size())
        self.update()

    # ------------------------------------------------------------------ #
    # GPU resource management                                              #
    # ------------------------------------------------------------------ #

    def _upload_mesh(self) -> None:
        """
        Build interleaved VBO [x,y,z, nx,ny,nz, u,v] and upload to GPU.

        Called from initializeGL (if painter is ready) or load_mesh (if GL
        context already exists). Releases any prior GPU buffers first.
        """
        self._release_gpu_resources()

        verts, norms, uvs, faces = self._painter.get_render_arrays()

        # Interleave: each vertex = [pos(3), normal(3), uv(2)] = 8 floats
        interleaved = np.hstack([verts, norms, uvs]).astype(np.float32)  # (N, 8)

        self._vbo = self._ctx.buffer(interleaved.tobytes())
        self._ibo = self._ctx.buffer(faces.tobytes())

        stride = 8 * 4  # 8 floats × 4 bytes
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            self._ibo,
        )

        # UV-mask highlight VAO: reuses the same VBO/IBO but only binds
        # in_position (3f) and in_uv (2f), skipping the 12-byte normal (12x).
        if self._uvmask_prog is not None:
            self._hl_mesh_vao = self._ctx.vertex_array(
                self._uvmask_prog,
                [(self._vbo, "3f 12x 2f", "in_position", "in_uv")],
                self._ibo,
            )

        # Rebuild projection VAOs to point at the new VBO
        self._rebuild_proj_vaos()

    def _upload_texture_full(self) -> None:
        """Upload the full albedo texture to a moderngl Texture (GPU)."""
        if self._albedo_tex is not None:
            self._albedo_tex.release()

        w, h = self._painter.get_texture_size()
        rgba_bytes = self._painter.get_texture_rgba()

        # moderngl expects RGBA data as bytes; components=4 means RGBA
        self._albedo_tex = self._ctx.texture(
            (w, h), components=4, data=rgba_bytes
        )
        self._albedo_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._albedo_tex.repeat_x = False
        self._albedo_tex.repeat_y = False

    def _upload_mesh_only(self) -> None:
        """
        Re-upload vertex position/normal data to the GPU without touching the texture.

        Called after sculpt operations — the texture is unchanged so we avoid
        the cost of reuploading it. Releases only the geometry buffers (VAO/VBO/IBO),
        then rebuilds them from the current painter state.
        """
        # Release geometry buffers only — _albedo_tex is intentionally preserved
        for attr in ("_vao", "_vbo", "_ibo", "_hl_mesh_vao"):
            resource = getattr(self, attr, None)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
            setattr(self, attr, None)

        verts, norms, uvs, faces = self._painter.get_render_arrays()
        interleaved = np.hstack([verts, norms, uvs]).astype(np.float32)

        self._vbo = self._ctx.buffer(interleaved.tobytes())
        self._ibo = self._ctx.buffer(faces.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            self._ibo,
        )
        if self._uvmask_prog is not None:
            self._hl_mesh_vao = self._ctx.vertex_array(
                self._uvmask_prog,
                [(self._vbo, "3f 12x 2f", "in_position", "in_uv")],
                self._ibo,
            )

        # Rebuild projection VAOs to point at the new VBO
        self._rebuild_proj_vaos()

    def _rebuild_proj_vaos(self) -> None:
        """
        Rebuild projection VAOs after a VBO change (sculpt op or new mesh load).

        The interleaved VBO is recreated by _upload_mesh() and _upload_mesh_only()
        whenever the mesh geometry changes. Any VAO that references the old VBO
        is stale and must be rebuilt. Textures and IBOs are reused unchanged.
        """
        if self._proj_prog is None or self._vbo is None or not self._proj_layers:
            return
        face_arr = np.asarray(self._painter._mesh.faces, dtype=np.int32)
        for lid, pdata in self._proj_layers.items():
            # Release the stale VAO (it referenced the old VBO)
            old_vao = pdata.get("vao")
            if old_vao is not None:
                try:
                    old_vao.release()
                except Exception:
                    pass

            # Rebuild IBO from current face array (face indices may have shifted
            # after sculpt-induced mesh operations like remove-floaters).
            params    = pdata.get("params", {})
            face_set  = params.get("face_set")
            if face_set is None:
                continue
            sel_idx      = np.array(sorted(face_set), dtype=np.int32)
            sel_ibo_data = face_arr[sel_idx].flatten().astype(np.uint32)
            old_ibo = pdata.get("ibo")
            if old_ibo is not None:
                try:
                    old_ibo.release()
                except Exception:
                    pass
            new_ibo = self._ctx.buffer(sel_ibo_data.tobytes())

            # Rebuild VAO bound to the new VBO
            new_vao = self._ctx.vertex_array(
                self._proj_prog,
                [(self._vbo, "3f 20x", "in_position")],
                new_ibo,
            )
            pdata["ibo"] = new_ibo
            pdata["vao"] = new_vao

    def _refresh_dirty_texture(self) -> None:
        """
        Upload only the modified rectangle of the texture to the GPU.

        MeshPainter accumulates the bounding rect of all paint operations
        since the last call. We read that rect and call glTexSubImage2D
        (via moderngl's write() method) to avoid reuploading the full texture.
        """
        if self._painter is None or self._albedo_tex is None:
            return

        rect = self._painter.get_dirty_rect()
        if rect is None:
            return

        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y
        if w <= 0 or h <= 0:
            return

        patch_bytes = self._painter.get_dirty_patch(x, y, x2, y2)
        # moderngl Texture.write() maps to glTexSubImage2D when viewport given
        self._albedo_tex.write(patch_bytes, viewport=(x, y, w, h))

    def _rebuild_scene_fbo(self, w: int, h: int) -> None:
        """
        (Re)build the offscreen scene FBO at the given pixel dimensions.

        Called from resizeGL() whenever the viewport size changes. The FBO
        contains a color texture (RGBA8) and a depth texture — both are
        needed by the screen-space overlay composite pass. Releases any
        previously allocated resources before allocating new ones.

        Must be called with the GL context current.
        """
        if self._ctx is None or w <= 0 or h <= 0:
            return

        # Release existing attachments before reallocating at new size
        for attr in ("_scene_fbo", "_scene_color_tex", "_scene_depth_tex"):
            res = getattr(self, attr, None)
            if res is not None:
                try:
                    res.release()
                except Exception:
                    pass
            setattr(self, attr, None)

        # RGBA8 color texture + depth texture — color for the copy pass,
        # depth for the overlay's mesh-pixel clipping test.
        self._scene_color_tex = self._ctx.texture((w, h), 4)
        self._scene_depth_tex = self._ctx.depth_texture((w, h))
        self._scene_fbo = self._ctx.framebuffer(
            color_attachments=[self._scene_color_tex],
            depth_attachment=self._scene_depth_tex,
        )

        # Single-channel layer mask texture — written each frame by PIL-rasterizing
        # the committed layer's polygon after reprojecting its 3D anchor points
        # through the current camera. 1-channel (R8) matches the overlay shader's
        # sel_mask sampler and minimises per-frame upload bandwidth.
        if self._layer_mask_tex is not None:
            try:
                self._layer_mask_tex.release()
            except Exception:
                pass
            self._layer_mask_tex = None
        self._layer_mask_tex = self._ctx.texture((w, h), 1)
        log.debug("ViewportWidget: scene FBO rebuilt at %dx%d", w, h)

    def _release_gpu_resources(self) -> None:
        """Release mesh GPU buffers and textures (called on each new mesh load)."""
        for attr in ("_hl_mesh_vao", "_vao", "_vbo", "_ibo", "_albedo_tex"):
            resource = getattr(self, attr, None)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
            setattr(self, attr, None)
        # Note: gizmo ring buffers are NOT released here. _upload_mesh() calls
        # _release_gpu_resources() and we want the gizmo to survive a mesh upload.
        # Gizmo teardown is handled explicitly in reset() and _release_environment().

        # Release polygon selection highlight resources — face indices are
        # invalidated whenever the mesh geometry changes (new load or op).
        self._release_pending_vao()
        self._release_layer_vaos()

    def _build_environment(self) -> None:
        """
        Compile environment shaders and upload static geometry to the GPU.

        Background: a fullscreen NDC quad (two triangles) placed at z=0.9999
        so it renders behind all scene geometry without needing special ordering.

        Ground grid: a large XZ quad (−200..+200 units) with line details
        computed entirely in the fragment shader using fwidth() for AA.

        Called once from initializeGL(); persists across mesh loads.
        """
        # --- Background gradient quad ---
        self._bg_prog = self._ctx.program(
            vertex_shader=BG_VERT_SHADER,
            fragment_shader=BG_FRAG_SHADER,
        )
        # Fullscreen quad as two triangles in NDC space
        bg_verts = np.array([
            -1.0, -1.0,   1.0, -1.0,   1.0,  1.0,
            -1.0, -1.0,   1.0,  1.0,  -1.0,  1.0,
        ], dtype=np.float32)
        self._bg_vbo = self._ctx.buffer(bg_verts.tobytes())
        self._bg_vao = self._ctx.vertex_array(
            self._bg_prog, [(self._bg_vbo, "2f", "in_pos")]
        )

        # --- Ground grid quad ---
        self._grid_prog = self._ctx.program(
            vertex_shader=GRID_VERT_SHADER,
            fragment_shader=GRID_FRAG_SHADER,
        )
        # Large flat XZ quad; Y is set per-frame via the grid_y uniform
        r = 200.0
        grid_verts = np.array([
            -r, -r,   r, -r,   r,  r,
            -r, -r,   r,  r,  -r,  r,
        ], dtype=np.float32)
        self._grid_vbo = self._ctx.buffer(grid_verts.tobytes())
        self._grid_vao = self._ctx.vertex_array(
            self._grid_prog, [(self._grid_vbo, "2f", "in_xz")]
        )

    def _release_environment(self) -> None:
        """Release background and grid GPU resources (called on full context teardown)."""
        for attr in ("_bg_vao", "_bg_vbo", "_bg_prog",
                     "_grid_vao", "_grid_vbo", "_grid_prog"):
            resource = getattr(self, attr, None)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
            setattr(self, attr, None)

    def _build_gizmo_rings(self) -> None:
        """
        Build three LINE_LOOP ring VAOs for the rotation gizmo (X/Y/Z axes).

        Rings are thin TRIANGLE_STRIP bands (±4% of radius) so they appear
        slightly thicker than a 1px line while staying clean and non-chunky.
        Each ring also has a small filled diamond handle — a visual grab point
        at the leading edge of the ring (angle 0) that makes it easy to click.

        X ring (red)   — lies in the YZ plane (rotates model around world X)
        Y ring (green) — lies in the XZ plane (rotates model around world Y)
        Z ring (blue)  — lies in the XY plane (rotates model around world Z)
        """
        if self._ctx is None or self._gizmo_prog is None or self._painter is None:
            return

        self._release_gizmo()

        bbox_min, bbox_max = self._painter.get_bbox()
        diagonal = float(np.linalg.norm(bbox_max - bbox_min))
        radius   = diagonal * 0.15   # compact cluster near the model
        inner_r  = radius * 0.96     # ±4% band — slightly thick, not fat
        outer_r  = radius * 1.04

        cx = float(self._mesh_centroid[0]) - diagonal * 0.35
        cy = float(bbox_max[1]) + radius * 1.5
        cz = float(self._mesh_centroid[2])
        self._gizmo_center = np.array([cx, cy, cz], dtype=np.float32)

        # Diamond handle size — 18% of ring radius on each arm
        ds = radius * 0.18

        N = 128
        angles_strip = np.linspace(0, 2 * math.pi, N + 1, dtype=np.float32)
        cos_s, sin_s = np.cos(angles_strip), np.sin(angles_strip)
        angles_mid   = np.linspace(0, 2 * math.pi, N, endpoint=False, dtype=np.float32)
        cos_m, sin_m = np.cos(angles_mid), np.sin(angles_mid)

        def _strip(inner_pts, outer_pts):
            """Interleave inner/outer vertex pairs into a TRIANGLE_STRIP array."""
            buf = np.empty((2 * (N + 1), 3), dtype=np.float32)
            buf[0::2] = inner_pts
            buf[1::2] = outer_pts
            return buf

        def _diamond(center, radial, tangent):
            """
            Four-vertex filled diamond at `center`.

            `radial`  — direction pointing outward along the ring radius.
            `tangent` — direction along the ring tangent at angle 0.
            Returns 6 float32 vertices (2 triangles) forming a rhombus.
            """
            v0 = center + radial  * ds   # outer tip
            v1 = center + tangent * ds   # side tip
            v2 = center - radial  * ds   # inner tip
            v3 = center - tangent * ds   # other side tip
            # Two triangles: (v0,v1,v2) and (v0,v2,v3)
            return np.array([v0, v1, v2, v0, v2, v3], dtype=np.float32)

        # --- X ring: annulus in the YZ plane ---
        xi = np.stack([np.full(N+1, cx), cy + inner_r * cos_s, cz + inner_r * sin_s], axis=1)
        xo = np.stack([np.full(N+1, cx), cy + outer_r * cos_s, cz + outer_r * sin_s], axis=1)
        x_mid = np.stack([np.full(N, cx), cy + radius * cos_m, cz + radius * sin_m], axis=1)
        # Diamond at angle 0 on X ring: center=(cx, cy+radius, cz), radial=(0,1,0), tangent=(0,0,1)
        x_diam = _diamond(
            np.array([cx, cy + radius, cz]),
            np.array([0.0,  1.0, 0.0]),
            np.array([0.0,  0.0, 1.0]),
        )

        # --- Y ring: annulus in the XZ plane ---
        yi = np.stack([cx + inner_r * cos_s, np.full(N+1, cy), cz + inner_r * sin_s], axis=1)
        yo = np.stack([cx + outer_r * cos_s, np.full(N+1, cy), cz + outer_r * sin_s], axis=1)
        y_mid = np.stack([cx + radius * cos_m, np.full(N, cy), cz + radius * sin_m], axis=1)
        y_diam = _diamond(
            np.array([cx + radius, cy, cz]),
            np.array([ 1.0, 0.0, 0.0]),
            np.array([ 0.0, 0.0, 1.0]),
        )

        # --- Z ring: annulus in the XY plane ---
        zi = np.stack([cx + inner_r * cos_s, cy + inner_r * sin_s, np.full(N+1, cz)], axis=1)
        zo = np.stack([cx + outer_r * cos_s, cy + outer_r * sin_s, np.full(N+1, cz)], axis=1)
        z_mid = np.stack([cx + radius * cos_m, cy + radius * sin_m, np.full(N, cz)], axis=1)
        z_diam = _diamond(
            np.array([cx + radius, cy, cz]),
            np.array([ 1.0, 0.0, 0.0]),
            np.array([ 0.0, 1.0, 0.0]),
        )

        for ax_name, strip_data, mid, diam_data in (
            ("x", _strip(xi, xo), x_mid, x_diam),
            ("y", _strip(yi, yo), y_mid, y_diam),
            ("z", _strip(zi, zo), z_mid, z_diam),
        ):
            vbo = self._ctx.buffer(strip_data.tobytes())
            vao = self._ctx.vertex_array(
                self._gizmo_prog, [(vbo, "3f", "in_position")]
            )
            self._gizmo_vbos[ax_name] = vbo
            self._gizmo_vaos[ax_name] = vao
            self._gizmo_ring_pts[ax_name] = mid   # midline for hit testing

            dvbo = self._ctx.buffer(diam_data.tobytes())
            dvao = self._ctx.vertex_array(
                self._gizmo_prog, [(dvbo, "3f", "in_position")]
            )
            self._gizmo_diamond_vbos[ax_name] = dvbo
            self._gizmo_diamond_vaos[ax_name] = dvao

    def _release_gizmo(self) -> None:
        """Release gizmo GPU buffers (called before rebuilding or on teardown)."""
        # Release ring band buffers
        for axis in list(self._gizmo_vaos.keys()):
            try:
                self._gizmo_vaos[axis].release()
                self._gizmo_vbos[axis].release()
            except Exception:
                pass
        self._gizmo_vaos.clear()
        self._gizmo_vbos.clear()
        self._gizmo_ring_pts.clear()
        # Release diamond handle buffers
        for axis in list(self._gizmo_diamond_vaos.keys()):
            try:
                self._gizmo_diamond_vaos[axis].release()
                self._gizmo_diamond_vbos[axis].release()
            except Exception:
                pass
        self._gizmo_diamond_vaos.clear()
        self._gizmo_diamond_vbos.clear()

    # ------------------------------------------------------------------ #
    # Model rotation helpers                                               #
    # ------------------------------------------------------------------ #

    def _model_mat_4x4(self) -> np.ndarray:
        """
        Compute the 4×4 model matrix from _model_rot and _mesh_centroid.

        The model matrix rotates the mesh around its centroid:
            M = T(centroid) @ R4x4 @ T(-centroid)

        This lets the rotation look "in place" rather than orbiting the origin.
        Returns a float32 (4,4) array ready for OpenGL (column-major via .T).
        """
        c = self._mesh_centroid.astype(np.float64)
        R = self._model_rot.astype(np.float64)

        # Build full 4×4 rotation matrix
        R4 = np.eye(4, dtype=np.float64)
        R4[:3, :3] = R

        # Translate to origin, rotate, translate back
        T_pos = np.eye(4, dtype=np.float64)
        T_pos[:3, 3] = c
        T_neg = np.eye(4, dtype=np.float64)
        T_neg[:3, 3] = -c

        return (T_pos @ R4 @ T_neg).astype(np.float32)

    def _to_model_ray(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a world-space ray into model space.

        Since the model matrix rotates about the centroid, the inverse transform
        is: translate to centroid, apply R^T (= R^-1 for orthogonal R), translate back.
        The direction is only rotation-transformed (no translation component).
        """
        R_inv = self._model_rot.T.astype(np.float64)          # R^-1 = R^T
        c     = self._mesh_centroid.astype(np.float64)
        o     = R_inv @ (origin.astype(np.float64) - c) + c   # transform origin
        d     = R_inv @ direction.astype(np.float64)           # transform direction
        norm  = np.linalg.norm(d)
        if norm > 1e-9:
            d /= norm
        return o, d

    def _model_pt_to_world(self, pt: np.ndarray) -> np.ndarray:
        """
        Transform a model-space point back to world space.

        Applies the forward rotation about the centroid:
            world_pt = R @ (pt - centroid) + centroid
        """
        R = self._model_rot.astype(np.float64)
        c = self._mesh_centroid.astype(np.float64)
        return R @ (pt.astype(np.float64) - c) + c

    def _gizmo_hit_test(self, x: int, y: int) -> "str | None":
        """
        Test if a screen pixel (x, y) is within 10px of any gizmo ring.

        Projects each ring's 64 vertices to screen space and measures the
        minimum distance to the cursor. Returns "x", "y", or "z" for the
        closest ring within threshold, or None if no ring is close enough.
        """
        if not self._gizmo_ring_pts or self._camera is None:
            return None

        threshold = 14  # pixels — generous enough for 1px LINE_LOOP rings
        best_axis: "str | None" = None
        best_dist = float("inf")

        for axis, pts in self._gizmo_ring_pts.items():
            for pt in pts:
                screen = self._world_to_screen(pt.astype(np.float64))
                if screen is None:
                    continue
                dist = math.hypot(screen[0] - x, screen[1] - y)
                if dist < threshold and dist < best_dist:
                    best_dist = dist
                    best_axis = axis

        return best_axis

    # ------------------------------------------------------------------ #
    # Mouse input                                                          #
    # ------------------------------------------------------------------ #

    def _world_to_screen(self, world_pt: np.ndarray) -> tuple[int, int] | None:
        """
        Project a world-space point into widget-local pixel coordinates.

        Returns None if the camera is not ready or the point is behind the
        near plane (w <= 0 after projection).
        """
        if self._camera is None:
            return None
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return None
        aspect = w / h
        view = self._camera.get_view_matrix()
        proj = self._camera.get_projection_matrix(aspect)
        pt_h = np.array([*world_pt, 1.0], dtype=np.float64)
        clip = proj @ view @ pt_h
        if clip[3] <= 0:
            return None  # behind near plane
        ndc = clip[:3] / clip[3]
        px = round((ndc[0] + 1.0) * 0.5 * w)
        py = round((1.0 - ndc[1]) * 0.5 * h)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        return (px, py)

    def _compute_lmb_anchor(
        self, click_x: int, click_y: int
    ) -> tuple[tuple[int, int], np.ndarray]:
        """
        Determine the orbit pivot world point and its screen position for an
        LMB press at (click_x, click_y).

        Priority:
          1. Ray-cast against mesh surface — if hit, that point is the pivot.
          2. Miss — bbox centre X/Z at camera's Y elevation (clamped to bbox),
             so orbiting pivots at the model's height nearest the camera.

        Returns (screen_pos, world_pivot). The cursor is warped to screen_pos
        once on press; no per-frame pinning is done so orbit stays stable.
        """
        fallback_screen = (click_x, click_y)

        if self._camera is None or self._painter is None:
            fp = self._camera.focal_point.copy() if self._camera else np.zeros(3)
            return fallback_screen, fp

        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return fallback_screen, self._camera.focal_point.copy()

        aspect = w / h

        # Convert click pixel to NDC (Y flipped: pixel-top = NDC +1)
        ndc_x = (click_x / w) * 2.0 - 1.0
        ndc_y = 1.0 - (click_y / h) * 2.0

        # Unproject to world-space ray, then transform into model space so the
        # BVH intersection is correct even when the mesh has been rotated.
        origin, direction = self._camera.unproject_ray(ndc_x, ndc_y, aspect)
        model_origin, model_dir = self._to_model_ray(origin, direction)

        # 1. Try mesh surface hit
        hit_model = self._painter.ray_cast_world_pos(model_origin, model_dir)
        if hit_model is not None:
            # Transform the model-space hit back to world space for screen projection
            hit_pos = self._model_pt_to_world(hit_model)
            screen = self._world_to_screen(hit_pos) or fallback_screen
            return screen, hit_pos

        # 2. Miss — model centre at camera Y elevation
        bbox_min, bbox_max = self._painter.get_bbox()
        centre   = (bbox_min.astype(np.float64) + bbox_max.astype(np.float64)) * 0.5
        cam_y    = float(self._camera.position[1])
        pivot_y  = float(np.clip(cam_y, float(bbox_min[1]), float(bbox_max[1])))
        world_pt = np.array([centre[0], pivot_y, centre[2]], dtype=np.float64)
        screen   = self._world_to_screen(world_pt) or fallback_screen
        return screen, world_pt

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        x, y = event.position().x(), event.position().y()
        self._last_mouse_pos = (int(x), int(y))

        btn = event.button()

        # Track all held buttons by name for chord binding checks
        _BTN_NAMES = {
            Qt.MouseButton.LeftButton:   "left",
            Qt.MouseButton.MiddleButton: "middle",
            Qt.MouseButton.RightButton:  "right",
        }
        if btn in _BTN_NAMES:
            self._mouse_btns.add(_BTN_NAMES[btn])

        if btn == Qt.MouseButton.MiddleButton:
            self._mmb_held = True

        elif btn == Qt.MouseButton.LeftButton:
            self._lmb_held = True
            self._lmb_cursor_origin_global = QCursor.pos()

            # --- Rotate tool: gizmo ring hit test → start drag ---
            if self._tool == "rotate":
                axis = self._gizmo_hit_test(int(x), int(y))
                if axis is not None:
                    # Hit a ring — lock axis; rotation accumulates per-frame
                    # in mouseMoveEvent via dx/dy deltas (no snapshot needed).
                    self._gizmo_axis = axis
                return  # rotate tool owns LMB entirely (gizmo or no-op)

            # --- Polygon select: commit anchor immediately on LMB press ---
            # Anchor is placed the moment the user clicks.  If they then drag
            # to orbit the camera, the first mouseMoveEvent detects the drag and
            # pops the anchor back out so the polygon isn't distorted.
            # _poly_pending_press is still set so mouseReleaseEvent can clean up.
            if self._tool == "poly_select":
                ix, iy = int(x), int(y)
                # Close-loop check: pressing within 15px of first anchor finalises.
                if len(self._poly_points) >= 3:
                    fx, fy = self._poly_points[0]
                    if math.hypot(ix - fx, iy - fy) < 15:
                        self._poly_pending_press = None
                        self._poly_orbit_dragged = False
                        self._finalize_poly_selection()
                        return
                # Commit anchor immediately.
                self._poly_points.append((ix, iy))
                if self._camera is not None and self._painter is not None:
                    w_vp, h_vp = self.width(), self.height()
                    ndc_x = (ix / w_vp) * 2.0 - 1.0
                    ndc_y = 1.0 - (iy / h_vp) * 2.0
                    origin, direction = self._camera.unproject_ray(
                        ndc_x, ndc_y, w_vp / max(h_vp, 1)
                    )
                    model_origin, model_dir = self._to_model_ray(origin, direction)
                    hit_model = self._painter.ray_cast_world_pos(model_origin, model_dir)
                    self._poly_points_3d.append(
                        hit_model.tolist() if hit_model is not None else None
                    )
                self._poly_overlay.set_points(self._poly_points, (ix, iy))
                self._poly_overlay.update()
                self.history_changed.emit(True, self._history.can_redo)
                # Track press so the move event can roll back if orbit starts.
                self._poly_pending_press = (ix, iy)
                self._poly_orbit_dragged = False
                return  # poly select owns LMB entirely

            # --- Tool mode: LMB activates the tool on the mesh only ---
            # Region fill fires on press (single click); brush starts its first
            # stroke here and continues on drag via mouseMoveEvent.
            # Clicking in empty space does nothing — ray_cast returns None.
            if self._tool:
                # Snapshot before the stroke begins. geometry=True for sculpt
                # operations (they change vertex positions); geometry=False for
                # paint tools (texture-only changes).
                _SCULPT_TOOLS = {"inflate", "deflate", "smooth", "flatten"}
                _needs_geo = self._tool in _SCULPT_TOOLS
                if self._painter is not None:
                    self._history.push_snapshot(self._painter, geometry=_needs_geo)
                    self.history_changed.emit(
                        self._history.can_undo, self._history.can_redo
                    )
                self._handle_paint(int(x), int(y))
                return  # skip orbit cursor setup — tool owns LMB while active

            # --- Navigation mode (no tool selected): orbit cursor setup ---
            # Capture global cursor position at press — restored on release so
            # the cursor reappears exactly where the user first clicked.
            # If the click lands on the mesh, hide the system cursor to give
            # an "attached to model" feel while orbiting. Restored on release.
            if self._camera is not None and self._painter is not None:
                w, h = self.width(), self.height()
                if w > 0 and h > 0:
                    ndc_x = (int(x) / w) * 2.0 - 1.0
                    ndc_y = 1.0 - (int(y) / h) * 2.0
                    origin, direction = self._camera.unproject_ray(
                        ndc_x, ndc_y, w / h)
                    # Transform into model space before BVH intersection
                    m_origin, m_dir = self._to_model_ray(origin, direction)
                    hit_model = self._painter.ray_cast_world_pos(m_origin, m_dir)
                    # Convert model-space hit back to world space for dot placement
                    hit = self._model_pt_to_world(hit_model) if hit_model is not None else None
                    if hit is not None:
                        # Hit — cursor hides, white dot anchors to the 3D
                        # surface hit point and tracks it through orbit.
                        self.setCursor(Qt.CursorShape.BlankCursor)
                        self._dot_world_pos = hit
                        self._dot_overlay.place(int(x), int(y))
                        self._dot_overlay.show()
                    else:
                        # Miss — snap cursor to projected focal point and show
                        # the white dot on the actual model surface at the same
                        # screen height as the click.
                        #
                        # Strategy: cast a second ray through the model's centre
                        # screen column at the cursor's Y. This ray almost always
                        # hits the mesh and lands on the visible surface rather
                        # than a fictitious interior point. Falls back to the
                        # bbox-centre-at-cursor-height geometry if the centre ray
                        # also misses (e.g. cursor Y is above/below the mesh).
                        bbox_min, bbox_max = self._painter.get_bbox()
                        centre = (
                            bbox_min.astype(np.float64)
                            + bbox_max.astype(np.float64)
                        ) * 0.5
                        centre_screen = self._world_to_screen(centre)
                        dot_world: np.ndarray | None = None
                        if centre_screen is not None:
                            # Ray through (model_centre_x, cursor_y) in screen space
                            cen_ndc_x = (centre_screen[0] / w) * 2.0 - 1.0
                            cen_ndc_y = 1.0 - (int(y) / h) * 2.0
                            c_origin, c_dir = self._camera.unproject_ray(
                                cen_ndc_x, cen_ndc_y, w / h
                            )
                            c_m_origin, c_m_dir = self._to_model_ray(c_origin, c_dir)
                            _hit_m = self._painter.ray_cast_world_pos(c_m_origin, c_m_dir)
                            dot_world = (
                                self._model_pt_to_world(_hit_m)
                                if _hit_m is not None else None
                            )
                        if dot_world is None:
                            # Fallback: bbox centre XZ at the world Y that
                            # corresponds to the cursor's vertical position.
                            dx_ = float(origin[0] - centre[0])
                            dz_ = float(origin[2] - centre[2])
                            denom_ = float(direction[0] ** 2 + direction[2] ** 2)
                            if abs(denom_) > 1e-6:
                                t_ = -(
                                    dx_ * direction[0] + dz_ * direction[2]
                                ) / denom_
                                world_y = float(origin[1] + t_ * direction[1])
                            else:
                                world_y = float(centre[1])
                            world_y = float(
                                np.clip(
                                    world_y, float(bbox_min[1]), float(bbox_max[1])
                                )
                            )
                            dot_world = np.array(
                                [centre[0], world_y, centre[2]], dtype=np.float64
                            )
                        dot_screen = self._world_to_screen(dot_world)
                        if dot_screen is not None:
                            self._dot_world_pos = dot_world
                            self._dot_is_miss = True
                            self._dot_overlay.place(*dot_screen)
                            self._dot_overlay.show()
                        # Snap cursor to projected focal point (existing behaviour)
                        # Do NOT overwrite _last_mouse_pos here. Set skip flag to
                        # absorb the first synthetic move event after the warp.
                        screen = self._world_to_screen(self._camera.focal_point)
                        if screen is not None:
                            QCursor.setPos(self.mapToGlobal(QPoint(*screen)))
                        self.setCursor(Qt.CursorShape.BlankCursor)
                        self._orbit_skip_one = True

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Finalize the polygon selection on double-click (requires ≥ 3 anchors)."""
        if self._tool == "poly_select" and len(self._poly_points) >= 3:
            self._finalize_poly_selection()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x, y = int(event.position().x()), int(event.position().y())

        if self._last_mouse_pos is None:
            self._last_mouse_pos = (x, y)
            return

        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)

        if self._camera is None:
            return

        # Pan takes priority — chord bindings (e.g. LMB+RMB) must be checked
        # before single-button actions so the chord isn't swallowed first.
        if self._mouse_binding_active("pan"):
            self._camera.pan(dx, dy)
            # A pan drag while poly_select is active means the user is navigating,
            # not placing an anchor — discard the pending press to avoid a stale point.
            if self._tool == "poly_select":
                self._poly_orbit_dragged = True
            self._camera_moved = True
            self.update()

        elif self._tool == "rotate":
            if self._gizmo_axis is not None and self._lmb_held:
                # Active gizmo drag — accumulate small per-frame rotation deltas.
                # Mapping pixel delta → angle gives smooth, linear feel with no
                # atan2 discontinuity. Sensitivity: 0.010 rad/px (~0.57°/px).
                _SENS = 0.010
                axis = self._gizmo_axis
                if axis == "x":
                    small_delta = dy * _SENS
                elif axis == "y":
                    small_delta = dx * _SENS
                else:  # "z"
                    small_delta = -dx * _SENS

                c, s = math.cos(small_delta), math.sin(small_delta)
                if axis == "x":
                    R_delta = np.array(
                        [[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32
                    )
                elif axis == "y":
                    R_delta = np.array(
                        [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32
                    )
                else:  # "z"
                    R_delta = np.array(
                        [[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32
                    )

                # Compose the small incremental rotation onto the accumulated total
                self._model_rot = R_delta @ self._model_rot
                self.update()
            else:
                # No active drag — update hover highlight as cursor moves over rings
                new_hover = self._gizmo_hit_test(x, y)
                if new_hover != self._gizmo_hover:
                    self._gizmo_hover = new_hover
                    self.update()  # repaint to show/clear ring highlight

        elif self._tool == "poly_select":
            # Update the rubber-band line from the last anchor to the cursor.
            # Fires on every move regardless of button state — live preview.
            self._poly_cursor = (x, y)
            if self._poly_points:
                self._poly_overlay.set_points(self._poly_points, (x, y))
                self._poly_overlay.update()

            # Allow camera orbit while poly_select is active.  If the user
            # starts dragging immediately after a press, roll back the anchor
            # that was committed on press — orbit drags should not plant points.
            #
            # Dead-zone: require at least 4px of total travel from the press
            # position before treating movement as intentional orbit.  Without
            # this, sub-pixel mouse jitter on a click (dx/dy = ±1) was enough
            # to trigger orbit and pop the just-placed anchor, making it feel
            # like the tool was confused between clicking and dragging.
            if self._mouse_binding_active("orbit") and (dx != 0 or dy != 0):
                is_orbit = self._poly_orbit_dragged  # already confirmed drag
                if not is_orbit and self._poly_pending_press is not None:
                    px, py = self._poly_pending_press
                    travel = math.hypot(x - px, y - py)
                    is_orbit = travel >= 4.0
                if is_orbit:
                    if (self._poly_pending_press is not None
                            and not self._poly_orbit_dragged
                            and self._poly_points):
                        # First confirmed orbit move — pop the just-committed anchor.
                        self._poly_points.pop()
                        if self._poly_points_3d:
                            self._poly_points_3d.pop()
                        self._poly_overlay.set_points(
                            self._poly_points,
                            (self._poly_pending_press[0], self._poly_pending_press[1]),
                        )
                        self._poly_overlay.update()
                    self._poly_orbit_dragged = True
                    self._camera.orbit(dx, dy)
                    self._camera_moved = True
                    self.update()

        elif self._tool in ("brush", "erase") and self._lmb_held:
            # Tool active — brush/erase stroke on drag; orbit is suspended.
            # Clicking or dragging over empty space does nothing (ray_cast
            # returns None inside _handle_paint).
            self._handle_paint(x, y)

        elif not self._tool and self._mouse_binding_active("orbit"):
            # No tool selected — LMB drag orbits the camera as normal.
            # After a miss-case cursor warp, absorb the very first move event
            # as a new anchor to prevent a large snap delta.
            if self._orbit_skip_one:
                self._orbit_skip_one = False
                self.update()
                return
            if dx != 0 or dy != 0:
                self._camera.orbit(dx, dy)
                self._camera_moved = True
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        btn = event.button()

        # Remove released button from the chord-tracking set
        _BTN_NAMES = {
            Qt.MouseButton.LeftButton:   "left",
            Qt.MouseButton.MiddleButton: "middle",
            Qt.MouseButton.RightButton:  "right",
        }
        self._mouse_btns.discard(_BTN_NAMES.get(btn, ""))

        if btn == Qt.MouseButton.MiddleButton:
            self._mmb_held = False

        elif btn == Qt.MouseButton.LeftButton:
            self._lmb_held = False
            self._orbit_skip_one = False
            # Rebuild BVH after sculpt drag — deferred from mouseMoveEvent
            # because rebuilding every drag pixel is too expensive (~50ms/rebuild).
            # Rebuilding on release keeps ray_cast accurate for the next stroke.
            _SCULPT_TOOLS = {"inflate", "deflate", "smooth", "flatten"}
            if self._tool in _SCULPT_TOOLS and self._painter is not None:
                self._painter._rebuild_bvh()
            # Poly select: anchor was committed on press; just reset drag state.
            if self._tool == "poly_select":
                self._lmb_cursor_origin_global = None
                self._poly_pending_press = None
                self._poly_orbit_dragged = False
                return
            # Clear gizmo drag state — drag is complete, hover re-evaluates on next move
            if self._gizmo_axis is not None:
                self._gizmo_axis = None
                self._gizmo_screen_ctr = None
                return  # gizmo drag — skip cursor restore logic below
            self._dot_overlay.hide()
            # Determine restore target. For on-mesh hits, warp cursor to the
            # dot's current projected screen position. For off-mesh clicks
            # (_dot_is_miss) or no dot, restore to the original click position.
            if self._dot_world_pos is not None and not self._dot_is_miss:
                restore_screen = self._world_to_screen(self._dot_world_pos)
                if restore_screen is not None:
                    restore_global = self.mapToGlobal(QPoint(*restore_screen))
                else:
                    restore_global = self._lmb_cursor_origin_global
            else:
                restore_global = self._lmb_cursor_origin_global
            self._dot_world_pos = None
            self._dot_is_miss = False
            # Restore cursor via CGWarpMouseCursorPosition (macOS native) —
            # Qt's QCursor.setPos() is silently ignored during an active drag.
            self.unsetCursor()
            if restore_global is not None:
                tgt = restore_global
                QTimer.singleShot(0, lambda: _warp_cursor(tgt))
            self._lmb_cursor_origin_global = None

        self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._camera is None:
            return
        # angleDelta().y() is 120 per notch (standard Qt)
        ticks = event.angleDelta().y() / 120.0

        # Cursor-targeted zoom: always dolly toward the world point under the
        # cursor. On a mesh hit that's the surface point. On a miss we project
        # the cursor ray onto the focal plane (perpendicular to the forward
        # vector, passing through focal_point) to find a unique target at the
        # same depth as the orbit centre — giving true directional zoom even
        # when hovering over empty space beside or above/below the model.
        direction: np.ndarray | None = None
        if self._painter is not None:
            w, h = self.width(), self.height()
            if w > 0 and h > 0:
                cursor_local = self.mapFromGlobal(QCursor.pos())
                cx, cy = cursor_local.x(), cursor_local.y()
                ndc_x = (cx / w) * 2.0 - 1.0
                ndc_y = 1.0 - (cy / h) * 2.0
                origin, ray_dir = self._camera.unproject_ray(ndc_x, ndc_y, w / h)
                m_orig, m_dir = self._to_model_ray(origin, ray_dir)
                hit_m = self._painter.ray_cast_world_pos(m_orig, m_dir)
                hit = self._model_pt_to_world(hit_m) if hit_m is not None else None
                if hit is not None:
                    direction = hit - self._camera.position
                else:
                    # Project cursor ray onto the focal plane to find a unique
                    # 3D target for this cursor position at the model's depth.
                    fwd = self._camera._forward()
                    denom = float(np.dot(ray_dir, fwd))
                    if abs(denom) > 1e-6:
                        t = float(np.dot(
                            self._camera.focal_point - origin, fwd
                        )) / denom
                        t = max(0.01, t)  # always in front of camera
                        target = origin + t * ray_dir
                    else:
                        target = self._camera.focal_point
                    direction = target - self._camera.position

        self._camera.dolly(ticks, direction)
        self._camera_moved = True
        self.update()

    # ------------------------------------------------------------------ #
    # Keyboard input                                                       #
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mods = event.modifiers()

        # Track alt modifier
        if key == Qt.Key.Key_Alt:
            self._alt_held = True

        # Escape cancels an in-progress polygon selection (clears anchor points)
        if key == Qt.Key.Key_Escape:
            if self._tool == "poly_select" and self._poly_points:
                self._cancel_poly_selection()
                return

        # Frame mesh on F
        if key == Qt.Key.Key_F and self._camera is not None and self._painter is not None:
            bbox_min, bbox_max = self._painter.get_bbox()
            self._camera.frame_mesh(bbox_min, bbox_max)
            self.update()
            return

        # Cmd+Z → Undo, Cmd+Y → Redo
        ctrl = Qt.KeyboardModifier.ControlModifier
        if key == Qt.Key.Key_Z and (mods & ctrl):
            self.undo()
            return
        if key == Qt.Key.Key_Y and (mods & ctrl):
            self.redo()
            return

        self._keys_held.add(key)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = event.key()

        if key == Qt.Key.Key_Alt:
            self._alt_held = False

        self._keys_held.discard(key)

    # ------------------------------------------------------------------ #
    # WASD movement tick — ~60fps                                          #
    # ------------------------------------------------------------------ #

    def _move_tick(self) -> None:
        """Translate the camera with WASD/Q/E keys at ~60fps."""
        if self._camera is None:
            return

        now = time.monotonic()
        dt = min(now - self._last_tick_time, 0.1)
        self._last_tick_time = now

        bindings = self._camera.settings.bindings
        forward_key  = self._qt_key(bindings.get("forward",  "W"))
        backward_key = self._qt_key(bindings.get("backward", "S"))
        left_key     = self._qt_key(bindings.get("left",     "A"))
        right_key    = self._qt_key(bindings.get("right",    "D"))
        up_key       = self._qt_key(bindings.get("up",       "E"))
        down_key     = self._qt_key(bindings.get("down",     "Q"))

        target_fwd   = float((1 if forward_key  in self._keys_held else 0) -
                             (1 if backward_key in self._keys_held else 0))
        target_right = float((1 if right_key    in self._keys_held else 0) -
                             (1 if left_key     in self._keys_held else 0))
        target_up    = float((1 if up_key       in self._keys_held else 0) -
                             (1 if down_key     in self._keys_held else 0))

        tau   = 0.08
        alpha = 1.0 - math.exp(-dt / tau)
        self._vel_fwd   += (target_fwd   - self._vel_fwd)   * alpha
        self._vel_right += (target_right - self._vel_right) * alpha
        self._vel_up    += (target_up    - self._vel_up)    * alpha

        speed   = self._camera.settings.keyboard_speed * dt
        fwd_v   = self._camera._forward()
        right_v = self._camera._right()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        delta = fwd_v * self._vel_fwd + right_v * self._vel_right + world_up * self._vel_up
        mag = float(np.linalg.norm(delta))
        if mag > 1e-4:
            move = delta * (speed * min(mag, 1.0) / mag)
            self._camera.position    += move
            self._camera.focal_point += move
            self._camera_moved = True
            self.update()
        elif abs(self._vel_fwd) > 1e-4 or abs(self._vel_right) > 1e-4 or abs(self._vel_up) > 1e-4:
            self.update()

    def _mouse_binding_active(self, action_key: str) -> bool:
        """
        Check whether the mouse binding for action_key matches the current
        held-button state and modifier keys.

        Supports single-button bindings ("left") and chord bindings ("left+right").
        All required buttons in the binding must be in _mouse_btns, and all
        listed modifiers must be active.
        """
        if self._camera is None:
            return False
        binding = self._camera.settings.bindings.get(action_key)
        if not isinstance(binding, dict):
            return False
        mouse = binding.get("mouse", "")
        if mouse:
            required_btns = set(mouse.split("+"))
            if not required_btns.issubset(self._mouse_btns):
                return False
        mods = binding.get("modifiers", [])
        for m in mods:
            if m == "Alt" and not self._alt_held:
                return False
        return True

    @staticmethod
    def _qt_key(key_str: str) -> Qt.Key:
        """Convert a binding string like 'W', 'Shift', 'Space' to a Qt.Key."""
        mapping = {
            "Shift": Qt.Key.Key_Shift,
            "Ctrl": Qt.Key.Key_Control,
            "Alt": Qt.Key.Key_Alt,
            "Space": Qt.Key.Key_Space,
        }
        if key_str in mapping:
            return mapping[key_str]
        # Single letter/digit keys
        return getattr(Qt.Key, f"Key_{key_str.upper()}", Qt.Key.Key_unknown)

    # ------------------------------------------------------------------ #
    # Paint interaction                                                    #
    # ------------------------------------------------------------------ #

    def _handle_paint(self, screen_x: int, screen_y: int) -> None:
        """
        Ray-cast a screen coordinate into the mesh and paint at the hit UV.

        Converts screen (x, y) → NDC → ray using the camera's unproject_ray
        method. Delegates to MeshPainter.paint_brush() or fill_region()
        depending on the active tool. Schedules a GPU texture refresh.
        """
        if self._painter is None or self._camera is None:
            return

        # Convert screen pixel to NDC [-1, 1]
        w, h = self.width(), self.height()
        ndc_x = (2.0 * screen_x / w) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / h)  # flip Y (OpenGL convention)

        aspect = w / max(h, 1)
        ray_origin, ray_dir = self._camera.unproject_ray(ndc_x, ndc_y, aspect)

        # Transform ray into model space so it intersects the BVH correctly
        # even when the mesh has been rotated via the gizmo.
        ray_origin, ray_dir = self._to_model_ray(ray_origin, ray_dir)

        result = self._painter.ray_cast(ray_origin, ray_dir)
        if result is None:
            return  # Ray missed the mesh

        face_idx, uv = result

        _SCULPT_TOOLS = {"inflate", "deflate", "smooth", "flatten"}

        if self._tool == "brush":
            self._painter.paint_brush(
                uv, self._paint_color, self._brush_size, self._brush_opacity
            )
        elif self._tool == "erase":
            # Restore original baked texture pixels under the brush
            self._painter.erase_brush(uv, self._brush_size, self._brush_opacity)
        elif self._tool == "region":
            self._painter.fill_region(face_idx, self._paint_color)
        elif self._tool in _SCULPT_TOOLS:
            # Sculpt hit point: the locations array from ray_cast gives us the
            # 3D model-space position. We use ray_cast_world_pos() to get it
            # (second BVH traversal, but sculpt calls are infrequent on drag).
            hit_pos = self._painter.ray_cast_world_pos(ray_origin, ray_dir)
            if hit_pos is not None:
                self._painter.sculpt(
                    hit_pos, self._tool,
                    self._sculpt_radius, self._sculpt_strength,
                )
                # Flag VBO for re-upload in the next paintGL call.
                # We don't call _upload_mesh_only() here because we're not
                # guaranteed to be inside the OpenGL context during mouse events.
                self._needs_mesh_upload = True

        self.update()  # triggers paintGL → _refresh_dirty_texture / VBO re-upload

    # ------------------------------------------------------------------ #
    # Polygon selection helpers                                            #
    # ------------------------------------------------------------------ #

    def _cancel_poly_selection(self) -> None:
        """
        Cancel the in-progress polygon selection, discarding all anchor points.

        Called when Escape is pressed while placing anchors, or when the tool
        is switched away from poly_select.
        """
        self._poly_points.clear()
        self._poly_points_3d.clear()
        self._poly_pending_press = None
        self._poly_orbit_dragged = False
        self._poly_overlay.set_points([], (0, 0))
        self._poly_overlay.update()
        # Re-sync undo button state — no anchors left, so reflect actual history.
        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

    def _reproject_poly_overlay(self, cursor: "tuple[int, int]") -> None:
        """
        Reproject world-space anchor positions through the current camera MVP.

        Called every frame from paintGL() when an in-progress polygon selection
        exists and the camera may have moved (orbit, pan, fly). Converts each
        entry in _poly_points_3d back to screen pixels via the live MVP and
        overwrites _poly_points so the overlay widget always draws the lasso
        anchors exactly on the mesh surface, not drifting in screen space.

        Only active when _poly_points_3d has entries — skips immediately if
        there are no in-progress anchors or the camera/context is not ready.
        """
        if not self._poly_points_3d or self._camera is None:
            return

        w, h = self.width(), self.height()
        if w < 1 or h < 1:
            return

        # Build the MVP matrix for the current frame so world-space anchor
        # positions can be transformed to clip space → NDC → screen pixels.
        aspect = w / h
        view   = self._camera.get_view_matrix()
        proj   = self._camera.get_projection_matrix(aspect)
        model  = self._model_mat_4x4()
        mvp    = (proj @ view @ model.astype(np.float64)).astype(np.float32)

        # Separate valid (mesh-hit) from None (off-mesh) entries.
        valid_indices = [i for i, p in enumerate(self._poly_points_3d) if p is not None]
        none_indices  = [i for i, p in enumerate(self._poly_points_3d) if p is None]

        if not valid_indices:
            # No mesh-hit anchors at all — nothing to reproject.
            return

        # Project the valid (mesh-hit) anchor positions.
        valid_3d = [self._poly_points_3d[i] for i in valid_indices]
        pts_w = np.array(valid_3d, dtype=np.float32)
        pts4  = np.hstack([pts_w, np.ones((len(pts_w), 1), np.float32)])
        clip  = (mvp @ pts4.T).T

        wc = np.where(np.abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])
        sx = ((clip[:, 0] / wc + 1.0) * 0.5 * w).astype(int)
        sy = ((1.0 - (clip[:, 1] / wc + 1.0) * 0.5) * h).astype(int)

        for k, i in enumerate(valid_indices):
            self._poly_points[i] = (int(sx[k]), int(sy[k]))

        # Estimate 3D positions for off-mesh (None) anchors so they also
        # reproject correctly when the camera moves, instead of staying frozen
        # at their original screen position while on-mesh anchors shift.
        # Use the average NDC z of valid hits as a stand-in depth — identical
        # to the approach in _finalize_poly_selection. Store the estimate back
        # into _poly_points_3d so subsequent reprojections are consistent.
        if none_indices:
            # Average NDC z from valid mesh hits.
            avg_ndc_z = float(np.mean(clip[:, 2] / wc))
            inv_mvp   = np.linalg.inv(mvp.astype(np.float64))
            for i in none_indices:
                px_i, py_i = self._poly_points[i]
                ndc_x_i = (px_i / w) * 2.0 - 1.0
                ndc_y_i = 1.0 - (py_i / h) * 2.0
                ndc_pt  = np.array([ndc_x_i, ndc_y_i, avg_ndc_z, 1.0])
                world_h = inv_mvp @ ndc_pt
                if abs(world_h[3]) > 1e-8:
                    est_3d = (world_h[:3] / world_h[3]).tolist()
                    self._poly_points_3d[i] = est_3d
                    # Project estimated position back to screen.
                    est_h = np.array([*est_3d, 1.0], dtype=np.float64)
                    ec    = mvp.astype(np.float64) @ est_h
                    ew    = ec[3] if abs(ec[3]) > 1e-8 else 1e-8
                    self._poly_points[i] = (
                        int((ec[0] / ew + 1.0) * 0.5 * w),
                        int((1.0 - (ec[1] / ew + 1.0) * 0.5) * h),
                    )

        # Use the last reprojected anchor as the rubber-band cursor endpoint.
        # _poly_cursor is stale during WASD (mouse doesn't move), so pointing the
        # rubber-band at the cursor would shoot a line off into the distance.
        # The rubber-band reappears at the correct position on the next mouse move.
        draw_cursor = self._poly_points[-1] if self._poly_points else cursor
        self._poly_overlay.set_points(self._poly_points, draw_cursor)

    def _reproject_pending_mask(self) -> None:
        """
        Rebuild the pending-selection screen mask from the stored 3D anchor points.

        Called from paintGL() whenever the camera moves and a finalized pending
        selection exists.  The teal overlay is a screen-space texture frozen at
        selection time — without reprojection it floats in screen space while the
        mesh moves under it during orbit.  This method reprojects the 3D anchor
        polygon through the current MVP and re-rasterizes the PIL mask so the
        overlay stays glued to the mesh surface.

        Also updates _pending_poly_pts_2d so that delete_layer_faces always uses
        the most recent camera state rather than the original frozen view.
        """
        if (self._pending_poly_pts_3d is None
                or self._camera is None
                or self._ctx is None):
            return

        w, h = self.width(), self.height()
        if w < 1 or h < 1:
            return

        # Reproject 3D anchor points through the current MVP.
        aspect = w / h
        view   = self._camera.get_view_matrix()
        proj   = self._camera.get_projection_matrix(aspect)
        model  = self._model_mat_4x4()
        mvp    = (proj @ view @ model.astype(np.float64)).astype(np.float32)

        pts_w = np.array([p for p in self._pending_poly_pts_3d if p is not None],
                         dtype=np.float32)
        if len(pts_w) < 3:
            return

        pts4 = np.hstack([pts_w, np.ones((len(pts_w), 1), np.float32)])
        clip  = (mvp @ pts4.T).T
        wc    = np.where(np.abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])
        sx    = ((clip[:, 0] / wc + 1.0) * 0.5 * w)
        sy    = ((1.0 - (clip[:, 1] / wc + 1.0) * 0.5) * h)

        new_pts_2d = [(float(sx[k]), float(sy[k])) for k in range(len(pts_w))]

        # Rasterise the reprojected polygon into a new mask texture.
        from PIL import Image as _PilImg, ImageDraw as _PilDraw  # noqa: PLC0415
        mask_pil = _PilImg.new("L", (w, h), 0)
        _PilDraw.Draw(mask_pil).polygon(new_pts_2d, fill=255)
        mask_data = np.array(mask_pil, dtype=np.uint8)

        # Called from inside paintGL() — the GL context is already current.
        # Do NOT call makeCurrent()/doneCurrent() here: re-entrant context
        # activation is undefined on macOS and causes a SIGSEGV in the driver.
        try:
            if self._pending_screen_mask_tex is not None:
                try:
                    self._pending_screen_mask_tex.release()
                except Exception:
                    pass
                self._pending_screen_mask_tex = None
            tex = self._ctx.texture((w, h), 1, mask_data.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._pending_screen_mask_tex = tex
            # Keep _pending_poly_pts_2d in sync with the reprojected polygon so
            # delete_layer_faces uses the current camera view, not the frozen one.
            self._pending_poly_pts_2d  = new_pts_2d
            self._pending_view_mat     = view.astype(np.float64).copy()
            self._pending_proj_mat     = proj.astype(np.float64).copy()
            self._pending_model_mat    = model.astype(np.float64).copy()
        except Exception:
            log.exception("ViewportWidget: _reproject_pending_mask failed")

    def _finalize_poly_selection(self) -> None:
        """
        Close the polygon and determine exactly which mesh faces fall inside it.

        Uses a pure CPU/numpy approach: projects every mesh vertex through the
        current MVP matrix to screen space, then tests each face's vertices and
        centroid against a PIL-rasterized polygon mask.

        This shares the exact same coordinate space as the drawn polygon — no
        OpenGL FBO, no Retina/DPR mismatch, no pixel-readback ambiguity. The
        face-ID FBO approach was removed because it consistently selected wrong
        faces due to viewport/DPR coordinate mismatches that could not be reliably
        corrected on all display configurations.

        No-op if fewer than 3 anchor points are placed, or if no mesh is loaded.
        """
        if len(self._poly_points) < 3 or self._painter is None or self._camera is None:
            return

        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return

        # ------------------------------------------------------------------ #
        # 1.  Rasterize the drawn polygon into a 2-D boolean mask             #
        # ------------------------------------------------------------------ #
        # PIL ImageDraw.polygon fills the exact interior of the user's polygon
        # in pixel-perfect screen space. The mask is used both for face selection
        # (steps 2-3) and for the GPU texture that drives the teal overlay display.
        from PIL import Image as _PilImg, ImageDraw as _PilDraw  # noqa: PLC0415
        mask_pil = _PilImg.new("L", (w, h), 0)
        _PilDraw.Draw(mask_pil).polygon(
            [(px, py) for px, py in self._poly_points], fill=255
        )
        mask = np.array(mask_pil, dtype=np.uint8) > 0  # bool (H, W)

        # ------------------------------------------------------------------ #
        # 2.  Project every mesh vertex to screen space via MVP               #
        # ------------------------------------------------------------------ #
        # Compute the same MVP used by paintGL so the projection is identical
        # to what the user sees on screen.
        aspect = w / h
        view   = self._camera.get_view_matrix()
        proj   = self._camera.get_projection_matrix(aspect)
        model  = self._model_mat_4x4()
        mvp    = (proj @ view @ model.astype(np.float64)).astype(np.float64)

        # Snapshot the 2D polygon and camera state for cookie-cutter deletion.
        # These are stored on the pending selection and transferred to the layer
        # in save_pending_as_layer so that delete_layer_faces can clip boundary
        # triangles exactly along the polygon outline at deletion time.
        self._pending_poly_pts_2d = list(self._poly_points)
        self._pending_view_mat    = view.astype(np.float64).copy()
        self._pending_proj_mat    = proj.astype(np.float64).copy()
        self._pending_model_mat   = model.astype(np.float64).copy()

        faces_arr = np.asarray(self._painter._mesh.faces, dtype=np.int32)   # (M, 3)
        verts_arr = np.asarray(self._painter._mesh.vertices, dtype=np.float32)  # (N, 3)
        n_faces = len(faces_arr)
        n_verts = len(verts_arr)

        # Homogeneous coords (N, 4), multiply through MVP in one batched op.
        verts_h = np.hstack([verts_arr.astype(np.float64),
                              np.ones((n_verts, 1), np.float64)])  # (N, 4)
        clip    = (mvp @ verts_h.T).T                              # (N, 4) clip space
        # Perspective divide → NDC [-1, 1]
        wc   = clip[:, 3:4]
        wc   = np.where(np.abs(wc) < 1e-8, 1e-8, wc)
        ndc  = clip[:, :3] / wc                                    # (N, 3)
        # NDC → screen pixels with top-left origin (matching PIL/Qt convention).
        # Y is flipped because NDC +1 = top of screen = y=0 in screen coords.
        sx = ((ndc[:, 0] + 1.0) * 0.5 * w)    # (N,) float x
        sy = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * h)  # (N,) float y
        # Integer pixel indices clamped to image bounds for mask lookup.
        isx = np.clip(sx.astype(np.int32), 0, w - 1)  # (N,)
        isy = np.clip(sy.astype(np.int32), 0, h - 1)  # (N,)

        # ------------------------------------------------------------------ #
        # 3.  Test each face against the polygon mask                         #
        # ------------------------------------------------------------------ #
        # A face is selected if ANY of its 3 vertices OR its centroid falls
        # inside the drawn polygon.  The centroid test catches faces whose
        # vertices are all just outside the polygon boundary but whose interior
        # is clearly inside it — common for large triangles at shallow angles.
        vi0, vi1, vi2 = faces_arr[:, 0], faces_arr[:, 1], faces_arr[:, 2]

        # Test face centroid against the polygon mask.
        # Centroid = average of the three projected vertex screen positions.
        # Using the centroid means "this face is mostly inside the polygon" —
        # it spans the full selection area without over-selecting partial
        # boundary faces, and avoids the strip/cluster artifacts that "all 3
        # vertices inside" produces at the polygon edges.
        cx_f = (sx[vi0] + sx[vi1] + sx[vi2]) / 3.0   # float centroid x (N,)
        cy_f = (sy[vi0] + sy[vi1] + sy[vi2]) / 3.0   # float centroid y (N,)
        cx_i = np.clip(cx_f.astype(np.int32), 0, w - 1)
        cy_i = np.clip(cy_f.astype(np.int32), 0, h - 1)
        face_inside = mask[cy_i, cx_i]                 # (M,) bool

        # Centroid-only test: no front-face culling.
        # Front-face culling was tried but Apple Object Capture OBJ files use
        # CW winding, so trimesh face_normals are inverted relative to the camera
        # convention. The culling was selecting the wrong faces (discarding ~86%
        # of the correct front-visible faces). Centroid-only is clean enough on
        # its own — back faces of a closed mesh project to different screen regions.
        selected: set[int] = set(int(i) for i in np.where(face_inside)[0])

        # ------------------------------------------------------------------ #
        # 4.  Average face normal for back-face culling of committed layers   #
        # ------------------------------------------------------------------ #
        # Committed layer rendering dot-tests this normal against the camera
        # direction each frame and skips drawing the highlight when the layer
        # is facing away from the viewer.
        # avg_normal-based back-face culling disabled: Apple Object Capture OBJ
        # files use CW winding, so trimesh face_normals are inverted. Computing
        # avg_normal from those inverted normals caused committed layers to always
        # be culled (dot test always fired). Setting None disables the cull —
        # layers render from all camera angles, which is the correct behaviour.
        self._pending_poly_avg_normal = None

        # ------------------------------------------------------------------ #
        # 5.  3D anchor positions for committed-layer reprojection            #
        # ------------------------------------------------------------------ #
        # _poly_points_3d is populated at click time via raycasting in
        # mousePressEvent. Use those model-space positions for per-frame
        # reprojection that keeps the pending overlay glued to the mesh surface.
        #
        # For anchor points that missed the mesh (None entries), estimate a 3D
        # position by unprojecting the 2D screen point at the average depth of
        # the valid hits. This ensures _pending_poly_pts_3d is always populated
        # (as long as at least 1 anchor hit the mesh), so _reproject_pending_mask
        # always runs during orbit and the overlay tracks the model correctly.
        valid_3d = [p for p in self._poly_points_3d if p is not None]
        if valid_3d:
            # Compute average model-space depth of valid hits to use as stand-in
            # depth for off-mesh anchor points.
            inv_mvp = np.linalg.inv(mvp)
            avg_valid = np.mean(valid_3d, axis=0).astype(np.float64)
            avg_h     = np.append(avg_valid, 1.0)
            avg_clip  = mvp @ avg_h
            avg_ndc_z = avg_clip[2] / avg_clip[3] if abs(avg_clip[3]) > 1e-8 else 0.0

            full_3d: list = []
            for i, p3d in enumerate(self._poly_points_3d):
                if p3d is not None:
                    full_3d.append(p3d)
                else:
                    # Unproject the 2D screen position at the average NDC depth.
                    sx_i, sy_i = self._poly_points[i] if i < len(self._poly_points) \
                                 else self._poly_points[-1]
                    ndc_x_i = (sx_i / w) * 2.0 - 1.0
                    ndc_y_i = 1.0 - (sy_i / h) * 2.0
                    ndc_pt  = np.array([ndc_x_i, ndc_y_i, avg_ndc_z, 1.0])
                    world_h = inv_mvp @ ndc_pt
                    if abs(world_h[3]) > 1e-8:
                        full_3d.append((world_h[:3] / world_h[3]).tolist())
                    else:
                        full_3d.append(avg_valid.tolist())
            self._pending_poly_pts_3d = full_3d if len(full_3d) >= 3 else None
        else:
            self._pending_poly_pts_3d = None

        # ------------------------------------------------------------------ #
        # 6.  Upload polygon mask as GPU texture for the screen-space overlay #
        # ------------------------------------------------------------------ #
        # The SCREEN_OVERLAY_FRAG shader samples this single-channel texture
        # alongside the scene depth texture to composite the teal selection
        # highlight — pixel-perfect, no UV atlas, no triangle-edge jagging.
        try:
            self.makeCurrent()
            mask_data = np.array(mask_pil, dtype=np.uint8)
            if self._pending_screen_mask_tex is not None:
                try:
                    self._pending_screen_mask_tex.release()
                except Exception:
                    pass
                self._pending_screen_mask_tex = None
            pending_mask = self._ctx.texture((w, h), 1, mask_data.tobytes())
            pending_mask.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._pending_screen_mask_tex = pending_mask
            log.debug(
                "ViewportWidget: screen mask uploaded (%dx%d, %d selected faces)",
                w, h, len(selected),
            )
        except Exception:
            log.exception("ViewportWidget: screen mask upload failed")
            self._pending_screen_mask_tex = None
        finally:
            self.doneCurrent()

        # ------------------------------------------------------------------ #
        # 7.  Commit or discard result                                        #
        # ------------------------------------------------------------------ #
        # Clear anchor points and overlay regardless of whether any faces were
        # selected — the polygon is consumed either way. Also clear the deferred
        # press state so a Release2 from a double-click doesn't plant a stale anchor.
        self._poly_points.clear()
        self._poly_points_3d.clear()
        self._poly_pending_press = None
        self._poly_orbit_dragged = False
        self._poly_overlay.set_points([], (0, 0))
        self._poly_overlay.update()
        # Re-sync undo button state with actual geometry history.
        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

        # Always proceed when cookie-cutter data is available, even if no face
        # centroids landed inside the polygon (small polygon that fits inside one
        # mesh triangle). The cookie-cutter uses the stored polygon + matrices
        # directly, so _pending_faces being empty is fine on that path.
        # Emit at least 1 so the Save button enables in the UI.
        self._pending_faces = selected
        self.update()
        emit_count = len(selected) if selected else (1 if self._pending_poly_pts_2d else 0)
        if emit_count == 0:
            return
        self.selection_ready.emit(emit_count)

    def _build_face_vao(
        self, face_indices: "set[int]"
    ) -> "tuple[moderngl.VertexArray, moderngl.Buffer]":
        """
        Build a flat-position VAO from a set of face indices.

        Extracts the 3D vertex positions for each selected face into a flat
        (K*3, 3) array and uploads to a moderngl Buffer. Returns (vao, vbo)
        bound to _hl_prog so the selection can be drawn with the highlight shader.

        Must be called with the GL context current (inside paintGL or after
        makeCurrent()).
        """
        faces_arr = np.asarray(self._painter._mesh.faces)          # (M, 3)
        verts_arr = np.asarray(self._painter._mesh.vertices, dtype=np.float32)  # (N, 3)

        sel       = np.array(sorted(face_indices), dtype=np.int64)
        tri_verts = verts_arr[faces_arr[sel].reshape(-1)]           # (K*3, 3)

        vbo = self._ctx.buffer(tri_verts.tobytes())
        vao = self._ctx.vertex_array(
            self._hl_prog, [(vbo, "3f", "in_position")]
        )
        return vao, vbo

    def _release_pending_vao(self) -> None:
        """Release the pending selection GPU resources (VAO, VBO, and screen mask)."""
        for attr in ("_pending_vao", "_pending_vbo", "_pending_screen_mask_tex"):
            resource = getattr(self, attr, None)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
            setattr(self, attr, None)

    def _release_layer_vaos(self) -> None:
        """Release any legacy GPU resources stored in layer dicts."""
        for layer in self._layers.values():
            # Release any legacy VAO/texture resources (face_vao, screen_mask_tex etc.)
            # that may exist from prior code versions. Current layers only store
            # poly_pts_3d (plain Python list) which needs no GPU release.
            for key in ("face_vao", "face_vbo", "screen_mask_tex", "vao", "vbo", "uv_mask_tex"):
                resource = layer.get(key)
                if resource is not None:
                    try:
                        resource.release()
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    # Public API — polygon selection layers                                #
    # ------------------------------------------------------------------ #

    def get_face_count(self) -> int:
        """Return the total face count of the loaded mesh, or 0."""
        if self._painter is None:
            return 0
        return len(self._painter._mesh.faces)

    def save_pending_as_layer(
        self, name: str, color: "tuple[float, float, float, float]"
    ) -> int:
        """
        Promote the pending polygon selection to a named, persistent layer.

        Builds a highlight VAO from the pending face set, assigns a unique
        layer_id, and stores it in _layers so it renders each frame.
        Clears the pending selection highlight.

        Parameters
        ----------
        name  : Human-readable label for the layer row.
        color : RGBA color (0.0–1.0 each) for the layer highlight overlay.

        Returns the new layer_id, or -1 if there is no pending selection or
        the GL context is unavailable.
        """
        # Allow saving even when _pending_faces is empty: the cookie-cutter path
        # works from the stored polygon + camera matrices and doesn't require
        # face centroids to have been detected inside the polygon.
        has_cookie = self._pending_poly_pts_2d is not None
        if (not self._pending_faces and not has_cookie) or self._ctx is None or self._painter is None:
            return -1

        layer_id = self._next_layer_id
        self._next_layer_id += 1

        # Transfer the 3D anchor points and face normal from the pending selection.
        # poly_pts_3d → reprojected through current camera each frame
        # avg_normal  → used to cull the layer when facing away from camera
        poly_pts_3d = self._pending_poly_pts_3d
        self._pending_poly_pts_3d = None
        avg_normal = self._pending_poly_avg_normal
        self._pending_poly_avg_normal = None

        # Transfer the cookie-cutter data: the original 2D polygon and the
        # exact camera matrices at the moment of selection. delete_layer_faces
        # uses these to clip boundary triangles along the polygon outline.
        poly_pts_2d = self._pending_poly_pts_2d
        self._pending_poly_pts_2d = None
        view_mat  = self._pending_view_mat;  self._pending_view_mat  = None
        proj_mat  = self._pending_proj_mat;  self._pending_proj_mat  = None
        model_mat = self._pending_model_mat; self._pending_model_mat = None

        self._layers[layer_id] = {
            "name":         name,
            "faces":        set(self._pending_faces),  # kept for future painting ops
            "visible":      True,
            "color":        color,
            "poly_pts_3d":  poly_pts_3d,   # 3D anchors → reproject each frame
            "avg_normal":   avg_normal,    # world-space face normal for culling
            # Cookie-cutter data — exact polygon + camera state at selection time
            "poly_pts_2d":  poly_pts_2d,
            "view_mat":     view_mat,
            "proj_mat":     proj_mat,
            "model_mat":    model_mat,
        }

        # Discard the pending teal highlight now that it's been promoted
        self._pending_faces = None
        self._release_pending_vao()
        self.update()

        return layer_id

    def set_active_layer(self, layer_id: int) -> None:
        """
        Mark a layer as the active selection (chosen in the layers panel).

        The active layer renders with boosted alpha so it stands out visually
        from other saved layers. Clears the previous active layer's boost.

        Called by ViewportView when the layers panel emits layer_selected.
        """
        self._active_layer_id = layer_id
        self.update()

    def set_layer_color(self, layer_id: int, color: "tuple[float, float, float, float]") -> None:
        """Update the RGBA highlight color of a saved layer in real-time."""
        if layer_id in self._layers:
            self._layers[layer_id]["color"] = color
            self.update()

    def set_layer_visibility(self, layer_id: int, visible: bool) -> None:
        """Toggle a saved layer's highlight overlay and projected texture on or off."""
        if layer_id in self._layers:
            self._layers[layer_id]["visible"] = visible
        # Toggle shader projection layer visibility (no GPU teardown needed —
        # the paintGL pass simply skips layers where visible=False).
        if layer_id in self._proj_layers:
            self._proj_layers[layer_id]["visible"] = visible
        # Also sync painter visibility state for undo/redo consistency.
        if self._painter is not None:
            self._painter.set_layer_texture_visible(layer_id, visible)
        self.update()

    def delete_layer(self, layer_id: int) -> None:
        """Remove a saved layer and release all associated GPU resources."""
        if layer_id not in self._layers:
            return
        layer = self._layers.pop(layer_id)
        if self._ctx is not None:
            self.makeCurrent()
            # Release highlight/selection GPU resources
            for key in ("face_vao", "face_vbo", "screen_mask_tex", "vao", "vbo", "uv_mask_tex"):
                resource = layer.get(key)
                if resource is not None:
                    try:
                        resource.release()
                    except Exception:
                        pass
            # Release projection GPU resources (texture + IBO + VAO)
            pdata = self._proj_layers.pop(layer_id, None)
            if pdata is not None:
                for key in ("gpu_tex", "ibo", "vao"):
                    res = pdata.get(key)
                    if res is not None:
                        try:
                            res.release()
                        except Exception:
                            pass
            self.doneCurrent()
        if self._painter is not None:
            self._painter.remove_layer_texture(layer_id)
        self.update()

    def _read_scene_depth_buffer(self) -> "np.ndarray | None":
        """
        Read self._scene_depth_tex from GPU to CPU.

        Returns a float32 numpy array of shape (h, w) with GL depth values in
        [0, 1] (0 = near plane, 1 = far plane), in OpenGL convention (row 0 is
        the bottom of the screen).  Returns None if the texture is unavailable
        or the readback fails.

        Safe to call from outside paintGL — uses makeCurrent()/doneCurrent().
        """
        if self._scene_depth_tex is None:
            return None
        try:
            self.makeCurrent()
            raw   = self._scene_depth_tex.read()
            tex_w, tex_h = self._scene_depth_tex.size          # moderngl: (w, h)
            return np.frombuffer(raw, dtype=np.float32).reshape(tex_h, tex_w)
        except Exception:
            log.exception("ViewportWidget: scene depth buffer readback failed")
            return None
        finally:
            self.doneCurrent()

    def _cookie_cutter_clip(
        self,
        poly_pts_2d: list,
        view_mat: np.ndarray,
        proj_mat: np.ndarray,
        model_mat: np.ndarray,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray | None]":
        """
        Clip the mesh against a screen-space polygon using the Sutherland-Hodgman
        / Shapely difference approach (cookie-cutter deletion).

        For each triangle in the mesh:
          - Fully inside the polygon  → deleted (not included in output)
          - Fully outside the polygon → kept unchanged
          - Straddling the boundary   → the outside fragment is kept; new
            vertices are added at the exact screen-space crossing points,
            with 3D positions and UV coordinates interpolated via barycentric
            coords on the original triangle

        The result is a hole whose boundary follows the drawn polygon exactly,
        not the jagged triangle-edge approximation produced by whole-face deletion.

        Parameters
        ----------
        poly_pts_2d : list of (x, y) floats — screen-space polygon (same coords
                      as _poly_points at selection time)
        view_mat    : 4×4 float64 view matrix at selection time
        proj_mat    : 4×4 float64 projection matrix at selection time
        model_mat   : 4×4 float64 model matrix at selection time

        Returns
        -------
        (new_vertices, new_faces, new_uvs)  — ready to drop into a new Trimesh.
        new_uvs is None if the painter has no UV data.
        """
        from shapely.geometry import Polygon as _SPoly  # noqa: PLC0415

        mesh      = self._painter._mesh
        old_verts = np.asarray(mesh.vertices,  dtype=np.float64)   # (V, 3)
        old_faces = np.asarray(mesh.faces,     dtype=np.int64)     # (F, 3)
        old_uvs   = self._painter._uvs                              # (V, 2) | None
        n_verts   = len(old_verts)
        n_faces   = len(old_faces)

        w = float(self.width())
        h = float(self.height())

        # ------------------------------------------------------------------ #
        # 1.  Project all vertices to screen space using stored matrices      #
        # ------------------------------------------------------------------ #
        mvp    = (proj_mat @ view_mat @ model_mat)
        verts_h = np.hstack([old_verts,
                              np.ones((n_verts, 1), dtype=np.float64)])  # (V,4)
        clip   = (mvp @ verts_h.T).T                                      # (V,4)
        wc     = np.where(np.abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])
        ndc    = clip[:, :3] / wc[:, np.newaxis]
        sx     = (ndc[:, 0] + 1.0) * 0.5 * w                             # (V,)
        sy     = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * h                    # (V,)

        # ------------------------------------------------------------------ #
        # 2.  Build Shapely clip polygon (the user's drawn polygon)           #
        # ------------------------------------------------------------------ #
        clip_poly = _SPoly(poly_pts_2d)
        if not clip_poly.is_valid:
            clip_poly = clip_poly.buffer(0)  # repair self-intersections

        # ------------------------------------------------------------------ #
        # 3.  Pre-filter: fast numpy classification for fully-inside /        #
        #     fully-outside faces to avoid per-face Shapely overhead on       #
        #     Ultra-density meshes (400 K faces).                             #
        # ------------------------------------------------------------------ #
        # Rasterise the polygon into a pixel mask for the fast test
        from PIL import Image as _PI, ImageDraw as _PID  # noqa: PLC0415
        mask_img = _PI.new("L", (int(w), int(h)), 0)
        _PID.Draw(mask_img).polygon([(px, py) for px, py in poly_pts_2d],
                                    fill=255)
        mask = np.array(mask_img, dtype=np.uint8) > 0  # bool (H, W)

        vi0, vi1, vi2 = old_faces[:, 0], old_faces[:, 1], old_faces[:, 2]
        isx = np.clip(sx.astype(np.int32), 0, int(w) - 1)
        isy = np.clip(sy.astype(np.int32), 0, int(h) - 1)

        in_v0 = mask[isy[vi0], isx[vi0]]
        in_v1 = mask[isy[vi1], isx[vi1]]
        in_v2 = mask[isy[vi2], isx[vi2]]

        # ------------------------------------------------------------------ #
        # Screen-space winding test for front-face culling.                   #
        # We avoid trimesh face_normals here because Apple Object Capture OBJ #
        # files use CW winding, making those normals appear inverted.         #
        #                                                                      #
        # Instead, compute the 2-D signed area of each projected triangle:    #
        #   signed_area = (v1−v0) × (v2−v0)  (2-D cross product)             #
        # In Y-down screen space, CW winding → signed_area < 0 → front-face  #
        # (matches OpenGL's CCW = front-face convention in Y-up NDC).         #
        # ------------------------------------------------------------------ #
        p0x, p0y = sx[vi0], sy[vi0]
        p1x, p1y = sx[vi1], sy[vi1]
        p2x, p2y = sx[vi2], sy[vi2]
        signed_area   = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
        front_facing  = signed_area < 0   # (F,) bool — CW in Y-down = front

        # ------------------------------------------------------------------ #
        # Centroid pixel coords (needed for mask lookup and depth lookup)     #
        # ------------------------------------------------------------------ #
        cx_pix = np.clip((isx[vi0] + isx[vi1] + isx[vi2]) // 3, 0, int(w) - 1)
        cy_pix = np.clip((isy[vi0] + isy[vi1] + isy[vi2]) // 3, 0, int(h) - 1)
        centroid_in_mask = mask[cy_pix, cx_pix]                        # (F,) bool

        # ------------------------------------------------------------------ #
        # GPU depth buffer visibility test                                     #
        #                                                                      #
        # Read back the scene depth texture (rendered by the last paintGL).   #
        # Convert each face's centroid from NDC depth to GL [0,1] and compare #
        # against the buffer value at that pixel.  A face whose GL depth      #
        # exceeds the buffer is occluded (behind the visible surface) and     #
        # should be kept.  This replaces all heuristic depth-margin math.     #
        #                                                                      #
        # Winding test (front_facing) is still used as a first-pass filter    #
        # because it is fast and eliminates obviously back-facing geometry     #
        # before the more expensive buffer lookup.                             #
        # ------------------------------------------------------------------ #
        # NDC depth for each face centroid, converted to GL [0, 1] range.
        face_ndc_z  = (ndc[vi0, 2] + ndc[vi1, 2] + ndc[vi2, 2]) / 3.0  # (F,)
        face_gl_dep = (face_ndc_z + 1.0) * 0.5                           # (F,)

        depth_buf = self._read_scene_depth_buffer()
        if depth_buf is not None:
            tex_h, tex_w = depth_buf.shape
            vp_w, vp_h   = int(w), int(h)
            # Scene depth texture uses OpenGL convention: row 0 = bottom of screen.
            # cx_pix/cy_pix are screen-space (top-left origin). Flip Y for lookup.
            if tex_w == vp_w and tex_h == vp_h:
                gl_row = (tex_h - 1) - cy_pix
                gl_col = cx_pix
            else:
                # Rare: viewport resized between last paintGL and delete call.
                gl_col = np.clip(cx_pix * tex_w  // vp_w, 0, tex_w  - 1)
                gl_row = np.clip((vp_h - 1 - cy_pix) * tex_h // vp_h, 0, tex_h - 1)
            buf_depth    = depth_buf[gl_row, gl_col]                    # (F,) float32
            # Small epsilon (~0.1%) handles surface roughness and floating-point
            # precision at the boundary. A face at exactly the visible surface
            # depth must pass; only truly occluded faces (behind it) are blocked.
            _EPS          = 0.001
            depth_visible = face_gl_dep <= (buf_depth + _EPS)           # (F,) bool
        else:
            # Depth texture unavailable — fall back to accepting all front-facing
            # faces. Conservative (may pass some back faces) but never silently
            # drops valid geometry. Matches pre-depth-readback behaviour.
            log.warning("ViewportWidget: depth buffer unavailable; skipping depth gate")
            depth_visible = np.ones(n_faces, dtype=bool)

        # ------------------------------------------------------------------ #
        # Face classification                                                  #
        # ------------------------------------------------------------------ #
        all_in  = in_v0 & in_v1 & in_v2
        all_out = ~in_v0 & ~in_v1 & ~in_v2
        mixed   = ~all_in & ~all_out                                    # some in, some out

        # Fully-inside: all verts inside polygon, front-facing, depth-visible
        fully_inside_mask = all_in & front_facing & depth_visible

        # Straddling boundary faces: no depth gate — boundary triangles can be
        # slightly occluded at the polygon edge due to surface roughness, and we
        # always want to Shapely-clip them to preserve the hole shape exactly.
        straddling_mask   = mixed & front_facing & depth_visible

        # Edge case: drawn polygon is entirely INSIDE a single mesh triangle.
        # Case 1: face centroid inside the polygon mask (polygon > face).
        centroid_inside = all_out & centroid_in_mask & front_facing & depth_visible

        # Case 2: a clip polygon vertex falls inside a mesh face (polygon < face —
        # tip/corner of drawn shape fits inside one triangle).
        clip_vert_in_face = np.zeros(n_faces, dtype=bool)
        candidate_ao = all_out & front_facing & depth_visible
        if candidate_ao.any():
            ax, ay = sx[vi0][candidate_ao], sy[vi0][candidate_ao]
            bx, by = sx[vi1][candidate_ao], sy[vi1][candidate_ao]
            cx, cy = sx[vi2][candidate_ao], sy[vi2][candidate_ao]
            for cpx, cpy in poly_pts_2d:
                d0 = (cpx - bx) * (ay - by) - (ax - bx) * (cpy - by)
                d1 = (cpx - cx) * (by - cy) - (bx - cx) * (cpy - cy)
                d2 = (cpx - ax) * (cy - ay) - (cx - ax) * (cpy - ay)
                inside = ~((d0 < 0) | (d1 < 0) | (d2 < 0)) | \
                         ~((d0 > 0) | (d1 > 0) | (d2 > 0))
                clip_vert_in_face[candidate_ao] |= inside

        polygon_inside_face = centroid_inside | (clip_vert_in_face & all_out)
        straddling_mask     = straddling_mask | polygon_inside_face

        # Everything else is kept unchanged
        fully_outside_mask = ~fully_inside_mask & ~straddling_mask

        # ------------------------------------------------------------------ #
        # 4.  Accumulate output geometry                                      #
        # ------------------------------------------------------------------ #
        # Start with original vertex arrays; new boundary vertices are appended.
        out_verts = list(old_verts)
        out_uvs   = list(old_uvs) if old_uvs is not None else None
        out_faces: list[list[int]] = []

        def _add_vert(pos3d: np.ndarray,
                      uv2d:  "np.ndarray | None") -> int:
            """Append a new vertex and return its index."""
            idx = len(out_verts)
            out_verts.append(pos3d)
            if out_uvs is not None:
                out_uvs.append(uv2d if uv2d is not None
                               else np.zeros(2, dtype=np.float32))
            return idx

        def _bary(p: np.ndarray,
                  a: np.ndarray,
                  b: np.ndarray,
                  c: np.ndarray) -> np.ndarray:
            """
            Barycentric coordinates of 2-D point p in triangle (a, b, c).
            Returns (u, v, w) such that p ≈ u*a + v*b + w*c.
            """
            v0 = b - a;  v1 = c - a;  v2 = p - a
            d00 = float(np.dot(v0, v0))
            d01 = float(np.dot(v0, v1))
            d11 = float(np.dot(v1, v1))
            d20 = float(np.dot(v2, v0))
            d21 = float(np.dot(v2, v1))
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                return np.array([1.0, 0.0, 0.0])
            vv = (d11 * d20 - d01 * d21) / denom
            ww = (d00 * d21 - d01 * d20) / denom
            uu = 1.0 - vv - ww
            bary = np.array([uu, vv, ww])
            # Clamp + renormalise to guard against floating-point overshoot
            bary = np.clip(bary, 0.0, 1.0)
            s = bary.sum()
            return bary / s if s > 1e-12 else np.array([1.0, 0.0, 0.0])

        def _interp(pt2d: np.ndarray,
                    i0: int, i1: int, i2: int) -> "tuple[np.ndarray, np.ndarray|None]":
            """
            Interpolate 3-D position (and UV) for a screen-space point pt2d
            that lies on or inside triangle (i0, i1, i2).
            """
            a2d = np.array([sx[i0], sy[i0]])
            b2d = np.array([sx[i1], sy[i1]])
            c2d = np.array([sx[i2], sy[i2]])
            bary = _bary(pt2d, a2d, b2d, c2d)
            pos3d = (bary[0] * old_verts[i0]
                   + bary[1] * old_verts[i1]
                   + bary[2] * old_verts[i2])
            uv2d = None
            if old_uvs is not None:
                uv2d = (bary[0] * old_uvs[i0]
                      + bary[1] * old_uvs[i1]
                      + bary[2] * old_uvs[i2]).astype(np.float32)
            return pos3d, uv2d

        def _fan(indices: list[int]) -> list[list[int]]:
            """Fan-triangulate a convex/simple polygon given as vertex indices."""
            if len(indices) < 3:
                return []
            tris = []
            v0 = indices[0]
            for k in range(1, len(indices) - 1):
                tris.append([v0, indices[k], indices[k + 1]])
            return tris

        # --- Fully-outside faces: keep unchanged ---
        for fi in np.where(fully_outside_mask)[0]:
            out_faces.append([int(old_faces[fi, 0]),
                               int(old_faces[fi, 1]),
                               int(old_faces[fi, 2])])

        # --- Fully-inside faces: dropped (not appended) ---

        # --- Straddling faces: Shapely clip → triangulate outside fragment ---
        for fi in np.where(straddling_mask)[0]:
            i0, i1, i2 = int(old_faces[fi, 0]), \
                          int(old_faces[fi, 1]), \
                          int(old_faces[fi, 2])

            p0 = np.array([sx[i0], sy[i0]])
            p1 = np.array([sx[i1], sy[i1]])
            p2 = np.array([sx[i2], sy[i2]])

            try:
                tri_2d = _SPoly([tuple(p0), tuple(p1), tuple(p2)])
                if not tri_2d.is_valid:
                    tri_2d = tri_2d.buffer(0)
            except Exception:
                out_faces.append([i0, i1, i2])
                continue

            if tri_2d.area < 1e-6:
                # Degenerate triangle — keep as-is
                out_faces.append([i0, i1, i2])
                continue

            try:
                outside = tri_2d.difference(clip_poly)
            except Exception:
                out_faces.append([i0, i1, i2])
                continue

            if outside.is_empty:
                continue  # Fully covered — delete

            if outside.area >= tri_2d.area * 0.999:
                out_faces.append([i0, i1, i2])  # Effectively untouched
                continue

            # Collect output polygons (may be Polygon or MultiPolygon)
            # Triangulate the outside fragment using constrained Delaunay
            # triangulation (Shapely 2.x).  Fan triangulation only works for
            # convex polygons — when the clip polygon has sharp corners the
            # outside fragment can be L-shaped or otherwise non-convex, and
            # fan triangulation produces incorrect overlapping triangles that
            # leave corner artifacts in the hole boundary.
            import shapely as _shapely  # noqa: PLC0415
            try:
                tri_collection = _shapely.constrained_delaunay_triangles(outside)
                output_tris = list(tri_collection.geoms)
            except Exception:
                # Fallback: treat the whole outside as one polygon and fan it
                polys_fb = (list(outside.geoms)
                            if outside.geom_type == "MultiPolygon"
                            else [outside])
                output_tris = []
                for poly_fb in polys_fb:
                    coords_fb = list(poly_fb.exterior.coords)[:-1]
                    if len(coords_fb) >= 3:
                        vi_fb: list[int] = []
                        for pt in coords_fb:
                            pt2d = np.array(pt, dtype=np.float64)
                            if np.linalg.norm(pt2d - p0) < 0.5:
                                vi_fb.append(i0)
                            elif np.linalg.norm(pt2d - p1) < 0.5:
                                vi_fb.append(i1)
                            elif np.linalg.norm(pt2d - p2) < 0.5:
                                vi_fb.append(i2)
                            else:
                                pos3d, uv2d = _interp(pt2d, i0, i1, i2)
                                vi_fb.append(_add_vert(pos3d, uv2d))
                        out_faces.extend(_fan(vi_fb))
                continue  # skip the CDT loop below

            for tri_geom in output_tris:
                if tri_geom.is_empty:
                    continue
                coords = list(tri_geom.exterior.coords)[:-1]  # always 3 pts
                if len(coords) != 3:
                    continue

                # Map each CDT triangle vertex to a mesh vertex index.
                vert_indices: list[int] = []
                for pt in coords:
                    pt2d = np.array(pt, dtype=np.float64)
                    if np.linalg.norm(pt2d - p0) < 0.5:
                        vert_indices.append(i0)
                    elif np.linalg.norm(pt2d - p1) < 0.5:
                        vert_indices.append(i1)
                    elif np.linalg.norm(pt2d - p2) < 0.5:
                        vert_indices.append(i2)
                    else:
                        pos3d, uv2d = _interp(pt2d, i0, i1, i2)
                        vert_indices.append(_add_vert(pos3d, uv2d))

                if len(vert_indices) == 3:
                    out_faces.append(vert_indices)

        # ------------------------------------------------------------------ #
        # 5.  Convert accumulated lists to numpy arrays                       #
        # ------------------------------------------------------------------ #
        new_verts = np.array(out_verts, dtype=np.float64)
        new_faces = (np.array(out_faces, dtype=np.int64)
                     if out_faces
                     else np.zeros((0, 3), dtype=np.int64))
        new_uvs   = (np.array(out_uvs, dtype=np.float32)
                     if out_uvs is not None
                     else None)

        return new_verts, new_faces, new_uvs

    # ---------------------------------------------------------------------- #

    def delete_layer_faces(self, layer_id: int) -> None:
        """
        Permanently remove a layer's geometry from the mesh using cookie-cutter
        clipping when polygon data is available, or simple whole-face deletion
        as a fallback.

        Cookie-cutter path (preferred):
          Uses the 2D screen-space polygon and camera matrices stored at
          selection time to clip every boundary triangle exactly along the
          polygon outline via Shapely.  New vertices are interpolated at the
          crossing points so the resulting hole matches the drawn polygon shape
          precisely rather than following jagged triangle edges.

        Fallback path:
          If the layer pre-dates cookie-cutter support (no poly_pts_2d stored),
          falls back to the original whole-face keep-mask approach.

        Both paths:
          1. Push a full geometry snapshot for undo (Ctrl+Z restores the mesh)
          2. Rebuild the trimesh with new vertices/faces, removing orphaned verts
          3. Remap UVs to the new vertex layout
          4. Rebuild all topology caches (BVH, adjacency, seam partners)
          5. Schedule a GPU VBO/IBO re-upload and trigger a repaint

        Args:
            layer_id: ID of the layer whose faces should be deleted from the mesh.
        """
        if self._painter is None or layer_id not in self._layers:
            return

        layer = self._layers[layer_id]
        mesh  = self._painter._mesh

        # --- 1. Snapshot for undo BEFORE any modification ---
        self._history.push_snapshot(self._painter, geometry=True)

        # ------------------------------------------------------------------ #
        # Cookie-cutter path: exact polygon clipping via Shapely              #
        # ------------------------------------------------------------------ #
        poly_pts_2d = layer.get("poly_pts_2d")
        view_mat    = layer.get("view_mat")
        proj_mat    = layer.get("proj_mat")
        model_mat   = layer.get("model_mat")

        if (poly_pts_2d is not None and view_mat is not None
                and proj_mat is not None and model_mat is not None):
            log.info("ViewportWidget: delete_layer_faces — cookie-cutter path")
            new_vertices, new_faces_arr, new_uvs = self._cookie_cutter_clip(
                poly_pts_2d, view_mat, proj_mat, model_mat
            )
        else:
            # ---------------------------------------------------------------- #
            # Fallback: whole-face deletion (pre-cookie-cutter layers)         #
            # ---------------------------------------------------------------- #
            log.info("ViewportWidget: delete_layer_faces — whole-face fallback")
            faces_to_del = layer.get("faces", set())
            if not faces_to_del:
                return

            n_faces   = len(mesh.faces)
            old_verts = np.asarray(mesh.vertices, dtype=np.float64)
            old_faces = np.asarray(mesh.faces,    dtype=np.int64)

            keep_mask = np.ones(n_faces, dtype=bool)
            for fi in faces_to_del:
                if fi < n_faces:
                    keep_mask[fi] = False

            new_faces_raw = old_faces[keep_mask]
            used_verts    = np.unique(new_faces_raw.ravel())
            vert_remap    = np.full(len(old_verts), -1, dtype=np.int64)
            vert_remap[used_verts] = np.arange(len(used_verts), dtype=np.int64)

            new_faces_arr = vert_remap[new_faces_raw]
            new_vertices  = old_verts[used_verts]
            new_uvs       = (self._painter._uvs[used_verts]
                             if self._painter._uvs is not None else None)

        # ------------------------------------------------------------------ #
        # Shared finish: compact vertices, rebuild mesh + topology            #
        # ------------------------------------------------------------------ #
        # Remove orphaned vertices (vertices not referenced by any face).
        # The cookie-cutter path may add new vertices but never leaves orphans;
        # the fallback path removes faces so some original vertices become orphans.
        if len(new_faces_arr) == 0:
            log.warning("ViewportWidget: delete_layer_faces produced empty mesh — aborting")
            return

        new_faces_i64 = np.asarray(new_faces_arr, dtype=np.int64)
        used_verts    = np.unique(new_faces_i64.ravel())
        vert_remap    = np.full(len(new_vertices), -1, dtype=np.int64)
        vert_remap[used_verts] = np.arange(len(used_verts), dtype=np.int64)

        final_verts = new_vertices[used_verts]
        final_faces = vert_remap[new_faces_i64].astype(np.uint32)
        final_uvs   = new_uvs[used_verts] if new_uvs is not None else None

        import trimesh as _trimesh
        self._painter._mesh = _trimesh.Trimesh(
            vertices=final_verts,
            faces=final_faces,
            process=False,
        )
        if final_uvs is not None:
            self._painter._uvs = final_uvs

        self._painter._rebuild_topology()

        # Remove the layer and schedule GPU re-upload
        self.delete_layer(layer_id)
        self._needs_mesh_upload = True
        self.update()

        # Notify UI so the Undo button enables immediately after face deletion.
        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

    def project_texture_to_layer(
        self,
        layer_id: int,
        texture_path: str,
        rotate: float,
        scale: float,
        offset_x: float,
        offset_y: float,
    ) -> None:
        """
        Set up a shader-based planar texture projection for a saved layer.

        Replaces the old UV-atlas baking approach. Rather than rasterizing the
        projection into the PIL albedo buffer (which produced jagged UV-seam
        artifacts), this method:

            1. Calls painter.setup_layer_projection() to compute the projection
               frame (normal, right, up, normalization bounds) and store params.
            2. Uploads the source texture to GPU as a trilinear-mipmap texture
               on texture unit 1.
            3. Builds a face-subset index buffer (IBO) containing only the
               selected faces so the projection pass draws exactly those faces.
            4. Builds a VAO binding in_position from the existing interleaved VBO.
            5. Stores all GPU resources in _proj_layers[layer_id].

        The projection is rendered in paintGL() as a second pass over the mesh
        — shape is perfect (matches face selection in 3D), quality is GPU-filtered.

        Parameters
        ----------
        layer_id     : Layer whose face set defines the projection area.
        texture_path : Absolute path to the source texture (PNG, JPG …).
        rotate       : Counter-clockwise rotation in degrees.
        scale        : Uniform scale (1.0 = fill selection exactly).
        offset_x     : Horizontal shift (-1.0 to 1.0).
        offset_y     : Vertical shift (-1.0 to 1.0).
        """
        import math as _math
        import traceback as _tb
        log.info("project_texture_to_layer: called layer=%d path=%s", layer_id, texture_path)
        try:
            self._project_texture_to_layer_impl(layer_id, texture_path, rotate, scale, offset_x, offset_y)
        except Exception:
            log.error("project_texture_to_layer: UNHANDLED EXCEPTION\n%s", _tb.format_exc())
        return

    def _project_texture_to_layer_impl(
        self,
        layer_id: int,
        texture_path: str,
        rotate: float,
        scale: float,
        offset_x: float,
        offset_y: float,
    ) -> None:
        """Inner implementation — called from project_texture_to_layer which wraps in try/except."""
        if self._painter is None or layer_id not in self._layers:
            log.warning(
                "project_texture_to_layer: layer %d not found or no painter (layers=%s)",
                layer_id, list(self._layers.keys())
            )
            return

        # Load source texture from disk — CPU only, no GL context needed yet.
        try:
            from PIL import Image as _Image
            texture_img = _Image.open(texture_path).convert("RGBA")
        except Exception as e:
            log.warning("project_texture_to_layer: could not load %s: %s", texture_path, e)
            return

        # Snapshot BEFORE applying projection so it is undoable.
        self._history.push_snapshot(self._painter, geometry=False)
        self.history_changed.emit(self._history.can_undo, self._history.can_redo)

        face_set = set(self._layers[layer_id]["faces"])

        # For cookie-cutter layers the face set may be empty — recompute from polygon.
        if not face_set:
            layer     = self._layers[layer_id]
            pts_2d    = layer.get("poly_pts_2d")
            view_mat  = layer.get("view_mat")
            proj_mat  = layer.get("proj_mat")
            model_mat = layer.get("model_mat")
            if pts_2d and view_mat is not None and proj_mat is not None and model_mat is not None:
                from PIL import Image as _PilFace, ImageDraw as _PilFaceD
                w_fb, h_fb = self.width(), self.height()
                if w_fb > 0 and h_fb > 0:
                    _mask_pil = _PilFace.new("L", (w_fb, h_fb), 0)
                    _PilFaceD.Draw(_mask_pil).polygon(pts_2d, fill=255)
                    _mask_np  = np.array(_mask_pil, dtype=np.uint8)
                    _mvp_r    = (proj_mat @ view_mat @ model_mat.astype(np.float64)).astype(np.float32)
                    _verts    = np.asarray(self._painter._mesh.vertices, dtype=np.float32)
                    _faces_np = np.asarray(self._painter._mesh.faces,    dtype=np.int32)
                    _verts_h  = np.hstack([_verts, np.ones((_verts.shape[0], 1), np.float32)])
                    _clip     = (_mvp_r @ _verts_h.T).T
                    _wc       = np.where(np.abs(_clip[:, 3]) < 1e-8, 1e-8, _clip[:, 3])
                    _sx       = ((_clip[:, 0] / _wc + 1.0) * 0.5 * w_fb).astype(int)
                    _sy       = ((1.0 - (_clip[:, 1] / _wc + 1.0) * 0.5) * h_fb).astype(int)
                    _cx       = np.clip((_sx[_faces_np[:, 0]] + _sx[_faces_np[:, 1]] + _sx[_faces_np[:, 2]]) // 3, 0, w_fb - 1)
                    _cy       = np.clip((_sy[_faces_np[:, 0]] + _sy[_faces_np[:, 1]] + _sy[_faces_np[:, 2]]) // 3, 0, h_fb - 1)
                    _inside   = _mask_np[_cy, _cx] > 0
                    face_set  = set(int(i) for i in np.where(_inside)[0])
            if not face_set:
                log.warning("project_texture_to_layer: layer %d has no faces and polygon recompute failed", layer_id)
                return

        # Compute projection frame (CPU — no GL needed).
        params = self._painter.setup_layer_projection(
            layer_id,
            face_set,
            texture_img,
            rotate_deg=rotate,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        if params is None:
            log.warning("project_texture_to_layer: setup_layer_projection returned None")
            return

        # ------------------------------------------------------------------
        # Defer all GPU resource creation to paintGL so it runs inside Qt's
        # managed GL frame (context guaranteed current, no makeCurrent needed).
        # Store everything the paintGL setup pass will need.
        # ------------------------------------------------------------------
        self._pending_proj_setups[layer_id] = {
            "texture_img": texture_img,   # RGBA PIL image
            "face_set":    face_set,
            "params":      params,
            "visible":     True,
        }
        log.info("project_texture_to_layer: queued pending setup for layer %d (%d faces)",
                 layer_id, len(face_set))

        # Repaint — projection pass runs next frame
        self.update()

    def save_baked_albedo(self) -> bool:
        """
        Bake all visible shader projection layers onto the UV atlas and write
        the result to albedo.png on disk.

        Called by the app layer just before export_mesh() so that projected
        textures are embedded in the exported file rather than existing only
        as GPU-side runtime state.

        Returns True if the bake was performed and saved, False if there was
        nothing to bake (no painter, no projection layers) — in which case
        the existing albedo.png is used unchanged.
        """
        if self._painter is None:
            return False
        if not self._proj_layers and not self._painter._layer_projections:
            return False

        log.info("ViewportWidget: baking %d projection layer(s) for export",
                 len(self._painter._layer_projections))
        baked = self._painter.bake_projections_to_atlas()
        self._painter._albedo_path.parent.mkdir(parents=True, exist_ok=True)
        baked.save(str(self._painter._albedo_path), format="PNG")
        log.info("ViewportWidget: baked albedo saved to %s",
                 self._painter._albedo_path)
        return True

    def get_pending_face_count(self) -> int:
        """Return the face count of the pending polygon selection, or 0."""
        return len(self._pending_faces) if self._pending_faces else 0

    def clear_pending_selection(self) -> None:
        """Discard the pending polygon selection without saving it as a layer."""
        self._pending_faces = None
        self._pending_rebuild = False
        self._release_pending_vao()
        self.update()
