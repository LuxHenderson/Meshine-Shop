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

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import (
    QColor, QCursor, QKeyEvent, QMouseEvent, QPainter, QWheelEvent,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ------------------------------------------------------------------ #
        # moderngl context + GL resources (set in initializeGL / load_mesh)   #
        # ------------------------------------------------------------------ #
        self._ctx: moderngl.Context | None = None

        # Mesh rendering resources (released and rebuilt on each mesh load)
        self._prog: moderngl.Program | None = None
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
        # Set True by load_mesh(); cleared in paintGL after GPU upload.
        # Ensures GPU upload happens inside the guaranteed-current GL context.
        self._needs_upload: bool = False

        # ------------------------------------------------------------------ #
        # Paint tool state                                                     #
        # ------------------------------------------------------------------ #
        self._tool: str = ""                  # "" | "brush" | "region" | "rotate"
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

    def reset_model_rotation(self) -> None:
        """
        Snap the mesh back to its original orientation (identity rotation).

        Called by the Reset button in the tools panel. Clears _model_rot to
        the identity matrix so the MVP shader receives no model transform.
        """
        self._model_rot = np.eye(3, dtype=np.float32)
        self._gizmo_axis = None
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

        # Release mesh GPU resources (environment resources are preserved
        # across mesh loads and will be rebuilt by initializeGL on re-show)
        if self._ctx is not None:
            self.makeCurrent()
            self._release_gpu_resources()
            self._release_gizmo()     # explicit teardown (not part of _release_gpu_resources)
            self._release_environment()
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
        qt_fbo.use()

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

        # Clear depth to 1.0 and color to black; the background gradient quad
        # will immediately overwrite every color pixel, so the clear color
        # doesn't matter visually — we just need a clean depth buffer.
        qt_fbo.clear(0.0, 0.0, 0.0)

        # --- 1. Background gradient (fullscreen NDC quad) ---
        # Depth test and depth write are disabled so the quad paints every
        # pixel without interfering with scene depth values.
        if self._bg_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._bg_vao.render()
            self._ctx.enable(moderngl.DEPTH_TEST)

        # If no camera is ready, stop here — grid + mesh need camera matrices.
        if self._camera is None:
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
            # No mesh yet — environment still renders cleanly above
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

    def _release_gpu_resources(self) -> None:
        """Release mesh GPU buffers and textures (called on each new mesh load)."""
        for attr in ("_vao", "_vbo", "_ibo", "_albedo_tex"):
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

            # --- Tool mode: LMB activates the tool on the mesh only ---
            # Region fill fires on press (single click); brush starts its first
            # stroke here and continues on drag via mouseMoveEvent.
            # Clicking in empty space does nothing — ray_cast returns None.
            if self._tool:
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

        elif self._tool == "brush" and self._lmb_held:
            # Tool active — brush stroke on drag; orbit is suspended.
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
        self.update()

    # ------------------------------------------------------------------ #
    # Keyboard input                                                       #
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()

        # Track alt modifier
        if key == Qt.Key.Key_Alt:
            self._alt_held = True

        # Frame mesh on F
        if key == Qt.Key.Key_F and self._camera is not None and self._painter is not None:
            bbox_min, bbox_max = self._painter.get_bbox()
            self._camera.frame_mesh(bbox_min, bbox_max)
            self.update()
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

        if self._tool == "brush":
            self._painter.paint_brush(
                uv, self._paint_color, self._brush_size, self._brush_opacity
            )
        elif self._tool == "region":
            self._painter.fill_region(face_idx, self._paint_color)

        self.update()  # triggers paintGL → _refresh_dirty_texture
