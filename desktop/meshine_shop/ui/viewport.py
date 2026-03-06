"""
3D viewport widget — OpenGL rendering, camera navigation, and texture painting.

This is the heart of the Phase 5 Viewport feature. ViewportWidget is a
QOpenGLWidget subclass that uses moderngl to render the UV-unwrapped,
photo-textured mesh in real-time. It handles all mouse and keyboard input for
UE-style camera navigation and routes paint interactions to MeshPainter.

Rendering pipeline (per frame):
    1. Bind albedo texture (photo-texture or painted version)
    2. Compute MVP = projection × view from ViewportCamera
    3. Draw all triangles via a single VAO render call
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
        self._prog: moderngl.Program | None = None
        self._vao: moderngl.VertexArray | None = None
        self._vbo: moderngl.Buffer | None = None
        self._ibo: moderngl.Buffer | None = None
        self._albedo_tex: moderngl.Texture | None = None

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
        self._tool: str = "brush"             # "brush" | "region"
        self._paint_color: tuple = (220, 100, 60)  # default rust-orange
        self._brush_size: int = 12            # pixels
        self._brush_opacity: float = 1.0

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

        # Flag that GPU resources need to be uploaded on the next paintGL call.
        # Deferring to paintGL() guarantees we're inside Qt's OpenGL context
        # (makeCurrent() has already been called by Qt for us).
        self._needs_upload = True

        self.update()

    def set_tool(self, tool: str) -> None:
        """Switch active paint tool. tool must be 'brush' or 'region'."""
        self._tool = tool

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

        # Release GPU resources
        if self._ctx is not None:
            self.makeCurrent()
            self._release_gpu_resources()
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
                log.info("ViewportWidget: GPU upload complete — vao=%s tex=%s",
                         self._vao, self._albedo_tex)
            except Exception:
                log.exception("ViewportWidget: GPU upload failed")

        if self._painter is None or self._vao is None:
            # No mesh loaded (or upload failed) — show clear background
            qt_fbo.clear(0.18, 0.18, 0.18)
            return

        # Clear color + depth
        qt_fbo.clear(0.18, 0.18, 0.18)

        # Upload any texture pixels that were painted since the last frame
        self._refresh_dirty_texture()

        # Compute MVP matrices
        aspect = self.width() / max(self.height(), 1)
        view = self._camera.get_view_matrix()
        proj = self._camera.get_projection_matrix(aspect)
        mvp = (proj @ view).astype("f4")
        mv = view.astype("f4")

        # Upload uniforms — OpenGL expects column-major (Fortran order).
        # numpy stores matrices row-major, so .T.tobytes() transposes before
        # serialising, giving OpenGL the correct column-major layout.
        self._prog["MVP"].write(mvp.T.tobytes())
        self._prog["MV"].write(mv.T.tobytes())
        self._prog["albedo"].value = 0   # texture unit 0

        # Bind texture and draw
        self._albedo_tex.use(0)
        self._vao.render()

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
        """Release all GPU buffers and textures."""
        for attr in ("_vao", "_vbo", "_ibo", "_albedo_tex"):
            resource = getattr(self, attr, None)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
            setattr(self, attr, None)

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

        # Unproject to world-space ray
        origin, direction = self._camera.unproject_ray(ndc_x, ndc_y, aspect)

        # 1. Try mesh surface hit
        hit_pos = self._painter.ray_cast_world_pos(origin, direction)
        if hit_pos is not None:
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
            # Capture global cursor position at press — restored on release so
            # the cursor reappears exactly where the user first clicked.
            self._lmb_cursor_origin_global = QCursor.pos()
            # If the click lands on the mesh, hide the system cursor to give
            # an "attached to model" feel while orbiting. Restored on release.
            if self._camera is not None and self._painter is not None:
                w, h = self.width(), self.height()
                if w > 0 and h > 0:
                    ndc_x = (int(x) / w) * 2.0 - 1.0
                    ndc_y = 1.0 - (int(y) / h) * 2.0
                    origin, direction = self._camera.unproject_ray(
                        ndc_x, ndc_y, w / h)
                    hit = self._painter.ray_cast_world_pos(origin, direction)
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
                            dot_world = self._painter.ray_cast_world_pos(
                                c_origin, c_dir
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

        elif self._mouse_binding_active("orbit"):
            # LMB drag → orbit around the pivot.
            # After a miss-case cursor warp, absorb the very first move event
            # as a new anchor. The warp can deliver an event from either the
            # click position or the warped position — either could produce a
            # huge delta. Eating it here prevents that snap entirely.
            if self._orbit_skip_one:
                self._orbit_skip_one = False
                self.update()
                return
            if dx != 0 or dy != 0:
                self._camera.orbit(dx, dy)
            self.update()

        elif self._lmb_held and self._tool == "brush" and not self._alt_held:
            # LMB drag (no chord, no alt) → brush stroke
            self._handle_paint(x, y)

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
                hit = self._painter.ray_cast_world_pos(origin, ray_dir)
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
