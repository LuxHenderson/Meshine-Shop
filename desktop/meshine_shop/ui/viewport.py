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
    - RMB held + mouse drag   → fly look (yaw/pitch)
    - RMB held + WASD/Q/E     → fly move (translate in view space)
    - RMB held + Shift         → 3× speed boost
    - Alt + LMB drag          → orbit around focal point
    - Scroll wheel             → dolly (zoom in/out)
    - MMB drag                 → pan (shift focal point + position)
    - F key                    → frame mesh (auto-position camera)

Texture painting:
    - Brush tool: click/drag on mesh → ray cast → UV paint → GPU upload
    - Region tool: click on mesh → BFS flood fill → GPU upload
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

try:
    import moderngl
    _HAS_MODERNGL = True
except ImportError:
    _HAS_MODERNGL = False

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from meshine_shop.core.mesh_painter import MeshPainter
from meshine_shop.core.viewport_camera import ViewportCamera

log = logging.getLogger(__name__)

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
        self._fly_mode: bool = False          # True while RMB is held
        self._alt_held: bool = False          # True while Alt is held
        self._mmb_held: bool = False          # True while MMB is held
        self._lmb_held: bool = False          # True while LMB is held (for paint drag)
        self._keys_held: set[Qt.Key] = set()
        self._last_mouse_pos: tuple[int, int] | None = None

        # Fly tick timer — fires at ~60fps to apply WASD movement while keys held
        self._fly_timer = QTimer(self)
        self._fly_timer.setInterval(16)       # ~60fps
        self._fly_timer.timeout.connect(self._fly_tick)
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
        """Clear all state and stop the fly timer. Returns widget to idle."""
        self._fly_timer.stop()
        self._painter = None
        self._camera = None
        self._needs_upload = False
        self._fly_mode = False
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        x, y = event.position().x(), event.position().y()
        self._last_mouse_pos = (int(x), int(y))

        btn = event.button()

        if btn == Qt.MouseButton.RightButton:
            # Enter fly mode
            self._fly_mode = True
            self._last_tick_time = time.monotonic()
            self._fly_timer.start()
            self.setCursor(Qt.CursorShape.BlankCursor)

        elif btn == Qt.MouseButton.MiddleButton:
            self._mmb_held = True

        elif btn == Qt.MouseButton.LeftButton:
            self._lmb_held = True
            if not self._alt_held and self._camera is not None:
                # Paint on LMB press (not when Alt is held — that's orbit)
                self._handle_paint(int(x), int(y))

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

        if self._fly_mode:
            # RMB drag → fly look (yaw/pitch)
            self._camera.fly_look(dx, dy)
            self.update()

        elif self._alt_held and self._lmb_held:
            # Alt + LMB drag → orbit
            self._camera.orbit(dx, dy)
            self.update()

        elif self._mmb_held:
            # MMB drag → pan
            self._camera.pan(dx, dy)
            self.update()

        elif self._lmb_held and self._tool == "brush" and not self._alt_held:
            # LMB drag → brush stroke
            self._handle_paint(x, y)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        btn = event.button()

        if btn == Qt.MouseButton.RightButton:
            self._fly_mode = False
            self._fly_timer.stop()
            self.unsetCursor()

        elif btn == Qt.MouseButton.MiddleButton:
            self._mmb_held = False

        elif btn == Qt.MouseButton.LeftButton:
            self._lmb_held = False

        self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._camera is None:
            return
        # angleDelta().y() is 120 per notch (standard Qt)
        ticks = event.angleDelta().y() / 120.0
        self._camera.dolly(ticks)
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
    # Fly tick — WASD movement at ~60fps                                   #
    # ------------------------------------------------------------------ #

    def _fly_tick(self) -> None:
        """
        Called by QTimer at ~60fps while in fly mode.

        Reads held keys and translates the camera in the view-space directions
        that match the UE fly navigation model.
        """
        if not self._fly_mode or self._camera is None:
            return

        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        # Clamp dt to avoid large jumps after a pause
        dt = min(dt, 0.1)

        # Read movement keys (defaults match DEFAULT_BINDINGS)
        bindings = self._camera.settings.bindings
        forward_key = self._qt_key(bindings.get("forward", "W"))
        backward_key = self._qt_key(bindings.get("backward", "S"))
        left_key = self._qt_key(bindings.get("left", "A"))
        right_key = self._qt_key(bindings.get("right", "D"))
        up_key = self._qt_key(bindings.get("up", "E"))
        down_key = self._qt_key(bindings.get("down", "Q"))
        boost_key = self._qt_key(bindings.get("boost", "Shift"))

        fwd = (1 if forward_key in self._keys_held else 0) - (
            1 if backward_key in self._keys_held else 0
        )
        right = (1 if right_key in self._keys_held else 0) - (
            1 if left_key in self._keys_held else 0
        )
        up = (1 if up_key in self._keys_held else 0) - (
            1 if down_key in self._keys_held else 0
        )
        boost = boost_key in self._keys_held

        if fwd != 0 or right != 0 or up != 0:
            self._camera.fly_move(fwd, right, up, dt, boost=boost)
            self.update()

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
