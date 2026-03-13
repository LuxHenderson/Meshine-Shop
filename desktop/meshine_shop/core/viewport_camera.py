"""
3D viewport camera for Meshine Shop.

This module provides the ViewportCamera class — a pure Python/numpy camera with
Unreal Engine-style navigation controls and fully user-configurable keybindings.
It has no Qt or OpenGL dependencies; the QOpenGLWidget queries it for matrices.

Navigation modes (defaults, all rebindable via ViewportSettings.bindings):
    Orbit:      LMB drag               → orbit around focal_point
    Dolly:      Scroll wheel           → zoom toward cursor (surface hit or focal plane)
    Pan:        MMB drag               → translate camera + focal_point perpendicular to view
    Frame:      F key                  → reset camera to frame the full mesh

Settings and bindings are stored in ~/.meshine_shop/config.json under the
"viewport" key and survive app restarts. The factory DEFAULT_BINDINGS constant
is used by the controls dialog ↺ reset-to-default buttons.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Default keybinding definitions
# ---------------------------------------------------------------------------
# Each binding is either:
#   Mouse action: {"modifiers": [str, ...], "mouse": "left"|"right"|"middle"|"scroll"}
#   Keyboard key: a string key name matching Qt.Key name without "Key_" prefix
#
# These are the factory defaults. The controls dialog exposes ↺ buttons to
# restore any individual binding back to its entry here.

DEFAULT_BINDINGS: dict = {
    # Mouse controls
    "orbit":    {"modifiers": [],      "mouse": "left"},
    "pan":      {"modifiers": [],      "mouse": "middle"},
    "dolly":    {"modifiers": [],      "mouse": "scroll"},
    # Keyboard controls
    "forward":  "W",
    "left":     "A",
    "backward": "S",
    "right":    "D",
    "up":       "E",
    "down":     "Q",
    "frame":    "F",
}

# Preset schemes — selecting one overwrites all bindings at once.
BINDING_PRESETS: dict[str, dict] = {
    "Default": DEFAULT_BINDINGS.copy(),
    "Unreal Engine": {
        "orbit":    {"modifiers": [],      "mouse": "left"},
        "pan":      {"modifiers": ["Alt"], "mouse": "right"},
        "dolly":    {"modifiers": [],      "mouse": "scroll"},
        "forward":  "W",
        "left":     "A",
        "backward": "S",
        "right":    "D",
        "up":       "E",
        "down":     "Q",
        "frame":    "F",
    },
}


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class ViewportSettings:
    """
    All user-configurable viewport parameters.

    Stored in ~/.meshine_shop/config.json under the "viewport" key.
    The controls dialog reads and writes this through the camera instance.
    """
    mouse_sensitivity: float = 0.3    # Degrees per pixel for orbit
    keyboard_speed: float = 5.0       # World units per second for WASD movement
    scroll_speed: float = 1.0         # World units per scroll tick
    invert_y: bool = False            # Invert vertical look direction
    invert_x: bool = False            # Invert horizontal look direction
    scheme: str = "Default"           # Active preset name ("Custom" if modified)
    bindings: dict = field(default_factory=lambda: dict(DEFAULT_BINDINGS))


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    """Return the path to the shared Meshine Shop config file."""
    return Path.home() / ".meshine_shop" / "config.json"


def _load_config() -> dict:
    """Load the full config.json, returning {} on missing/corrupt file."""
    path = _config_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(config: dict) -> None:
    """Write the full config dict back to config.json."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class ViewportCamera:
    """
    Perspective camera with Unreal Engine-style navigation controls.

    All navigation inputs are processed by the QOpenGLWidget and forwarded
    here as deltas or discrete events. The camera updates its position,
    yaw, and pitch, then the widget calls get_view_matrix() each frame.

    Coordinate system: right-handed, Y-up.
    """

    def __init__(self):
        # --- Camera pose ---
        # Position in world space.
        self.position: np.ndarray = np.array([0.0, 0.0, 3.0], dtype=np.float64)

        # Euler angles (radians).
        # Yaw: rotation around world-Y (left/right look). 0 = looking toward -Z.
        # Pitch: rotation around local-X (up/down look). Clamped to ±89°.
        self.yaw: float = 0.0
        self.pitch: float = 0.0

        # The point the orbit mode rotates around. Updated by frame_mesh()
        # and pan operations.
        self.focal_point: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)


        # --- Settings (loaded from config or defaulted) ---
        self.settings: ViewportSettings = ViewportSettings()
        self.load_settings()

    # -----------------------------------------------------------------------
    # Settings persistence
    # -----------------------------------------------------------------------

    def load_settings(self) -> None:
        """Read viewport settings from ~/.meshine_shop/config.json."""
        config = _load_config()
        vp = config.get("viewport", {})

        s = self.settings
        s.mouse_sensitivity = float(vp.get("mouse_sensitivity", s.mouse_sensitivity))
        s.keyboard_speed    = float(vp.get("keyboard_speed",    s.keyboard_speed))
        s.scroll_speed      = float(vp.get("scroll_speed",      s.scroll_speed))
        s.invert_y          = bool (vp.get("invert_y",          s.invert_y))
        s.invert_x          = bool (vp.get("invert_x",          s.invert_x))
        s.scheme            = str  (vp.get("scheme",            s.scheme))

        # Load bindings — merge with defaults so new actions added in future
        # releases don't silently break on existing config files.
        saved_bindings = vp.get("bindings", {})
        s.bindings = {**DEFAULT_BINDINGS, **saved_bindings}

    def save_settings(self) -> None:
        """Write current viewport settings back to ~/.meshine_shop/config.json."""
        config = _load_config()
        s = self.settings
        config["viewport"] = {
            "mouse_sensitivity": s.mouse_sensitivity,
            "keyboard_speed":    s.keyboard_speed,
            "scroll_speed":      s.scroll_speed,
            "invert_y":          s.invert_y,
            "invert_x":          s.invert_x,
            "scheme":            s.scheme,
            "bindings":          s.bindings,
        }
        _save_config(config)

    def get_bindings(self) -> dict:
        """Return a copy of the current bindings for the controls dialog."""
        return dict(self.settings.bindings)

    def apply_bindings(self, bindings: dict, scheme: str = "Custom") -> None:
        """
        Apply a new bindings dict (from the controls dialog Save action).

        Args:
            bindings: Full bindings dict with all action keys.
            scheme:   Display name of the preset or "Custom".
        """
        self.settings.bindings = dict(bindings)
        self.settings.scheme = scheme
        self.save_settings()

    # -----------------------------------------------------------------------
    # Matrix generation
    # -----------------------------------------------------------------------

    def _forward(self) -> np.ndarray:
        """Unit vector pointing in the camera's look direction."""
        # Spherical → Cartesian for right-handed Y-up:
        #   forward.x =  cos(pitch) * sin(yaw)
        #   forward.y = -sin(pitch)           (negative because pitch up = look up)
        #   forward.z = -cos(pitch) * cos(yaw) (negative Z is "into screen")
        p, y = self.pitch, self.yaw
        fx =  math.cos(p) * math.sin(y)
        fy = -math.sin(p)
        fz = -math.cos(p) * math.cos(y)
        return np.array([fx, fy, fz], dtype=np.float64)

    def _right(self) -> np.ndarray:
        """Unit vector pointing to the camera's right."""
        fwd = self._forward()
        up  = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(fwd, up)
        n = np.linalg.norm(r)
        return r / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])

    def get_view_matrix(self) -> np.ndarray:
        """
        Compute a right-handed look-at view matrix (column-major float64).

        The matrix transforms world-space positions into camera/eye space.
        """
        fwd = self._forward()
        up  = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r   = self._right()
        # Recompute orthogonal up from right × forward
        u   = np.cross(r, fwd)

        # Build the 4×4 look-at view matrix.
        # Row 0 = right, Row 1 = up, Row 2 = -forward (OpenGL convention).
        view = np.eye(4, dtype=np.float64)
        view[0, :3] =  r
        view[1, :3] =  u
        view[2, :3] = -fwd
        view[0, 3]  = -np.dot(r,    self.position)
        view[1, 3]  = -np.dot(u,    self.position)
        view[2, 3]  =  np.dot(fwd,  self.position)
        return view

    def get_projection_matrix(self, aspect: float,
                               fov: float = 60.0,
                               near: float = 0.01,
                               far: float = 1000.0) -> np.ndarray:
        """
        Compute a standard perspective projection matrix.

        Args:
            aspect: Viewport width / height ratio.
            fov:    Vertical field of view in degrees.
            near:   Near clip plane distance.
            far:    Far clip plane distance.

        Returns:
            4×4 perspective projection matrix (column-major float64).
        """
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def frame_mesh(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
        """
        Position and orient the camera to frame the full mesh bounding box.

        Places the camera back along the -Z axis at a distance that ensures
        the full diagonal of the bounding box is visible with FOV=60°.
        Also resets yaw and pitch to a slightly elevated front-facing angle.

        Args:
            bbox_min: (3,) array — minimum corner of the mesh AABB.
            bbox_max: (3,) array — maximum corner of the mesh AABB.
        """
        centroid = (bbox_min + bbox_max) * 0.5
        diagonal = float(np.linalg.norm(bbox_max - bbox_min))

        # Distance to fit the diagonal in a 60° FOV with some padding.
        dist = (diagonal / 2.0) / math.tan(math.radians(30.0)) * 1.2

        self.focal_point = centroid.copy()
        # Slightly elevated front view — camera above centroid, looking down at model.
        self.yaw   = 0.0
        self.pitch = math.radians(20.0)  # positive pitch = camera high, looking down
        fwd = self._forward()
        self.position = centroid - fwd * dist

    def orbit(self, dx: float, dy: float) -> None:
        """
        Orbit camera position around focal_point.

        Rotates the position vector around the focal point by the mouse delta.
        The look direction is always updated to aim at focal_point.

        Args:
            dx: Horizontal mouse delta (positive = orbit right).
            dy: Vertical mouse delta (positive = orbit up).
        """
        sens = self.settings.mouse_sensitivity
        x_sign = -1.0 if self.settings.invert_x else 1.0
        y_sign = -1.0 if self.settings.invert_y else 1.0

        d_yaw   = math.radians(dx * sens * x_sign)
        d_pitch = math.radians(dy * sens * y_sign)

        # Vector from focal_point to camera.
        offset = self.position - self.focal_point
        dist = float(np.linalg.norm(offset))
        if dist < 1e-6:
            return

        # Current spherical angles.
        cur_yaw   = math.atan2(offset[0], -offset[2])
        cur_pitch = math.asin(max(-1.0, min(1.0, offset[1] / dist)))

        new_yaw   = cur_yaw + d_yaw
        new_pitch = max(-math.radians(89), min(math.radians(89),
                                               cur_pitch - d_pitch))

        # Reconstruct position from new spherical coords.
        self.position = self.focal_point + np.array([
            dist * math.cos(new_pitch) * math.sin(new_yaw),
            dist * math.sin(new_pitch),
           -dist * math.cos(new_pitch) * math.cos(new_yaw),
        ], dtype=np.float64)

        # Aim camera at focal_point.
        to_focal = self.focal_point - self.position
        self.yaw   = math.atan2(to_focal[0], -to_focal[2])
        self.pitch = math.asin(max(-1.0, min(1.0,
                                   -to_focal[1] / np.linalg.norm(to_focal))))

    def dolly(self, ticks: float, direction: "np.ndarray | None" = None) -> None:
        """
        Zoom by moving the camera along a direction.

        Step size is proportional to the distance to the target point so
        each scroll tick moves the same visual percentage of remaining
        distance regardless of how close the camera is. This prevents the
        "accelerating lurch" that fixed-unit dolly produces when close to
        the model. Factor 0.15 = ~15% of distance per tick at scroll_speed=1.0.

        Args:
            ticks:     Scroll wheel ticks. Positive = zoom in, negative = zoom out.
            direction: World-space vector to move along. If None, uses the camera's
                       forward vector. Pass (target_point - position) to zoom toward
                       a specific world point (e.g. what's under the cursor).
        """
        # Proportional step factor — each tick covers this fraction of the
        # current distance. Gives consistent visual speed at any zoom level.
        _PROPORTION = 0.15

        if direction is None:
            d = self._forward()
            # Use distance to focal point as the reference distance.
            dist = max(0.01, float(np.linalg.norm(self.focal_point - self.position)))
            self.position += d * (ticks * self.settings.scroll_speed * dist * _PROPORTION)
        else:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                return
            d = direction / norm
            # norm is the distance to the target point — ideal reference for
            # proportional movement toward exactly what's under the cursor.
            dist = max(0.01, norm)
            delta = d * (ticks * self.settings.scroll_speed * dist * _PROPORTION)
            # Move both position and focal_point by the same delta so the
            # orbital relationship (distance, yaw, pitch) stays intact.
            # Without this, orbit after a cursor-targeted zoom snaps because
            # position - focal_point points in a different direction than the
            # stored yaw/pitch.
            self.position    += delta
            self.focal_point += delta

    def pan(self, dx: float, dy: float) -> None:
        """
        Pan the camera by translating both position and focal_point perpendicular
        to the view direction.

        Args:
            dx: Horizontal pixel delta.
            dy: Vertical pixel delta.
        """
        sens = self.settings.mouse_sensitivity * 0.01  # Pan is slower than look
        r   = self._right()
        # Use the camera's actual up vector (perpendicular to both right and
        # forward) instead of world Y. World Y causes arcing/spherical motion
        # when the camera has any pitch because the vertical pan axis is then
        # diagonal relative to the screen's vertical at non-zero pitch angles.
        up  = np.cross(r, self._forward())

        delta = -r * (dx * sens) + up * (dy * sens)
        self.position    += delta
        self.focal_point += delta

    # -----------------------------------------------------------------------
    # Ray unprojection (for painting)
    # -----------------------------------------------------------------------

    def unproject_ray(self, ndc_x: float, ndc_y: float,
                      aspect: float, fov: float = 60.0
                      ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a world-space ray from a normalised device coordinate.

        Used by the viewport to turn a mouse click into a ray for
        MeshPainter.ray_cast().

        Args:
            ndc_x:  Normalised X in [−1, +1] (left to right).
            ndc_y:  Normalised Y in [−1, +1] (bottom to top).
            aspect: Viewport width / height.
            fov:    Vertical FOV in degrees (must match projection matrix).

        Returns:
            (ray_origin, ray_direction) — both world-space (3,) float64 arrays.
            ray_direction is normalised.
        """
        # Reconstruct the camera basis.
        fwd = self._forward()
        r   = self._right()
        u   = np.cross(r, fwd)

        # Half-extents of the near plane at distance 1 from the camera.
        half_h = math.tan(math.radians(fov) / 2.0)
        half_w = half_h * aspect

        # Direction in world space.
        direction = fwd + r * (ndc_x * half_w) + u * (ndc_y * half_h)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm

        return self.position.copy(), direction
