"""
Viewport tools panel and keybinding controls dialog.

ViewportToolsPanel — the 200px fixed-width right-side panel on the Viewport
page. Contains tool selector buttons (Brush / Region Select), a color swatch,
size and opacity sliders, and a gear button that opens the controls dialog.

ViewportControlsDialog — a full keybinding customizer. Two-column layout:
left = category list, right = active bindings table with per-action
✕ (clear), ↺ (reset to default), and ✎ (rebind) buttons. Also contains
movement speed sliders. All settings are read from and written to
ViewportCamera.settings, which persists to ~/.meshine_shop/config.json.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from meshine_shop.core.viewport_camera import (
    DEFAULT_BINDINGS,
    ViewportCamera,
    ViewportSettings,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Action display metadata                                                       #
# --------------------------------------------------------------------------- #

# Human-readable names for each binding key in DEFAULT_BINDINGS.
# Used to populate the controls dialog rows in the correct order.

MOUSE_ACTIONS = [
    ("orbit",    "Orbit"),
    ("pan",      "Pan"),
    ("dolly",    "Dolly"),
    ("fly_look", "Fly Look"),
]

KEY_ACTIONS = [
    ("forward",  "Move Forward"),
    ("backward", "Move Backward"),
    ("left",     "Move Left"),
    ("right",    "Move Right"),
    ("up",       "Move Up"),
    ("down",     "Move Down"),
    ("boost",    "Speed Boost"),
    ("frame",    "Frame Mesh"),
]

# Category names shown in the left list of the controls dialog
CATEGORIES = ["Mouse Controls", "Keyboard Controls", "Movement Speed"]


def _binding_to_label(key: str, binding) -> str:
    """
    Convert a raw binding value to a human-readable string for display.

    Mouse bindings are dicts like {"modifiers": ["Alt"], "mouse": "left"}.
    Keyboard bindings are plain strings like "W", "Shift".
    """
    if isinstance(binding, dict):
        mods = binding.get("modifiers", [])
        mouse = binding.get("mouse", "")
        scroll = binding.get("scroll", "")
        parts = mods + (
            [f"{mouse.title()} Mouse Button"] if mouse else []
        ) + (["Scroll Wheel"] if scroll else [])
        return " + ".join(parts) if parts else "(none)"
    return str(binding) if binding else "(none)"


# --------------------------------------------------------------------------- #
# Binding row widget                                                            #
# --------------------------------------------------------------------------- #


class _BindingRow(QWidget):
    """
    One row in the keybinding table: [Action label] [Binding pill] [✕] [↺] [✎]

    Signals:
        clear_requested(action_key) — user pressed ✕
        reset_requested(action_key) — user pressed ↺ (restore factory default)
        edit_requested(action_key)  — user pressed ✎ (enter capture mode)
    """

    clear_requested = Signal(str)
    reset_requested = Signal(str)
    edit_requested = Signal(str)

    def __init__(
        self,
        action_key: str,
        action_label: str,
        current_binding,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._action_key = action_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        # Action label (left-aligned, fixed width)
        self._label = QLabel(action_label)
        self._label.setMinimumWidth(120)
        self._label.setObjectName("binding_action_label")
        layout.addWidget(self._label)

        # Binding pill (shows current binding)
        self._pill = QLabel()
        self._pill.setObjectName("binding_pill")
        self._pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pill.setMinimumWidth(180)
        self._pill.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.set_binding(current_binding)
        layout.addWidget(self._pill)

        # ✕ clear button
        clear_btn = QPushButton("✕")
        clear_btn.setObjectName("binding_icon_btn")
        clear_btn.setFixedSize(24, 24)
        clear_btn.setToolTip("Clear binding")
        clear_btn.clicked.connect(lambda: self.clear_requested.emit(self._action_key))
        layout.addWidget(clear_btn)

        # ↺ reset button
        reset_btn = QPushButton("↺")
        reset_btn.setObjectName("binding_icon_btn")
        reset_btn.setFixedSize(24, 24)
        reset_btn.setToolTip("Reset to default")
        reset_btn.clicked.connect(lambda: self.reset_requested.emit(self._action_key))
        layout.addWidget(reset_btn)

        # ✎ edit / rebind button
        edit_btn = QPushButton("✎")
        edit_btn.setObjectName("binding_icon_btn")
        edit_btn.setFixedSize(24, 24)
        edit_btn.setToolTip("Rebind")
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(self._action_key))
        layout.addWidget(edit_btn)

    def set_binding(self, binding) -> None:
        """Update the displayed binding pill text."""
        self._pill.setText(_binding_to_label(self._action_key, binding))

    def set_capturing(self, capturing: bool) -> None:
        """Show 'Press a key or button...' while waiting for input."""
        if capturing:
            self._pill.setText("Press a key / click a mouse button…")
        # When not capturing, caller should call set_binding() to restore text


# --------------------------------------------------------------------------- #
# Panel pages inside the controls dialog                                        #
# --------------------------------------------------------------------------- #


class _MouseControlsPage(QWidget):
    """The Mouse Controls panel — one _BindingRow per mouse action."""

    bindings_changed = Signal()

    def __init__(
        self, camera: ViewportCamera, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self._rows: dict[str, _BindingRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        header = QLabel("Mouse Controls")
        header.setObjectName("controls_section_header")
        layout.addWidget(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("controls_divider")
        layout.addWidget(divider)

        for action_key, action_label in MOUSE_ACTIONS:
            current = self._camera.settings.bindings.get(action_key)
            row = _BindingRow(action_key, action_label, current)
            row.clear_requested.connect(self._on_clear)
            row.reset_requested.connect(self._on_reset)
            row.edit_requested.connect(self._on_edit)
            layout.addWidget(row)
            self._rows[action_key] = row

        layout.addStretch()
        self._capturing: str | None = None  # key being captured

    def refresh(self) -> None:
        """Refresh all pills from the current camera settings."""
        for key, row in self._rows.items():
            row.set_binding(self._camera.settings.bindings.get(key))

    def _on_clear(self, action_key: str) -> None:
        bindings = dict(self._camera.settings.bindings)
        bindings[action_key] = {}
        self._camera.apply_bindings(bindings, scheme="Custom")
        self._rows[action_key].set_binding({})
        self.bindings_changed.emit()

    def _on_reset(self, action_key: str) -> None:
        bindings = dict(self._camera.settings.bindings)
        bindings[action_key] = DEFAULT_BINDINGS[action_key]
        self._camera.apply_bindings(bindings, scheme="Custom")
        self._rows[action_key].set_binding(DEFAULT_BINDINGS[action_key])
        self.bindings_changed.emit()

    def _on_edit(self, action_key: str) -> None:
        # For mouse bindings the dialog just shows a message — full capture
        # mode requires a native event filter which is beyond scope for now.
        self._rows[action_key].set_capturing(True)
        self._capturing = action_key
        log.debug("Mouse binding capture not yet implemented for %s", action_key)
        # Immediately cancel capture for now (future work: native event filter)
        self._rows[action_key].set_binding(
            self._camera.settings.bindings.get(action_key)
        )
        self._capturing = None


class _KeyboardControlsPage(QWidget):
    """The Keyboard Controls panel — one _BindingRow per keyboard action."""

    bindings_changed = Signal()

    def __init__(
        self, camera: ViewportCamera, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self._rows: dict[str, _BindingRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        header = QLabel("Keyboard Controls")
        header.setObjectName("controls_section_header")
        layout.addWidget(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("controls_divider")
        layout.addWidget(divider)

        for action_key, action_label in KEY_ACTIONS:
            current = self._camera.settings.bindings.get(action_key)
            row = _BindingRow(action_key, action_label, current)
            row.clear_requested.connect(self._on_clear)
            row.reset_requested.connect(self._on_reset)
            row.edit_requested.connect(self._on_edit)
            layout.addWidget(row)
            self._rows[action_key] = row

        layout.addStretch()

    def refresh(self) -> None:
        """Refresh all pills from the current camera settings."""
        for key, row in self._rows.items():
            row.set_binding(self._camera.settings.bindings.get(key))

    def _on_clear(self, action_key: str) -> None:
        bindings = dict(self._camera.settings.bindings)
        bindings[action_key] = ""
        self._camera.apply_bindings(bindings, scheme="Custom")
        self._rows[action_key].set_binding("")
        self.bindings_changed.emit()

    def _on_reset(self, action_key: str) -> None:
        bindings = dict(self._camera.settings.bindings)
        bindings[action_key] = DEFAULT_BINDINGS[action_key]
        self._camera.apply_bindings(bindings, scheme="Custom")
        self._rows[action_key].set_binding(DEFAULT_BINDINGS[action_key])
        self.bindings_changed.emit()

    def _on_edit(self, action_key: str) -> None:
        # Keyboard capture: grab next key press via a temporary dialog
        capture_dlg = _KeyCaptureDialog(action_key, self)
        if capture_dlg.exec() == QDialog.DialogCode.Accepted:
            new_key = capture_dlg.captured_key
            if new_key:
                bindings = dict(self._camera.settings.bindings)
                bindings[action_key] = new_key
                self._camera.apply_bindings(bindings, scheme="Custom")
                self._rows[action_key].set_binding(new_key)
                self.bindings_changed.emit()


class _KeyCaptureDialog(QDialog):
    """
    Tiny modal dialog that grabs the next keypress and returns it.

    Shown when the user presses ✎ on a keyboard binding. The user presses
    any key and the dialog closes, storing the key name as a string.
    """

    def __init__(self, action_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rebind")
        self.setFixedSize(300, 120)
        self.captured_key: str = ""

        layout = QVBoxLayout(self)
        msg = QLabel(f"Press a key to bind to:\n{action_label}")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg)

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        layout.addWidget(cancel)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        # Map Qt key code back to a human-readable string
        text = event.text().upper().strip()
        if not text:
            # Special keys
            special = {
                Qt.Key.Key_Shift: "Shift",
                Qt.Key.Key_Control: "Ctrl",
                Qt.Key.Key_Alt: "Alt",
                Qt.Key.Key_Space: "Space",
                Qt.Key.Key_Escape: "Escape",
                Qt.Key.Key_Tab: "Tab",
            }
            text = special.get(key, "")
        if text:
            self.captured_key = text
            self.accept()


class _MovementSpeedPage(QWidget):
    """
    The Movement Speed panel — sliders for sensitivity, speed, scroll, invert Y.
    """

    def __init__(
        self, camera: ViewportCamera, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._camera = camera

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Movement Speed")
        header.setObjectName("controls_section_header")
        layout.addWidget(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("controls_divider")
        layout.addWidget(divider)

        # Mouse sensitivity
        self._mouse_sensitivity = self._add_double_spinbox(
            layout,
            "Mouse Sensitivity",
            camera.settings.mouse_sensitivity,
            0.1, 2.0, 0.1,
        )

        # Keyboard speed
        self._keyboard_speed = self._add_double_spinbox(
            layout,
            "Keyboard Speed",
            camera.settings.keyboard_speed,
            1.0, 20.0, 0.5,
        )

        # Scroll speed
        self._scroll_speed = self._add_double_spinbox(
            layout,
            "Scroll Speed",
            camera.settings.scroll_speed,
            0.1, 5.0, 0.1,
        )

        # Invert Y checkbox
        self._invert_y = QCheckBox("Invert Y Axis")
        self._invert_y.setChecked(camera.settings.invert_y)
        layout.addWidget(self._invert_y)

        layout.addStretch()

    def _add_double_spinbox(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        step: float,
    ) -> QDoubleSpinBox:
        """Add a labeled spinbox row and return the spinbox widget."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(140)
        row.addWidget(lbl)
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(value)
        spinbox.setDecimals(1)
        row.addWidget(spinbox)
        parent_layout.addLayout(row)
        return spinbox

    def apply_to_camera(self) -> None:
        """Write current widget values back to the camera settings object."""
        s = self._camera.settings
        s.mouse_sensitivity = self._mouse_sensitivity.value()
        s.keyboard_speed = self._keyboard_speed.value()
        s.scroll_speed = self._scroll_speed.value()
        s.invert_y = self._invert_y.isChecked()
        self._camera.save_settings()


# --------------------------------------------------------------------------- #
# Controls dialog                                                               #
# --------------------------------------------------------------------------- #


class ViewportControlsDialog(QDialog):
    """
    Full keybinding and speed customizer dialog.

    Two-column layout: left = category list (Mouse / Keyboard / Movement Speed),
    right = QStackedWidget showing the selected category's controls.

    Footer: Save Changes button + Close button.
    """

    def __init__(
        self, camera: ViewportCamera, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self.setWindowTitle("Viewport Controls")
        self.setMinimumSize(680, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ------------------------------------------------------------------ #
        # Two-column body (category list | content panel)                     #
        # ------------------------------------------------------------------ #
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left: category list
        self._category_list = QListWidget()
        self._category_list.setObjectName("controls_category_list")
        self._category_list.setFixedWidth(160)
        self._category_list.setFrameShape(QFrame.Shape.NoFrame)
        for cat in CATEGORIES:
            item = QListWidgetItem(cat)
            self._category_list.addItem(item)
        self._category_list.setCurrentRow(0)
        self._category_list.currentRowChanged.connect(self._on_category_changed)
        body.addWidget(self._category_list)

        # Vertical divider between list and content
        v_divider = QFrame()
        v_divider.setFrameShape(QFrame.Shape.VLine)
        v_divider.setObjectName("controls_divider")
        body.addWidget(v_divider)

        # Right: stacked content pages
        self._stack = QStackedWidget()

        self._mouse_page = _MouseControlsPage(camera)
        self._stack.addWidget(self._mouse_page)

        self._keyboard_page = _KeyboardControlsPage(camera)
        self._stack.addWidget(self._keyboard_page)

        self._speed_page = _MovementSpeedPage(camera)
        self._stack.addWidget(self._speed_page)

        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        # ------------------------------------------------------------------ #
        # Footer                                                               #
        # ------------------------------------------------------------------ #
        footer_divider = QFrame()
        footer_divider.setFrameShape(QFrame.Shape.HLine)
        footer_divider.setObjectName("controls_divider")
        root.addWidget(footer_divider)

        footer = QHBoxLayout()
        footer.setContentsMargins(12, 8, 12, 8)
        footer.addStretch()

        save_btn = QPushButton("Save Changes")
        save_btn.setObjectName("primary_button")
        save_btn.clicked.connect(self._save)
        footer.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)

        root.addLayout(footer)

    def _on_category_changed(self, row: int) -> None:
        self._stack.setCurrentIndex(row)

    def _save(self) -> None:
        """Write speed settings and close."""
        self._speed_page.apply_to_camera()
        self._camera.save_settings()
        self.accept()


# --------------------------------------------------------------------------- #
# Tools panel                                                                   #
# --------------------------------------------------------------------------- #


class ViewportToolsPanel(QWidget):
    """
    200px fixed-width panel on the RIGHT side of the Viewport page.

    Contains:
    - Tool selector buttons (Brush / Region, mutually exclusive)
    - Color swatch button → QColorDialog
    - Size slider (brush radius 1–64 px)
    - Opacity slider (1–100%)
    - ⚙ Controls button → ViewportControlsDialog

    Signals:
        tool_changed(str)         — "brush" or "region"
        color_changed(tuple)      — (R, G, B) int tuple
        brush_size_changed(int)   — radius in pixels
        opacity_changed(float)    — 0.0–1.0
    """

    tool_changed = Signal(str)
    color_changed = Signal(tuple)
    brush_size_changed = Signal(int)
    opacity_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewport_tools_panel")
        self.setFixedWidth(200)

        # Camera reference — set by ViewportView when the mesh is loaded
        self._camera: ViewportCamera | None = None

        self._current_color: tuple = (220, 100, 60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)

        # ------------------------------------------------------------------ #
        # Section: Tools                                                       #
        # ------------------------------------------------------------------ #
        tools_header = QLabel("Tools")
        tools_header.setObjectName("tools_section_header")
        layout.addWidget(tools_header)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)

        # Mutually exclusive checkable tool buttons
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._brush_btn = QPushButton("🖌")
        self._brush_btn.setObjectName("tool_button")
        self._brush_btn.setCheckable(True)
        self._brush_btn.setChecked(True)
        self._brush_btn.setToolTip("Brush — paint on the mesh surface")
        self._brush_btn.setFixedSize(40, 40)
        self._btn_group.addButton(self._brush_btn)
        tool_row.addWidget(self._brush_btn)

        self._region_btn = QPushButton("⬤")
        self._region_btn.setObjectName("tool_button")
        self._region_btn.setCheckable(True)
        self._region_btn.setToolTip("Region Select — flood-fill a connected region")
        self._region_btn.setFixedSize(40, 40)
        self._btn_group.addButton(self._region_btn)
        tool_row.addWidget(self._region_btn)

        tool_row.addStretch()
        layout.addLayout(tool_row)

        self._brush_btn.toggled.connect(
            lambda checked: self.tool_changed.emit("brush") if checked else None
        )
        self._region_btn.toggled.connect(
            lambda checked: self.tool_changed.emit("region") if checked else None
        )

        # ------------------------------------------------------------------ #
        # Divider                                                              #
        # ------------------------------------------------------------------ #
        div1 = QFrame()
        div1.setFrameShape(QFrame.Shape.HLine)
        div1.setObjectName("tools_divider")
        layout.addWidget(div1)

        # ------------------------------------------------------------------ #
        # Color swatch                                                         #
        # ------------------------------------------------------------------ #
        color_label = QLabel("Color")
        color_label.setObjectName("tools_label")
        layout.addWidget(color_label)

        self._color_btn = QPushButton()
        self._color_btn.setObjectName("color_swatch")
        self._color_btn.setMinimumHeight(30)
        self._color_btn.setToolTip("Click to choose paint color")
        self._color_btn.clicked.connect(self._pick_color)
        self._update_color_swatch()
        layout.addWidget(self._color_btn)

        # ------------------------------------------------------------------ #
        # Size slider                                                          #
        # ------------------------------------------------------------------ #
        size_row = QHBoxLayout()
        size_lbl = QLabel("Size")
        size_lbl.setObjectName("tools_label")
        size_lbl.setMinimumWidth(50)
        size_row.addWidget(size_lbl)

        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 64)
        self._size_slider.setValue(12)
        size_row.addWidget(self._size_slider)

        self._size_readout = QLabel("12")
        self._size_readout.setObjectName("tools_readout")
        self._size_readout.setMinimumWidth(24)
        self._size_readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        size_row.addWidget(self._size_readout)
        layout.addLayout(size_row)

        self._size_slider.valueChanged.connect(self._on_size_changed)

        # ------------------------------------------------------------------ #
        # Opacity slider                                                       #
        # ------------------------------------------------------------------ #
        opacity_row = QHBoxLayout()
        opacity_lbl = QLabel("Opacity")
        opacity_lbl.setObjectName("tools_label")
        opacity_lbl.setMinimumWidth(50)
        opacity_row.addWidget(opacity_lbl)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(1, 100)
        self._opacity_slider.setValue(100)
        opacity_row.addWidget(self._opacity_slider)

        self._opacity_readout = QLabel("100%")
        self._opacity_readout.setObjectName("tools_readout")
        self._opacity_readout.setMinimumWidth(30)
        self._opacity_readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        opacity_row.addWidget(self._opacity_readout)
        layout.addLayout(opacity_row)

        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)

        # ------------------------------------------------------------------ #
        # Spacer                                                               #
        # ------------------------------------------------------------------ #
        layout.addStretch()

        # ------------------------------------------------------------------ #
        # Divider above gear button                                            #
        # ------------------------------------------------------------------ #
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setObjectName("tools_divider")
        layout.addWidget(div2)

        # ------------------------------------------------------------------ #
        # ⚙ Controls button                                                   #
        # ------------------------------------------------------------------ #
        controls_btn = QPushButton("⚙  Controls")
        controls_btn.setObjectName("controls_gear_btn")
        controls_btn.setToolTip("Open viewport navigation controls settings")
        controls_btn.clicked.connect(self._open_controls_dialog)
        layout.addWidget(controls_btn)

    # ------------------------------------------------------------------ #
    # Camera reference                                                     #
    # ------------------------------------------------------------------ #

    def set_camera(self, camera: ViewportCamera) -> None:
        """
        Called by ViewportView after a mesh is loaded to connect the camera
        to the tools panel (needed by the controls dialog).
        """
        self._camera = camera

    # ------------------------------------------------------------------ #
    # Slot handlers                                                        #
    # ------------------------------------------------------------------ #

    def _pick_color(self) -> None:
        """Open QColorDialog and emit color_changed on selection."""
        initial = QColor(*self._current_color)
        color = QColorDialog.getColor(initial, self, "Pick Paint Color")
        if color.isValid():
            self._current_color = (color.red(), color.green(), color.blue())
            self._update_color_swatch()
            self.color_changed.emit(self._current_color)

    def _update_color_swatch(self) -> None:
        """Set the button background to the current color."""
        r, g, b = self._current_color
        self._color_btn.setStyleSheet(
            f"QPushButton#color_swatch {{ background-color: rgb({r},{g},{b}); }}"
        )

    def _on_size_changed(self, value: int) -> None:
        self._size_readout.setText(str(value))
        self.brush_size_changed.emit(value)

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_readout.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)

    def _open_controls_dialog(self) -> None:
        """Open the keybinding controls dialog."""
        if self._camera is None:
            # No mesh loaded yet — create a temporary camera just to show the dialog
            cam = ViewportCamera()
            cam.load_settings()
        else:
            cam = self._camera
        dlg = ViewportControlsDialog(cam, self)
        dlg.exec()
