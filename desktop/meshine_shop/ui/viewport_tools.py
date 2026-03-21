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
]

KEY_ACTIONS = [
    ("forward",  "Move Forward"),
    ("backward", "Move Backward"),
    ("left",     "Move Left"),
    ("right",    "Move Right"),
    ("up",       "Move Up"),
    ("down",     "Move Down"),
    ("frame",    "Frame Mesh"),
]

# Category names shown in the left list of the controls dialog
CATEGORIES = ["Mouse Controls", "Keyboard Controls", "Movement Speed"]


def _binding_to_label(key: str, binding) -> str:
    """
    Convert a raw binding value to a human-readable string for display.

    Mouse bindings are dicts like {"modifiers": ["Alt"], "mouse": "left"} or
    chord bindings like {"modifiers": [], "mouse": "left+right"}.
    Keyboard bindings are plain strings like "W", "Shift".
    """
    if isinstance(binding, dict):
        mods = binding.get("modifiers", [])
        mouse = binding.get("mouse", "")
        scroll = binding.get("scroll", "")
        # Support multi-button chords stored as "left+right" etc.
        btn_parts = (
            [f"{b.title()} Mouse Button" for b in mouse.split("+")]
            if mouse else []
        )
        parts = mods + btn_parts + (["Scroll Wheel"] if scroll else [])
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
        # Open the mouse capture dialog and wait for a click or scroll.
        capture_dlg = _MouseCaptureDialog(action_key, self)
        if capture_dlg.exec() == QDialog.DialogCode.Accepted:
            new_binding = capture_dlg.captured_binding
            if new_binding:
                bindings = dict(self._camera.settings.bindings)
                bindings[action_key] = new_binding
                self._camera.apply_bindings(bindings, scheme="Custom")
                self._rows[action_key].set_binding(new_binding)
                self.bindings_changed.emit()


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


class _MouseCaptureDialog(QDialog):
    """
    Modal dialog that captures mouse button chords and scroll gestures.

    Uses a QApplication event filter to intercept mouse events from all
    child widgets (not just the dialog background), fixing the bug where
    clicking on a label or empty area would be silently ignored.

    Chord detection: the user holds one or more buttons simultaneously.
    The status label updates live showing what's held. On release of all
    buttons the chord is confirmed and the dialog closes. This lets the
    user bind e.g. LMB+RMB by pressing both then releasing.

    Stored binding format:
        Single button: {"modifiers": [...], "mouse": "left"}
        Chord:         {"modifiers": [...], "mouse": "left+right"}
        Scroll:        {"modifiers": [...], "scroll": "y"}
    """

    from PySide6.QtWidgets import QApplication as _QApp

    # Map Qt mouse button enum values to binding string names
    _BUTTON_NAMES = {
        Qt.MouseButton.LeftButton:   "left",
        Qt.MouseButton.MiddleButton: "middle",
        Qt.MouseButton.RightButton:  "right",
    }

    def __init__(self, action_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rebind Mouse")
        self.setFixedSize(340, 155)
        self.captured_binding: dict = {}

        # Set of button names ("left", "right", etc.) currently held down
        self._btns_held: set[str] = set()
        # Binding built from the current held state (confirmed on release)
        self._pending: dict = {}

        layout = QVBoxLayout(self)

        # Instruction
        instr = QLabel(f"Hold button(s) then release to bind to:\n{action_label}")
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instr)

        # Live status showing what's currently held
        self._status = QLabel("(waiting for input…)")
        self._status.setObjectName("binding_pill")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        # Cancel — its clicked signal fires on release inside the button,
        # before our eventFilter sees the release, so reject() is called first
        # and the dialog is already closed when we check _btns_held.
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self._cancel_btn)

        # Install globally so we catch events on child widgets too
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    def done(self, result: int) -> None:
        """Remove the app-wide event filter before closing."""
        from PySide6.QtWidgets import QApplication
        QApplication.instance().removeEventFilter(self)
        super().done(result)

    def _in_dialog(self, widget) -> bool:
        """Return True if widget is this dialog or a descendant of it."""
        w = widget
        while w is not None:
            if w is self:
                return True
            w = w.parent()
        return False

    def _active_modifiers(self) -> list[str]:
        """Return currently held modifier key names."""
        from PySide6.QtWidgets import QApplication
        mods = QApplication.keyboardModifiers()
        result = []
        if mods & Qt.KeyboardModifier.AltModifier:
            result.append("Alt")
        if mods & Qt.KeyboardModifier.ControlModifier:
            result.append("Ctrl")
        if mods & Qt.KeyboardModifier.ShiftModifier:
            result.append("Shift")
        return result

    def _update_status(self) -> None:
        """Refresh the status label and rebuild _pending from current state."""
        if not self._btns_held:
            self._status.setText("(waiting for input…)")
            self._pending = {}
            return
        mods = self._active_modifiers()
        # Sort for consistent display and storage ordering
        btn_parts = [f"{b.title()} Mouse Button" for b in sorted(self._btns_held)]
        self._status.setText(" + ".join(mods + btn_parts))
        self._pending = {
            "modifiers": mods,
            "mouse": "+".join(sorted(self._btns_held)),
        }

    def eventFilter(self, obj, event) -> bool:
        """
        Intercept mouse press/release/wheel events from anywhere inside the
        dialog so we catch clicks on labels and empty areas, not just the
        dialog background.
        """
        from PySide6.QtCore import QEvent

        if not self._in_dialog(obj):
            return False  # ignore events from outside this dialog

        t = event.type()

        if t == QEvent.Type.MouseButtonPress:
            btn_name = self._BUTTON_NAMES.get(event.button())
            if btn_name:
                # Skip if the press is on the Cancel button — let it reject normally
                if obj is self._cancel_btn:
                    return False
                self._btns_held.add(btn_name)
                self._update_status()
            return False  # don't consume — let normal Qt routing continue

        if t == QEvent.Type.MouseButtonRelease:
            btn_name = self._BUTTON_NAMES.get(event.button())
            if btn_name and btn_name in self._btns_held:
                self._btns_held.discard(btn_name)
                if not self._btns_held and self._pending:
                    # All buttons released — confirm the chord
                    self.captured_binding = self._pending
                    self.accept()
            return False

        if t == QEvent.Type.Wheel:
            self.captured_binding = {
                "modifiers": self._active_modifiers(),
                "scroll": "y",
            }
            self.accept()
            return True  # consume the scroll so it doesn't scroll the dialog

        return False


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

        # Invert axis checkboxes — laid out side-by-side
        invert_row = QHBoxLayout()
        self._invert_x = QCheckBox("Invert X Axis")
        self._invert_x.setChecked(camera.settings.invert_x)
        self._invert_y = QCheckBox("Invert Y Axis")
        self._invert_y.setChecked(camera.settings.invert_y)
        invert_row.addWidget(self._invert_x)
        invert_row.addWidget(self._invert_y)
        invert_row.addStretch()
        layout.addLayout(invert_row)

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
        s.invert_x = self._invert_x.isChecked()
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
    # Emitted when the user clicks the "Reset Rotation" button
    reset_rotation_requested = Signal()
    # Emitted when the user clicks "Normalize Scale"
    normalize_scale_requested = Signal()
    # Emitted when Undo / Redo buttons are clicked
    undo_requested = Signal()
    redo_requested = Signal()
    # Emitted when the sculpt brush radius or strength sliders change
    sculpt_radius_changed = Signal(float)
    sculpt_strength_changed = Signal(float)

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

        # --- Tool row 1: Polygon Select | Flatten | Inflate ---
        # Column layout: Poly/Rotate | Flatten/Smooth | Inflate/Deflate
        tool_row1 = QHBoxLayout()
        tool_row1.setSpacing(6)

        # Polygon select tool — click to place anchor points defining a custom boundary
        self._poly_btn = QPushButton("⬡")
        self._poly_btn.setObjectName("tool_button")
        self._poly_btn.setCheckable(True)
        self._poly_btn.setToolTip("Polygon Select — click to place anchor points")
        self._poly_btn.setFixedSize(54, 54)
        tool_row1.addWidget(self._poly_btn)

        # Flatten: project vertices onto best-fit plane (PCA) — sits above Smooth
        self._flatten_btn = QPushButton("⊥")
        self._flatten_btn.setObjectName("tool_button")
        self._flatten_btn.setCheckable(True)
        self._flatten_btn.setToolTip("Flatten — level surface to best-fit plane")
        self._flatten_btn.setFixedSize(54, 54)
        tool_row1.addWidget(self._flatten_btn)

        # Inflate: push vertices outward along their normals — sits above Deflate
        self._inflate_btn = QPushButton("▲")
        self._inflate_btn.setObjectName("tool_button")
        self._inflate_btn.setCheckable(True)
        self._inflate_btn.setToolTip("Inflate — push surface outward along normals")
        self._inflate_btn.setFixedSize(54, 54)
        tool_row1.addWidget(self._inflate_btn)

        tool_row1.addStretch()
        layout.addLayout(tool_row1)

        # --- Tool row 2: Rotate | Smooth | Deflate ---
        tool_row2 = QHBoxLayout()
        tool_row2.setSpacing(6)

        # Rotate tool — sits below Polygon Select
        self._rotate_btn = QPushButton("⟳")
        self._rotate_btn.setObjectName("tool_button")
        self._rotate_btn.setCheckable(True)
        self._rotate_btn.setToolTip("Rotate — drag the X/Y/Z rings to rotate the mesh")
        self._rotate_btn.setFixedSize(54, 54)
        tool_row2.addWidget(self._rotate_btn)

        # Smooth: Laplacian vertex averaging — sits below Flatten
        self._smooth_btn = QPushButton("≋")
        self._smooth_btn.setObjectName("tool_button")
        self._smooth_btn.setCheckable(True)
        self._smooth_btn.setToolTip("Smooth — average vertex positions (Laplacian)")
        self._smooth_btn.setFixedSize(54, 54)
        tool_row2.addWidget(self._smooth_btn)

        # Deflate: pull vertices inward along their normals — sits below Inflate
        self._deflate_btn = QPushButton("▼")
        self._deflate_btn.setObjectName("tool_button")
        self._deflate_btn.setCheckable(True)
        self._deflate_btn.setToolTip("Deflate — pull surface inward along normals")
        self._deflate_btn.setFixedSize(54, 54)
        tool_row2.addWidget(self._deflate_btn)

        tool_row2.addStretch()
        layout.addLayout(tool_row2)

        # Sculpt strength slider
        str_row = QHBoxLayout()
        str_lbl = QLabel("Strength")
        str_lbl.setObjectName("tools_label")
        str_lbl.setMinimumWidth(50)
        str_row.addWidget(str_lbl)

        # Range 1–100, maps to 0.001–0.05 world-space displacement strength
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(1, 100)
        self._strength_slider.setValue(20)    # default: 0.010
        str_row.addWidget(self._strength_slider)

        self._strength_readout = QLabel("0.010")
        self._strength_readout.setObjectName("tools_readout")
        self._strength_readout.setMinimumWidth(36)
        self._strength_readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        str_row.addWidget(self._strength_readout)
        layout.addLayout(str_row)

        self._strength_slider.valueChanged.connect(self._on_strength_changed)

        # Sculpt radius slider
        rad_row = QHBoxLayout()
        rad_lbl = QLabel("Radius")
        rad_lbl.setObjectName("tools_label")
        rad_lbl.setMinimumWidth(50)
        rad_row.addWidget(rad_lbl)

        # Range 1–100, maps to 0.005–0.50 world-space radius
        self._radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._radius_slider.setRange(1, 100)
        self._radius_slider.setValue(30)      # default: 0.05
        rad_row.addWidget(self._radius_slider)

        self._radius_readout = QLabel("0.050")
        self._radius_readout.setObjectName("tools_readout")
        self._radius_readout.setMinimumWidth(36)
        self._radius_readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        rad_row.addWidget(self._radius_readout)
        layout.addLayout(rad_row)

        self._radius_slider.valueChanged.connect(self._on_radius_changed)

        # Wire click handlers — each button manages mutual exclusivity and
        # emits tool_changed("") when toggled off so the viewport knows no
        # tool is active. Brush and region are wired below where their buttons
        # are created (in the color section).
        self._poly_btn.clicked.connect(self._on_poly_clicked)
        self._rotate_btn.clicked.connect(self._on_rotate_clicked)

        # Wire sculpt button click handlers
        self._inflate_btn.clicked.connect(self._on_inflate_clicked)
        self._deflate_btn.clicked.connect(self._on_deflate_clicked)
        self._smooth_btn.clicked.connect(self._on_smooth_clicked)
        self._flatten_btn.clicked.connect(self._on_flatten_clicked)

        # ------------------------------------------------------------------ #
        # Divider                                                              #
        # ------------------------------------------------------------------ #
        div1 = QFrame()
        div1.setFrameShape(QFrame.Shape.HLine)
        div1.setObjectName("tools_divider")
        layout.addWidget(div1)

        # ------------------------------------------------------------------ #
        # Brush / Region Select — live in the color section since they use   #
        # the color, size, and opacity controls directly below.              #
        # ------------------------------------------------------------------ #
        brush_row = QHBoxLayout()
        brush_row.setSpacing(6)

        # Checkable tool buttons — manually exclusive (same logic as other tools).
        self._brush_btn = QPushButton("🖌")
        self._brush_btn.setObjectName("tool_button")
        self._brush_btn.setCheckable(True)
        self._brush_btn.setToolTip("Brush — paint on the mesh surface")
        self._brush_btn.setFixedSize(54, 54)
        brush_row.addWidget(self._brush_btn)

        self._region_btn = QPushButton("⬤")
        self._region_btn.setObjectName("tool_button")
        self._region_btn.setCheckable(True)
        self._region_btn.setToolTip("Region Select — flood-fill a connected region")
        self._region_btn.setFixedSize(54, 54)
        brush_row.addWidget(self._region_btn)

        # Erase tool — restores original baked texture pixels within the brush radius
        self._erase_btn = QPushButton("◌")
        self._erase_btn.setObjectName("tool_button")
        self._erase_btn.setCheckable(True)
        self._erase_btn.setToolTip("Erase — restore original texture under the brush")
        self._erase_btn.setFixedSize(54, 54)
        brush_row.addWidget(self._erase_btn)

        # Wire brush/region/erase click handlers here since buttons are defined here
        self._brush_btn.clicked.connect(self._on_brush_clicked)
        self._region_btn.clicked.connect(self._on_region_clicked)
        self._erase_btn.clicked.connect(self._on_erase_clicked)

        brush_row.addStretch()
        layout.addLayout(brush_row)

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
        # Undo / Redo buttons — side by side, same style as Reset Rotation.   #
        # Disabled until there is history to step through; enabled/disabled   #
        # by update_undo_redo_state() called from ViewportWidget after each   #
        # operation.                                                           #
        # ------------------------------------------------------------------ #
        undo_redo_row = QHBoxLayout()
        undo_redo_row.setSpacing(6)

        self._undo_btn = QPushButton("↩  Undo")
        self._undo_btn.setObjectName("reset_rotation_btn")  # reuse same QSS style
        self._undo_btn.setToolTip("Undo the last edit  (Cmd+Z)")
        self._undo_btn.setEnabled(False)
        # NoFocus prevents the macOS focus ring from staying on after a click
        self._undo_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        undo_redo_row.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("↪  Redo")
        self._redo_btn.setObjectName("reset_rotation_btn")  # reuse same QSS style
        self._redo_btn.setToolTip("Redo the last undone edit  (Cmd+Shift+Z)")
        self._redo_btn.setEnabled(False)
        self._redo_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._redo_btn.clicked.connect(self.redo_requested.emit)
        undo_redo_row.addWidget(self._redo_btn)

        layout.addLayout(undo_redo_row)

        # ------------------------------------------------------------------ #
        # Reset Rotation button (always visible; no-op when not rotated)      #
        # ------------------------------------------------------------------ #
        self._reset_rot_btn = QPushButton("↺  Reset Rotation")
        self._reset_rot_btn.setObjectName("reset_rotation_btn")
        self._reset_rot_btn.setToolTip(
            "Snap the mesh back to its original orientation"
        )
        self._reset_rot_btn.clicked.connect(self.reset_rotation_requested.emit)
        layout.addWidget(self._reset_rot_btn)

        # ------------------------------------------------------------------ #
        # Normalize Scale button — rescales mesh so longest axis = 1.0 unit. #
        # Ensures consistent scale across export targets (UE5, Blender,       #
        # web viewers, AR/VR platforms) that do not handle scale on import.   #
        # ------------------------------------------------------------------ #
        self._normalize_btn = QPushButton("⤢  Normalize Scale")
        self._normalize_btn.setObjectName("reset_rotation_btn")  # reuse QSS style
        self._normalize_btn.setToolTip(
            "Rescale mesh so its longest axis = 1.0 unit\n"
            "Recommended before exporting to web, AR/VR, or any platform\n"
            "that does not handle scale at import."
        )
        self._normalize_btn.clicked.connect(self.normalize_scale_requested.emit)
        layout.addWidget(self._normalize_btn)

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
    # Tool toggle handlers                                                 #
    # ------------------------------------------------------------------ #

    def _on_brush_clicked(self) -> None:
        """Toggle brush on/off; deactivate all other tools if active."""
        if self._brush_btn.isChecked():
            self._deactivate_all_except(self._brush_btn)
            self.tool_changed.emit("brush")
        else:
            self.tool_changed.emit("")

    def _on_region_clicked(self) -> None:
        """Toggle region select on/off; deactivate all other tools if active."""
        if self._region_btn.isChecked():
            self._deactivate_all_except(self._region_btn)
            self.tool_changed.emit("region")
        else:
            self.tool_changed.emit("")

    def _on_poly_clicked(self) -> None:
        """Toggle polygon select on/off; deactivate all other tools if active."""
        if self._poly_btn.isChecked():
            self._deactivate_all_except(self._poly_btn)
            self.tool_changed.emit("poly_select")
        else:
            self.tool_changed.emit("")

    def _on_rotate_clicked(self) -> None:
        """Toggle rotate on/off; deactivate all other tools if active."""
        if self._rotate_btn.isChecked():
            self._deactivate_all_except(self._rotate_btn)
            self.tool_changed.emit("rotate")
        else:
            self.tool_changed.emit("")

    def _on_inflate_clicked(self) -> None:
        """Toggle inflate sculpt brush; deactivate all other tools."""
        if self._inflate_btn.isChecked():
            self._deactivate_all_except(self._inflate_btn)
            self.tool_changed.emit("inflate")
        else:
            self.tool_changed.emit("")

    def _on_deflate_clicked(self) -> None:
        """Toggle deflate sculpt brush; deactivate all other tools."""
        if self._deflate_btn.isChecked():
            self._deactivate_all_except(self._deflate_btn)
            self.tool_changed.emit("deflate")
        else:
            self.tool_changed.emit("")

    def _on_smooth_clicked(self) -> None:
        """Toggle smooth sculpt brush; deactivate all other tools."""
        if self._smooth_btn.isChecked():
            self._deactivate_all_except(self._smooth_btn)
            self.tool_changed.emit("smooth")
        else:
            self.tool_changed.emit("")

    def _on_flatten_clicked(self) -> None:
        """Toggle flatten sculpt brush; deactivate all other tools."""
        if self._flatten_btn.isChecked():
            self._deactivate_all_except(self._flatten_btn)
            self.tool_changed.emit("flatten")
        else:
            self.tool_changed.emit("")

    def _on_erase_clicked(self) -> None:
        """Toggle erase on/off; deactivate all other tools if active."""
        if self._erase_btn.isChecked():
            self._deactivate_all_except(self._erase_btn)
            self.tool_changed.emit("erase")
        else:
            self.tool_changed.emit("")

    def _deactivate_all_except(self, keep: QPushButton) -> None:
        """Uncheck all tool buttons except the given one."""
        for btn in (
            self._brush_btn, self._region_btn, self._erase_btn,
            self._poly_btn, self._rotate_btn,
            self._inflate_btn, self._deflate_btn,
            self._smooth_btn, self._flatten_btn,
        ):
            if btn is not keep:
                btn.setChecked(False)

    def _on_strength_changed(self, value: int) -> None:
        """Map slider 1–100 → strength 0.001–0.05 and emit signal."""
        # Linear map: 1 → 0.001, 100 → 0.05
        strength = 0.001 + (value - 1) / 99.0 * (0.05 - 0.001)
        self._strength_readout.setText(f"{strength:.3f}")
        self.sculpt_strength_changed.emit(strength)

    def _on_radius_changed(self, value: int) -> None:
        """Map slider 1–100 → radius 0.005–0.50 and emit signal."""
        # Linear map: 1 → 0.005, 100 → 0.50
        radius = 0.005 + (value - 1) / 99.0 * (0.50 - 0.005)
        self._radius_readout.setText(f"{radius:.3f}")
        self.sculpt_radius_changed.emit(radius)

    # ------------------------------------------------------------------ #
    # Camera reference                                                     #
    # ------------------------------------------------------------------ #

    def set_camera(self, camera: ViewportCamera) -> None:
        """
        Called by ViewportView after a mesh is loaded to connect the camera
        to the tools panel (needed by the controls dialog).
        """
        self._camera = camera

    def update_undo_redo_state(self, can_undo: bool, can_redo: bool) -> None:
        """
        Enable or disable the Undo/Redo buttons to reflect current history depth.

        Called by ViewportWidget after every snapshot push, undo, or redo so
        the button enabled states always match what's actually possible.

        Parameters
        ----------
        can_undo : bool
            True when there is at least one step to undo.
        can_redo : bool
            True when there is at least one step to redo.
        """
        self._undo_btn.setEnabled(can_undo)
        self._redo_btn.setEnabled(can_redo)

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
