"""
Layers panel for the Meshine Shop viewport.

ViewportLayersPanel — the 200px fixed-width left-side panel on the Viewport
page. Displays saved polygon selections as named layer entries, each with
a visibility toggle, auto-assigned color swatch, and a delete button.

Workflow:
    1. User activates the polygon-select tool and draws a boundary on the mesh.
    2. ViewportWidget emits selection_ready(face_count).
    3. set_pending_selection(count) is called — enables the Save button and
       shows a "pending" indicator so the user knows there is an unsaved selection.
    4. User clicks "Save as Layer" — save_layer_requested signal is emitted with
       (name, color_rgba). The app layer calls viewport.save_pending_as_layer()
       and then calls add_layer() here to append a row.
    5. Each layer row: color swatch  |  name  |  eye toggle  |  delete btn
       Double-clicking the name enters an inline rename editor.
"""

from __future__ import annotations

import logging
import webbrowser

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

# Palette of RGBA highlight colors (0-1 range) cycled through as layers are created.
# Chosen to be visually distinct against the dark charcoal background.
_LAYER_COLORS: list[tuple[float, float, float, float]] = [
    (0.86, 0.27, 0.27, 0.40),  # crimson
    (0.27, 0.67, 0.86, 0.40),  # sky blue
    (0.27, 0.86, 0.47, 0.40),  # green
    (0.86, 0.67, 0.27, 0.40),  # amber
    (0.67, 0.27, 0.86, 0.40),  # violet
    (0.86, 0.47, 0.27, 0.40),  # orange
    (0.27, 0.86, 0.86, 0.40),  # teal
    (0.86, 0.27, 0.67, 0.40),  # pink
]


class _LayerRow(QWidget):
    """
    A single layer entry row in the layers list.

    Layout (left to right):
        [ color swatch ] [ name label / line edit ] [ eye btn ] [ delete btn ]

    Clicking anywhere on the row body (not the buttons) emits row_clicked so
    the panel can track the active layer and highlight this row.
    Double-clicking the name label enters an inline edit mode.

    The eye button (👁) replaces the old checkbox — it toggles model visibility
    and dims itself when hidden so the state is immediately obvious.
    """

    # Emitted when the eye button is toggled. True = visible.
    visibility_changed = Signal(bool)
    # Emitted when the user picks a new color via the swatch. RGBA float tuple.
    color_changed = Signal(tuple)
    # Emitted when the user renames the layer (new name string).
    renamed = Signal(str)
    # Emitted when the delete button is clicked.
    delete_requested = Signal()
    # Emitted when the row body is clicked (not the eye/delete buttons).
    row_clicked = Signal()

    def __init__(self, name: str, color: tuple[float, float, float, float]) -> None:
        super().__init__()
        self._name = name
        self._visible = True
        self._is_selected = False   # drives paintEvent background fill

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(8)

        # Color swatch — clickable square that opens a color picker.
        # The picked color updates both this swatch and the viewport overlay.
        r, g, b, _ = color
        self._swatch_color: tuple[float, float, float, float] = color  # current RGBA (0-1)
        self._swatch = QPushButton()
        self._swatch.setObjectName("layer_swatch")
        self._swatch.setFixedSize(14, 14)
        self._swatch.setToolTip("Click to change layer color")
        self._swatch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._swatch.clicked.connect(self._on_swatch_clicked)
        self._apply_swatch_color(r, g, b)
        layout.addWidget(self._swatch)

        # Name label — double-click to enter inline rename mode
        self._name_label = QLabel(name)
        self._name_label.setObjectName("layer_name_label")
        self._name_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._name_label.mouseDoubleClickEvent = lambda _: self._start_rename()
        layout.addWidget(self._name_label)

        # Inline rename editor — hidden until double-click activates it
        self._edit = QLineEdit(name)
        self._edit.setObjectName("layer_name_edit")
        self._edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._edit.hide()
        self._edit.returnPressed.connect(self._commit_rename)
        self._edit.editingFinished.connect(self._commit_rename)
        layout.addWidget(self._edit)

        # Eye button — toggles layer visibility on the model. Uses ◉ (FISHEYE)
        # rather than the 👁 emoji so it renders as a plain monochrome unicode
        # symbol consistent with the tool-panel button aesthetic. Dims when
        # hidden so the state is immediately clear without a separate label.
        self._eye_btn = QPushButton("◉")
        self._eye_btn.setObjectName("layer_eye_btn")
        self._eye_btn.setFixedSize(22, 22)
        self._eye_btn.setToolTip("Toggle layer visibility on model")
        self._eye_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._eye_btn.clicked.connect(self._on_eye_clicked)
        layout.addWidget(self._eye_btn)

        # Delete button — red × to remove this layer
        self._del_btn = QPushButton("×")
        self._del_btn.setObjectName("layer_delete_btn")
        self._del_btn.setFixedSize(18, 18)
        self._del_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(self._del_btn)

    def _apply_swatch_color(self, r: float, g: float, b: float) -> None:
        """Update the swatch button's background to the given 0-1 RGB values."""
        hex_color = QColor(int(r * 255), int(g * 255), int(b * 255)).name()
        self._swatch.setStyleSheet(
            f"QPushButton#layer_swatch {{"
            f" background-color: {hex_color};"
            f" border: 1px solid #555555;"
            f" border-radius: 2px;"
            f"}}"
            f"QPushButton#layer_swatch:hover {{"
            f" border-color: #e0e0e0;"
            f"}}"
        )

    def _on_swatch_clicked(self) -> None:
        """Open a color picker and update the layer highlight color if confirmed."""
        r, g, b, a = self._swatch_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255))
        picked = QColorDialog.getColor(
            initial, self, "Choose Layer Color"
        )
        if not picked.isValid():
            return  # user cancelled
        # Preserve existing alpha; only the RGB changes
        nr = picked.red()   / 255.0
        ng = picked.green() / 255.0
        nb = picked.blue()  / 255.0
        self._swatch_color = (nr, ng, nb, a)
        self._apply_swatch_color(nr, ng, nb)
        self.color_changed.emit(self._swatch_color)

    def _on_eye_clicked(self) -> None:
        """Toggle visibility state and update eye button appearance."""
        self._visible = not self._visible
        # Bright when visible, dim when hidden — gives instant visual feedback
        if self._visible:
            self._eye_btn.setStyleSheet("")          # reset to QSS default
        else:
            self._eye_btn.setStyleSheet("color: #404040;")
        self.visibility_changed.emit(self._visible)

    def paintEvent(self, event) -> None:
        """
        Draw the crimson selection highlight behind all child widgets.

        Using paintEvent rather than stylesheet background-color avoids the
        Qt cascade ambiguity where QWidget { background-color } on a custom
        subclass is silently ignored. fillRect paints edge-to-edge across the
        full row before child widgets render on top.
        """
        if self._is_selected:
            painter = QPainter(self)
            # Crimson accent at ~18% opacity (46 / 255 ≈ 0.18)
            painter.fillRect(self.rect(), QColor(220, 53, 69, 46))
            painter.end()
        super().paintEvent(event)

    def set_selected(self, selected: bool) -> None:
        """
        Highlight or un-highlight this row as the active layer.

        The visual "pop" is entirely vertical — layout top/bottom margins expand
        by 3px each (6px total) so the row grows upward and downward without
        any horizontal shift. The background fill is drawn in paintEvent so it
        spans the full row width reliably regardless of child widget styles.
        """
        self._is_selected = selected
        if selected:
            # Expand layout margins vertically: +3px top, +3px bottom
            self.layout().setContentsMargins(8, 7, 6, 7)
        else:
            # Restore default margins
            self.layout().setContentsMargins(8, 4, 6, 4)
        # Trigger repaint so paintEvent reflects the new state immediately
        self.update()

    def mousePressEvent(self, event) -> None:
        """Emit row_clicked when the row background area is pressed.

        Qt only delivers mousePressEvent to the container when the click lands
        on the widget's own background — child button clicks are consumed by
        the buttons themselves, so this never double-fires.
        """
        self.row_clicked.emit()
        super().mousePressEvent(event)

    def _start_rename(self) -> None:
        """Switch the name label to an inline text editor."""
        self._name_label.hide()
        self._edit.setText(self._name)
        self._edit.show()
        self._edit.setFocus()
        self._edit.selectAll()

    def _commit_rename(self) -> None:
        """Commit the rename and switch back to the label."""
        new_name = self._edit.text().strip() or self._name
        self._name = new_name
        self._name_label.setText(new_name)
        self._edit.hide()
        self._name_label.show()
        self.renamed.emit(new_name)


class ViewportLayersPanel(QWidget):
    """
    Left-side panel (200px fixed) for managing polygon selection layers.

    Signals
    -------
    save_layer_requested(str, tuple)
        Emitted when the user clicks "Save as Layer". Carries the auto-generated
        name and the auto-assigned RGBA color. The app layer should call
        viewport.save_pending_as_layer(name, color) then add_layer() here.
    layer_visibility_changed(int, bool)
        Emitted when a layer's visibility checkbox is toggled.
    layer_deleted(int)
        Emitted when a layer's delete button is clicked.
    """

    save_layer_requested = Signal(str, tuple)   # (name, rgba_tuple)
    layer_visibility_changed = Signal(int, bool)
    layer_color_changed = Signal(int, tuple)    # (layer_id, new_rgba_tuple)
    layer_deleted = Signal(int)
    # Emitted when the user clicks a layer row to make it the active layer.
    layer_selected = Signal(int)
    # Emitted when the user wants to permanently delete the selected layer's
    # faces from the mesh geometry. The viewport handles the topology change
    # and undo snapshot; the panel removes the row after the signal is emitted.
    delete_faces_requested = Signal(int)        # layer_id
    # Emitted when the user clicks "Apply to Layer" with a loaded texture.
    # Args: (layer_id, texture_path, rotate_deg, scale, offset_x, offset_y)
    project_texture_requested = Signal(int, str, float, float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewport_layers_panel")
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # --- Header ---
        header = QLabel("LAYERS")
        header.setObjectName("tools_section_header")
        layout.addWidget(header)

        # --- Divider ---
        divider = QFrame()
        divider.setObjectName("tools_divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        # --- Pending indicator ---
        # Shows how many faces are in the current (unsaved) polygon selection.
        # Hidden when no selection is pending.
        self._pending_label = QLabel("")
        self._pending_label.setObjectName("viewport_placeholder")
        self._pending_label.setWordWrap(True)
        self._pending_label.hide()
        layout.addWidget(self._pending_label)

        # --- Save as Layer button ---
        # Disabled until a polygon selection has been finalized.
        self._save_btn = QPushButton("Save as Layer")
        self._save_btn.setObjectName("save_layer_btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_btn)

        # --- Delete Faces button ---
        # Permanently removes the selected layer's faces from the mesh geometry.
        # Enabled only when a layer is selected. This is the viewport-based
        # alternative to preprocessing background removal — select unwanted
        # geometry (base, stool, background surfaces), then delete it directly
        # from the mesh. The operation is undoable via Ctrl+Z.
        self._delete_faces_btn = QPushButton("Delete Faces")
        self._delete_faces_btn.setObjectName("save_layer_btn")
        self._delete_faces_btn.setEnabled(False)
        self._delete_faces_btn.setToolTip(
            "Permanently remove the selected layer's faces from the mesh.\n"
            "Use this to delete background geometry (bases, tables, floors).\n"
            "Undoable with Ctrl+Z."
        )
        self._delete_faces_btn.clicked.connect(self._on_delete_faces_clicked)
        layout.addWidget(self._delete_faces_btn)

        # Second divider below the action buttons
        divider2 = QFrame()
        divider2.setObjectName("tools_divider")
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setFrameShadow(QFrame.Shadow.Plain)
        divider2.setFixedHeight(1)
        layout.addWidget(divider2)

        # --- Scrollable layer list ---
        # Each saved layer appears as a _LayerRow widget inside this scroll area.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_container)
        layout.addWidget(scroll, 1)  # stretch=1 → fills remaining panel height

        # --- Empty-state placeholder ---
        # Shown inside the list area when no layers exist yet.
        self._placeholder = QLabel(
            "No layers yet.\n\nUse the polygon\nselect tool to\nisolate a region."
        )
        self._placeholder.setObjectName("viewport_placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._placeholder.setWordWrap(True)
        self._placeholder.setContentsMargins(8, 4, 0, 0)
        self._list_layout.insertWidget(0, self._placeholder)

        # ------------------------------------------------------------------ #
        # Texture Projection section                                        #
        # Sits below the layer list. Controls activate when a layer is      #
        # selected and a texture has been imported.                         #
        # ------------------------------------------------------------------ #

        # Divider separating layer list from texture controls
        tex_divider = QFrame()
        tex_divider.setObjectName("tools_divider")
        tex_divider.setFrameShape(QFrame.Shape.HLine)
        tex_divider.setFrameShadow(QFrame.Shadow.Plain)
        tex_divider.setFixedHeight(1)
        layout.addWidget(tex_divider)

        # Section header row: "TEXTURE" label on left, Polyhaven link on right.
        # Polyhaven is a free PBR texture library — a useful starting point
        # for sourcing projection textures without leaving the app's orbit.
        tex_header_row = QHBoxLayout()
        tex_header_row.setContentsMargins(0, 0, 0, 0)
        tex_header_row.setSpacing(4)

        tex_header = QLabel("TEXTURE")
        tex_header.setObjectName("tools_section_header")
        tex_header_row.addWidget(tex_header, 1)

        # Small link button that opens Polyhaven.com in the system browser.
        # Styled as a flat muted label-link to keep visual weight low.
        self._polyhaven_btn = QPushButton("Polyhaven ↗")
        self._polyhaven_btn.setObjectName("polyhaven_btn")
        self._polyhaven_btn.setToolTip("Open Polyhaven.com — free PBR textures")
        self._polyhaven_btn.setFixedHeight(18)
        self._polyhaven_btn.clicked.connect(
            lambda: webbrowser.open("https://polyhaven.com")
        )
        tex_header_row.addWidget(self._polyhaven_btn)
        layout.addLayout(tex_header_row)

        # "Import Texture..." button — opens a file dialog to pick a texture.
        # Enabled only when a layer is selected.
        self._import_tex_btn = QPushButton("Import Texture…")
        self._import_tex_btn.setObjectName("save_layer_btn")
        self._import_tex_btn.setEnabled(False)
        self._import_tex_btn.setToolTip("Load a PNG or JPG to project onto the selected layer")
        self._import_tex_btn.clicked.connect(self._on_import_texture)
        layout.addWidget(self._import_tex_btn)

        # Label showing the name of the currently loaded texture file.
        # Hidden until a texture is imported.
        self._tex_filename_label = QLabel("")
        self._tex_filename_label.setObjectName("viewport_placeholder")
        self._tex_filename_label.setWordWrap(False)
        self._tex_filename_label.hide()
        layout.addWidget(self._tex_filename_label)

        # --- Material controls (Rotate / Scale / Offset X / Offset Y) ---
        # Each control is a [label] [slider] [value] row using a compact
        # horizontal layout. Sliders use integer ranges scaled to the real
        # float values they represent.

        # Rotate: 0–360° integer maps directly to degrees.
        self._rot_slider, self._rot_val = self._make_slider_row(
            layout, "Rotate", 0, 360, 0, suffix="°"
        )

        # Scale: slider range 10–500 → 0.10× to 5.00× (divided by 100).
        self._scale_slider, self._scale_val = self._make_slider_row(
            layout, "Scale", 10, 500, 100, suffix="×", divisor=100.0, decimals=2
        )

        # Offset X: slider range -100–100 → -1.00 to 1.00 (divided by 100).
        self._offx_slider, self._offx_val = self._make_slider_row(
            layout, "Off X", -100, 100, 0, suffix="", divisor=100.0, decimals=2
        )

        # Offset Y: slider range -100–100 → -1.00 to 1.00 (divided by 100).
        self._offy_slider, self._offy_val = self._make_slider_row(
            layout, "Off Y", -100, 100, 0, suffix="", divisor=100.0, decimals=2
        )

        # "Apply to Layer" — triggers the texture projection into the albedo.
        # Enabled only when a layer is selected AND a texture has been imported.
        self._apply_tex_btn = QPushButton("Apply to Layer")
        self._apply_tex_btn.setObjectName("save_layer_btn")
        self._apply_tex_btn.setEnabled(False)
        self._apply_tex_btn.setToolTip("Project the loaded texture onto the selected layer")
        self._apply_tex_btn.clicked.connect(self._on_apply_texture)
        layout.addWidget(self._apply_tex_btn)

        # Internal state
        self._layer_rows: dict[int, _LayerRow] = {}   # layer_id → row widget
        self._layer_nums: dict[int, int] = {}        # layer_id → assigned number
        self._pending_face_count: int = 0
        self._selected_id: int | None = None   # layer_id of the currently selected row
        self._texture_path: str | None = None  # path to the currently loaded texture

        # Live preview: re-project whenever a slider is released so the user
        # can see the result before committing with Apply to Layer.
        # A short single-shot timer prevents rapid repeated emissions when
        # the user drags a slider quickly (fires 100ms after release settles).
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._on_apply_texture)

        # Wire all sliders: valueChanged gives live label updates (already
        # wired in _make_slider_row), and sliderReleased fires the preview.
        for slider in (self._rot_slider, self._scale_slider,
                       self._offx_slider, self._offy_slider):
            slider.sliderReleased.connect(self._schedule_preview)

    # ------------------------------------------------------------------ #
    # Slots — called by ViewportView signal wiring                        #
    # ------------------------------------------------------------------ #

    def set_pending_selection(self, face_count: int) -> None:
        """
        Called when the viewport finalizes a polygon selection.

        Shows a pending indicator and enables the Save button so the user
        knows there is an unsaved selection ready to be promoted to a layer.

        Args:
            face_count: Number of mesh faces inside the finalized polygon.
        """
        self._pending_face_count = face_count
        if face_count > 0:
            self._pending_label.setText(
                f"{face_count:,} faces\nselected — click\n\"Save as Layer\"\nto keep."
            )
            self._pending_label.show()
            self._save_btn.setEnabled(True)
        else:
            self._pending_label.hide()
            self._save_btn.setEnabled(False)

    def add_layer(self, layer_id: int, name: str, color: tuple) -> None:
        """
        Append a new layer row to the list.

        Called by ViewportView after viewport.save_pending_as_layer() succeeds.

        Args:
            layer_id: Unique integer ID assigned by the viewport.
            name:     Layer display name.
            color:    RGBA float tuple (0-1 each) for the highlight color.
        """
        # Hide placeholder once the first layer is added
        self._placeholder.hide()

        # Track the assigned number for this layer so _on_save_clicked can
        # always continue from the highest number in use.
        try:
            self._layer_nums[layer_id] = int(name.rsplit(" ", 1)[-1])
        except (ValueError, IndexError):
            self._layer_nums[layer_id] = len(self._layer_rows) + 1

        row = _LayerRow(name, color)
        # Wire row signals → panel signals carrying the layer_id
        row.visibility_changed.connect(
            lambda vis, lid=layer_id: self.layer_visibility_changed.emit(lid, vis)
        )
        row.color_changed.connect(
            lambda rgba, lid=layer_id: self.layer_color_changed.emit(lid, rgba)
        )
        row.delete_requested.connect(lambda lid=layer_id: self._on_delete(lid))
        # Clicking the row body selects it and notifies the viewport
        row.row_clicked.connect(lambda lid=layer_id: self._select_layer(lid))

        # Insert before the stretch at the bottom of the list
        stretch_idx = self._list_layout.count() - 1
        self._list_layout.insertWidget(stretch_idx, row)
        self._layer_rows[layer_id] = row

    def reset(self) -> None:
        """Clear all layer rows and return to placeholder state."""
        for row in list(self._layer_rows.values()):
            self._list_layout.removeWidget(row)
            row.deleteLater()
        self._layer_rows.clear()
        self._layer_nums.clear()
        self._pending_face_count = 0
        self._pending_label.hide()
        self._save_btn.setEnabled(False)
        self._delete_faces_btn.setEnabled(False)
        self._placeholder.show()
        self._selected_id = None

        # Clear texture projection state
        self._texture_path = None
        self._tex_filename_label.hide()
        self._rot_slider.setValue(0)
        self._scale_slider.setValue(100)
        self._offx_slider.setValue(0)
        self._offy_slider.setValue(0)
        self._update_texture_section()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _select_layer(self, layer_id: int) -> None:
        """
        Toggle selection on the clicked row and notify the viewport.

        If the row is already selected, clicking it again deselects it
        (sets selected_id to None and emits -1 so the viewport clears any
        active-layer boost). Otherwise selects the clicked row exclusively.
        Also updates the texture projection section's enabled state to match
        the new selection.
        """
        if self._selected_id == layer_id:
            # Second click on the active row — deselect it
            for row in self._layer_rows.values():
                row.set_selected(False)
            self._selected_id = None
            self.layer_selected.emit(-1)
        else:
            for lid, row in self._layer_rows.items():
                row.set_selected(lid == layer_id)
            self._selected_id = layer_id
            self.layer_selected.emit(layer_id)

        # Re-evaluate button enable states whenever selection changes.
        self._delete_faces_btn.setEnabled(self._selected_id is not None)
        self._update_texture_section()

    def _make_slider_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        min_val: int,
        max_val: int,
        default_val: int,
        suffix: str = "",
        divisor: float = 1.0,
        decimals: int = 0,
    ) -> tuple[QSlider, QLabel]:
        """
        Build a compact [label][slider][value] row and add it to parent_layout.

        Returns the QSlider and the value QLabel so callers can read/reset them.
        The value label is updated live as the slider moves.

        Parameters
        ----------
        label_text  : Short left-aligned label (≤6 chars to fit in 200px panel).
        min_val     : Slider integer minimum.
        max_val     : Slider integer maximum.
        default_val : Initial slider position.
        suffix      : String appended to the displayed value (e.g. "°" or "×").
        divisor     : Divide slider integer value by this to get display float.
        decimals    : Decimal places in the displayed value.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        lbl = QLabel(label_text)
        lbl.setObjectName("slider_label")
        lbl.setFixedWidth(40)
        row.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setEnabled(False)
        row.addWidget(slider, 1)

        # Display label — shows the current value as a formatted float or int
        def _fmt(v: int) -> str:
            real = v / divisor
            if decimals == 0:
                return f"{int(real)}{suffix}"
            return f"{real:.{decimals}f}{suffix}"

        val_lbl = QLabel(_fmt(default_val))
        val_lbl.setObjectName("slider_value")
        val_lbl.setFixedWidth(36)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(val_lbl)

        # Wire slider → value label update
        slider.valueChanged.connect(lambda v, lbl=val_lbl, f=_fmt: lbl.setText(f(v)))

        parent_layout.addLayout(row)
        return slider, val_lbl

    def _update_texture_section(self) -> None:
        """
        Enable or disable texture projection controls based on current state.

        Rules:
            Import Texture — enabled when a layer is selected.
            Sliders        — enabled when a layer is selected.
            Apply to Layer — enabled when a layer is selected AND a texture
                             has been imported.
        """
        has_layer = self._selected_id is not None
        has_tex   = self._texture_path is not None

        self._import_tex_btn.setEnabled(has_layer)
        self._rot_slider.setEnabled(has_layer)
        self._scale_slider.setEnabled(has_layer)
        self._offx_slider.setEnabled(has_layer)
        self._offy_slider.setEnabled(has_layer)
        self._apply_tex_btn.setEnabled(has_layer and has_tex)

    def _on_import_texture(self) -> None:
        """
        Open a file dialog and store the selected texture path.

        Accepts PNG, JPG, JPEG, and TIFF — common formats for PBR albedo maps.
        Updates the filename label so the user can confirm what's loaded.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Texture",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.exr)",
        )
        if not path:
            return  # user cancelled

        self._texture_path = path

        # Show the filename (not the full path) so it fits the 200px panel.
        # Truncate with ellipsis if still too long.
        filename = path.split("/")[-1].split("\\")[-1]
        if len(filename) > 22:
            filename = filename[:20] + "…"
        self._tex_filename_label.setText(filename)
        self._tex_filename_label.show()

        # Activate the Apply button now that both a layer and a texture exist.
        self._update_texture_section()

        # Immediately project a preview at the current slider values so the
        # user can see the texture on the mesh without needing to hit Apply.
        # Adjusting sliders will update the preview automatically on release.
        self._on_apply_texture()

    def _on_apply_texture(self) -> None:
        """
        Emit project_texture_requested with the current material control values.

        The viewport receives this signal and delegates to MeshPainter.project_texture(),
        which rasterizes the material transform into the albedo buffer. The operation
        is undoable via the normal Undo button.
        """
        if self._selected_id is None or self._texture_path is None:
            return

        rotate  = float(self._rot_slider.value())              # degrees (0–360)
        scale   = float(self._scale_slider.value()) / 100.0    # 0.10–5.00×
        offset_x = float(self._offx_slider.value()) / 100.0   # -1.00–1.00
        offset_y = float(self._offy_slider.value()) / 100.0   # -1.00–1.00

        self.project_texture_requested.emit(
            self._selected_id,
            self._texture_path,
            rotate,
            scale,
            offset_x,
            offset_y,
        )

    def _on_save_clicked(self) -> None:
        """Assign an auto-name and color, then emit save_layer_requested."""
        # next_num = one above the highest number currently in use, or 1 if
        # no layers exist. This means numbering always continues forward from
        # the highest existing layer (e.g. Layer 2 exists → next is Layer 3)
        # but resets to 1 when all layers have been deleted.
        next_num = max(self._layer_nums.values()) + 1 if self._layer_nums else 1
        name  = f"Layer {next_num}"
        color = _LAYER_COLORS[(next_num - 1) % len(_LAYER_COLORS)]

        # Disable Save button immediately — user must draw a new polygon
        # before saving another layer.
        self._pending_label.hide()
        self._save_btn.setEnabled(False)
        self._pending_face_count = 0

        self.save_layer_requested.emit(name, color)

    def _on_delete(self, layer_id: int) -> None:
        """Remove the row widget and emit layer_deleted."""
        self._layer_nums.pop(layer_id, None)
        row = self._layer_rows.pop(layer_id, None)
        if row is not None:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        # Clear selection state if the deleted layer was selected
        if self._selected_id == layer_id:
            self._selected_id = None
            self._delete_faces_btn.setEnabled(False)
            self._update_texture_section()
        # Show placeholder again if all layers were deleted
        if not self._layer_rows:
            self._placeholder.show()
        self.layer_deleted.emit(layer_id)

    def _on_delete_faces_clicked(self) -> None:
        """
        Emit delete_faces_requested for the currently selected layer.

        The viewport handles the actual geometry change and undo snapshot.
        This method also removes the row from the panel since the layer
        will no longer exist after the faces are deleted from the mesh.
        """
        if self._selected_id is None:
            return
        layer_id = self._selected_id
        # Notify the viewport first — it needs the layer to still be in
        # self._layers when delete_layer_faces runs. If _on_delete fired first,
        # the layer_deleted signal would call viewport.delete_layer(), removing
        # the layer before delete_layer_faces could find it.
        # delete_layer_faces handles its own internal viewport.delete_layer()
        # cleanup; _on_delete here only removes the panel UI row and re-emits
        # layer_deleted, which calls viewport.delete_layer() again harmlessly
        # (layer is already gone — the method just returns early).
        self.delete_faces_requested.emit(layer_id)
        # Now clean up the panel UI row
        self._on_delete(layer_id)

    def _schedule_preview(self) -> None:
        """
        Restart the debounce timer on slider release.

        Connected to sliderReleased on all four material sliders. Starts
        the 100ms single-shot timer so a preview projection fires shortly
        after the user finishes dragging — prevents hammering project_texture
        on every intermediate tick while the slider is in motion.
        """
        if self._selected_id is not None and self._texture_path is not None:
            self._preview_timer.start()
