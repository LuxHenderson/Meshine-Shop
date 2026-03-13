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
    5. Each layer row: visibility checkbox  |  color swatch  |  name  |  delete btn
       Double-clicking the name enters an inline rename editor.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(8)

        # Checkbox — reserved for future use (e.g. multi-select operations).
        # Not wired to visibility; the eye button owns model show/hide.
        self._check = QCheckBox()
        self._check.setFixedSize(16, 16)
        self._check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self._check)

        # Spacer between checkbox and the swatch/name so the row breathes
        layout.addSpacing(4)

        # Color swatch — small colored square showing the layer highlight color
        self._swatch = QLabel()
        self._swatch.setObjectName("layer_swatch")
        self._swatch.setFixedSize(12, 12)
        r, g, b, _ = color
        hex_color = QColor(int(r * 255), int(g * 255), int(b * 255)).name()
        self._swatch.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #555555; border-radius: 2px;"
        )
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

        # Eye button — toggles layer visibility on the model. Dims when hidden
        # so the state is immediately clear without a separate label.
        self._eye_btn = QPushButton("👁")
        self._eye_btn.setObjectName("layer_eye_btn")
        self._eye_btn.setFixedSize(20, 20)
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

    def _on_eye_clicked(self) -> None:
        """Toggle visibility state and update eye button appearance."""
        self._visible = not self._visible
        # Bright when visible, dim when hidden — gives instant visual feedback
        if self._visible:
            self._eye_btn.setStyleSheet("")          # reset to QSS default
        else:
            self._eye_btn.setStyleSheet("color: #404040;")
        self.visibility_changed.emit(self._visible)

    def set_selected(self, selected: bool) -> None:
        """
        Highlight or un-highlight this row as the active layer.

        Uses a subtle blue-tinted background to indicate selection without
        clashing with the layer color swatches.
        """
        if selected:
            self.setStyleSheet(
                "QWidget { background-color: #35354a; border-radius: 3px; }"
            )
        else:
            self.setStyleSheet("")

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
    layer_deleted = Signal(int)
    # Emitted when the user clicks a layer row to make it the active layer.
    layer_selected = Signal(int)

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

        # Second divider below the save button
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
        self._list_layout.insertWidget(0, self._placeholder)

        # Internal state
        self._layer_rows: dict[int, _LayerRow] = {}   # layer_id → row widget
        self._pending_face_count: int = 0
        self._layer_count: int = 0   # used for auto-naming ("Layer 1", "Layer 2", ...)
        self._selected_id: int | None = None   # layer_id of the currently selected row

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

        row = _LayerRow(name, color)
        # Wire row signals → panel signals carrying the layer_id
        row.visibility_changed.connect(
            lambda vis, lid=layer_id: self.layer_visibility_changed.emit(lid, vis)
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
        self._pending_face_count = 0
        self._pending_label.hide()
        self._save_btn.setEnabled(False)
        self._placeholder.show()
        self._layer_count = 0
        self._selected_id = None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _select_layer(self, layer_id: int) -> None:
        """
        Mark the clicked row as selected and notify the viewport.

        Deselects all other rows first so only one row is highlighted at a time.
        Emits layer_selected so the viewport can boost that layer's overlay.
        """
        for lid, row in self._layer_rows.items():
            row.set_selected(lid == layer_id)
        self._selected_id = layer_id
        self.layer_selected.emit(layer_id)

    def _on_save_clicked(self) -> None:
        """Assign an auto-name and color, then emit save_layer_requested."""
        self._layer_count += 1
        name  = f"Layer {self._layer_count}"
        color = _LAYER_COLORS[(self._layer_count - 1) % len(_LAYER_COLORS)]

        # Disable Save button immediately — user must draw a new polygon
        # before saving another layer.
        self._pending_label.hide()
        self._save_btn.setEnabled(False)
        self._pending_face_count = 0

        self.save_layer_requested.emit(name, color)

    def _on_delete(self, layer_id: int) -> None:
        """Remove the row widget and emit layer_deleted."""
        row = self._layer_rows.pop(layer_id, None)
        if row is not None:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        # Show placeholder again if all layers were deleted
        if not self._layer_rows:
            self._placeholder.show()
        self.layer_deleted.emit(layer_id)
