"""
Layers panel for the Meshine Shop viewport.

ViewportLayersPanel — the 200px fixed-width left-side panel on the Viewport
page. Mirrors the visual structure of ViewportToolsPanel on the right, with
the border on the right edge instead of the left.

Current state: structural placeholder awaiting the full layer system (Steps
13–15 of the tooling plan). The panel header, divider, and empty-state label
are all in place; the layer list and per-layer controls will be inserted here
when polygon selection is implemented.

Future contents (Steps 13–15):
    - Layer list (QListWidget): one row per named selection
      • Visibility toggle checkbox
      • Layer name (double-click to rename)
      • Delete button
    - "New Layer" button (enabled once polygon selection creates a selection)
    - Per-layer options: sculpt mask toggle, texture mask toggle
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ViewportLayersPanel(QWidget):
    """
    Left-side panel on the Viewport page for managing mesh layers.

    Fixed at 200px width to match the right-side tools panel. Styled with
    a right border so it visually separates from the OpenGL viewport area.

    The panel is fully functional structurally — it just has no layers to
    show yet. When the polygon selection tool is added, created layers will
    be appended to self._layer_list (a QListWidget added in Step 13).
    """

    def __init__(self):
        super().__init__()
        # Object name allows QSS to target this panel specifically.
        self.setObjectName("viewport_layers_panel")
        # Fixed width mirrors the right tools panel so the viewport sits
        # symmetrically between two equal-width panels.
        self.setFixedWidth(200)

        # Vertical layout with comfortable internal padding.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # --- Section header ---
        # All-caps small label, same style as the tools panel section headers.
        layers_header = QLabel("LAYERS")
        layers_header.setObjectName("tools_section_header")
        layout.addWidget(layers_header)

        # --- Horizontal divider below header ---
        divider = QFrame()
        divider.setObjectName("tools_divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        # --- Empty-state placeholder ---
        # Shown until the user creates a layer via the polygon selection tool.
        # Word-wrapped so the message fits within the 200px panel width.
        self._placeholder = QLabel(
            "No layers yet.\n\nUse the polygon select tool to isolate a region "
            "and save it as a layer."
        )
        self._placeholder.setObjectName("viewport_placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        # Stretch pushes all content to the top so future layer list items
        # stack downward naturally as they're added.
        layout.addStretch()
