"""
Sidebar navigation widget for Meshine Shop.

The sidebar provides the primary navigation for the application. It contains
three nav buttons — Import, Process, and Export — which correspond to the
three main workflow stages of the photogrammetry pipeline.

Navigation uses Qt's signal/slot system: when a button is clicked, the
sidebar emits a nav_changed signal with the button's index. The main
content area listens for this signal and swaps to the corresponding view.

QButtonGroup is used with exclusive mode to ensure only one nav button
can be active at a time, which mirrors the single-view-at-a-time behavior
of the stacked widget in MainContent.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QButtonGroup,
)
from PySide6.QtCore import Signal, Qt


class Sidebar(QWidget):
    # Custom signal emitted when the user clicks a nav button.
    # Carries the integer index of the selected button (0=Import,
    # 1=Process, 2=Export), which maps directly to the stacked
    # widget indices in MainContent.
    nav_changed = Signal(int)

    # Navigation item definitions: (display label, tooltip description).
    # These are iterated to build the nav buttons dynamically, making it
    # easy to add new sections later without touching the layout code.
    NAV_ITEMS = [
        ("Import", "Drag & drop photo sets"),
        ("Process", "Run the pipeline"),
        ("Export", "Download game-ready assets"),
    ]

    def __init__(self):
        super().__init__()
        # Object name is used by QSS to target sidebar-specific styles.
        self.setObjectName("sidebar")
        # Fixed width ensures the sidebar never stretches or shrinks
        # when the window is resized — only the main content area flexes.
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(4)

        # App branding — displayed at the top of the sidebar in crimson
        # with letter-spacing for a clean, logo-like appearance.
        title = QLabel("MESHINE SHOP")
        title.setStyleSheet(
            "color: #dc3545; font-size: 16px; font-weight: 700; "
            "letter-spacing: 2px; padding: 0 8px 20px 8px;"
        )
        layout.addWidget(title)

        # QButtonGroup enforces mutual exclusivity — clicking one button
        # automatically unchecks the others. Each button is assigned an
        # integer ID matching its position, which becomes the signal payload.
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        for i, (label, _tooltip) in enumerate(self.NAV_ITEMS):
            btn = QPushButton(label)
            # Object name allows QSS to style all nav buttons uniformly.
            btn.setObjectName("nav_button")
            # Checkable buttons maintain a pressed/active state, which
            # QSS uses to apply the :checked style (crimson text).
            btn.setCheckable(True)
            btn.setToolTip(_tooltip)
            self.button_group.addButton(btn, i)
            layout.addWidget(btn)

        # Stretch pushes the version label to the bottom of the sidebar.
        layout.addStretch()

        # Version indicator — dimmed so it's visible but unobtrusive.
        # Useful for debugging and user-facing release identification.
        version = QLabel("v0.1.0")
        version.setStyleSheet("color: #4a5568; font-size: 11px; padding: 0 8px;")
        layout.addWidget(version)

        # Default to the Import view on launch — this is the natural
        # starting point of the workflow (import photos first).
        self.button_group.button(0).setChecked(True)

        # Forward the button group's idClicked signal as our nav_changed signal.
        # This decouples the sidebar's internal implementation from consumers —
        # they just listen for nav_changed(int) without knowing about QButtonGroup.
        self.button_group.idClicked.connect(self.nav_changed.emit)
