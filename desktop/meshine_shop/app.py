"""
Main application window for Meshine Shop.

This module defines the top-level QMainWindow that serves as the shell
for the entire application. It assembles the core layout:
    - A fixed-width sidebar for navigation (Import / Process / Export)
    - A main content area that swaps views based on sidebar selection
    - A status bar for real-time feedback

The charcoal + crimson dark theme is applied globally here via QSS
(Qt Style Sheets), which styles every child widget in the window.
This approach was chosen over per-widget styling because it keeps
all visual design centralized in one place (styles.py) and makes
future theme changes trivial.
"""

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QStatusBar
from PySide6.QtCore import Qt
from meshine_shop.ui.sidebar import Sidebar
from meshine_shop.ui.main_window import MainContent
from meshine_shop.ui.styles import DARK_THEME


class MeshineShopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meshine Shop")

        # Minimum size prevents the window from being resized too small
        # to display the UI properly. Default size of 1280x800 provides
        # a comfortable workspace without being overwhelming.
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # Apply the global dark theme stylesheet to every widget in the window.
        # See ui/styles.py for the full QSS definition.
        self.setStyleSheet(DARK_THEME)

        # Central widget — required by QMainWindow as the root container.
        # All primary UI elements are children of this widget.
        central = QWidget()
        self.setCentralWidget(central)

        # Horizontal layout: sidebar on the left, main content on the right.
        # Zero margins and spacing ensure the sidebar and content area sit
        # flush against each other with no gaps.
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar handles navigation between the three main views.
        self.sidebar = Sidebar()
        # MainContent uses a QStackedWidget to swap between Import/Process/Export.
        self.main_content = MainContent()

        # The sidebar is fixed-width (220px); the main content stretches to
        # fill remaining space (stretch factor of 1).
        layout.addWidget(self.sidebar)
        layout.addWidget(self.main_content, 1)

        # Connect sidebar navigation clicks to the main content view switcher.
        # When a nav button is clicked, the sidebar emits nav_changed(index),
        # which triggers MainContent.switch_view(index) to swap the visible panel.
        self.sidebar.nav_changed.connect(self.main_content.switch_view)

        # Status bar provides real-time feedback at the bottom of the window.
        # It will display pipeline progress, error messages, and file counts
        # as the app evolves beyond scaffolding.
        status_bar = QStatusBar()
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)
