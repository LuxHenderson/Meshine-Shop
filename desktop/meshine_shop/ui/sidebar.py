"""
Top navigation bar for Meshine Shop.

Replaces the original vertical sidebar with a slim horizontal bar that sits
above the main content area. The four workflow tabs — Import, Process,
Viewport, Export — are evenly distributed across the full bar width so each
one takes identical space regardless of label length.

The same nav_changed(int) signal and QButtonGroup wiring from the original
Sidebar are preserved exactly, so nothing in app.py needs to change beyond
swapping to a vertical layout. Callers still refer to self.sidebar and
self.sidebar.button_group as before.

Design decisions:
    - Fixed height (40px) keeps the chrome minimal so the viewport gets maximum space.
    - Equal stretch weights on every button ensure perfectly even spacing at any window width.
    - Active indicator is a crimson bottom border (tab-strip convention) rather than a
      left border, which reads better on a horizontal bar.
    - "MESHINE SHOP" branding is the window title; the bar is navigation only.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QButtonGroup
from PySide6.QtCore import Signal


class Sidebar(QWidget):
    """
    Slim horizontal top navigation bar.

    Emits nav_changed(int) when the user selects a tab, matching the
    original Sidebar contract so all downstream wiring is unchanged.
    """

    # Same signal as the original Sidebar — downstream code is unaffected.
    nav_changed = Signal(int)

    # Tab definitions: (display label, tooltip).
    # Order matches the QStackedWidget indices in MainContent.
    NAV_ITEMS = [
        ("Import",   "Drag & drop photo sets"),
        ("Process",  "Run the pipeline"),
        ("Viewport", "Inspect and paint textures"),
        ("Export",   "Download game-ready assets"),
    ]

    def __init__(self):
        super().__init__()
        # Object name used by QSS to apply the top-bar background.
        self.setObjectName("top_nav_bar")
        # Fixed height keeps the bar slim — just tall enough for comfortable
        # click targets without wasting vertical space above the content area.
        self.setFixedHeight(40)

        # Horizontal layout with no margins or inter-widget spacing.
        # Even distribution is achieved by assigning stretch=1 to every button
        # (see the addWidget calls below), not by adding spacers.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # QButtonGroup enforces mutual exclusivity — clicking one tab
        # automatically unchecks the others, same as the original sidebar.
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        for i, (label, tooltip) in enumerate(self.NAV_ITEMS):
            btn = QPushButton(label)
            # Shared object name so all tab buttons receive the same QSS rules.
            btn.setObjectName("nav_button")
            # Checkable so Qt tracks the active/pressed state for QSS :checked.
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            self.button_group.addButton(btn, i)
            # stretch=1 on every button → they all receive equal width as the
            # window resizes, keeping the tabs perfectly evenly spaced.
            layout.addWidget(btn, 1)

        # Default to Import on launch — the natural workflow start point.
        self.button_group.button(0).setChecked(True)

        # Forward the button group's idClicked signal as nav_changed so callers
        # don't need to know about QButtonGroup internals.
        self.button_group.idClicked.connect(self.nav_changed.emit)
