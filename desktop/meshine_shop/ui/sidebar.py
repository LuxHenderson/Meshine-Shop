from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QButtonGroup,
)
from PySide6.QtCore import Signal, Qt


class Sidebar(QWidget):
    nav_changed = Signal(int)

    NAV_ITEMS = [
        ("Import", "Drag & drop photo sets"),
        ("Process", "Run the pipeline"),
        ("Export", "Download game-ready assets"),
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(4)

        # Logo / title
        title = QLabel("MESHINE SHOP")
        title.setStyleSheet(
            "color: #dc3545; font-size: 16px; font-weight: 700; "
            "letter-spacing: 2px; padding: 0 8px 20px 8px;"
        )
        layout.addWidget(title)

        # Navigation buttons
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        for i, (label, _tooltip) in enumerate(self.NAV_ITEMS):
            btn = QPushButton(label)
            btn.setObjectName("nav_button")
            btn.setCheckable(True)
            btn.setToolTip(_tooltip)
            self.button_group.addButton(btn, i)
            layout.addWidget(btn)

        layout.addStretch()

        # Version label at bottom
        version = QLabel("v0.1.0")
        version.setStyleSheet("color: #4a5568; font-size: 11px; padding: 0 8px;")
        layout.addWidget(version)

        # Select first button by default
        self.button_group.button(0).setChecked(True)
        self.button_group.idClicked.connect(self.nav_changed.emit)
