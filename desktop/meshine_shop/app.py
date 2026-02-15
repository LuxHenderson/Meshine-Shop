from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QStatusBar
from PySide6.QtCore import Qt
from meshine_shop.ui.sidebar import Sidebar
from meshine_shop.ui.main_window import MainContent
from meshine_shop.ui.styles import DARK_THEME


class MeshineShopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meshine Shop")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self.setStyleSheet(DARK_THEME)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.main_content = MainContent()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.main_content, 1)

        self.sidebar.nav_changed.connect(self.main_content.switch_view)

        status_bar = QStatusBar()
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)
