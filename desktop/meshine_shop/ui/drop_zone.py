from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal


class DropZone(QWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName("drop_zone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(300)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon = QLabel("+")
        icon.setObjectName("drop_icon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        title = QLabel("Drop photos or point clouds here")
        title.setObjectName("drop_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Supports JPEG, PNG, TIFF, and PLY files")
        subtitle.setObjectName("drop_subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("")  # triggers :hover via re-render

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [url.toLocalFile() for url in urls]
        if paths:
            self.files_dropped.emit(paths)
