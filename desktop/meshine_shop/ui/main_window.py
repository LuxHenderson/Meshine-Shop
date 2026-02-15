from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QStackedWidget,
)
from PySide6.QtCore import Qt
from meshine_shop.ui.drop_zone import DropZone
from meshine_shop.ui.processing_queue import ProcessingQueue


class ImportView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        header = QLabel("Import")
        header.setObjectName("section_title")
        layout.addWidget(header)

        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        self.file_count_label = QLabel("")
        self.file_count_label.setStyleSheet("color: #dc3545; font-size: 13px;")
        layout.addWidget(self.file_count_label)

        layout.addStretch()

    def _on_files_dropped(self, paths):
        count = len(paths)
        self.file_count_label.setText(f"{count} file{'s' if count != 1 else ''} ready for processing")


class ProcessView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        self.queue = ProcessingQueue()
        layout.addWidget(self.queue)


class ExportView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        header = QLabel("Export")
        header.setObjectName("section_title")
        layout.addWidget(header)

        placeholder = QLabel(
            "Export settings will appear here once an asset has been processed."
        )
        placeholder.setStyleSheet("color: #999999;")
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)

        layout.addStretch()


class MainContent(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.stack.addWidget(ImportView())
        self.stack.addWidget(ProcessView())
        self.stack.addWidget(ExportView())

        layout.addWidget(self.stack)

    def switch_view(self, index: int):
        self.stack.setCurrentIndex(index)
