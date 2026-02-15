from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class ProcessingQueue(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("queue_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Processing Queue")
        title.setObjectName("section_title")
        layout.addWidget(title)

        empty_msg = QLabel(
            "No jobs in the queue.\nImport a photo set to get started."
        )
        empty_msg.setObjectName("queue_empty")
        empty_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_msg)

        layout.addStretch()
