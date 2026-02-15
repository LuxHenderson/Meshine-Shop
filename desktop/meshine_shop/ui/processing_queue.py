"""
Processing queue widget for Meshine Shop.

Displays the list of active and completed processing jobs. Each job
represents a photo set moving through the photogrammetry pipeline
(ingest -> features -> sparse -> dense -> mesh -> texture).

Currently shows an empty state. In Phase 1c, this will be expanded to:
    - Display active jobs with their current pipeline stage
    - Show per-stage progress bars
    - Stream log output from COLMAP in real-time
    - Allow cancellation of in-progress jobs
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class ProcessingQueue(QWidget):
    def __init__(self):
        super().__init__()
        # Object name enables the QSS "card" styling (dark background,
        # rounded corners) defined in styles.py.
        self.setObjectName("queue_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Section title for the queue panel.
        title = QLabel("Processing Queue")
        title.setObjectName("section_title")
        layout.addWidget(title)

        # Empty state message — shown when no jobs are queued.
        # This will be replaced with a dynamic job list once the
        # pipeline is integrated in Phase 1b/1c.
        empty_msg = QLabel(
            "No jobs in the queue.\nImport a photo set to get started."
        )
        empty_msg.setObjectName("queue_empty")
        empty_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_msg)

        # Stretch pushes content to the top of the panel.
        layout.addStretch()
