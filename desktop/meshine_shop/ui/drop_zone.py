"""
Drag-and-drop file import widget for Meshine Shop.

The DropZone is the primary input mechanism for the Import view. Users
drag photo sets (JPEG, PNG, TIFF) or point cloud files (PLY) from their
file system directly into this widget to start the photogrammetry pipeline.

It uses Qt's drag-and-drop event system:
    - dragEnterEvent: Validates that the dragged data contains file URLs
    - dragLeaveEvent: Resets visual state when the drag leaves the zone
    - dropEvent: Extracts file paths from the dropped URLs and emits them

The files_dropped signal carries a list of file path strings, which the
parent ImportView connects to for further processing (validation, queueing).
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal


class DropZone(QWidget):
    # Emitted when files are successfully dropped onto the widget.
    # Carries a list of absolute file path strings extracted from
    # the drop event's MIME data.
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        # Object name enables QSS styling (dashed border, hover effects).
        self.setObjectName("drop_zone")
        # Enable this widget to receive drag-and-drop events from the OS.
        self.setAcceptDrops(True)
        # Minimum height ensures the drop zone is a large enough target
        # for comfortable drag-and-drop interaction.
        self.setMinimumHeight(300)

        # Center-aligned vertical layout for the icon, title, and subtitle.
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        # "+" icon — a simple text character styled large in crimson.
        # Serves as a visual focal point that signals "add files here."
        icon = QLabel("+")
        icon.setObjectName("drop_icon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        # Primary instruction text — tells the user what to do.
        title = QLabel("Drop photos or point clouds here")
        title.setObjectName("drop_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Secondary hint — lists the supported file formats so users
        # know what they can import without checking documentation.
        subtitle = QLabel("Supports JPEG, PNG, TIFF, and PLY files")
        subtitle.setObjectName("drop_subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event):
        """Accept the drag if it contains file URLs (from the OS file manager)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # Reset stylesheet to trigger a re-render, which activates
            # the QSS :hover pseudo-class for visual drag-over feedback.
            self.setStyleSheet("")

    def dragLeaveEvent(self, event):
        """Reset visual state when the drag leaves the drop zone."""
        self.setStyleSheet("")

    def dropEvent(self, event):
        """Extract file paths from the dropped URLs and emit them."""
        urls = event.mimeData().urls()
        # Convert Qt URL objects to local file path strings.
        paths = [url.toLocalFile() for url in urls]
        if paths:
            self.files_dropped.emit(paths)
