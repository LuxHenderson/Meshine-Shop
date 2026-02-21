"""
Drag-and-drop file import widget for Meshine Shop.

The DropZone is the primary input mechanism for the Import view. Users
drag photo sets (JPEG, PNG, TIFF, HEIC) or entire folders of photos from
their file system directly into this widget to start the photogrammetry
pipeline.

It uses Qt's drag-and-drop event system:
    - dragEnterEvent: Validates that the dragged data contains file URLs
    - dragLeaveEvent: Resets visual state when the drag leaves the zone
    - dropEvent: Extracts file paths from the dropped URLs and emits them

When a dropped path is a directory, the widget recursively scans it for
supported image files — this lets users import large datasets (100+ photos)
by dropping a single folder instead of selecting individual files.

The files_dropped signal carries a list of file path strings, which the
parent ImportView connects to for further processing (validation, queueing).
"""

from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal

from meshine_shop.core.engine import SUPPORTED_IMAGE_EXTENSIONS


def collect_images_from_paths(paths: list[str]) -> list[str]:
    """
    Given a mix of file and directory paths, return all supported image files.

    For each path:
    - If it's a directory, recursively walk it and collect image files
    - If it's a file with a supported extension, include it directly
    - Unsupported files are silently skipped (the ingest stage handles
      detailed validation and error reporting later)

    Returns a sorted list for deterministic ordering across runs.
    """
    image_paths = []

    for raw_path in paths:
        p = Path(raw_path)

        if p.is_dir():
            # Recursively find all image files inside the directory.
            # rglob("*") walks all subdirectories, then we filter by extension.
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    image_paths.append(str(child))

        elif p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            # Single file with a supported extension — include directly.
            image_paths.append(str(p))

    # Sort for deterministic ordering regardless of filesystem walk order.
    image_paths.sort()
    return image_paths


class DropZone(QWidget):
    # Emitted when files are successfully dropped onto the widget.
    # Carries a list of absolute file path strings — either individual
    # image files or all images found inside dropped folders.
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
        title = QLabel("Drop photos or a folder here")
        title.setObjectName("drop_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Secondary hint — lists the supported file formats and mentions
        # folder support so users know they can import entire directories.
        subtitle = QLabel("Supports JPEG, PNG, TIFF, HEIC")
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
        """
        Extract file paths from the dropped URLs, scan folders, and emit results.

        Handles both individual files and directories. When a directory is
        dropped, it's recursively scanned for supported image files so the
        user can import an entire photo dataset in one drag-and-drop action.
        """
        urls = event.mimeData().urls()
        # Convert Qt URL objects to local file path strings.
        raw_paths = [url.toLocalFile() for url in urls]
        # Resolve folders into individual image file paths.
        image_paths = collect_images_from_paths(raw_paths)
        if image_paths:
            self.files_dropped.emit(image_paths)
