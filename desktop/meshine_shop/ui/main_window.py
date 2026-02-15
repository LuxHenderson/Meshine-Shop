"""
Main content area and view definitions for Meshine Shop.

This module contains the three primary views of the application:
    - ImportView:  Drag-and-drop zone for importing photo sets
    - ProcessView: Processing queue showing pipeline progress
    - ExportView:  Export settings and format options

These views are stacked inside a QStackedWidget (MainContent), which
shows only one view at a time. The sidebar's nav_changed signal drives
which view is visible — index 0=Import, 1=Process, 2=Export.

This stacked approach was chosen over tab widgets or page navigation
because it provides instant view switching with zero transition cost,
and gives us full control over the navigation UI (the sidebar) rather
than relying on Qt's built-in tab styling.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QStackedWidget, QPushButton,
)
from PySide6.QtCore import Qt, Signal
from meshine_shop.ui.drop_zone import DropZone
from meshine_shop.ui.processing_queue import ProcessingQueue


class ImportView(QWidget):
    """
    The first step in the workflow — importing source data.

    Contains a drag-and-drop zone for photo sets, a label showing how many
    files have been loaded, and a "Start Processing" button that kicks off
    the photogrammetry pipeline.

    Signal flow:
        1. User drops files → DropZone emits files_dropped(list)
        2. ImportView stores the paths and enables the Start button
        3. User clicks Start → ImportView emits start_requested(list)
        4. App layer receives start_requested and creates the pipeline worker
    """

    # Emitted when the user clicks "Start Processing". Carries the list
    # of file paths that were dropped into the import zone.
    start_requested = Signal(list)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Section header — identifies this view for the user.
        header = QLabel("Import")
        header.setObjectName("section_title")
        layout.addWidget(header)

        # The drop zone handles all drag-and-drop interaction.
        # Its files_dropped signal is connected below to update
        # the file count label when files are imported.
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        # File count feedback — shows the user how many files they've
        # imported, confirming the drop was successful. Uses crimson
        # accent color to draw attention to the confirmation.
        self.file_count_label = QLabel("")
        self.file_count_label.setStyleSheet("color: #dc3545; font-size: 13px;")
        layout.addWidget(self.file_count_label)

        # "Start Processing" button — disabled until files are dropped.
        # Clicking this triggers the full photogrammetry pipeline via
        # the start_requested signal, which the app layer listens for.
        self._start_btn = QPushButton("Start Processing")
        self._start_btn.setObjectName("start_button")
        self._start_btn.setEnabled(False)
        self._start_btn.setFixedHeight(44)
        self._start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self._start_btn)

        # Stretch pushes all content to the top of the view.
        layout.addStretch()

        # Store dropped file paths for when the user clicks Start.
        self._pending_paths: list[str] = []

    def _on_files_dropped(self, paths):
        """Update the file count label and enable the Start button."""
        self._pending_paths = paths
        count = len(paths)
        self.file_count_label.setText(
            f"{count} file{'s' if count != 1 else ''} ready for processing"
        )
        # Enable the Start button now that we have files to process.
        self._start_btn.setEnabled(True)

    def _on_start_clicked(self):
        """Emit the start_requested signal with the pending file paths."""
        if self._pending_paths:
            # Disable the button to prevent double-clicks while processing.
            self._start_btn.setEnabled(False)
            self._start_btn.setText("Processing...")
            self.start_requested.emit(self._pending_paths)

    def reset_start_button(self):
        """Re-enable the Start button after pipeline completes or errors."""
        self._start_btn.setEnabled(bool(self._pending_paths))
        self._start_btn.setText("Start Processing")


class ProcessView(QWidget):
    """
    The second step — running the photogrammetry pipeline.

    Contains the processing queue panel, which shows active jobs with
    their current pipeline stage and progress indicators.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # The queue widget manages its own layout and displays
        # real-time pipeline stage progress via signal connections.
        self.queue = ProcessingQueue()
        layout.addWidget(self.queue)


class ExportView(QWidget):
    """
    The third step — exporting game-ready assets.

    Will eventually contain format selection (FBX/glTF), quality presets
    (mobile, desktop, cinematic), and export path configuration. Currently
    shows a placeholder message until Phase 1d/2e implement export logic.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Section header for the Export view.
        header = QLabel("Export")
        header.setObjectName("section_title")
        layout.addWidget(header)

        # Placeholder text — communicates that export functionality
        # depends on having a processed asset first.
        placeholder = QLabel(
            "Export settings will appear here once an asset has been processed."
        )
        placeholder.setStyleSheet("color: #999999;")
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)

        layout.addStretch()


class MainContent(QWidget):
    """
    Container that manages view switching via QStackedWidget.

    QStackedWidget holds all three views in a stack, displaying only
    one at a time. The switch_view method (connected to the sidebar's
    nav_changed signal) controls which view is visible.

    View index mapping:
        0 = ImportView
        1 = ProcessView
        2 = ExportView

    The import_view and process_view attributes are exposed publicly so
    the app layer can connect signals to them (e.g., start_requested from
    ImportView, and worker signals to ProcessView's queue).
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create named references to views so the app can access them
        # for signal wiring. Previously these were anonymous addWidget calls.
        self.import_view = ImportView()
        self.process_view = ProcessView()
        self.export_view = ExportView()

        # QStackedWidget shows one child at a time. Views are added in
        # the same order as the sidebar buttons, so their indices match.
        self.stack = QStackedWidget()
        self.stack.addWidget(self.import_view)     # index 0
        self.stack.addWidget(self.process_view)    # index 1
        self.stack.addWidget(self.export_view)     # index 2

        layout.addWidget(self.stack)

    def switch_view(self, index: int):
        """Switch the visible view to match the sidebar selection."""
        self.stack.setCurrentIndex(index)
