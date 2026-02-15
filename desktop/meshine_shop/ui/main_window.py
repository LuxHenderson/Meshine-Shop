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
    QWidget, QVBoxLayout, QLabel, QStackedWidget,
)
from PySide6.QtCore import Qt
from meshine_shop.ui.drop_zone import DropZone
from meshine_shop.ui.processing_queue import ProcessingQueue


class ImportView(QWidget):
    """
    The first step in the workflow — importing source data.

    Contains a drag-and-drop zone for photo sets and a label that
    updates to show how many files have been loaded. This view will
    eventually include file validation, thumbnail previews, and the
    ability to start the processing pipeline from here.
    """

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

        # Stretch pushes all content to the top of the view.
        layout.addStretch()

    def _on_files_dropped(self, paths):
        """Update the file count label when files are dropped into the zone."""
        count = len(paths)
        self.file_count_label.setText(f"{count} file{'s' if count != 1 else ''} ready for processing")


class ProcessView(QWidget):
    """
    The second step — running the photogrammetry pipeline.

    Contains the processing queue panel, which will show active jobs,
    their current pipeline stage, and progress indicators. Currently
    displays an empty state message until Phase 1b/1c add actual
    pipeline execution.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # The queue widget manages its own layout and will be expanded
        # in Phase 1c to show real-time pipeline stage progress.
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
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # QStackedWidget shows one child at a time. Views are added in
        # the same order as the sidebar buttons, so their indices match.
        self.stack = QStackedWidget()
        self.stack.addWidget(ImportView())
        self.stack.addWidget(ProcessView())
        self.stack.addWidget(ExportView())

        layout.addWidget(self.stack)

    def switch_view(self, index: int):
        """Switch the visible view to match the sidebar selection."""
        self.stack.setCurrentIndex(index)
