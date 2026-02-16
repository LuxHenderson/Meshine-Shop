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

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget,
    QPushButton, QComboBox, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from meshine_shop.ui.drop_zone import DropZone
from meshine_shop.ui.processing_queue import ProcessingQueue
from meshine_shop.core.pipeline import EXPORT_FORMATS
from meshine_shop.core.workspace import WorkspacePaths


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

    def reset(self):
        """Clear all imported files and return to the initial empty state.

        Called by the app layer when the user resets the pipeline, so the
        Import view is clean for a fresh job — no stale file list lingering.
        """
        self._pending_paths = []
        self.file_count_label.setText("")
        self._start_btn.setEnabled(False)
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
    The third step — exporting the reconstructed mesh.

    Toggles between two states:
        - Placeholder: shown when no mesh is available yet
        - Active: shows mesh info, format selector, and export button

    After the pipeline completes, the app calls set_mesh_ready() with the
    workspace paths and mesh stats. The user picks a format, clicks Export,
    and a save dialog lets them choose where to save the file.

    Signal flow:
        1. App calls set_mesh_ready(workspace, mesh_info) after pipeline completes
        2. User selects format from dropdown and clicks "Export"
        3. QFileDialog opens for the user to choose a save location
        4. ExportView emits export_requested(source_ply, dest_path)
        5. App layer calls the exporter and reports success/error back
    """

    # Emitted when the user confirms an export. Carries the source PLY path
    # and the destination path (with the chosen format extension).
    export_requested = Signal(str, str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Section header — always visible.
        header = QLabel("Export")
        header.setObjectName("section_title")
        layout.addWidget(header)

        # --- Placeholder state ---
        # Shown when no mesh has been processed yet.
        self._placeholder = QLabel(
            "Export settings will appear here once an asset has been processed."
        )
        self._placeholder.setStyleSheet("color: #999999;")
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        # --- Active state container ---
        # Contains mesh info, format selector, export button, and feedback.
        # Hidden until set_mesh_ready() is called.
        self._active_container = QWidget()
        active_layout = QVBoxLayout(self._active_container)
        active_layout.setContentsMargins(0, 0, 0, 0)
        active_layout.setSpacing(16)

        # Mesh info panel — shows vertex count, triangle count, file size.
        # Gives the user confidence that the pipeline produced valid output
        # before they commit to exporting.
        self._mesh_info = QLabel("")
        self._mesh_info.setObjectName("mesh_info")
        self._mesh_info.setWordWrap(True)
        active_layout.addWidget(self._mesh_info)

        # Format selector row — dropdown + label on one line.
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(12)

        format_label = QLabel("Format:")
        format_label.setObjectName("export_label")
        format_label.setFixedWidth(60)
        format_layout.addWidget(format_label)

        # QComboBox populated with the export formats defined in pipeline.py.
        # The user selects a format before clicking Export.
        self._format_combo = QComboBox()
        self._format_combo.setObjectName("format_combo")
        for label in EXPORT_FORMATS:
            self._format_combo.addItem(label)
        self._format_combo.setFixedWidth(200)
        format_layout.addWidget(self._format_combo)

        format_layout.addStretch()
        active_layout.addWidget(format_row)

        # Export button — opens a save dialog then emits export_requested.
        # Uses the same crimson accent as the Start Processing button so
        # the primary action is always visually consistent.
        self._export_btn = QPushButton("Choose Location && Export")
        self._export_btn.setObjectName("export_button")
        self._export_btn.setFixedHeight(44)
        self._export_btn.clicked.connect(self._on_export_clicked)
        active_layout.addWidget(self._export_btn)

        # Feedback label — shows success or error messages after export.
        self._feedback = QLabel("")
        self._feedback.setObjectName("export_feedback")
        self._feedback.setWordWrap(True)
        active_layout.addWidget(self._feedback)

        active_layout.addStretch()
        layout.addWidget(self._active_container)
        self._active_container.hide()

        layout.addStretch()

        # Store the source PLY path once the pipeline provides it.
        self._source_ply: Path | None = None

    def set_mesh_ready(self, workspace: WorkspacePaths, mesh_info: dict):
        """
        Switch from placeholder to active state with mesh details.

        Called by the app layer after the pipeline completes successfully.
        Populates the mesh info label and enables the export controls.

        Args:
            workspace: The workspace containing the pipeline output.
            mesh_info: Dict with 'vertices', 'triangles', 'file_size_mb' keys.
        """
        self._source_ply = workspace.mesh / "meshed.ply"

        # Format mesh stats into a readable summary.
        self._mesh_info.setText(
            f"Vertices: {mesh_info['vertices']:,}    "
            f"Triangles: {mesh_info['triangles']:,}    "
            f"File size: {mesh_info['file_size_mb']} MB"
        )

        # Switch to active state.
        self._placeholder.hide()
        self._active_container.show()
        self._feedback.setText("")

    def set_export_success(self, dest_path: str):
        """Show a success message after a successful export."""
        self._feedback.setStyleSheet("color: #28a745; font-size: 13px;")
        self._feedback.setText(f"Exported to: {dest_path}")

    def set_export_error(self, message: str):
        """Show an error message if export fails."""
        self._feedback.setStyleSheet("color: #dc3545; font-size: 13px;")
        self._feedback.setText(f"Export failed: {message}")

    def reset(self):
        """Return to placeholder state. Called when the pipeline is reset."""
        self._source_ply = None
        self._active_container.hide()
        self._placeholder.show()
        self._feedback.setText("")

    def _on_export_clicked(self):
        """Open a save dialog and emit export_requested with the chosen path."""
        if self._source_ply is None:
            return

        # Get the selected format's file extension from the pipeline constants.
        selected_label = self._format_combo.currentText()
        extension = EXPORT_FORMATS[selected_label]

        # Build the file filter string for the save dialog.
        # e.g., "OBJ Files (*.obj)" or "glTF Binary Files (*.glb)"
        format_name = selected_label.split(" (")[0]
        file_filter = f"{format_name} Files (*{extension})"

        # Default filename based on the format.
        default_name = f"mesh{extension}"

        # Open a native save dialog so the user picks where to save.
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mesh",
            default_name,
            file_filter,
        )

        # User cancelled the dialog.
        if not dest_path:
            return

        # Ensure the file has the correct extension (some OS dialogs
        # don't append it automatically).
        if not dest_path.lower().endswith(extension):
            dest_path += extension

        # Emit the signal for the app layer to handle the actual export.
        self.export_requested.emit(str(self._source_ply), dest_path)


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
