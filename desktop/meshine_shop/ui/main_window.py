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
from meshine_shop.ui.drop_zone import DropZone, collect_images_from_paths
from meshine_shop.ui.processing_queue import ProcessingQueue
from meshine_shop.core.pipeline import EXPORT_FORMATS, QUALITY_PRESETS
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
    # of file paths and the selected quality preset string.
    start_requested = Signal(list, str)

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

        # "Browse Folder" button — alternative to drag-and-drop for users
        # who prefer a native directory picker. Opens QFileDialog to select
        # a folder, then scans it for supported images the same way as a
        # folder drop.
        self._browse_btn = QPushButton("Browse Folder")
        self._browse_btn.setObjectName("browse_button")
        self._browse_btn.setFixedSize(200, 36)
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        layout.addWidget(self._browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Quality preset selector — lets the user choose a target polygon
        # budget before starting the pipeline. The preset determines how
        # aggressively the mesh decimation stage reduces triangle count.
        # Placed directly after Browse Folder for even spacing between buttons.
        self._quality_combo = QComboBox()
        self._quality_combo.setObjectName("quality_combo")
        for label in QUALITY_PRESETS:
            self._quality_combo.addItem(label)
        # Default to PC preset (index 1) — the middle-tier option.
        self._quality_combo.setCurrentIndex(1)
        self._quality_combo.setFixedWidth(200)
        layout.addWidget(self._quality_combo, alignment=Qt.AlignmentFlag.AlignCenter)

        # "Start Processing" button — disabled until files are dropped.
        # Clicking this triggers the full photogrammetry pipeline via
        # the start_requested signal, which the app layer listens for.
        self._start_btn = QPushButton("Start Processing")
        self._start_btn.setObjectName("start_button")
        self._start_btn.setEnabled(False)
        self._start_btn.setFixedSize(200, 36)
        self._start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self._start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # File count row — shows how many images are loaded, with a Clear
        # button to reset the selection if the user picked the wrong files.
        # Positioned below the three main controls (Browse / Quality / Start)
        # so they remain evenly spaced.
        file_count_row = QWidget()
        file_count_layout = QHBoxLayout(file_count_row)
        file_count_layout.setContentsMargins(0, 0, 0, 0)
        file_count_layout.setSpacing(12)

        file_count_layout.addStretch()

        self.file_count_label = QLabel("")
        self.file_count_label.setStyleSheet("color: #dc3545; font-size: 13px;")
        file_count_layout.addWidget(self.file_count_label)

        # "Clear" button — hidden until files are loaded. Resets the import
        # selection so the user can start over without navigating away.
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("clear_button")
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setFixedWidth(70)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        self._clear_btn.hide()
        file_count_layout.addWidget(self._clear_btn)

        file_count_layout.addStretch()
        layout.addWidget(file_count_row, alignment=Qt.AlignmentFlag.AlignCenter)

        # Stretch pushes all content to the top of the view.
        layout.addStretch()

        # Store dropped file paths for when the user clicks Start.
        self._pending_paths: list[str] = []

    def _on_browse_clicked(self):
        """Open a native directory picker and scan the selected folder for images."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Photo Folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        # User cancelled the dialog.
        if not folder:
            return

        # Scan the selected folder for supported image files, then feed
        # the results through the same handler as drag-and-drop.
        image_paths = collect_images_from_paths([folder])
        if image_paths:
            self._on_files_dropped(image_paths)
        else:
            # No images found — show a message instead of enabling Start.
            self.file_count_label.setText("No supported images found in folder")

    def _on_clear_clicked(self):
        """Clear the current file selection so the user can start over."""
        self._pending_paths = []
        self.file_count_label.setText("")
        self._clear_btn.hide()
        self._start_btn.setEnabled(False)

    def _on_files_dropped(self, paths):
        """Update the file count label, show Clear button, and enable Start."""
        self._pending_paths = paths
        count = len(paths)
        self.file_count_label.setText(
            f"{count} image{'s' if count != 1 else ''} ready for processing"
        )
        # Show the Clear button so the user can reset if they picked wrong files.
        self._clear_btn.show()
        # Enable the Start button now that we have files to process.
        self._start_btn.setEnabled(True)

    def _on_start_clicked(self):
        """Emit the start_requested signal with paths and quality preset."""
        if self._pending_paths:
            # Disable the button to prevent double-clicks while processing.
            self._start_btn.setEnabled(False)
            self._start_btn.setText("Processing...")
            # Include the selected quality preset so the worker knows
            # the target triangle count for the decimation stage.
            preset = self._quality_combo.currentText()
            self.start_requested.emit(self._pending_paths, preset)

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
        self._clear_btn.hide()
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

        # Section header — matches the other views' layout style.
        header = QLabel("Process")
        header.setObjectName("section_title")
        layout.addWidget(header)

        # Stretch before and after the queue — centers it vertically on the
        # page, matching the Export view's content positioning.
        layout.addStretch()

        # The queue widget manages its own layout and displays
        # real-time pipeline stage progress via signal connections.
        # AlignHCenter centers it within the content area.
        self.queue = ProcessingQueue()
        layout.addWidget(self.queue, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch()


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

    # Emitted when the user confirms an export. Carries the source mesh path
    # (OBJ with UVs from Phase 2b, or PLY fallback) and the destination path
    # (with the chosen format extension).
    export_requested = Signal(str, str)

    # Emitted when the user clicks Reset to clear the pipeline and start
    # a new job from scratch. The app layer handles cancellation, UI reset,
    # and switching back to the Import view.
    reset_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Section header — always visible.
        header = QLabel("Export")
        header.setObjectName("section_title")
        layout.addWidget(header)

        # Stretch before content — pushes everything to vertical center,
        # mirroring how the drop zone fills space on the Import page.
        layout.addStretch()

        # Placeholder — shown when no mesh has been processed yet.
        # Do NOT pass alignment to addWidget here — combining wordWrap with
        # addWidget alignment causes Qt to constrain the label to a tiny width
        # and cut the text off. Instead, the label fills the available width
        # and its own AlignCenter centers the text within that space.
        self._placeholder = QLabel(
            "Export settings will appear here once an asset has been processed."
        )
        self._placeholder.setStyleSheet("color: #999999; font-size: 13px;")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder)

        # Mesh stats — hidden until set_mesh_ready() is called.
        self._vertices_label = QLabel("")
        self._vertices_label.setObjectName("mesh_info")
        self._vertices_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._vertices_label.hide()
        layout.addWidget(self._vertices_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._triangles_label = QLabel("")
        self._triangles_label.setObjectName("mesh_info")
        self._triangles_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._triangles_label.hide()
        layout.addWidget(self._triangles_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._filesize_label = QLabel("")
        self._filesize_label.setObjectName("mesh_info")
        self._filesize_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._filesize_label.hide()
        layout.addWidget(self._filesize_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Texture availability indicator — shows which PBR maps were baked
        # during Phase 2c. Hidden until set_mesh_ready() confirms they exist.
        self._textures_label = QLabel("")
        self._textures_label.setObjectName("mesh_info")
        self._textures_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._textures_label.hide()
        layout.addWidget(self._textures_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Format selector — centered dropdown without a label, matching
        # the quality preset dropdown on the Import page.
        self._format_combo = QComboBox()
        self._format_combo.setObjectName("format_combo")
        for label in EXPORT_FORMATS:
            self._format_combo.addItem(label)
        self._format_combo.setFixedWidth(200)
        self._format_combo.hide()
        layout.addWidget(self._format_combo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Export button — same fixed size as Start Processing button.
        self._export_btn = QPushButton("Choose Location && Export")
        self._export_btn.setObjectName("export_button")
        self._export_btn.setFixedSize(200, 36)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_btn.hide()
        layout.addWidget(self._export_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Reset button — returns to Import view for a fresh job.
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setObjectName("reset_button")
        self._reset_btn.setFixedSize(200, 36)
        self._reset_btn.clicked.connect(lambda: self.reset_requested.emit())
        self._reset_btn.hide()
        layout.addWidget(self._reset_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Feedback label — success or error message after export attempt.
        # Do NOT pass alignment to addWidget here — combining wordWrap with
        # addWidget alignment causes Qt to constrain the label to a tiny width
        # and cut the text off. The label's own setAlignment centers the text.
        self._feedback = QLabel("")
        self._feedback.setObjectName("export_feedback")
        self._feedback.setWordWrap(True)
        self._feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._feedback)

        # Stretch after content — balances the top stretch to keep
        # everything vertically centered on the page.
        layout.addStretch()

        # Store the source mesh path once the pipeline provides it.
        # Phase 2b: will be meshed_uv.obj (OBJ with UV coords) if available,
        # otherwise falls back to meshed.ply (decimated-only PLY).
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
        # Prefer the UV-unwrapped OBJ (Phase 2b output) — it carries UV
        # coordinates needed for texture baking. Fall back to raw PLY.
        mesh_uv_obj = workspace.mesh / "meshed_uv.obj"
        mesh_ply = workspace.mesh / "meshed.ply"
        self._source_ply = mesh_uv_obj if mesh_uv_obj.exists() else mesh_ply

        # Populate each stat on its own centered line inside the info panel.
        self._vertices_label.setText(f"Vertices: {mesh_info['vertices']:,}")
        self._triangles_label.setText(f"Triangles: {mesh_info['triangles']:,}")
        self._filesize_label.setText(f"File size: {mesh_info['file_size_mb']} MB")

        # Check which PBR texture maps were baked during Phase 2c + 2d.
        # Report which maps are present so the user knows the export will
        # be textured (or untextured if baking failed/was skipped).
        if hasattr(workspace, "textures"):
            baked_maps = []
            if (workspace.textures / "albedo.png").exists():
                baked_maps.append("Albedo")
            if (workspace.textures / "normal.png").exists():
                baked_maps.append("Normal")
            if (workspace.textures / "ao.png").exists():
                baked_maps.append("AO")
            if (workspace.textures / "roughness.png").exists():
                baked_maps.append("Roughness")
            if (workspace.textures / "metallic.png").exists():
                baked_maps.append("Metallic")

            if baked_maps:
                self._textures_label.setText(
                    f"Textures: {', '.join(baked_maps)}"
                )
                self._textures_label.show()
            else:
                self._textures_label.hide()
        else:
            self._textures_label.hide()

        # Switch from placeholder to active state — show stats in the
        # panel and reveal the controls below it.
        self._placeholder.hide()
        self._vertices_label.show()
        self._triangles_label.show()
        self._filesize_label.show()
        self._format_combo.show()
        self._export_btn.show()
        self._reset_btn.show()
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
        self._vertices_label.hide()
        self._triangles_label.hide()
        self._filesize_label.hide()
        self._textures_label.hide()
        self._format_combo.hide()
        self._export_btn.hide()
        self._reset_btn.hide()
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
