"""
Main application window for Meshine Shop.

This module defines the top-level QMainWindow that serves as the shell
for the entire application. It assembles the core layout:
    - A fixed-width sidebar for navigation (Import / Process / Export)
    - A main content area that swaps views based on sidebar selection
    - A status bar for real-time feedback

Beyond layout, this is the orchestration hub for the pipeline:
    1. ImportView emits start_requested(paths) when the user clicks Start
    2. This module creates a workspace, engine, and worker
    3. Worker signals are connected to the ProcessingQueue for live updates
    4. The status bar reflects current pipeline state

The charcoal + crimson dark theme is applied globally here via QSS
(Qt Style Sheets), which styles every child widget in the window.
This approach was chosen over per-widget styling because it keeps
all visual design centralized in one place (styles.py) and makes
future theme changes trivial.
"""

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QStatusBar, QFrame
from PySide6.QtCore import Qt

from meshine_shop.ui.sidebar import Sidebar
from meshine_shop.ui.main_window import MainContent
from meshine_shop.ui.styles import DARK_THEME
from meshine_shop.core.engine_factory import create_best_engine
from meshine_shop.core.worker import PipelineWorker
from meshine_shop.core.workspace import create_workspace
from meshine_shop.core.pipeline import STAGE_DISPLAY_NAMES
from meshine_shop.core.exporter import load_mesh_info, export_mesh


class MeshineShopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meshine Shop")

        # Minimum size prevents the window from being resized too small
        # to display the UI properly. Default size of 1280x800 provides
        # a comfortable workspace without being overwhelming.
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # Apply the global dark theme stylesheet to every widget in the window.
        # See ui/styles.py for the full QSS definition.
        self.setStyleSheet(DARK_THEME)

        # Central widget — required by QMainWindow as the root container.
        # All primary UI elements are children of this widget.
        central = QWidget()
        self.setCentralWidget(central)

        # Horizontal layout: sidebar on the left, main content on the right.
        # Zero margins and spacing ensure the sidebar and content area sit
        # flush against each other with no gaps.
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar handles navigation between the three main views.
        self.sidebar = Sidebar()
        # MainContent uses a QStackedWidget to swap between Import/Process/Export.
        self.main_content = MainContent()

        # Vertical separator — a 1px QFrame line that visually divides the
        # sidebar menu from the main content area. More reliable than the CSS
        # border-right approach because QFrame always renders at exactly 1px.
        separator = QFrame()
        separator.setObjectName("sidebar_separator")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setFixedWidth(1)

        # The sidebar is fixed-width (220px); the main content stretches to
        # fill remaining space (stretch factor of 1).
        layout.addWidget(self.sidebar)
        layout.addWidget(separator)
        layout.addWidget(self.main_content, 1)

        # Connect sidebar navigation clicks to the main content view switcher.
        # When a nav button is clicked, the sidebar emits nav_changed(index),
        # which triggers MainContent.switch_view(index) to swap the visible panel.
        self.sidebar.nav_changed.connect(self.main_content.switch_view)

        # Status bar provides real-time feedback at the bottom of the window.
        # It displays pipeline progress, error messages, and status updates.
        status_bar = QStatusBar()
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)

        # Connect the Import view's "Start Processing" button to the pipeline
        # orchestration method. This is the entry point for the entire pipeline.
        self.main_content.import_view.start_requested.connect(self._start_pipeline)

        # Connect the Reset button in the Export view to the reset handler.
        # This lets the user clear the current job and return to the Import view.
        self.main_content.export_view.reset_requested.connect(self._reset_pipeline)

        # Connect the Export view's export button to the export handler.
        self.main_content.export_view.export_requested.connect(self._run_export)

        # Keep a reference to the active worker to prevent garbage collection.
        # QThread must be stored as an instance attribute or it gets destroyed
        # when the local variable goes out of scope.
        self._worker: PipelineWorker | None = None

        # Store the workspace from the current pipeline run so the Export view
        # can locate the output mesh after the pipeline completes.
        self._workspace = None

    def _start_pipeline(self, image_paths: list[str],
                        quality_preset: str = "PC (65K triangles)"):
        """
        Create a workspace, engine, and worker, then start the pipeline.

        This method is called when the user clicks "Start Processing" in
        the Import view. It:
        1. Creates a fresh workspace directory for this reconstruction job
        2. Auto-selects the best available engine for this platform
        3. Creates a PipelineWorker on a background thread
        4. Connects all worker signals to the UI
        5. Starts the worker thread

        The engine factory detects available hardware and picks the best
        engine automatically:
            macOS + Apple Silicon → Apple Object Capture (Metal)
            Windows + NVIDIA     → COLMAP (CUDA)
            Fallback             → COLMAP (CPU)

        The worker runs on a background thread so the UI stays responsive.
        All communication back to the UI happens via Qt signals, which are
        automatically marshaled to the main thread.

        Args:
            image_paths:    List of absolute file path strings to process.
            quality_preset: Selected quality preset label from the Import view.
        """
        # Create a fresh workspace for this job and store it so the Export
        # view can find the output mesh after the pipeline completes.
        workspace = create_workspace()
        self._workspace = workspace

        # Auto-select the best engine for this platform.
        # Pass the quality preset so the factory can tune the Apple Object
        # Capture detail level (Mobile → reduced, PC/Cinematic → full).
        # The factory returns both the engine instance and a display name
        # so we can show the user which engine is processing their data.
        engine, engine_name = create_best_engine(quality_preset)
        self.statusBar().showMessage(f"Engine: {engine_name} | Workspace: {workspace.root}")

        # Create the background worker with the user's quality preset.
        # The preset determines the target triangle count for decimation.
        self._worker = PipelineWorker(engine, image_paths, workspace, quality_preset)

        # Get a reference to the processing queue for signal connections.
        queue = self.main_content.process_view.queue

        # Initialize the queue UI — switch from empty state to active state
        # with all stages showing as "pending."
        queue.start_pipeline()

        # --- Connect worker signals to the processing queue ---

        # When a stage starts, mark it as running in the queue.
        self._worker.stage_started.connect(queue.set_stage_started)

        # When a stage completes, mark it as done in the queue.
        self._worker.stage_completed.connect(queue.set_stage_completed)

        # Progress messages update the status text on the active stage row.
        self._worker.progress.connect(queue.set_progress)

        # Errors mark the stage as failed and display the error message.
        self._worker.error.connect(queue.set_stage_error)

        # When the full pipeline finishes, show the completion banner.
        self._worker.pipeline_finished.connect(queue.set_finished)

        # --- Connect worker signals to the status bar ---

        # Show the current stage name in the status bar as each stage starts.
        self._worker.stage_started.connect(
            lambda stage: self.statusBar().showMessage(
                f"Running: {STAGE_DISPLAY_NAMES.get(stage, stage)}"
            )
        )

        # Show completion in the status bar.
        self._worker.pipeline_finished.connect(
            lambda: self.statusBar().showMessage("Pipeline complete!")
        )

        # When the pipeline finishes, prepare the Export view with mesh info
        # and auto-switch to it so the user can export immediately.
        self._worker.pipeline_finished.connect(self._on_pipeline_finished)

        # Show errors in the status bar.
        self._worker.error.connect(
            lambda stage, msg: self.statusBar().showMessage(
                f"Error in {STAGE_DISPLAY_NAMES.get(stage, stage)}: {msg}"
            )
        )

        # --- Connect worker signals to post-pipeline actions ---

        # Re-enable the Start button after the pipeline finishes or errors.
        self._worker.pipeline_finished.connect(
            self.main_content.import_view.reset_start_button
        )
        self._worker.error.connect(
            lambda _stage, _msg: self.main_content.import_view.reset_start_button()
        )

        # Auto-switch to the Process view so the user can watch progress.
        self.main_content.switch_view(1)
        # Also update the sidebar to reflect the view change.
        self.sidebar.button_group.button(1).setChecked(True)

        # Start the worker thread — this calls worker.run() on the background thread.
        self._worker.start()

    def _on_pipeline_finished(self):
        """
        Prepare the Export view after the pipeline completes successfully.

        Loads mesh stats from the pipeline's output mesh and passes them to
        the Export view, then auto-switches to the Export tab so the user
        can export immediately without manually navigating.

        Prefers meshed_uv.obj (UV-unwrapped, Phase 2b output) over meshed.ply
        (decimated-only). The OBJ carries UV coordinates needed for texture
        baking in Phase 2c. Falls back to meshed.ply if UV unwrapping somehow
        didn't produce output.
        """
        if self._workspace is None:
            return

        # Prefer the UV-unwrapped OBJ produced by Phase 2b.
        # Fall back to the raw decimated PLY if the OBJ isn't present.
        mesh_uv_obj = self._workspace.mesh / "meshed_uv.obj"
        mesh_ply = self._workspace.mesh / "meshed.ply"

        if mesh_uv_obj.exists():
            mesh_file = mesh_uv_obj
        elif mesh_ply.exists():
            mesh_file = mesh_ply
        else:
            self.statusBar().showMessage("Pipeline complete, but no mesh file found.")
            return

        # Load mesh stats (vertex count, triangle count, file size) for display.
        mesh_info = load_mesh_info(mesh_file)

        # Populate the Export view with mesh details and switch to it.
        self.main_content.export_view.set_mesh_ready(self._workspace, mesh_info)
        self.main_content.switch_view(2)
        self.sidebar.button_group.button(2).setChecked(True)

    def _run_export(self, source_mesh: str, dest_path: str):
        """
        Execute the mesh export when the user clicks Export in the Export view.

        Converts the pipeline's mesh output to the user's chosen format (OBJ or
        glTF Binary) using trimesh, then reports success or failure back to the
        Export view. The source may be a UV-unwrapped OBJ (Phase 2b) or a raw
        PLY — trimesh handles both transparently.

        Args:
            source_mesh: Path to the pipeline's output mesh (OBJ or PLY).
            dest_path:   User-chosen save location with format extension.
        """
        from pathlib import Path

        from PySide6.QtWidgets import QMessageBox

        try:
            # Pass the workspace so the exporter can locate baked textures
            # (workspace/textures/albedo.png etc.) and embed them in the export.
            # export_mesh returns the actual output path — for GLB this is
            # dest_path itself; for OBJ/FBX with textures it's the bundle folder.
            output_path = export_mesh(Path(source_mesh), Path(dest_path), self._workspace)
            self.main_content.export_view.set_export_success(str(output_path))
            self.statusBar().showMessage(f"Exported to: {output_path}")
        except Exception as e:
            # Show the full error in an unmissable dialog as well as the
            # inline feedback label. This makes export failures easy to debug.
            self.main_content.export_view.set_export_error(str(e))
            self.statusBar().showMessage(f"Export failed: {e}")
            QMessageBox.critical(self, "Export Failed", str(e))

    def _reset_pipeline(self):
        """
        Cancel any running pipeline and reset the UI to its initial state.

        Called when the user clicks the Reset button in the processing queue.
        This method:
        1. Cancels the background worker (cooperative — takes effect between stages)
        2. Resets the processing queue to its empty state
        3. Re-enables the Start button on the Import view
        4. Switches back to the Import view
        5. Updates the status bar
        """
        # Cancel the worker if one is running. The cancellation flag is checked
        # between pipeline stages, so the current stage will finish before stopping.
        if self._worker is not None:
            self._worker.cancel()
            # Wait for the worker thread to finish (with a timeout to avoid freezing).
            self._worker.wait(5000)
            self._worker = None

        # Reset the processing queue back to its empty state.
        self.main_content.process_view.queue.reset()

        # Reset the Export view back to its placeholder state.
        self.main_content.export_view.reset()

        # Clear the stored workspace reference.
        self._workspace = None

        # Reset the Import view — clears dropped files, file count label,
        # and disables the Start button so the user starts fresh.
        self.main_content.import_view.reset()

        # Switch back to the Import view.
        self.main_content.switch_view(0)
        self.sidebar.button_group.button(0).setChecked(True)

        # Update the status bar.
        self.statusBar().showMessage("Ready")
