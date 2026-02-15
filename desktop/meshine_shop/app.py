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

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QStatusBar
from PySide6.QtCore import Qt

from meshine_shop.ui.sidebar import Sidebar
from meshine_shop.ui.main_window import MainContent
from meshine_shop.ui.styles import DARK_THEME
from meshine_shop.core.engine import ColmapEngine
from meshine_shop.core.worker import PipelineWorker
from meshine_shop.core.workspace import create_workspace
from meshine_shop.core.pipeline import STAGE_DISPLAY_NAMES


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

        # The sidebar is fixed-width (220px); the main content stretches to
        # fill remaining space (stretch factor of 1).
        layout.addWidget(self.sidebar)
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

        # Connect the Reset button in the processing queue to the reset handler.
        # This lets the user cancel a running pipeline and return to the Import view.
        self.main_content.process_view.queue.reset_requested.connect(self._reset_pipeline)

        # Keep a reference to the active worker to prevent garbage collection.
        # QThread must be stored as an instance attribute or it gets destroyed
        # when the local variable goes out of scope.
        self._worker: PipelineWorker | None = None

    def _start_pipeline(self, image_paths: list[str]):
        """
        Create a workspace, engine, and worker, then start the pipeline.

        This method is called when the user clicks "Start Processing" in
        the Import view. It:
        1. Creates a fresh workspace directory for this reconstruction job
        2. Instantiates the COLMAP engine
        3. Creates a PipelineWorker on a background thread
        4. Connects all worker signals to the UI
        5. Starts the worker thread

        The worker runs on a background thread so the UI stays responsive.
        All communication back to the UI happens via Qt signals, which are
        automatically marshaled to the main thread.
        """
        # Create a fresh workspace for this job.
        workspace = create_workspace()
        self.statusBar().showMessage(f"Workspace: {workspace.root}")

        # Instantiate the COLMAP engine.
        engine = ColmapEngine()

        # Create the background worker.
        self._worker = PipelineWorker(engine, image_paths, workspace)

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

        # Re-enable the Start button so the user can start a new job.
        self.main_content.import_view.reset_start_button()

        # Switch back to the Import view.
        self.main_content.switch_view(0)
        self.sidebar.button_group.button(0).setChecked(True)

        # Update the status bar.
        self.statusBar().showMessage("Ready")
