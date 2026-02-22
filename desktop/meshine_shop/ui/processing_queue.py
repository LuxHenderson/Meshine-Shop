"""
Processing queue widget for Meshine Shop.

Displays the real-time status of the photogrammetry pipeline as it runs.
Each pipeline stage gets its own row with a status indicator, stage name,
and a message showing what's currently happening.

The queue toggles between two states:
    - Empty state: Shows a placeholder message when no job is running
    - Active state: Shows stage rows with live progress updates

Public methods (called by the app via signal connections):
    start_pipeline()           — Switch to active state, reset all rows
    set_stage_started(stage)   — Mark a stage as currently running
    set_stage_completed(stage) — Mark a stage as done
    set_stage_error(stage, msg)— Mark a stage as failed
    set_progress(stage, msg)   — Update the status message for a stage
    set_finished()             — Show completion banner
    reset()                    — Return to empty state

Signals:
    reset_requested            — Emitted when the user clicks the Reset button
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal

from meshine_shop.core.pipeline import STAGE_ORDER, STAGE_DISPLAY_NAMES


class StageRow(QWidget):
    """
    A single row in the processing queue representing one pipeline stage.

    Each row displays:
        - A status indicator (unicode character: pending, running, done, error)
        - The stage's human-readable display name
        - A status message that updates as the stage progresses

    This is a private implementation detail of ProcessingQueue — it is not
    used elsewhere in the app.
    """

    # Unicode characters for each status state.
    # These render cleanly across macOS and Windows without requiring icon files.
    ICON_PENDING = "\u25CB"   # ○ hollow circle
    ICON_RUNNING = "\u25CF"   # ● filled circle
    ICON_DONE = "\u2713"      # ✓ checkmark
    ICON_ERROR = "\u2717"     # ✗ cross mark

    def __init__(self, stage_key: str):
        super().__init__()
        self.setObjectName("stage_row")
        self._stage_key = stage_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)

        # Status indicator — a single character that changes based on state.
        self._icon = QLabel(self.ICON_PENDING)
        self._icon.setObjectName("stage_status_pending")
        self._icon.setFixedWidth(20)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon)

        # Stage name — the human-readable label (e.g., "Extracting Features").
        self._name = QLabel(STAGE_DISPLAY_NAMES.get(stage_key, stage_key))
        self._name.setObjectName("stage_name")
        self._name.setFixedWidth(180)
        layout.addWidget(self._name)

        # Status message — updates in real-time as the engine reports progress.
        self._message = QLabel("")
        self._message.setObjectName("stage_message")
        self._message.setStyleSheet("color: #999999;")
        layout.addWidget(self._message, 1)

    def set_pending(self):
        """Reset to pending state (stage hasn't started yet)."""
        self._icon.setText(self.ICON_PENDING)
        self._icon.setObjectName("stage_status_pending")
        # Force QSS re-evaluation after changing the object name.
        self._icon.setStyleSheet("")
        self._message.setText("")

    def set_running(self):
        """Mark as currently running (crimson indicator)."""
        self._icon.setText(self.ICON_RUNNING)
        self._icon.setObjectName("stage_status_running")
        self._icon.setStyleSheet("")
        self._message.setText("Running...")

    def set_done(self):
        """Mark as successfully completed (green checkmark)."""
        self._icon.setText(self.ICON_DONE)
        self._icon.setObjectName("stage_status_done")
        self._icon.setStyleSheet("")

    def set_error(self, message: str):
        """Mark as failed with an error message."""
        self._icon.setText(self.ICON_ERROR)
        self._icon.setObjectName("stage_status_error")
        self._icon.setStyleSheet("")
        self._message.setText(message)
        self._message.setStyleSheet("color: #dc3545;")

    def set_message(self, message: str):
        """Update the status message text (used during running state)."""
        self._message.setText(message)


class ProcessingQueue(QWidget):
    """
    Container widget that displays pipeline progress or an empty state.

    Manages a collection of StageRow widgets (one per pipeline stage) and
    provides public methods for the app to update stage statuses in response
    to worker signals.
    """

    # Emitted when the user clicks the Reset button. The app layer listens
    # for this to cancel any running worker, re-enable the Start button,
    # and switch back to the Import view.
    reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("queue_panel")
        # Cap width so the queue can be horizontally centered in ProcessView
        # rather than stretching edge-to-edge across the content area.
        self.setMaximumWidth(480)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        # Empty state message — shown when no job is running. Matches the
        # Export view's placeholder style: left-aligned, muted gray text.
        self._empty_msg = QLabel(
            "No jobs in the queue. Import a photo set to get started."
        )
        self._empty_msg.setStyleSheet("color: #999999;")
        self._empty_msg.setWordWrap(True)
        self._layout.addWidget(self._empty_msg)

        # Stage rows container — hidden until a pipeline starts.
        self._stages_container = QWidget()
        stages_layout = QVBoxLayout(self._stages_container)
        stages_layout.setContentsMargins(0, 8, 0, 0)
        stages_layout.setSpacing(2)

        # Create a StageRow for each pipeline stage.
        self._stage_rows: dict[str, StageRow] = {}
        for stage_key in STAGE_ORDER:
            row = StageRow(stage_key)
            self._stage_rows[stage_key] = row
            stages_layout.addWidget(row)

        self._layout.addWidget(self._stages_container)
        self._stages_container.hide()

        # Completion banner — shown when all stages finish.
        self._finished_label = QLabel("Pipeline complete!")
        self._finished_label.setObjectName("pipeline_complete")
        self._finished_label.setStyleSheet(
            "color: #28a745; font-size: 14px; font-weight: 600; padding: 8px 0;"
        )
        self._finished_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._finished_label.hide()
        self._layout.addWidget(self._finished_label)

        self._layout.addStretch()

    def start_pipeline(self):
        """Switch from empty state to active state and reset all stage rows."""
        self._empty_msg.hide()
        self._finished_label.hide()
        self._stages_container.show()

        # Reset all rows to pending.
        for row in self._stage_rows.values():
            row.set_pending()

    def set_stage_started(self, stage: str):
        """Mark a stage as currently running."""
        if stage in self._stage_rows:
            self._stage_rows[stage].set_running()

    def set_stage_completed(self, stage: str):
        """Mark a stage as successfully completed."""
        if stage in self._stage_rows:
            self._stage_rows[stage].set_done()

    def set_stage_error(self, stage: str, message: str):
        """Mark a stage as failed with an error message."""
        if stage in self._stage_rows:
            self._stage_rows[stage].set_error(message)

    def set_progress(self, stage: str, message: str):
        """Update the status message for a running stage."""
        if stage in self._stage_rows:
            self._stage_rows[stage].set_message(message)

    def set_finished(self):
        """Show the completion banner below the stage rows."""
        self._finished_label.show()

    def reset(self):
        """Return the queue to its empty state.

        Hides the stage rows and completion banner. Shows the empty state
        placeholder message. Called by the app layer when the user resets
        from the Export view.
        """
        self._stages_container.hide()
        self._finished_label.hide()
        self._empty_msg.show()

        # Reset all rows so they're clean for the next pipeline run.
        for row in self._stage_rows.values():
            row.set_pending()
