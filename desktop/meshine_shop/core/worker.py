"""
Background pipeline worker for Meshine Shop.

This module provides the QThread subclass that runs the photogrammetry
pipeline on a background thread, keeping the UI responsive during
what can be a minutes-to-hours processing operation.

Communication between the worker thread and the UI main thread is handled
entirely through Qt signals. Qt's signal/slot mechanism automatically
marshals cross-thread signals via QueuedConnection (the default when
sender and receiver live on different threads), so:
    - The worker never touches any widget directly
    - Slot methods in the UI always execute on the main thread
    - No explicit mutex, lock, or QMetaObject.invokeMethod is needed

The worker receives a ReconstructionEngine via constructor injection
rather than creating one itself. This keeps engine creation in the UI
layer and makes the worker testable with mock engines.
"""

from pathlib import Path
from PySide6.QtCore import QThread, Signal

from meshine_shop.core.pipeline import PipelineStage, STAGE_ORDER, QUALITY_PRESETS
from meshine_shop.core.engine import ReconstructionEngine
from meshine_shop.core.workspace import WorkspacePaths


class PipelineWorker(QThread):
    """
    Runs the full photogrammetry pipeline on a background thread.

    Iterates through each stage in STAGE_ORDER, calling the corresponding
    engine method. Emits signals before, during, and after each stage so
    the UI can update in real-time.

    Signals:
        stage_started(str)    — Emitted when a stage begins. Payload is the stage name.
        stage_completed(str)  — Emitted when a stage finishes successfully.
        progress(str, str)    — (stage_name, message) for status updates within a stage.
        error(str, str)       — (stage_name, error_message) when a stage fails.
        pipeline_finished()   — Emitted when ALL stages complete successfully.
    """

    stage_started = Signal(str)
    stage_completed = Signal(str)
    progress = Signal(str, str)
    error = Signal(str, str)
    pipeline_finished = Signal()

    def __init__(self, engine: ReconstructionEngine,
                 image_paths: list[str], workspace: WorkspacePaths,
                 quality_preset: str = "PC (25K triangles)"):
        super().__init__()
        self._engine = engine
        self._image_paths = image_paths
        self._workspace = workspace

        # Look up the target triangle count from the quality preset.
        # This value is passed to the engine's decimate() method.
        self._target_faces = QUALITY_PRESETS.get(quality_preset, 25_000)

        # Cooperative cancellation flag. Checked between stages (not mid-stage).
        # Python's GIL makes single boolean reads/writes thread-safe, so no
        # explicit lock is needed for this flag.
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the pipeline. Takes effect between stages."""
        self._cancelled = True

    def run(self):
        """
        Execute each pipeline stage in sequence.

        This method runs on the BACKGROUND THREAD — never access Qt widgets
        from here. All UI communication goes through signal emissions, which
        Qt automatically marshals to the main thread.

        On error, the pipeline stops at the failing stage and emits the error
        signal. It does NOT continue to subsequent stages.
        """
        # Map each stage constant to its corresponding engine method.
        # The ingest stage is special — it also needs the image paths,
        # so it gets a lambda wrapper that includes them. The decimation
        # stage similarly needs the target face count from the quality preset.
        # All other stages take (workspace, on_progress) directly.
        stage_methods = {
            PipelineStage.INGEST: lambda ws, cb: self._engine.ingest(
                self._image_paths, ws, cb
            ),
            PipelineStage.FEATURES: self._engine.extract_features,
            PipelineStage.SPARSE: self._engine.sparse_reconstruct,
            PipelineStage.DENSE: self._engine.dense_reconstruct,
            PipelineStage.MESH: self._engine.mesh_reconstruct,
            PipelineStage.TEXTURE: self._engine.texture_map,
            PipelineStage.DECIMATION: lambda ws, cb: self._engine.decimate(
                ws, cb, self._target_faces
            ),
            # UV unwrapping reads the decimated PLY and writes meshed_uv.obj
            # with xatlas UV coordinates. Required for Phase 2c texture baking.
            PipelineStage.UV_UNWRAP: self._engine.unwrap_uv,
            # Texture baking reads meshed_uv.obj and writes albedo.png,
            # normal.png, ao.png to workspace.textures/. Non-fatal: the
            # engine implementation logs a warning and continues on failure.
            PipelineStage.TEXTURE_BAKE: self._engine.bake_textures,
        }

        for stage in STAGE_ORDER:
            # Check for cancellation between stages.
            if self._cancelled:
                return

            # Notify the UI that this stage is starting.
            self.stage_started.emit(stage)

            try:
                # Call the engine method for this stage.
                # Each method receives the workspace and a progress callback.
                method = stage_methods[stage]
                method(self._workspace, self._make_progress_callback(stage))

                # Notify the UI that this stage completed successfully.
                self.stage_completed.emit(stage)

            except Exception as e:
                # Any exception stops the pipeline. The error signal carries
                # the stage name and a human-readable error message.
                self.error.emit(stage, str(e))
                return

        # All stages completed without error.
        self.pipeline_finished.emit()

    def _make_progress_callback(self, stage: str):
        """
        Create a progress callback closure for the given stage.

        Returns a function that, when called with a message string,
        emits the progress signal with the stage name prepended.
        This way engine methods don't need to know their own stage name.
        """
        def callback(message: str):
            self.progress.emit(stage, message)
        return callback
