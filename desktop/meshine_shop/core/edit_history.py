"""
Undo/Redo snapshot stack for Meshine Shop's viewport editing operations.

EditHistory maintains two stacks — undos and redos — where each entry is a
_Snapshot capturing the full editor state at a single point in time. Snapshots
are taken before any destructive operation (paint stroke, sculpt stroke, or
mesh operation) so every change can be reversed with a single undo.

Design decisions:
    - deque(maxlen=50): the oldest snapshot is silently dropped when the stack
      is full, keeping memory bounded without manual eviction logic.
    - geometry=True/False flag: paint-only operations skip copying vertex/face
      arrays (they're unchanged) so the snapshot only carries the PIL image.
      This halves memory cost for the most frequent operation (brush painting).
    - Redos are cleared on every new push_snapshot() call because a new edit
      branches the timeline — the old redo chain is no longer reachable.
    - No Qt or OpenGL imports — this is pure Python/numpy/PIL so it can be
      unit-tested independently of the GUI.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Maximum number of undo steps kept in memory.
# At ~9 MB per full snapshot (50k-vertex mesh + 1024×1024 texture) this caps
# memory usage at roughly 450 MB worst-case, which is acceptable on modern hardware.
MAX_HISTORY: int = 50


@dataclass
class _Snapshot:
    """
    Immutable record of editor state at one point in time.

    geometry_included indicates whether vertex/face/normal arrays are populated.
    When False (paint-only snapshot), the mesh geometry fields are all None and
    only the albedo image is restored on undo.
    """
    # Whether the geometry arrays are populated in this snapshot.
    geometry_included: bool

    # Geometry state — copies of the mesh arrays at snapshot time.
    # All None when geometry_included=False (paint-only snapshot).
    vertices: np.ndarray | None   # (N, 3) float32
    faces:    np.ndarray | None   # (M, 3) uint32
    normals:  np.ndarray | None   # (N, 3) float32

    # Texture state — always included regardless of geometry_included.
    albedo: Image.Image            # PIL Image copy


class EditHistory:
    """
    Undo/redo snapshot stack for viewport editing operations.

    Usage:
        history = EditHistory()

        # Before any destructive operation:
        history.push_snapshot(painter, geometry=False)   # paint-only
        history.push_snapshot(painter, geometry=True)    # mesh-changing

        # Undo / Redo:
        snap = history.undo()
        if snap:
            painter.restore_snapshot(snap)
            viewport.reupload_from_painter()

        snap = history.redo()
        if snap:
            painter.restore_snapshot(snap)
            viewport.reupload_from_painter()
    """

    def __init__(self) -> None:
        # Undo stack — deque with maxlen so oldest entries auto-drop when full
        self._undos: deque[_Snapshot] = deque(maxlen=MAX_HISTORY)
        # Redo stack — plain list, cleared whenever a new snapshot is pushed
        self._redos: list[_Snapshot] = []

    # ------------------------------------------------------------------ #
    # State queries                                                        #
    # ------------------------------------------------------------------ #

    @property
    def can_undo(self) -> bool:
        """True when at least one undo step is available."""
        return len(self._undos) > 0

    @property
    def can_redo(self) -> bool:
        """True when at least one redo step is available."""
        return len(self._redos) > 0

    # ------------------------------------------------------------------ #
    # Snapshot creation                                                    #
    # ------------------------------------------------------------------ #

    def push_snapshot(self, painter, geometry: bool = True) -> None:
        """
        Capture the current painter state and push it onto the undo stack.

        Call this BEFORE any destructive operation so the pre-edit state is
        saved and can be restored by undo().

        Parameters
        ----------
        painter : MeshPainter
            The active painter instance — its get_snapshot_data() method is
            called to retrieve deep copies of the current state.
        geometry : bool
            True  → snapshot vertices, faces, normals + texture (sculpt / mesh ops)
            False → snapshot texture only (brush paint / region fill)
        """
        verts, faces, norms, albedo = painter.get_snapshot_data(geometry=geometry)

        snap = _Snapshot(
            geometry_included=geometry,
            vertices=verts,
            faces=faces,
            normals=norms,
            albedo=albedo,
        )
        self._undos.append(snap)

        # Any new edit invalidates the redo chain — future can't be predicted
        # once the timeline branches.
        self._redos.clear()

        log.debug(
            "EditHistory: pushed snapshot (geometry=%s, undo_depth=%d)",
            geometry, len(self._undos),
        )

    # ------------------------------------------------------------------ #
    # Undo / Redo                                                          #
    # ------------------------------------------------------------------ #

    def undo(self) -> _Snapshot | None:
        """
        Pop the most recent snapshot from the undo stack.

        The caller is responsible for capturing the CURRENT state onto the
        redo stack before calling this, so the operation can be re-applied.
        Returns the snapshot to restore, or None if nothing to undo.
        """
        if not self._undos:
            log.debug("EditHistory: nothing to undo")
            return None

        snap = self._undos.pop()
        log.debug(
            "EditHistory: undo (undo_depth=%d, redo_depth=%d)",
            len(self._undos), len(self._redos),
        )
        return snap

    def redo(self) -> _Snapshot | None:
        """
        Pop the most recent snapshot from the redo stack.

        Returns the snapshot to restore, or None if nothing to redo.
        """
        if not self._redos:
            log.debug("EditHistory: nothing to redo")
            return None

        snap = self._redos.pop()
        log.debug(
            "EditHistory: redo (undo_depth=%d, redo_depth=%d)",
            len(self._undos), len(self._redos),
        )
        return snap

    def push_redo(self, painter, geometry: bool = True) -> None:
        """
        Capture the current painter state onto the REDO stack.

        Called by ViewportWidget.undo() just before restoring the undo snapshot,
        so the undone state is preserved for redo.
        """
        verts, faces, norms, albedo = painter.get_snapshot_data(geometry=geometry)
        snap = _Snapshot(
            geometry_included=geometry,
            vertices=verts,
            faces=faces,
            normals=norms,
            albedo=albedo,
        )
        self._redos.append(snap)

    def push_undo_only(self, painter, geometry: bool = True) -> None:
        """
        Capture the current painter state onto the UNDO stack WITHOUT clearing redo.

        Used by ViewportWidget.redo() to save the current state before applying
        a redo snapshot. Using push_snapshot() would clear _redos before we could
        pop from it, breaking redo entirely.
        """
        verts, faces, norms, albedo = painter.get_snapshot_data(geometry=geometry)
        snap = _Snapshot(
            geometry_included=geometry,
            vertices=verts,
            faces=faces,
            normals=norms,
            albedo=albedo,
        )
        self._undos.append(snap)
        log.debug(
            "EditHistory: push_undo_only (undo_depth=%d, redo_depth=%d)",
            len(self._undos), len(self._redos),
        )

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Clear all undo and redo history. Called when a new mesh is loaded."""
        self._undos.clear()
        self._redos.clear()
        log.debug("EditHistory: history cleared")
