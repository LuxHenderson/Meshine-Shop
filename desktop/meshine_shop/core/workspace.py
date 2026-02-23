"""
Workspace directory manager for Meshine Shop.

Each reconstruction job needs a structured directory tree where COLMAP
stores its intermediate and output files. This module handles creating
that structure so neither the engine nor the UI has to construct paths
manually.

Workspaces are timestamped and stored under ~/MeshineShop/workspaces/
by default. Each workspace contains:
    images/   — source photos (copied or symlinked from the user's originals)
    sparse/   — sparse reconstruction output (camera poses, sparse point cloud)
    dense/    — dense reconstruction output (depth maps, fused point cloud)
    mesh/     — mesh files (meshed.ply, meshed_uv.obj, USDZ extraction)
    textures/ — baked PBR textures (Phase 2c): albedo.png, normal.png, ao.png
    database.db — COLMAP's SQLite database (features, matches, etc.)

Using a dedicated workspace per job keeps reconstructions isolated and
makes cleanup straightforward.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class WorkspacePaths:
    """
    Typed container for all paths within a reconstruction workspace.

    Using a dataclass instead of raw Path objects prevents path-construction
    bugs scattered across the codebase — every module references these
    named attributes instead of building paths with string concatenation.
    """
    root: Path       # Top-level workspace directory
    images: Path     # Source images directory
    database: Path   # COLMAP SQLite database file path
    sparse: Path     # Sparse reconstruction output directory
    dense: Path      # Dense reconstruction output directory
    mesh: Path       # Mesh and texture output directory
    textures: Path   # Baked texture maps (Phase 2c) — albedo.png, normal.png, ao.png


def create_workspace(base_dir: Path | None = None) -> WorkspacePaths:
    """
    Create a fresh, timestamped workspace with all required subdirectories.

    Args:
        base_dir: Parent directory for workspaces. Defaults to
                  ~/MeshineShop/workspaces/ if not specified.

    Returns:
        WorkspacePaths with all directories created on disk.

    The timestamp format (YYYYMMDD_HHMMSS) ensures workspaces sort
    chronologically and are human-readable in a file browser.
    """
    if base_dir is None:
        base_dir = Path.home() / "MeshineShop" / "workspaces"

    # Generate a unique, timestamped directory name.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_dir / f"job_{timestamp}"

    # Define all subdirectory paths.
    paths = WorkspacePaths(
        root=root,
        images=root / "images",
        database=root / "database.db",
        sparse=root / "sparse",
        dense=root / "dense",
        mesh=root / "mesh",
        # Phase 2c: baked texture maps live here, separate from raw mesh files.
        textures=root / "textures",
    )

    # Create all directories. parents=True creates the full path including
    # base_dir if it doesn't exist yet (first-time setup).
    # exist_ok=True prevents errors if called twice (shouldn't happen, but safe).
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.sparse.mkdir(parents=True, exist_ok=True)
    paths.dense.mkdir(parents=True, exist_ok=True)
    paths.mesh.mkdir(parents=True, exist_ok=True)
    paths.textures.mkdir(parents=True, exist_ok=True)

    return paths
