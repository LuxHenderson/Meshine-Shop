"""
Mesh export module for Meshine Shop.

Converts the pipeline's PLY mesh output into standard 3D formats that
any modeling tool (Blender, Unity, Unreal) can import. This is the final
step in the Phase 1 end-to-end workflow: photos → reconstruction → export.

Supported export formats:
    - OBJ (.obj)   — Universal format, widest compatibility
    - glTF Binary (.glb) — Modern standard, preferred by web and game engines

Uses trimesh for format conversion. trimesh handles the vertex/face data
translation between formats, including normals and vertex colors when present.
"""

from pathlib import Path
import trimesh


def load_mesh_info(source_ply: Path) -> dict:
    """
    Load a PLY mesh and return summary information for the UI.

    Returns a dict with:
        - vertices: int — number of vertices in the mesh
        - triangles: int — number of triangular faces
        - file_size_mb: float — source file size in megabytes

    This is called when the pipeline completes so the Export view can
    display mesh stats before the user decides on a format.
    """
    mesh = trimesh.load(source_ply)

    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "file_size_mb": round(source_ply.stat().st_size / (1024 * 1024), 2),
    }


def export_mesh(source_ply: Path, dest_path: Path) -> None:
    """
    Convert a PLY mesh to the format implied by dest_path's extension.

    Args:
        source_ply: Path to the pipeline's output PLY mesh.
        dest_path:  Where to save the exported file. The file extension
                    determines the format (.obj → OBJ, .glb → glTF Binary).

    Raises:
        ValueError: If the destination extension isn't a supported format.
        FileNotFoundError: If the source PLY doesn't exist.

    trimesh infers the export format from the file extension, so we just
    need to validate that the extension is one we support and let trimesh
    handle the actual conversion.
    """
    supported = {".obj", ".glb"}

    if not source_ply.exists():
        raise FileNotFoundError(f"Source mesh not found: {source_ply}")

    extension = dest_path.suffix.lower()
    if extension not in supported:
        raise ValueError(
            f"Unsupported export format '{extension}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )

    # Load the PLY mesh.
    mesh = trimesh.load(source_ply)

    # Export to the target format. trimesh uses the file extension to
    # determine the output format automatically.
    mesh.export(str(dest_path))
