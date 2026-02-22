"""
Apple Object Capture engine for Meshine Shop.

This module provides a ReconstructionEngine implementation that uses Apple's
PhotogrammetrySession API (RealityKit Object Capture) for high-quality
photogrammetry on macOS with Apple Silicon.

Unlike COLMAP, which exposes individual pipeline stages (feature extraction,
matching, sparse/dense reconstruction, meshing, texturing), Object Capture
runs the entire reconstruction as a single operation. This engine maps that
single operation onto our 6-stage pipeline interface so the UI shows familiar
progress stages regardless of which engine is running.

Communication with the Swift API happens via a subprocess — PhotogrammetrySession
is a Swift-only API that cannot be accessed through PyObjC (which only bridges
Objective-C). The Swift CLI tool (apple-photogrammetry) outputs JSON lines to
stdout, which this engine parses for real-time progress updates.

Stage mapping:
    Ingest      → Shared ingest_images() — same as COLMAP
    Features    → Runs the full Object Capture session (reports combined progress)
    Sparse      → Auto-completes (handled by Object Capture internally)
    Dense       → Auto-completes (handled by Object Capture internally)
    Mesh        → Converts Object Capture output to PLY for the export pipeline
    Texture     → Auto-completes (Object Capture bakes textures automatically)
    Decimation  → Shared PyMeshLab decimation — same as COLMAP
    UV Unwrap   → Shared xatlas UV parametrization — same as COLMAP
"""

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import trimesh

from meshine_shop.core.engine import ReconstructionEngine, EngineError, ingest_images


# ---------------------------------------------------------------------------
# Swift CLI binary location
# ---------------------------------------------------------------------------
# The CLI binary is built by Swift Package Manager into the apple_photogrammetry
# package directory. During development it lives in .build/debug/; for
# distribution it will be bundled alongside the Python app.

def _find_cli_binary() -> Path | None:
    """
    Locate the apple-photogrammetry Swift CLI binary.

    Searches in order:
    1. The SPM build directory (development)
    2. The system PATH (installed/bundled)

    Returns None if the binary is not found.
    """
    # Development path — relative to this file's location.
    # This file is at desktop/meshine_shop/core/apple_engine.py
    # The CLI is at desktop/apple_photogrammetry/.build/debug/apple-photogrammetry
    dev_path = (
        Path(__file__).resolve().parent.parent.parent
        / "apple_photogrammetry" / ".build" / "debug" / "apple-photogrammetry"
    )
    if dev_path.exists():
        return dev_path

    # Release build path — check for optimized build too.
    release_path = (
        Path(__file__).resolve().parent.parent.parent
        / "apple_photogrammetry" / ".build" / "release" / "apple-photogrammetry"
    )
    if release_path.exists():
        return release_path

    # Fall back to system PATH (for bundled/installed distributions).
    system_path = shutil.which("apple-photogrammetry")
    if system_path:
        return Path(system_path)

    return None


def is_apple_object_capture_available() -> bool:
    """
    Check whether this machine supports Apple Object Capture.

    Verifies three conditions:
    1. Running on macOS (Object Capture is Apple-only)
    2. The Swift CLI binary exists and is executable
    3. The CLI reports {"supported": true} (checks macOS version + hardware)

    This function is called by the engine factory at startup to decide
    which engine to use. It's intentionally a module-level function
    (not a method) so it can be called without instantiating the engine.
    """
    # Quick platform check before trying to run the CLI.
    if platform.system() != "Darwin":
        return False

    cli_path = _find_cli_binary()
    if cli_path is None:
        return False

    # Ask the CLI to check hardware/OS support.
    try:
        result = subprocess.run(
            [str(cli_path), "--check-support"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            return data.get("supported", False)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

    return False


class AppleObjectCaptureEngine(ReconstructionEngine):
    """
    Photogrammetry engine using Apple's Object Capture API (RealityKit).

    Produces high-quality meshes (~250K polygons at 'full' detail) with
    PBR textures on macOS 12+ with Apple Silicon. Uses Metal GPU acceleration
    via the Neural Engine for fast processing.

    The engine communicates with the PhotogrammetrySession API through a
    lightweight Swift CLI subprocess that outputs JSON progress lines.
    """

    def __init__(self, detail: str = "full"):
        """
        Initialize the Apple Object Capture engine.

        Args:
            detail: Quality level for reconstruction. One of:
                    preview  (~25K polys, fast validation)
                    reduced  (~25K polys, with textures)
                    medium   (~100K polys, good for AR)
                    full     (~250K polys, game-ready — default)
                    raw      (~30M polys, professional post-production)
        """
        self._detail = detail

        # Locate the Swift CLI binary. If it's not found, the engine
        # shouldn't have been selected by the factory — but we handle
        # it gracefully just in case.
        self._cli_path = _find_cli_binary()
        if self._cli_path is None:
            raise EngineError(
                "Apple Object Capture CLI binary not found. "
                "Build it with: cd desktop/apple_photogrammetry && swift build"
            )

    # --- Pipeline stage implementations ---

    def ingest(self, image_paths: list[str], workspace, on_progress):
        """
        Validate and convert source images into the workspace.

        Uses the shared ingest_images() function — identical behavior to
        COLMAP. Both engines need JPEG-normalized images in workspace/images/.
        """
        ingest_images(image_paths, workspace, on_progress)

    def extract_features(self, workspace, on_progress):
        """
        Run the full Apple Object Capture reconstruction.

        This is where the real work happens. Object Capture doesn't expose
        individual stages — it runs feature extraction, matching, sparse
        reconstruction, dense reconstruction, meshing, and texturing as a
        single operation. We run it during the "features" stage and report
        progress as the combined reconstruction advances.

        The output is a USDZ file (Apple's native 3D format) that we'll
        convert to PLY in the mesh_reconstruct stage for consistency with
        the export pipeline. USDZ is the only format guaranteed to work
        with all versions of PhotogrammetrySession.
        """
        on_progress("Starting Apple Object Capture (Metal-accelerated)...")

        # Object Capture reads directly from the image directory.
        input_dir = str(workspace.images)

        # Pass the mesh output directory to the CLI. The CLI will create:
        # - object_capture_output.usdz (intermediate, Apple's native format)
        # - meshed.obj (final, converted via ModelIO for trimesh compatibility)
        output_dir = str(workspace.mesh)

        # Run the Swift CLI as a subprocess with line-buffered stdout
        # so we get real-time JSON progress updates.
        try:
            process = subprocess.Popen(
                [
                    str(self._cli_path),
                    input_dir,
                    output_dir,
                    "--detail", self._detail,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # Line buffering for real-time progress.
                bufsize=1,
            )

            # Read JSON lines from stdout as they arrive.
            # Each line is a self-contained JSON object with an "event" key.
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Skip non-JSON output (shouldn't happen, but safe).
                    continue

                event = data.get("event")

                if event == "started":
                    on_progress("Object Capture session initialized")

                elif event == "progress":
                    fraction = data.get("fraction", 0)
                    percent = int(fraction * 100)
                    on_progress(f"Reconstructing: {percent}% complete")

                elif event == "complete":
                    on_progress("Object Capture reconstruction complete")

                elif event == "error":
                    message = data.get("message", "Unknown error")
                    raise EngineError(
                        f"Apple Object Capture failed: {message}"
                    )

            # Wait for the process to finish and check exit code.
            process.wait(timeout=3600)  # 1 hour timeout for large datasets.

            if process.returncode != 0:
                # Read stderr for additional error context.
                stderr_output = process.stderr.read() if process.stderr else ""
                raise EngineError(
                    f"Apple Object Capture exited with code {process.returncode}. "
                    f"{stderr_output[:500]}"
                )

        except EngineError:
            raise
        except subprocess.TimeoutExpired:
            process.kill()
            raise EngineError(
                "Apple Object Capture timed out after 1 hour. "
                "Try using fewer images or a lower detail level."
            )
        except Exception as e:
            raise EngineError(f"Failed to run Apple Object Capture: {e}")

        # Verify the USDZ output was created by the CLI.
        output_usdz = workspace.mesh / "object_capture_output.usdz"
        if not output_usdz.exists():
            raise EngineError(
                "Apple Object Capture completed but no output file was created. "
                "The image set may not have enough overlap for reconstruction."
            )

        on_progress("Reconstruction complete — USDZ with PBR textures saved")

    def sparse_reconstruct(self, workspace, on_progress):
        """
        Auto-complete — Object Capture handles this internally.

        The full reconstruction ran during extract_features(). This stage
        exists only to satisfy the pipeline interface and show a completion
        checkmark in the UI.
        """
        on_progress("Sparse reconstruction handled by Object Capture")

    def dense_reconstruct(self, workspace, on_progress):
        """
        Auto-complete — Object Capture handles this internally.

        Object Capture uses Metal-accelerated dense reconstruction
        automatically. No separate step needed.
        """
        on_progress("Dense reconstruction handled by Object Capture (Metal GPU)")

    def mesh_reconstruct(self, workspace, on_progress):
        """
        Convert Object Capture's USDZ output to PLY for the export pipeline.

        The existing export flow expects a PLY file at workspace.mesh/meshed.ply.
        Object Capture produces a USDZ file (zip archive containing .usdc geometry
        and PBR texture images). We extract the geometry using Pixar's USD Python
        library (usd-core) and write it as PLY via trimesh.

        This keeps the export pipeline engine-agnostic — it always reads from
        the same PLY location regardless of which engine produced the mesh.
        """
        on_progress("Converting Object Capture USDZ to PLY...")

        source_usdz = workspace.mesh / "object_capture_output.usdz"
        output_ply = workspace.mesh / "meshed.ply"

        if not source_usdz.exists():
            raise EngineError(
                "Object Capture USDZ output not found. "
                "The reconstruction may have failed during the features stage."
            )

        try:
            import zipfile
            import numpy as np
            from pxr import Usd, UsdGeom

            # USDZ is a zip archive containing a .usdc file (geometry) and
            # texture images. Extract to a temp directory so USD can read it.
            extract_dir = workspace.mesh / "usdz_extracted"
            extract_dir.mkdir(exist_ok=True)

            on_progress("Extracting USDZ archive...")
            with zipfile.ZipFile(str(source_usdz), 'r') as zf:
                zf.extractall(str(extract_dir))

            # Find the .usdc geometry file inside the extracted archive.
            usdc_files = list(extract_dir.glob("*.usdc"))
            if not usdc_files:
                raise EngineError(
                    "No .usdc geometry file found inside USDZ archive."
                )

            # Open the USD stage and find the mesh geometry.
            on_progress("Reading mesh geometry from USD...")
            stage = Usd.Stage.Open(str(usdc_files[0]))

            mesh_prim = None
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    mesh_prim = UsdGeom.Mesh(prim)
                    break

            if mesh_prim is None:
                raise EngineError(
                    "No mesh geometry found in USD file."
                )

            # Extract vertices and face indices from the USD mesh.
            points = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float64)
            face_indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get())
            face_counts = np.array(mesh_prim.GetFaceVertexCountsAttr().Get())

            on_progress(
                f"Mesh loaded: {len(points):,} vertices, {len(face_counts):,} triangles"
            )

            # Build a trimesh object from the USD geometry data.
            # Object Capture always produces triangle meshes (face_counts all 3),
            # so we can reshape the face indices directly.
            faces = face_indices.reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=points, faces=faces)

            # Save as PLY — the format the export pipeline expects.
            mesh.export(str(output_ply))
            on_progress(
                f"Mesh converted to PLY: {len(points):,} vertices, "
                f"{len(faces):,} triangles"
            )

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Failed to convert Object Capture output: {e}")

    def texture_map(self, workspace, on_progress):
        """
        Auto-complete — Object Capture bakes textures automatically.

        The OBJ output from Object Capture includes diffuse, normal, and
        AO texture maps. No separate texturing step is needed.
        """
        on_progress("Textures baked by Object Capture (PBR maps included)")

    def decimate(self, workspace, on_progress, target_faces=25_000):
        """
        Reduce mesh polygon count using quadric edge collapse decimation.

        Uses PyMeshLab's simplification_quadric_edge_collapse_decimation.
        Object Capture typically produces ~100K triangles at 'full' detail,
        so this stage is important for mobile and PC quality presets.

        If the mesh already has fewer triangles than the target (e.g.,
        Cinematic preset with a small dataset), decimation is skipped.

        Args:
            workspace:    WorkspacePaths containing the mesh to decimate.
            on_progress:  Callback for status messages shown in the UI.
            target_faces: Target triangle count from the quality preset.
        """
        import pymeshlab

        mesh_path = workspace.mesh / "meshed.ply"

        if not mesh_path.exists():
            raise EngineError(
                "No mesh file found for decimation. "
                "The mesh reconstruction stage may have failed."
            )

        on_progress("Loading mesh for decimation...")

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_path))

            # Get the current triangle count to decide if decimation is needed.
            current_faces = ms.current_mesh().face_number()

            if current_faces <= target_faces:
                on_progress(
                    f"Mesh already has {current_faces:,} triangles "
                    f"(target: {target_faces:,}) — skipping decimation"
                )
                return

            on_progress(
                f"Decimating from {current_faces:,} to {target_faces:,} triangles..."
            )

            # Quadric edge collapse decimation — iteratively collapses edges
            # with the lowest quadric error metric until the target face count
            # is reached. This preserves geometric detail where it matters most.
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces,
                preservenormal=True,
            )

            final_faces = ms.current_mesh().face_number()
            final_verts = ms.current_mesh().vertex_number()

            # Overwrite the original mesh with the decimated version.
            ms.save_current_mesh(str(mesh_path))

            on_progress(
                f"Decimation complete: {final_verts:,} vertices, "
                f"{final_faces:,} triangles"
            )

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Mesh decimation failed: {e}")

    def unwrap_uv(self, workspace, on_progress):
        """
        Generate non-overlapping UV coordinates for the decimated mesh.

        Identical implementation to ColmapEngine — UV unwrapping is engine-agnostic.
        Both engines produce meshed.ply after decimation; this stage reads that
        file and produces meshed_uv.obj with UV coordinates embedded.

        Uses xatlas UV atlas packer to cut and unfold the mesh surface into
        non-overlapping UV islands packed efficiently into [0,1]² space.

        Input:  workspace.mesh/meshed.ply  (decimated mesh, no UVs)
        Output: workspace.mesh/meshed_uv.obj  (UV-mapped mesh for Phase 2c baking)
        """
        import numpy as np
        import xatlas
        from trimesh.visual.texture import TextureVisuals

        mesh_path = workspace.mesh / "meshed.ply"
        output_obj = workspace.mesh / "meshed_uv.obj"

        if not mesh_path.exists():
            raise EngineError(
                "No mesh file found for UV unwrapping. "
                "The mesh decimation stage may have failed."
            )

        on_progress("Loading mesh for UV unwrapping...")

        try:
            # Load the decimated PLY mesh via trimesh.
            source = trimesh.load(str(mesh_path))

            # xatlas requires float32 positions and uint32 face indices.
            vertices = np.array(source.vertices, dtype=np.float32)
            faces = np.array(source.faces, dtype=np.uint32)

            original_vertex_count = len(vertices)
            original_face_count = len(faces)
            on_progress(
                f"Mesh loaded: {original_vertex_count:,} vertices, "
                f"{original_face_count:,} triangles"
            )

            on_progress("Running xatlas UV parametrization (this may take a moment)...")

            # xatlas.parametrize generates non-overlapping UV islands.
            # Returns:
            #   vmapping:    (N,) int32 — maps each output vertex to its original
            #   new_indices: (F, 3) int32 — new triangle face connectivity
            #   uvs:         (N, 2) float32 — UV coordinates in [0, 1]² space
            vmapping, new_indices, uvs = xatlas.parametrize(vertices, faces)

            # Reconstruct new vertex positions using the vertex mapping.
            # xatlas may split vertices at UV seam edges, so output count >= input.
            new_positions = vertices[vmapping]
            new_vertex_count = len(new_positions)
            seam_splits = new_vertex_count - original_vertex_count

            on_progress(
                f"UV unwrapping complete: {new_vertex_count:,} vertices "
                f"(+{seam_splits:,} seam splits), {len(new_indices):,} triangles"
            )

            # Build a new trimesh with UV texture coordinates attached.
            uv_visuals = TextureVisuals(uv=uvs)
            mesh_with_uvs = trimesh.Trimesh(
                vertices=new_positions,
                faces=new_indices,
                visual=uv_visuals,
                # Skip auto-processing — merging vertices would destroy UV seams.
                process=False,
            )

            # Export as OBJ with UV coordinates embedded as "vt" lines.
            mesh_with_uvs.export(str(output_obj))
            on_progress(f"UV mesh saved: {output_obj.name}")

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"UV unwrapping failed: {e}")
