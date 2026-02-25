"""
Photogrammetry engine abstraction and COLMAP implementation.

This module defines the interface that any photogrammetry engine must
satisfy (ReconstructionEngine), plus the concrete COLMAP implementation
(ColmapEngine) that wraps the pycolmap Python API.

The abstraction layer is intentionally thin — one method per pipeline stage,
each receiving a workspace path and a progress callback. This keeps things
simple for a solo developer while still allowing engine swaps in the future
(e.g., switching from COLMAP to OpenMVG+OpenMVS) without touching the
worker or UI code.

COLMAP integration strategy:
    - pycolmap Python API for most stages (no subprocess overhead)
    - COLMAP CLI subprocess as fallback for:
        * Dense reconstruction (patch_match_stereo requires CUDA, unavailable on macOS)
        * Texture mapping (not exposed in pycolmap Python API)
    - On CPU-only systems (macOS), dense reconstruction is skipped and
      meshing proceeds directly from the sparse point cloud. This produces
      lower quality but still functional output — acceptable for v1.0.
"""

import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import pycolmap
import open3d as o3d
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF/HEIC support with Pillow so Image.open() can read Apple's
# default photo format. Without this, iPhones photos (which are HEIC internally
# even when they have .JPEG extensions) cannot be processed by COLMAP.
register_heif_opener()


class EngineError(Exception):
    """
    Raised when a pipeline stage fails within the engine.

    Wraps the underlying error (pycolmap exception, subprocess failure, etc.)
    with a human-readable message that includes the stage name and likely cause.
    The worker layer catches this and forwards it to the UI via signals.
    """
    pass


# ---------------------------------------------------------------------------
# Shared ingest logic
# ---------------------------------------------------------------------------
# Both ColmapEngine and AppleObjectCaptureEngine need to validate and convert
# source images before processing. This shared function keeps that logic in
# one place so both engines produce identical workspace/images/ directories.

# Extensions we'll attempt to open with Pillow. HEIC/HEIF included
# because iPhones use this format by default.
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".heif",
}


def ingest_images(image_paths: list[str], workspace, on_progress) -> None:
    """
    Validate source images and copy them into the workspace as JPEG.

    Shared by all engine implementations. Checks that each file exists and
    has a supported image extension, then converts every image to standard
    JPEG via Pillow. This handles HEIC/HEIF images (Apple's default format)
    transparently — iPhones often save HEIC data with .JPEG extensions, so
    we can't trust the extension alone.

    Files are written (not symlinked) to workspace/images/ so the originals
    are never modified by any engine.

    Args:
        image_paths: List of absolute file path strings from the user's drop.
        workspace:   WorkspacePaths with an images directory to write into.
        on_progress: Callback for status messages shown in the UI.

    Raises:
        EngineError: If no valid images are found after filtering.
    """
    on_progress("Validating input files...")

    valid_count = 0
    skipped = []

    for path_str in image_paths:
        path = Path(path_str)

        # Check the file exists.
        if not path.is_file():
            skipped.append(f"{path.name} (not found)")
            continue

        # Check the file has a supported image extension.
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            skipped.append(f"{path.name} (unsupported format)")
            continue

        # Convert every image to standard JPEG via Pillow. This ensures
        # any engine receives files it can actually read, regardless of the
        # source format (HEIC, 16-bit PNG, TIFF, etc.).
        dest = workspace.images / (path.stem + ".jpg")
        try:
            with Image.open(path) as img:
                # Convert to RGB in case the image has an alpha channel
                # or is in a color mode JPEG doesn't support (e.g., RGBA, P).
                rgb_img = img.convert("RGB")
                rgb_img.save(dest, "JPEG", quality=95)
            valid_count += 1
        except Exception as e:
            skipped.append(f"{path.name} (conversion failed: {e})")
            continue

    if valid_count == 0:
        raise EngineError(
            "No valid images found. Supported formats: "
            "JPEG, PNG, TIFF, BMP, HEIC. Please check your input files."
        )

    if skipped:
        on_progress(f"Imported {valid_count} images, skipped {len(skipped)}: {', '.join(skipped[:3])}")
    else:
        on_progress(f"Imported {valid_count} images")


# ---------------------------------------------------------------------------
# Configuration constants for COLMAP
# ---------------------------------------------------------------------------
# These are defined at module level so they're easy to find and tweak.
# pycolmap uses option objects with sensible defaults; we only override
# settings that matter for our use case.

# Maximum features per image. 8192 balances quality vs. CPU processing time.
# Higher values improve reconstruction quality but increase feature matching
# time quadratically (every feature must be compared against every other).
MAX_NUM_FEATURES = 8192


def _find_colmap_cli() -> Path | None:
    """
    Locate the COLMAP command-line binary on the system.

    Checks common install locations. Returns None if COLMAP CLI is not found.
    The CLI is needed as a fallback for stages that pycolmap doesn't expose
    (texture mapping) or that require CUDA (dense reconstruction).
    """
    # shutil.which searches PATH — works on macOS and Windows.
    cli_path = shutil.which("colmap")
    if cli_path:
        return Path(cli_path)
    return None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ReconstructionEngine(ABC):
    """
    Abstract interface for a photogrammetry reconstruction engine.

    Each method corresponds to one pipeline stage (defined in pipeline.py).
    Implementations receive a workspace path and a progress callback they
    can call with a status message string to report progress.

    Contract:
        - Returning normally means the stage succeeded.
        - Raising EngineError means the stage failed with a known issue.
        - Raising any other exception means an unexpected failure.
    """

    @abstractmethod
    def ingest(self, image_paths: list[str], workspace: "WorkspacePaths",
               on_progress: Callable[[str], None]) -> None:
        """Validate and copy source images into the workspace."""
        ...

    @abstractmethod
    def extract_features(self, workspace: "WorkspacePaths",
                         on_progress: Callable[[str], None]) -> None:
        """Detect SIFT keypoints and descriptors in each image."""
        ...

    @abstractmethod
    def sparse_reconstruct(self, workspace: "WorkspacePaths",
                           on_progress: Callable[[str], None]) -> None:
        """Match features across images and reconstruct camera poses + sparse point cloud."""
        ...

    @abstractmethod
    def dense_reconstruct(self, workspace: "WorkspacePaths",
                          on_progress: Callable[[str], None]) -> None:
        """Generate a dense point cloud from camera poses and images."""
        ...

    @abstractmethod
    def mesh_reconstruct(self, workspace: "WorkspacePaths",
                         on_progress: Callable[[str], None]) -> None:
        """Convert the point cloud into a triangulated surface mesh."""
        ...

    @abstractmethod
    def texture_map(self, workspace: "WorkspacePaths",
                    on_progress: Callable[[str], None]) -> None:
        """Project source photo colors onto the mesh surface."""
        ...

    @abstractmethod
    def decimate(self, workspace: "WorkspacePaths",
                 on_progress: Callable[[str], None],
                 target_faces: int = 25_000) -> None:
        """Reduce mesh polygon count to the target triangle budget."""
        ...

    @abstractmethod
    def unwrap_uv(self, workspace: "WorkspacePaths",
                  on_progress: Callable[[str], None]) -> None:
        """Generate UV coordinates for the decimated mesh using xatlas.

        Reads workspace.mesh/meshed.ply and writes workspace.mesh/meshed_uv.obj
        with UV texture coordinates embedded in OBJ format. The OBJ is the
        intermediate handoff file for Phase 2c texture baking.
        """
        ...

    @abstractmethod
    def bake_textures(self, workspace: "WorkspacePaths",
                      on_progress: Callable[[str], None]) -> None:
        """Bake PBR texture maps onto the UV-mapped decimated mesh.

        Reads workspace.mesh/meshed_uv.obj (produced by Phase 2b UV unwrapping)
        and writes three texture images to workspace.textures/:
            albedo.png  — diffuse/base color from the reconstruction source
            normal.png  — tangent-space normal map from mesh vertex normals
            ao.png      — ambient occlusion from hemisphere ray casting

        The color source differs by engine:
            COLMAP:  colored dense or sparse point cloud
            Apple:   diffuse texture extracted from object_capture_output.usdz

        This stage is non-fatal: if baking fails for any reason, the pipeline
        continues with an untextured mesh (same pattern as the texture_map stage).
        """
        ...


# ---------------------------------------------------------------------------
# COLMAP implementation
# ---------------------------------------------------------------------------

class ColmapEngine(ReconstructionEngine):
    """
    Concrete photogrammetry engine wrapping COLMAP via the pycolmap Python API.

    Uses pycolmap for feature extraction, matching, sparse reconstruction,
    and meshing. Falls back to the COLMAP CLI for dense reconstruction
    (requires CUDA) and texture mapping (not exposed in pycolmap).

    On CPU-only systems (e.g., macOS with Apple Silicon), dense reconstruction
    is skipped — meshing proceeds from the sparse point cloud. The output
    quality is lower but the pipeline still completes end-to-end.
    """

    def __init__(self):
        # Check for CUDA availability at construction time.
        # This determines whether we can run dense reconstruction.
        self._has_cuda = pycolmap.has_cuda

        # Locate the CLI binary for fallback operations.
        self._cli_path = _find_colmap_cli()

    def ingest(self, image_paths: list[str], workspace, on_progress):
        """
        Validate source images and copy them into the workspace.

        Delegates to the shared ingest_images() function which handles
        format validation, HEIC conversion, and JPEG normalization.
        See ingest_images() docstring for full details.
        """
        ingest_images(image_paths, workspace, on_progress)

    def extract_features(self, workspace, on_progress):
        """
        Extract SIFT features from all images in the workspace.

        Uses pycolmap.extract_features() which detects keypoints and computes
        descriptors for each image, storing them in the COLMAP database.
        SIFT (Scale-Invariant Feature Transform) is the standard feature
        detector for photogrammetry — robust to scale and rotation changes.
        """
        on_progress("Extracting SIFT features from images...")

        try:
            # Configure SIFT extraction options.
            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_num_features = MAX_NUM_FEATURES

            extraction_options = pycolmap.FeatureExtractionOptions()
            extraction_options.sift = sift_options

            pycolmap.extract_features(
                database_path=str(workspace.database),
                image_path=str(workspace.images),
                extraction_options=extraction_options,
            )
            on_progress("Feature extraction complete")
        except Exception as e:
            raise EngineError(f"Feature extraction failed: {e}")

    def sparse_reconstruct(self, workspace, on_progress):
        """
        Match features across images and run incremental SfM.

        This stage has two substeps:
        1. Exhaustive matching — every image's features are compared against
           every other image to find correspondences. This is O(n^2) but
           produces the most complete results for small-to-medium datasets.
        2. Incremental mapping — COLMAP registers images one by one, triangulating
           3D points and optimizing camera poses via bundle adjustment.

        The output is a sparse 3D point cloud with estimated camera positions
        for each image.
        """
        # Step 1: Feature matching
        on_progress("Matching features across images (this may take a while)...")
        try:
            pycolmap.match_exhaustive(
                database_path=str(workspace.database),
            )
        except Exception as e:
            raise EngineError(f"Feature matching failed: {e}")

        # Step 2: Incremental SfM
        on_progress("Running incremental Structure-from-Motion...")
        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(workspace.database),
                image_path=str(workspace.images),
                output_path=str(workspace.sparse),
            )
        except Exception as e:
            raise EngineError(f"Sparse reconstruction failed: {e}")

        # Validate the reconstruction produced usable results.
        if not reconstructions:
            raise EngineError(
                "Sparse reconstruction produced no results. This usually means "
                "the images don't have enough visual overlap. Try adding more "
                "photos with overlapping views of the subject."
            )

        # Report how many images were successfully registered.
        # The first (largest) reconstruction is the primary one.
        best = reconstructions[0] if isinstance(reconstructions, list) else list(reconstructions.values())[0]
        num_images = best.num_images()
        num_points = best.num_points3D()
        total_input = len(list(workspace.images.iterdir()))
        on_progress(
            f"Sparse reconstruction complete: {num_images}/{total_input} images "
            f"registered, {num_points} 3D points"
        )

        # Warn if fewer than half the images were registered — this suggests
        # poor overlap or image quality issues.
        if num_images < total_input * 0.5:
            on_progress(
                f"Warning: Only {num_images}/{total_input} images were registered. "
                "Consider adding more overlapping photos for better results."
            )

    def dense_reconstruct(self, workspace, on_progress):
        """
        Generate a dense point cloud from camera poses and images.

        Dense reconstruction uses patch-match stereo to compute depth maps
        for each image, then fuses them into a dense point cloud.

        IMPORTANT: Patch-match stereo requires CUDA (NVIDIA GPU). On macOS
        with Apple Silicon, CUDA is not available. In this case, we skip
        dense reconstruction entirely and proceed to meshing from the sparse
        point cloud. This is a known limitation documented in the README.

        On systems with CUDA, this stage produces a significantly higher-quality
        point cloud that results in better mesh output.
        """
        if not self._has_cuda:
            on_progress(
                "Dense reconstruction skipped (requires CUDA GPU). "
                "Proceeding with sparse point cloud for meshing."
            )
            return

        # Undistort images — required preparation for patch-match stereo.
        # This reprojects images to remove lens distortion using the camera
        # models estimated during sparse reconstruction.
        on_progress("Undistorting images for dense reconstruction...")
        try:
            # Find the sparse reconstruction directory (COLMAP creates numbered subdirs).
            sparse_subdirs = sorted(workspace.sparse.iterdir())
            if not sparse_subdirs:
                raise EngineError("No sparse reconstruction found for dense processing.")
            sparse_model_path = sparse_subdirs[0]

            pycolmap.undistort_images(
                output_path=str(workspace.dense),
                input_path=str(sparse_model_path),
                image_path=str(workspace.images),
            )
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Image undistortion failed: {e}")

        # Patch-match stereo — compute depth maps for each image.
        on_progress("Computing depth maps via patch-match stereo...")
        try:
            pycolmap.patch_match_stereo(
                workspace_path=str(workspace.dense),
            )
        except Exception as e:
            raise EngineError(f"Patch-match stereo failed: {e}")

        # Stereo fusion — merge depth maps into a dense point cloud.
        on_progress("Fusing depth maps into dense point cloud...")
        try:
            fused_path = workspace.dense / "fused.ply"
            pycolmap.stereo_fusion(
                output_path=str(fused_path),
                workspace_path=str(workspace.dense),
            )
            on_progress("Dense reconstruction complete")
        except Exception as e:
            raise EngineError(f"Stereo fusion failed: {e}")

    def mesh_reconstruct(self, workspace, on_progress):
        """
        Convert the point cloud into a triangulated surface mesh.

        Uses Poisson surface reconstruction via Open3D, which takes an oriented
        point cloud and produces a watertight triangle mesh. Poisson meshing
        requires point normals — dense point clouds from COLMAP include them,
        but sparse point clouds do not. For sparse data, we estimate normals
        using Open3D's PCA-based normal estimation before meshing.

        Input source depends on whether dense reconstruction ran:
        - With CUDA: Uses the dense fused.ply point cloud (higher quality)
        - Without CUDA: Uses the sparse reconstruction (lower quality but functional)
        """
        on_progress("Building surface mesh via Poisson reconstruction...")

        # Determine the input point cloud path.
        # Prefer the dense fused point cloud if it exists.
        fused_ply = workspace.dense / "fused.ply"
        if fused_ply.exists():
            input_path = str(fused_ply)
            on_progress("Using dense point cloud for meshing")
        else:
            # Fall back to sparse reconstruction — find the first model directory.
            sparse_subdirs = sorted(workspace.sparse.iterdir())
            if not sparse_subdirs:
                raise EngineError(
                    "No point cloud data found for meshing. "
                    "Sparse reconstruction may have failed."
                )
            # Export sparse points to PLY for Poisson meshing.
            sparse_model_path = sparse_subdirs[0]
            sparse_ply = workspace.mesh / "sparse_input.ply"

            try:
                reconstruction = pycolmap.Reconstruction()
                reconstruction.read(str(sparse_model_path))
                reconstruction.export_PLY(str(sparse_ply))
                input_path = str(sparse_ply)
                on_progress("Using sparse point cloud for meshing (dense was skipped)")
            except Exception as e:
                raise EngineError(f"Failed to export sparse point cloud: {e}")

        # Load the point cloud with Open3D for normal estimation and meshing.
        # Open3D's Poisson reconstruction is more robust than pycolmap's —
        # it handles missing normals gracefully and produces better results
        # from sparse data.
        output_path = workspace.mesh / "meshed.ply"
        try:
            pcd = o3d.io.read_point_cloud(input_path)

            if len(pcd.points) == 0:
                raise EngineError("Point cloud is empty — nothing to mesh.")

            on_progress(f"Loaded {len(pcd.points)} points")

            # Estimate normals if the point cloud doesn't have them.
            # This is always the case for sparse reconstructions, and
            # sometimes for dense ones depending on the fusion method.
            if not pcd.has_normals():
                on_progress("Estimating surface normals...")
                # k=20 neighbors provides stable normal estimates for typical
                # photogrammetry point densities. The search radius of 0.1
                # adapts to the point cloud scale.
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=20
                    )
                )
                # Orient normals consistently so Poisson reconstruction
                # can determine inside vs. outside of the surface.
                pcd.orient_normals_consistent_tangent_plane(k=15)

            # Run Poisson surface reconstruction via Open3D.
            # depth=9 controls octree depth — higher values produce more
            # detail but take longer. 9 is a good balance for photogrammetry.
            on_progress("Running Poisson surface reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )

            if len(mesh.vertices) == 0:
                raise EngineError(
                    "Poisson reconstruction produced an empty mesh. "
                    "The point cloud may not have enough points."
                )

            # Remove low-density vertices to clean up the mesh edges.
            # Poisson reconstruction creates a watertight surface that extends
            # beyond the actual data — trimming removes these phantom surfaces.
            import numpy as np
            density_array = np.asarray(densities)
            # Remove vertices below the 5th percentile density — these are
            # typically spurious extensions of the Poisson surface.
            threshold = np.quantile(density_array, 0.05)
            vertices_to_remove = density_array < threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            # Save the final mesh.
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            on_progress(
                f"Mesh reconstruction complete: "
                f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
            )
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Mesh reconstruction failed: {e}")

    def texture_map(self, workspace, on_progress):
        """
        Project source photo colors onto the mesh surface.

        Texture mapping is not exposed in pycolmap's Python API, so we
        fall back to the COLMAP CLI (colmap texture_mapper). If the CLI
        is not available, this stage is skipped with a warning — the mesh
        will still be exported, just without texture.

        This CLI fallback is the only subprocess call in the pipeline.
        All other stages use the pycolmap Python API directly.
        """
        if self._cli_path is None:
            on_progress(
                "Texture mapping skipped (COLMAP CLI not found). "
                "The mesh will be exported without texture."
            )
            return

        on_progress("Mapping textures onto mesh surface...")

        # Find the sparse reconstruction model for camera poses.
        sparse_subdirs = sorted(workspace.sparse.iterdir())
        if not sparse_subdirs:
            on_progress("Texture mapping skipped (no sparse model found)")
            return

        sparse_model_path = sparse_subdirs[0]
        mesh_file = workspace.mesh / "meshed.ply"

        if not mesh_file.exists():
            on_progress("Texture mapping skipped (no mesh file found)")
            return

        # Run the COLMAP texture mapper via CLI subprocess.
        # This is the one place we use subprocess — pycolmap doesn't
        # expose texture mapping in its Python API.
        try:
            result = subprocess.run(
                [
                    str(self._cli_path), "texture_mapper",
                    "--workspace_path", str(workspace.root),
                    "--input_path", str(sparse_model_path),
                    "--output_path", str(workspace.mesh),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for large datasets
            )

            if result.returncode != 0:
                # Non-fatal — the mesh still exists, just untextured.
                on_progress(
                    f"Texture mapping had issues (exit code {result.returncode}). "
                    "Mesh exported without texture."
                )
            else:
                on_progress("Texture mapping complete")

        except subprocess.TimeoutExpired:
            on_progress("Texture mapping timed out. Mesh exported without texture.")
        except Exception as e:
            on_progress(f"Texture mapping failed: {e}. Mesh exported without texture.")

    def decimate(self, workspace, on_progress, target_faces=25_000):
        """
        Reduce mesh polygon count using quadric edge collapse decimation.

        Uses Open3D's simplify_quadric_decimation — the same QEM algorithm
        as PyMeshLab but MIT-licensed, commercial-friendly, and already a
        core dependency for AO baking. If the source mesh already has fewer
        triangles than the target, decimation is skipped to avoid unnecessary
        processing or upscaling artifacts.

        The decimated mesh overwrites the original meshed.ply so the export
        pipeline always reads the final optimized version.

        Args:
            workspace:    WorkspacePaths containing the mesh to decimate.
            on_progress:  Callback for status messages shown in the UI.
            target_faces: Target triangle count from the quality preset.
        """
        import open3d as o3d

        mesh_path = workspace.mesh / "meshed.ply"

        if not mesh_path.exists():
            raise EngineError(
                "No mesh file found for decimation. "
                "The mesh reconstruction stage may have failed."
            )

        on_progress("Loading mesh for decimation...")

        try:
            # Load the PLY mesh into Open3D.
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            # Get the current triangle count to decide if decimation is needed.
            current_faces = len(mesh.triangles)

            if current_faces <= target_faces:
                on_progress(
                    f"Mesh already has {current_faces:,} triangles "
                    f"(target: {target_faces:,}) — skipping decimation"
                )
                return

            on_progress(
                f"Decimating from {current_faces:,} to {target_faces:,} triangles..."
            )

            # Quadric error metrics decimation — iteratively collapses edges
            # with the lowest quadric error until the target face count is
            # reached. Preserves geometric detail where it matters most.
            mesh = mesh.simplify_quadric_decimation(target_faces)

            final_faces = len(mesh.triangles)
            final_verts = len(mesh.vertices)

            # Overwrite the original mesh with the decimated version.
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)

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

        Uses xatlas — the same UV atlas packer used inside Blender, lightmap
        tools, and many game studios. xatlas automatically cuts the mesh into
        UV islands and packs them efficiently into a [0,1]² atlas with no
        overlaps (a requirement for texture baking).

        UV seam splits: xatlas may introduce additional vertices where it cuts
        the mesh open to unfold it flat. The output vertex count is therefore
        >= the input. This is expected and normal.

        Input:  workspace.mesh/meshed.ply  (decimated mesh, no UVs)
        Output: workspace.mesh/meshed_uv.obj  (UV-mapped mesh, OBJ format)

        OBJ format was chosen because it natively stores UV coordinates as
        "vt" lines, making it the natural intermediate format for Phase 2c
        texture baking. Final export format (OBJ or glTF) is still chosen
        by the user in the Export view.
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
            import trimesh as _trimesh

            # Load the decimated PLY mesh via trimesh.
            source = _trimesh.load(str(mesh_path))

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

            # Reconstruct new vertex positions from the vertex mapping.
            # xatlas may split vertices at UV seam edges (where the surface
            # is cut open to unfold it flat), so len(vmapping) >= original count.
            new_positions = vertices[vmapping]
            new_vertex_count = len(new_positions)
            seam_splits = new_vertex_count - original_vertex_count

            on_progress(
                f"UV unwrapping complete: {new_vertex_count:,} vertices "
                f"(+{seam_splits:,} seam splits), {len(new_indices):,} triangles"
            )

            # Build a new trimesh with UV texture coordinates attached.
            # TextureVisuals stores the UV map; trimesh writes it as "vt" lines
            # in the OBJ file, which is what texture bakers expect.
            uv_visuals = TextureVisuals(uv=uvs)
            mesh_with_uvs = _trimesh.Trimesh(
                vertices=new_positions,
                faces=new_indices,
                visual=uv_visuals,
                # Skip auto-processing — merging vertices would destroy UV seams.
                process=False,
            )

            # Export as OBJ. trimesh writes UV coords as "vt" lines and
            # references them in face definitions (f v1/vt1 v2/vt2 v3/vt3).
            mesh_with_uvs.export(str(output_obj))
            on_progress(f"UV mesh saved: {output_obj.name}")

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"UV unwrapping failed: {e}")

    def bake_textures(self, workspace, on_progress):
        """
        Bake albedo, normal map, and AO textures from the COLMAP reconstruction.

        Color source priority:
        1. workspace.dense/fused.ply — colored dense point cloud (requires CUDA)
        2. workspace.sparse/0/points3D.ply — sparse colored points (always present)

        Each point in the COLMAP reconstruction carries an RGB color taken from
        the source photos during feature matching. We project those colors onto
        the decimated mesh by finding the nearest source point for each vertex.

        Output files written to workspace.textures/:
            albedo.png — diffuse color from point cloud nearest-neighbour
            normal.png — tangent-space normals from mesh vertex normals
            ao.png     — ambient occlusion from Open3D hemisphere ray casting

        Non-fatal: logs a warning and continues if baking fails at any step.
        Subsequent stages (export) will still work — the mesh just won't
        carry texture data.
        """
        import trimesh as _trimesh
        from meshine_shop.core.texture_baker import (
            bake_albedo_from_pointcloud,
            vertex_colors_to_texture,
            bake_normal_map,
            bake_ao,
        )

        uv_obj = workspace.mesh / "meshed_uv.obj"
        if not uv_obj.exists():
            on_progress(
                "Texture baking skipped (meshed_uv.obj not found). "
                "UV unwrapping may have failed."
            )
            return

        # Load the UV-mapped decimated mesh — the target for all baking.
        on_progress("Loading UV-mapped mesh for texture baking...")
        try:
            mesh = _trimesh.load(str(uv_obj), process=False)
        except Exception as e:
            on_progress(f"Texture baking skipped: could not load mesh ({e})")
            return

        # Verify the mesh has UV coordinates (required for rasterization).
        if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
            on_progress("Texture baking skipped: mesh has no UV coordinates")
            return

        import numpy as np

        uvs = np.array(mesh.visual.uv, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        # --- Albedo: find the best colored point cloud source ---
        vertex_colors = None

        # Try dense point cloud first (higher density, better quality).
        dense_pcd = workspace.dense / "fused.ply"
        if dense_pcd.exists():
            on_progress("Using dense point cloud for albedo colors...")
            vertex_colors = bake_albedo_from_pointcloud(mesh, dense_pcd, on_progress)

        # Fall back to sparse points (always produced by COLMAP).
        if vertex_colors is None:
            sparse_subdirs = sorted(workspace.sparse.iterdir())
            if sparse_subdirs:
                # COLMAP exports sparse points as points3D.ply in the model directory.
                sparse_pcd = sparse_subdirs[0] / "points3D.ply"
                if sparse_pcd.exists():
                    on_progress("Using sparse point cloud for albedo colors...")
                    vertex_colors = bake_albedo_from_pointcloud(
                        mesh, sparse_pcd, on_progress
                    )

        # If we have vertex colors, rasterize them to a texture.
        if vertex_colors is not None:
            try:
                on_progress("Rasterizing albedo to texture (this may take a moment)...")
                albedo_img = vertex_colors_to_texture(uvs, faces, vertex_colors)
                albedo_img.save(str(workspace.textures / "albedo.png"))
                on_progress("Albedo texture saved: albedo.png")
            except Exception as e:
                on_progress(f"Albedo baking failed: {e}")
        else:
            on_progress(
                "No colored point cloud found — albedo baking skipped. "
                "The mesh will export without color texture."
            )

        # --- Normal map: always available from mesh vertex normals ---
        try:
            on_progress("Baking tangent-space normal map...")
            normal_img = bake_normal_map(mesh)
            normal_img.save(str(workspace.textures / "normal.png"))
            on_progress("Normal map saved: normal.png")
        except Exception as e:
            on_progress(f"Normal map baking failed: {e}")

        # --- AO: hemisphere ray casting via Open3D ---
        try:
            ao_img = bake_ao(mesh, num_rays=64, on_progress=on_progress)
            ao_img.save(str(workspace.textures / "ao.png"))
            on_progress("AO map saved: ao.png")
        except Exception as e:
            on_progress(f"AO baking failed: {e}")

        on_progress(
            "Texture baking complete. "
            "Textures saved to workspace/textures/"
        )
