"""
Photogrammetry processing pipeline — stage definitions.

This module defines the discrete stages of the photogrammetry reconstruction
workflow. Each stage represents a distinct, logged step that the user can
monitor in the UI's processing queue.

The pipeline follows the standard photogrammetry reconstruction sequence:
    1. Ingest — Load and validate input images
    2. Feature Extraction — Detect keypoints and descriptors in each image
    3. Sparse Reconstruction (SfM) — Match features across images and
       triangulate camera positions + a sparse 3D point cloud
    4. Dense Reconstruction — Generate a high-density point cloud from
       the camera positions established in step 3
    5. Mesh Reconstruction — Convert the dense point cloud into a
       triangulated surface mesh
    6. Texture Mapping — Project the original photo colors onto the
       mesh surface to create a textured 3D model
    7. Mesh Decimation — Reduce polygon count to match the user's chosen
       quality preset (Mobile / PC / Cinematic)
    8. UV Unwrapping — Generate non-overlapping UV coordinates for the
       decimated mesh using xatlas. UV maps are required for texture
       baking (Phase 2c) and PBR material export to game engines.

The stage constants defined here are used throughout the app to track
progress, update the UI, and log output consistently.
"""


class PipelineStage:
    """
    String constants identifying each pipeline stage.

    These are used as keys for progress tracking, logging, and UI updates.
    Using a class with constants (rather than an enum) keeps things simple
    and allows direct string comparison without .value access.
    """
    INGEST = "ingest"
    FEATURES = "feature_extraction"
    SPARSE = "sparse_reconstruction"
    DENSE = "dense_reconstruction"
    MESH = "mesh_reconstruction"
    TEXTURE = "texture_mapping"
    DECIMATION = "decimation"
    UV_UNWRAP = "uv_unwrapping"


# Ordered list of stages — defines the sequence the pipeline executes.
# The UI uses this to render progress indicators in the correct order
# and to determine which stage comes next.
STAGE_ORDER = [
    PipelineStage.INGEST,
    PipelineStage.FEATURES,
    PipelineStage.SPARSE,
    PipelineStage.DENSE,
    PipelineStage.MESH,
    PipelineStage.TEXTURE,
    PipelineStage.DECIMATION,
    PipelineStage.UV_UNWRAP,
]

# Human-readable display names for each stage, used in the processing
# queue UI to show users what's happening in plain language.
STAGE_DISPLAY_NAMES = {
    PipelineStage.INGEST: "Importing Images",
    PipelineStage.FEATURES: "Extracting Features",
    PipelineStage.SPARSE: "Sparse Reconstruction",
    PipelineStage.DENSE: "Dense Reconstruction",
    PipelineStage.MESH: "Building Mesh",
    PipelineStage.TEXTURE: "Texture Mapping",
    PipelineStage.DECIMATION: "Mesh Decimation",
    PipelineStage.UV_UNWRAP: "UV Unwrapping",
}

# Quality presets for mesh decimation. Each preset maps to a target
# triangle count appropriate for the target platform. The decimation
# stage uses these to determine how aggressively to reduce the mesh.
QUALITY_PRESETS = {
    "Mobile (5K triangles)": 5_000,
    "PC (25K triangles)": 25_000,
    "Cinematic (100K triangles)": 100_000,
}

# Export format options presented to the user in the Export view.
# Keys are human-readable labels; values are file extensions used by
# trimesh to determine the output format.
EXPORT_FORMATS = {
    "OBJ (.obj)": ".obj",
    "glTF Binary (.glb)": ".glb",
}
