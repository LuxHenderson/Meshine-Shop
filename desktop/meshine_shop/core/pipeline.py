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

COLMAP (via pycolmap) will be the engine executing these stages.
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
}
