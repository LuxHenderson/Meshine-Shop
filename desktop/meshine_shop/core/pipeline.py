"""
Photogrammetry processing pipeline.

This module will orchestrate the full reconstruction workflow:
    1. Image ingestion
    2. Feature extraction
    3. Sparse point cloud (SfM)
    4. Dense point cloud
    5. Mesh reconstruction
    6. Texture mapping

Each stage will be a discrete, logged step visible in the UI.
"""


class PipelineStage:
    INGEST = "ingest"
    FEATURES = "feature_extraction"
    SPARSE = "sparse_reconstruction"
    DENSE = "dense_reconstruction"
    MESH = "mesh_reconstruction"
    TEXTURE = "texture_mapping"


STAGE_ORDER = [
    PipelineStage.INGEST,
    PipelineStage.FEATURES,
    PipelineStage.SPARSE,
    PipelineStage.DENSE,
    PipelineStage.MESH,
    PipelineStage.TEXTURE,
]
