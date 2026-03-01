"""
Mesh segmentation for the Meshine Shop AI texturing pipeline.

Automatically divides a UV-mapped mesh into semantically meaningful regions so
that per-segment material prompts can drive separate Stability AI API calls.
This produces multi-material textures (e.g. leather body + chrome blades + skin
face) from a single combined user prompt like "leather body with chrome blades".

Two-tier segmentation algorithm
--------------------------------
Tier 1 — Connected components (preferred):
    Uses trimesh.graph.connected_components() to find genuinely disconnected
    mesh pieces. A character holding a weapon, a product on a stand, or a shoe
    with separate lace pieces will naturally segment here. Tiny components
    (< 1% of total faces) are merged into the nearest large component by
    centroid distance to avoid wasting API credits on invisible micro-fragments.

Tier 2 — K-means fallback (single-piece meshes):
    When the mesh is one connected piece (e.g. a scanned rock, mug, or shoe
    sole), spatial K-means clustering on normalized vertex positions divides it
    into k=3 volumetric regions. Each face is assigned to the cluster that the
    majority of its three vertices belong to. Region boundaries are spatial, not
    semantic, but still enable meaningfully different materials per region.

Segment count is capped at max_segments=5 to keep Stability AI credit usage
predictable (N segments × 6 views = N×6 API calls per run).

Public API
----------
segment_mesh(mesh, max_segments=5) -> list[np.ndarray]
    Returns a list of vertex-index arrays, sorted by segment size (largest
    first). Each array contains the integer indices into mesh.vertices that
    belong to that segment. Segments are mutually exclusive and together cover
    all vertices exactly once.

assign_segment_prompts(segment_vertex_lists, user_prompt) -> list[str]
    Parses the user prompt for material keywords in left-to-right order.
    Returns one prompt string per segment. Fewer keywords than segments: last
    keyword repeats. No keywords found: full prompt used for all segments.
"""

import numpy as np
import trimesh
import trimesh.graph
from scipy.cluster.vq import kmeans2
from scipy import stats
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_keyword_positions(prompt: str) -> list[tuple[int, str]]:
    """
    Scan the prompt for all known material keywords and return their positions
    in left-to-right order, resolving substring overlaps.

    MATERIAL_PRESETS is imported lazily from ai_texture_gen to avoid a
    circular import at module level (mesh_segmentation ← ai_texture_gen,
    ai_texture_gen imports mesh_segmentation inside function bodies).

    Deduplication rule: if keyword A is entirely contained within keyword B's
    character span (same or earlier start position, same or later end), A is
    dropped. Example: "metal" inside "polished metal" → only "polished metal"
    is returned.

    Args:
        prompt: The user's material description string.

    Returns:
        List of (character_position, keyword) pairs sorted by position.
    """
    # Lazy import avoids circular dependency at module level.
    from meshine_shop.core.ai_texture_gen import MATERIAL_PRESETS

    lower = prompt.lower()

    # Collect all matches: (position, keyword).
    hits: list[tuple[int, str]] = []
    for keyword in MATERIAL_PRESETS:
        pos = lower.find(keyword)
        if pos != -1:
            hits.append((pos, keyword))

    # Sort by position ascending (left-to-right in the prompt string).
    hits.sort(key=lambda x: x[0])

    # Deduplicate: remove any hit that is entirely within the span of another hit.
    # A hit (pos_a, kw_a) is dominated if there exists (pos_b, kw_b) such that:
    #   pos_b <= pos_a  AND  pos_b + len(kw_b) >= pos_a + len(kw_a)
    # i.e. kw_b's span fully contains kw_a's span.
    cleaned: list[tuple[int, str]] = []
    for i, (pos_a, kw_a) in enumerate(hits):
        dominated = False
        end_a = pos_a + len(kw_a)
        for j, (pos_b, kw_b) in enumerate(hits):
            if i == j:
                continue
            end_b = pos_b + len(kw_b)
            if pos_b <= pos_a and end_b >= end_a and len(kw_b) > len(kw_a):
                dominated = True
                break
        if not dominated:
            cleaned.append((pos_a, kw_a))

    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_mesh(
    mesh: "trimesh.Trimesh",
    max_segments: int = 5,
) -> list[np.ndarray]:
    """
    Divide the mesh into discrete vertex-index segments using a two-tier strategy.

    Tier 1 (connected components) is used when the mesh has multiple disconnected
    pieces. Tier 2 (K-means on vertex positions) is used when the mesh is a single
    connected piece.

    The returned segments:
    - Are mutually exclusive (no vertex appears in two segments).
    - Together cover ALL vertices (coverage is guaranteed by a final check that
      appends any uncovered vertices to the largest segment).
    - Are sorted descending by size (largest segment first).

    Args:
        mesh:         A trimesh.Trimesh with valid faces and vertices.
        max_segments: Maximum number of output segments. Excess segments beyond
                      this cap are merged into the last kept segment.

    Returns:
        List of np.ndarray (dtype int64), each containing vertex indices.
    """
    # Guard: empty mesh — return one segment covering all vertices.
    if len(mesh.faces) == 0:
        return [np.arange(len(mesh.vertices), dtype=np.int64)]

    # Guard: trivial cap.
    if max_segments < 1:
        max_segments = 1

    # ---- Tier 1: Connected components ----
    # trimesh returns a list of face-index arrays, one per connected component.
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),  # include all faces, even isolated ones
    )

    if len(components) > 1:
        vert_segments = _tier1_connected_components(mesh, components, max_segments)
    else:
        # Single connected piece — fall through to K-means.
        vert_segments = _tier2_kmeans(mesh, max_segments)

    # ---- Final coverage check ----
    # Guarantee every vertex is in exactly one segment.
    if vert_segments:
        covered = np.concatenate(vert_segments)
        all_verts = np.arange(len(mesh.vertices), dtype=np.int64)
        uncovered = np.setdiff1d(all_verts, covered)
        if len(uncovered) > 0:
            # Append orphan vertices to the largest segment.
            vert_segments[0] = np.unique(
                np.concatenate([vert_segments[0], uncovered])
            )
    else:
        # Fallback: something went wrong — return one segment for the whole mesh.
        vert_segments = [np.arange(len(mesh.vertices), dtype=np.int64)]

    # Sort descending by size (largest segment first).
    vert_segments.sort(key=len, reverse=True)
    return vert_segments


def assign_segment_prompts(
    segment_vertex_lists: list[np.ndarray],
    user_prompt: str,
) -> list[str]:
    """
    Map one material prompt string to each segment.

    Parsing rules:
    - All material keywords are found in the prompt in left-to-right order.
    - Segment 0 (largest) gets the first keyword, segment 1 the second, etc.
    - If there are more segments than keywords, the last keyword repeats.
    - If no material keywords are found, all segments receive the full user
      prompt unchanged (same behavior as before segmentation was added).

    Each segment prompt is prefixed with "{keyword} surface, " so that
    _material_pbr_params() always picks up the correct primary material via
    the earliest-position logic already in ai_texture_gen.py.

    Args:
        segment_vertex_lists: Output of segment_mesh() — list of vertex-index
                               arrays, sorted largest first.
        user_prompt:          The user's full material description string.

    Returns:
        List of prompt strings, one per segment, same length as
        segment_vertex_lists.
    """
    n = len(segment_vertex_lists)
    keyword_hits = _find_keyword_positions(user_prompt)

    # No material keywords found — use the full prompt for all segments.
    if not keyword_hits:
        return [user_prompt] * n

    # Extract keywords in left-to-right order.
    keywords = [kw for _, kw in keyword_hits]

    # Build per-segment prompts.
    prompts: list[str] = []
    for i in range(n):
        # Repeat last keyword when there are more segments than keywords.
        kw = keywords[min(i, len(keywords) - 1)]
        # Prefix ensures _material_pbr_params() picks the right material even
        # when the full prompt contains multiple material keywords further along.
        seg_prompt = f"{kw} surface, {user_prompt}"
        prompts.append(seg_prompt)

    return prompts


# ---------------------------------------------------------------------------
# Internal implementation helpers
# ---------------------------------------------------------------------------

def _tier1_connected_components(
    mesh: "trimesh.Trimesh",
    components: list[np.ndarray],
    max_segments: int,
) -> list[np.ndarray]:
    """
    Convert trimesh face-index component arrays to vertex-index segment arrays,
    merge tiny fragments, and cap at max_segments.

    Args:
        mesh:        The source mesh.
        components:  Output of trimesh.graph.connected_components — list of
                     face-index arrays, one per disconnected piece.
        max_segments: Cap on the number of returned segments.

    Returns:
        List of vertex-index arrays (dtype int64).
    """
    total_faces = len(mesh.faces)
    # Threshold: components with fewer faces than this are merged into the
    # nearest large component to avoid wasting API calls on micro-fragments.
    min_faces = max(1, int(0.01 * total_faces))

    # Split into large and small components.
    large_face_arrays: list[np.ndarray] = []
    small_face_arrays: list[np.ndarray] = []
    for comp in components:
        if len(comp) >= min_faces:
            large_face_arrays.append(comp)
        else:
            small_face_arrays.append(comp)

    # If everything is tiny (pathological mesh), treat all as large.
    if not large_face_arrays:
        large_face_arrays = list(components)
        small_face_arrays = []

    # Convert face-index arrays to vertex-index arrays for large components.
    large_verts: list[list] = [
        list(np.unique(mesh.faces[f])) for f in large_face_arrays
    ]

    # Merge small components into the nearest large component by centroid distance.
    if small_face_arrays and large_verts:
        # Build centroids for each large component.
        large_centroids = np.array([
            mesh.vertices[np.unique(mesh.faces[f])].mean(axis=0)
            for f in large_face_arrays
        ])
        tree = cKDTree(large_centroids)

        for small_f in small_face_arrays:
            small_v = np.unique(mesh.faces[small_f])
            small_centroid = mesh.vertices[small_v].mean(axis=0)
            _, nearest_idx = tree.query(small_centroid)
            large_verts[nearest_idx].extend(small_v.tolist())

    # De-duplicate vertex indices within each large component after merging.
    vert_segments = [np.unique(np.array(vl, dtype=np.int64)) for vl in large_verts]

    # Sort descending by size before capping.
    vert_segments.sort(key=len, reverse=True)

    # Cap at max_segments: merge overflow into the last kept segment.
    if len(vert_segments) > max_segments:
        overflow = np.concatenate(vert_segments[max_segments:])
        vert_segments[max_segments - 1] = np.unique(
            np.concatenate([vert_segments[max_segments - 1], overflow])
        )
        vert_segments = vert_segments[:max_segments]

    return vert_segments


def _tier2_kmeans(
    mesh: "trimesh.Trimesh",
    max_segments: int,
) -> list[np.ndarray]:
    """
    Partition a single connected mesh into spatial regions using K-means on
    normalized vertex positions.

    K is fixed at 3 (default), capped by max_segments. Each face is assigned
    to the cluster that the majority of its three vertices belong to, then
    vertex-index arrays are built from the face assignments.

    Args:
        mesh:        A single connected trimesh.Trimesh.
        max_segments: Upper bound on k (number of clusters).

    Returns:
        List of vertex-index arrays (dtype int64), sorted descending by size.
    """
    k = min(3, max_segments, len(mesh.vertices))

    # Normalize vertex positions to [-1, 1] so the coordinate range does not
    # skew the Euclidean distance metric used by K-means.
    verts = np.array(mesh.vertices, dtype=np.float32)
    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    v_range = np.maximum(v_max - v_min, 1e-6)
    verts_norm = (verts - v_min) / v_range * 2.0 - 1.0  # (V, 3) in [-1, 1]

    # Run K-means. minit='points' selects k random data points as initial
    # centroids — more stable than 'random' (random points in data range).
    try:
        _, vertex_labels = kmeans2(
            verts_norm, k=k, minit="points", iter=20, missing="warn"
        )
    except Exception:
        # If K-means fails (e.g. too few unique vertices), return one segment.
        return [np.arange(len(mesh.vertices), dtype=np.int64)]

    # Assign each face to the cluster that its majority vertex belongs to.
    # scipy.stats.mode returns shape (F,) in modern scipy; .ravel() is a
    # defensive call for older versions that returned (F, 1).
    face_vertex_labels = vertex_labels[mesh.faces]  # (F, 3)
    face_cluster = stats.mode(face_vertex_labels, axis=1).mode.ravel().astype(int)

    # Strictly assign each VERTEX to one cluster using majority-vote across all
    # faces that reference it. Vertices on cluster boundaries would otherwise
    # appear in multiple segments (violating the strict-partition guarantee)
    # because they are shared by faces of different clusters.
    #
    # For each cluster, scatter-add 1 to every vertex referenced by faces in
    # that cluster. Then argmax gives each vertex its winning cluster.
    n_verts_local = len(mesh.vertices)
    cluster_votes = np.zeros((n_verts_local, k), dtype=np.int32)
    for cid in range(k):
        face_mask = face_cluster == cid
        verts_in_cid = mesh.faces[face_mask].ravel()   # all vertex refs in this cluster
        np.add.at(cluster_votes[:, cid], verts_in_cid, 1)

    vertex_assignment = cluster_votes.argmax(axis=1)  # (V,) — one cluster per vertex

    # Build vertex-index arrays from strict vertex assignment.
    vert_segments: list[np.ndarray] = []
    for cid in range(k):
        seg_verts = np.where(vertex_assignment == cid)[0].astype(np.int64)
        if len(seg_verts) > 0:
            vert_segments.append(seg_verts)

    # Sort descending by size.
    vert_segments.sort(key=len, reverse=True)

    # Fallback: if K-means produced nothing, return one segment.
    if not vert_segments:
        return [np.arange(len(mesh.vertices), dtype=np.int64)]

    return vert_segments
