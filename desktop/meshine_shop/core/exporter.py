"""
Mesh export module for Meshine Shop.

Converts the pipeline's mesh output into standard 3D formats that any
modeling tool (Blender, Unity, Unreal) can import. This is the final
user-facing step of the pipeline.

Source mesh formats supported (all loaded via trimesh):
    - OBJ (.obj)   — Phase 2b output with UV coordinates (meshed_uv.obj)
    - PLY (.ply)   — Decimated-only fallback if UV unwrapping was skipped

Supported export formats:
    - OBJ (.obj)         — Universal format, widest compatibility
    - glTF Binary (.glb) — Modern standard, preferred by web and game engines
    - FBX (.fbx)         — Autodesk format, primary for Unreal Engine and DCC tools
                           Requires Assimp CLI on PATH: macOS: brew install assimp

When Phase 2c + 2d texture baking has run, the workspace will contain up to
five PBR maps in workspace/textures/. These are embedded into exports:
    - glTF Binary (.glb): all textures embedded as a PBR material — fully
      self-contained, no external PNG files needed. Roughness and metallic are
      packed into a single metallicRoughnessTexture (G=roughness, B=metallic)
      per the glTF 2.0 specification. Vertex tangents (TANGENT attribute) are
      pre-computed and embedded so all glTF viewers decode normal maps correctly
      without runtime approximation. This is Phase 2i.
    - OBJ (.obj): a bundle folder is created at dest_path.parent/dest_path.stem/,
      containing the .obj, an .mtl file, and all available texture PNGs. The MTL
      uses standard PBR extension directives (map_Pr, map_Pm) for roughness and
      metallic so the bundle loads correctly in Blender and similar tools.

Uses trimesh for format conversion and PBR material attachment.
"""

import shutil
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Vertex tangent computation — Phase 2i
# ---------------------------------------------------------------------------

def _compute_vertex_tangents(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """
    Compute per-vertex tangent vectors for correct normal map rendering in glTF.

    The glTF 2.0 spec requires a TANGENT vertex attribute for any material that
    uses a normal map (normalTexture). Without it, viewers must generate tangents
    at runtime — but different implementations (three.js, Babylon.js, Blender,
    Unity, Unreal) use slightly different tangent generation algorithms, causing
    inconsistent and often incorrect normal map rendering.

    Pre-computing tangents using the standard MikkTSpace-compatible algorithm
    and embedding them in the GLB guarantees that every compliant viewer will
    decode the normal map identically, no matter where the file is opened.

    Algorithm — UV partial derivatives method:
        For each triangle, the tangent T and bitangent B are the 3D vectors that
        align with the U and V axes of the UV mapping respectively. They are
        derived from the ratio of 3D edge vectors to UV edge vectors:

            r = 1 / (dU1*dV2 - dU2*dV1)     # UV signed area reciprocal
            T = (dV2 * edge1 - dV1 * edge2) * r
            B = (dU1 * edge2 - dU2 * edge1) * r

        Per-vertex tangents are the area-weighted sum of adjacent face tangents,
        Gram-Schmidt orthogonalized against the vertex normal. The w component
        encodes tangent basis handedness (±1), telling the shader which direction
        to derive the bitangent: B = w * (N × T).

    Args:
        mesh: trimesh.Trimesh with UV coordinates in mesh.visual.uv.

    Returns:
        (N, 4) float32 array — (Tx, Ty, Tz, Tw) per vertex, or None if the mesh
        has no UV coordinates or tangent computation fails.
    """
    # Guard: UV coordinates are required for tangent computation.
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        return None

    uvs = np.array(mesh.visual.uv, dtype=np.float64)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    # Guard: UV count must match vertex count for consistent indexing.
    if len(uvs) != len(vertices) or len(faces) == 0:
        return None

    # Vertex normals are needed for Gram-Schmidt orthogonalization.
    # trimesh auto-computes them from face normals if not stored explicitly.
    normals = np.array(mesh.vertex_normals, dtype=np.float64)

    n_verts = len(vertices)

    # Accumulators for per-vertex tangent and bitangent sums.
    # Using float64 throughout to avoid precision loss in the accumulation step.
    tan1 = np.zeros((n_verts, 3), dtype=np.float64)  # tangent accumulator
    tan2 = np.zeros((n_verts, 3), dtype=np.float64)  # bitangent accumulator

    # --- Step 1: Compute per-triangle tangent and bitangent ---
    # Index into vertex positions and UVs for all three corners at once.
    v0 = vertices[faces[:, 0]]   # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    uv0 = uvs[faces[:, 0]]      # (F, 2)
    uv1 = uvs[faces[:, 1]]
    uv2 = uvs[faces[:, 2]]

    # 3D edge vectors along the triangle sides.
    edge1 = v1 - v0              # (F, 3)
    edge2 = v2 - v0

    # UV edge vectors (differences in the texture domain).
    duv1 = uv1 - uv0            # (F, 2)
    duv2 = uv2 - uv0

    # Denominator: signed area of the UV triangle.
    # Non-zero only for non-degenerate UV triangles.
    denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]  # (F,)
    valid = np.abs(denom) > 1e-8

    # Safe reciprocal — degenerate triangles contribute zero tangent/bitangent.
    r = np.where(valid, 1.0 / np.where(valid, denom, 1.0), 0.0)  # (F,)
    r3 = r[:, np.newaxis]                                          # (F, 1)

    # Per-face tangent: the 3D direction that aligns with the U axis of the UV.
    tang  = (duv2[:, 1:2] * edge1 - duv1[:, 1:2] * edge2) * r3   # (F, 3)

    # Per-face bitangent: the 3D direction that aligns with the V axis of the UV.
    bitan = (duv1[:, 0:1] * edge2 - duv2[:, 0:1] * edge1) * r3   # (F, 3)

    # --- Step 2: Accumulate per-triangle contributions into per-vertex sums ---
    # Each vertex accumulates the tangent from all triangles that share it.
    # np.add.at is the correct tool here (regular indexing doesn't accumulate).
    np.add.at(tan1, faces[:, 0], tang)
    np.add.at(tan1, faces[:, 1], tang)
    np.add.at(tan1, faces[:, 2], tang)

    np.add.at(tan2, faces[:, 0], bitan)
    np.add.at(tan2, faces[:, 1], bitan)
    np.add.at(tan2, faces[:, 2], bitan)

    # --- Step 3: Gram-Schmidt orthogonalize tangent against vertex normal ---
    # The accumulated tangent may not be perfectly perpendicular to the normal
    # (due to UV seams and averaging). Gram-Schmidt projects the tangent onto
    # the plane perpendicular to the normal, then normalizes.
    dot = np.sum(normals * tan1, axis=1, keepdims=True)  # N·T per vertex
    tangents = tan1 - normals * dot                       # T - N*(N·T)

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)  # |T|

    # Normalize. For degenerate tangents (|T| ≈ 0), fall back to a stable
    # perpendicular: prefer [1,0,0] unless the normal is nearly parallel to it,
    # in which case use [0,1,0]. This keeps the tangent space well-formed.
    degenerate = norms[:, 0] <= 1e-8
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    tangents = tangents / safe_norms

    if degenerate.any():
        # Compute a stable fallback for each degenerate vertex individually.
        # Vectorized: pick [1,0,0] when N is not nearly parallel to X-axis,
        # else use [0,1,0].
        parallel_to_x = np.abs(normals[:, 0]) >= 0.9
        fallback = np.where(
            parallel_to_x[:, np.newaxis],
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        )
        dot_fb = np.sum(normals * fallback, axis=1, keepdims=True)
        fb_tang = fallback - normals * dot_fb
        fb_norm = np.linalg.norm(fb_tang, axis=1, keepdims=True)
        fb_safe = np.where(fb_norm > 1e-8, fb_norm, 1.0)
        fb_tang = fb_tang / fb_safe
        tangents[degenerate] = fb_tang[degenerate]

    # --- Step 4: Compute handedness (w component) ---
    # w = sign(N × T · B): +1 if the tangent-bitangent-normal basis is
    # right-handed, -1 if it's mirrored (common at UV seams and flipped UVs).
    # The shader uses: B = w * (N × T)  — so w must be preserved in the export.
    cross = np.cross(normals, tangents)               # N × T
    handedness = np.sign(np.sum(cross * tan2, axis=1))
    handedness = np.where(handedness == 0, 1.0, handedness)  # default +1

    # Pack tangent (xyz) + handedness (w) into the required (N, 4) layout.
    result = np.concatenate(
        [tangents, handedness[:, np.newaxis]], axis=1
    )

    return result.astype(np.float32)


def _fix_glb_tangent_attribute(glb_path: Path) -> None:
    """
    Post-process a GLB file to fix two trimesh output issues in one JSON pass.

    Fix 1 — TANGENT attribute name:
        trimesh prefixes all vertex_attributes entries with '_' (treating them
        as application-specific custom attributes per glTF spec). TANGENT is a
        standard glTF 2.0 attribute and must NOT have the underscore prefix —
        viewers look for exactly 'TANGENT', not '_TANGENT'.
        Without this: MESH_PRIMITIVE_GENERATED_TANGENT_SPACE warning.
        With this: viewers use the embedded tangents for correct normal maps.

    Fix 2 — bufferView.target missing:
        trimesh omits the optional 'target' field on glTF bufferViews.
        The glTF spec recommends setting it so the GPU driver knows the intended
        use before any mesh is instantiated:
            34962 (GL_ARRAY_BUFFER)         — vertex attribute data
            34963 (GL_ELEMENT_ARRAY_BUFFER) — index data
        Without this: BUFFER_VIEW_TARGET_MISSING hints from the validator.
        With this: the file passes the validator hint-free.

    GLB format (glTF 2.0 spec §5.1):
        Bytes 0–11:    header (magic "glTF", version=2, total_length)
        Bytes 12+:     JSON chunk (4-byte length, 4-byte type "JSON", data)
        After JSON:    BIN  chunk (4-byte length, 4-byte type "BIN\\0", data)

    JSON data is padded to 4-byte boundary with trailing spaces (0x20).
    The chunk length field includes the padding. BIN chunk is never touched.

    Args:
        glb_path: Path to the GLB file to fix (modified in-place).
    """
    import json
    import struct

    with open(glb_path, "rb") as f:
        data = bytearray(f.read())

    # Validate GLB magic and version.
    if len(data) < 20 or data[:4] != b"glTF":
        return  # Not a GLB, skip silently.

    version = struct.unpack_from("<I", data, 4)[0]
    if version != 2:
        return  # Only glTF 2.0 is supported.

    # JSON chunk starts at byte 12.
    json_chunk_length = struct.unpack_from("<I", data, 12)[0]  # incl. padding
    json_chunk_type = data[16:20]

    if json_chunk_type != b"JSON":
        return  # Unexpected chunk order, don't touch the file.

    # Extract and parse JSON (strip trailing space/null padding first).
    json_raw = data[20 : 20 + json_chunk_length]
    json_str = json_raw.rstrip(b" ").rstrip(b"\x00").decode("utf-8")

    try:
        gltf = json.loads(json_str)
    except json.JSONDecodeError:
        return  # Malformed JSON, don't touch the file.

    modified = False

    # --- Fix 1: rename '_TANGENT' → 'TANGENT' in mesh primitive attributes ---
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "_TANGENT" in attrs:
                # Preserve the accessor index — only the key name changes.
                # The binary vertex data is unaffected.
                attrs["TANGENT"] = attrs.pop("_TANGENT")
                modified = True

    # --- Fix 2: add bufferView.target for vertex and index buffer views ---
    # Build a mapping: bufferView index → intended GL target constant.
    # We walk every mesh primitive to find which accessors are used as
    # indices (target=34963) and which are used as vertex attributes (34962).
    # An accessor's bufferView gets the target implied by its first reference
    # (a bufferView should be used for only one purpose).
    GL_ARRAY_BUFFER         = 34962  # vertex attribute data
    GL_ELEMENT_ARRAY_BUFFER = 34963  # index / element data

    bv_targets: dict[int, int] = {}  # bufferView index → GL target

    accessors = gltf.get("accessors", [])

    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            # Indices accessor → element buffer
            idx_accessor = prim.get("indices")
            if idx_accessor is not None and idx_accessor < len(accessors):
                bv = accessors[idx_accessor].get("bufferView")
                if bv is not None:
                    bv_targets[bv] = GL_ELEMENT_ARRAY_BUFFER

            # Attribute accessors → vertex array buffer
            for accessor_idx in prim.get("attributes", {}).values():
                if accessor_idx < len(accessors):
                    bv = accessors[accessor_idx].get("bufferView")
                    if bv is not None and bv not in bv_targets:
                        # Don't overwrite index target if somehow shared
                        bv_targets[bv] = GL_ARRAY_BUFFER

    # Write the target values into each bufferView that needs one.
    buffer_views = gltf.get("bufferViews", [])
    for bv_idx, target in bv_targets.items():
        if bv_idx < len(buffer_views):
            if "target" not in buffer_views[bv_idx]:
                buffer_views[bv_idx]["target"] = target
                modified = True

    if not modified:
        return  # Nothing changed — no need to rewrite the file.

    # Re-serialize the JSON (compact — no extra whitespace).
    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")

    # Pad to 4-byte boundary with ASCII spaces per glTF spec §5.1.
    pad_needed = (-len(new_json_bytes)) % 4
    new_json_padded = new_json_bytes + b" " * pad_needed
    new_json_chunk_length = len(new_json_padded)

    # The BIN chunk starts immediately after the original JSON chunk.
    bin_chunk_start = 20 + json_chunk_length
    bin_chunk = bytes(data[bin_chunk_start:])  # Unchanged.

    # New total length: header(12) + JSON chunk header(8) + JSON + BIN.
    new_total = 12 + 8 + new_json_chunk_length + len(bin_chunk)

    # Reconstruct the GLB binary with the patched JSON.
    new_data = bytearray()
    new_data += b"glTF"                                     # magic
    new_data += struct.pack("<I", 2)                        # version = 2
    new_data += struct.pack("<I", new_total)                # total length
    new_data += struct.pack("<I", new_json_chunk_length)    # JSON chunk length
    new_data += b"JSON"                                     # JSON chunk type
    new_data += new_json_padded                             # updated JSON
    new_data += bin_chunk                                   # BIN chunk (intact)

    with open(glb_path, "wb") as f:
        f.write(new_data)


def _load_trimesh(source_mesh: Path, **kwargs) -> trimesh.Trimesh:
    """
    Load a mesh file via trimesh and ensure we always get a Trimesh object.

    trimesh.load() can return either a trimesh.Trimesh or a trimesh.Scene
    depending on whether the file contains multiple material groups.
    OBJ files with 'usemtl' directives often return a Scene with a single
    geometry. This helper collapses the Scene into a single Trimesh so the
    rest of the export pipeline can rely on a consistent type.

    Args:
        source_mesh: Path to the mesh file (.obj or .ply).
        **kwargs:    Forwarded to trimesh.load() (e.g. process=False).

    Returns:
        trimesh.Trimesh — a single merged mesh object.
    """
    result = trimesh.load(source_mesh, **kwargs)

    if isinstance(result, trimesh.Scene):
        # Scene with (possibly many) sub-geometries — merge into one Trimesh.
        # This preserves UV coordinates and vertex data across all sub-meshes.
        geometries = list(result.geometry.values())
        if len(geometries) == 1:
            # Single geometry inside the Scene — unwrap it directly.
            result = geometries[0]
        else:
            # Multiple geometries — concatenate. UV / visual data may be lost
            # across the merge boundary, but geometry is preserved.
            result = trimesh.util.concatenate(geometries)

    return result


def load_mesh_info(source_mesh: Path) -> dict:
    """
    Load a mesh file and return summary information for the UI.

    Accepts any format supported by trimesh (PLY, OBJ, etc.).

    Returns a dict with:
        - vertices: int — number of vertices in the mesh
        - triangles: int — number of triangular faces
        - file_size_mb: float — source file size in megabytes

    This is called when the pipeline completes so the Export view can
    display mesh stats before the user decides on an export format.
    """
    mesh = _load_trimesh(source_mesh)

    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "file_size_mb": round(source_mesh.stat().st_size / (1024 * 1024), 2),
    }


def export_mesh(source_mesh: Path, dest_path: Path, workspace=None) -> Path:
    """
    Convert a mesh file to the format implied by dest_path's extension,
    optionally embedding baked PBR textures from the workspace.

    The source may be a PLY or OBJ file — trimesh handles both. When the
    source is the Phase 2b UV-unwrapped OBJ (meshed_uv.obj), UV coordinates
    are preserved in the export.

    If workspace is provided and workspace.textures/ contains baked textures
    (albedo.png, normal.png, ao.png, roughness.png, metallic.png), they are
    included:
        - glTF Binary (.glb): all textures embedded as PBR material. Roughness
          and metallic are packed into metallicRoughnessTexture (G/B channels)
          per glTF 2.0 spec. Fully self-contained, no separate PNG files needed.
        - OBJ (.obj): all files written into a bundle folder at
          dest_path.parent/stem/. MTL uses map_Pr/map_Pm for roughness/metallic.

    Args:
        source_mesh: Path to the pipeline's output mesh (OBJ or PLY).
        dest_path:   Where to save the exported file. The file extension
                     determines the format (.obj → OBJ, .glb → glTF Binary).
        workspace:   Optional WorkspacePaths. Passed when the app wants to
                     include baked textures in the export. If None, textures
                     are not included (geometry-only export).

    Returns:
        Path — the actual output location. For GLB this is dest_path itself.
        For OBJ with textures this is the bundle folder path.

    Raises:
        ValueError: If the destination extension isn't a supported format.
        FileNotFoundError: If the source mesh doesn't exist.
    """
    supported = {".obj", ".glb", ".fbx"}

    if not source_mesh.exists():
        raise FileNotFoundError(f"Source mesh not found: {source_mesh}")

    extension = dest_path.suffix.lower()
    if extension not in supported:
        raise ValueError(
            f"Unsupported export format '{extension}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )

    # Load the mesh via the helper that collapses Scene → Trimesh.
    # process=False preserves UV seam vertices from Phase 2b xatlas output.
    mesh = _load_trimesh(source_mesh, process=False)

    # --- Detect available baked textures ---
    # Check whether Phase 2c + 2d produced texture files in the workspace.
    albedo_path = None
    normal_path = None
    ao_path = None
    roughness_path = None
    metallic_path = None

    if workspace is not None:
        albedo_candidate   = workspace.textures / "albedo.png"
        normal_candidate   = workspace.textures / "normal.png"
        ao_candidate       = workspace.textures / "ao.png"
        roughness_candidate = workspace.textures / "roughness.png"
        metallic_candidate  = workspace.textures / "metallic.png"

        if albedo_candidate.exists():
            albedo_path = albedo_candidate
        if normal_candidate.exists():
            normal_path = normal_candidate
        if ao_candidate.exists():
            ao_path = ao_candidate
        if roughness_candidate.exists():
            roughness_path = roughness_candidate
        if metallic_candidate.exists():
            metallic_path = metallic_candidate

    has_textures = albedo_path is not None or normal_path is not None

    # --- Attach PBR material if textures are available ---
    if has_textures:
        from PIL import Image
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals

        # Load texture images as PIL Images.
        # PBRMaterial accepts PIL Images and handles encoding for glTF export.
        albedo_img    = Image.open(albedo_path)    if albedo_path    else None
        normal_img    = Image.open(normal_path)    if normal_path    else None
        ao_img        = Image.open(ao_path)        if ao_path        else None
        roughness_img = Image.open(roughness_path) if roughness_path else None
        metallic_img  = Image.open(metallic_path)  if metallic_path  else None

        # Build a PBR material with the available textures.
        # PBRMaterial maps to glTF's pbrMetallicRoughness extension.
        material_kwargs = {}

        if albedo_img:
            material_kwargs["baseColorTexture"] = albedo_img

        if normal_img:
            material_kwargs["normalTexture"] = normal_img

        if ao_img:
            # glTF uses the R channel of occlusionTexture for AO.
            # Convert greyscale AO to RGB so trimesh encodes it correctly.
            material_kwargs["occlusionTexture"] = ao_img.convert("RGB")

        if roughness_img or metallic_img:
            # glTF 2.0 spec: metallicRoughnessTexture packs roughness in the
            # G channel and metallic in the B channel of a single RGB image.
            # Build the combined texture from the available greyscale maps.
            ref_img = roughness_img if roughness_img else metallic_img
            size = ref_img.size  # (width, height)
            r_ch = Image.new("L", size, 0)   # R channel — unused (kept at 0)
            g_ch = roughness_img.convert("L") if roughness_img else Image.new("L", size, 128)
            b_ch = metallic_img.convert("L")  if metallic_img  else Image.new("L", size, 0)
            mr_texture = Image.merge("RGB", (r_ch, g_ch, b_ch))
            material_kwargs["metallicRoughnessTexture"] = mr_texture

        material = PBRMaterial(**material_kwargs)

        # Attach the material to the mesh while preserving the UV coordinates.
        # We keep the existing TextureVisuals UV array and replace the material.
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            mesh.visual = TextureVisuals(
                uv=mesh.visual.uv,
                material=material,
            )
        else:
            # Mesh has no UVs — attach material anyway (may not render correctly
            # in viewers, but at least the texture data is present in the file).
            mesh.visual.material = material

    # --- Export to the chosen format ---
    if extension == ".glb":
        # Pre-compute and embed vertex tangents when a normal map is present.
        #
        # glTF 2.0 spec §3.9.3: if a material uses normalTexture, the mesh
        # primitive MUST supply a TANGENT vertex attribute, or viewers must
        # generate tangents at runtime using an implementation-defined method.
        # Runtime tangent generation differs between three.js, Babylon.js,
        # Blender, Unity, and Unreal — causing inconsistent normal map rendering.
        # Pre-embedding ensures every viewer produces the same correct result.
        #
        # We only compute tangents when we have a normal map (the only texture
        # that uses the tangent space). Meshes without normal maps are unaffected.
        if has_textures and normal_img is not None:
            tangents = _compute_vertex_tangents(mesh)
            if tangents is not None:
                # TANGENT is the standard glTF attribute name (uppercase).
                # trimesh's GLTF exporter includes vertex_attributes entries
                # as mesh primitive attributes, so this will appear in the file
                # as the required TANGENT accessor.
                mesh.vertex_attributes["TANGENT"] = tangents

        # glTF Binary: trimesh serialises the entire mesh + materials + textures
        # into a single self-contained binary file. No separate PNGs needed.
        mesh.export(str(dest_path))

        # Post-process: rename '_TANGENT' → 'TANGENT' in the GLB JSON header.
        # trimesh prefixes all vertex_attributes with '_' (correct for custom
        # attributes), but TANGENT is a standard glTF attribute that must NOT
        # have the underscore. This binary patch fixes the key in the JSON chunk
        # without touching the vertex data — the bytes are already correct.
        _fix_glb_tangent_attribute(dest_path)

        return dest_path

    elif extension == ".obj":
        # OBJ export: the format requires multiple files (geometry + MTL + PNGs).
        # _export_obj_with_textures bundles them into a subfolder to keep the
        # user's destination directory tidy. Returns the bundle folder path.
        return _export_obj_with_textures(
            mesh, dest_path,
            albedo_path, normal_path, ao_path,
            roughness_path, metallic_path,
        )

    elif extension == ".fbx":
        # FBX export: geometry-only FBX (no embedded PBR material) bundled
        # with texture PNGs. DCC tools and Unreal Engine expect textures
        # delivered alongside the FBX and assigned via their own material editors.
        return _export_fbx(
            mesh, dest_path,
            albedo_path, normal_path, ao_path,
            roughness_path, metallic_path,
        )

    # Should never be reached given the supported check above.
    return dest_path


def _export_obj_with_textures(mesh, dest_path, albedo_path, normal_path, ao_path,
                              roughness_path=None, metallic_path=None) -> Path:
    """
    Export mesh as OBJ and bundle the texture files + MTL in a subfolder.

    OBJ format requires multiple files (geometry + MTL + textures). Rather than
    scattering these alongside the user's chosen dest_path, we create a bundle
    folder named dest_path.stem in dest_path.parent and write everything there.

    Example: if dest_path is ~/Desktop/mesh.obj and textures exist, creates:
        ~/Desktop/mesh/
            mesh.obj         — geometry (UV coordinates included)
            mesh.mtl         — material referencing the texture images
            albedo.png       — copied from workspace.textures/
            normal.png       — copied from workspace.textures/
            ao.png           — copied from workspace.textures/
            roughness.png    — copied from workspace.textures/ (Phase 2d)
            metallic.png     — copied from workspace.textures/ (Phase 2d)

    If no textures are available, the .obj is written directly to dest_path
    (single file, no subfolder needed).

    Args:
        mesh:          trimesh.Trimesh to export.
        dest_path:     Path the user chose (e.g. ~/Desktop/mesh.obj).
        albedo_path:   Path to albedo.png (or None).
        normal_path:   Path to normal.png (or None).
        ao_path:       Path to ao.png (or None).
        roughness_path: Path to roughness.png (or None).
        metallic_path:  Path to metallic.png (or None).

    Returns:
        Path — bundle folder (if textures were written) or the OBJ file path.
    """
    # Determine whether we have any textures to bundle.
    texture_names = {}  # maps role → filename string

    if albedo_path and albedo_path.exists():
        texture_names["albedo"] = "albedo.png"
    if normal_path and normal_path.exists():
        texture_names["normal"] = "normal.png"
    if ao_path and ao_path.exists():
        texture_names["ao"] = "ao.png"
    if roughness_path and roughness_path.exists():
        texture_names["roughness"] = "roughness.png"
    if metallic_path and metallic_path.exists():
        texture_names["metallic"] = "metallic.png"

    if not texture_names:
        # No textures — write the OBJ directly to the user's chosen path.
        mesh.export(str(dest_path))
        return dest_path

    # Textures are present — create a bundle folder to keep everything together.
    stem = dest_path.stem
    bundle_dir = dest_path.parent / stem
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Actual OBJ and MTL paths inside the bundle folder.
    actual_obj = bundle_dir / dest_path.name
    mtl_path = bundle_dir / f"{stem}.mtl"

    # Export the mesh geometry as OBJ inside the bundle folder.
    mesh.export(str(actual_obj))

    # Copy texture images into the bundle folder.
    if albedo_path and albedo_path.exists():
        shutil.copy(albedo_path, bundle_dir / "albedo.png")
    if normal_path and normal_path.exists():
        shutil.copy(normal_path, bundle_dir / "normal.png")
    if ao_path and ao_path.exists():
        shutil.copy(ao_path, bundle_dir / "ao.png")
    if roughness_path and roughness_path.exists():
        shutil.copy(roughness_path, bundle_dir / "roughness.png")
    if metallic_path and metallic_path.exists():
        shutil.copy(metallic_path, bundle_dir / "metallic.png")

    # Write the MTL file referencing all available maps.
    # Uses standard OBJ PBR extension directives supported by Blender and
    # most game engine importers: map_Kd (diffuse), map_bump (normal),
    # map_Ka (AO), map_Pr (roughness), map_Pm (metallic).
    with open(mtl_path, "w") as f:
        f.write("# Generated by Meshine Shop\n")
        f.write(f"newmtl {stem}_material\n")

        if "albedo" in texture_names:
            f.write(f"map_Kd {texture_names['albedo']}\n")

        if "normal" in texture_names:
            # map_bump is the standard OBJ normal map directive.
            f.write(f"map_bump {texture_names['normal']}\n")

        if "ao" in texture_names:
            # map_Ka is ambient — the closest standard OBJ channel for AO.
            f.write(f"map_Ka {texture_names['ao']}\n")

        if "roughness" in texture_names:
            # map_Pr is the OBJ PBR extension roughness directive.
            f.write(f"map_Pr {texture_names['roughness']}\n")

        if "metallic" in texture_names:
            # map_Pm is the OBJ PBR extension metallic directive.
            f.write(f"map_Pm {texture_names['metallic']}\n")

    # Ensure the OBJ file references our MTL file.
    # trimesh may have already written a 'mtllib' line from its own material —
    # we overwrite it to point to our custom MTL.
    try:
        with open(actual_obj, "r") as f:
            lines = f.readlines()

        mtl_line = f"mtllib {stem}.mtl\n"
        if lines and lines[0].startswith("mtllib"):
            lines[0] = mtl_line
        else:
            lines.insert(0, mtl_line)

        with open(actual_obj, "w") as f:
            f.writelines(lines)
    except Exception:
        # Non-fatal — OBJ is still valid geometry, just may not auto-load the MTL.
        pass

    # Return the bundle folder path so the app can display it in the success message.
    return bundle_dir


def _export_fbx(mesh, dest_path, albedo_path, normal_path, ao_path,
                roughness_path=None, metallic_path=None) -> Path:
    """
    Export mesh as FBX and bundle texture files in a subfolder.

    trimesh has no native FBX exporter. The conversion pipeline is:
        1. Write mesh to a temporary OBJ (trimesh handles this natively).
        2. Call the assimp CLI ('assimp export <in> <out>') to convert OBJ → FBX.
        3. Delete the temp directory.

    The assimp CLI is part of the standard assimp package:
        macOS:   brew install assimp
        Windows: download from https://github.com/assimp/assimp/releases
                 and add the bin directory to PATH.

    Textures are NOT embedded in the FBX binary — they are delivered as
    separate PNGs in the bundle folder. This matches how Unreal Engine, Maya,
    and 3ds Max expect FBX assets: geometry in the FBX, textures assigned
    manually via the DCC tool's material editor.

    Example output for ~/Desktop/mesh.fbx:
        ~/Desktop/mesh/
            mesh.fbx         — geometry with UV coordinates
            albedo.png       — PBR base colour
            normal.png       — tangent-space normal map
            ao.png           — ambient occlusion
            roughness.png    — surface roughness (Phase 2d)
            metallic.png     — metallic mask (Phase 2d)

    Args:
        mesh:           trimesh.Trimesh to export.
        dest_path:      Path the user chose (e.g. ~/Desktop/mesh.fbx).
        albedo_path:    Path to albedo.png (or None).
        normal_path:    Path to normal.png (or None).
        ao_path:        Path to ao.png (or None).
        roughness_path: Path to roughness.png (or None).
        metallic_path:  Path to metallic.png (or None).

    Returns:
        Path — the bundle folder containing the FBX and texture files.

    Raises:
        RuntimeError: If the assimp CLI is not on PATH, or if conversion fails.
    """
    import subprocess
    import tempfile

    # Guard: the assimp CLI must be available on PATH.
    # 'brew install assimp' on macOS installs both the shared library and the
    # CLI tool. We use the CLI here because pyassimp's library discovery does
    # not search Homebrew paths on macOS.
    assimp_bin = shutil.which("assimp")
    if not assimp_bin:
        raise RuntimeError(
            "FBX export requires the Assimp CLI tool ('assimp' on PATH).\n"
            "macOS:   brew install assimp\n"
            "Windows: download from https://github.com/assimp/assimp/releases\n"
            "         and add the bin directory to your PATH."
        )

    # Create the bundle folder for the FBX and its texture companions.
    stem = dest_path.stem
    bundle_dir = dest_path.parent / stem
    bundle_dir.mkdir(parents=True, exist_ok=True)

    actual_fbx = bundle_dir / dest_path.name

    # Export geometry via trimesh → temp OBJ → assimp CLI → FBX binary.
    #
    # trimesh OBJ export preserves UV coordinates from Phase 2b (xatlas).
    # assimp derives the output format from the .fbx extension automatically.
    temp_dir = Path(tempfile.mkdtemp())
    temp_obj = temp_dir / "temp_mesh.obj"
    try:
        # Step 1: write geometry to a temporary OBJ.
        # trimesh may also write a companion .mtl — both land in temp_dir.
        mesh.export(str(temp_obj))

        # Step 2: invoke the assimp CLI to convert OBJ → FBX binary.
        result = subprocess.run(
            [assimp_bin, "export", str(temp_obj), str(actual_fbx)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        cli_output = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"assimp CLI failed (exit {result.returncode}).\n{cli_output}"
            )
        # Sanity-check: assimp may exit 0 yet still fail to write the file.
        if not actual_fbx.exists():
            raise RuntimeError(
                f"assimp reported success but produced no output file.\n{cli_output}"
            )
    finally:
        # Step 3: clean up the temp directory regardless of success or failure.
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Copy all available PBR texture maps into the bundle folder.
    # Users can drag the entire folder into Unreal Engine, Maya, or Blender
    # and assign each map to the corresponding material slot.
    texture_sources = [
        (albedo_path,    "albedo.png"),
        (normal_path,    "normal.png"),
        (ao_path,        "ao.png"),
        (roughness_path, "roughness.png"),
        (metallic_path,  "metallic.png"),
    ]
    for src, name in texture_sources:
        if src and Path(src).exists():
            shutil.copy(src, bundle_dir / name)

    # Return the bundle folder so the app can surface the path to the user.
    return bundle_dir
