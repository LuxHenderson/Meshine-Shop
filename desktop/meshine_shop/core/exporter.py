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
      per the glTF 2.0 specification.
    - OBJ (.obj): a bundle folder is created at dest_path.parent/dest_path.stem/,
      containing the .obj, an .mtl file, and all available texture PNGs. The MTL
      uses standard PBR extension directives (map_Pr, map_Pm) for roughness and
      metallic so the bundle loads correctly in Blender and similar tools.

Uses trimesh for format conversion and PBR material attachment.
"""

import shutil
from pathlib import Path

import trimesh


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
        # glTF Binary: trimesh serialises the entire mesh + materials + textures
        # into a single self-contained binary file. No separate PNGs needed.
        mesh.export(str(dest_path))
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
