# Meshine Shop — Case Study

## The Real-World Problem

Game developers and 3D artists working with photogrammetry face a fragmented toolchain. Capturing a real-world object and turning it into a usable game asset currently requires:

1. A capture app (or manual photography)
2. A reconstruction engine (COLMAP, RealityCapture, Metashape)
3. A mesh editor for cleanup and decimation (Blender, MeshLab)
4. A UV unwrapping tool
5. A texture baker
6. A material editor for PBR maps
7. An export tool to get it into the right format for Unity or Unreal

Each tool has its own interface, its own file formats, its own quirks. Intermediate files pile up. Quality degrades at handoff points. The process is slow, error-prone, and inaccessible to anyone who isn't already a 3D pipeline expert.

**Meshine Shop consolidates this entire workflow into a single desktop application.** Drop photos in, get a game-ready .FBX or .glTF out — with optimized meshes, clean UVs, and PBR textures.

## Constraints

- **Solo developer.** Every architectural decision has to be practical for one person to build, maintain, and extend.
- **No budget for commercial tools.** The entire stack is built on open-source technology — COLMAP, Open3D, xatlas, trimesh — to eliminate licensing costs and ensure the tool can be freely distributed commercially.
- **Cross-platform from day one.** The app must work on macOS and Windows without maintaining two separate codebases.
- **Portfolio flagship.** This isn't a weekend project — it's designed to demonstrate full-stack systems thinking, from GPU-bound compute pipelines to polished UI/UX.

## Thought Process and MVP Decisions

### Why a desktop app, not a web app?

Photogrammetry is computationally intensive — it involves gigabytes of image data, GPU-heavy 3D reconstruction, and local file system access. A web app would require uploading all that data to a server, introduce latency, and create hosting costs. A desktop app keeps everything local, fast, and free.

### Why Python + Qt, not Electron?

The core pipeline depends entirely on Python libraries: COLMAP (pycolmap), PyTorch, PyMeshLab, Open3D, trimesh. Using Electron would mean running Python as a subprocess behind a Node.js shell — adding IPC overhead, making debugging harder, and splitting the codebase across two languages. Python + PySide6 keeps everything in one process, one language, one ecosystem. It's also what the professional tools in this space use (Blender, Maya, Houdini are all C++/Qt).

### Why COLMAP?

Three reasons: (1) It has a Python API via pycolmap, so we can call it directly without subprocess management. (2) It's the most documented and researched photogrammetry engine available. (3) Its reconstruction quality competes with commercial tools like RealityCapture. We designed the pipeline with an abstraction layer so the engine is swappable — but COLMAP is the right starting point.

### What makes v1.0 the MVP?

The MVP is the shortest path to a complete workflow: photos in, game-ready mesh out. That means:
- Phase 1 proves the pipeline works end-to-end (photos → mesh)
- Phase 2 makes the output actually usable (decimation, UVs, textures, PBR)
- Phase 3 adds a differentiating feature (live LiDAR capture)
- Phase 4 packages it for distribution

Each phase delivers standalone value. You can ship after Phase 2 and have a useful tool. Phase 3 is what makes it stand out.

## Architecture Reasoning

### Pipeline Architecture

The app follows a strict sequential pipeline: each stage takes input from the previous stage and produces output for the next. This was chosen over a more flexible graph-based architecture because:

1. Photogrammetry is inherently sequential — you can't do dense reconstruction without sparse reconstruction first.
2. Sequential stages are easy to log, monitor, and resume if interrupted.
3. Each stage maps cleanly to a UI progress step, giving users clear feedback.

### Monorepo Structure

The project uses a monorepo with `desktop/`, `mobile/`, and `shared-protocols/` directories. This keeps the desktop app, iOS companion, and their shared streaming protocol definitions in one repository. It simplifies version coordination between the phone and desktop, and makes the full scope of the project visible in a single place — important for a portfolio piece.

### Centralized Theming

All visual styling is defined in a single `styles.py` file using Qt Style Sheets (QSS). This mirrors the CSS-in-one-file approach used in design systems. It means changing the color scheme (which we did — from blue to charcoal/crimson) is a one-file operation, not a hunt through every widget.

## Challenges and Solutions

### Challenge: Qt doesn't have "live reload"

**Problem:** Web development has hot module replacement. Qt apps have to be manually restarted to see changes, which slows down UI iteration.

**Solution:** Built a custom dev runner using the `watchfiles` library (Rust-based file watcher) that monitors the source directory and automatically kills and restarts the app process on any file change. This gives a web-dev-like feedback loop without the complexity of a hot-reload system.

### Challenge: COLMAP doesn't support GPU on macOS

**Problem:** Apple Silicon uses Metal, not CUDA. COLMAP's GPU acceleration requires CUDA, so it falls back to CPU on Mac.

**Solution:** Designed the pipeline with an engine-agnostic abstraction layer (`ReconstructionEngine` ABC) so the photogrammetry engine is swappable. This paid off immediately — when Apple Object Capture was integrated in Phase 1e, it slotted in as a second engine behind the same interface. macOS now uses Metal-accelerated reconstruction via Object Capture (producing ~50K vertices with PBR textures), while COLMAP remains the engine for Windows with CUDA. The engine factory auto-detects the best available engine at startup — no user configuration needed.

### Challenge: iPhone photos aren't really JPEG

**Problem:** Users drag-and-drop photos from their iPhone into the app. The files have `.JPEG` extensions, but macOS silently preserves the original HEIC format. COLMAP cannot read HEIC — it silently fails to extract features from every image, producing an empty database and a misleading "not enough visual overlap" error.

**Solution:** Added a Pillow-based image conversion layer in the ingest stage. Every image passes through `PIL.Image.open()` → `convert("RGB")` → `save(JPEG)` before entering the workspace. The `pillow-heif` library registers HEIC support with Pillow, so the conversion is transparent regardless of the source format. This also future-proofs the ingest against other unusual formats (16-bit PNG, TIFF variants, etc.).

**Lesson:** Never trust file extensions. Always validate the actual format — and when working with Apple's ecosystem, assume HEIC until proven otherwise.

### Challenge: Poisson meshing requires normals that sparse reconstructions don't have

**Problem:** On macOS (no CUDA), dense reconstruction is skipped and the pipeline falls back to sparse point cloud data. COLMAP's sparse PLY export contains only XYZ coordinates and RGB color — no surface normals. Poisson surface reconstruction requires oriented normals to determine inside vs. outside of the surface. Without them, `pycolmap.poisson_meshing()` silently produces a 0-byte output file.

**Solution:** Replaced pycolmap's Poisson meshing with Open3D's implementation, which provides automatic normal estimation. The pipeline now: (1) loads the sparse PLY, (2) estimates normals via PCA on k-nearest neighbors, (3) orients normals consistently using tangent plane propagation, (4) runs Poisson reconstruction, and (5) trims low-density vertices to remove phantom surface extensions. This produces a usable mesh (~5,800 vertices, ~11,500 triangles from a 17,500-point sparse cloud).

**Lesson:** CPU-only fallback paths need end-to-end testing with real data. The sparse-to-mesh path was architecturally correct but broke at the data format level — a gap that only showed up when running the full pipeline with actual iPhone photos.

### Challenge: PLY is not a universal format

**Problem:** The pipeline's output is a `.ply` file — a format that COLMAP and Open3D use natively, but that most game engines and 3D modeling tools don't import directly. Unity expects `.fbx` or `.glTF`. Blender prefers `.obj` or `.glTF`. The pipeline was technically complete but the output was stranded in a format only pipeline tools could read.

**Solution:** Added trimesh as an export layer. trimesh loads the PLY mesh (vertices, faces, vertex colors) and writes it out as `.obj` or `.glb` (glTF Binary). The Export view presents mesh stats (vertex count, triangle count, file size) so the user can verify the output before committing to an export. A native save dialog lets the user choose the destination and format.

**Lesson:** "End-to-end" means end-to-end. A pipeline that produces output in an intermediate format isn't complete — it needs to deliver files in the formats the user's downstream tools actually consume.

### Challenge: PyMeshLab is GPL — a commercial distribution blocker

**Problem:** PyMeshLab (used for QEM decimation in Phase 2a) is licensed under GPL v3. GPL is a copyleft license: any software that uses a GPL library must itself be distributed under GPL. This is fine for open-source tools, but incompatible with the goal of eventually monetizing Meshine Shop. Identifying the licensing status of every dependency before building a commercial product is a requirement, not an afterthought.

**Solution:** Swapped PyMeshLab out for Open3D (MIT licensed), which was already in the dependency stack for AO ray casting. Open3D's `mesh.simplify_quadric_decimation(target_faces)` uses the same QEM algorithm as PyMeshLab — the quality is identical and the API is cleaner. UV coordinates are not affected by the swap because UVs are generated after decimation (by xatlas in Phase 2b), not before. The removal dropped the only GPL dependency from the project; the entire stack is now MIT/BSD/Apache/LGPL, which is compatible with commercial distribution.

**Lesson:** Dependency licensing is not a legal formality — it's a build constraint. GPL in any runtime dependency closes the commercial distribution path. Audit the full dependency tree against your intended licensing model before committing to a library, especially for libraries that do heavy lifting (mesh processing, image codecs, ML frameworks).

### Challenge: pyassimp can't find the Assimp library on macOS

**Problem:** Phase 2e required FBX export. trimesh has no native FBX exporter — calling `mesh.export("*.fbx")` raises a `ValueError`. The intended solution was `pyassimp`, Python bindings for the Assimp format conversion library. `brew install assimp` was run, `pyassimp` was added via Poetry, and `pyassimp.export(scene, path, 'fbx')` was called. Every attempt raised `AssimpError: assimp library not found`.

The root cause: `pyassimp`'s library discovery code reads `additional_dirs` at module import time — hardcoded to `/usr/lib/`, `/usr/local/lib/`, and `LD_LIBRARY_PATH`. Homebrew on Apple Silicon installs to `/opt/homebrew/lib/`, which is not on this list and is not automatically added to `LD_LIBRARY_PATH`. The Python package couldn't load its own C backend even though the system library was correctly installed.

**Solution:** Abandoned `pyassimp` entirely and replaced it with a `subprocess` call to the Assimp CLI tool. `brew install assimp` installs both the shared library and the `assimp` CLI binary at `/opt/homebrew/bin/assimp`. The FBX conversion pipeline became: trimesh → temp OBJ → `assimp export <in> <out>` → FBX binary → delete temp. `shutil.which("assimp")` locates the binary on PATH, and a `subprocess.run()` call handles the conversion. `pyassimp` was removed from `pyproject.toml` entirely.

**Lesson:** Python bindings for system C/C++ libraries often have fragile dynamic library discovery. On macOS with Homebrew, assume the default search paths will not include `/opt/homebrew/lib/` unless you explicitly configure them. When a library ships a CLI tool alongside its Python bindings, the CLI is often the more reliable integration point — it handles its own library loading, doesn't require Python ctypes path manipulation, and produces the same output.

### Challenge: Apple's PhotogrammetrySession is Swift-only

**Problem:** Apple's Object Capture API (`PhotogrammetrySession` in RealityKit) produces dramatically better results than COLMAP on macOS — ~50K vertices with PBR textures vs. ~5,800 sparse vertices. But the API is Swift-only. It cannot be accessed via PyObjC (which only bridges Objective-C), and there is no Python binding. The entire desktop app is Python + PySide6, so calling this API directly from the application was impossible.

**Solution:** Built a lightweight Swift CLI tool (`apple_photogrammetry/`) that wraps `PhotogrammetrySession` and communicates with the Python app via a JSON lines protocol over stdout. The CLI accepts an input directory and output path, runs the reconstruction, and streams structured progress updates (`{"stage": "...", "progress": 0.75}`) that the Python worker parses in real time. This keeps the integration clean — the Swift code is a thin wrapper around Apple's API, and the Python code treats it as just another subprocess with a well-defined contract.

**Lesson:** When a platform API locks you into a specific language, a subprocess bridge with a structured protocol is often cleaner than FFI or binding generators. The JSON lines protocol is trivial to parse, easy to debug, and naturally decouples the two codebases.

### Challenge: Swift CLI process hangs after completion

**Problem:** After the Apple Object Capture pipeline finished processing, the Swift CLI subprocess would not exit. The Python `stdout` reader would hang indefinitely at the "Extracting Features" stage, even though all stages had completed. The pipeline appeared frozen.

**Solution:** The bug was a subtle Swift language trap. Inside a `for try await output in session.outputs` loop, the `.processingComplete` case used a bare `break` statement. In Swift, `break` inside a `switch` only exits the switch — it does not break out of the enclosing `for` loop. The loop continued waiting for more output that would never come. The fix was a labeled loop: `outputLoop: for try await output in session.outputs { ... break outputLoop }`. This explicitly breaks the `for` loop when processing completes, allowing the CLI to exit cleanly.

**Lesson:** In Swift, `break` inside a `switch` is a no-op that only exits the switch. To break out of an enclosing loop from inside a switch case, you need a labeled loop. This is a well-known gotcha, but it produces no compiler warning — the code looks correct and compiles fine.

### Challenge: Object Capture outputs USDZ, not PLY

**Problem:** Apple Object Capture produces a `.usdz` file — a compressed USD archive containing geometry, PBR textures (diffuse, normal, roughness, AO), and material definitions. The existing export pipeline expected `.ply` input. USDZ is not a format that trimesh, Open3D, or any of the Python mesh libraries can read natively. The high-quality reconstruction was stranded in a format the rest of the pipeline couldn't consume.

**Solution:** Added Pixar's `usd-core` Python library to extract geometry from the USDZ archive. The conversion pipeline opens the USD stage, finds the mesh prim, reads the vertex positions and face indices, and writes them out as a standard PLY file that the existing export flow can handle. This approach was chosen over Apple's ModelIO framework (which hung indefinitely in testing) because `usd-core` is the canonical USD implementation and provides reliable, headless operation.

**Lesson:** Format conversion between proprietary and open formats is often the hardest part of integrating platform-specific APIs. The reconstruction itself was easy — Apple's API is excellent. Getting the output into a format the rest of the pipeline could use required a separate library and a custom conversion step.

### Challenge: PyMeshLab's API is not PyPI-documented

**Problem:** PyMeshLab wraps MeshLab's filter system — a C++ application where filters are discovered at runtime and called by name string. There is no autocomplete, no type hints, and the Python docs are sparse. Finding the correct filter name (`meshing_decimation_quadric_edge_collapse`) and its parameter names (`targetfacenum`, `preservenormal`, `preservetopology`, `optimalplacement`) required cross-referencing the MeshLab C++ source and community forum posts. An incorrect filter name raises a generic runtime error with no suggestion of what went wrong.

**Solution:** Treated PyMeshLab like an undocumented runtime API and verified each call against MeshLab's filter documentation directly. The key insight is that `ms.apply_filter("filter_name", **params)` takes keyword arguments that map exactly to the filter's XML-defined parameter names — once you find those names in the source, the API is predictable. The working call was documented thoroughly in the codebase so future filter additions don't require the same research.

**Lesson:** When wrapping a C++ application's plugin system in Python, expect a discovery gap. Plan for time to dig through source code or community resources rather than relying on the Python wrapper's documentation alone.

### Challenge: xatlas splits mesh vertices at UV seams

**Problem:** UV unwrapping requires cutting the 3D mesh surface along seam edges to flatten it into 2D. xatlas performs these cuts, which introduces additional vertices — a single vertex in the input mesh may become two or more vertices in the UV-mapped output (one per UV island boundary). The output vertex count is always >= the input. This was initially confusing: after running xatlas, the mesh appeared to have more vertices than after decimation, seemingly undoing optimization work.

**Solution:** Documented and exposed the seam-split count in the pipeline progress message (`+N seam splits`). The vertex increase is expected behavior — it's geometrically equivalent to the original mesh, just with UV coordinates attached. The new vertex positions are correctly reconstructed using xatlas's `vmapping` output array: `new_positions = vertices[vmapping]` maps each output vertex back to its original input vertex. Setting `process=False` on the trimesh constructor is required to prevent trimesh from re-merging duplicate vertices, which would corrupt the UV layout.

**Lesson:** xatlas's output format (`vmapping`, `new_indices`, `uvs`) is a remapping, not a new mesh from scratch. Understanding that `vmapping[i]` gives the original vertex index for output vertex `i` makes the reconstruction straightforward — but the API documentation doesn't make this obvious.

### Challenge: Apple's USDZ archive has a non-obvious directory structure

**Problem:** Apple Object Capture packages its reconstruction output as a USDZ file — a ZIP archive containing a `.usdc` geometry file and PNG texture images. The `mesh_reconstruct` stage extracts this archive to `workspace/mesh/usdz_extracted/`. During Phase 2c texture baking, `bake_albedo_from_usdz()` searched for the diffuse texture using `glob("*.png")` at the root of the extracted directory. This found nothing — because Object Capture places texture images in a numbered subdirectory inside the archive (`usdz_extracted/0/baked_mesh_*_tex0.png`), not at the archive root. The `.usdc` geometry file sits at the root, but the textures are one level deeper.

**Solution:** Changed all texture searches in `bake_albedo_from_usdz()` from `glob()` to `rglob()` so the search recurses into subdirectories. Also added `"tex"` to the diffuse texture name pattern list — Apple names its diffuse map `*_tex0.png`, while the other PBR maps use `_ao0`, `_norm0`, `_roughness0` suffixes. Adding a non-diffuse exclusion filter (`_ao`, `_norm`, `_rough`, `_metal`, `_disp`) ensures the diffuse map is selected correctly even when the `rglob` finds all five texture types.

**Lesson:** Archive extraction preserves internal directory structure. `usdz_extracted/` is not flat — it mirrors the archive's internal layout. Never assume extracted archives place all files at the top level; always use recursive search or inspect the archive structure first.

### Challenge: OBJ format requires multiple files but users expected one

**Problem:** After implementing Phase 2c texture baking, the OBJ export path correctly bundled the geometry, MTL material file, and three texture PNGs. But users were surprised to see five separate files appear in their destination folder instead of a single file. OBJ format is fundamentally multi-file — the geometry references an `.mtl` file which in turn references texture image paths — but this is not obvious if you're coming from GLB (which is self-contained binary).

**Solution:** Changed the OBJ export path to create a named bundle folder (`dest_path.parent/dest_path.stem/`) and write all five files inside it: `mesh.obj`, `mesh.mtl`, `albedo.png`, `normal.png`, `ao.png`. The `export_mesh()` function now returns the actual output path — the bundle folder for OBJ, the file itself for GLB — so the success message can show the user exactly where to find their export. The Export view's success message now reads "Exported to: ~/Desktop/mesh/" (the folder) rather than pointing to a specific file that may contain a relative MTL reference.

**Lesson:** File format choices have UX implications that aren't immediately obvious from the technical spec. When a format requires multiple files to function, the application should manage that complexity for the user rather than exposing it as scattered files in their destination folder.

*More challenges will be documented as development progresses through Phase 3.*

## Results and Impact

*This section will be updated as the project reaches functional milestones.*

### Phase 1a Complete
- Fully scaffolded desktop application with professional UI
- Working development workflow with auto-restart
- Project architecture designed for 14-week build cycle
- All code comprehensively commented for maintainability

### Phase 1b Complete
- End-to-end photogrammetry pipeline: 65 iPhone photos → 3D mesh in one click
- COLMAP integration via pycolmap with automatic HEIC image conversion
- Background QThread processing with live stage-by-stage UI updates
- Graceful CPU-only fallback: skips dense reconstruction, estimates normals, meshes from sparse data
- Pipeline produces usable output: ~5,800 vertices, ~11,500 triangles from 65 source photos
- Reset button allows re-running without restarting the app

### Phase 1c + 1d Complete
- Stage-by-stage UI feedback already delivered as part of 1b's processing queue
- Mesh export to OBJ and glTF Binary via trimesh — the pipeline now delivers files that Unity, Unreal, and Blender can import directly
- Export view shows mesh stats before export, with format selection and native save dialog
- Auto-transition from pipeline completion to Export view for a seamless workflow
- Full reset flow: clears all three views (Import, Process, Export) back to clean state

### Phase 1e Complete — Apple Object Capture Integration
- Metal-accelerated reconstruction on macOS via RealityKit's PhotogrammetrySession API
- Output quality leap: ~50K vertices, ~100K triangles with PBR textures (vs. ~5,800 sparse vertices from COLMAP on Mac)
- Engine-agnostic architecture validated — Apple Object Capture slots in behind the same `ReconstructionEngine` interface as COLMAP
- Swift CLI subprocess bridge with JSON lines protocol for real-time progress streaming
- USDZ-to-PLY format conversion via Pixar USD library for pipeline compatibility
- Engine factory auto-detects the best engine at startup — no user configuration needed

### Phase 1f Complete — UI Refinements
- Folder import: drag-and-drop entire directories with recursive image scanning
- Browse Folder button as an alternative to drag-and-drop for native directory picker
- Clear button to reset file selection before processing
- Unified outlined button style (charcoal + crimson accent) across all three views
- Consistent layout: all views share the same header + content pattern
- Reset button moved from Process to Export page for logical workflow placement
- Phase 1 goal achieved: photos in → standard 3D mesh out, end-to-end in one application

### Phase 2a Complete — Mesh Decimation
- PyMeshLab quadric edge collapse decimation with three quality presets: Mobile (5K), PC (25K), Cinematic (100K triangles)
- Preset selected on the Import page before processing — no post-processing decision required
- Decimation runs as pipeline stage 7, taking the reconstructed mesh and reducing it to the target polygon budget
- Both COLMAP and Apple Object Capture engines share the same decimation implementation (engine-agnostic)
- Mesh stats in the Export view reflect the decimated count, confirming the target was hit
- Unlocks practical game-engine import: raw reconstruction meshes (200K–1M+ triangles) are too heavy for real-time rendering; decimated meshes are game-ready

### Phase 2b Complete — Automatic UV Unwrapping
- xatlas automatic UV atlas generation added as pipeline stage 8 (after decimation)
- Non-overlapping UV islands packed into [0,1]² space — the standard UV layout expected by game engines and texture bakers
- Handles vertex splitting at UV seams automatically: where xatlas cuts the mesh surface to unfold it, new vertices are introduced at seam edges. This is normal UV behavior and is tracked in the pipeline progress message
- Output saved as `meshed_uv.obj` in the workspace — OBJ format chosen because it natively stores UV coordinates as `vt` lines in the face definitions
- Export view now sources from the UV-mapped OBJ instead of the raw PLY, so UV coordinates are present in all exported files (OBJ and glTF Binary)
- Establishes the foundation for Phase 2c: the UV map defines how photo color data will be projected onto the mesh surface during texture baking

### Phase 2c Complete — PBR Texture Baking
- Three PBR texture maps baked onto the UV-mapped decimated mesh as pipeline stage 9
- **Albedo**: color source differs by engine — Apple path extracts the diffuse PNG from the USDZ archive and transfers colors via Open3D KDTreeFlann nearest-neighbor lookup from original mesh vertices to decimated vertices; COLMAP path samples the colored dense or sparse point cloud. Both rasterize to a 2048×2048 RGB texture via UV-space barycentric scan-fill
- **Normal map**: tangent-space normals computed from the mesh's own vertex normals — TBN matrix built per vertex from UV edge gradients (accumulated per face, Gram-Schmidt orthogonalized), world-space normals transformed to tangent space, encoded as RGB PNG
- **AO**: Open3D `RaycastingScene` hemisphere ray casting — 64 cosine-weighted rays per vertex, all submitted in one batch call, `AO = 1 - mean(hits > 0)`. Runs on the C++ BVH with SIMD acceleration
- All three maps written to `workspace/textures/albedo.png`, `normal.png`, `ao.png`
- Export view shows which PBR maps are present before the user exports (Textures: Albedo, Normal, AO)
- GLB exports embed all three textures as a `PBRMaterial` (self-contained, no external files); OBJ exports bundle the mesh + MTL + textures into a named subfolder
- Non-fatal design: if baking fails at any step, the pipeline continues with untextured geometry rather than failing the entire job

### Phase 2d Complete — PBR Material Estimation (Roughness + Metallic)
- Two additional PBR maps derived from image-space analysis of the baked albedo — no additional geometry passes required
- **Roughness**: `specularity = Value × (1 − Saturation)` in HSV space — bright, desaturated pixels are specular/glossy (low roughness); saturated or dark pixels are rough. Blended with AO crevice factor: crevices get a +15% roughness boost. Clamped to [0.2, 1.0] to prevent unphysical perfect-mirror values
- **Metallic**: `score = (1 − S) × V` with a soft threshold at 0.7 (ramp to 1.0 at 0.9) — highly desaturated, high-value pixels are treated as metallic. Conservative threshold avoids false positives on light-colored diffuse surfaces; most surfaces will correctly produce near-zero metallic values
- Both maps saved as grayscale 2048×2048 PNGs to `workspace/textures/roughness.png` and `metallic.png`
- GLB: packed into a single `metallicRoughnessTexture` per the glTF 2.0 specification (G channel = roughness, B channel = metallic)
- OBJ: `map_Pr` (roughness) and `map_Pm` (metallic) directives added to the MTL file — supported by Blender, Substance Painter, and most current game engine importers
- Export view now shows all five maps: "Textures: Albedo, Normal, AO, Roughness, Metallic"

### Phase 2e Complete — FBX Export
- FBX added as a third export format alongside OBJ and GLB
- Conversion pipeline: trimesh exports the UV-mapped mesh to a temporary OBJ → the Assimp CLI (`assimp export <in> <out>`) converts it to FBX binary → temp files deleted
- Bundle folder output matches OBJ convention: `mesh/mesh.fbx` + all five texture PNGs in the same folder. Unreal Engine, Maya, and 3ds Max all expect FBX assets delivered this way — geometry in the FBX, textures assigned manually via the DCC tool's material editor
- Requires `brew install assimp` (macOS) — the CLI tool is the integration point rather than the `pyassimp` Python bindings, which have fragile dynamic library discovery on macOS (see challenge above)
- Graceful failure: if the `assimp` binary is not on PATH, a clear error dialog appears with installation instructions rather than a silent failure or crash

## What I'd Improve

*This section will be updated as development reveals areas for iteration.*

- **Image conversion overhead:** Converting every image through Pillow adds processing time at ingest. A smarter approach would detect the actual format first (via file magic bytes) and only convert when necessary.
- **COLMAP on macOS is still sparse-only:** Apple Object Capture solved the macOS quality problem, but COLMAP remains the only option on Windows. Ensuring COLMAP's CUDA path is thoroughly tested on Windows hardware is a priority before Phase 3.
- **Heuristic PBR maps vs. extracted maps:** Phase 2d derives roughness and metallic via image-space HSV analysis — fast and effective for v1.0. Apple Object Capture's USDZ output includes actual roughness and metallic PBR maps (`*_roughness0.png`, `*_metalness0.png`). Extracting these directly from the USDZ archive (as is already done for albedo in Phase 2c) would give ground-truth PBR values on macOS rather than estimates. This is a clear v1.1 target.
- **Performance optimization for large datasets:** COLMAP sequential matching, Apple Object Capture detail level control, and image downscaling at ingest are all deferred to post-v1.0.

## What This Proves

- **Systems architecture:** Choosing the right tools for each layer of a complex application — not defaulting to the most popular option, but the most effective one for the domain.
- **Pipeline design:** Breaking a multi-stage computational workflow into discrete, observable, resumable steps — the same pattern used in data engineering, CI/CD, and game asset pipelines.
- **Cross-language integration:** Bridging Python, Swift, and C++ (via COLMAP) across a single application using clean subprocess protocols and abstract interfaces — without resorting to fragile FFI or binding generators.
- **Full-stack thinking:** From low-level GPU compute (COLMAP, Metal via Object Capture) to high-level UI/UX (Qt theming, drag-and-drop interaction) to developer experience (file watcher, Poetry, monorepo structure).
- **Scope management:** Phased delivery with standalone value at each milestone — not trying to build everything at once, but shipping incrementally while maintaining a coherent architecture.
- **Technical writing:** Every module is documented with what it does and why it exists. The codebase is designed to be read, not just executed.
