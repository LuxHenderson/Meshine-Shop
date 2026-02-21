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
- **No budget for commercial tools.** The entire stack is built on open-source technology — COLMAP, PyMeshLab, xatlas, PyTorch — to eliminate licensing costs and ensure the tool can be freely distributed.
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

*More challenges will be documented as development progresses through Phases 2–4.*

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

## What I'd Improve

*This section will be updated as development reveals areas for iteration.*

- **Initial consideration:** Evaluate whether a hybrid approach (Rust core + Python bindings) would yield better performance for the compute-heavy pipeline stages, while maintaining the Python UI layer.
- **Image conversion overhead:** Converting every image through Pillow adds processing time at ingest. A smarter approach would detect the actual format first (via file magic bytes) and only convert when necessary.
- **Texture carry-over from USDZ:** Object Capture produces PBR textures (diffuse, normal, roughness, AO) inside its USDZ output, but the current PLY conversion extracts geometry only. Carrying those textures through to the final export would deliver fully textured models without the Phase 2 texture baking step.
- **COLMAP on macOS is still sparse-only:** Apple Object Capture solved the macOS quality problem, but COLMAP remains the only option on Windows. Ensuring COLMAP's CUDA path is thoroughly tested on Windows hardware is a priority before Phase 2.

## What This Proves

- **Systems architecture:** Choosing the right tools for each layer of a complex application — not defaulting to the most popular option, but the most effective one for the domain.
- **Pipeline design:** Breaking a multi-stage computational workflow into discrete, observable, resumable steps — the same pattern used in data engineering, CI/CD, and game asset pipelines.
- **Cross-language integration:** Bridging Python, Swift, and C++ (via COLMAP) across a single application using clean subprocess protocols and abstract interfaces — without resorting to fragile FFI or binding generators.
- **Full-stack thinking:** From low-level GPU compute (COLMAP, Metal via Object Capture) to high-level UI/UX (Qt theming, drag-and-drop interaction) to developer experience (file watcher, Poetry, monorepo structure).
- **Scope management:** Phased delivery with standalone value at each milestone — not trying to build everything at once, but shipping incrementally while maintaining a coherent architecture.
- **Technical writing:** Every module is documented with what it does and why it exists. The codebase is designed to be read, not just executed.
