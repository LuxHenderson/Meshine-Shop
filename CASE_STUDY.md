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

**Solution:** Accepted CPU-mode for macOS development (still fast enough for small-to-medium datasets). The architecture is engine-agnostic, so Metal-accelerated alternatives can be integrated later. Windows users with NVIDIA GPUs get full CUDA acceleration.

*More challenges will be documented as development progresses through Phases 1b–4.*

## Results and Impact

*This section will be updated as the project reaches functional milestones.*

### Phase 1a Complete
- Fully scaffolded desktop application with professional UI
- Working development workflow with auto-restart
- Project architecture designed for 14-week build cycle
- All code comprehensively commented for maintainability

## What I'd Improve

*This section will be updated as development reveals areas for iteration.*

- **Initial consideration:** Evaluate whether a hybrid approach (Rust core + Python bindings) would yield better performance for the compute-heavy pipeline stages, while maintaining the Python UI layer.

## What This Proves

- **Systems architecture:** Choosing the right tools for each layer of a complex application — not defaulting to the most popular option, but the most effective one for the domain.
- **Pipeline design:** Breaking a multi-stage computational workflow into discrete, observable, resumable steps — the same pattern used in data engineering, CI/CD, and game asset pipelines.
- **Full-stack thinking:** From low-level GPU compute (COLMAP, PyTorch) to high-level UI/UX (Qt theming, drag-and-drop interaction) to developer experience (file watcher, Poetry, monorepo structure).
- **Scope management:** Phased delivery with standalone value at each milestone — not trying to build everything at once, but shipping incrementally while maintaining a coherent architecture.
- **Technical writing:** Every module is documented with what it does and why it exists. The codebase is designed to be read, not just executed.
