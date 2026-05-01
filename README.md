# Meshine Shop

A standalone desktop application that converts photogrammetry data into game-ready 3D assets.

## Problem Statement

Game developers and 3D artists need to convert real-world objects into usable game assets. The current workflow involves chaining together multiple disconnected tools — capture software, reconstruction engines, mesh optimizers, UV unwrappers, texture bakers — each with its own interface, file formats, and learning curve. There is no single, streamlined tool that takes raw photogrammetry data in and delivers a game-ready asset out.

Meshine Shop solves this by unifying the entire pipeline — from photo import (or live LiDAR capture) through reconstruction, optimization, and export — into one cohesive desktop application.

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11 | Direct access to the ML/CV/3D ecosystem without bridging |
| Desktop Framework | PySide6 (Qt) | Industry-standard for professional 3D tools (Blender, Maya, Substance use Qt). Native OpenGL support for the 3D viewport |
| Photogrammetry (macOS) | Apple Object Capture (RealityKit) | Metal-accelerated, ~250K vertices at full detail with PBR textures. Accessed via Swift CLI subprocess |
| Photogrammetry (Windows) | COLMAP (via pycolmap) | CUDA-accelerated dense reconstruction. Cross-platform fallback on all systems |
| Swift CLI Bridge | Swift Package Manager | Wraps Apple's PhotogrammetrySession API — a Swift-only API that cannot be accessed via PyObjC |
| USD Processing | usd-core (Pixar USD) | Reads Object Capture's USDZ output (geometry + PBR textures) for format conversion |
| Mesh Processing | Open3D | Decimation (QEM), point cloud manipulation, AO ray casting — MIT licensed, fully commercial-friendly |
| UV Unwrapping | xatlas | Battle-tested automatic UV generation used in production pipelines |
| 3D Viewport | moderngl (OpenGL 4.1 core) + PySide6 QOpenGLWidget | GPU-accelerated render loop with full GL context control and custom GLSL shaders |
| Export | trimesh + Assimp CLI | OBJ/GLB via trimesh; FBX via the Assimp CLI (OBJ → FBX conversion) |
| Mobile Companion | Swift + ARKit (iOS) | LiDAR live capture streaming to desktop (Phase 3) |
| Streaming Protocol | WebSocket or gRPC | Real-time depth/RGB/pose data from phone to desktop |
| Dependency Management | Poetry | Clean dependency resolution, virtual environments, and lock files |
| Packaging | PyInstaller | Cross-platform desktop installers (macOS + Windows) |

## Architecture Overview

Meshine Shop follows a **pipeline architecture** — data flows sequentially through discrete, logged stages:

```
Photos/LiDAR → Ingest → Feature Extraction → Sparse Cloud (SfM) → Dense Cloud → Mesh → Texture → Decimation → UV Unwrap → Texture Baking → Export
```

The application uses an **engine-agnostic design** — a `ReconstructionEngine` abstract base class defines the pipeline interface, and an engine factory auto-detects the best available engine at startup:

| Platform | Engine | Acceleration | Output Quality |
|---|---|---|---|
| macOS + Apple Silicon | Apple Object Capture | Metal GPU + Neural Engine | ~250K vertices, PBR textures |
| Windows + NVIDIA GPU | COLMAP | CUDA | Dense reconstruction, high quality |
| Any (fallback) | COLMAP | CPU | Sparse reconstruction, lower quality |

Both engines implement the same 9-stage interface, so the worker, processing queue, and export flow are completely engine-agnostic — they don't know or care which engine is running.

The application is structured as a **monorepo** with three top-level directories:

```
Meshine Shop/
├── desktop/                    # Python + PySide6 desktop application
│   ├── meshine_shop/           # Main application package
│   │   ├── core/               # Pipeline logic, engine implementations
│   │   │   ├── engine.py       # ReconstructionEngine ABC + COLMAP engine
│   │   │   ├── apple_engine.py # Apple Object Capture engine
│   │   │   ├── engine_factory.py # Auto-detects best engine per platform
│   │   │   ├── worker.py       # QThread background pipeline runner
│   │   │   ├── pipeline.py     # Stage definitions and constants
│   │   │   ├── workspace.py    # Workspace directory management
│   │   │   ├── exporter.py     # Mesh format conversion + texture embedding
│   │   │   ├── texture_baker.py # PBR texture baking (albedo, normal, AO)
│   │   │   ├── edit_history.py # Undo/redo command stack (UndoStack + ICommand ABC)
│   │   │   ├── mesh_painter.py # BVH, sculpt brushes, paint buffer, projection bake
│   │   │   └── viewport_camera.py # Camera math (view/projection matrices, UE-style navigation)
│   │   └── ui/                 # Qt widgets, views, styling
│   │       ├── main_window.py  # Import, Process, Viewport, Export views
│   │       ├── drop_zone.py    # Drag-and-drop + folder import
│   │       ├── processing_queue.py # Live pipeline stage display
│   │       ├── sidebar.py      # Navigation sidebar
│   │       ├── viewport.py     # QOpenGLWidget — moderngl render loop, projection shaders, overlays
│   │       ├── viewport_tools.py # Tool panel (brush, sculpt, polygon select, settings)
│   │       ├── viewport_layers.py # Layers panel (saved selections, visibility, rename, delete)
│   │       └── styles.py       # Centralized QSS dark theme
│   ├── apple_photogrammetry/   # Swift CLI wrapping PhotogrammetrySession
│   │   ├── Package.swift       # SPM manifest (macOS 14+, RealityKit)
│   │   └── Sources/main.swift  # CLI: JSON progress protocol over stdout
│   └── scripts/                # Development tooling (file watcher)
├── mobile/                     # iOS companion app (Phase 3)
└── shared-protocols/           # Streaming protocol definitions (Phase 3)
```

### Key Design Decisions

**Python + PySide6 over Electron:** The entire computational backbone — COLMAP, Open3D, trimesh — lives in Python. Electron would require constant IPC bridging between Node.js and Python, adding latency and failure points. PySide6 keeps everything in one process, one language, one ecosystem.

**Engine-agnostic pipeline:** The `ReconstructionEngine` ABC lets us swap photogrammetry engines without touching the UI, worker, or export code. Apple Object Capture on macOS and COLMAP on Windows both implement the same 8-stage interface. The engine factory picks the best one automatically at startup.

**Swift CLI subprocess for Object Capture:** Apple's `PhotogrammetrySession` is a Swift-only API — it cannot be bridged via PyObjC (which only handles Objective-C). A lightweight Swift CLI tool wraps the API and communicates with the Python app via JSON lines on stdout. This keeps the integration clean and the Python app self-contained.

**QStackedWidget for view management:** Views are swapped instantly via a stacked widget rather than recreated on each navigation event. This gives zero-cost view transitions and allows each view to maintain its state across switches.

**Centralized QSS theming:** All visual styling lives in a single `styles.py` file rather than being scattered across widgets. This makes theme changes trivial and keeps UI code focused on layout and behavior.

## v1.0 Features

### Reconstruction Engines
- **Apple Object Capture** (macOS): Metal-accelerated reconstruction producing ~250K vertices at `full` detail with PBR textures (diffuse, normal, roughness, AO, metallic) via RealityKit's PhotogrammetrySession API. Quality preset controls detail level: Mobile → `reduced`, PC/Cinematic/Ultra → `full`
- **COLMAP** (cross-platform): Full SfM/MVS pipeline with CUDA acceleration on Windows; CPU fallback on all systems
- Engine factory auto-detects the best engine at startup — no user configuration needed

### Import
- Drag-and-drop import accepting individual files or entire folders of photos
- Browse Folder button for native directory picker as an alternative to drag-and-drop
- Recursive folder scanning — automatically finds all supported images in nested directories
- Clear button to reset file selection before processing
- Quality preset selector (Mobile / PC / Cinematic / Ultra) to choose target polygon budget before processing
- Supported formats: JPEG, PNG, TIFF, HEIC (automatic HEIC-to-JPEG conversion for iPhone photos)

### Processing
- Full 9-stage photogrammetry pipeline: ingest → feature extraction → sparse reconstruction → dense reconstruction → mesh → texture mapping → decimation → UV unwrapping → texture baking
- Background processing via QThread — UI stays responsive during pipeline execution
- Live processing queue with per-stage status indicators (pending / running / done / error)
- Real-time progress messages in the queue and status bar
- USDZ-to-PLY format conversion via Pixar USD library (for Object Capture output)
- **Mesh decimation**: Open3D quadric error metrics (QEM) decimation to quality preset triangle targets — MIT licensed, production-safe
- **UV unwrapping**: xatlas automatic UV atlas generation — non-overlapping islands packed into [0,1]² space; output is OBJ with embedded UV coordinates
- **Texture baking**: five PBR maps baked onto the UV-mapped mesh — albedo, normal, AO, roughness, metallic. Saved to workspace/textures/
- **PBR material estimation**: roughness and metallic maps derived from image-space HSV analysis of the baked albedo (COLMAP path). Roughness: `specularity = V × (1 − S)`, blended with AO crevice factor. Metallic: soft threshold on desaturated high-value pixels
- **Mesh geometry improvements**: isolated floating mesh fragments removed before decimation (connected component analysis); Laplacian smoothing pre-decimation reduces reconstruction noise; polygon budgets: Mobile (15K), PC (65K), Cinematic (200K), Ultra (400K)
- **Texel-space PBR baking**: Apple Object Capture path bakes all five USDZ maps (albedo, normal, AO, roughness, metallic) at texel-space resolution using BVH surface proximity + Cramér's-rule barycentric UV interpolation. For each output texel, the nearest surface point on the original 250K-triangle USDZ mesh is found via trimesh ProximityQuery; barycentric coordinates locate that point within its USDZ triangle; the per-face-vertex USDZ UV is interpolated at those weights, then sampled from the original texture. Texture dilation (4px) applied to all maps to prevent UV island seam bleeding
- **Apple Object Capture quality control**: quality preset controls the Object Capture detail level — Mobile uses `reduced` (~25K source polys, faster), PC/Cinematic/Ultra use `full` (~250K source polys, maximum detail)
- **Vertex tangent embedding**: pre-computed MikkTSpace-compatible TANGENT vectors embedded in every GLB with a normal map — required by glTF 2.0 spec for correct normal map decoding across all viewers
- **Preset-scaled texture resolution**: Mobile → 1024×1024, PC → 2048×2048, Cinematic → 4096×4096, Ultra → 8192×8192. Ultra's 8K UV atlas delivers 4× the texel density of Cinematic, enabling sub-millimeter surface detail and skin microstructure
- **Selective PBR correction**: Apple Object Capture's roughness and metallic maps are calibrated for RealityKit — raw values produce chrome/liquid-metal in standard glTF renderers. Correction uses the metallic map as a surface-type mask: organic zones (leather, skin, fabric) get roughness floor 0.60 and metallic clamped to 0; genuine metal zones (scissors, buckles, rivets — metallic > 0.30) are left untouched so chrome parts render as chrome
- **Albedo clarity enhancement**: shadow lift (`shadow_lift=0.22`, `shadow_threshold=0.35`) raises only sub-threshold pixels proportionally — dark leather becomes visible without brightening skin or highlights. 1.5× saturation boost amplifies colour differences in lifted dark areas

### Export
- Mesh export to OBJ (.obj), glTF Binary (.glb), and FBX (.fbx)
- Source mesh is the UV-mapped OBJ — UV coordinates preserved in all exports
- **Full PBR export**: all five maps (albedo, normal, AO, roughness, metallic) included when present
  - **GLB**: all textures embedded as a self-contained PBR material per the glTF 2.0 spec — roughness and metallic packed into a single `metallicRoughnessTexture` (G=roughness, B=metallic). Vertex tangents embedded (TANGENT attribute) so normal maps render correctly in all glTF viewers without runtime approximation
  - **OBJ**: bundle folder with `.obj`, `.mtl` (using `map_Pr`/`map_Pm` PBR extension directives), and all five PNG maps
  - **FBX**: bundle folder with `mesh.fbx` + all five PNG maps — textures delivered externally per Unreal Engine and Maya convention. Requires `assimp` CLI on PATH (`brew install assimp`)
- **Viewport projection baking**: if the user has applied shader-based texture projections in the viewport, those layers are baked onto the albedo atlas before export — the exported GLB/OBJ/FBX carries the painted result, not raw GPU state
- Export view shows mesh stats and which PBR maps are available before exporting
- Format selector dropdown with native save dialog
- Auto-transition to Export view after pipeline completes
- Reset button to clear the job and start a new reconstruction

### 3D Viewport

The viewport sits between the Process and Export pages. The pipeline auto-navigates here after reconstruction completes so the user can inspect and edit the asset before exporting.

**Rendering**
- OpenGL 4.1 core profile via moderngl inside a PySide6 QOpenGLWidget
- Diffuse + ambient GLSL shaders: `texture(albedo, uv).rgb * (diffuse * 0.6 + 0.4)` — photo-realistic lighting from a single directional light without shadow maps
- GPU-uploaded interleaved VBO (position + normal + UV per vertex) with a single draw call per frame
- Dirty-rect texture streaming: only modified regions are re-uploaded via `glTexSubImage2D` after paint operations

**Camera Navigation (Unreal Engine-style)**

| Input | Behaviour |
|---|---|
| RMB + mouse drag | Fly look (yaw/pitch) |
| RMB + WASD | Move forward/back/left/right |
| RMB + Q/E | Move down/up |
| RMB + Shift | 3× speed boost |
| Alt + LMB drag | Orbit around focal point |
| Scroll wheel | Dolly along view direction |
| MMB drag | Pan focal point |
| F key | Frame mesh — resets camera to see full bounding box |

**Sculpt Brushes**
- Inflate, Deflate, Smooth, Flatten — real-time deformation applied via trimesh BVH ray casting
- Seam-safe: edits propagate to duplicate vertices across UV seam boundaries to prevent cracks
- Undo/redo via a full command stack (`UndoStack` + `ICommand` ABC) — Ctrl+Z / Ctrl+Shift+Z

**Polygon Selection + Layers**
- Lasso tool: click to place polygon anchor points (committed instantly on press), Enter or double-click to finalize
- **Pixel-accurate face selection via Shapely intersection**: every mesh vertex projected to screen space via MVP; faces included when their screen-space triangle intersects the drawn polygon (Shapely `intersects()`) — not just faces whose centroid is inside. A numpy bounding-box pre-filter skips non-candidates before Shapely is invoked. Any face with any polygon overlap is included; the GLSL mask clips exterior portions precisely. No FBO, no DPR mismatch
- **Pending overlay**: teal screen-space mask; reprojects from 3D anchor points each frame so the overlay tracks the mesh surface during orbit before saving
- **In-progress anchor tracking**: world-space 3D positions are stored for each anchor at click time via BVH ray cast. On every camera movement — orbit, pan, WASD fly navigation — anchor positions are reprojected through the current camera MVP so the lasso overlay stays glued to the mesh surface. The rubber-band preview line suppresses itself during camera-only movement and reappears naturally on the next mouse move.
- **Committed overlay**: per-frame 3D anchor reprojection — anchor points stored as world-space surface positions, reprojected through the current camera MVP each frame. The highlight stays glued to the model surface from any orbit angle or WASD camera movement without drifting.
- **Layers panel** (left sidebar): each saved polygon selection appears as a named layer row with an eye toggle, color swatch, inline rename (double-click), and delete button. All layers default to crimson; per-layer color is user-adjustable via the swatch
- **Delete Faces — cookie-cutter clipping**: removes geometry with exact polygon-shaped holes rather than whole-face deletion. Uses Shapely `difference()` on screen-space projected triangles; boundary triangles are split exactly along the polygon outline and re-triangulated via constrained Delaunay triangulation. New boundary vertices are created at exact crossing points with 3D positions and UVs interpolated via barycentric coordinates. Supports Undo (Ctrl+Z)
- **GPU depth buffer back-face guard**: face visibility determined by reading back the scene depth texture rendered by the last `paintGL` frame. Each candidate face's NDC centroid depth is compared against the buffer value at that pixel — faces deeper than the rendered surface are occluded and kept, eliminating blowout on the back of the mesh regardless of camera angle or surface curvature

**Shader-Based Texture Projection**
- Textures are projected onto the mesh via a dedicated GLSL second render pass — planar UV projection entirely in world-space, not UV-atlas space
- Projection frame (Right, Up, Normal vectors) passed as shader uniforms; planar UVs computed in the fragment shader at GPU framerate
- Source textures uploaded as trilinear mipmap `LINEAR_MIPMAP_LINEAR` moderngl textures for highest quality sampling at all distances and angles
- **GLSL analytical polygon mask**: the drawn lasso polygon is uploaded as `uniform vec2 poly_verts[64]` and tested per-fragment via a winding-number point-in-polygon algorithm entirely in GLSL. SDF edge distance drives a `smoothstep` 1.5px AA fringe. Zero CPU rasterization, zero texture upload per frame — mathematically exact sub-pixel edges at GPU framerate
- **Back-face discard**: the camera's world-space forward direction at projection time is stored per layer as `proj_forward`. Each frame the shader discards fragments where the screen-space derivative normal (`dFdx`/`dFdy` cross product) no longer agrees with the original projection direction — hiding the projection when the camera orbits behind the mesh
- **Depth test fix**: the projection pass re-draws the same faces as the base pass at equal depth. OpenGL's default `GL_LESS` discards equal-depth fragments (making the projection invisible). Fixed by setting `depth_func = "<="` before the projection pass and restoring `"<"` after
- Each projection layer is independent: separate face-subset IBO, separate texture, separate polygon mask, toggled via the layers panel eye icon
- **Export baking**: at export time, `bake_projections_to_atlas()` in `mesh_painter.py` implements the shader math in numpy — per-face UV triangle rasterization, barycentric interpolation of 3D positions, planar UV computation, bilinear texture sampling, alpha compositing. Runs at 2× supersampling resolution then LANCZOS downsamples to atlas size for anti-aliased results

**Mesh Operations** *(partially implemented — in active development)*
- **Smooth Mesh**: single-pass Taubin smooth per click (alternating shrink/inflate Laplacian, `λ=0.1 / μ=-0.11`) — reduces reconstruction noise without mesh shrinkage. Seam-safe: edits propagate to UV-seam duplicate vertex pairs to prevent cracks. Volume is near-stable across repeated applications. Undoable. *Status: working.*
- **Fill Holes**: closes open boundary loops — useful after Delete Faces or to patch scan coverage gaps. Traces directed boundary half-edges into closed loops, splits figure-8 / bowtie boundaries at pinch-point vertices, then fan-triangulates each loop from its centroid. New patch vertices receive UV (0,0). Undoable. *Status: implemented, in verification.*
- **Remove Floaters**: splits the mesh into connected components and discards any fragment below 100 faces — eliminates background debris and reconstruction artefacts while preserving the main subject. Undoable. *Status: working.*
- **Decimate**: quadric error metric (QEM) polygon reduction via a target-face-count dialog showing the current count and defaulting to 50% reduction. UV coordinates are transferred to the new topology via KD-tree nearest-vertex lookup. Undoable. *Status: working.*

### UI/UX
- Charcoal + crimson dark theme with cohesive outlined button styling
- Top navigation bar with crimson active indicator and horizontal separator
- Import / Process / Viewport / Export views — all content vertically and horizontally centered
- Quality preset dropdown on Import page (Mobile / PC / Cinematic / Ultra); format dropdown on Export page
- Development file watcher with auto-restart on save
- Cross-platform targeting (macOS + Windows)

## Roadmap

### v1.0 — Complete
- [x] Project scaffolding and app shell
- [x] COLMAP photogrammetry engine integration (SfM/MVS, CUDA on Windows, CPU fallback)
- [x] Apple Object Capture integration (Metal-accelerated, ~250K vertices with PBR textures)
- [x] Engine-agnostic pipeline: factory auto-selects best engine per platform
- [x] Full 9-stage processing pipeline with live stage-by-stage UI feedback
- [x] Mesh decimation with quality presets (Mobile 15K / PC 65K / Cinematic 200K / Ultra 400K)
- [x] Automatic UV unwrapping (xatlas — non-overlapping atlas packed into [0,1]²)
- [x] Five-map PBR texture baking: albedo, normal, AO, roughness, metallic
- [x] Texel-space PBR baking from USDZ (BVH surface proximity + Cramér's-rule barycentric UV interpolation — full photographic resolution, zero Voronoi artifacts)
- [x] Preset-scaled texture resolution (Mobile 1024 / PC 2048 / Cinematic 4096 / Ultra 8192)
- [x] Selective PBR correction for Object Capture maps (organic zones corrected, metal zones preserved)
- [x] Albedo clarity enhancement (shadow lift + saturation boost for dark-subject visibility)
- [x] Vertex tangent embedding in GLB (MikkTSpace-compatible TANGENT for spec-compliant normal map rendering)
- [x] Multi-format export: OBJ, glTF Binary (GLB), FBX — all with full PBR texture bundles
- [x] Viewport projection baking: painted layers composited onto UV atlas at export time
- [x] Custom OpenGL 4.1 viewport (moderngl + PySide6 QOpenGLWidget): textured mesh render, diffuse shading, deferred VBO upload
- [x] Unreal Engine-style camera navigation (fly, orbit, pan, zoom, frame)
- [x] Sculpt brushes: inflate, deflate, smooth, flatten — BVH ray cast hit detection, seam-safe deformation
- [x] Undo/redo command stack (ICommand ABC — Ctrl+Z / Ctrl+Shift+Z)
- [x] Polygon lasso selection: Shapely triangle–polygon intersection (pixel-accurate boundary faces), GLSL analytical polygon mask (winding-number, 1.5px AA), 3D anchor reprojection on orbit/fly
- [x] Layer system: save, toggle visibility, rename, delete, per-layer color, crimson default
- [x] Delete Faces: cookie-cutter clipping via Shapely + constrained Delaunay re-triangulation, GPU depth buffer back-face guard
- [x] Shader-based texture projection: GLSL second render pass, planar UV in world space, proj_forward back-face discard, GLSL analytical polygon mask
- [x] Smooth Mesh operation: conservative single-pass Taubin per click, seam-safe, undoable
- [x] Remove Floaters operation: connected component analysis, discards fragments < 100 faces, undoable
- [x] Decimate operation: QEM reduction to user-specified target, UV transfer via KD-tree, undoable

### In Progress — Paused
- 🔧 **Fill Holes**: boundary loop tracer + figure-8 split + fan triangulation implemented — pending full verification against Delete Faces output
- 🔧 **Mesh Operations stability**: Smooth, Remove Floaters, and Decimate are working; overall mesh operations suite needs additional real-world testing before being considered stable
- 🔧 **Depth overlay distance verification**: fix applied (threshold 0.9995 → 0.9999) but not yet verified at medium/far camera distances

### Planned for Future Releases
- **Windows / CUDA validation** — full pipeline tested on an NVIDIA RTX system; COLMAP CUDA path verified end-to-end
- **Full-model texture projection** — project a texture across the entire mesh surface in a single operation, not just within a polygon selection
- **Previous Jobs section** — reopen past reconstruction jobs in the viewport for continued editing; replace auto-trash-on-close with persistent job storage
- **Mesh merge tool** — combine multiple photogrammetry captures of the same subject into a single unified mesh
- **Performance optimization** — COLMAP sequential/vocab-tree matching for large datasets; image downscaling at ingest; BVH proximity bake caching for repeated exports on the same mesh
- **LiDAR live capture** — iOS companion app (ARKit + LiDAR) streaming depth + RGB + pose to desktop via WebSocket/gRPC
- **Viewport settings dialog** — camera sensitivity, keybindings customization
- **Desktop installers** — PyInstaller bundles for macOS and Windows; first-run tutorial
- **Batch processing** — multiple assets processed sequentially in one session

## Setup Instructions

### Prerequisites

- macOS or Windows
- Python 3.11
- Poetry (dependency manager)
- COLMAP (photogrammetry engine — required on Windows, optional on macOS)
- Swift toolchain (macOS only — for building the Object Capture CLI)

### macOS Install

```bash
# Install Python 3.11 and Poetry via Homebrew
brew install python@3.11
brew install poetry

# COLMAP is optional on macOS (Object Capture is preferred)
brew install colmap

# Assimp is required for FBX export
brew install assimp

# Build the Apple Object Capture CLI
cd desktop/apple_photogrammetry
swift build
```

### Project Setup

```bash
# Clone the repository
git clone git@github.com:LuxHenderson/Meshine-Shop.git
cd Meshine-Shop/desktop

# Install dependencies
poetry env use python3.11
poetry install
```

### Running the App

```bash
# Normal launch
cd desktop
poetry run python -m meshine_shop

# Development mode (auto-restart on file changes)
cd desktop
poetry run python scripts/dev.py
```

## Known Limitations

- **COLMAP on macOS is sparse-only:** Apple Silicon uses Metal, not CUDA. Dense reconstruction is skipped; meshing proceeds from the sparse point cloud. On macOS, Apple Object Capture is strongly preferred and automatically selected.
- **iPhone photos are HEIC internally:** Even when named `.JPEG`, iPhone photos are HEIC format. The app handles this automatically via Pillow + pillow-heif conversion.
- **macOS-only testing so far:** Cross-platform support is architected in (engine factory, platform-agnostic pipeline, PySide6 for UI) but Windows has not been tested. CUDA path is in the COLMAP engine but unverified on real hardware.
- **Mesh operations are functional but early:** Smooth, Remove Floaters, and Decimate are working. Fill Holes is implemented and under verification. None have been stress-tested against the full range of reconstruction output topologies.
- **Texture projection back-face bleed-through:** Minor fragmented bleed-through of the projected texture is visible from some rear camera angles. The `proj_forward` shader discard handles most cases; a small amount of bleed-through is accepted for v1.0.
- **Fill Holes UV quality:** Vertices introduced by hole-filling receive UV (0,0) — the patch geometry is untextured. This is acceptable for structural closure (watertight export) but not for textured patches.
