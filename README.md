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
| Photogrammetry (macOS) | Apple Object Capture (RealityKit) | Metal-accelerated, ~50K vertices at full detail with PBR textures. Accessed via Swift CLI subprocess |
| Photogrammetry (Windows) | COLMAP (via pycolmap) | CUDA-accelerated dense reconstruction. Cross-platform fallback on all systems |
| Swift CLI Bridge | Swift Package Manager | Wraps Apple's PhotogrammetrySession API — a Swift-only API that cannot be accessed via PyObjC |
| USD Processing | usd-core (Pixar USD) | Reads Object Capture's USDZ output (geometry + PBR textures) for format conversion |
| Mesh Processing | Open3D | Decimation (QEM), point cloud manipulation, AO ray casting — MIT licensed, fully commercial-friendly |
| UV Unwrapping | xatlas | Battle-tested automatic UV generation used in production pipelines |
| 3D Viewport | Qt OpenGL widget + Open3D | GPU-accelerated rendering without browser overhead |
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
| macOS + Apple Silicon | Apple Object Capture | Metal GPU + Neural Engine | ~50K vertices, PBR textures |
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
│   │   │   └── texture_baker.py # PBR texture baking (albedo, normal, AO)
│   │   └── ui/                 # Qt widgets, views, styling
│   │       ├── main_window.py  # Import, Process, Export views
│   │       ├── drop_zone.py    # Drag-and-drop + folder import
│   │       ├── processing_queue.py # Live pipeline stage display
│   │       ├── sidebar.py      # Navigation sidebar
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

## Current Features (Phase 1 + Phase 2a–2e Complete)

### Reconstruction Engines
- **Apple Object Capture** (macOS): Metal-accelerated reconstruction producing ~50K vertices with PBR textures (diffuse, normal, roughness, AO) via RealityKit's PhotogrammetrySession API
- **COLMAP** (cross-platform): Full SfM/MVS pipeline with CUDA acceleration on Windows; CPU fallback on all systems
- Engine factory auto-detects the best engine at startup — no user configuration needed

### Import
- Drag-and-drop import accepting individual files or entire folders of photos
- Browse Folder button for native directory picker as an alternative to drag-and-drop
- Recursive folder scanning — automatically finds all supported images in nested directories
- Clear button to reset file selection before processing
- Quality preset selector (Mobile / PC / Cinematic) to choose target polygon budget before processing
- Supported formats: JPEG, PNG, TIFF, HEIC (automatic HEIC-to-JPEG conversion for iPhone photos)

### Processing
- Full 9-stage photogrammetry pipeline: ingest → feature extraction → sparse reconstruction → dense reconstruction → mesh → texture mapping → decimation → UV unwrapping → texture baking
- Background processing via QThread — UI stays responsive during pipeline execution
- Live processing queue with per-stage status indicators (pending / running / done / error)
- Real-time progress messages in the queue and status bar
- USDZ-to-PLY format conversion via Pixar USD library (for Object Capture output)
- **Mesh decimation** (Phase 2a): Open3D quadric error metrics (QEM) decimation to Mobile (5K), PC (25K), or Cinematic (100K) triangle targets — MIT licensed, production-safe
- **UV unwrapping** (Phase 2b): xatlas automatic UV atlas generation — non-overlapping islands packed into [0,1]² space; output is OBJ with embedded UV coordinates
- **Texture baking** (Phase 2c): three PBR maps baked onto the UV-mapped mesh — albedo (diffuse color from USDZ textures or COLMAP point cloud), tangent-space normal map (from mesh vertex normals), and ambient occlusion (hemisphere ray casting via Open3D). All three saved as 2048×2048 PNGs to workspace/textures/
- **PBR material estimation** (Phase 2d): roughness and metallic maps derived from image-space HSV analysis of the baked albedo. Roughness: `specularity = V × (1 − S)`, blended with AO crevice factor. Metallic: soft threshold on desaturated high-value pixels. Both maps added to workspace/textures/ and included in all exports

### Export
- Mesh export to OBJ (.obj), glTF Binary (.glb), and FBX (.fbx)
- Source mesh is the UV-mapped OBJ (Phase 2b output) — UV coordinates preserved in all exports
- **Full PBR export** (Phase 2c–2d): all five maps (albedo, normal, AO, roughness, metallic) included when present
  - **GLB**: all textures embedded as a self-contained PBR material per the glTF 2.0 spec — roughness and metallic packed into a single `metallicRoughnessTexture` (G=roughness, B=metallic)
  - **OBJ**: bundle folder with `.obj`, `.mtl` (using `map_Pr`/`map_Pm` PBR extension directives), and all five PNG maps
  - **FBX** (Phase 2e): bundle folder with `mesh.fbx` + all five PNG maps — textures delivered externally per Unreal Engine and Maya convention. Requires `assimp` CLI on PATH (`brew install assimp`)
- Export view shows mesh stats and which PBR maps are available before exporting
- Format selector dropdown with native save dialog
- Auto-transition to Export view after pipeline completes
- Reset button to clear the job and start a new reconstruction

### UI/UX
- Charcoal + crimson dark theme with cohesive outlined button styling
- Sidebar navigation with crimson left-accent active indicator and vertical separator
- Import / Process / Export views — all content vertically and horizontally centered
- Quality preset dropdown on Import page; format dropdown on Export page
- Development file watcher with auto-restart on save
- Cross-platform targeting (macOS + Windows)

## Roadmap

### Phase 1: Core Pipeline (Complete)
- [x] 1a — Project scaffolding and app shell
- [x] 1b — COLMAP photogrammetry engine integration
- [x] 1c — Processing pipeline with stage-by-stage UI feedback
- [x] 1d — Basic mesh export (.OBJ / .glTF)
- [x] 1e — Apple Object Capture integration (macOS high-quality reconstruction)
- [x] 1f — UI refinements (folder import, browse button, consistent styling)

### Phase 2: Game-Ready Optimization
- [x] 2a — Mesh decimation with quality presets
- [x] 2b — Automatic UV unwrapping (xatlas)
- [x] 2c — Texture baking (albedo, normals, AO)
- [x] 2d — PBR material estimation (roughness + metallic from image-space HSV analysis)
- [x] 2e — FBX export with texture bundle (via Assimp CLI)

### Phase 3: LiDAR Live Capture
- [ ] 3a — iOS companion app (ARKit + LiDAR)
- [ ] 3b — WebSocket/gRPC streaming protocol
- [ ] 3c — Real-time 3D point cloud preview
- [ ] 3d — Capture-to-pipeline handoff

### Phase 4: Polish & Packaging
- [ ] Desktop installers (macOS + Windows)
- [ ] Quality presets (Mobile / PC / Cinematic)
- [ ] Batch processing for multiple assets
- [ ] First-run tutorial and documentation

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
- **macOS-only testing so far:** Cross-platform support is architected in but Windows testing has not started.

## Future Improvements

- Plugin system for custom pipeline stages
- Cloud processing offload for large datasets
- Android companion app for LiDAR capture
- Real-time collaborative scanning (multiple phones contributing to one reconstruction)
- Performance optimization: Apple Object Capture detail level control, COLMAP sequential/vocab-tree matching for large datasets, image preprocessing (auto-resize inputs)
