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
| Photogrammetry Engine | COLMAP (via pycolmap) | Best-documented open-source SfM/MVS engine. Python API avoids subprocess overhead |
| Mesh Processing | PyMeshLab, Open3D | Decimation, UV unwrapping, point cloud manipulation — all native Python |
| UV Unwrapping | xatlas | Battle-tested automatic UV generation used in production pipelines |
| AI/ML | PyTorch | PBR material estimation (roughness/metallic maps) via local inference |
| 3D Viewport | Qt OpenGL widget + Open3D | GPU-accelerated rendering without browser overhead |
| Export | trimesh, pygltflib | Native Python .glTF and .FBX writing |
| Mobile Companion | Swift + ARKit (iOS) | LiDAR live capture streaming to desktop (Phase 3) |
| Streaming Protocol | WebSocket or gRPC | Real-time depth/RGB/pose data from phone to desktop |
| Dependency Management | Poetry | Clean dependency resolution, virtual environments, and lock files |
| Packaging | PyInstaller | Cross-platform desktop installers (macOS + Windows) |

## Architecture Overview

Meshine Shop follows a **pipeline architecture** — data flows sequentially through discrete, logged stages:

```
Photos/LiDAR → Ingest → Feature Extraction → Sparse Cloud (SfM) → Dense Cloud → Mesh → Texture → Export
```

The application is structured as a **monorepo** with three top-level directories:

```
Meshine Shop/
├── desktop/              # Python + PySide6 desktop application
│   ├── meshine_shop/     # Main application package
│   │   ├── core/         # Pipeline logic, COLMAP integration
│   │   └── ui/           # Qt widgets, views, styling
│   └── scripts/          # Development tooling (file watcher)
├── mobile/               # iOS companion app (Phase 3)
└── shared-protocols/     # Streaming protocol definitions (Phase 3)
```

### Key Design Decisions

**Python + PySide6 over Electron:** The entire computational backbone — COLMAP, PyTorch, PyMeshLab, Open3D — lives in Python. Electron would require constant IPC bridging between Node.js and Python, adding latency and failure points. PySide6 keeps everything in one process, one language, one ecosystem.

**COLMAP over OpenMVG + OpenMVS:** COLMAP provides a Python API (pycolmap), eliminating subprocess overhead. It's the most documented and widely used open-source photogrammetry engine. The pipeline architecture includes an abstraction layer so alternative engines can be added later without refactoring.

**QStackedWidget for view management:** Views are swapped instantly via a stacked widget rather than recreated on each navigation event. This gives zero-cost view transitions and allows each view to maintain its state across switches.

**Centralized QSS theming:** All visual styling lives in a single `styles.py` file rather than being scattered across widgets. This makes theme changes trivial and keeps UI code focused on layout and behavior.

## Current Features (Phase 1 — Core Pipeline Complete)

- Desktop application shell with charcoal + crimson dark theme
- Sidebar navigation (Import / Process / Export views)
- Drag-and-drop file import zone accepting JPEG, PNG, TIFF, HEIC, and PLY files
- Automatic HEIC-to-JPEG conversion for iPhone photos (via Pillow + pillow-heif)
- Full COLMAP photogrammetry pipeline: ingest → feature extraction → sparse reconstruction → dense reconstruction → mesh → texture mapping
- Background processing via QThread — UI stays responsive during pipeline execution
- Live processing queue with per-stage status indicators (pending / running / done / error)
- Real-time progress messages in the queue and status bar
- Poisson surface reconstruction via Open3D with automatic normal estimation
- Dense reconstruction support on CUDA-enabled systems; graceful fallback to sparse meshing on CPU-only (macOS)
- Reset button to cancel/clear the pipeline and start a new job — resets all views to clean state
- Mesh export to OBJ (.obj) and glTF Binary (.glb) via trimesh
- Export view with mesh stats (vertex count, triangle count, file size), format selector, and native save dialog
- Auto-transition to Export view after pipeline completes
- Development file watcher with auto-restart on save
- Cross-platform targeting (macOS + Windows)

## Roadmap

### Phase 1: Core Pipeline (Complete)
- [x] 1a — Project scaffolding and app shell
- [x] 1b — COLMAP photogrammetry engine integration
- [x] 1c — Processing pipeline with stage-by-stage UI feedback
- [x] 1d — Basic mesh export (.OBJ / .glTF)

### Phase 2: Game-Ready Optimization
- [ ] 2a — Mesh decimation with quality presets
- [ ] 2b — Automatic UV unwrapping (xatlas)
- [ ] 2c — Texture baking (albedo, normals, AO)
- [ ] 2d — AI-driven PBR material estimation
- [ ] 2e — .FBX and .glTF export with material definitions

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
- COLMAP (photogrammetry engine)

### macOS Install

```bash
# Install Python 3.11, Poetry, and COLMAP via Homebrew
brew install python@3.11
brew install poetry
brew install colmap
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

- **No GPU acceleration for COLMAP on macOS:** Apple Silicon uses Metal, not CUDA. Dense reconstruction (patch-match stereo) is skipped on macOS; meshing proceeds from the sparse point cloud. Windows with NVIDIA GPUs get full CUDA acceleration.
- **Sparse-only meshing on macOS:** Without dense reconstruction, the Poisson mesh is built from the sparse point cloud (~17k points for 65 images). Quality is lower than dense-based meshing but the pipeline completes end-to-end.
- **iPhone photos are HEIC internally:** Even when named `.JPEG`, iPhone photos are HEIC format. The app handles this automatically via Pillow + pillow-heif conversion, but users should be aware of the extra processing step.
- **macOS-only testing so far:** Cross-platform support is architected in but Windows testing has not started.

## Future Improvements

- GPU-accelerated reconstruction via Metal compute shaders (macOS) or CUDA (Windows)
- Plugin system for custom pipeline stages
- Cloud processing offload for large datasets
- Android companion app for LiDAR capture
- Real-time collaborative scanning (multiple phones contributing to one reconstruction)
