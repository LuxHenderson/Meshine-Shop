"""
Engine selection factory for Meshine Shop.

This module decides which photogrammetry engine to use based on the
current platform and available hardware. The goal is to always pick
the highest-quality engine that will run on the user's machine:

    macOS + Apple Silicon  → Apple Object Capture (Metal-accelerated)
    macOS + Intel          → COLMAP (CPU-only fallback)
    Windows + NVIDIA GPU   → COLMAP (CUDA-accelerated dense reconstruction)
    Windows + no GPU       → COLMAP (CPU-only fallback)

The factory is called once per pipeline run in app.py. The returned
engine implements ReconstructionEngine, so the worker, processing queue,
and export flow don't need to know which engine is active — they all
share the same interface.
"""

from meshine_shop.core.engine import ReconstructionEngine, ColmapEngine


def create_best_engine() -> tuple[ReconstructionEngine, str]:
    """
    Detect the best available engine and return it with a display name.

    Returns:
        A tuple of (engine_instance, engine_display_name).
        The display name is shown in the status bar so the user knows
        which engine is processing their data.

    The detection order is:
    1. Apple Object Capture (macOS only, requires compatible hardware)
    2. COLMAP (cross-platform fallback, always available)

    Apple Object Capture is preferred on macOS because it produces
    dramatically higher quality output (~250K polygons at 'full' detail)
    compared to COLMAP's CPU-only sparse path (~5K polygons). On Windows,
    COLMAP with CUDA is the primary engine and produces comparable quality
    via dense reconstruction.
    """
    # Try Apple Object Capture first (macOS only).
    # Import lazily to avoid errors on Windows where the module
    # references macOS-specific paths.
    try:
        from meshine_shop.core.apple_engine import (
            is_apple_object_capture_available,
            AppleObjectCaptureEngine,
        )

        if is_apple_object_capture_available():
            engine = AppleObjectCaptureEngine(detail="full")
            return engine, "Apple Object Capture"
    except Exception:
        # If anything goes wrong during detection (missing binary,
        # import error, etc.), fall through to COLMAP silently.
        pass

    # Fall back to COLMAP — always available as the baseline engine.
    return ColmapEngine(), "COLMAP"
