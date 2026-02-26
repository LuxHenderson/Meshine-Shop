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

Quality preset → Apple Object Capture detail level mapping:
    Mobile  (15K target)  → "reduced" — ~25K polys before decimation,
                            fast processing suitable for mobile iterations.
    PC      (65K target)  → "full" — ~250K polys, game-ready quality.
    Cinematic (200K target) → "full" — ~250K polys; we decimate only lightly.
                            "raw" (30M polys) is intentionally avoided because
                            the processing time would be impractical.
"""

from meshine_shop.core.engine import ReconstructionEngine, ColmapEngine


def create_best_engine(
    quality_preset: str = "PC (65K triangles)",
) -> tuple[ReconstructionEngine, str]:
    """
    Detect the best available engine and return it with a display name.

    Args:
        quality_preset: The user's selected quality preset label (from
                        QUALITY_PRESETS in pipeline.py). Used to select
                        the Apple Object Capture detail level so the raw
                        reconstruction polygon count matches the intended
                        output tier.

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
    # Map quality preset label to Apple Object Capture detail level.
    # "Mobile" uses "reduced" for faster processing — Apple still produces
    # ~25K polys which we decimate to 15K. PC and Cinematic use "full"
    # to get the highest quality source (~250K polys) before decimation.
    apple_detail = "reduced" if "Mobile" in quality_preset else "full"

    # Map quality preset to texture baking resolution.
    # Texture detail (texels per triangle) scales with polygon budget:
    #   Mobile   (15K tri)  → 1024×1024  — small assets, fast baking
    #   PC       (65K tri)  → 2048×2048  — standard game asset quality
    #   Cinematic (200K tri) → 4096×4096 — 4× more texels for fine skin/surface detail
    if "Mobile" in quality_preset:
        texture_size = 1024
    elif "Cinematic" in quality_preset:
        texture_size = 4096
    else:
        texture_size = 2048  # PC preset — default

    # Try Apple Object Capture first (macOS only).
    # Import lazily to avoid errors on Windows where the module
    # references macOS-specific paths.
    try:
        from meshine_shop.core.apple_engine import (
            is_apple_object_capture_available,
            AppleObjectCaptureEngine,
        )

        if is_apple_object_capture_available():
            engine = AppleObjectCaptureEngine(
                detail=apple_detail, texture_size=texture_size
            )
            return engine, f"Apple Object Capture ({apple_detail})"
    except Exception:
        # If anything goes wrong during detection (missing binary,
        # import error, etc.), fall through to COLMAP silently.
        pass

    # Fall back to COLMAP — always available as the baseline engine.
    return ColmapEngine(texture_size=texture_size), "COLMAP"
