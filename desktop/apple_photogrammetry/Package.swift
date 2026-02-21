// swift-tools-version: 5.9

// Swift Package Manager manifest for the Apple Object Capture CLI tool.
//
// This package builds a lightweight command-line executable that wraps
// Apple's PhotogrammetrySession API (RealityKit Object Capture). The
// Meshine Shop Python app calls this binary as a subprocess and reads
// JSON progress lines from stdout to drive the UI.
//
// Requires macOS 12+ (Monterey) — the minimum version that includes
// the Object Capture API in RealityKit.

import PackageDescription

let package = Package(
    name: "apple-photogrammetry",
    // macOS 12 is the minimum for PhotogrammetrySession.
    // We target .v14 (Sonoma) to access the latest improvements
    // while remaining compatible with most Apple Silicon Macs.
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "apple-photogrammetry",
            path: "Sources"
        )
    ]
)
