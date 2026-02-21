/// Apple Object Capture CLI — Photogrammetry subprocess for Meshine Shop.
///
/// This command-line tool wraps Apple's PhotogrammetrySession API (RealityKit)
/// and outputs structured JSON lines to stdout so the Python desktop app can
/// parse progress in real time.
///
/// The CLI produces a USDZ file (Apple's native 3D format) which contains
/// the mesh geometry (.usdc) and PBR textures (roughness, AO, normal, diffuse,
/// displacement). The Python app handles conversion to PLY/OBJ using the
/// Pixar USD Python library (usd-core).
///
/// Usage:
///   apple-photogrammetry <input-dir> <output-dir> [--detail <level>]
///   apple-photogrammetry --check-support
///
/// Arguments:
///   <input-dir>    Directory containing source photographs (JPEG, HEIC, PNG)
///   <output-dir>   Directory where object_capture_output.usdz will be written
///   --detail       Quality level: preview, reduced, medium, full, raw
///                  Default: full (~250K polygons with PBR textures)
///   --check-support  Print {"supported": true/false} and exit. Used by the
///                    Python app at startup to determine engine availability.
///
/// Output protocol (JSON lines to stdout):
///   {"event": "started"}
///   {"event": "progress", "fraction": 0.42}
///   {"event": "complete", "output": "/path/to/object_capture_output.usdz"}
///   {"event": "error", "message": "Human-readable error description"}
///
/// Exit codes:
///   0 = success
///   1 = error (details in JSON output)
///   2 = invalid arguments

import Foundation
import RealityKit

// MARK: - JSON Output Helpers

/// Print a single JSON line to stdout and flush immediately.
/// The Python app reads these line-by-line for real-time progress updates.
func emit(_ dict: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: dict),
       let json = String(data: data, encoding: .utf8) {
        print(json)
        // Force flush so the Python subprocess reader gets lines immediately
        // rather than waiting for the output buffer to fill.
        fflush(stdout)
    }
}

/// Emit a progress event with a fraction between 0.0 and 1.0.
func emitProgress(_ fraction: Double) {
    emit(["event": "progress", "fraction": fraction])
}

/// Emit a fatal error and exit with code 1.
func emitErrorAndExit(_ message: String) -> Never {
    emit(["event": "error", "message": message])
    exit(1)
}

// MARK: - Support Check

/// Check whether this machine supports Apple Object Capture.
/// Requires macOS 12+ and compatible hardware (Apple Silicon preferred,
/// Intel with AMD GPU + 16GB RAM also works).
func checkSupport() {
    if #available(macOS 12.0, *) {
        let supported = PhotogrammetrySession.isSupported
        emit(["supported": supported])
    } else {
        emit(["supported": false])
    }
    exit(0)
}

// MARK: - Detail Level Parsing

/// Convert a CLI string argument into a PhotogrammetrySession detail level.
/// Returns nil for unrecognized strings so the caller can show a usage error.
@available(macOS 12.0, *)
func parseDetail(_ string: String) -> PhotogrammetrySession.Request.Detail? {
    switch string.lowercased() {
    case "preview":  return .preview
    case "reduced":  return .reduced
    case "medium":   return .medium
    case "full":     return .full
    case "raw":      return .raw
    default:         return nil
    }
}

// MARK: - Main Entry Point

/// Parse command-line arguments and run the photogrammetry session.
/// This function never returns — it calls exit() when done.
@available(macOS 12.0, *)
func runPhotogrammetry() async {
    let args = CommandLine.arguments

    // Handle --check-support flag (can appear anywhere in args).
    if args.contains("--check-support") {
        checkSupport()
    }

    // Validate minimum arguments: program name + input dir + output dir.
    guard args.count >= 3 else {
        fputs("""
        Usage: apple-photogrammetry <input-dir> <output-dir> [--detail <level>]
               apple-photogrammetry --check-support

        Detail levels: preview, reduced, medium, full (default), raw
        """, stderr)
        exit(2)
    }

    let inputDir = URL(fileURLWithPath: args[1], isDirectory: true)
    let outputDir = URL(fileURLWithPath: args[2], isDirectory: true)

    // The intermediate USDZ file — Object Capture's native output format.
    // This is guaranteed to work with PhotogrammetrySession.
    let usdzFile = outputDir.appendingPathComponent("object_capture_output.usdz")

    // Parse optional --detail flag. Default to "full" which produces
    // ~250K polygons with PBR textures — the best balance of quality
    // and processing time for game assets.
    var detail: PhotogrammetrySession.Request.Detail = .full
    if let detailIndex = args.firstIndex(of: "--detail"),
       detailIndex + 1 < args.count {
        guard let parsed = parseDetail(args[detailIndex + 1]) else {
            emitErrorAndExit("Invalid detail level: \(args[detailIndex + 1]). Use: preview, reduced, medium, full, raw")
        }
        detail = parsed
    }

    // Verify the input directory exists.
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: inputDir.path) else {
        emitErrorAndExit("Input directory does not exist: \(inputDir.path)")
    }

    // Ensure the output directory exists.
    if !fileManager.fileExists(atPath: outputDir.path) {
        do {
            try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)
        } catch {
            emitErrorAndExit("Failed to create output directory: \(error.localizedDescription)")
        }
    }

    // Create the PhotogrammetrySession with the input image directory.
    // The session scans the directory for supported image formats automatically.
    let session: PhotogrammetrySession
    do {
        // Configure the session. We use the default configuration which
        // enables all automatic optimizations (masking, ordering, etc.).
        var config = PhotogrammetrySession.Configuration()
        // Let Object Capture handle object masking automatically.
        config.isObjectMaskingEnabled = true
        session = try PhotogrammetrySession(input: inputDir, configuration: config)
    } catch {
        emitErrorAndExit("Failed to create photogrammetry session: \(error.localizedDescription)")
    }

    emit(["event": "started"])

    // Track whether we received a completion event.
    var completedSuccessfully = false

    // Listen for session output on a background task.
    // PhotogrammetrySession.Outputs is an AsyncSequence that emits
    // progress updates and completion/error events.
    // Use a labeled loop so we can break out of the `for` from inside
    // the `switch`. A bare `break` inside a switch only exits the switch,
    // not the enclosing for loop — this was causing the process to hang
    // after completion because the output listener never terminated.
    let outputTask = Task {
        do {
            outputLoop: for try await output in session.outputs {
                switch output {
                case .requestProgress(_, let fractionComplete):
                    // fractionComplete is 0.0 to 1.0 for the current request.
                    emitProgress(fractionComplete)

                case .requestComplete(_, _):
                    // The USDZ model file has been written.
                    completedSuccessfully = true

                case .requestError(_, let requestError):
                    emitErrorAndExit("Photogrammetry request failed: \(requestError.localizedDescription)")

                case .processingComplete:
                    // All requests are done. Break the labeled for loop
                    // so the task completes and the process can exit.
                    break outputLoop

                case .processingCancelled:
                    emitErrorAndExit("Photogrammetry session was cancelled")

                case .invalidSample(let id, let reason):
                    // Non-fatal — just a warning about a single image.
                    // Don't emit as error; the session continues without it.
                    fputs("Warning: Invalid sample \(id): \(reason)\n", stderr)

                case .skippedSample(let id):
                    // Non-fatal — session skipped an image it couldn't use.
                    fputs("Warning: Skipped sample \(id)\n", stderr)

                case .automaticDownsampling:
                    // Session is downsampling images to fit memory constraints.
                    fputs("Info: Automatic downsampling enabled\n", stderr)

                case .requestProgressInfo(_, _):
                    // Extended progress info available in newer macOS versions.
                    // We already handle .requestProgress for the fraction.
                    break

                case .stitchingIncomplete:
                    // Some images couldn't be stitched into the reconstruction.
                    // Non-fatal — the session continues with what it has.
                    fputs("Warning: Some images could not be stitched\n", stderr)

                case .inputComplete:
                    // All input images have been read. Processing continues.
                    break

                @unknown default:
                    // Future API additions — ignore gracefully.
                    break
                }
            }
        } catch {
            emitErrorAndExit("Session output error: \(error.localizedDescription)")
        }
    }

    // Submit the reconstruction request. Always output USDZ — Apple's native
    // format that is guaranteed to work with PhotogrammetrySession.
    do {
        try session.process(requests: [
            .modelFile(url: usdzFile, detail: detail)
        ])
    } catch {
        emitErrorAndExit("Failed to submit photogrammetry request: \(error.localizedDescription)")
    }

    // Wait for the output listener to complete.
    await outputTask.value

    // Verify the USDZ was created.
    guard completedSuccessfully, fileManager.fileExists(atPath: usdzFile.path) else {
        emitErrorAndExit("Object Capture completed but no USDZ file was created.")
    }

    // Report the USDZ path back to the Python app. The Python engine
    // handles conversion from USDZ to PLY using the Pixar USD library
    // (usd-core), which is more reliable than ModelIO for large meshes.
    emit([
        "event": "complete",
        "output": usdzFile.path
    ])

    exit(0)
}

// MARK: - Entry Point Dispatch

// Handle --check-support without the @available guard since it has
// its own availability check internally.
if CommandLine.arguments.contains("--check-support") {
    checkSupport()
}

// The main photogrammetry function requires macOS 12+.
if #available(macOS 12.0, *) {
    // Run the async entry point. We use a semaphore to keep the process
    // alive while the async task runs, since Swift CLI tools exit when
    // the main function returns.
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        await runPhotogrammetry()
        semaphore.signal()
    }
    semaphore.wait()
} else {
    emitErrorAndExit("Apple Object Capture requires macOS 12.0 or later. Current system is not supported.")
}
