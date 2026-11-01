// Record one app's windows + that app's audio to an .mp4 with ScreenCaptureKit (macOS 15+).
// Stops when the app process exits, or on SIGINT/SIGTERM. No loopback audio driver needed.
//
//   swiftc -O tools/screenrec.swift -o ~/.cache/zelda-rl/screenrec
//   ~/.cache/zelda-rl/screenrec --pid <PID> --out video.mp4 [--fps 60] [--max-width 1920]
//
// Needs Screen Recording permission for the terminal app that launches it (System Settings > Privacy & Security).
//
// SEGMENTS, and why. macOS stops a capture stream on its own when the display configuration changes --
// which is exactly what Dolphin does when emulation starts and it goes fullscreen:
//
//     Error Domain=com.apple.ScreenCaptureKit.SCStreamErrorDomain Code=-3821 "Stream was stopped by the system"
//
// Measured 2026-11-01: a 61.3 s winning Gohma episode produced 18.6 s of video that did not contain the
// kill, and the old code printed "screenrec: saved <file>" over it and exited 0. So a stream that dies
// while the target process is still alive is now RESTARTED into the next segment, and the exit status
// tells the caller whether the recording is whole:
//
//     0  the recording ended because the process exited or we were signalled (complete)
//     1  usage / no window / permission (nothing recorded)
//     3  the stream died and could not be restarted (the file is SHORT -- do not trust it)
//
// Segments are written as <out-stem>.partNN.mp4 and listed on stdout as "screenrec: segment <path>".
// Concatenating them is the caller's job (scripts/record_wins.py does it with ffmpeg).
import AppKit
import AVFoundation
import Foundation
import ScreenCaptureKit

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

func note(_ msg: String) { print("screenrec: \(msg)"); fflush(stdout) }

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

/// Per-segment state. `failed` means the SYSTEM stopped us; `done` means the file was finalized.
final class Recorder: NSObject, SCStreamDelegate, SCRecordingOutputDelegate {
    private let lock = NSLock()
    private var _done = false
    private var _failed = false
    var done: Bool { lock.lock(); defer { lock.unlock() }; return _done }
    var failed: Bool { lock.lock(); defer { lock.unlock() }; return _failed }
    func reset() { lock.lock(); _done = false; _failed = false; lock.unlock() }
    private func finish(failed: Bool) {
        lock.lock(); _done = true; _failed = _failed || failed; lock.unlock()
    }
    func recordingOutputDidStartRecording(_ recordingOutput: SCRecordingOutput) { note("recording started") }
    func recordingOutputDidFinishRecording(_ recordingOutput: SCRecordingOutput) { finish(failed: false) }
    func recordingOutput(_ recordingOutput: SCRecordingOutput, didFailWithError error: Error) {
        FileHandle.standardError.write("screenrec: recording failed: \(error)\n".data(using: .utf8)!)
        finish(failed: true)
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("screenrec: stream stopped: \(error)\n".data(using: .utf8)!)
        finish(failed: true)
    }
}

final class Flag {
    private let lock = NSLock()
    private var v = false
    var value: Bool { lock.lock(); defer { lock.unlock() }; return v }
    func set() { lock.lock(); v = true; lock.unlock() }
}

guard let pidStr = arg("--pid"), let pid = Int32(pidStr), let out = arg("--out") else {
    fail("usage: screenrec --pid PID --out file.mp4 [--fps 60] [--max-width 1920]")
}
let fps = Int32(arg("--fps") ?? "60") ?? 60
let maxWidth = CGFloat(Double(arg("--max-width") ?? "1920") ?? 1920)
let recorder = Recorder()
let stopRequested = Flag()
let outURL = URL(fileURLWithPath: out)
let stem = outURL.deletingPathExtension().path

func segmentPath(_ n: Int) -> String { String(format: "%@.part%02d.mp4", stem, n) }

// ScreenCaptureKit adds a menu-bar recording indicator, which needs a running AppKit application (no Dock icon).
let application = NSApplication.shared
application.setActivationPolicy(.accessory)

signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM] {
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { stopRequested.set() }
    src.resume()
    signalSources.append(src)
}

/// Build a fresh content filter for `pid`. Re-queried per segment so a fullscreen switch, a moved window
/// or a re-created window is picked up instead of being captured from a stale description.
func currentFilter(waitSeconds: Int) async throws -> (SCContentFilter, SCRunningApplication, SCDisplay)? {
    for _ in 0..<(waitSeconds * 2) {
        // onScreenWindowsOnly MUST be false. Dolphin's fullscreen window sits on its own Space and drops
        // out of the on-screen listing whenever that Space is not frontmost, which ends the capture with
        // Code=-3815 "Failed to find any displays or windows to capture" partway through the episode:
        // measured 2026-11-01, four Deku episodes in a row kept 1.6-10.3 s of 14.6-19.2 s episodes.
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        let window = content.windows
            .filter { $0.owningApplication?.processID == pid && $0.frame.width >= 320 && $0.frame.height >= 240 }
            .max { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }
        if let window, let app = window.owningApplication {
            let display = content.displays.first { $0.frame.intersects(window.frame) } ?? content.displays[0]
            // WINDOW-scoped, not display-scoped. Dolphin goes fullscreen when emulation starts and macOS
            // puts a fullscreen window on its OWN Space, so a capture of the display records the empty
            // desktop behind it: measured 2026-11-01, a 21 s Deku clip and a 2.8 s Gohma clip that both
            // passed a duration check were solid BLACK for their whole length. A desktop-independent
            // window filter follows the window across Spaces and crops to the game, with no letterboxing.
            return (SCContentFilter(desktopIndependentWindow: window), app, display)
        }
        if kill(pid, 0) != 0 { return nil }
        try await Task.sleep(nanoseconds: 500_000_000)
    }
    return nil
}

Task {
    var segment = 0
    var wrote: [String] = []
    do {
        while kill(pid, 0) == 0 && !stopRequested.value {
            guard let (filter, app, display) = try await currentFilter(waitSeconds: segment == 0 ? 120 : 10) else {
                break                                   // the process went away while we looked for its window
            }
            let path = segmentPath(segment)
            let cfg = SCStreamConfiguration()
            let scale = min(CGFloat(filter.pointPixelScale), maxWidth / filter.contentRect.width)
            cfg.width = Int(filter.contentRect.width * scale) & ~1
            cfg.height = Int(filter.contentRect.height * scale) & ~1
            cfg.minimumFrameInterval = CMTime(value: 1, timescale: fps)
            cfg.showsCursor = false
            cfg.capturesAudio = true              // audio from the apps in the filter only
            cfg.excludesCurrentProcessAudio = true
            cfg.sampleRate = 48000
            cfg.channelCount = 2
            cfg.queueDepth = 8

            recorder.reset()
            let s = SCStream(filter: filter, configuration: cfg, delegate: recorder)
            let outCfg = SCRecordingOutputConfiguration()
            outCfg.outputURL = URL(fileURLWithPath: path)
            outCfg.outputFileType = .mp4
            outCfg.videoCodecType = .h264
            try s.addRecordingOutput(SCRecordingOutput(configuration: outCfg, delegate: recorder))
            try await s.startCapture()
            note("capturing \(app.applicationName) (pid \(pid)) on display \(display.displayID) at "
                 + "\(cfg.width)x\(cfg.height), \(fps) fps + audio -> \(path)")
            wrote.append(path)

            while kill(pid, 0) == 0 && !stopRequested.value && !recorder.failed {
                try await Task.sleep(nanoseconds: 250_000_000)
            }
            let systemKilledIt = recorder.failed
            try? await s.stopCapture()
            for _ in 0..<150 where !recorder.done {     // up to 15 s for the file to be finalized
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            note("segment \(path)")
            segment += 1
            if !systemKilledIt { break }                // normal end: process exited or we were signalled
            note("the system stopped the stream; restarting into the next segment")
            try await Task.sleep(nanoseconds: 500_000_000)
        }
    } catch {
        let m = "screenrec: \(error) (grant Screen Recording to your terminal in System Settings > "
              + "Privacy & Security, then restart it)\n"
        FileHandle.standardError.write(m.data(using: .utf8)!)
        exit(wrote.isEmpty ? 1 : 3)
    }
    if wrote.isEmpty { fail("screenrec: no window for pid \(pid); nothing recorded") }
    // Still alive and not asked to stop means the loop fell out on an error it could not recover.
    let incomplete = kill(pid, 0) == 0 && !stopRequested.value
    note("segments=\(wrote.count) \(incomplete ? "INCOMPLETE" : "complete")")
    exit(incomplete ? 3 : 0)
}
application.run()
