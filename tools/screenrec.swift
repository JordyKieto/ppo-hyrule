// Record one app's windows + that app's audio to an .mp4 with ScreenCaptureKit (macOS 15+).
// Stops when the app process exits, or on SIGINT/SIGTERM. No loopback audio driver needed.
//
//   swiftc -O tools/screenrec.swift -o ~/.cache/zelda-rl/screenrec
//   ~/.cache/zelda-rl/screenrec --pid <PID> --out video.mp4 [--fps 60] [--max-width 1920]
//
// Needs Screen Recording permission for the terminal app that launches it (System Settings > Privacy & Security).
import AppKit
import AVFoundation
import Foundation
import ScreenCaptureKit

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

final class Recorder: NSObject, SCStreamDelegate, SCRecordingOutputDelegate {
    private let lock = NSLock()
    private var _done = false
    var done: Bool { lock.lock(); defer { lock.unlock() }; return _done }
    func finish() { lock.lock(); _done = true; lock.unlock() }
    func recordingOutputDidStartRecording(_ recordingOutput: SCRecordingOutput) { print("screenrec: recording started") }
    func recordingOutputDidFinishRecording(_ recordingOutput: SCRecordingOutput) { finish() }
    func recordingOutput(_ recordingOutput: SCRecordingOutput, didFailWithError error: Error) {
        FileHandle.standardError.write("screenrec: recording failed: \(error)\n".data(using: .utf8)!)
        finish()
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("screenrec: stream stopped: \(error)\n".data(using: .utf8)!)
        finish()
    }
}

guard let pidStr = arg("--pid"), let pid = Int32(pidStr), let out = arg("--out") else {
    fail("usage: screenrec --pid PID --out file.mp4 [--fps 60] [--max-width 1920]")
}
let fps = Int32(arg("--fps") ?? "60") ?? 60
let maxWidth = CGFloat(Double(arg("--max-width") ?? "1920") ?? 1920)
let recorder = Recorder()
var stream: SCStream?

func stop(_ reason: String) async {
    print("screenrec: stopping (\(reason))")
    try? await stream?.stopCapture()
    for _ in 0..<150 where !recorder.done {  // wait up to 15 s for the file to be finalized
        try? await Task.sleep(nanoseconds: 100_000_000)
    }
    print("screenrec: saved \(out)")
    exit(0)
}

// ScreenCaptureKit adds a menu-bar recording indicator, which needs a running AppKit application (no Dock icon).
let application = NSApplication.shared
application.setActivationPolicy(.accessory)

signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM] {
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { Task { await stop("signal") } }
    src.resume()
    signalSources.append(src)
}

Task {
    do {
        var content: SCShareableContent?
        var window: SCWindow?
        for _ in 0..<240 {  // wait up to 2 min for the app's main window
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            window = content?.windows
                .filter { $0.owningApplication?.processID == pid && $0.frame.width >= 320 && $0.frame.height >= 240 }
                .max { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }
            if window != nil { break }
            if kill(pid, 0) != 0 { fail("screenrec: process \(pid) exited before showing a window") }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        guard let content, let window, let app = window.owningApplication else {
            fail("screenrec: no window for pid \(pid)")
        }
        let display = content.displays.first { $0.frame.intersects(window.frame) } ?? content.displays[0]
        let filter = SCContentFilter(display: display, including: [app], exceptingWindows: [])
        let cfg = SCStreamConfiguration()
        let scale = min(CGFloat(filter.pointPixelScale), maxWidth / filter.contentRect.width)  // e.g. 1080p from a Retina display
        cfg.width = Int(filter.contentRect.width * scale) & ~1
        cfg.height = Int(filter.contentRect.height * scale) & ~1
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: fps)
        cfg.showsCursor = false
        cfg.capturesAudio = true              // audio from the apps in the filter only
        cfg.excludesCurrentProcessAudio = true
        cfg.sampleRate = 48000
        cfg.channelCount = 2
        cfg.queueDepth = 8

        let s = SCStream(filter: filter, configuration: cfg, delegate: recorder)
        let outCfg = SCRecordingOutputConfiguration()
        outCfg.outputURL = URL(fileURLWithPath: out)
        outCfg.outputFileType = .mp4
        outCfg.videoCodecType = .h264
        try s.addRecordingOutput(SCRecordingOutput(configuration: outCfg, delegate: recorder))
        try await s.startCapture()
        stream = s
        print("screenrec: capturing \(app.applicationName) (pid \(pid)) on display \(display.displayID) at \(cfg.width)x\(cfg.height), \(fps) fps + audio -> \(out)")

        while kill(pid, 0) == 0 {  // stop when the recorded process exits
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        await stop("process \(pid) exited")
    } catch {
        fail("screenrec: \(error) (grant Screen Recording to your terminal in System Settings > Privacy & Security, then restart it)")
    }
}
application.run()
