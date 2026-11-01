"""Record a policy until it WINS, keep the winning episodes, throw the losses away.

One attempt = one `scripts/play.py` invocation playing exactly one episode on one start slot, with
`tools/screenrec.swift` capturing that Dolphin window (and its audio) for the life of the process. The
episode's outcome is read from play.py's own `■ slot N: <END>` line; SUCCESS is kept, everything else has
its .mp4 deleted immediately. One episode's video exists on disk at a time, which matters because a 60 fps
capture of a 300-second episode is hundreds of megabytes and this machine runs at 99% full.

    .venv/bin/python scripts/record_wins.py \
        --ckpt checkpoints/kept/gohma-kl-v15-r4_latest.pt --slot 7 --slot 5 \
        --slots-dir ~/Library/Application\\ Support/Dolphin/StateSaves \
        --episode-seconds 300 --wins 1 --attempts 12 --out-dir media/gohma-v15

Why the manifest matters more than the videos (AGENTS 27, and §30's "one lucky draw with a percentage
printed on it"): a kept clip of a 8%-per-episode event is NOT a demonstration of a typical episode, and a
video with no attempt count attached invites exactly that reading. So every attempt is written to
`<out-dir>/attempts.json` -- slot, outcome, boss HP taken, whether the file was kept -- and the summary
line prints `wins/attempts`. Cite that, never the clip alone.

EXPERIMENT-CONTRACT: exempt -- this process opens no env and loads no model. It launches play.py, which
pins its own torch threads (AGENTS 28, enforced there), and sets OMP/MKL to 1 for the child as well.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REC_SRC = REPO / "tools" / "screenrec.swift"
REC_BIN = Path.home() / ".cache" / "zelda-rl" / "screenrec"
DOLPHIN = Path.home() / ".cache" / "zelda-rl" / "Dolphin.app" / "Contents" / "MacOS" / "Dolphin"
PLAYS = REPO / "runs" / "play"                           # play.py drops a ~0.3 GB instance dir here per episode
PY_DBG = REPO / ".venv" / "bin" / "python-dbg"          # live RAM attach needs the signed interpreter
FROZEN_MAX = 0.75          # above this the clip is a still picture, not a recording of a game
END_RE = re.compile(r"■ slot (\d+): ([A-Z_]+) \| killed (\d+)/(\d+)")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_seconds(p: Path) -> float | None:
    """The mp4's real duration, from ffprobe. None if ffprobe is not installed."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True, timeout=60)
        return round(float(r.stdout.strip()), 1)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def fight_summary(text: str) -> dict:
    """What happened in the episode, from play.py's HUD lines, so a clip can be chosen from numbers.

    A 3-second win inside Gohma's opening stun and a 61-second win with a chase, spawned larvae and Deku
    Nut stuns are both "SUCCESS"; these are the columns that tell them apart.
    """
    out: dict = {}
    larvae = [int(n) for n in re.findall(r"larvae (\d+)", text)]
    if larvae:
        out["larvae_peak"] = max(larvae)
        # The HUD samples the ALIVE count, so every step down is a larva that died.
        out["larvae_killed"] = sum(max(0, a - b) for a, b in zip(larvae, larvae[1:]))
    out["nut_stuns"] = len(re.findall(r"'nut_stun': True", text))
    out["seed_stuns"] = len(re.findall(r"'seed_stun': True", text))
    out["stun_events"] = len(re.findall(r"'boss_stunned': True", text))
    out["boss_hits"] = sum(int(n) for n in re.findall(r"'boss_hit': (\d+)", text))
    kill_t = re.search(r"killed \d+/\d+ at \[([\d., ]*)\] s", text)
    if kill_t and kill_t.group(1).strip():
        out["kill_times_s"] = [float(x) for x in kill_t.group(1).split(",")]
    states = re.findall(r"gohma \d+hp ([a-z_]+)", text)
    if states:
        out["gohma_states_seen"] = sorted(set(states))
    return out


def join_segments(segs: list[Path], mp4: Path) -> None:
    """Concatenate the recorder's segments into one .mp4 and remove them.

    One segment is the normal case and is just a rename. Several means the system killed the stream and
    screenrec restarted it, and the segments can differ in resolution (a fullscreen switch changes the
    display mode), so a stream copy is tried first and a re-encode is the fallback.
    """
    if len(segs) == 1:
        segs[0].replace(mp4)
        return
    lst = mp4.with_suffix(".segments.txt")
    lst.write_text("".join(f"file '{s.name}'\n" for s in segs))
    base = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst.name]
    for tail in (["-c", "copy"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"]):
        r = subprocess.run(base + tail + [mp4.name], cwd=mp4.parent, capture_output=True, text=True)
        if r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0:
            break
    else:
        print(f"\n!! could not join {len(segs)} segment(s); leaving them in place", flush=True)
        return
    lst.unlink(missing_ok=True)
    for s_ in segs:
        s_.unlink(missing_ok=True)


def frozen_fraction(p: Path) -> float | None:
    """Fraction of the clip that is a FROZEN picture, via ffmpeg's freezedetect.

    A duration check is not enough. A DISPLAY-scoped capture of a FULLSCREEN Dolphin records the empty
    desktop behind it -- macOS puts a fullscreen window on its own Space -- producing a file of exactly
    the right length, with audio, that decodes without error and shows no game at all. Two such clips
    were reported "complete" on 2026-11-01 before anyone looked at a frame.

    Brightness cannot catch it: the Deku Tree interior is genuinely dark and a known-good clip measures
    the same mean luma as a blank desktop. Motion can -- a desktop does not move. Measured on the three
    clips of 2026-11-01: the blank one is 85% frozen, real gameplay 46-58%.
    """
    total = probe_seconds(p)
    if not total:
        return None
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(p), "-vf", "freezedetect=n=0.003:d=1",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=600)
        frozen = sum(float(m) for m in re.findall(r"freeze_duration: (\d+\.?\d*)", r.stderr))
        return round(min(frozen, total) / total, 3)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _unused_blackdetect(p: Path) -> float | None:
    """Fraction of the clip that is a black picture, via ffmpeg's blackdetect.

    This exists because a duration check is not enough. A display-scoped capture of a FULLSCREEN Dolphin
    records the empty desktop behind it -- macOS puts a fullscreen window on its own Space -- and that
    produces a file of exactly the right length, with audio, that decodes without error and is solid
    black. Two such clips were reported as complete on 2026-11-01 before anyone looked at a frame.
    """
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(p), "-vf", "blackdetect=d=0.5:pix_th=0.10",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=600)
        total = probe_seconds(p) or 0.0
        black = sum(float(m) for m in re.findall(r"black_duration:(\d+\.?\d*)", r.stderr))
        return round(black / total, 3) if total else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def dolphin_pids() -> set[int]:
    """Dolphins alive right now. `ps -eo` + a bracketed pattern, never `pgrep -f` (AGENTS 23)."""
    r = subprocess.run(["ps", "-eo", "pid=,command="], capture_output=True, text=True)
    out = set()
    for line in r.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if "MacOS/Dolphin" in cmd and "-u" in cmd:
            out.add(int(pid))
    return out


def build_recorder() -> None:
    if REC_BIN.exists() and REC_BIN.stat().st_mtime >= REC_SRC.stat().st_mtime:
        return
    print("building the screen recorder...", flush=True)
    REC_BIN.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["swiftc", "-O", str(REC_SRC), "-o", str(REC_BIN)], check=True)


def attempt(args, slot: int, idx: int, out_dir: Path) -> dict:
    """Play one episode on `slot`, recording it. Returns the manifest row."""
    stem = out_dir / f"{args.tag}_s{slot:02d}_try{idx:02d}"
    mp4, log = stem.with_suffix(".mp4"), stem.with_suffix(".log")
    env = dict(os.environ, DOLPHIN_BIN=str(DOLPHIN), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    cmd = [str(PY_DBG), "-u", str(REPO / "scripts" / "play.py"), str(args.ckpt),
           "--slot", str(slot), "-n", "1", "--end-hold", str(args.end_hold),
           "--hud-hz", "0.5", "--slots-dir", str(args.slots_dir)]
    if args.episode_seconds:
        cmd += ["--episode-seconds", str(args.episode_seconds)]
    if args.fullscreen:
        cmd.append("--fullscreen")
    if args.deterministic:
        cmd.append("--deterministic")

    # play.py gives every episode a fresh `runs/play/<ts>/instance` Dolphin user dir -- NAND, config and a
    # copy of the savestates, measured at 210-550 MB EACH. Nothing deleted them, so a 14-attempt Gohma run
    # would leak ~5 GB on a volume with 9 GiB free and die writing a savestate (zelda_rl/tmpdirs.py docstring
    # is the same failure on the pods). So snapshot the dir now and reap the attempt's own instance after.
    plays_before = set(PLAYS.glob("*")) if PLAYS.is_dir() else set()
    before = dolphin_pids()
    t0 = time.time()
    with log.open("w") as lf:
        play = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)

        pid, rec = None, None
        for _ in range(int(args.boot_timeout * 2)):
            new = dolphin_pids() - before
            if new:
                pid = sorted(new)[-1]
                break
            if play.poll() is not None:
                break
            time.sleep(0.5)

        if pid is None:
            play.kill()
            return dict(slot=slot, attempt=idx, outcome="NO_DOLPHIN", kept=False,
                        log=str(log.relative_to(REPO)))

        # Attach the recorder only once the EPISODE has started, not when the Dolphin process appears.
        # Dolphin applies `Fullscreen: True` when emulation begins, and that display change makes macOS stop
        # a display-scoped SCStream: "Code=-3821 Stream was stopped by the system". Measured on the first
        # smoke attempt -- a 61 s winning episode produced an 18.6 s clip of the BOOT and nothing else.
        # play.py prints "── slot N ep" immediately after env.reset(), by which point fullscreen has settled.
        # The "▶ <ckpt>" line is printed once make_env has booted Dolphin and attached to its RAM, which
        # means emulation -- and therefore the fullscreen switch -- has already happened. Waiting for the
        # later "── slot" line instead costs the start of the episode, and slot 7 can be won in 3.3 s.
        t_boot = time.time()
        while time.time() - t_boot < args.boot_timeout:
            if "▶ " in log.read_text():
                time.sleep(args.capture_settle)             # let the fullscreen transition finish
                break
            if play.poll() is not None:
                break
            time.sleep(0.25)
        else:
            print("(env never came up; recording the window as-is)", end=" ", flush=True)
        t_rec = time.time()

        reclog = stem.with_suffix(".rec.log")               # the recorder's own diagnosis of a missing video
        with reclog.open("w") as rf:
            rec = subprocess.Popen([str(REC_BIN), "--pid", str(pid), "--out", str(mp4),
                                    "--fps", str(args.fps), "--max-width", str(args.max_width)],
                                   stdout=rf, stderr=subprocess.STDOUT)
        try:
            play.wait(timeout=args.attempt_timeout)
        except subprocess.TimeoutExpired:
            play.send_signal(signal.SIGINT)
            play.wait(timeout=60)
        if rec.poll() is None:
            rec.send_signal(signal.SIGINT)
        try:
            rec_rc = rec.wait(timeout=90)
        except subprocess.TimeoutExpired:
            rec.kill()
            rec_rc = -9

    for p in dolphin_pids() - before:                  # a dead player leaves its Dolphin running
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass

    freed = 0
    for d in (set(PLAYS.glob("*")) if PLAYS.is_dir() else set()) - plays_before:
        inst = d / "instance"                          # keep episodes.jsonl, drop the emulator user dir
        if inst.is_dir():
            try:
                freed += sum(f.stat().st_size for f in inst.rglob("*") if f.is_file())
            except OSError:
                pass
            shutil.rmtree(inst, ignore_errors=True)

    text = log.read_text()
    m = END_RE.search(text)
    row = dict(slot=slot, attempt=idx, wall_s=round(time.time() - t0, 1),
               outcome=(m.group(2) if m else "NO_RESULT"),
               killed=(f"{m.group(3)}/{m.group(4)}" if m else None),
               log=str(log.relative_to(REPO)), instance_mb_freed=round(freed / 1e6))
    hp = re.findall(r"gohma (\d+)hp", text)
    if hp:
        row["boss_hp_start"], row["boss_hp_end"] = int(hp[0]), int(hp[-1])
        row["boss_damage"] = int(hp[0]) - int(hp[-1])

    row["recorded_s"] = round(time.time() - t_rec, 1)      # how long the recorder was attached for
    row["recorder_exit"] = rec_rc                          # 3 = the stream died and did not come back
    segs = sorted(stem.parent.glob(stem.name + ".part*.mp4"))
    row["segments"] = len(segs)
    if segs:
        join_segments(segs, mp4)
    if rec_rc not in (0, None):
        row["recorder_error"] = (f"screenrec exited {rec_rc}: the capture did not run to the end of the "
                                 f"episode, see {reclog.relative_to(REPO)}")

    # Measure the clip on EVERY attempt, win or lose, before a loss deletes it. Otherwise the only
    # attempts that say whether the capture survived are the rare ones, and a capture regression would
    # not be visible until it had already ruined the asset we were hunting for (AGENTS 16).
    # The recorder attaches before the episode starts, so a complete clip is at least as long as the
    # episode. Comparing against the whole recording window instead would count the end-hold and
    # Dolphin's teardown as missing footage and cry wolf on a good clip (AGENTS 36).
    ep_s = max((float(t) for t in re.findall(r"t=\s*([\d.]+)s", text)), default=0.0)
    row["episode_seconds"] = ep_s
    dur = probe_seconds(mp4) if (mp4.exists() and mp4.stat().st_size > 0) else None
    row["video_seconds"] = dur
    row["frozen_fraction"] = blk = frozen_fraction(mp4) if dur is not None else None
    row["capture_complete"] = bool(dur is not None and ep_s and dur >= 0.9 * ep_s and rec_rc == 0
                                   and (blk is None or blk < FROZEN_MAX))
    row["accepted_reason"] = None

    row.update(fight_summary(text))

    # The acceptance test, in code so it cannot be eyeballed. A SUCCESS that fails it is a real win and
    # a useless video, so it is recorded as an attempt, its clip is dropped, and the run keeps going.
    reject = []
    if args.min_episode_seconds and ep_s < args.min_episode_seconds:
        reject.append(f"episode {ep_s:.1f}s < {args.min_episode_seconds:.0f}s")
    if args.require_larva_kill and not row.get("larvae_killed"):
        reject.append("no larva killed")
    if args.require_boss_damage and row.get("boss_damage", 0) < args.require_boss_damage:
        reject.append(f"boss damage {row.get('boss_damage')} < {args.require_boss_damage}")
    row["rejected_by_acceptance"] = "; ".join(reject) or None

    win = row["outcome"] == "SUCCESS" and not reject
    if row["outcome"] == "SUCCESS" and reject:
        print(f"\n!! SUCCESS but REJECTED as an asset: {row['rejected_by_acceptance']}", flush=True)
    if win and dur is not None:
        row.update(kept=True, video=str(mp4.relative_to(REPO)), bytes=mp4.stat().st_size, md5=md5(mp4))
        if rec_rc != 0:
            # The recorder itself says the stream died and did not come back. Printing success over that
            # file is the exact failure this repo keeps paying for, so it is not a win.
            row["kept"] = False
            print(f"\n!! NOT COUNTED AS A WIN: {row['recorder_error']}", flush=True)
        elif blk is not None and blk >= FROZEN_MAX:
            row["kept"] = False
            row["video_blank"] = (f"{blk:.0%} of the clip is a frozen picture: the capture most likely "
                                  f"recorded the desktop behind the fullscreen window, not the game")
            print(f"\n!! NOT COUNTED AS A WIN: {row['video_blank']}", flush=True)
        elif ep_s and dur < 0.9 * ep_s:
            # The kill is the LAST second of the episode, so a short clip is a failed attempt, not an
            # asset. Keep the file (re-playing the episode costs minutes) but do not count it as a win.
            row["kept"] = False
            row["video_truncated"] = (f"{dur:.1f} s of video for a {ep_s:.1f} s episode: the capture was "
                                      f"cut short and the kill is NOT on film, see {reclog.relative_to(REPO)}")
            print(f"\n!! TRUNCATED, NOT COUNTED AS A WIN: {row['video_truncated']}", flush=True)
    elif win:
        # A win with no video is a RECORDING failure, not a lost episode: say so instead of filing it as a loss.
        row["kept"] = False
        row["video_error"] = (f"screenrec produced no usable file ({'0 bytes' if mp4.exists() else 'missing'}); "
                              f"see {reclog.relative_to(REPO)} -- usually macOS Screen Recording permission")
        print(f"\n!! WON but the video is unusable: {row['video_error']}", flush=True)
        if mp4.exists():
            mp4.unlink()
    else:
        row["kept"] = False
        if mp4.exists():
            mp4.unlink()                               # 11 GiB free: a loss does not get to keep 200 MB
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slot", type=int, action="append", required=True,
                    help="start slot to try; repeatable, cycled in order")
    ap.add_argument("--slots-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None, help="file prefix (default: the out-dir's name)")
    ap.add_argument("--wins", type=int, default=1, help="stop after this many winning episodes")
    ap.add_argument("--attempts", type=int, default=12, help="give up after this many episodes")
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fullscreen", action="store_true", default=True)
    ap.add_argument("--windowed", dest="fullscreen", action="store_false")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--end-hold", type=float, default=4.0)
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    ap.add_argument("--min-episode-seconds", type=float, default=0.0,
                    help="reject a win whose episode was shorter than this (a 3 s kill inside Gohma's "
                         "opening stun is a SUCCESS and a worthless clip)")
    ap.add_argument("--require-larva-kill", action="store_true",
                    help="reject a Gohma win in which no larva was killed")
    ap.add_argument("--require-boss-damage", type=int, default=0,
                    help="reject a win that took less than this much boss HP (10 = a full-health fight)")
    ap.add_argument("--capture-settle", type=float, default=2.0,
                    help="seconds to wait after the episode starts before attaching the recorder")
    ap.add_argument("--attempt-timeout", type=float, default=900.0)
    args = ap.parse_args()

    args.slots_dir = Path(os.path.expanduser(args.slots_dir))
    if not args.slots_dir.is_dir():
        raise SystemExit(f"--slots-dir does not exist: {args.slots_dir}")
    if not Path(args.ckpt).is_file():
        raise SystemExit(f"--ckpt does not exist: {args.ckpt}")
    if not PY_DBG.is_file():
        raise SystemExit(f"the signed interpreter is missing: {PY_DBG} (docs/MAP.md)")
    out_dir = Path(args.out_dir).resolve()      # relative paths broke every row's log=relative_to(REPO)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.tag = args.tag or out_dir.name
    build_recorder()
    # A 300 s episode is long enough for the display to idle-sleep, and a display change is exactly what
    # makes macOS stop the capture. Hold the system awake for as long as this process lives.
    subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    manifest = out_dir / "attempts.json"
    rows = json.loads(manifest.read_text())["attempts"] if manifest.is_file() else []
    base = len(rows)
    wins = sum(1 for r in rows if r.get("kept"))
    print(f"ckpt {args.ckpt}\nslots {args.slot} from {args.slots_dir}\n"
          f"target {args.wins} win(s) in <= {args.attempts} attempt(s); {base} already on record\n",
          flush=True)

    for i in range(args.attempts):
        slot = args.slot[i % len(args.slot)]
        idx = base + i + 1
        print(f"[{idx}] slot {slot} ...", end=" ", flush=True)
        row = attempt(args, slot, idx, out_dir)
        rows.append(row)
        manifest.write_text(json.dumps({"ckpt": str(args.ckpt), "slots_dir": str(args.slots_dir),
                                        "attempts": rows}, indent=2) + "\n")
        dmg = f" dmg {row['boss_damage']}/{row['boss_hp_start']}" if "boss_damage" in row else ""
        cap = (f" [capture {row.get('video_seconds')}s of {row.get('episode_seconds')}s ep"
               f" frozen {row.get('frozen_fraction')}"
               f"{'' if row.get('capture_complete') else ' INCOMPLETE'}]")
        print(f"{row['outcome']}{dmg} in {row.get('wall_s', 0):.0f}s{cap}"
              f"{'  <- KEPT ' + row['video'] if row.get('kept') else ''}", flush=True)
        if row.get("kept"):
            wins += 1
            if wins >= args.wins:
                break

    kept = [r for r in rows if r.get("kept")]
    print(f"\n{wins} win(s) kept from {len(rows)} attempt(s) -> {manifest}")
    for r in kept:
        print(f"  slot {r['slot']:>2}  {r['video']}  {r['bytes'] / 1e6:.0f} MB  md5 {r['md5'][:12]}")
    if not kept:
        print("  nothing kept. The attempt logs are still there; raise --attempts or pick an easier slot.")


if __name__ == "__main__":
    main()
