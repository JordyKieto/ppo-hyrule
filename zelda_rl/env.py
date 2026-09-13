"""Goal-conditioned OoT environment on a live Dolphin instance (pipes for input, zero-copy RAM for observations)."""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zelda_rl.dolphin import HOTKEY_MAP, DolphinInstance, build_user_dir
from zelda_rl.goal import GoalConfig, GoalMetrics, goal_metrics, reward
from zelda_rl.live import attach_n64_ram
from zelda_rl.obs import PLAY_STATE, GameState, featurize, featurize_v2, read_state, read_state_v2

# 9 stick directions (None = neutral; else stick angle, 0 = right, pi/2 = up/away from camera) x 3 buttons.
DIRECTIONS: list[float | None] = [None] + [k * math.pi / 4 for k in range(8)]
BUTTONS = [None, "A", "B"]           # A: roll/talk/advance text, B: sword (grass)
ACTIONS = [(d, b) for d in DIRECTIONS for b in range(len(BUTTONS))]
N_ACTIONS = len(ACTIONS)
# combat: verified live -- GC L (held) = Z-target, GC R (held) = shield, B = sword, A = action / jump attack when locked
COMBAT_BUTTONS = [(), ("A",), ("B",), ("L",), ("R",), ("L", "B"), ("L", "A")]
COMBAT_BUTTON_NAMES = ["", "+A", "+B", "+Z", "+Shield", "+Z+B", "+Z+A"]
COMBAT_ACTIONS = [(d, b) for d in DIRECTIONS for b in range(len(COMBAT_BUTTONS))]
N_COMBAT_ACTIONS = len(COMBAT_ACTIONS)


def action_name(a: int, task: str = "goal") -> str:
    if task == "combat":
        d, b = divmod(a, len(COMBAT_BUTTONS))
        return DIR_NAMES[d] + COMBAT_BUTTON_NAMES[b]
    d, b = divmod(a, len(BUTTONS))
    return DIR_NAMES[d] + ("+" + BUTTONS[b] if b else "")
DIR_NAMES = ["-", "E", "NE", "N", "NW", "W", "SW", "S", "SE"]
# v2: directions are WORLD angles in the x/z plane (atan2(dz, dx)); the env converts them to a stick angle using the
# camera yaw. Stick "up" moves Link along the camera's forward vector; STICK_HANDEDNESS (+1/-1) says which way stick
# angles rotate in world x/z. Calibrated by scripts/calibrate_stick.py.
STICK_HANDEDNESS = -1


class EnvError(RuntimeError):
    def __init__(self, msg: str, idx: int = -1):
        super().__init__(msg)
        self.idx = idx


@dataclass
class EnvConfig:
    episode_vi: int = 3600            # 60 s of game time
    vi_per_step: int = 6              # 10 Hz decisions
    video_backend: str = "Metal"
    obs_version: int = 2
    task: str = "goal"                # "goal" (reach a target state) | "combat" (defeat enemies, zelda_rl/combat.py)
    stall_timeout: float = 10.0       # seconds without VI progress before EnvError (visible-window play uses more)
    combat: "object" = None           # CombatConfig when task == "combat"
    unlimited_speed: bool = False
    goal: GoalConfig = field(default_factory=GoalConfig)

    @property
    def max_steps(self) -> int:
        return self.episode_vi // self.vi_per_step


class DolphinGoalEnv:
    def __init__(self, idx: int, instance_dir: Path, slot_files: dict[int, Path], slot_states: dict[int, GameState],
                 goal: GameState, cfg: EnvConfig):
        self.idx, self.cfg, self.goal = idx, cfg, goal
        self.slot_states = slot_states
        shutil.rmtree(instance_dir, ignore_errors=True)
        ud = build_user_dir(instance_dir.resolve(), dump_frames=False, unlimited_speed=cfg.unlimited_speed)
        for k, f in slot_files.items():
            shutil.copy(f, ud / "StateSaves" / f"NACE01.s{k:02d}")
        self.inst = DolphinInstance(ud, video_backend=cfg.video_backend)
        self._read = read_state_v2 if cfg.obs_version >= 2 else read_state
        self.start_flags = None
        if cfg.task == "combat":
            from zelda_rl.combat import CombatConfig, CombatTracker
            self.combat_cfg = cfg.combat or CombatConfig()
            self.tracker = CombatTracker(self.combat_cfg)
        self.ram = attach_n64_ram(self.inst.proc.pid)
        self._ram_refresh = getattr(self.ram, "refresh", None)
        # Load immediately while booting: first loads sent later (during the title/attract sequence) often don't fire.
        self.load_slot(min(slot_files), timeout=60)
        self.state: GameState | None = None
        self._next_vi = 0
        self._last_vi_change = time.perf_counter()

    # ---- resets ----------------------------------------------------------------------------
    def load_slot(self, slot: int, timeout: float = 30.0) -> float:
        """Load a savestate slot and wait until RAM shows the VI clock jump to the slot's saved value. Returns seconds.

        Press ONCE and wait. Only re-press if the game provably kept running without loading (clock advanced
        >300 VIs past the press). Re-pressing while a load is still in progress (slow under CPU load) can hang Dolphin.
        """
        target = self.slot_states[slot]
        key = HOTKEY_MAP[f"Load State/Load State Slot {slot}"]
        t0 = time.perf_counter()
        self.inst.pad.send("SET MAIN 0.5 0.5", "RELEASE A", "RELEASE B")
        s = read_state(self.ram)
        if s.vi == target.vi and s.scene == target.scene:      # untouched since this slot was just loaded
            return 0.0
        while time.perf_counter() - t0 < timeout:
            press_vi = prev_vi = read_state(self.ram).vi
            self.inst.hk.send(f"PRESS {key}")
            t_press = time.perf_counter()
            released = False
            while True:
                if not released and time.perf_counter() - t_press > 0.1:
                    self.inst.hk.send(f"RELEASE {key}")
                    released = True
                s = read_state(self.ram)
                if abs(s.vi - prev_vi) > 30 and abs(s.vi - target.vi) < 60 and s.scene == target.scene:
                    if not released:
                        self.inst.hk.send(f"RELEASE {key}")
                    return time.perf_counter() - t0
                prev_vi = s.vi
                missed = s.vi - press_vi > 300 and abs(s.vi - target.vi) >= 60
                if missed or time.perf_counter() - t_press > 5.0 or time.perf_counter() - t0 > timeout:
                    break
                time.sleep(0.001)
            if not released:
                self.inst.hk.send(f"RELEASE {key}")
            time.sleep(0.2)
        raise EnvError(f"env{self.idx}: loading slot {slot} timed out", self.idx)

    def reset(self, slot: int) -> np.ndarray:
        self.load_slot(slot)
        self.state = self._read(self.ram)
        self.start_flags = getattr(self.state, "flags", None)
        if self.cfg.task == "combat":
            from zelda_rl.combat import read_combat_state
            self.start_state = self.state
            self.cstate = read_combat_state(self.ram)
            self.tracker.reset(self.cstate, self.cfg.max_steps)
        self.slot, self.t = slot, 0
        self.prev_pos = self.state.xyz
        self.prev_action: tuple[float | None, int] = (None, 0)
        self._next_vi = self.state.vi + self.cfg.vi_per_step
        self._last_vi_change = time.perf_counter()
        return self._obs()

    def input_check(self, wait_vi: int = 12) -> dict:
        """Does the game see our pad? (frozen states read zeros). Uses the game's own controller struct."""
        self.inst.pad.send("SET MAIN 1.0 0.5")
        vi0 = read_state(self.ram).vi
        t0 = time.perf_counter()
        while read_state(self.ram).vi < vi0 + wait_vi and time.perf_counter() - t0 < 3:
            time.sleep(0.005)
        stick_x = np.frombuffer(bytes(self.ram[(PLAY_STATE + 0x16) & 0x3FFFFF:(PLAY_STATE + 0x17) & 0x3FFFFF]), np.int8)[0]
        self.inst.pad.send("SET MAIN 0.5 0.5")
        vi_rate = (read_state(self.ram).vi - vi0) / max(time.perf_counter() - t0, 1e-6)
        return {"game_sees_stick": int(stick_x) > 40, "stick_x": int(stick_x), "vi_per_s": round(vi_rate, 1)}

    # ---- stepping --------------------------------------------------------------------------
    def send_action(self, a: int) -> None:
        if self.cfg.task == "combat":
            d, b = COMBAT_ACTIONS[a]
            held = COMBAT_BUTTONS[b]
            x, y = (0.5, 0.5) if d is None else self._stick(d)
            self.inst.pad.send(f"SET MAIN {x:.3f} {y:.3f}", *[f"{'PRESS' if n in held else 'RELEASE'} {n}" for n in ("A", "B", "L", "R")])
            self.prev_action = (d, b)
            return
        d, b = ACTIONS[a]
        x, y = (0.5, 0.5) if d is None else self._stick(d)
        cmds = [f"SET MAIN {x:.3f} {y:.3f}"]
        cmds += [f"PRESS {n}" if i == b else f"RELEASE {n}" for i, n in enumerate(BUTTONS) if n]
        self.inst.pad.send(*cmds)
        self.prev_action = (d, b)

    def _refresh(self) -> None:
        """Copying RAM readers (Linux fallback) expose refresh(); zero-copy views don't need it."""
        if self._ram_refresh is not None:
            self._ram_refresh()

    def _stick(self, d: float) -> tuple[float, float]:
        stick = d
        if self.cfg.obs_version >= 2:          # world-relative: rotate by camera yaw into stick space
            stick = math.pi / 2 + STICK_HANDEDNESS * (d - self.state.cam_yaw)
        return 0.5 + 0.5 * math.cos(stick), 0.5 + 0.5 * math.sin(stick)

    def ready(self) -> bool:
        self._refresh()
        vi = read_state_vi(self.ram)
        now = time.perf_counter()
        if vi != getattr(self, "_seen_vi", None):
            self._seen_vi, self._last_vi_change = vi, now
        elif now - self._last_vi_change > self.cfg.stall_timeout:
            raise EnvError(f"env{self.idx}: emulation stalled (VI stuck at {vi})", self.idx)
        elif self.cfg.stall_timeout > 10 and now - self._last_vi_change > 3 and now - getattr(self, "_last_pause_msg", 0) > 5:
            self._last_pause_msg = now
            print(f"  (Dolphin paused {now - self._last_vi_change:.0f}s: bring its window to the front / un-minimize it)", flush=True)
        return vi >= self._next_vi

    def observe(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        prev = self.state
        self._refresh()
        cur = self._read(self.ram)
        self.t += 1
        if self.cfg.task == "combat":
            from zelda_rl.combat import read_combat_state
            cprev, self.cstate = self.cstate, read_combat_state(self.ram)
            r, terminated, events = self.tracker.step(cprev, self.cstate)
            truncated = not terminated and self.t >= self.cfg.max_steps
            info = {"events": events, "vi_elapsed": cur.vi - prev.vi, "state": cur, "episode": self.tracker.ep}
            self.prev_pos, self.state = prev.xyz, cur
            self._next_vi = cur.vi + self.cfg.vi_per_step
            return self._obs(), r, terminated, truncated, info
        r, m, terminated = reward(prev, cur, self.goal, self.cfg.goal, self.start_flags)
        truncated = not terminated and self.t >= self.cfg.max_steps
        info = {"metrics": m, "vi_elapsed": cur.vi - prev.vi, "state": cur}
        self.prev_pos, self.state = prev.xyz, cur
        self._next_vi = cur.vi + self.cfg.vi_per_step
        return self._obs(), r, terminated, truncated, info

    def metrics(self) -> GoalMetrics:
        return goal_metrics(self.state, self.goal, self.cfg.goal, self.start_flags)

    def _obs(self) -> np.ndarray:
        tl = 1.0 - self.t / self.cfg.max_steps
        if self.cfg.task == "combat":
            from zelda_rl.combat import combat_features
            from zelda_rl.obs import GLOBAL_DIM_V2
            v2 = featurize_v2(self.state, self.start_state, self.prev_pos, tl, self.prev_action[0], self.prev_action[1],
                              self.start_flags)
            return np.concatenate([v2[:GLOBAL_DIM_V2], combat_features(self.cstate, self.tracker, tl), v2[GLOBAL_DIM_V2:]])
        if self.cfg.obs_version >= 2:
            return featurize_v2(self.state, self.goal, self.prev_pos, tl, self.prev_action[0], self.prev_action[1],
                                self.start_flags)
        return featurize(self.state, self.goal, self.prev_pos, tl, self.prev_action[0], self.prev_action[1])

    def diagnose(self) -> dict:
        """Snapshot for stall post-mortems: our RAM view vs fresh reads, game scene/pos, Dolphin process state."""
        import subprocess
        import sys as _sys
        from zelda_rl.obs import VI_COUNTER
        d = {"pid": self.inst.proc.pid, "alive": self.inst.proc.poll() is None}
        try:
            s = read_state(self.ram)
            d.update(view_vi=s.vi, scene=s.scene, pos=[round(v) for v in s.pos], rupees=s.rupees)
            time.sleep(0.5)
            d["view_vi_after_0.5s"] = read_state(self.ram).vi
        except Exception as e:  # noqa: BLE001
            d["view_error"] = repr(e)
        if _sys.platform == "darwin":
            try:
                from zelda_rl.machmem import MachTask, find_mem1_views
                t = MachTask(self.inst.proc.pid)
                off = 0xF64120 + (VI_COUNTER & 0x3FFFFF)
                d["fresh_vi_per_view"] = {hex(v): int.from_bytes(t.read(v + off, 4), "big") for v in find_mem1_views(t)}
            except Exception as e:  # noqa: BLE001
                d["fresh_error"] = repr(e)
            ps = subprocess.run(["ps", "-o", "%cpu=,state=", "-p", str(self.inst.proc.pid)], capture_output=True, text=True)
            d["ps_cpu_state"] = ps.stdout.strip()
            sample = self.inst.user_dir.parent / f"stall_sample_env{self.idx}_{int(time.time())}.txt"
            subprocess.run(["sample", str(self.inst.proc.pid), "1", "-file", str(sample)], capture_output=True)
            d["stack_sample"] = str(sample)
        return d

    def close(self):
        self.inst.stop()


def read_state_vi(ram) -> int:
    from zelda_rl.obs import VI_COUNTER, _u32
    return _u32(ram, VI_COUNTER)
