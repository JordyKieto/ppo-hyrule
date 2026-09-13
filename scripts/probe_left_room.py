"""Why does slot 1 end as left_room? Random actions from slot 1; log the reason + Link pos/scene/room each second."""
import os, sys, time, random, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig, N_COMBAT_ACTIONS
from zelda_rl.obs import read_state_file_v2
from zelda_rl.combat import CombatConfig
D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
files = {1: D / "NACE01.s01"}; states = {1: read_state_file_v2(files[1])}
cfg = EnvConfig(video_backend="Null", unlimited_speed=True, obs_version=3, task="combat", combat=CombatConfig(), episode_vi=60 * 30)
env = DolphinGoalEnv(0, Path("runs/instances/probe_left"), files, states, states[1], cfg)
random.seed(0)
for ep in range(6):
    env.reset(1); s0 = env.cstate
    for t in range(cfg.max_steps):
        env.send_action(random.randrange(N_COMBAT_ACTIONS))
        while not env.ready(): time.sleep(0.0005)
        _, r, term, trunc, info = env.observe()
        c = env.cstate
        if term or trunc:
            e = info["episode"]
            print(f"ep{ep}: end={e['end']} reason={e.get('left_reason')} t={t/10:.1f}s start=({s0.pos[0]:.0f},{s0.pos[1]:.0f},{s0.pos[2]:.0f}) "
                  f"end=({c.pos[0]:.0f},{c.pos[1]:.0f},{c.pos[2]:.0f}) scene={c.scene} room={c.room}", flush=True)
            break
env.close()
