"""Watch the dream-trained policy play in the real emulator.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/play_dream_policy.py --slot 1

The policy was optimised entirely inside the learned latent dynamics and has never seen the emulator.
Here the real game state is encoded with the (frozen) joint encoder and the resulting latent is handed
straight to it, so this is the authoritative-simulator evaluation: whatever it does here is what the
dream was worth.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import DEKU_STATES_DIR, EnemyRows, check_slots, encode_state  # noqa: E402
from zelda_rl.bc import BCNet, Norm, featurise  # noqa: E402
from zelda_rl.env import action_name  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402
from zelda_rl.ppo import ActorCritic  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="checkpoints/dynamics/dream_ppo.pt")
    ap.add_argument("--model", default="/tmp/joint_3e-4.pt", help="joint encoder used to make the latent")
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fast", action="store_true", help="uncapped speed (default: real time, watchable)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--hud-hz", type=float, default=3.0)
    a = ap.parse_args()
    torch.set_num_threads(2)

    mk = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = mk["latent"], mk["n_actions"]
    enc = BCNet(link_dim=mk["link_dim"], row_dim=mk["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(mk["encoder"]); enc.eval()
    norm = Norm.load(mk["norm"])

    pk = torch.load(a.policy, weights_only=False, map_location="cpu")
    pol = ActorCritic(LAT, NA, pk["config"]["hidden"])
    pol.load_state_dict(pk["model"]); pol.eval()
    hist = pk.get("history", [])
    print(f"policy {a.policy}  (trained on {hist[-1]['transitions']:,} dream transitions, "
          f"final dream return {hist[-1]['return']:.1f}; real-data baseline {pk.get('baseline', float('nan')):.2f})")

    babas = check_slots(a.slots_dir, [a.slot])
    if not babas[a.slot]:
        raise SystemExit(f"slot {a.slot} has no live Deku Baba in {a.slots_dir}")
    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))
    out = Path("runs/dream_play") / time.strftime("%Y%m%d-%H%M%S")
    env, _ = make_env(cfg, out / "instance", slots=[a.slot], slots_dir=a.slots_dir,
                      video_backend="Null" if a.headless else "Metal", unlimited_speed=a.fast,
                      audio=not a.headless)
    if not a.headless and sys.platform == "darwin":
        import subprocess
        subprocess.run(["osascript", "-e", 'tell application "System Events" to set frontmost of (first '
                        f'process whose unix id is {env.inst.proc.pid}) to true'], capture_output=True)

    @torch.no_grad()
    def act(state335):
        f = norm.apply(featurise(state335[None]))
        z = enc.encode(torch.from_numpy(f["link"]), torch.from_numpy(f["rows"]),
                       torch.from_numpy(f["ids"]), torch.from_numpy(f["mask"]))
        act_, _, _, v = pol.act(z, deterministic=a.deterministic)
        return int(act_[0]), float(v[0])

    try:
        for n in range(a.episodes):
            rows = EnemyRows()
            env.reset(a.slot)
            tr = env.tracker
            print(f"\n── slot {a.slot} ep {n + 1}: Link {env.cstate.hp / 16:.1f} hearts | "
                  f"targets {tr.targets_start} | {'argmax' if a.deterministic else 'sampled'}")
            t_hud, ep = 0.0, None
            for t in range(env.cfg.max_steps):
                ac, v = act(encode_state(env.cstate, rows))
                env.send_action(ac)
                while not env.ready():
                    time.sleep(0.0005)
                _, r, term, trunc, info = env.observe()
                c, ev = env.cstate, info["events"]
                now = time.perf_counter()
                if ev or term or now - t_hud >= 1.0 / a.hud_hz:
                    t_hud = now
                    live = tr.live_targets(c)
                    near = min((round(float(np.linalg.norm(np.subtract(e.pos, c.pos)))) for e in live),
                               default=None)
                    print(f"  t={t * env.cfg.vi_per_step / 60:5.1f}s {action_name(ac, 'combat'):12s} "
                          f"V={v:6.2f}  hp {c.hp / 16:.1f} | babas {len(live)}/{tr.targets_start} "
                          f"nearest {near} | R={info['episode']['return']:6.2f}"
                          + (f" | EVENTS {ev}" if ev else ""))
                if term or trunc:
                    ep = dict(info["episode"])
                    break
            ep = ep or dict(env.tracker.ep)
            print(f"■ {ep['end'].upper()} | killed {ep['defeated']}/{ep['targets_start']} | "
                  f"hearts lost {ep['hp_lost'] / 16:.1f} | {ep['length']} steps | return {ep['return']:.1f}")
    except KeyboardInterrupt:
        print("\n interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
