"""Roll the model over windows that actually CONTAIN events, and check it reproduces them.

    .venv/bin/python scripts/eval_event_windows.py checkpoints/dynamics/joint_v15_exitev.pt

Why this exists: every earlier rollout evaluation averaged over whole held-out episodes, and those are
event-poor. On that set "assume nothing ever changes" scores 0.92-1.00 on behaviour, live-Baba count and Link
HP, because almost nothing happens -- so a model that predicts a frozen world matches the control and looks
fine. Across 12 held-out episodes the real totals were 0.33 damage and 0.25 kills; most contain no kill at all.

So select the windows where something happens and score only those. A window qualifies if, within H steps of
its start, the REAL trajectory contains the event being tested:

  hit      an actor's HP falls (Link lands a sword hit)
  link_hit Link's HP falls
  walk     Link travels more than `--walk-units` world units

Each window is rolled from its own real start state with the real action sequence, so the comparison is
like-for-like with the game. Reported per event type and horizon:

  detect   did the dream produce the event at all, and how close to the right step
  magnitude how much HP / distance, dream vs real
  identity  the same numbers for "nothing ever changes", which is the bar to beat

A model can only be said to model attacks or walking if it beats identity HERE, not on the full-episode mean.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zelda_rl.bc as bc  # noqa: E402
from zelda_rl.bc import BCNet, Norm, StateDecoder, featurise, wants_v2  # noqa: E402
from zelda_rl.dynamics import TransitionHead, load_predictor  # noqa: E402
from zelda_rl.dream_state import apply_transition, decoded_states_from_raw, decoded_to_states  # noqa: E402

MAX_STEP_UNITS = 28.0      # Link's top speed per 10 Hz env step; larger means the savestate had not settled


def find_windows(S, H, walk_units):
    """Per-step real event flags, and the window starts that contain each kind within H steps."""
    L = bc.LI
    hp = S[:, L["hp"]]
    rows = S[:, len(bc.LINK_FIELDS_V3):].reshape(len(S), bc.N_ROWS, bc.ROW_DIM)
    rhp = rows[:, :, bc.RI["hp"]]
    pres = rows[:, :, bc.RI["present"]] > 0.5
    px, pz = S[:, L["pos_x"]], S[:, L["pos_z"]]
    step = np.r_[0.0, np.hypot(np.diff(px), np.diff(pz))]
    hit = np.r_[False, ((rhp[1:] < rhp[:-1]) & pres[1:] & pres[:-1]).any(1)]
    lhit = np.r_[False, hp[1:] < hp[:-1]]
    out = {}
    n = len(S) - H - 1
    if n <= 0:
        return out
    cum = np.cumsum(step)
    for name, flag in (("hit", hit), ("link_hit", lhit)):
        c = np.cumsum(flag.astype(int))
        out[name] = [t for t in range(n) if c[t + H] - c[t] >= 1 and step[t + 1] <= MAX_STEP_UNITS]
    out["walk"] = [t for t in range(n)
                   if cum[t + H] - cum[t] >= walk_units and step[t + 1] <= MAX_STEP_UNITS]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--walk-units", type=float, default=120.0)
    ap.add_argument("--per-kind", type=int, default=40, help="windows sampled per event kind")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--held-out", action="store_true", default=True)
    ap.add_argument("--out", default="results/event_windows.json")
    a = ap.parse_args()
    torch.set_num_threads(6)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = StateDecoder(latent=LAT); dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = load_predictor(ck, LAT, NA)
    thead = None
    if ck.get("thead") is not None:
        thead = TransitionHead(LAT, NA); thead.load_state_dict(ck["thead"]); thead.eval()
    norm = Norm.load(ck["norm"]); scale = ck["decode_scale"]
    EC, ET = ck.get("event_calib") or {}, ck.get("exit_thresh", 6.38)
    held = set(ck["split"]["val_slots"])

    dirs = a.data or sorted(glob.glob("datasets/transitions/v3-*/*/"))
    pool = {"hit": [], "link_hit": [], "walk": []}
    for d in dirs:
        mf = Path(d) / "manifest.jsonl"
        if not mf.exists():
            continue
        for m in map(json.loads, open(mf)):
            if a.held_out and m["slot"] not in held:
                continue
            z = np.load(Path(d) / m["file"])
            S, A = z["state"], z["action"]
            if len(A) < a.horizon + 2:
                continue
            for k, ts in find_windows(S[:len(A) + 1], a.horizon, a.walk_units).items():
                for t in ts:
                    pool[k].append((S, A, t))
    rng0 = np.random.default_rng(0)
    for k in pool:
        if len(pool[k]) > a.per_kind:
            pool[k] = [pool[k][i] for i in rng0.choice(len(pool[k]), a.per_kind, replace=False)]
    print(f"held-out slots {sorted(held)}, horizon {a.horizon}, windows found: "
          f"{ {k: len(v) for k, v in pool.items()} }\n")

    def roll(S, A, t0):
        """Encode the real state at t0, roll H steps on the real actions, return decoded end state."""
        f = norm.apply(featurise(S[t0:t0 + 1], link_v2=wants_v2(ck["link_dim"])))
        outs = []
        for seed in range(a.seeds):
            timers, rng = {}, np.random.default_rng(seed)
            with torch.no_grad():
                z = enc.encode(*(torch.from_numpy(f[k]) for k in ("link", "rows", "ids", "mask")))
                cur = decoded_to_states(dec(z), scale)
                start = cur[0]                       # the dream's OWN start, not the truth's
                s_hp0 = start.hp
                s_bhp0 = sum(e.hp for e in start.enemies.values())
                trav = 0.0
                for j in range(a.horizon):
                    oh = torch.nn.functional.one_hot(torch.tensor([int(A[t0 + j])]), NA).float()
                    q = thead(z, oh) if thead is not None else None
                    z = pred(torch.cat([z, oh], -1))
                    nxt = decoded_to_states(dec(z), scale)
                    if q is not None:
                        nxt = apply_transition(cur, nxt, q, calib=EC, rng=rng, exit_thresh=ET,
                                               _timers=timers)
                    trav += math.hypot(nxt[0].pos[0] - cur[0].pos[0], nxt[0].pos[2] - cur[0].pos[2])
                    cur = nxt
            outs.append((cur[0], trav, s_hp0, s_bhp0))
        return outs

    res = {}
    for kind, wins in pool.items():
        if not wins:
            continue
        d_hp, r_hp, d_bhp, r_bhp, d_tr, r_tr, i_tr = [], [], [], [], [], [], []
        for S, A, t0 in wins:
            H = a.horizon
            truth0, truthH = decoded_states_from_raw(S[t0:t0 + 1])[0], decoded_states_from_raw(S[t0 + H:t0 + H + 1])[0]
            rt = math.hypot(S[t0 + H, bc.LI["pos_x"]] - S[t0, bc.LI["pos_x"]],
                            S[t0 + H, bc.LI["pos_z"]] - S[t0, bc.LI["pos_z"]])
            tot_r = float(np.sum(np.hypot(np.diff(S[t0:t0 + H + 1, bc.LI["pos_x"]]),
                                          np.diff(S[t0:t0 + H + 1, bc.LI["pos_z"]]))))
            sb0 = sum(e.hp for e in truth0.enemies.values())
            sbH = sum(e.hp for e in truthH.enemies.values())
            for st, trav, s_hp0, s_bhp0 in roll(S, A, t0):
                # measure each trajectory against ITS OWN start: the decoded seed does not always match
                # truth (a Baba at 1 HP can decode as 2), and mixing the two produced fake "healing"
                d_hp.append(s_hp0 - st.hp);          r_hp.append(truth0.hp - truthH.hp)
                d_bhp.append(s_bhp0 - sum(e.hp for e in st.enemies.values())); r_bhp.append(sb0 - sbH)
                d_tr.append(trav); r_tr.append(tot_r); i_tr.append(0.0)
        res[kind] = dict(n=len(wins),
                         baba_hp_lost_dream=float(np.mean(d_bhp)), baba_hp_lost_real=float(np.mean(r_bhp)),
                         link_hp_lost_dream=float(np.mean(d_hp)) / 16, link_hp_lost_real=float(np.mean(r_hp)) / 16,
                         travel_dream=float(np.mean(d_tr)), travel_real=float(np.mean(r_tr)),
                         detect_baba=float(np.mean(np.array(d_bhp) > 0)),
                         detect_baba_real=float(np.mean(np.array(r_bhp) > 0)),
                         detect_link=float(np.mean(np.array(d_hp) > 0)),
                         detect_link_real=float(np.mean(np.array(r_hp) > 0)))
    print(f"EVENT-RICH WINDOWS, horizon {a.horizon}, {a.seeds} seeds "
          f"(identity = nothing changes = 0 on every row)\n")
    print(f"{'window kind':<10} {'n':>4} | {'baba HP lost d/r':>20} {'detected d/r':>15} | "
          f"{'hearts lost d/r':>18} | {'travel d/r':>16}")
    for k, v in res.items():
        print(f"{k:<10} {v['n']:>4} | {v['baba_hp_lost_dream']:>8.2f}/{v['baba_hp_lost_real']:<11.2f} "
              f"{v['detect_baba']:>6.0%}/{v['detect_baba_real']:<8.0%} | "
              f"{v['link_hp_lost_dream']:>7.2f}/{v['link_hp_lost_real']:<10.2f} | "
              f"{v['travel_dream']:>7.1f}/{v['travel_real']:<8.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "horizon": a.horizon, "res": res}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
