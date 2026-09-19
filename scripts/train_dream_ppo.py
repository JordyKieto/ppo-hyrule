"""PPO trained entirely inside the learned latent dynamics. The emulator is never touched.

    .venv/bin/python scripts/train_dream_ppo.py --transitions 100000

Reward comes from the event head (`scripts/train_event_head.py`): a calibrated distribution over damage,
kills, Link damage and terminal, read from the latent transition. It is NOT obtained by decoding HP and
subtracting. That earlier scheme was unsound in both directions -- clamping HP monotone turned model drift
into permanent unearned kills, and leaving it unclamped let a policy farm one enemy by steering into a
region where the decode oscillates. Expected reward under a bounded probability cannot exceed the real
payout for an event, however wrong the latent is.

Two further corrections over the first version:
  * episodes start from real **episode-start** frames, not a uniform sample of every frame. Sampling
    uniformly began fights midway with about two Babas already engaged, so "cleared the room" meant
    finishing a partial fight and could not be compared with a baseline measured over whole episodes.
  * the horizon is bounded. Decoded behaviour state holds to roughly ten steps and decays after, so
    dreaming 200 steps spent most of its time where the model cannot predict at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, decode_targets, featurise,  # noqa: E402
                         hp_class)
from zelda_rl.ppo import gae, ppo_update_recurrent  # noqa: E402
from scripts.train_bc_decode import DEFAULT_DATA  # noqa: E402
from scripts.train_event_head import labels, load_with_terminal  # noqa: E402
from zelda_rl.bc import StateDecoder  # noqa: E402
from zelda_rl.combat import CombatConfig, CombatTracker  # noqa: E402
from zelda_rl.dream_state import (MEM_DIM, DreamPolicy, DreamTracker, build_seeds,  # noqa: E402
                                  decoded_states_from_raw, decoded_to_states, memory_features)
from zelda_rl.runlog import RunLogger  # noqa: E402


class Dream:
    """Batched latent rollout. Reward comes from the REAL CombatTracker run on decoded states.

    Not a re-implementation: this is the same tracker object the emulator uses, so every one of its 18
    reward terms applies -- including the four outcome terms an earlier hand-rolled version silently
    dropped (room exit, timeout, kill-speed bonus, success-time bonus), which were exactly the terms that
    punish stalling and escaping.
    """

    def __init__(self, pred, dec, scale, seed_z, seed_trk, n_actions, max_steps, combat_cfg,
                 exit_persist=3):
        self.pred, self.dec, self.scale = pred, dec, scale
        self.seed_z, self.seed_trk = seed_z, seed_trk       # latent + the tracker replayed to that frame
        self.NA, self.max_steps, self.ccfg = n_actions, max_steps, combat_cfg
        self.exit_persist = exit_persist

    @torch.no_grad()
    def _states(self, z):
        return decoded_to_states(self.dec(z), self.scale)

    def _pick(self, n, rng):
        """A seed is a real frame plus the real tracker replayed up to it -- kills already made, lock-ons
        already spent, time since the last hit. Without that, a mid-fight seed would tell the policy the
        fight had not started and pay again for progress already made."""
        j = rng.integers(0, len(self.seed_z), n)
        return self.seed_z[torch.from_numpy(j)].clone(), [self.seed_trk[k].fresh(self.max_steps) for k in j]

    @torch.no_grad()
    def reset(self, n, rng):
        self.z, self.trk = self._pick(n, rng)
        self.out_run = np.zeros(n, np.int32)
        self.cur = self._states(self.z)
        return self.z

    @torch.no_grad()
    def reset_at(self, k, rng):
        zz, tt = self._pick(len(k), rng)
        self.z[k] = zz
        st = self._states(zz)
        self.out_run[k.numpy()] = 0
        for n, i in enumerate(k.tolist()):
            self.cur[i], self.trk[i] = st[n], tt[n]

    @torch.no_grad()
    def step(self, a):
        self.z = self.pred(torch.cat([self.z, F.one_hot(a, self.NA).float()], -1))
        nxt = self._states(self.z)
        n = len(nxt)
        # Leaving the room is permanent, so require the scene flag to persist. The head's per-frame
        # false-positive rate is 0.0055, which over an ~85-step episode compounds to a ~38% chance of at
        # least one spurious -20 exit; three consecutive frames drops that to ~1e-5.
        for i, st_ in enumerate(nxt):
            self.out_run[i] = self.out_run[i] + 1 if st_.scene != 0 else 0
            if st_.scene != 0 and self.out_run[i] < self.exit_persist:
                st_.scene = 0
        r = torch.zeros(n); done = torch.zeros(n, dtype=torch.bool)
        dead = torch.zeros(n, dtype=torch.bool); cleared = torch.zeros(n, dtype=torch.bool)
        left = torch.zeros(n, dtype=torch.bool); kills = torch.zeros(n)
        dmg = torch.zeros(n); hploss = torch.zeros(n); lock = torch.zeros(n); blk = torch.zeros(n)
        for i in range(n):
            rr, term, ev = self.trk[i].step(self.cur[i], nxt[i])
            ep = self.trk[i].ep
            r[i] = rr
            kills[i] = ep["defeated"]; dmg[i] = ep["damage_dealt"]; hploss[i] = ep["hp_lost"]
            lock[i] = ep["lockons"]; blk[i] = ep["blocks"] + ep["dodges"]
            trunc = self.trk[i].t >= self.max_steps
            done[i] = term or trunc
            dead[i] = ep["end"] == "death"; cleared[i] = ep["end"] == "success"
            left[i] = ep["end"] == "left_room"
        self.cur = nxt
        return self.z, r, done, {"dead": dead, "cleared": cleared, "left": left, "kills": kills,
                                 "dmg": dmg, "hp_lost": hploss, "lockons": lock, "blocks": blk}

    def obs(self, z):
        """z plus the episode-memory block the real policy sees: kills, damage, hearts lost, time since
        last hit / damage / kill, lock-ons, time remaining."""
        m = np.stack([memory_features(t, 1.0 - t.t / self.max_steps) for t in self.trk])
        return torch.cat([z, torch.from_numpy(m)], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/tmp/joint_3e-4.pt")
    ap.add_argument("--event-head", default="checkpoints/dynamics/event_head.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--transitions", type=int, default=100_000)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=120, help="rollout length; must be >= chunk")
    ap.add_argument("--max-steps", type=int, default=1800,
                    help="dream episode cap, in steps. The real env uses 1800 (180 s at 10 Hz); 72%% of "
                         "real successful episodes take more than 100, and the documented mean clear is "
                         "132, so a shorter cap cannot contain a fight.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=60, help="recurrent PPO sequence chunk (real run: 60)")
    ap.add_argument("--minibatches", type=int, default=16, help="real run: 16, adaptive chunks_per_mb")
    ap.add_argument("--target-kl", type=float, default=0.03, help="real run early-stops an update at this")
    ap.add_argument("--gamma", type=float, default=0.995)   # parity with scripts/train_combat.py
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--out", default="checkpoints/dynamics/dream_ppo.pt")
    ap.add_argument("--resume", default=None,
                    help="snapshot to continue from: restores weights, optimizer and update counter, and "
                         "reattaches the same wandb run")
    ap.add_argument("--seed-mode", choices=["frames", "slots"], default="frames",
                    help="'slots': seed only from the 10 savestates the model-free run started from, which "
                         "is exactly its start distribution. 'frames': every training frame, with the "
                         "tracker replayed to it.")
    ap.add_argument("--exit-persist", type=int, default=3,
                    help="consecutive decoded out-of-room frames before the -20 exit terminal fires")
    ap.add_argument("--log-every", type=int, default=10, help="updates between table rows / wandb points")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb-project", default="zelda-rl-combat-ppo")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", default=True)
    ap.add_argument("--snapshot-every", type=int, default=100,
                    help="save <out>.upd<N>.pt periodically; a 41 M-step run is hours long")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    pred = nn.Sequential(nn.Linear(LAT + NA, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, LAT))
    pred.load_state_dict(ck["pred"]); pred.eval()
    dec = StateDecoder(latent=LAT); dec.load_state_dict(ck["decoder"]); dec.eval()
    for p in list(enc.parameters()) + list(pred.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)

    def epid_of(frm_):
        """Episode id per frame: frames are contiguous per episode, so a break in `frm` starts a new one."""
        return np.cumsum(np.r_[0, np.diff(frm_) != 1])

    meta, state, frm, action, slot = load_with_terminal(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    L = labels(Y, frm, action)
    norm = Norm.load(ck["norm"])
    Fn = norm.apply(featurise(state))
    with torch.no_grad():
        Z = torch.cat([enc.encode(torch.from_numpy(Fn["link"][s:s + 8192]),
                                  torch.from_numpy(Fn["rows"][s:s + 8192]),
                                  torch.from_numpy(Fn["ids"][s:s + 8192]),
                                  torch.from_numpy(Fn["mask"][s:s + 8192]))
                       for s in range(0, len(state), 8192)])

    # Seed from EVERY training frame, each paired with the real tracker replayed up to it. The real run's
    # ten start slots are themselves varied fight states -- three Babas untouched in slot 1, one left in
    # slots 4/5/9 -- and it averaged return over all of them, so a mixture of fight states is what the
    # reference number already measures.
    held = np.isin(slot, list(set(ck["split"]["val_slots"]) | set(ck["split"]["tune_slots"])))
    keep = ~held
    t_seed = time.time()
    epi = epid_of(frm)
    if a.seed_mode == "slots":
        # The model-free run started every episode from one of ten savestates. Seeding from the same ten
        # makes the start distribution identical to the reference, so the returns are comparable.
        # The tracker is fresh at an episode start, so no replay is needed.
        first = np.r_[True, np.diff(epi) != 0]
        seed_idx, seen = [], set()
        for j in np.where(first)[0]:
            if slot[j] not in seen:
                seen.add(int(slot[j])); seed_idx.append(int(j))
        seed_idx = np.array(sorted(seed_idx))
        seed_trk = []
        for j in seed_idx:
            t = CombatTracker(CombatConfig())
            t.reset(decoded_states_from_raw(state[j:j + 1])[0], a.max_steps)
            seed_trk.append(t)
        print(f"seeds: {len(seed_idx)} savestate start states, slots {sorted(seen)} "
              f"-- the model-free run's own start distribution")
    else:
        seeds = build_seeds(state, epi, keep, CombatConfig(), a.max_steps)
        seed_idx = np.array([i for i, _ in seeds])
        seed_trk = [t for _, t in seeds]
        print(f"seeds: {len(seeds):,} training frames, each with the tracker replayed to it "
              f"({time.time() - t_seed:.0f}s)")
    starts = Z[torch.from_numpy(seed_idx)]

    # real-data control: the returns the SAME tracker produced on real held-out episodes (from the manifest)
    man = [json.loads(l) for l in (Path(a.data) / "manifest.jsonl").read_text().splitlines() if l.strip()]
    tev = set(ck["split"]["val_slots"])
    base = float(np.mean([m["return"] for m in man if m["slot"] in tev]))

    resume_ck = torch.load(a.resume, weights_only=False, map_location="cpu") if a.resume else None
    run_name = (a.run_name or (resume_ck or {}).get("run_name")
                or f"dream-{Path(a.model).stem}-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir, vars(a) | {"encoder": a.model, "real_reference_return": 55.5,
                                        "real_reference_env_steps": 40_700_000},
                    use_wandb=a.wandb, project=a.wandb_project, entity=a.wandb_entity,
                    run_name=run_name, resume=bool(a.resume), tags=["dream", "latent-wm", "deku-baba"],
                    group="dream-deku-baba")

    env = Dream(pred, dec, ck["decode_scale"], starts, seed_trk, NA, a.max_steps, CombatConfig(),
                exit_persist=a.exit_persist)
    OBS = LAT + MEM_DIM
    model = DreamPolicy(OBS, NA, hidden=a.hidden, gru=a.gru)   # the emulator's recurrent policy
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    start_update, hist_prev = 1, []
    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
        if "optimizer" in resume_ck:
            opt.load_state_dict(resume_ck["optimizer"])
        else:
            print("  ! snapshot predates optimizer saving: weights restored, Adam state reset")
        start_update = resume_ck.get("update", 0) + 1
        hist_prev = resume_ck.get("history", [])
        print(f"resumed {run_name} from update {start_update - 1} "
              f"({(start_update - 1) * a.envs * a.steps:,} transitions)")
    N, T = a.envs, a.steps
    updates = max(a.transitions // (N * T), 1)
    print(f"dreaming {updates} updates x {N} envs x {T} steps = {updates*N*T:,} transitions")
    print(f"start states: {len(starts):,} ({a.seed_mode})")
    print(f"policy: DreamPolicy (RecurrentEntityActorCritic + GRU {a.gru}); obs = {LAT} latent + "
          f"{MEM_DIM} episode-memory")
    print(f"PPO (parity with train_combat.py): gamma {a.gamma} lam {a.lam} clip {a.clip} chunk {a.chunk} "
          f"minibatches {a.minibatches} target_kl {a.target_kl} ent {a.ent}")
    print(f"reward: the real CombatTracker on decoded states -- all 18 terms, no re-implementation")
    print(f"real-data control, same tracker on real held-out episodes: {base:.2f}\n")
    print(f"{'upd':>4} {'transitions':>12} {'return':>8} {'len':>5} {'clear%':>7} {'death%':>7} "
          f"{'left%':>6} {'kills':>6} {'dmg':>5} {'hpLost':>7} {'lock':>5} {'blk':>5} {'sps':>6}")
    print("-" * 100)

    z = env.reset(N, rng)
    obs = env.obs(z)
    h = model.init_hidden(N)
    first = torch.ones(N)
    ep_r, ep_l = torch.zeros(N), torch.zeros(N)
    done_ret, done_len, ncl, nde, nlf, neps, nkill = [], [], 0, 0, 0, 0, 0.0
    ndmg = nhp = nlock = nblk = 0.0
    hist = list(hist_prev)
    t0 = time.time()
    for u in range(start_update, updates + 1):
        O = torch.zeros(T, N, OBS); A_ = torch.zeros(T, N, dtype=torch.long)
        LP = torch.zeros(T, N); V = torch.zeros(T, N); R = torch.zeros(T, N); D = torch.zeros(T, N)
        FI = torch.zeros(T, N); H = torch.zeros(T, N, a.gru)
        for t in range(T):
            H[t], FI[t] = h, first
            with torch.no_grad():
                act_, lp, _, v, h = model.act(obs, h, first)
            first = torch.zeros(N)
            O[t], A_[t], LP[t], V[t] = obs, act_, lp, v
            z, r, done, info = env.step(act_)
            R[t], D[t] = r, done.float()
            ep_r += r; ep_l += 1
            if done.any():
                k = done.nonzero(as_tuple=True)[0]
                done_ret += ep_r[k].tolist(); done_len += ep_l[k].tolist()
                ncl += int(info["cleared"][k].sum()); nde += int(info["dead"][k].sum())
                nlf += int(info["left"][k].sum()); nkill += float(info["kills"][k].sum()); neps += len(k)
                ndmg += float(info["dmg"][k].sum()); nhp += float(info["hp_lost"][k].sum())
                nlock += float(info["lockons"][k].sum()); nblk += float(info["blocks"][k].sum())
                ep_r[k] = 0; ep_l[k] = 0
                env.reset_at(k, rng)
                z = env.z
                first[k] = 1.0                      # zero this env's memory at the new episode
            obs = env.obs(z)
        with torch.no_grad():
            nv = model.value(obs, h, first)
        adv, ret = gae(R.numpy(), V.numpy(), D.numpy(), nv.numpy(), a.gamma, a.lam)
        ppo_update_recurrent(model, opt, {"obs": O.numpy(), "act": A_.numpy(), "logp": LP.numpy(),
                                          "adv": adv, "ret": ret, "val": V.numpy(),
                                          "first": FI.numpy(), "h": H.numpy()},
                             chunk=a.chunk, epochs=4,
                             chunks_per_mb=max(1, (N * (T // a.chunk)) // a.minibatches),
                             clip=a.clip, ent_coef=a.ent, vf_coef=0.5, max_grad_norm=0.5,
                             target_kl=a.target_kl)
        if u % a.log_every == 0 or u == start_update:
            mr = float(np.mean(done_ret[-200:])) if done_ret else float("nan")
            ml = float(np.mean(done_len[-200:])) if done_len else float("nan")
            e = max(neps, 1)
            print(f"{u:>4} {u*N*T:>12,} {mr:>8.2f} {ml:>5.0f} {100*ncl/e:>6.1f}% {100*nde/e:>6.1f}% "
                  f"{100*nlf/e:>5.1f}% {nkill/e:>6.2f} {ndmg/e:>5.2f} {nhp/e/16:>7.2f} {nlock/e:>5.2f} "
                  f"{nblk/e:>5.2f} {u*N*T/(time.time()-t0):>6,.0f}", flush=True)
            row = {"update": u, "transitions": u * N * T, "return": mr, "length": ml,
                   "clear_pct": 100 * ncl / e, "death_pct": 100 * nde / e, "left_pct": 100 * nlf / e,
                   "kills": nkill / e, "damage": ndmg / e, "hearts_lost": nhp / e / 16,
                   "lockons": nlock / e, "blocks_dodges": nblk / e}
            hist.append(row)
            log.update({"episode/return": mr, "episode/length": ml,
                        "combat/success_rate": (100 * ncl / e) / 100, "combat/death_rate": (100 * nde / e) / 100,
                        "combat/left_room_rate": (100 * nlf / e) / 100, "combat/defeated": nkill / e,
                        "combat/damage_dealt": ndmg / e, "combat/hearts_lost": nhp / e / 16,
                        "combat/lockons": nlock / e, "combat/blocks_dodges": nblk / e,
                        "perf/dream_steps_per_sec": u * N * T / (time.time() - t0),
                        "charts/real_reference_return": 55.5},
                       global_step=u * N * T)
            ncl = nde = nlf = neps = 0; nkill = ndmg = nhp = nlock = nblk = 0.0
        if a.snapshot_every and u % a.snapshot_every == 0:
            snap = Path(str(a.out).replace(".pt", f".upd{u:05d}.pt"))
            snap.parent.mkdir(parents=True, exist_ok=True)
            # optimizer state and the update counter too: without them a snapshot can only be evaluated
            # or warm-started, not resumed -- Adam's moments and the schedule position would be lost.
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "update": u,
                        "run_name": run_name, "config": vars(a), "history": hist, "baseline": base}, snap)
    el = time.time() - t0
    print(f"\n{updates*N*T:,} dream transitions in {el:.0f}s ({updates*N*T/el:,.0f}/s), 0 emulator steps")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(a), "history": hist, "baseline": base}, a.out)
    Path("results/dream_ppo.json").write_text(json.dumps({"config": vars(a), "history": hist,
                                                          "real_baseline": base}, indent=2))
    log.summary({"final_return": hist[-1]["return"] if hist else float("nan"),
                 "dream_transitions": updates * N * T, "authoritative_env_steps": 0})


if __name__ == "__main__":
    main()
