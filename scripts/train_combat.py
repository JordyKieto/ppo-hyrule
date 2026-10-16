"""PPO combat task: defeat the Deku Babas in the Deku Tree's first room without dying or leaving it.

Start states = the savestate slots (varied Link HP, Babas damaged/defeated). Rewards are per-step RAM deltas
(zelda_rl/combat.py); episodes end on success, death (game over), leaving the room, or the 180 s cap.
Rollout = one full horizon per env (early-ended envs reset and continue), then one PPO update.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/train_combat.py --num-envs 3 --run-name deku-combat-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import band_metrics, deficit_weights, COMBAT_DIM, COMBAT_FIELDS, CombatConfig  # noqa: E402
from zelda_rl.env import N_COMBAT_ACTIONS, N_COMBAT_ACTIONS_V5, DolphinGoalEnv, EnvConfig, EnvError, action_name  # noqa: E402
from zelda_rl.obs import (ACTOR_DIM_V2, ACTOR_FIELDS_V2, GLOBAL_DIM_V2, GLOBAL_FIELDS_V2, N_ACTORS,  # noqa: E402
                          read_state_file_v2)
from zelda_rl.ppo import build_model, gae, ppo_update, ppo_update_recurrent  # noqa: E402
from zelda_rl.runlog import LOGGER, RunLogger  # noqa: E402

# Gohma profile overrides (boss room: large arena, no exits once the door shuts; the fight takes 2-3 minutes)
GOHMA_DEFAULTS = {"target_ids": (0x0028,), "room_radius": 3000.0, "fall_drop": 400.0, "timeout_penalty_per_target": 10.0,
                  "success_bonus": 20.0, "success_time_bonus": 30.0, "lockon_cap": 8}

DEFAULT_SLOTS_DIR = (os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves")
                     if sys.platform == "darwin" else "assets/states")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=time.strftime("combat-%Y%m%d-%H%M%S"))
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="initialise weights from a checkpoint; a smaller obs/action spec "
                    "(e.g. the obs-v4 Deku Baba policy) is widened function-preservingly (zelda_rl/transfer.py)")
    ap.add_argument("--transfer-action-margin", type=float, default=2.0, help="when widening a checkpoint: new action "
                    "logits start this far below the old mean (2.0 ~ 2-3%% total mass, 0.2 ~ 10%%)")
    ap.add_argument("--profile", default="deku_baba", choices=["deku_baba", "gohma"], help="combat task profile")
    ap.add_argument("--n-actors", type=int, default=0,
                    help="keep only the N nearest actor-table rows (0 = all 16). The table is 736 of the "
                         "1033-dim v7 observation and at most 4 slots are ever occupied in the Gohma room, so "
                         "~550 dims describe nothing. obs_dim: 4 -> 481, 6 -> 573, 8 -> 665, 16 -> 1033; the "
                         "SOLVED Deku task was 483 on this same 256-wide net (docs/DEKU_VS_GOHMA_SPEC.md). "
                         "Gohma keeps her own 69-dim block regardless, so she cannot drop out")
    ap.add_argument("--obs-version", type=int, default=None, help="4 (Deku Baba v4) | 5 (+boss/larva/item block, "
                    "+slingshot actions); default 4 for deku_baba, 5 for gohma")
    ap.add_argument("--num-envs", type=int, default=3)
    ap.add_argument("--total-updates", type=int, default=200)
    ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    ap.add_argument("--start-slots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--bc-data", default=None,
                    help="directory of human demonstrations (ep*.npz from scripts/record_human.py). With "
                         "--bc-coef > 0, every PPO minibatch also pays cross-entropy on an equal number of "
                         "human windows, so the policy is regularised against the DEMONSTRATIONS for the whole "
                         "run instead of only starting near them (--init-from decays within a few updates)")
    ap.add_argument("--bc-coef", type=float, default=0.0,
                    help="weight on the BC loss at update 1. 0 = off. Report bcL/bcA in the log line to see "
                         "it bite, and bcC for the coefficient actually applied this update")
    ap.add_argument("--bc-coef-final", type=float, default=None,
                    help="decay --bc-coef linearly to this FLOOR by --bc-coef-updates, then hold it there "
                         "forever (DAPG: anchor early, let PPO take over, but keep a long tail). Decaying to "
                         "0 would leave --init-from-only behaviour, which is what the term exists to avoid. "
                         "Default: no decay, the coefficient is constant")
    ap.add_argument("--bc-coef-updates", type=int, default=50,
                    help="updates over which --bc-coef reaches --bc-coef-final")
    ap.add_argument("--kl-anchor", default=None,
                    help="path to a KNOWN-GOOD checkpoint. Adds a KL penalty from the training policy toward "
                         "it, bounding the DISTANCE TRAVELLED rather than the step size. --target-kl only "
                         "early-stops the epoch loop and in v13 never fired (measured KL 0.005-0.007 vs a "
                         "0.03 threshold) while the defeat rate drifted 7.26%% -> 6.08%% over ~23 updates. "
                         "From puffertank E39, where this was the only arm that stopped the equivalent decline.")
    ap.add_argument("--kl-coef", type=float, default=0.3,
                    help="weight on the anchor KL. E39 measured coef 1.0 holding the policy AT the anchor "
                         "without improving it, and 0.3 letting it improve, so 0.3 is the default.")
    ap.add_argument("--death-scale", type=float, default=1.0,
                    help="multiply the death penalty by this. 0 removes it. E39's ablation: removing the death "
                         "penalty delayed an equivalent collapse by ~10 updates. Our own slot-8 data has the "
                         "same signature -- 20%% death, 0 wins in 225 episodes, surviving by not engaging.")
    ap.add_argument("--reward", choices=["full", "simple", "damage"], default="full",
                    help="gohma only. `simple` applies combat.GOHMA_SIMPLE: drops the entrance-era slingshot "
                         "shaping cluster and the distraction terms, and pays seed/nut stun credit only when "
                         "the stun is converted into damage. Every OUTCOME term is unchanged. See "
                         "docs/GOHMA_FOCUS.md for the measurements behind it.")
    ap.add_argument("--slot-sampling", default="fixed", choices=["fixed", "deficit"],
                    help="`deficit` reweights the start-slot distribution toward the slots the policy has NOT "
                         "solved, from its own measured per-slot success. PPO optimises expected return UNDER "
                         "the sampling distribution, so the curriculum weights ARE part of the objective: v16 "
                         "spent 30%% of its episodes on the slot it already wins 28%% of the time and 13%% on "
                         "the one it wins 0.5%% of. Fixed is the default so every earlier run reproduces.")
    ap.add_argument("--slot-deficit-ratio", type=float, default=4.0,
                    help="with --slot-sampling deficit: the least-solved slot gets exactly this many times "
                         "the sampling weight of the most-solved, whatever the absolute rates are")
    ap.add_argument("--slot-deficit-floor", type=float, default=0.06,
                    help="minimum share of episodes per slot, so a solved slot keeps being practised. v12 is "
                         "the warning: a hand-inverted curriculum starved what it had learned and stalled")
    ap.add_argument("--slot-warmup", type=int, default=3,
                    help="updates to run on --slot-weights before adapting, to measure the rates first")
    ap.add_argument("--slot-ema", type=float, default=0.3,
                    help="EMA weight on each update's per-slot success when tracking the deficit")
    ap.add_argument("--slot-weights", default="", help="curriculum: slot=weight,... (others 1), e.g. '4=2.33,5=2.33,6=2.33' "
                    "makes half of all episodes start from slots 4-6")
    ap.add_argument("--episode-seconds", type=float, default=180.0)
    ap.add_argument("--vi-per-step", type=int, default=6)
    ap.add_argument("--video-backend", default="Null")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--no-recurrent", dest="recurrent", action="store_false", default=True,
                    help="disable the GRU memory (stateless entity MLP)")
    ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=60, help="recurrent PPO sequence length (steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--combat-json", default=None, help="JSON dict overriding CombatConfig reward weights")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "zelda-rl-combat-ppo"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--fleet-experiment", default=None)
    ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world-size", type=int, default=1)
    ap.add_argument("--fleet-launch-id", default="0")
    ap.add_argument("--fleet-sync", default="avg", choices=["avg", "none"])
    ap.add_argument("--fleet-aggregator", type=int, default=0, help="rank that averages weights (pick a fast pod)")
    ap.add_argument("--publish-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.out_dir) / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.torch_threads)

    start_slots = [int(s) for s in args.start_slots.split(",")]
    slot_files = {k: Path(args.slots_dir) / f"NACE01.s{k:02d}" for k in start_slots}
    slot_states = {k: read_state_file_v2(f) for k, f in slot_files.items()}
    obs_version = args.obs_version or (7 if args.profile == "gohma" else 4)
    if args.profile == "gohma" and args.episode_seconds == 180.0:     # boss fight: 5-minute episodes unless overridden
        args.episode_seconds = 300.0
    profile_defaults = GOHMA_DEFAULTS if args.profile == "gohma" else {}
    if args.death_scale != 1.0:
        base = GOHMA_DEFAULTS.get("g_death_penalty", 30.0) if args.profile == "gohma" else 20.0
        key = "g_death_penalty" if args.profile == "gohma" else "death_penalty"
        profile_defaults = {**profile_defaults, key: base * args.death_scale}
        LOGGER.info("death penalty scaled x%.2f -> %.2f", args.death_scale, base * args.death_scale)
    if args.profile == "gohma" and args.reward == "damage":
        from zelda_rl.combat import GOHMA_DAMAGE
        profile_defaults = {**profile_defaults, **GOHMA_DAMAGE}
        LOGGER.info("reward preset `damage`: outcomes only -- boss_damage x (1 + %.1f x fraction of her HP "
                    "gone), boss_defeat, timeout, death x --death-scale. %d shaping terms zeroed.",
                    CombatConfig().g_boss_hp_progress,
                    sum(1 for v in GOHMA_DAMAGE.values() if v == 0.0))
    elif args.profile == "gohma" and args.reward == "simple":
        from zelda_rl.combat import GOHMA_SIMPLE
        profile_defaults = {**profile_defaults, **GOHMA_SIMPLE}
        LOGGER.info("reward preset `simple`: %d terms zeroed, stun credit paid on conversion only",
                    sum(1 for v in GOHMA_SIMPLE.values() if v == 0.0))
    ccfg = CombatConfig(profile=args.profile, max_seconds=args.episode_seconds,
                        **{**profile_defaults, **(json.loads(args.combat_json) if args.combat_json else {})})
    env_cfg = EnvConfig(episode_vi=int(args.episode_seconds * 60), vi_per_step=args.vi_per_step,
                        video_backend=args.video_backend, unlimited_speed=not args.realtime, obs_version=obs_version,
                        task="combat", combat=ccfg, n_actors=args.n_actors)
    T, N = env_cfg.max_steps, args.num_envs
    from zelda_rl.ego import ACTOR_DIM_V6, ACTOR_DIM_V7, ACTOR_FIELDS_V6, ACTOR_FIELDS_V7, AIM_DIM, AIM_FIELDS, EGO_V7_DIM, EGO_V7_FIELDS
    from zelda_rl.gohma import BOSS_DIM, BOSS_FIELDS
    ego_dim, ego_fields = (EGO_V7_DIM, EGO_V7_FIELDS) if obs_version >= 7 else (AIM_DIM, AIM_FIELDS) if obs_version >= 6 else (0, [])
    global_dim = GLOBAL_DIM_V2 + COMBAT_DIM + (BOSS_DIM if obs_version >= 5 else 0) + ego_dim
    actor_dim = ACTOR_DIM_V7 if obs_version >= 7 else ACTOR_DIM_V6 if obs_version >= 6 else ACTOR_DIM_V2
    n_actors = args.n_actors if args.n_actors > 0 else N_ACTORS
    obs_dim = global_dim + n_actors * actor_dim
    n_actions = N_COMBAT_ACTIONS_V5 if obs_version >= 5 else N_COMBAT_ACTIONS
    config = {**vars(args), "task": "combat", "obs_version": obs_version, "obs_dim": obs_dim, "global_dim": global_dim,
              "n_actors": n_actors, "actor_dim": actor_dim, "n_actions": n_actions, "action_frame": "world",
              "obs_fields": GLOBAL_FIELDS_V2 + COMBAT_FIELDS + (BOSS_FIELDS if obs_version >= 5 else []) + ego_fields,
              "actor_fields": ACTOR_FIELDS_V7 if obs_version >= 7 else ACTOR_FIELDS_V6 if obs_version >= 6 else ACTOR_FIELDS_V2,
              "combat_cfg": asdict(ccfg), "steps_per_rollout": T, "recurrent": args.recurrent, "gru": args.gru, "batch_size": T * N, "goal_slot": start_slots[0],
              "goal_cfg": {}}
    log = RunLogger(run_dir, config, use_wandb=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
                    run_name=args.run_name, resume=args.resume, tags=["combat", "deku-tree", args.profile],
                    group=args.wandb_group or args.fleet_experiment)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    link = None
    if args.fleet_experiment:
        from zelda_rl.fleet_link import FleetLink
        link = FleetLink(args.fleet_experiment, args.fleet_rank, args.fleet_world_size, args.fleet_launch_id)
    LOGGER.info("combat run %s | obs %d (global %d) | actions %d | %d envs x %d steps (%.0fs @ %d Hz) | slots %s",
                run_dir, obs_dim, global_dim, n_actions, N, T, args.episode_seconds, 60 // args.vi_per_step,
                start_slots)

    def make_env(i: int) -> DolphinGoalEnv:
        return DolphinGoalEnv(i, run_dir / "instances" / f"env{i}", slot_files, slot_states,
                              slot_states[start_slots[0]], env_cfg)

    with ThreadPoolExecutor(N) as pool:
        envs = list(pool.map(make_env, range(N)))
    usable = []
    for k in start_slots:
        envs[0].load_slot(k)
        chk = envs[0].input_check(wait_vi=60)
        if not chk["game_sees_stick"]:        # some boss states (red eye, stunned) ignore input for ~1 s after loading
            envs[0].load_slot(k)
            chk = envs[0].input_check(wait_vi=240)
        (usable.append(k) if chk["game_sees_stick"] else LOGGER.warning("slot %d ignores input -> excluded (%s)", k, chk))
    LOGGER.info("usable start slots: %s", usable)
    wmap = {int(k): float(v) for k, v in (kv.split("=") for kv in args.slot_weights.split(",") if kv)}
    slot_p = np.array([wmap.get(k, 1.0) for k in usable], np.float64)
    slot_p /= slot_p.sum()
    LOGGER.info("start-slot probabilities: %s", {k: round(float(p), 3) for k, p in zip(usable, slot_p)})
    slot_ema: dict[int, float] = {}          # EMA of per-slot success, the signal --slot-sampling deficit uses
    if args.slot_sampling == "deficit":
        LOGGER.info("slot sampling: DEFICIT (ratio %.1f, floor %.2f, warmup %d updates, ema %.2f) -- the "
                    "distribution will move toward the slots this policy does not win",
                    args.slot_deficit_ratio, args.slot_deficit_floor, args.slot_warmup, args.slot_ema)

    torch.manual_seed(1234)
    model = build_model(config)
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start_update, global_step, episode_count = 1, 0, 0
    latest = run_dir / "checkpoints" / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_update, global_step, episode_count = ck["update"] + 1, ck["global_step"], ck["episode_count"]
    elif args.init_from:
        src = args.init_from.format(rank=args.fleet_rank)          # e.g. runs/deku-combat-gru-v2-r{rank}/checkpoints/update_00030.pt
        ck = torch.load(src, weights_only=False)
        src_cfg = ck["config"]
        if (src_cfg["global_dim"], src_cfg["n_actions"], src_cfg["actor_dim"]) == (global_dim, n_actions, actor_dim):
            model.load_state_dict(ck["model"])
            LOGGER.info("initialised weights from %s", src)
        else:
            from zelda_rl.transfer import transfer_state_dict
            src_model = build_model(src_cfg)
            src_model.load_state_dict(ck["model"])
            model.load_state_dict(transfer_state_dict(src_model, src_cfg, config,
                                                      new_action_logit_margin=args.transfer_action_margin))
            LOGGER.info("transferred weights from %s (obs global %d -> %d, actor %d -> %d, actions %d -> %d, new-action margin %.2f)",
                        src, src_cfg["global_dim"], global_dim, src_cfg["actor_dim"], actor_dim, src_cfg["n_actions"], n_actions,
                        args.transfer_action_margin)

    rec = bool(getattr(model, "recurrent", False))
    # ---- KL anchor: a frozen known-good policy. Its log-probs are recorded DURING the rollout so its GRU
    # carries its own hidden state; reusing the trainee's state would be a different network's memory.
    anchor = None
    bc = None
    if args.bc_data and args.bc_coef > 0:
        from zelda_rl.bc_demos import coef_at, describe, load_bc_chunks
        bc = load_bc_chunks(args.bc_data, args.chunk, obs_dim=obs_dim, n_actions=n_actions)
        if args.bc_coef_final is not None:
            LOGGER.info("BC regulariser %s: coef %.3f -> %.3f over %d updates, then held (long tail): %s",
                        args.bc_data, args.bc_coef, args.bc_coef_final, args.bc_coef_updates, describe(bc))
        else:
            LOGGER.info("BC regulariser %s (coef %.3f, constant): %s", args.bc_data, args.bc_coef, describe(bc))
    elif args.bc_data or args.bc_coef > 0:
        raise SystemExit("--bc-data and --bc-coef must be given together (coef > 0), or neither")

    if args.kl_anchor:
        from zelda_rl.policy_io import load_policy
        anchor, a_cfg, _ = load_policy(Path(args.kl_anchor))
        if a_cfg.get("n_actions") != n_actions or a_cfg.get("obs_dim") != obs_dim:
            raise SystemExit(f"--kl-anchor spec mismatch: anchor is obs {a_cfg.get('obs_dim')} / "
                             f"{a_cfg.get('n_actions')} actions, this run is {obs_dim} / {n_actions}. "
                             f"Widen the anchor first (zelda_rl/transfer.py) or pick a matching checkpoint.")
        for q in anchor.parameters():
            q.requires_grad_(False)
        anchor.eval()
        LOGGER.info("KL anchor %s (coef %.3f) -- penalising drift from this policy", args.kl_anchor, args.kl_coef)
    anc_buf = np.zeros((T, N, n_actions), np.float32) if anchor is not None else None
    h_buf = np.zeros((T, N, args.gru), np.float32)
    first_buf = np.zeros((T, N), np.float32)
    obs_buf = np.zeros((T, N, obs_dim), np.float32)
    act_buf = np.zeros((T, N), np.int64)
    logp_buf, val_buf, rew_buf, done_buf = (np.zeros((T, N), np.float32) for _ in range(4))
    # `succ` is mix-weighted and has misled every cross-version decision on this task, so the banded columns
    # sit right next to it: bal = unweighted mean over slots (mix-free), hard = the same excluding the near
    # band, then near/mid/far by engagement distance, then nz = how many slots are non-zero at all.
    # dmgB / dmgH lead, because boss damage is what the task is about and it is the only signal with enough
    # resolution to show a trend (AGENTS 34). succ is kept but is the LAST thing to read.
    band_hdr = (f" {'dmgB':>5} {'dmgH':>5} {'bal':>5} {'hard':>5} {'near':>5} {'mid':>5} {'far':>5} {'nz':>5}"
                if args.profile == "gohma" else "")
    hdr = (f"{'upd':>4} {'steps':>9} {'eps':>5} {'succ':>5}{band_hdr} {'kill%':>6} {'death':>6} {'left':>5} "
           f"{'hpLost':>6} {'dmg':>5} {'lock':>5} {'blk':>4} {'return':>7} {'(out/shape)':>13} {'kl':>6} "
           f"{'ent':>5} {'sps':>5}" + (" klA" if args.kl_anchor else "")
           + ("   bcL   bcA    bcC" if bc is not None else ""))

    try:
        for update in range(start_update, args.total_updates + 1):
            for attempt in range(3):
                try:
                    t_roll = time.perf_counter()
                    eps: list[dict] = []
                    slot_of = [0] * N
                    acts = Counter()

                    def begin(i):
                        slot_of[i] = int(rng.choice(usable, p=slot_p))
                        return envs[i].reset(slot_of[i])

                    obs = np.stack([begin(i) for i in range(N)])
                    h = model.init_hidden(N) if rec else None
                    h_a = anchor.init_hidden(N) if (anchor is not None and rec) else None
                    first = torch.ones(N)
                    model.eval()
                    for t in range(T):
                        obs_buf[t] = obs
                        with torch.no_grad():
                            if rec:
                                h_buf[t], first_buf[t] = h.numpy(), first.numpy()
                                a, logp, _, v, h = model.act(torch.as_tensor(obs), h, first)
                            else:
                                a, logp, _, v = model.act(torch.as_tensor(obs))
                        if anchor is not None:
                            with torch.no_grad():
                                x = torch.as_tensor(obs)
                                lg, _, h_a = anchor.step(x, h_a, first) if rec else (anchor.actor(x), None, None)
                                anc_buf[t] = torch.log_softmax(lg, -1).numpy()
                        first = torch.zeros(N)
                        a = a.numpy()
                        act_buf[t], logp_buf[t], val_buf[t] = a, logp.numpy(), v.numpy()
                        for i, e in enumerate(envs):
                            e.send_action(int(a[i]))
                            acts[int(a[i])] += 1
                        pending, results = set(range(N)), [None] * N
                        while pending:
                            for i in list(pending):
                                if envs[i].ready():
                                    results[i] = envs[i].observe()
                                    pending.discard(i)
                            if pending:
                                time.sleep(0.0005)
                        for i, (o, r, term, trunc, info) in enumerate(results):
                            rew_buf[t, i], done_buf[t, i] = r, float(term)
                            if term or trunc or t == T - 1:
                                ep = dict(info["episode"], slot=slot_of[i], env=i, update=update, truncated=not term)
                                eps.append(ep)
                                if term and ep["end"] in ("success", "death"):
                                    LOGGER.info("  env%d slot%d %s at %.1fs | killed %d/%d | hp lost %.1f hearts | return %.1f",
                                                i, ep["slot"], ep["end"].upper(), ep["length"] * args.vi_per_step / 60,
                                                ep["defeated"], ep["targets_start"], ep["hp_lost"] / 16, ep["return"])
                                if (term or trunc) and t < T - 1:
                                    o = begin(i)
                                    first[i] = 1.0                # new episode: zero this env's memory next step
                                    if h_a is not None:
                                        h_a[i] = 0.0              # and the anchor's, on the same boundary
                            obs[i] = o
                    rollout_s = time.perf_counter() - t_roll
                    break
                except EnvError as err:
                    i_bad = err.idx if err.idx >= 0 else 0
                    LOGGER.warning("rollout aborted (%s) -> restarting env%d", err, i_bad)
                    envs[i_bad].close()
                    envs[i_bad] = make_env(i_bad)
            else:
                raise RuntimeError(f"update {update}: 3 rollout attempts failed")

            with torch.no_grad():
                next_v = (model.value(torch.as_tensor(obs), h, first) if rec else model.value(torch.as_tensor(obs))).numpy()
            np.nan_to_num(obs_buf, copy=False); np.nan_to_num(rew_buf, copy=False)
            adv, ret = gae(rew_buf, val_buf, done_buf, next_v, args.gamma, args.gae_lambda)
            batch = {"obs": obs_buf.reshape(T * N, -1), "act": act_buf.reshape(-1), "logp": logp_buf.reshape(-1),
                     "adv": adv.reshape(-1), "ret": ret.reshape(-1), "val": val_buf.reshape(-1)}
            model.train()
            t_upd = time.perf_counter()
            if rec:
                rbatch = {"obs": obs_buf, "act": act_buf, "logp": logp_buf, "adv": adv.astype(np.float32),
                          "ret": ret.astype(np.float32), "val": val_buf, "first": first_buf, "h": h_buf}
                if anc_buf is not None:
                    rbatch["anchor_logp"] = anc_buf
                n_chunks = N * (T // args.chunk)
                bc_coef_now = 0.0 if bc is None else (
                    args.bc_coef if args.bc_coef_final is None
                    else coef_at(update, args.bc_coef, args.bc_coef_final, args.bc_coef_updates))
                stats = ppo_update_recurrent(model, opt, rbatch, chunk=args.chunk, epochs=args.epochs,
                                             chunks_per_mb=max(1, n_chunks // args.minibatches), clip=args.clip,
                                             ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                                             max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                                             kl_coef=(args.kl_coef if anchor is not None else 0.0),
                                             bc=bc, bc_coef=bc_coef_now)
            else:
                stats = ppo_update(model, opt, batch, epochs=args.epochs, minibatches=args.minibatches, clip=args.clip,
                                   ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                                   target_kl=args.target_kl)
            update_s = time.perf_counter() - t_upd
            sync = {}
            if link is not None and args.fleet_world_size > 1 and args.fleet_sync == "avg":
                try:
                    sync = link.average_weights(model, update, aggregator=args.fleet_aggregator)
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("fleet sync failed at update %d: %s", update, e)
            global_step += T * N
            episode_count += len(eps)
            for ep in eps:
                log.episode(ep, global_step)

            m = lambda k, es=eps: float(np.mean([float(e[k] or 0) for e in es])) if es else float("nan")
            frac = lambda cond, es=eps: float(np.mean([cond(e) for e in es])) if es else float("nan")
            kill = lambda e: e["defeated"] / e["targets_start"] if e["targets_start"] else 1.0
            metrics = {
                "combat/success_rate": frac(lambda e: e["success"]), "combat/kill_frac": float(np.mean([kill(e) for e in eps])),
                "combat/death_rate": frac(lambda e: e["died"]), "combat/left_room_rate": frac(lambda e: e["left_room"]),
                "combat/timeout_rate": frac(lambda e: e["end"] == "timeout"),
                "combat/hp_lost_hearts": m("hp_lost") / 16, "combat/damage_dealt": m("damage_dealt"),
                "combat/lockons": m("lockons"), "combat/lock_steps": m("lock_steps"), "combat/shield_steps": m("shield_steps"),
                "combat/blocks": m("blocks"), "combat/dodges": m("dodges"), "combat/lunges_faced": m("lunges"),
                "combat/lunge_hits_taken": m("lunge_hits_taken"), "combat/stun_hits": m("stun_hits"),
                "combat/block_rate": (float(np.sum([e["blocks"] for e in eps])) / max(1.0, float(np.sum([e["lunges"] for e in eps])))),
                "combat/kills": m("defeated"), "combat/babas_at_start": m("targets_start"),
                "combat/link_hp_end_hearts": m("hp_end") / 16, "combat/link_hp_start_hearts": m("hp_start") / 16,
                "combat/first_kill_s": float(np.mean([e["kill_times_s"][0] for e in eps if e["kill_times_s"]]))
                if any(e["kill_times_s"] for e in eps) else float("nan"),
                "combat/kill_speed_mult": float(np.mean([x for e in eps for x in e["kill_mults"]]))
                if any(e["kill_mults"] for e in eps) else float("nan"),
                **{f"combat/end_{k}": frac(lambda e, k=k: e["end"] == k) for k in ("success", "death", "left_room", "timeout")},
                # per reward component: mean contribution per episode (sums to episode/return)
                **{f"reward/{k}": float(np.mean([e["r_parts"].get(k, 0.0) for e in eps]))
                   for k in sorted({k for e in eps for k in e["r_parts"]} | {"success", "death", "link_damage", "time_penalty"})},
                "combat/time_to_clear_s": float(np.mean([e["time_to_clear_s"] * args.vi_per_step / 60 for e in eps
                                                         if e["time_to_clear_s"]])) if any(e["time_to_clear_s"] for e in eps) else float("nan"),
                "episode/count": episode_count, "episode/return": m("return"), "episode/r_outcome": m("r_outcome"),
                "episode/r_shaping": m("r_shaping"), "episode/length_s": m("length") * args.vi_per_step / 60,
                **{f"ppo/{k}": v for k, v in stats.items()},
                "perf/env_steps_per_sec": T * N / rollout_s, "perf/rollout_seconds": rollout_s, "perf/update_seconds": update_s,
                **{f"fleet/{k}": v for k, v in sync.items() if k != "weight_hash"}, "update": update,
            }
            if args.profile == "gohma" and eps:
                n_seeds = float(np.sum([e["seeds_fired"] for e in eps]))
                metrics.update({
                    "gohma/defeat_rate": frac(lambda e: e["boss_defeated"]), "gohma/boss_damage": m("boss_damage"),
                    "gohma/boss_hp_end": m("boss_hp_end"), "gohma/seed_stuns": m("seed_stuns"), "gohma/stuns": m("stuns"),
                    "gohma/seeds_fired": m("seeds_fired"), "gohma/red_eye_seeds": m("red_eye_seeds"),
                    "gohma/nut_stuns": m("nut_stuns"), "gohma/nuts_thrown": m("nuts_thrown"),
                    "gohma/seed_hit_rate": float(np.sum([e["seed_stuns"] for e in eps])) / max(1.0, n_seeds),
                    "gohma/stun_hits": m("stun_hits"), "gohma/larva_kills": m("larva_kills"), "gohma/larvae_seen": m("larvae_seen"),
                    "gohma/boss_attacks": m("boss_attacks"), "gohma/boss_blocks": m("boss_blocks"),
                    "gohma/vulnerable_steps": m("vulnerable_steps"), "gohma/vulnerable_lock_steps": m("vulnerable_lock_steps"),
                    "gohma/sling_aim_steps": m("sling_aim_steps"), "gohma/sling_ready_steps": m("sling_ready_steps"), "gohma/elevated_seeds": m("elevated_seeds"), "gohma/seed_misses": m("seed_misses"), "gohma/nut_misses": m("nut_misses"),
                    "gohma/larva_kills_sling": m("larva_kills_sling"), "gohma/grass_cut": m("grass_cut"), "gohma/dropped_red_eye_shots": m("dropped_red_eye_shots"), "gohma/unfired_red_eye_windows": m("unfired_red_eye_windows"), "gohma/seeds_picked": m("seeds_picked"),
                    "gohma/nuts_picked": m("nuts_picked"),
                    "gohma/time_to_defeat_s": float(np.mean([e["time_to_defeat_s"] * args.vi_per_step / 60 for e in eps
                                                            if e["time_to_defeat_s"]])) if any(e["time_to_defeat_s"] for e in eps) else float("nan"),
                })
            for k in usable:
                se = [e for e in eps if e["slot"] == k]
                if se and args.profile == "gohma":
                    metrics[f"by_slot/s{k}/boss_damage"] = m("boss_damage", se)
                    metrics[f"by_slot/s{k}/seed_stuns"] = m("seed_stuns", se)
                if se:
                    metrics[f"by_slot/s{k}/success_rate"] = frac(lambda e: e["success"], se)
                    metrics[f"by_slot/s{k}/kill_frac"] = float(np.mean([kill(e) for e in se]))
                    metrics[f"by_slot/s{k}/death_rate"] = frac(lambda e: e["died"], se)
                    metrics[f"by_slot/s{k}/kills"] = m("defeated", se)
                    metrics[f"by_slot/s{k}/hp_lost_hearts"] = m("hp_lost", se) / 16
                    metrics[f"by_slot/s{k}/return"] = m("return", se)

            # THE HEADLINE, banded by engagement distance, plus the two mix-free aggregates.
            # One implementation, shared with tests/test_band_metrics.py (AGENTS 25).
            if args.profile == "gohma" and eps:
                metrics.update(band_metrics(eps))

            # ---- adaptive start-slot distribution. Recomputed from the EMA of this policy's OWN per-slot
            # success, so compute flows to the starts it cannot do. `success_balanced` is mix-invariant by
            # construction, which is what makes a moving distribution measurable at all.
            if args.slot_sampling == "deficit" and eps:
                for k in usable:
                    se = [e for e in eps if e["slot"] == k]
                    if se:
                        r = float(np.mean([bool(e["success"]) for e in se]))
                        slot_ema[k] = r if k not in slot_ema else \
                            (1 - args.slot_ema) * slot_ema[k] + args.slot_ema * r
                if update >= start_update + args.slot_warmup and len(slot_ema) == len(usable):
                    newp = deficit_weights(slot_ema, args.slot_deficit_ratio, args.slot_deficit_floor)
                    slot_p = np.array([newp[k] for k in usable], np.float64)
                    slot_p /= slot_p.sum()
                for k, pk in zip(usable, slot_p):
                    metrics[f"slot_p/s{k}"] = float(pk)
                    if k in slot_ema:
                        metrics[f"slot_ema/s{k}"] = slot_ema[k]
            log.update(metrics, global_step)
            if link is not None:
                hb = {"update": update, "global_step": global_step, "run": args.run_name, "weight_hash": sync.get("weight_hash"),
                      **{k.split("/")[-1]: v for k, v in metrics.items() if k.startswith(("combat/", "gohma/")) or k == "episode/return"},
                      "env_steps_per_sec": metrics["perf/env_steps_per_sec"]}
                link.heartbeat(hb)
                if update % args.publish_every == 0:
                    link.publish_policy(model, config, update, global_step, {"episode_count": episode_count, **hb})

            top = ", ".join(f"{action_name(a, 'combat')}:{c / sum(acts.values()):.0%}" for a, c in acts.most_common(3))
            if update == start_update or update % 20 == 0:
                LOGGER.info(hdr)
            def _pc(key, w=5):
                v = metrics.get(key)
                return f"{v:>{w}.1%}" if isinstance(v, float) and v == v else f"{'-':>{w}}"
            def _f2(key):
                v = metrics.get(key)
                return f"{v:>5.2f}" if isinstance(v, float) and v == v else f"{'-':>5}"
            band_line = ("" if args.profile != "gohma" else
                         f" {_f2('combat/boss_damage_balanced')} {_f2('combat/boss_damage_hard')}"
                         f" {_pc('combat/success_balanced')} {_pc('combat/success_hard')} "
                         f"{_pc('by_band/near/success_rate')} {_pc('by_band/mid/success_rate')} "
                         f"{_pc('by_band/far/success_rate')} "
                         f"{str(int(metrics.get('combat/slots_nonzero', 0))) + '/' + str(int(metrics.get('combat/slots_seen', 0))):>5}")
            LOGGER.info(f"{update:>4} {global_step:>9,} {episode_count:>5} {metrics['combat/success_rate']:>5.0%}"
                        f"{band_line} "
                        f"{metrics['combat/kill_frac']:>6.0%} {metrics['combat/death_rate']:>6.0%} {metrics['combat/left_room_rate']:>5.0%} "
                        f"{metrics['combat/hp_lost_hearts']:>6.2f} {metrics['combat/damage_dealt']:>5.2f} {metrics['combat/lockons']:>5.2f} "
                        f"{metrics['combat/blocks']:>4.1f} {metrics['episode/return']:>7.2f} "
                        f"({metrics['episode/r_outcome']:>5.1f}/{metrics['episode/r_shaping']:>5.1f}) {stats['approx_kl']:>6.3f} "
                        f"{stats['entropy']:>5.2f} {metrics['perf/env_steps_per_sec']:>5.0f}"
                        + (f" klA {stats['kl_anchor']:>6.3f}" if stats.get("kl_anchor") == stats.get("kl_anchor")
                           and "kl_anchor" in stats else "")
                        + (f" {stats.get('bc_loss', float('nan')):>5.2f} {100 * stats.get('bc_acc', float('nan')):>4.0f}%"
                           f" {bc_coef_now:>6.4f}" if bc is not None else "")
                        + f"  top: {top}"
                        + (f"  sync {sync.get('sync_ranks')}r {sync.get('weight_hash')}" if sync else ""))
            if update % args.save_every == 0 or update == args.total_updates:
                ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "update": update, "global_step": global_step,
                      "episode_count": episode_count, "config": config}
                torch.save(ck, run_dir / "checkpoints" / f"update_{update:05d}.pt")
                torch.save(ck, latest)
    finally:
        for e in envs:
            e.close()
        log.close()


if __name__ == "__main__":
    main()
