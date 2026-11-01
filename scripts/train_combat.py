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
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import band_metrics, deficit_weights, COMBAT_DIM, COMBAT_FIELDS, CombatConfig  # noqa: E402
from zelda_rl.env import N_COMBAT_ACTIONS, N_COMBAT_ACTIONS_V5, DolphinGoalEnv, EnvConfig, EnvError, action_name  # noqa: E402
from zelda_rl.obs import (ACTOR_DIM_V2, ACTOR_FIELDS_V2, GLOBAL_DIM_V2, GLOBAL_FIELDS_V2, N_ACTORS,  # noqa: E402
                          read_state_file_v2)
from zelda_rl import ckpt_s3
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
    ap.add_argument("--bc-adaptive-frac", type=float, default=0.0,
                    help="ADAPTIVE regularisation: ignore --bc-coef and set the weight each minibatch so the "
                         "BC term is this fraction of |policy loss|. At a constant 0.1 the term contributed "
                         "~0.44 against a policy loss of 0.01-0.1 and dominated; this chooses the RATIO")
    ap.add_argument("--bc-qfilter", type=int, default=0, metavar="STEPS",
                    help="Q-FILTER (outcome form): pay the BC loss only on demonstration frames where the "
                         "boss loses HP within STEPS. Nair et al. 2018 compare Q(s,a_demo) vs Q(s,a_pi); this "
                         "critic is a V head so there is no Q, and the human's realised damage is used "
                         "instead. 30 steps = 3 s keeps ~12%% of frames, 60 keeps ~21%%")
    ap.add_argument("--bc-latent-coef", type=float, default=0.0,
                    help="LATENT ALIGNMENT: MSE between this policy's GRU state and --bc-teacher's on the "
                         "demonstration frames. Aligns the representation instead of the action, which "
                         "matters because v15 and the human agree on only 3%% of actions while both winning")
    ap.add_argument("--rvs-data", action="append", default=None, metavar="DIR",
                    help="repeatable. RETURN-CONDITIONED PPO: the policy's input gains the 3 conditioning "
                         "dims of zelda_rl/rvs.py and is trained BOTH by PPO (acting with the target prompt "
                         "g=(1,1,0)) and by cross-entropy on these corpora with each step's ACTUAL outcome. "
                         "Losing episodes become coverage instead of corruption, because the action that "
                         "died is labelled with the outcome that followed it")
    ap.add_argument("--rvs-coef", type=float, default=0.5, help="weight on the return-conditioned CE term")
    ap.add_argument("--rvs-max-windows", type=int, default=12000, metavar="N",
                    help="memory budget for the static corpus. Every file is READ (so composition is exact "
                         "and every winning window is kept), but loss windows are strided to this budget: "
                         "9,340 episodes as 60-step float32 windows is ~20 GB, which does not fit beside "
                         "training on a 32 GB pod")
    ap.add_argument("--rvs-buffer", type=int, default=6000, metavar="WINDOWS",
                    help="DYNA-STYLE refresh: after every update the policy's own episodes are relabelled "
                         "with the outcome they actually produced and appended to the RvS buffer, capped "
                         "here, oldest evicted. Keep it above the static corpus so the human wins survive")
    ap.add_argument("--rvs-fresh-share", type=float, default=0.5,
                    help="HOW MUCH of the conditioning gradient comes from the policy's OWN fresh episodes, "
                         "as a share in [0,1). 0 = the static corpus alone; 0.9 = almost pure "
                         "self-imitation. The fresh windows' weights are rescaled to hit this exactly, so "
                         "the knob is influence and not window count")
    ap.add_argument("--rvs-static-only", action="store_true",
                    help="shorthand for --rvs-fresh-share 0")
    ap.add_argument("--rvs-win-share", type=float, default=0.3,
                    help="TARGET share of the conditioning gradient that winning play should carry. The "
                         "multiplier is solved from the corpus to hit it, because a fixed multiplier is the "
                         "wrong knob: 8x on a pod-local 48%%-win mix put wins at 83%% of the gradient, which "
                         "turns RvS back into anchored BC and throws away what the losses were for")
    ap.add_argument("--rvs-win-weight", type=float, default=0.0,
                    help="override the solved multiplier (0 = solve it from --rvs-win-share)")
    ap.add_argument("--bc-peer", default=None, metavar="CKPT",
                    help="MUTUAL alignment: load this checkpoint as a SECOND TRAINABLE network (normally the "
                         "BC clone). It keeps training on the demonstrations while both encoders are pulled "
                         "into a common representation space -- nothing is frozen, unlike --bc-teacher. "
                         "Needs --align-coef > 0")
    ap.add_argument("--align-coef", type=float, default=0.0,
                    help="weight on the mutual alignment term between this policy and --bc-peer")
    ap.add_argument("--align-kind", choices=["mse", "infonce"], default="infonce",
                    help="`mse` forces the two encodings to be numerically equal; `infonce` (default) only "
                         "requires each encoder to IDENTIFY the state in the other's space, which is "
                         "scale- and basis-free and is the contrastive form used by arXiv 2605.28372")
    ap.add_argument("--peer-bc-coef", type=float, default=1.0,
                    help="weight on the peer's OWN behaviour-cloning loss, so it stays a clone of the human "
                         "while it moves toward the shared space")
    ap.add_argument("--peer-lr", type=float, default=1e-4,
                    help="the peer's own learning rate; lower than the policy's by default so the shared "
                         "space is not dragged by a network with no return signal")
    ap.add_argument("--bc-latent-site", choices=["gru", "proj"], default="gru",
                    help="WHAT to align: `gru` the recurrent state, `proj` the pre-GRU encoding of the "
                         "observation (a pure function of the obs, no episode history in it)")
    ap.add_argument("--bc-latent-states", choices=["demo", "rollout"], default="demo",
                    help="WHERE to align: `demo` the human's frames, `rollout` the states this policy "
                         "actually visits. Aligning on demo frames shapes the representation where the "
                         "demonstrator went; aligning on rollout states shapes it where the policy is scored")
    ap.add_argument("--ibrl", action="store_true",
                    help="IBRL (arXiv 2311.02198) actor proposal: each step, compare the policy's sampled "
                         "action against --kl-anchor's greedy action with a learned Q head and EXECUTE the "
                         "better one. No BC term in the loss at all -- the imitation policy influences "
                         "behaviour by proposing actions, which the paper reports beats loss mixing. "
                         "Requires --kl-anchor; set --kl-coef 0 to keep the loss pure")
    ap.add_argument("--q-coef", type=float, default=0.5, help="weight on the IBRL Q head's regression loss")
    ap.add_argument("--latent-mode", choices=["state", "delta", "projected"], default="state",
                    help="`state` MSE on the latent itself; `delta` MSE on h_t - h_(t-1), matching how the "
                         "two policies UPDATE their belief rather than what it is; `projected` FitNets-style, "
                         "comparing a LEARNED LINEAR MAP of the student's latent to the teacher's, which asks "
                         "only that the student's latent contain the teacher's information up to a linear "
                         "transform instead of sharing its basis")
    ap.add_argument("--bc-teacher", default=None,
                    help="frozen policy for --bc-latent-coef, normally the BC checkpoint")
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
    ap.add_argument("--reward", choices=["full", "simple", "damage", "behaviour"], default="full",
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
    elif args.profile == "gohma" and args.reward == "behaviour":
        from zelda_rl.combat import GOHMA_BEHAVIOUR, GOHMA_SIMPLE
        profile_defaults = {**profile_defaults, **GOHMA_BEHAVIOUR}
        changed = {k: v for k, v in GOHMA_BEHAVIOUR.items() if GOHMA_SIMPLE.get(k, getattr(CombatConfig(), k)) != v}
        LOGGER.info("reward preset `behaviour`: `simple` plus the five watched behaviours -- %d weights "
                    "changed vs `simple`: %s", len(changed),
                    ", ".join(f"{k}={v}" for k, v in sorted(changed.items())))
        LOGGER.info("reward preset `behaviour`: the red-eye slingshot cluster stays at ZERO on purpose "
                    "(g_red_eye_seed, g_sling_red_eye, g_sling_ready, g_lock_red_eye, g_elevated_seed, "
                    "g_unfired_red_eye, g_drop_red_eye_shot) -- it was removed on measurement, not forgotten")
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
    def _with_prompt(o):
        """Append the inference prompt g=(1,1,0) to an env observation: during RL the policy always acts
        conditioned on the outcome we want, while the corpus term teaches it what each outcome looks like."""
        from zelda_rl.rvs import TARGET
        o = np.asarray(o, np.float32)
        g = np.asarray(TARGET, np.float32)
        if o.ndim == 1:
            return np.concatenate([o, g])
        return np.concatenate([o, np.tile(g, (o.shape[0], 1))], axis=1)

    rvs_cond = 0
    if args.rvs_data:
        from zelda_rl.rvs import COND_DIM
        rvs_cond = COND_DIM
        global_dim = global_dim + COND_DIM        # the prompt rides in the global block
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
    bc = bc_teacher = peer = peer_opt = latent_proj = None
    opt_extra = []
    # Aligning on ROLLOUT states needs NO demonstrations: the rollout branch in zelda_rl/ppo.py reads the
    # policy's own obs and sits OUTSIDE the `bc is not None` gate. Demanding --bc-data for it killed all
    # nine gohma-la arms 15 s after launch, with a message that read backwards (AGENTS 36).
    _terms = (args.bc_coef > 0 or args.bc_adaptive_frac > 0 or args.bc_latent_coef > 0
              or args.align_coef > 0)
    _demo_needed = (args.bc_coef > 0 or args.bc_adaptive_frac > 0 or args.align_coef > 0
                    or (args.bc_latent_coef > 0 and args.bc_latent_states == "demo"))
    if _demo_needed and not args.bc_data:
        raise SystemExit("--bc-coef / --bc-adaptive-frac / --align-coef, and --bc-latent-coef with "
                         "--bc-latent-states demo, all read demonstrations: pass --bc-data. "
                         "--bc-latent-states rollout does NOT need it.")
    if args.bc_data and not _terms:
        raise SystemExit("--bc-data is read by nothing: set one of --bc-coef / --bc-adaptive-frac / "
                         "--bc-latent-coef / --align-coef > 0")
    if _terms:
        from zelda_rl.bc_demos import coef_at, describe, load_bc_chunks
        if args.bc_data:
            bc = load_bc_chunks(args.bc_data, args.chunk, obs_dim=obs_dim, n_actions=n_actions,
                                qfilter_steps=args.bc_qfilter)
        if args.bc_latent_coef > 0:
            if not args.bc_teacher:
                raise SystemExit("--bc-latent-coef needs --bc-teacher (the frozen policy to align to)")
            from zelda_rl.policy_io import load_policy
            bc_teacher, t_cfg, _ = load_policy(Path(args.bc_teacher))
            if t_cfg.get("obs_dim") != obs_dim or t_cfg.get("n_actions") != n_actions:
                raise SystemExit(f"--bc-teacher spec mismatch: teacher obs {t_cfg.get('obs_dim')} / "
                                 f"{t_cfg.get('n_actions')} actions vs this run's {obs_dim} / {n_actions}")
            for q in bc_teacher.parameters():
                q.requires_grad_(False)
            bc_teacher.eval()
            LOGGER.info("latent alignment to %s (coef %.3f) mode=%s at the %s latent on %s states, "
                        "teacher frozen", args.bc_teacher, args.bc_latent_coef, args.latent_mode,
                        args.bc_latent_site, args.bc_latent_states)
            if args.latent_mode == "projected":
                # The regressor is trained WITH the policy: it is part of the alignment objective, not of the
                # policy, and it is deliberately not saved into the checkpoint's model state.
                w = args.gru if args.bc_latent_site == "gru" else args.gru
                latent_proj = torch.nn.Linear(w, w)
                opt_extra = list(latent_proj.parameters())
                LOGGER.info("FitNets regressor: Linear(%d, %d) trained alongside the policy", w, w)
        if args.bc_peer:
            if args.align_coef <= 0:
                raise SystemExit("--bc-peer needs --align-coef > 0")
            from zelda_rl.policy_io import load_policy
            peer, p_cfg, _ = load_policy(Path(args.bc_peer))
            if p_cfg.get("obs_dim") != obs_dim or p_cfg.get("n_actions") != n_actions:
                raise SystemExit(f"--bc-peer spec mismatch: peer obs {p_cfg.get('obs_dim')} / "
                                 f"{p_cfg.get('n_actions')} actions vs this run's {obs_dim} / {n_actions}")
            peer.train()
            peer_opt = torch.optim.Adam(peer.parameters(), lr=args.peer_lr)
            LOGGER.info("MUTUAL alignment with %s: peer is TRAINABLE (lr %.1e, own BC coef %.2f), "
                        "alignment %s at coef %.3f on the proj latent",
                        args.bc_peer, args.peer_lr, args.peer_bc_coef, args.align_kind, args.align_coef)
        if args.bc_adaptive_frac > 0:
            LOGGER.info("ADAPTIVE BC weight: holding the BC term at %.3f of |policy loss| (--bc-coef ignored)",
                        args.bc_adaptive_frac)
        if bc is not None and args.bc_coef_final is not None:
            LOGGER.info("BC regulariser %s: coef %.3f -> %.3f over %d updates, then held (long tail): %s",
                        args.bc_data, args.bc_coef, args.bc_coef_final, args.bc_coef_updates, describe(bc))
        elif bc is not None:
            LOGGER.info("BC regulariser %s (coef %.3f, constant): %s", args.bc_data, args.bc_coef, describe(bc))
        elif args.bc_latent_coef > 0:
            LOGGER.info("no --bc-data: the ONLY demonstration-derived term is the latent alignment, on the "
                        "policy's own rollout states")

    rvs = rvs_idx = None
    if args.rvs_data:
        # Aliased imports: a bare `from ... import GLOBAL_FIELDS_V2` here would make that name local to the
        # whole function and break its module-level use earlier in main() (UnboundLocalError).
        from zelda_rl.combat import COMBAT_FIELDS as _CF
        from zelda_rl.ego import EGO_V7_FIELDS as _EF
        from zelda_rl.gohma import BOSS_FIELDS as _BF
        from zelda_rl.obs import GLOBAL_FIELDS_V2 as _GF
        rvs_idx, _o = {}, 0
        for _n, _w in _GF + _CF + _BF + _EF:
            rvs_idx[_n] = _o
            _o += _w
        from zelda_rl.rvs_data import load_rvs_windows
        rvs = load_rvs_windows(args.rvs_data, args.chunk, obs_dim=obs_dim, n_actions=n_actions,
                               win_weight=args.rvs_win_weight, win_share=args.rvs_win_share,
                               max_windows=args.rvs_max_windows)
        LOGGER.info("RvS corpus: %s", rvs["desc"])

    q_head = None
    if args.ibrl:
        if not args.kl_anchor:
            raise SystemExit("--ibrl needs --kl-anchor <bc checkpoint>: that policy is the action proposer")
        from zelda_rl.ppo import ActionValueHead
        q_head = ActionValueHead(args.gru, args.hidden, n_actions)
        opt_extra += list(q_head.parameters())
        LOGGER.info("IBRL actor proposal: the anchor proposes, a Q head picks. Q(%d->%d->%d), coef %.2f",
                    args.gru, args.hidden, n_actions, args.q_coef)

    if opt_extra:
        # The optimiser is built before this point, so a regressor created here would get NO gradient steps
        # and the FitNets arm would silently be the plain-MSE arm. Same failure class as the three
        # unreachable-gate bugs; add it as a param group rather than hoping.
        opt.add_param_group({"params": opt_extra, "lr": args.lr})
        LOGGER.info("added %d FitNets regressor tensors to the optimiser", len(opt_extra))

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
    # One behaviour counter per behaviour the `behaviour` preset pays for, BESIDE the headline damage columns
    # and never instead of them. They are dense where a defeat is a 1-10% binary, so they move first; if one
    # of them has not moved after several updates, that term is not working and waiting will not fix it.
    #   item = items collected/episode   hrt = hearts   grs = grass tufts cut   iapp = units closed to loot
    #   face = fraction of her live steps spent facing her   fcos = mean alignment over those steps
    #   lock = fraction of steps Z-locked   nutW = nuts wasted/episode   wst% = wasted / thrown
    #   reSd = red-eye seeds fired   conv = seed stuns / red-eye seeds   s2sw = s from knockdown to sword
    #   rch% = fraction of knockdowns Link reached at all
    beh_hdr = (f" {'item':>5} {'hrt':>4} {'grs':>4} {'iapp':>5} {'face':>5} {'fcos':>5} {'lock':>5}"
               f" {'nutW':>5} {'wst%':>5} {'reSd':>5} {'conv':>5} {'s2sw':>5} {'rch%':>5}"
               if args.profile == "gohma" else "")
    hdr = (f"{'upd':>4} {'steps':>9} {'eps':>5} {'succ':>5}{band_hdr}{beh_hdr} {'kill%':>6} {'death':>6} {'left':>5} "
           f"{'hpLost':>6} {'dmg':>5} {'lock':>5} {'blk':>4} {'return':>7} {'(out/shape)':>13} {'kl':>6} "
           f"{'ent':>5} {'sps':>5}" + (" klA" if args.kl_anchor else "")
           + ("   bcL   bcA    bcC" if bc is not None else "")
           + ("   bcZ" if bc_teacher is not None else "")
           + ("   algn  peerA" if peer is not None else "")
           + ("  bcPick    qL" if q_head is not None else "")
           + ("  rvsL  rvsA" if rvs is not None else ""))

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
                    if rvs_cond:
                        obs = _with_prompt(obs)
                    h = model.init_hidden(N) if rec else None
                    h_a = anchor.init_hidden(N) if (anchor is not None and rec) else None
                    first = torch.ones(N)
                    model.eval()
                    ibrl_bc_picks = ibrl_steps = 0
                    for t in range(T):
                        obs_buf[t] = obs
                        with torch.no_grad():
                            if rec:
                                h_buf[t], first_buf[t] = h.numpy(), first.numpy()
                                a, logp, _, v, h = model.act(torch.as_tensor(obs), h, first)
                            else:
                                a, logp, _, v = model.act(torch.as_tensor(obs))
                        anc_lg = None
                        if anchor is not None:
                            with torch.no_grad():
                                x = torch.as_tensor(obs)
                                lg, _, h_a = anchor.step(x, h_a, first) if rec else (anchor.actor(x), None, None)
                                anc_lg = lg
                                anc_buf[t] = torch.log_softmax(lg, -1).numpy()
                        if q_head is not None and anc_lg is not None:
                            # IBRL actor proposal: execute whichever of the two candidate actions the Q head
                            # scores higher, and record the EXECUTED action's log-prob under pi so PPO's
                            # ratio refers to what was actually done.
                            with torch.no_grad():
                                a_bc = anc_lg.argmax(-1)
                                q = q_head(h)
                                q_rl = q.gather(-1, a.unsqueeze(-1)).squeeze(-1)
                                q_bc = q.gather(-1, a_bc.unsqueeze(-1)).squeeze(-1)
                                take_bc = q_bc > q_rl
                                a = torch.where(take_bc, a_bc, a)
                                logp = Categorical(logits=model.pi(h)).log_prob(a)
                                ibrl_bc_picks += int(take_bc.sum())
                                ibrl_steps += int(take_bc.numel())
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
                            obs[i] = _with_prompt(o) if rvs_cond else o
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
                if rvs is not None and not args.rvs_static_only and args.rvs_fresh_share > 0:
                    from zelda_rl.rvs_data import merge_capped, rollout_windows
                    fresh = rollout_windows(rbatch, rvs_idx, args.chunk, rvs_cond, args.rvs_win_weight)
                    rvs = merge_capped(rvs, fresh, args.rvs_buffer, args.rvs_fresh_share)
                bc_coef_now = 0.0 if bc is None else (
                    args.bc_coef if args.bc_coef_final is None
                    else coef_at(update, args.bc_coef, args.bc_coef_final, args.bc_coef_updates))
                stats = ppo_update_recurrent(model, opt, rbatch, chunk=args.chunk, epochs=args.epochs,
                                             chunks_per_mb=max(1, n_chunks // args.minibatches), clip=args.clip,
                                             ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                                             max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                                             kl_coef=(args.kl_coef if anchor is not None else 0.0),
                                             bc=bc, bc_coef=bc_coef_now,
                                             bc_adaptive_frac=args.bc_adaptive_frac,
                                             bc_teacher=bc_teacher, bc_latent_coef=args.bc_latent_coef,
                                             bc_latent_site=args.bc_latent_site,
                                             bc_latent_states=args.bc_latent_states,
                                             peer=peer, peer_opt=peer_opt,
                                             peer_bc_coef=args.peer_bc_coef,
                                             align_coef=args.align_coef, align_kind=args.align_kind,
                                             latent_mode=args.latent_mode, latent_proj=latent_proj,
                                             q_head=q_head, q_coef=args.q_coef,
                                             rvs=rvs, rvs_coef=args.rvs_coef)
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
                    # ---- BEHAVIOUR COUNTERS, one per behaviour the `behaviour` preset pays for.
                    # These are DENSE where a defeat is a 1-10% binary: on this task a graded signal showed an
                    # 8.7x improvement across six versions that the binary reported as six consecutive
                    # failures (AGENTS 34). They say within a handful of updates whether a term bites at all,
                    # long before boss damage can, and they are emitted on EVERY arm -- a term only "works" if
                    # its counter moves ABOVE what the untreated control already does for free.
                    "gohma/items_picked": m("items_picked"), "gohma/hearts_picked": m("hearts_picked"),
                    "gohma/item_approach_units": m("item_approach_units"),
                    "gohma/nuts_wasted": m("nuts_wasted"),
                    "gohma/nut_waste_frac": float(np.sum([e["nuts_wasted"] for e in eps]))
                                            / max(1.0, float(np.sum([e["nuts_thrown"] for e in eps]))),
                    # mean alignment over the steps where facing her can lead to something, and the fraction
                    # of those steps actually spent facing her. Denominator is boss_live_steps, never episode
                    # length: a ratio whose denominator moves with her ceiling phases is not comparable.
                    "gohma/facing_cos": float(np.sum([e["facing_cos_sum"] for e in eps]))
                                        / max(1.0, float(np.sum([e["boss_live_steps"] for e in eps]))),
                    "gohma/facing_frac": float(np.sum([e["facing_steps"] for e in eps]))
                                         / max(1.0, float(np.sum([e["boss_live_steps"] for e in eps]))),
                    "gohma/lock_frac": float(np.sum([e["lock_steps"] for e in eps]))
                                       / max(1.0, float(np.sum([e["length"] for e in eps]))),
                    "gohma/red_eye_seed_conv": float(np.sum([e["seed_stuns"] for e in eps]))
                                               / max(1.0, float(np.sum([e["red_eye_seeds"] for e in eps]))),
                    # behaviour 5, measured directly: seconds from her knockdown to Link inside sword range,
                    # over the windows he reached at all, plus the fraction of windows he reached.
                    "gohma/stun_to_sword_s": (float(np.sum([e["stun_to_sword_steps"] for e in eps]))
                                              / max(1.0, float(np.sum([e["stun_reached"] for e in eps])))
                                              / ccfg.steps_per_second),
                    "gohma/stun_reach_frac": float(np.sum([e["stun_reached"] for e in eps]))
                                             / max(1.0, float(np.sum([e["stun_windows"] for e in eps]))),
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
            def _f(key, w=5, dp=2):
                v = metrics.get(key)
                return f"{v:>{w}.{dp}f}" if isinstance(v, float) and v == v else f"{'-':>{w}}"
            beh_line = ("" if args.profile != "gohma" else
                        f" {_f('gohma/items_picked')} {_f('gohma/hearts_picked', 4)} {_f('gohma/grass_cut', 4)}"
                        f" {_f('gohma/item_approach_units', 5, 0)} {_pc('gohma/facing_frac')}"
                        f" {_f('gohma/facing_cos')} {_pc('gohma/lock_frac')} {_f('gohma/nuts_wasted')}"
                        f" {_pc('gohma/nut_waste_frac')} {_f('gohma/red_eye_seeds')}"
                        f" {_pc('gohma/red_eye_seed_conv')} {_f('gohma/stun_to_sword_s', 5, 1)}"
                        f" {_pc('gohma/stun_reach_frac')}")
            LOGGER.info(f"{update:>4} {global_step:>9,} {episode_count:>5} {metrics['combat/success_rate']:>5.0%}"
                        f"{band_line}{beh_line} "
                        f"{metrics['combat/kill_frac']:>6.0%} {metrics['combat/death_rate']:>6.0%} {metrics['combat/left_room_rate']:>5.0%} "
                        f"{metrics['combat/hp_lost_hearts']:>6.2f} {metrics['combat/damage_dealt']:>5.2f} {metrics['combat/lockons']:>5.2f} "
                        f"{metrics['combat/blocks']:>4.1f} {metrics['episode/return']:>7.2f} "
                        f"({metrics['episode/r_outcome']:>5.1f}/{metrics['episode/r_shaping']:>5.1f}) {stats['approx_kl']:>6.3f} "
                        f"{stats['entropy']:>5.2f} {metrics['perf/env_steps_per_sec']:>5.0f}"
                        + (f" klA {stats['kl_anchor']:>6.3f}" if stats.get("kl_anchor") == stats.get("kl_anchor")
                           and "kl_anchor" in stats else "")
                        + (f" {stats.get('bc_loss', float('nan')):>5.2f} {100 * stats.get('bc_acc', float('nan')):>4.0f}%"
                           f" {stats.get('bc_coef_applied', bc_coef_now):>6.4f}" if bc is not None else "")
                        + (f" {stats.get('bc_latent', float('nan')):>5.2f}" if bc_teacher is not None else "")
                        + (f" {stats.get('align', float('nan')):>6.3f} {100 * stats.get('peer_acc', float('nan')):>5.0f}%"
                           if peer is not None else "")
                        + (f" {100 * ibrl_bc_picks / max(1, ibrl_steps):>6.1f}% {stats.get('q_loss', float('nan')):>5.2f}"
                           if q_head is not None else "")
                        + (f" {stats.get('rvs_loss', float('nan')):>5.2f} {100 * stats.get('rvs_acc', float('nan')):>4.0f}%"
                           if rvs is not None else "")
                        + f"  top: {top}"
                        + (f"  sync {sync.get('sync_ranks')}r {sync.get('weight_hash')}" if sync else ""))
            if update % args.save_every == 0 or update == args.total_updates:
                ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "update": update, "global_step": global_step,
                      "episode_count": episode_count, "config": config}
                torch.save(ck, run_dir / "checkpoints" / f"update_{update:05d}.pt")
                torch.save(ck, latest)
                # Back it up to S3 AS IT IS WRITTEN. A pod can vanish between two updates, and on
                # 2026-10-15 one took fifteen arms' weights with it because syncing was a manual step.
                ckpt_s3.upload(run_dir / "checkpoints" / f"update_{update:05d}.pt", args.run_name, update)
                ckpt_s3.upload(latest, args.run_name, update)
    finally:
        for e in envs:
            e.close()
        log.close()


if __name__ == "__main__":
    main()
