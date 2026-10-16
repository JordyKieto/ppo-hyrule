# Claim audit

One row per claim a paper could make, with the evidence class and the artefact it rests on. Classes are defined
in `../PROVENANCE.md`. **Nothing above U goes in a paper; nothing at U goes in without being closed first.**

The purpose of this file is that in past projects the numbers survived and the *provenance* did not, so a claim
could not be defended six months later. Fill a row when the measurement happens, not when the paper is written.

## RL track

| # | Claim | Class | Evidence |
|---|---|---|---|
| R1 | PPO learns OoT Deku Baba combat from RAM state alone, no pixels | **M** | `results/vxejacj7/` (124 updates), 78% fleet success at G120–150 |
| R2 | The solved policy clears all 10 start states | **Q** | 1 watched episode per slot at G150, `docs/COMBAT_TASK.md`. A rate needs a real eval |
| R3 | It cost $51 and 4 h 52 m on 20 CPU pods, ≈$1.3 per million env steps | **M** | wandb timestamps + fleet kill log, `PROVENANCE.md` §3.1 |
| R4 | An entity-factored policy with a GRU beats a flat MLP here | **U** | never run head-to-head. `ActorCritic` exists but was not ablated at scale |
| R5 | Queen Gohma is not solved | **M** | best full-HP defeat rate 0.64%, `results/kj9ypr7w/` |
| R6 | Reward simplification + action-space fix + engaged starts doubled Gohma's defeat rate | **I -- CONFOUNDED** | v10 3.29% -> v11 7.0% is pooled success, and v11 changed the START MIX as well as the reward (50% weight on slot 7, the 38-unit start). Per-slot, v11 improved slot 7 only. The reward changes may still be right; this number does not show it. R18/R19 |
| R7 | Two of the actions the task needs did not exist in the action space | **M** | `("L","R")` Z+Shield and `("L","CL")` Z+Nut absent until 2026-09-26; shield registered on 17.4% of presses |
| R8 | A graded HP curriculum beats both a finisher-heavy and an inverted one | **I -- v11 figure withdrawn** | the "v11 +0.700/update" half is a curriculum artefact: v11's climb was slot 7 alone (7.32->14.00%), nine slots at exactly 0.00% for 37 updates. v13's 0.64% full-HP stands. PROVENANCE 3.2 |
| R9 | The residual blocker is engagement DISTANCE, not boss HP | **M** | 10-slot suite, 80 eps, v14 u20: 18.75% (<200u, n=16) -> 4.17% (200-400u, n=24) -> 2.50% (>400u, n=40). `results/eval/v14_full_q_rank{0,1}.jsonl`. Earlier **I** was slot 8 alone |
| R10 | The spacing reward improved survival | **U → withdrawn** | `close_attack_steps` never left ~21%; the death-rate fall is not attributable to it |
| R11 | A KL anchor to the best known policy keeps a PPO fleet from drifting off it | **U** | `gohma-kl-v14`, `klA` steady at 0.012 over updates 15-24: the penalty is active and small. Says nothing yet about outcome |
| R12 | The KL anchor (or dropping the death penalty) improves Gohma's defeat rate | **U** | flat so far: fleet mean succ 8.4 +/- 1.5% against the v13 init's 7.4%. Two variables changed at once, so a positive result would not separate them |
| R13 | A per-episode (not per-round) start-slot assignment is required for a hold-out rate to mean anything | **M** | the round version sent 48/48 episodes of the first v13 hold-out to slot 1 and ran 0 episodes on slots 9-10 of the v14 full suite; both numbers withdrawn, PROVENANCE 3.2 |
| R14 | v14 generalises to unseen start states | **U -> cannot be measured on this curriculum** | held-out is by construction [1,2,9,10] = all 5 HP, all >=368u, so collinear with 'far'; trained slots at the same distance score the same. 0/32 is not a generalisation gap |
| R15 | Gohma was never given an approach reward in v1-v14 | **A** | `GohmaTracker.step` replaces `CombatTracker.step` with no `super()` call, so `w_approach` never fired; `approach_stunned` is gated on `GOHMA_STUNNED`. Read both implementations |
| R16 | An approach reward raises the defeat rate from distant starts | **U -- tested, NOT shown** | `gohma-kl-v15`, stopped at update 40. 9/160 vs v14's 18/559, one-sided Fisher p=0.122; every band overlaps. Direction consistent in all 4 bands and the near band moved least, which is the predicted asymmetry. ~400 far-band episodes needed for p~0.04; 72 collected |
| R17 | A net-progress shaping term is not farmable by oscillation | **M** | `tests/test_gohma_approach.py`: 12 in/out round trips pay the single net gain, not 12x it; wandering after the best pays nothing more |
| R18 | Pooled success rate cannot compare policies across curricula | **M** | identical per-slot skill under two mixes gives pooled rates 3.89 points apart and the same `success_balanced` to 1e-9; `tests/test_band_metrics.py`. Three version decisions were misread from it |
| R19 | v11's apparent momentum was one start slot, not learning | **M** | slot 7 38u: 7.32->14.00%, slope +0.298; slots 3/6/2/8/5/1/4 at exactly 0.00% across all 37 updates. W&B `by_slot/s*/success_rate` |
| R20 | v15 is the broadest policy to date | **M** | non-zero on 6/6 trained slots vs v11's 1/10; ~3.3% vs ~0.13% off the easy slot. It reads as the flattest run on pooled success |
| R21 | More training time was never the missing ingredient | **M** | `gohma-v10` 399 updates -> 1.21%; `gohma-v8` 129 updates from scratch -> 0.1%. Version trend table, PROVENANCE 3.2 |
| R22 | The held-out checkpoint selector could not distinguish checkpoints | **M** | ranked on 16-episode subsets at a 1.68% pooled rate, where 6.25% is a 20.9% draw from the null; selected 2/32 vs others 5/384, p=0.094. Needs ~256 episodes/checkpoint, 16x what it had |
| R23 | g_approach caused the plateau | **I -- CONFOUNDED** | the dose-response (v14 +0.012, v15 +0.004, v16 -0.017) cannot be separated from saturation: the slope tracks the STARTING level, and v18 ran v14's reward on v15's weights without recovering v14's slope. Not exonerated, not shown |
| R24 | Deficit-weighted sampling improves the unsolved starts | **U -> NEGATIVE** | v17 boss damage slope -0.0236 (t=-3.78) on the mix-invariant per-slot form, so not a mix artefact. Stopped |
| R25 | Gohma training improved steadily from v10 to v15 | **M** | boss damage 0.38 -> 3.32 of 10, every version's slope distinguishable from zero; return -12.6 -> +56.8. It was reported as six plateaus because the metric was a 1-10% binary. PROVENANCE 3.2 |
| R26 | A sparse binary cannot steer this task | **M** | the defeat rate called v10-v15 flat while boss damage rose 8.7x with |t|>2 at every step; the term added on the strength of the binary (g_approach) measurably reversed the damage slope. AGENTS 34 |
| R27 | v14's reward is the configuration that learns | **U** | `gohma-kl-v18`, v14's reward (no g_approach) on v15's weights. Judged on boss_damage_balanced slope against v14's +0.012 HP/update |
| R28 | Gohma boss damage saturates at 3.2-3.4 of 10 | **M** | eight versions, five reward configurations, every run starting near 3.3 fails to climb while every run starting near 2.6 does. The last six HP are the open problem |
| R29 | Outcome-only reward with the escalating damage multiplier | **U -- built, unrun** | `--reward damage` / `GOHMA_DAMAGE`, 39 shaping terms zeroed. Init not yet chosen; see GOHMA_FOCUS WHERE WE LEFT OFF |
| R30 | The credit horizon never matched the task | **M** | gamma 0.995 / gae_lambda 0.95 identical in all 19 exported runs; advantage horizon 1.8 s vs a 45-154 s fight; boss_defeat worth 0.04% of face value at episode start. GOHMA_CREDIT_PLAN 1.1 |
| R31 | Network capacity never grew with the problem | **M** | obs 483->1033, actions 63->108, params +10%, params-per-obs-dim halved 1,275 -> 658. hidden 384 restores 1,360 |
| R32 | A longer credit horizon raises boss damage | **U** | Phase 1 of GOHMA_CREDIT_PLAN, 3 arms incl. a control sharing the same init, gated on slope > +0.012 HP/update with \|t\|>2 |
| R33 | A long credit horizon needs a lower learning rate | **M** | 12-cell grid: at whole-episode credit, lr 3e-4 -> 3e-5 moves damage 0.76 -> 2.08 (2.7x); at 30s 1.21 -> 2.30; at v15's 1.8s horizon LR does nothing. GOHMA_CREDIT_PLAN 9.1 |
| R34 | The 9s/3e-5 cell beats v15's settings | **M -> WITHDRAWN** | grid said +32% (3.16 vs 2.40) on 1 pod per arm; matched confirmation, 7 vs 6 pods from one init, gives +0.065 +/- 0.117, t=+0.56. Winner's curse. AGENTS 35 |
| R35 | A 20s-horizon policy can be switched to a long horizon | **M -> NO** | tried at vf_coef 0.05 and 0.5; both destroyed v15 (3.275 -> 1.84, t=-3.30) with the critic never converging |
| R36 | 53% of the Gohma observation is dead weight | **M** | obs 1033 -> 481 (n_actors 16 -> 4) is within noise of the control across 22-36 updates per arm, with v15's weights loaded unchanged (missing=0). GOHMA_CREDIT_PLAN 10 |
| R37 | Restoring Deku's params-per-input ratio unlocks learning | **M -> NO** | n4 sits AT that ratio (1,413 vs Deku's 1,275) and is the worst-trending arm, slope -0.0249 t=-2.65. Best vs control +0.193 +/- 0.104, t=+1.85, not significant |
| R38 | The 120s episode cap and removed approach reward cost ~0.7 damage | **I** | every arm of every sweep on 2026-10-15, control included, sits below v15's 3.275 start |
| R39 | A human can beat Queen Gohma repeatedly and it can be recorded with inputs | **M** | 13/13 wins, 10,532 steps, 8 start states, raw pad + decoded action + camera yaw per frame; md5s in `datasets/human/gohma/MANIFEST.json`, gated by `scripts/verify_human_dataset.py`. PROVENANCE 3.3 |
| R40 | The recorded human action decoding is the exact inverse of what the env sends | **M** | `tests/test_record_human.py` round-trips every emittable action through the env's stick transform at 5 camera yaws; `--selftest` proves the pad offset through the pipe, 11 cases. An earlier version ignored the camera and was wrong for every non-neutral direction |
| R41 | Behaviour cloning learns the human's action choices | **M**, n=1 | 53.4% held-out action accuracy vs an 11.1% majority floor, 108-way; split by episode across slots. Single run, 2 held-out episodes (AGENTS 7) |
| R42 | The BC clone can fight Gohma | **M -> NO** | 80 held-out episodes, 8 per start on all 10 starts: **0 defeats**, 83.8% deaths, boss HP taken **0.14/10**, mean episode 41s. Against v15's 3.275 and v10's 0.38 start. `runs/eval_bc_baseline.jsonl` |
| R46 | Frame-level imitation accuracy predicts closed-loop competence | **M -> NO** | 53.4% held-out action accuracy (4.8x the majority floor) yields 1.4% of her HP and 0/80 wins. v15, which does beat her, agrees with the human on only 3% of frames |
| R47 | A constant BC coefficient of 0.1 degrades v15 | **M** | 12 ranks x 8 updates: damage 2.63 -> 2.03, slope -0.0857 +/- 0.0172, t=-4.98; 4-rank replicate -0.134 +/- 0.089 agrees in direction. bcA 3% -> 23% confirms the term was working |
| R43 | A BC loss on demonstrations regularises PPO toward them | **M** for the mechanism | `tests/test_bc_regulariser.py`: coef 1.0 drives agreement with the demonstrated action to 96.9% from an 11.1% chance floor; coef 0.0 reports no stats and reaches 0% |
| R44 | BC init + BC regularisation beats v15 | **M -> NO** | arm stopped after 6 updates x 12 ranks: boss damage 0.21 -> 0.35, rising but from 1/15th of v15's 3.275 and below from-scratch v10's 0.38 start. ~100 updates to reach where v15 already is |
| R45 | A BC loss on human wins improves our BEST policy (v15) | **M -> NO so far** | constant 0.1 declines (t=-4.98); decaying 0.1->0.02 is not distinguishable from it on the matched window (t=+0.65), declines in its own second half (t=-2.05) and in the replicate (t=-3.37). 0 of 26 updates above v15's 3.275. Filter / adaptive / latent variants untested |
| R49 | IBRL's action proposal helps on-policy PPO | **U -- instrument broken** | bcPick 54-78% with q_loss 2.5-10: the Q head is untrained when it starts gating, so the policy executes the clone (0.14 HP) on most steps and fell from 2.3 to 0.38 in 9 updates. Needs the proposal gated on a fitted Q before any claim |
| R60 | Oversampling decision moments raises BC's online damage | **M** | monotone at matched n=80: scale 1.5/6/12 -> 0.16/0.33/0.53 boss damage. Best stratified clone 0.53 vs 0.33 unstratified and 0.14 for the 13-game clone |
| R61 | Dropping the larva class is worth quadrupling the scale | **M** | `nolarva` at scale 3 scores 0.49 where scale 3 with larva sits at 0.26-0.33; the class correlates -0.598 with damage in the corpus |
| R62 | The seed-stun chain alone is sufficient for BC | **M -> NO** | `chainonly` (no sword, no larva) scores 0.34 against `nolarva`'s 0.49 at the same scale. The sword class carries real signal |
| R63 | Validation accuracy predicts a BC clone's online damage | **M -> NO** | r = -0.11 across five arms; the highest-accuracy arm (73.1%) is middling on damage and the best-damage arm is nearly the lowest-accuracy (71.8%) |
| R64 | Stratified BC can beat Gohma | **U, and the extrapolation is not credible** | 0 defeats in 400 episodes; 0.53 is 16% of v15 and 5.3% of a kill. A log-log fit needs scale ~4,022 for a kill, but the weighting saturates past ~60 where positioning already contributes nothing |
| R65 | Adding an RvS loss to PPO does nothing at any coefficient | **M** | 23 cells from one v15 init, matched updates 1–11: control (c=0, 5 arms) 2.512 ± sd 0.380; c=0.25 −0.112 ± 0.197; c=0.5 −0.160 ± 0.188; c=1.0 +0.056 ± 0.196. All \|t\| < 1 |
| R66 | Five replicates of ONE PPO config span 2.00–2.97 boss damage (sd 0.380) | **M** | the five c=0 arms of the RvS sweep, 11 matched updates. This spread is the floor for every cell-level claim in that grid |
| R67 | Continuing PPO from v15 under the simple reward LOSES boss damage | **M** | all five controls at 60 updates: slopes −0.0053, −0.0095, −0.0162, −0.0176, −0.0360 per update; three at \|t\| > 3.7; ending 1.33–2.27 vs v15's 3.275 |
| R68 | Pressing the BC clone's ACTIONS onto a strong policy's own online states harms it | **M** | `gohma-la` arm B (latent align + KL to the clone, coef 0.3): two of three seeds −0.107 (t=−5.50) and −0.110 (t=−4.83), landing ~1.1 vs the control's 1.53–2.67 |
| R69 | Aligning only the REPRESENTATION on online states is neutral | **M (null)** | arm A, 3 seeds: +0.032 (t=0.82), −0.066 (t=−1.64), +0.123 (t=0.74) — indistinguishable from the control on every seed |
| R70 | Pure RvS (no PPO) on human wins + PPO losses + PPO wins reaches 1.60 boss damage | **M** | pooled over 5 online evals, 120 episodes: 1.60 ± 0.11, vs the plain clone's 0.33 and the stratified clone's 0.53 on the same 100 games |
| R71 | Adding PPO wins to the human win class HELPS | **M** | M5 1.60 ± 0.11 vs M2 (human wins only in the class) 0.93 ± 0.17 → 0.67 ± 0.20, t = 3.3, pooled over 120 episodes each |
| R72 | Training RvS on wins ALONE is the worst corpus | **M** | M1 (100 human wins, no losses) pooled 0.30 ± 0.04 against 0.75–1.60 for every mix containing losses. Nothing labels the fatal action |
| R73 | A 24-episode online eval cannot rank these arms | **M** | per-eval swing ±0.75 exceeds the whole between-arm spread; every one of the five mixes was the maximum at some eval. Four single-eval rankings were quoted and one retraction was itself wrong (AGENTS §38) |
| R75 | **Pure return-conditioned BC defeats Gohma** — 8/240 episodes (3.3%), no PPO | **M** | `rvs_M5`, ten 24-episode online evals, `results/rvs_baselines.json`. Both BC clones of the same games: 0/400 (R64). Other mixes 0.4–1.0% |
| R76 | Pure RvS reaches 1.76 ± 0.11 boss damage but PLATEAUS there | **M** | ten evals, slope +0.0145 ± 0.0091/epoch, t=+1.60 — flat at the \|t\|>2 bar. ~half of v15's 3.275 |
| R77 | More PPO wins is not monotonically better: 200 beats all 495 | **M** | `rvs_M5` (200 wins) 1.76 ± 0.11 vs sweep cell `w495_s0.3` 1.42 ± 0.08 at the same win share, t=+2.43 |
| R78 | Dyna's refresh enriches the win class far too slowly to matter | **M (mechanical)** | `dyna_M5_s0`: 300 → 305 wins over five refreshes (+1.7%) while absorbing 120 of its own episodes. The earlier form of this row said the count "had not moved"; it moves, at 4–8% of absorbed episodes |
| R79 | Absorbing the policy's own episodes raises pure-RvS damage | **M → NO** | matched at equal budget, `dyna_M5_s0` 1.60 and 5/120 kills vs static `rvs_M5` 1.60 and 4/120 over its own first five evals. Own slopes flat: damage t=−0.15, kills t=0.00. Retracted en route: "all three Dyna arms are rising" was 3-eval fits against converged arms |
| R87 | Corpus size 800 → 1,800 at fixed composition: no detectable effect | **M (null)** | 1.47 ± 0.08 / 1.39 ± 0.09 / 1.64 ± 0.11, non-monotone; 1800−800 = +0.17 ± 0.14, t=1.2. Spread 0.25 against a 0.43 seed floor — one seed per size cannot resolve it |
| R88 | 10× the episodes at a diluted win fraction is worse than 2× at fixed composition | **M** | spout `la4` 7,729 eps at 6.1% wins → 1.25, against 1,800 eps at 37.5% wins → 1.64 |
| R89 | The Dyna/spout design has a hard corpus ceiling at ~8,000 episodes | **M (mechanical)** | `la3` OOM-killed at 8,345 eps: `load_all` holds ~2.9 MB of observations per episode and reloads the whole corpus per refresh, ~24 GB against the pod's 32 GB `memory.max` (`free` reported 503 GB) |
| R85 | No learning rate makes PPO improve the RvS policy | **M** | 5 LRs over 100×, matched over updates 1–6: best vs runner-up +0.09 ± 0.29, t=+0.30, a tie. The three lowest are flat (t=+0.5 to +1.2); the best arm is a 3e-4 seed flat at 1.90 |
| R86 | The PPO collapse is seed-dependent, not LR-dependent | **M** | two identical 3e-4 arms, 37 updates each: one t=−5.0 (2.10 → 0.89), one flat (1.79 → 1.81) |
| R83 | PPO initialised from the RvS policy holds its damage and its kill rate | **M → NO** | 7 updates: 1.96 ± 0.17 / 2.87% and 1.86 ± 0.18 / 5.03%, above the 1.76 it started from. PPO from v15 at the same config declined (R67). Flat slopes so far |
| R84 | Nothing else in the RvS phase is improving | **M** | 14 arms with slopes ± SE; the only significant ones are two low-level arms rising (1.15 at t=3.7, 0.86 at t=2.2) and one falling (`dyna_M5_s0`, t=−2.4). Two Dyna seeds at identical settings have opposite significant slopes |
| R81 | **Offline metrics INVERT against online damage across RvS corpora** | **M** | 14 arms: val WIN acc r=**−0.875** (t=−6.25), val acc −0.599, train acc −0.626, train loss **+0.624**. The best-fitting arms have the easiest corpus (`rvs_M3` 56.7% acc / 0.82 dmg vs `rvs_M5` 27.4% / 1.76). Sharpens R63's −0.11 |
| R82 | Within one arm, accuracy DOES track its own damage, weakly | **M** | mean r=+0.441 ± 0.097 over 14 arms for val WIN acc, +0.417 ± 0.086 for val acc. So accuracy is a progress signal and an anti-selector at the same time |
| R80 | The standout checkpoints are recoverable by md5 with their measured performance | **M** | `artifacts/MANIFEST.json`: `rvs_M5_s5.pt` b4aafda1… 1.76 ± 0.11 / 8-240 kills, plus both Dyna M5 seeds, M4 seed 4 and every `_bestdmg`, each with its `s3://zelda-rl/fleet/ckpt_auto/` key |
| R74 | The win-share / PPO-win-count shape of the M5 mix matters | **U — in flight** | 9 cells, `ppo_wins ∈ {0,100,495} × win_share ∈ {0.15,0.3,0.5}`, human wins + ~500 losses in every cell. No verdict until the top cell has 3+ pooled evals |
| R55 | Damage requires a stun, and `gohma_vulnerable` is the signal that identifies it | **M** | 678/678 recorded hits had `gohma_stunned` within 2 s; only 9/678 (1.3%) had `gohma_vulnerable` set. The vulnerable flag identifies almost no real damage event |
| R56 | The human's slingshot lands 87% of shots, and the productive shot is the red-eye one | **M** | 299 fires / 260 impacts; `ceiling_eye_red` is 52% of all shots at 92.9% land; `ceiling_wait` lands 7.1%. Aim-to-eye cosine 0.607 (hit) vs 0.270 (miss); first-person aim 0.115 vs 0.359, i.e. associated with MISSES |
| R57 | A seed impact is worth damage only in certain states | **M** | 371 impacts: stun follows 100% in fall/land/stunned, 95.2% floor_prepare, 40.6% ceiling_eye_red, and **0% across 71 impacts in ceiling_idle/ceiling_move**. 0.70 of 10 HP comes off in the 3.5 s after an impact |
| R58 | Larva kills and ceiling time help | **M -> NO, both hurt** | per-episode correlation with damage dealt: larva hits **-0.598**, time she is on the ceiling **-0.559** (n=100) |
| R59 | The recorded reward scales with damage | **M -> NO** | human return ~= +253.1 + 8.19 x damage, so ~80% is damage-independent; a 10-HP kill earns only 13% more than a 5-7 HP episode (+335 vs +295) |
| R53 | More human demonstrations make the clone win more | **M -> WEAKLY, and not on the task** | 13 -> 100 games: accuracy 53.4% -> 74.5%, damage 0.14 -> 0.33, defeats 0/80 -> 1/80. On FULL-HP starts both are 0/24 at 0.12 damage. The gain is entirely in the <200u band (0.46 -> 1.08); mid and far are 0.00 for both. `results/eval/bc_human_100_online.jsonl` |
| R54 | The human corpus is auditable and versioned | **M** | v1 digest `5663643e97f8`, 13 episodes; v2 digest `e57da6027a30`, 100 episodes, v1 a prefix of v2. Per-episode md5, slot, start state/HP/distance, outcome, recorded_utc and recorder commit. `scripts/dataset_manifest.py`, mirrored to S3 |
| R51 | Fifteen mechanisms for using human demonstrations all fail to beat the policy trained without them | **M** | v15 alone = 3.275. Best arm means: FitNets 2.87, transition 2.79, mutual 2.66, one-way latent 2.47, decay 2.32, constant 2.31, IBRL 0.38. Three arms produced ONE update above 3.275; none produced a trend. PROVENANCE 3.3 |
| R52 | The 2026-10-15 arms' checkpoints are recoverable | **M -> NO** | the whole 34-pod fleet vanished at ~20:14 with `fleet sync` never run on any arm. Only W&B histories survive (11 groups exported). No weights from any arm exist |
| R50 | Latent alignment preserves v15 where action-level BC does not | **I** | FitNets (mean 2.87, max 3.37) and transition-matching (2.79, 3.21) are flat at v15's level over 14 updates each, where every action-level arm declined. Suggestive, not measured: neither slope is distinguishable from zero and neither exceeds 3.275 on a trend |
| R48 | The decay beats a constant coefficient | **M -> NO** | matched u1-8: +0.0256 +/- 0.0396, t=+0.65. First reported as t=+1.92 from an unmatched window (decay u18-25 vs constant u1-8) -- retracted, see PROVENANCE 3.3 |

## WM track

| # | Claim | Class | Evidence |
|---|---|---|---|
| W1 | Six latent world-model architectures lose to the identity baseline | **M** | `docs/LATENT_WORLD_MODEL.md` §1–30 |
| W2 | Validation loss and rollout skill are anti-correlated here | **M** | val loss fell 1.8× while skill went +0.0026 → −0.41, `results/c3zurq8g/`; and the LR sweep |
| W3 | A teacher-forced pretrain plus an autoregressive fine-tune is positive to H5 and no further | **M** | `results/gymco2ix/`, `docs/WM_EXPERIMENT_REGIME.md` |
| W4 | At H=1800 the model is far worse than assuming nothing moves | **M** (thin) | skill −2.44; but only ~31 k of 52 M starts qualify and all are timeouts |
| W5 | A 1.35 M-param MLP matches or beats both token models at one step | **M** | +0.393 vs JEPA +0.379 best / token WM +0.188. Different split, so not strictly matched |
| W6 | The objective matters more than the architecture: L1 vs MSE | **M** | same net/data/epochs, aggregate skill −0.379 → +0.392 |
| W7 | Link's motion is not action-conditioned at one step | **M** | action ablation +0.002; and the action/displacement cosine is 0.006 at lag 1 |
| W8 | 43% of states are action-INSENSITIVE, which is why the average looks flat | **M** | 150 counterfactual pairs, `samples/wm/counterfactual_excerpt.npz` |
| W9 | A change gate fixes sparse regression heads | **M** (small) | `a_yaw` −3.71 → +0.00 and 3 others, full scale; but it cost position skill |
| W10 | Up-weighting the motion loss ×10 improves motion | **U** | −0.193 → −0.079, 3 epochs, one seed. The only thing that has ever moved it |
| W11 | Nav-grid resolution is the motion bottleneck | **U → doubtful** | +0.068 small scale, −0.012 at full scale; and the cell is 28.8×57.1, not the 142×66 I first published |
| W12 | The token WM's autoregressive rollout diverges by ~429 world units in a few hundred steps | **M** | `scripts/wm_replay.py`, reproducible per checkpoint |

## Method claims that need a control before use

| Claim | Missing control |
|---|---|
| any "X improved Y" on the WM | seed repeats — six seeds of one config once spanned 1.38×–3.43× on this project |
| any architecture comparison | matched epochs AND matched data subset; two of this session's contradictions were subset effects |
| any action-conditioning number | must be scored at lag ≥ 2 on the action-sensitive subset, not on all states |
| any arm ranked by online eval | the POOLED mean over every eval with its SE, and a gap wider than the combined SE. A single 24-episode eval swings ±0.75 (R73) |
