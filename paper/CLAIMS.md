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
| R45 | A BC loss on human wins improves our BEST policy (v15) | **U** | `gohma-v20-v15bc` launch 20261015175011, 12 ranks + `gohma-v15bc` 4 ranks as a second replicate. Read dmgB against v15's 3.275 with bcA beside it |

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
