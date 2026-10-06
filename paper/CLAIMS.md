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
| R6 | Reward simplification + action-space fix + engaged starts doubled Gohma's defeat rate | **M** | v10 3.29% → v11 7.0%, `results/1qk7isl2/` vs `results/p20rbwq9/` |
| R7 | Two of the actions the task needs did not exist in the action space | **M** | `("L","R")` Z+Shield and `("L","CL")` Z+Nut absent until 2026-09-26; shield registered on 17.4% of presses |
| R8 | A graded HP curriculum beats both a finisher-heavy and an inverted one | **M** | v11 +0.700/update but 1 full-HP win in 3,023; v12 +0.098; v13 0.64% full-HP, 21× v11 |
| R9 | The residual blocker is engagement DISTANCE, not boss HP | **M** | 10-slot suite, 80 eps, v14 u20: 18.75% (<200u, n=16) -> 4.17% (200-400u, n=24) -> 2.50% (>400u, n=40). `results/eval/v14_full_q_rank{0,1}.jsonl`. Earlier **I** was slot 8 alone |
| R10 | The spacing reward improved survival | **U → withdrawn** | `close_attack_steps` never left ~21%; the death-rate fall is not attributable to it |
| R11 | A KL anchor to the best known policy keeps a PPO fleet from drifting off it | **U** | `gohma-kl-v14`, `klA` steady at 0.012 over updates 15-24: the penalty is active and small. Says nothing yet about outcome |
| R12 | The KL anchor (or dropping the death penalty) improves Gohma's defeat rate | **U** | flat so far: fleet mean succ 8.4 +/- 1.5% against the v13 init's 7.4%. Two variables changed at once, so a positive result would not separate them |
| R13 | A per-episode (not per-round) start-slot assignment is required for a hold-out rate to mean anything | **M** | the round version sent 48/48 episodes of the first v13 hold-out to slot 1 and ran 0 episodes on slots 9-10 of the v14 full suite; both numbers withdrawn, PROVENANCE 3.2 |
| R14 | v14 generalises to unseen start states | **U -> cannot be measured on this curriculum** | held-out is by construction [1,2,9,10] = all 5 HP, all >=368u, so collinear with 'far'; trained slots at the same distance score the same. 0/32 is not a generalisation gap |
| R15 | Gohma was never given an approach reward in v1-v14 | **A** | `GohmaTracker.step` replaces `CombatTracker.step` with no `super()` call, so `w_approach` never fired; `approach_stunned` is gated on `GOHMA_STUNNED`. Read both implementations |
| R16 | An approach reward raises the defeat rate from distant starts | **U** | `gohma-kl-v15` launched 2026-10-05 from v14 u33, one variable changed. Judged on the by_distance bands of the held-out evaluator |
| R17 | A net-progress shaping term is not farmable by oscillation | **M** | `tests/test_gohma_approach.py`: 12 in/out round trips pay the single net gain, not 12x it; wandering after the best pays nothing more |

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
