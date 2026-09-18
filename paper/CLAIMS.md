# Claim audit

Every factual statement in [paper.md](paper.md) and [../README.md](../README.md), with what backs it.
A claim is only a result if its class is **M**. Update this file in the same commit as the claim.

- **M**: quantitatively measured, with a recoverable artifact (logs, manifest, dataset, checkpoint, code)
- **Q**: qualitatively observed (video, screenshot, author observation), no systematic measurement
- **I**: inferred interpretation of M/Q evidence
- **A**: confirmed by the author from first-hand knowledge of the experiment, not independently recoverable
- **U**: unsupported — needs evidence before it can be stated as fact
- **P**: proposed protocol, deliberately not yet run

Named sources: **COMBAT** = `docs/COMBAT_TASK.md` · **GOHMA** = `docs/GOHMA_TASK.md` · **DS** =
`datasets/transitions/deku-baba/20260918-134229/{dataset.json,manifest.jsonl}` · **RAM** = `docs/RAM_MAP.md` ·
**WM** = `docs/WORLD_MODEL.md`

## Model-free reference (§4)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 1 | Deku Baba task solved: final policy clears 10/10 start slots, 0 deaths | M | COMBAT per-slot table, 1 episode/slot, local Mac, sampled actions |
| 2 | 40.7 M authoritative-simulator steps to reach it | M | COMBAT; `global_step` in `deku-combat-gru-v2b_u00120.pt` |
| 3 | 77% success over fleet training episodes | M | COMBAT, wandb `zelda-rl-combat-ppo` run `vxejacj7` |
| 4 | Six slots cleared with no damage, including the two weakest in training (32% / 41%) | M | COMBAT per-slot table |
| 5 | Queen Gohma is not solved | A | Fleet was killed 2026-09-18 at update 11 of a 4000-update extension; no evaluation exists |
| 6 | The Deku Baba policy transfers as a warm start to Gohma | Q + A | GOHMA; `zelda_rl/transfer.py` widens the actor encoder. No ablation against from-scratch |
| 7 | Gohma observation v7 is 297 global dims, 90 actions | M | Checkpoint config of `gohma-v10_u00400.pt` |

## Structured state and encoding (§3)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 8 | State is read from live RAM in ~25 µs per step | M | `scripts/bench_latency.py`; figure quoted in `docs/FINDINGS.md` |
| 9 | RAM offsets are verified against live recordings, not guessed | M | RAM documents each offset with the trace that confirmed it; `docs/ADDING_OBSERVATIONS.md` gives the procedure |
| 10 | Dataset state vector is 335 dims = 15 Link + 16 × 20 actor | M | DS `dataset.json`: `state_dim`, `link_fields`, `enemy_fields` |
| 11 | Actor rows are permutation-stable within an episode | M | `EnemyRows` in `scripts/collect_transitions.py` assigns on first sight and never reassigns; verified by inspecting hp traces per row across an episode |
| 12 | No raw memory addresses appear in the encoding | M | Lock-on/Navi stored as row index; code review of `encode_state()` |
| 13 | 16 actor rows suffice; 0 actors were dropped in the released dataset | M | DS manifest `actors_dropped` sums to 0 across 168 episodes; start states hold at most 9 |
| 14 | Deku Baba action routine takes 11 mapped values | M | `DEKU_BABA_STATE` in `zelda_rl/combat.py`, decoded from a live fight recording |
| 15 | The decoded behaviour state is the field a policy most needs | I | Inferred from COMBAT's finding that the learned kill move is a timed Z+A against a telegraphed lunge; not measured by ablation |
| 16 | Action directions are world-frame, making the action–state pair one consistent frame | M | `zelda_rl/env.py` `_stick()` converts world angle by camera yaw; `action_frame: "world"` in checkpoint configs |
| 17 | Decision rate is 10 Hz (6 VI per step) | M | `vi_per_step` in DS `dataset.json` and every checkpoint config |

## Transition dataset (§5)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 18 | 109,817 transitions over 168 episodes | M | DS `dataset.json`: `transitions`, `episodes` |
| 19 | Per-checkpoint success/death/timeout and mean episode length (Table 2) | M | DS manifest, aggregated by `results/make_figures.py` |
| 20 | Success rises and episode length falls monotonically across the six tiers | M | Table 2; monotone in both columns G5→G150 |
| 21 | The final tier's 81% recovers the 77% from fleet training, so the replay harness is faithful | M + I | 81% is measured (DS); "faithful" is the inference. Different protocols — 36 episodes over 10 slots here vs. fleet-sampled episodes there — so this is agreement, not identity |
| 22 | All 10 start slots are covered | M | DS manifest distinct `slot` values |
| 23 | Dataset spans competence tiers rather than expert-only behaviour | M | Table 2 spans 27%→81% success |
| 24 | A world model trained only on expert data never sees the failure states a learning policy visits | I | Standard distribution-shift argument; not measured here |
| 25 | 17.6 min wall clock, ~104 steps/s aggregate, 10 instances, 8-core laptop | M | Collector's final summary line, `/tmp/collect_deku.log` |
| 26 | Independent episodes roughly double throughput vs. a lockstep batch at 10 instances | M | Measured both: lockstep ~50 steps/s, independent ~105–114 steps/s, same checkpoint and episode cap. Single paired comparison, not repeated |
| 27 | Episode `.npz` layout and that transition *t* = (state[t], action[t], state[t+1]) | M | DS `dataset.json` `layout`; array shapes `action[T]`, `state[T+1]` |

## The savestate trap (§5.4)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 28 | The first collection run evaluated the Deku Baba policy against Queen Gohma | M | Smoke-test output showed actor id `0x0028` (Boss_Goma) where `0x0055` (Deku Baba) was expected; live Dolphin slots dated 2026-09-14, the Gohma work |
| 29 | Every such episode ends as a timeout with zero targets, indistinguishable from a weak policy in aggregate | M | First smoke run: `targets_start: 0`, 100% timeout |
| 30 | The collector now refuses to run unless each start state holds a live target of the expected id | M | `check_slots()` in `scripts/collect_transitions.py`; verified it returns 1–3 for the Deku states and 0 for the Gohma states |
| 31 | This is the failure mode structured-state datasets are most exposed to | I | One incident. Stated as interpretation, not a general finding |

## Motivation (§1)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 32 | The 40.7 M-step run required a 20-pod fleet for ~6 hours | M | COMBAT version history: fleet start/stop timestamps 09:37→15:02 on 2026-09-13 |
| 33 | Fleet runs cost ≈ $10/hr at 20 CPU pods | A | Author's RunPod billing; no invoice committed |
| 34 | The usual motivation for pixel-based world models does not apply here | I | Follows from claim 8–9 (state is exact and cheap). An interpretation of scope, not a measurement |
| 35 | Enemy telegraphs must be answered within a few tenths of a second | M (partial) | GOHMA measured a Deku Nut stun landing 0.22 s after the throw and lasting ~2 s. The Baba lunge window is asserted, not timed |

## Behaviour-shaped latents (§1, §6) — the paper's central proposal

| # | Claim | Class | Evidence |
|---|---|---|---|
| 47 | A dynamics model over raw 335-d state spends capacity on quantities no policy reads | I | Argument from the encoding (§3.1): room props and underground actors occupy rows a policy's own observation largely ignores. Not measured — a per-field capacity ablation would be needed |
| 48 | The control-relevant fields are low-entropy and discrete, so a regression loss under-weights them | I | Follows from the encoding: behaviour state is 1-of-11, HP is a small integer, vs. continuous positions in the hundreds. Plausible but unquantified — measure per-field loss share before asserting |
| 49 | A frozen BC encoder cannot collapse the way jointly-learned self-predictive encoders can | I | Structural argument: `E` is fixed before `M` exists and was fit to a task a constant map cannot solve. Sound as reasoning, not an experimental result |
| 50 | A behaviour-shaped latent beats reconstruction-shaped and random latents at equal width | **P** | §7.2 ablation table. **Not run.** The comparison is the paper; it may well come out negative |
| 51 | Encoders shaped by more competent behaviour give better dynamics spaces | **P** | §6.1 competence axis. Enabled by the tiers in Table 2, not yet tested |
| 52 | Dreaming beats behaviour cloning alone | **P** | §6.4. The BC row is the one most likely to embarrass the method; 69% of dataset episodes already succeed |
| 53 | The encoder has blind spots inherited from the demonstrator | I → **P** | Argued in §2 and §8; §6.3's linear probe is designed to measure it. Currently reasoning only |
| 54 | The demonstrator policies are recurrent (GRU over the entity encoder) | M | `RecurrentEntityActorCritic`, `zelda_rl/ppo.py:129`; `recurrent: true` in every deku checkpoint config |
| 55 | The demonstrator's observation carried episode-memory fields absent from our structured state | M | `dataset.json["obs_fields"]` lists `time_left`, `mem_kills_frac`, `mem_since_last_kill`, `mem_lockons`, … ; `link_fields` does not |
| 56 | Therefore `a_t = π(s_t)` is not a well-posed cloning target and feedforward BC has a partial-observability ceiling | I | Follows from 54–55. The size of the ceiling is measured by the feedforward-vs-GRU ablation (§6.1), not yet run |
| 57 | The BC encoder reuses the architecture that solved the task model-free, not a new one | M | `BCNet.encode` mirrors `EntityActorCritic._features`: id embedding → shared per-entity MLP → masked mean+max pool → concat global → trunk. Differs only in trunk activation, widths, and a LayerNorm on the latent |
| 58 | `E` must be non-recurrent for the latent dynamics to be well-posed | I | Structural: a recurrent `E` makes `z_t` depend on history `M(z_t, a_t)` cannot observe. Reasoning, not measurement |

## Stage one, measured (§6) — the frozen encoder

Split is three-way by start slot: train {1,2,3,5,6,10}, threshold-tuning {4,8}, test {7,9}. All figures are
test-slot; all operating points were chosen on the tuning slots. Source: `checkpoints/bc/bc-decode-final.pt`
(`metrics` key) and `results/event_detection.json`, both regenerated by `scripts/train_bc_decode.py` +
`scripts/eval_decode_events.py`.

| # | Claim | Class | Evidence |
|---|---|---|---|
| 59 | Per-row identity survives masked mean+max pooling: row HP 0.908, row position MAE 7.2 units, Baba behaviour state 0.999 | M | Table 4; baselines 0.554 / 250.2 / 0.669 |
| 60 | Lock-on target row and the shield/damaged state bits reconstruct at 1.000 | M | Table 4 |
| 61 | HP is discrete, so classification beats regression: damage-event F1 0.286 -> 0.458 on the same split | M | Recorded before/after; the regression variant rounded a regressed HP |
| 62 | Baba HP never rises (0 of 891,459 present row-steps); Link healed once in 109,649 | M | Direct count over the released dataset |
| 63 | A per-episode running minimum on decoded HP raises damage F1 0.458 -> 0.746 and cuts false positives 63 -> 14 | M | Measured on the two-way split; it also costs recall by entrenching premature decreases |
| 64 | Light oversampling of HP-transition frames helps (5%) and heavy oversampling hurts (15%) | M | Ablation: damage F1 0.833 at 5%, 0.719 at 15%, 0.794 at 0% |
| 65 | Cross-head consistency (HP=0 implies dying or despawned) helps | M | Ablation at oversample 0: damage F1 0.794 -> 0.873, kill 0.756 -> 0.842 |
| 66 | Calibrating the detector and tuning its threshold was worth more than any architectural change | M + I | The gain is measured; "more than any architectural change" is a comparison across our own ablations, not a general claim |
| 67 | Reward-critical events are recoverable at AP 0.887 (damage), 0.895 (kill), 0.827 (Link damage) | M | Table 5. AP is threshold-free; F1 depends on an operating point that transferred poorly for Link damage |
| 68 | Gating damage on decoded `hit`-state entry does not help on its own | M | AP 0.523 vs 0.887 for HP-drop alone; only the sum combination beat HP on tuned F1 (0.862 vs 0.833) |
| 69 | Despawn corroboration adds nothing to kill detection | M | 0.944 F1 / 0.895 AP with and without, despite 164/164 true kills despawning |
| 70 | Tuning the threshold on the reported split inflates results | M | Link damage F1 was 0.898 tuned-on-test vs 0.629 with an honest tune split |
| 71 | These figures are an upper bound for the dynamics model | I | They autoencode `s_t` through `z_t`; §7 decodes from a predicted `ẑ_{t+1}`, which cannot be better |
| 72 | Reconstruction pressure costs action accuracy | M | Action top-1 0.057 (action-only) -> 0.049 (joint). Both far under the 0.220 ceiling |
| 73 | Test event counts are small (28 damage, 19 kill, 24 Link damage) | M | `results/event_detection.json`; differences of a few points are noise |

## Not claimed

| # | Statement | Status |
|---|---|---|
| 36 | Structured state is *better* than pixels for world models | **Not claimed** — no pixel baseline is run (§8) |
| 37 | The dynamics-model or policy architecture is novel | **Not claimed** |
| 38 | Any dream-trained policy works | **P** — §6 is a protocol; no dynamics model exists yet |
| 39 | Results generalise beyond OoT | **Not claimed** (§8) |
| 40 | One dynamics model covers both enemies | **P** — §7 condition 2, not run |
| 41 | A policy dreamed on Deku Baba transfers to Gohma | **P** — §7 condition 3. We predict failure and will report which fields break first |
| 42 | Sample efficiency beats the model-free reference | **P** — the comparison in Table 3 is the point of the study, and its denominator (40.7 M) is claim 2 |

## Open items that would change the paper

| # | Item | Why it matters |
|---|---|---|
| 43 | Behaviour-cloning baseline on the same dataset | **In progress.** 69% of dataset episodes succeed; without this row a dream-trained policy's competence cannot be separated from imitation. Trained by `scripts/train_bc.py`, measured in the emulator by `scripts/eval_bc.py` — action accuracy is not success, and only the emulator number goes in Table 3 |
| 44 | Held-out-by-slot split, fixed and committed | §6.2 claims generalisation across scenarios; the split must exist before any model is trained on it |
| 45 | Timing of the Baba lunge window | Claim 35 is partly unsupported; a trace measurement would close it |
| 46 | Gohma model-free reference | Without it, §7 has no ceiling to compare dreamed boss policies against |
