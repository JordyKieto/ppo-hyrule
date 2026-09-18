# Predict What Control Needs: Behaviour-Shaped Latents for World Models in a Commercial 3D Game

**Status: draft.** Sections 1–6 describe work that exists and is measured — including the frozen
behaviour-shaped encoder and what it retains (§6). Sections 7–8 are a *proposed protocol*: the dynamics
model and dream-trained policies are not built yet. Every quantitative statement is audited in
[CLAIMS.md](CLAIMS.md) with an evidence class; nothing below should be read as a result unless CLAIMS.md
marks it **M** (measured).

---

## Abstract

We study model-based reinforcement learning for melee and boss combat in *The Legend of Zelda: Ocarina of
Time* (OoT), running under an unmodified Dolphin emulator. Because the emulator's RAM is directly readable,
we work from **structured game state** — actor positions, velocities, health, lock-on targets, inventory,
and each enemy's *decoded action routine* — rather than pixels. This removes perception from the problem
and leaves dynamics, but it does not make the problem small: the state is a variable set of actors over
twenty fields each, and a model that predicts it faithfully is a model that reconstructs the game.

Our central claim is that this is the wrong target. **The easiest useful future to predict is not raw next
state, but state compressed through a representation already shaped by competent behaviour.** We train a
behaviour-cloning policy on trajectories from agents of varying skill, freeze its encoder `E`, and learn
dynamics in that space: `ẑ_{t+1} = M(E(s_t), a_t)` against the target `E(s_{t+1})`, with auxiliary heads
retaining the few raw quantities that policy optimisation needs (reward, terminals, health). The encoder
discards what competent play never depended on, so the dynamics model is not charged for reconstructing it.

This makes the representation itself the experimental variable. We can hold dynamics capacity fixed and
vary only the latent — behaviour-shaped, reconstruction-shaped (an autoencoder at equal width), or random
projection — and ask which supports policies that transfer. Because our dataset spans six competence tiers
of a single training run, we can further ask whether encoders shaped by *more competent* behaviour yield
better dynamics spaces. Policies are optimised entirely inside the learned model and evaluated in the
emulator, which is used only as an authoritative judge and never for policy gradients. A model-free PPO
agent that clears the task from 10/10 start states after 40.7 M environment steps sets the reference such
a policy must be measured against.

## 1. Introduction

Commercial 3D games are a demanding testbed for embodied RL: the dynamics are authored rather than
physical, the state is partially observable, and reward is sparse and tied to scripted enemy behaviour.
OoT combat in particular requires a policy to read an enemy's *animation state* — a Deku Baba telegraphs a
lunge, Queen Gohma's eye turns red before she is vulnerable — and to respond within a window of a few
tenths of a second.

Two facts shape our approach.

First, **the emulator is slow and serial.** One environment step costs six emulated video interrupts
(~100 ms of game time); at uncapped speed on an 8-core laptop we measure ~104 environment steps per second
across ten parallel Dolphin instances (§5.3). The 40.7 M-step model-free run required a 20-pod cloud fleet
for six hours. Any method that needs tens of millions of authoritative steps is therefore expensive to
iterate on.

Second, **the true state is cheap and exact.** Because Dolphin emulates an N64 whose 4 MB of RAM we can map
zero-copy into our process, we read a complete, noiseless description of the fight in ~25 µs — not a
rendering of it. The usual motivation for pixel-based world models (the true state is unavailable) does not
apply.

Together these suggest learning dynamics over structured state and moving policy optimisation off the
emulator entirely.

But structured state is not the same as *small* state. A fight holds up to sixteen actors over twenty fields
each, and most of that detail has no bearing on what a policy should do: the exact focus point of a room
prop, the velocity of a Baba that is still underground. A dynamics model asked to predict all 335 dimensions
spends its capacity on quantities no policy ever reads, and its loss is dominated by them. Worse, the fields
that matter most for control — an enemy's decoded action routine, the frame an HP decrement lands — are
low-entropy and discrete, so they contribute almost nothing to a regression objective while being exactly
what a policy needs.

This motivates the paper's organising idea: **choose the prediction target by what control needs, and let a
policy define that.** A behaviour-cloning encoder `E` trained on competent play is, by construction, a map
that keeps what competent action selection depended on. Learning `M` in `E`'s latent space rather than in
raw state changes the question from *can we reconstruct the game* to *can we predict the part of the game
that competent behaviour was sensitive to*. Our contributions, as scoped for this draft:

1. **A structured-state formalisation of OoT combat** (§3) with a fixed-width, permutation-stable encoding
   of a variable set of actors, including decoded enemy behaviour states.
2. **The behaviour-shaped latent as an experimental variable** (§6 measured, §7.2 proposed): dynamics capacity is held fixed while
   the latent varies — behaviour-cloned, autoencoded at equal width, or random — so the claim that
   behaviour-shaping helps is falsifiable rather than assumed.
3. **A competence axis** (§5): because the dataset spans six tiers of one training run, we can ask whether
   an encoder shaped by *better* behaviour is a better dynamics space, which is not a question a
   single-expert dataset can pose.
4. **An authoritative-simulator evaluation protocol** (§7.4) in which the emulator is used *only* for
   evaluation, never for policy gradient steps, making the sample-efficiency claim unambiguous.
5. **A model-free reference and a released dataset** (§4–5): a solved Deku Baba task at 40.7 M steps, and
   109,817 transitions with a dataset card and per-field schema.
6. **A proposed cross-scenario study** (§8): whether one latent and one dynamics model cover a trash mob
   (Deku Baba) and a boss (Queen Gohma) with distinct mechanics.

## 2. Related work and positioning

Dreamer-style agents learn a recurrent latent dynamics model from pixels and optimise a policy by
backpropagating through imagined rollouts; MuZero learns a value-equivalent model for planning. Both treat
the observation as an image. Our setting inverts that assumption: the state is directly observable and
*symbolic*, so the model's job is not perception but **dynamics** — object motion, scripted state machines,
damage resolution, and the consequences of a 63-way discrete action.

The nearest neighbours are object-centric and entity-factored world models, which impose a slot structure
on learned representations. We get that structure for free from RAM, which lets us ask a question those
methods cannot cleanly ask: *given perfect object factorisation, what should a dynamics model predict?*

Two lines bear directly on that. **Value-equivalent** models (MuZero and successors) answer "whatever
preserves value", learning a latent whose only obligation is to support planning, never reconstruction. Our
proposal is a cheaper cousin: rather than deriving the latent from returns through planning, we take it from
a policy trained by supervised imitation, which is stable, fast, and needs no environment interaction.
**Self-predictive and auxiliary-task representations** show that predicting one's own future embedding can
avoid the reconstruction trap, but they learn the encoder and dynamics jointly, so a degenerate constant
encoder is a live failure mode requiring stop-gradients or target networks. Freezing a behaviour-cloned
encoder sidesteps collapse by construction: `E` is fixed before `M` exists and was fit to a task —
predicting a competent agent's action — that a constant map cannot solve.

The cost is that `E` is only as good as the behaviour that shaped it, and it inherits that behaviour's blind
spots: state distinctions the demonstrator never had to make are invisible in `z`, and a policy optimised in
that space cannot learn to make them either. This is a real ceiling, and §6.3 is designed to detect it
rather than argue it away.

Positioning is deliberately narrow. We do not claim a novel architecture. The contribution is a
representation choice stated as a testable comparison, plus evidence about where it succeeds and fails in a
game whose dynamics we did not author and cannot inspect except through RAM.

## 3. Structured state and action space

### 3.1 Observation

State is read from N64 RAM at addresses verified against live recordings (`docs/RAM_MAP.md`). The transition
dataset stores a 335-dimensional vector per timestep:

| Block | Dims | Contents |
|---|---|---|
| Link | 15 | scene, room, HP, HP cap, `stateFlags1`, position (3), yaw, lock-on row, Navi-target row, held item, seeds, nuts, grass tufts standing |
| Actors | 16 × 20 | present, actor id, HP, position (3), action-routine offset, decoded behaviour state, shape yaw, yaw-to-player, action timer, category, params, world yaw, velocity (3), focus point (3) |

Two encoding decisions matter.

**Permutation stability.** The emulator exposes actors as a linked list whose order changes as actors spawn
and despawn. We assign each actor pointer a row on first sight and hold it for the whole episode, so row *j*
at time *t* and *t+1* is the same entity. Without this a dynamics model would have to learn that objects
teleport between rows. Lock-on and Navi targets are stored as *row indices*, not raw pointers, so the
encoding contains no memory addresses.

**Decoded behaviour.** A Deku Baba's action routine pointer (`actor+0x1B0`) takes eleven values that we have
mapped to named states (`hidden`, `prepare_lunge`, `lunge`, `pull_back`, `stunned`, `dying`, …). We store
both the raw pointer offset and the decoded index. This is the field a policy most needs and the one a
dynamics model is most likely to get wrong, so it is reported separately in §6.

### 3.2 Action

63 discrete actions: nine stick directions (neutral plus eight world-frame angles) × seven button
combinations (none, A, B, Z, shield, Z+B, Z+A), with three further item combinations reserved for the boss
profile (slingshot, Z+slingshot, Deku Nut) bringing the boss action space to 90. Directions are **world
angles**, converted to a stick deflection by the environment using the current camera yaw, so the
action–state pair is expressed in a single consistent frame.

Decisions are taken every 6 video interrupts (10 Hz). This is the transition Δt for the dynamics model and
is held fixed everywhere; changing it would invalidate both the dataset and the policies trained at it.

## 4. Model-free reference

A recurrent (GRU) entity-factored PPO policy was trained on the Deku Tree first room across a 20-pod CPU
fleet with periodic weight averaging.

**Table 1.** Model-free reference on Deku Baba (`docs/COMBAT_TASK.md`).

| Quantity | Value |
|---|---|
| Final policy | global update 150 (`deku-combat-gru-v2b_u00120.pt`) |
| Authoritative-simulator steps | 40.7 M |
| Observation | v4, 483 dims |
| Fleet training success (sampled episodes) | 77% |
| Local evaluation, 1 episode × 10 start states | **10/10 cleared, 0 deaths** |

This is the number any dream-trained policy must be compared against, and the 40.7 M is the denominator for
every sample-efficiency claim in §6.

The Queen Gohma boss task (`docs/GOHMA_TASK.md`) shares the RAM reader and reward bookkeeping, adds
slingshot and Deku Nut actions and observation v7 (297 global dims with per-frame seed-impact tracking), and
warm-starts from the Deku Baba policy. It is not solved; its partial runs are reference data, not a result.

## 5. The transition dataset

### 5.1 Composition

`scripts/collect_transitions.py` replays saved policies in headless Dolphin instances and records every
transition. Checkpoints are drawn from six points along the model-free training curve so the dataset spans
competence tiers rather than only expert behaviour — a world model trained solely on expert data never sees
the failure states a learning policy will visit.

**Table 2.** Dataset `datasets/transitions/deku-baba/20260918-134229` (measured).

| Checkpoint | Episodes | Transitions | Success | Death | Timeout | Mean episode steps |
|---|---|---|---|---|---|---|
| G5 | 15 | 20,782 | 27% | 0% | 73% | 1385 |
| G30 | 42 | 32,949 | 62% | 12% | 26% | 784 |
| G90 | 21 | 12,644 | 71% | 5% | 24% | 602 |
| G110 | 27 | 14,751 | 74% | 4% | 22% | 546 |
| G130 | 27 | 12,539 | 81% | 0% | 19% | 464 |
| G150 | 36 | 16,152 | 81% | 0% | 19% | 449 |
| **Total** | **168** | **109,817** | **69%** | **4%** | **27%** | 654 |

Success rises and episode length falls monotonically with training progress, and the final tier's 81%
recovers the 77% measured during fleet training — an independent check that the replay harness reproduces
the policies faithfully.

### 5.2 Layout and reproducibility

Each episode is one `.npz` holding `action[T]`, `reward[T]`, `state[T+1, 335]` and `obs[T, 483]`, so
transition *t* is `(state[t], action[t], state[t+1])`. `manifest.jsonl` carries one row per episode with
outcome, kills, HP, and the checkpoint and slot that produced it; `dataset.json` carries the full field
schema, action names, and per-checkpoint totals. See the dataset card at
[`datasets/transitions/deku-baba/README.md`](../datasets/transitions/deku-baba/README.md).

### 5.3 Collection cost

168 episodes and 109,817 transitions in 17.6 minutes wall clock on one 8-core laptop (10 parallel headless
Dolphin instances, ~104 steps/s aggregate). Episodes are collected independently rather than in lockstep;
a lockstep batch makes every tick wait for the slowest instance, which we measured at roughly half the
throughput at ten instances.

### 5.4 A trap worth recording

Our first collection run produced a plausible-looking dataset that was entirely worthless: the start
savestates had been overwritten with boss-room states from a later experiment, so the Deku Baba policy was
being evaluated against Queen Gohma. Every episode terminated as a timeout with zero targets, which is
indistinguishable from a weak policy in aggregate metrics. It was caught only by printing actor ids.
The collector now refuses to run unless every start state contains a live target of the expected actor id.
We report this because it is the failure mode structured-state datasets are most exposed to: the pipeline
does not crash, it silently changes the task.

---

## 6. Stage one, measured: the behaviour-shaped encoder

**This section is a result, not a protocol.** The dynamics model (§7) is still unbuilt.

We train a BC policy on the dataset of §5 with the architecture that solved the task model-free
(`EntityActorCritic`'s entity encoder: id embedding, a shared per-row MLP, masked mean+max pooling, concat
with the Link block, MLP trunk to `z ∈ R^128`). Off the same `z`, a decoder reconstructs the raw state
variables `CombatTracker.step()` reads. The reward is never regressed: it is computed from decoded state by
the real reward code.

Splits are **by start slot**, three ways: train {1,2,3,5,6,10}, threshold-tuning {4,8}, test {7,9}. Every
number below is on the test slots, at operating points chosen on the tuning slots only.

### 6.1 What survives the pooling

The worry was that masked mean+max pooling destroys per-row identity, making per-enemy HP and behaviour
state unrecoverable. It does not.

**Table 4.** Reconstruction of the reward function's inputs from a frozen 128-d latent (test slots).

| field | model | baseline | unit |
|---|---|---|---|
| Link HP | **0.988** | 0.909 | accuracy, 7 half-heart classes |
| Link position | **5.2** | 37.1 | MAE, world units |
| Link yaw | **1.8** | 74.4 | mean abs error, degrees |
| shielding / damaged bits | **1.000** | 0.987 / 0.994 | accuracy |
| lock-on target row | **1.000** | 0.767 | accuracy over 17 |
| row present | **0.996** | 0.543 | accuracy |
| row HP | **0.908** | 0.554 | accuracy, 3 classes |
| row position | **7.2** | 250.2 | MAE, world units |
| **Baba behaviour state** | **0.999** | 0.669 | accuracy over 10 |

The behaviour state — the field a policy most depends on and the one we most expected to lose — is the best
reconstructed of all.

### 6.2 Static reconstruction is not the same as usable reward

The reward reads HP *deltas*, and damage occurs once in 253 transitions. At a per-frame HP error rate `ε`,
each error fabricates roughly two spurious deltas, so precision ≈ `0.0019 / (0.0019 + 2ε)`: **three to four
nines of per-frame accuracy are needed before differencing a decoded state yields usable events.** Our first
attempt regressed HP and rounded it, which converts numeric jitter directly into fabricated damage, and
scored F1 0.286 / 0.491 / 0.059 despite looking excellent per-frame.

Four changes closed most of that gap, of which only the first two were the ones we expected:

1. **HP as classification, not regression.** It is discrete (Link: 7 half-heart values; Baba: 0/1/2), so
   classification removes the jitter channel entirely.
2. **A monotonicity constraint.** HP never rises: across the dataset a Baba's HP increases in **0 of 891,459**
   present row-steps, and Link healed **once in 109,649**. A running minimum per episode deletes every
   upward flicker. It is not free — it entrenches a premature decrease, which costs recall.
3. **Cross-head consistency.** Penalising a row decoded at 0 HP that is neither dying nor despawned.
4. **Light oversampling of HP-transition frames** (5% of each batch). At 15% it is actively harmful — the
   model hallucinates transitions — so this is a narrow optimum, not a free lever.

Two ideas that seemed obvious did **not** help: gating damage on the decoded `hit` state (a 46× lift in
ground truth, but the decoder does not localise the state *entry*), and corroborating kills with despawn
(100% of true kills despawn, but HP→0 probability was already saturated).

The single largest remaining win was neither of those: **calibrating the detector and choosing its
threshold**. argmax is an arbitrary operating point; scoring each candidate with `Σ_{a>b} P_t(a)·P_{t+1}(b)`
and picking a threshold on held-out tuning slots was worth more than any architectural change.

**Table 5.** Reward-critical event detection from the frozen latent (test slots, threshold fixed on tuning slots).

| event | detector | F1 | precision | recall | AP | TP/FP/FN |
|---|---|---|---|---|---|---|
| Baba damage | hit-entry + HP-drop | 0.862 | 0.833 | 0.893 | 0.828 | 25/5/3 |
| Baba damage | HP-drop probability | 0.833 | 0.781 | 0.893 | **0.887** | 25/7/3 |
| Baba kill | HP→0 probability | **0.944** | **1.000** | 0.895 | **0.895** | 17/0/2 |
| Link damage | HP-drop probability | 0.629 | 1.000 | 0.458 | **0.827** | 11/0/13 |

Average precision is threshold-free and is the figure to compare; F1 depends on an operating point that must
transfer across slots, and for Link damage it did not — the tuning slots chose a threshold whose recall
collapsed from 0.92 to 0.46 on test while precision stayed at 1.000. We report this rather than tuning on
the test slots, which had inflated the same number to F1 0.898.

### 6.3 What this licenses, and what it does not

A frozen behaviour-shaped latent retains the reward function's inputs well enough to recover its key events
at ~0.89 AP. That is the precondition for §7 and it is met.

It is also an **upper bound**. These numbers come from autoencoding `s_t` through `z_t = E(s_t)`; a dynamics
model decoding from a *predicted* `ẑ_{t+1}` can only do worse. And the counts are small — 28, 19 and 24 test
events — so differences of a few points are noise. The honest claim is that the representation is adequate,
not that reward recovery is solved.

One cost is visible throughout: action top-1 is 0.049 against a 0.220 Bayes ceiling and a 0.039 majority
baseline. Reconstruction pressure and action prediction compete for a 128-d latent, and reconstruction won.
Whether that trade helps or hurts a dreamed policy is exactly what §7 measures.


---

## 7. Proposed protocol: behaviour-shaped latents, dreaming, and transfer

**Not yet run.** This section fixes the protocol in advance so results cannot be chosen after the fact.

### 6.1 Stage one: the encoder

Train a behaviour-cloning policy `π_BC(a | s)` on the dataset of §5, with an architecture that respects the
actor-set structure: a shared per-row encoder applied to all sixteen actor rows, masked permutation-invariant
pooling, concatenation with the Link block, then an MLP to a latent `z ∈ R^d` and a 63-way action head.
Auxiliary heads off the same `z` predict reward, terminal, Link HP, and live-target count — the quantities
policy optimisation needs that action prediction alone might not preserve.

The encoder `E` is then **frozen**. Everything downstream consumes `z = E(s)` and never updates `E`.

**The demonstrator was not a function of state alone.** This is a subtlety the setup forces into the open.
Our structured state is Markov in the *game*: it is what the emulator's RAM says is true at time *t*. But
the policies that generated the data are recurrent (a GRU over the entity encoder) and their observation
carried explicit episode memory — time remaining, steps since the last hit, kills so far. So `a_t = π(s_t)`
is not a well-posed cloning target, and a feedforward `E` trained on it has an accuracy ceiling set by
partial observability rather than by capacity.

We resolve this by putting the recurrence **after** the encoder and outside the frozen artifact:

```
z_t     = E(s_t)                  # frozen, state-only — required for M(z_t, a_t) to be well-posed
h_t     = GRU(z_t, h_{t-1})       # BC policy memory, discarded after stage one
logits  = head(h_t)
```

`E` must not itself be recurrent, or `ẑ_{t+1} = M(z_t, a_t)` is ill-posed: `z` would depend on history the
dynamics model cannot see. Auxiliary heads therefore hang off `z_t`, not `h_t`, so the quantities the
dynamics model must decode remain properties of the state. The feedforward-vs-recurrent gap is reported as
an ablation, because it measures something we need to know anyway: **how much of competent play here was
memory rather than state**, and therefore whether a Markov latent is a sufficient basis for dreaming in this
domain at all. A large gap is evidence against the whole approach and we will report it as such.

Two further knobs define the "shaped by competent behaviour" axis, both ablations rather than assumptions:
the competence tiers included (§5 spans 27%→81% success), and whether to restrict to successful episodes.

### 6.2 Stage two: dynamics in latent space

Learn `ẑ_{t+1} = M(z_t, a_t)` against the target `E(s_{t+1})`, with `E` frozen, plus decoder heads for the
raw quantities optimisation requires (reward, terminals, HP, live targets). Note what is *not* required: a
decoder back to all 335 dimensions. The model is never charged for reconstructing the game.

**The comparison that makes this a claim rather than a preference.** Hold `M`'s architecture and parameter
count fixed and vary only the latent:

| Latent | Shaped by | What it tests |
|---|---|---|
| `E_BC` | competent action selection | the paper's proposal |
| `E_AE` | reconstruction of `s`, same width `d` | whether *behaviour* shaping matters, or merely compression |
| `E_rand` | nothing (frozen random projection) | whether the encoder is doing anything at all |
| identity | — | raw 335-d state: the baseline this paper argues against |

If `E_AE` matches `E_BC`, the contribution reduces to dimensionality reduction and we will say so. If
`E_rand` matches both, there is no contribution and we will say that too.

### 6.3 Metrics for the model itself

Latent-space error is not interpretable on its own — a latent can shrink its own error by discarding
information. So every dynamics model is scored through decoder heads on **raw, semantically named
quantities**, at horizons *h* ∈ {1, 5, 10, 50}, on episodes held out **by start slot**:

| Quantity | Metric |
|---|---|
| Enemy HP | exact-match accuracy; separately, recall of the timestep at which HP decrements (the hit event) |
| Decoded behaviour state | per-class accuracy, with `lunge` and `stunned` broken out as the decision-relevant classes |
| Terminals | precision/recall of death, room exit, success |
| Link position | mean absolute error in world units; at *h*=50, fraction of rollouts still within 100 units |

Compounding error is the expected failure mode; the horizon at which behaviour-state accuracy falls to
chance defines the usable dream length and therefore the imagination horizon in §7.4.

**The blind-spot probe.** Because `E` is frozen and fit to imitation, it may have discarded distinctions
competent play never needed. We test this directly: train linear probes from frozen `z` to held-out
semantic fields (nearest enemy's behaviour state, distance to nearest target, seeds remaining) and compare
against the same probes on raw normalised state. Fields where `z` probes far worse than raw state are the
encoder's blind spots, and we report them *before* policy results, since they predict where dreaming will
fail.

### 6.4 Stage three: policy learning in the dream, evaluation in the emulator

Train the same GRU entity-factored architecture used for the model-free reference, with PPO, on rollouts
generated **only** by `M`, starting from states sampled from the dataset. No emulator steps enter the policy
gradient. Then evaluate in Dolphin.

**Table 3.** Proposed evaluation. "Authoritative steps" counts *only* policy-training interaction; dataset
collection (109,817 steps) is a fixed one-time cost reported separately.

| Condition | Dream steps | Authoritative steps for training | Reported |
|---|---|---|---|
| Model-free reference (§4) | 0 | 40.7 M | 10/10 slots, 77% fleet success |
| Behaviour cloning alone | 0 | 0 | success over 10 slots × 10 episodes |
| Dream in `E_BC` | ≥ 10 M | **0** | same |
| Dream in `E_AE` / `E_rand` / raw | ≥ 10 M | 0 | same — the §7.2 ablation, carried through to transfer |
| Dream in `E_BC` + fine-tune | ≥ 10 M | 10^5 | same |

The behaviour-cloning row is the one that can most easily embarrass us: 69% of dataset episodes succeed, so
imitation alone may already be competent. **Dreaming must beat its own frozen encoder's policy to mean
anything** — otherwise we have shown only that BC works and the dynamics model is decoration.

Success criterion, fixed now: **dream-only clears ≥ 7/10 start slots at ≥ 50% per-slot success**, using the
same protocol as §4, *and* exceeds the behaviour-cloning row.

### 6.5 Failure modes we expect to report

Model exploitation — the policy finding states `M` scores well and the emulator does not — is the standard
pathology, diagnosed by replaying dream-optimal action sequences in the emulator and measuring divergence.
A behaviour-shaped latent may make this *worse*: `z` is only trustworthy where the demonstrator went, and a
policy free to optimise inside it is under no obligation to stay there. We will report the divergence curve
and the fraction of dreamed states whose `z` falls outside the encoder's training support, whether or not
transfer succeeds.

## 8. Proposed cross-scenario study

Deku Baba and Queen Gohma share a RAM reader but differ in mechanics: the boss requires ranged attacks
(slingshot seeds at a red eye), consumable management, and has a larva sub-population. Planned conditions:

Because the pipeline has two learned stages, "does it transfer" splits into two questions — whether the
*latent* transfers and whether the *dynamics* do — and they can be varied independently:

1. **Per-scenario, end to end** — its own encoder and dynamics per enemy; establishes the achievable ceiling.
2. **Joint** — one encoder and one dynamics model over both, with actor id as an input feature; tests whether
   capacity shared across enemies helps or hurts.
3. **Shared latent, per-scenario dynamics** — one encoder trained on both enemies, separate dynamics. This
   is the condition that isolates whether a behaviour-shaped representation is enemy-agnostic: fighting a
   Baba and fighting Gohma may share more at the level of *what competent play attends to* (a telegraphed
   window, a lock-on, a health bar) than at the level of raw dynamics.
4. **Held-out enemy** — encoder and dynamics from Deku Baba only, dream a Gohma policy, evaluate in the
   emulator. We expect this to fail, and the informative measurement is *where*: the §7.3 probe on Gohma
   states tells us whether the encoder lost the boss-specific fields (eye state, seed count) before the
   dynamics model ever had a chance.

## 9. Limitations

- **One game, two enemies.** Nothing here establishes generality across titles.
- **The encoder inherits the demonstrator's blind spots.** `E` is fit to predict a competent agent's
  actions, so state distinctions that agent never had to make are absent from `z`, and no amount of
  dreaming in `z` can recover them. This bounds the approach to roughly the demonstrator's competence
  envelope; §7.3's probe measures the bound but does not remove it.
- **Behaviour shaping needs behaviour.** The method presupposes a policy good enough to be worth imitating.
  It is therefore not a replacement for model-free RL but a way to amortise it — which makes the honest
  comparison in Table 3 cumulative cost, not just the dream-phase step count.
- **Structured state is game-specific.** The encoding depends on hand-verified RAM offsets; the method
  transfers to another game only with comparable reverse-engineering effort. This is a real cost and we do
  not amortise it away.
- **The dataset is on-policy to a specific training run.** Coverage spans six tiers of one PPO run, not the
  space of all behaviours. States reachable only by a differently-shaped policy are absent.
- **No pixel baseline.** We do not compare against a pixel-based world model on the same task, so we cannot
  claim structured state is *better* — only that it is sufficient and interpretable.
- **The authoritative simulator is itself an emulator.** "Ground truth" means Dolphin 2603a on NTSC 1.2;
  savestate compatibility is version-locked.

## 10. Reproducibility

Every table above is regenerated from committed artifacts by `results/make_figures.py`; no number in this
document is typed by hand. Run identity, data provenance, and the exact commands are in
[`../README.md`](../README.md) and the dataset card. The audit of which claims are measured, inferred, or
merely asserted is [CLAIMS.md](CLAIMS.md).
