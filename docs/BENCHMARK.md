# The benchmark: % of real, on windows where the competency happens

**This is the number that decides whether a world model is usable.** Not per-field skill, not
log-likelihood, not training loss. `scripts/eval_competencies.py`.

    .venv/bin/python scripts/eval_competencies.py --ckpt runs/p2-loop-oversample/loop_state.pt
    .venv/bin/python scripts/eval_competencies.py --ckpt ... --trace walk    # step-by-step side by side

## Why this and not per-field skill

`Link.pos` scores **+0.144** on the per-field metric: on the steps where Link moves, the model's SINGLE-STEP
error is 14% better than predicting no movement. Roll the same model forward on its own output for 20 steps
with the real action sequence and it travels **7.4% of the real distance**, ending further from the truth
than freezing Link at the start.

Both numbers are correct. A small single-step edge does not compound into a trajectory. Only the rollout
says whether the thing can be used as a dream, which is AGENTS.md section 8 restated per competency.

## Method

A window qualifies if the **REAL** trajectory contains the competency, so "real" is 1.00 by construction and
a broken window definition shows up as a real column that is not ~1.00. Selection uses the real trajectory
ONLY, never the model, so the same windows score any checkpoint.

Each window is rolled from its own real start state with the real action sequence, **full-state
autoregressive**: all 77 fields predicted each step and fed back through `decompose -> model -> recompose`,
exactly as a dream would, so errors compound across fields.

**The do-nothing control is reported beside every number and is not decoration.** It caught a broken metric
on its first run: `shield` originally counted steps-with-shield-up, and a frozen model scored **21 of a real
23**, because a shield already raised at step 0 stays raised for the whole window. Metrics that a frozen
state can satisfy are worthless. It now counts shield RAISE events; the dream's score went 13% -> **0.0%**.

| competency | measures | unit |
|---|---|---|
| `walk` | distance travelled | world units |
| `lockon` | Z-target ACQUISITIONS (`sf1` bit 15, 0 -> 1) | events |
| `shield` | shield RAISE events (`sf1` bit 22, 0 -> 1) | events |
| `approach` | distance closed to the nearest live Deku Baba | world units |
| `combat` | enemy HP removed | HP |

## Result of record

`runs/p2-loop-oversample` -- discovered graph (55 edges), 3-term objective, event oversampling K=50.
12 windows per competency, horizon 20, test slots {7,9}.

| competency | windows | real | dream | **% of real** | do-nothing |
|---|---|---|---|---|---|
| `walk` | 12 | 2136.0 | 158.4 | **7.4%** | 0.0 |
| `lockon` | 12 | 37.0 | 32.0 | **86.5%** | 0.0 |
| `shield` | 12 | 19.0 | 0.0 | **0.0%** | 0.0 |
| `approach` | 12 | 1502.7 | 45.1 | **3.0%** | 0.0 |
| `combat` | 11 | 12.0 | 0.0 | **0.0%** | 0.0 |

**Movement, approach, shield and combat are all effectively zero. Lock-on is the one competency that works.**

A caveat on lock-on: the metric counts HOW MANY acquisitions, not whether they occur at the right moment.
86.5% means the dream produces about the right number of Z-target events, not that they are the right ones.

## What a single trace looks like

`--trace walk`, real actions, clean walking episode, no room transition:

    step  action  |  REAL x   REAL z |  DREAM x  DREAM z |  real step  dream step
       0  N       |  131.49    78.47 |  131.49    78.47  |     0.00      0.00
       4  N       |  129.20    95.55 |  132.16    77.42  |    14.24      0.25
       8  N+A     |  129.87   168.36 |  131.91    76.59  |    23.62      0.30
      12  -+Z     |  134.58   267.25 |  129.77    75.93  |    24.75      0.72
      20          |  133.90   253.01 |  121.44    76.21  |    18.00      1.25

Three things this shows that the aggregate hides:

1. **Real steps are 14-25 units, dream steps 0.2-1.3** -- about 20x too small, consistently.
2. **The dream drifts the wrong way.** Link presses `N` and gains +175 in z; the dream moves -2 in z.
   It is not under-shooting a correct direction, it is going somewhere else.
3. **It ignores the action.** At steps 14-18 Link is stationary; the dream keeps accelerating.

## History on this metric

| model | walk % of real |
|---|---|
| latent world model (LATENT_WORLD_MODEL.md section 22) | dream position error 103 vs 50 for "never moves" |
| OOCDM at `G1` (CAUSAL_WORLD_MODEL.md section 15) | 4% |
| **OOCDM, discovered graph + oversampling** | **7.4%** |

The causal machinery has not moved this. 4% -> 7.4% is not a usable dream by any reading.
