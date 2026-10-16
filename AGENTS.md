# Rules for working in this repo

**Start with `docs/MAP.md`** — it is the index to 15 documents and the code layout, plus the environment facts
that have each cost a session (the signed interpreter, the Deku savestate provenance, the fleet's task-specific
S3 prefix). Do not grep blindly; look at the map, then read the whole document it points you to.

Every rule below is here because breaking it cost real time, and the specific incident is named so the rule
cannot be argued away as generic advice.

## 0. Orientation, in order, before touching anything

1. `docs/MAP.md` — what exists and where.
2. The rule list below.
3. The specific doc the map points to for your task, **in full**. Long docs bury their operational facts in
   tables below the examples: `RUNPOD.md` line 80 is the one that mattered most and it is 80 lines down.
4. The docstring of the module or script you are about to change.

## 1. A number is not a result until its controls are run

Never report, act on, or build on a metric before running all three controls. Run them BEFORE spending
compute on the thing the metric seems to justify, not after.

| control | question it answers |
|---|---|
| **identity / do-nothing** | does the model beat "nothing changes"? A zero-init residual predictor IS the identity, so skill = 0 is the floor, not a bad score. |
| **untrained, same architecture** | did *training* do anything, or is this the shape of the map? |
| **free-signal ablation** | is the signal a clock, an episode constant, or a copy of the input? |

**The incident.** The competent PPO policy's `proj` latent scored skill **+0.742** as a predictive surface
against our encoder's −0.48, and the obvious conclusion — "reuse the competent policy as the encoder" — was
reported before any control ran. The controls then destroyed it completely:

- same architecture, **random untrained weights**: **+0.617**. Competence contributed nothing.
- minus the **control echo**: **+0.240**. `obs` dims 22:26 are `prev_action_*`, a literal copy of `a_t`
  into `obs_{t+1}`.
- minus **velocity**: **+0.053**. With velocity in `z`, next position is linear extrapolation.
- minus **`time_left` and the `goal_*` block**: **−0.012**. A deterministic countdown and episode constants.

The whole +0.742 was artifacts. Cost: days, and a WM/PPO run was nearly launched on it.

## 2. Predictability and informativeness must be reported together

A latent that never moves is trivially predictable and useless; a constant encoder is the global optimum of
any pure prediction objective. So **every** skill number is reported next to:

- `move/rand` = mean |z_{t+1} − z_t| / mean |z_i − z_j|. Low means the latent barely moves per step.
- a **decode probe** (ridge is fine) for the quantities the reward actually reads: position, yaw, HP, counts.

Two columns, always. `scripts/eval_policy_latent.py` is the template.

## 3. Never compute an offset, index, or field width by hand

Derive it from the definition in code and assert it.

**The incident.** The `obs` echo slice was hand-counted as `slice(23, 27)`. `GLOBAL_FIELDS_V2` sums to 22, so
that slice zeroed half the echo and one unrelated `state_flags` bit — which is why the first echo ablation
falsely reported +0.575 surviving. Fixed by building the slices from `GLOBAL_FIELDS_V2` itself.

Same rule for RAM offsets: read them from the [OoT decompilation](https://github.com/zeldaret/oot), never from
recall. See `docs/ADDING_OBSERVATIONS.md`. Retail Player fields are `decomp offset − 0x10`.

## 4. Read the whole definition before claiming a field is absent

**The incident.** "Our encoder never saw relative positions or velocity" was asserted to the user after
reading only the first half of `featurise()`. Both were already there — `d = rpos - lpos`, `dist`, bearing and
per-actor velocity in the row block, and Link's `speed`/`vel_xyz` in the v2 link block. The claim was wrong
and had to be retracted in the next message.

`grep` the symbol, open the function, read it to the `return`. Do not infer absence from a partial read.

## 5. Control echo: check every input vector for a copy of the action

An encoder input that contains the previous action makes the next latent trivially predictable and teaches the
world model nothing. This has now bitten twice: once in our own features (`CONTROL_ECHO_FEATS`, removed — see
`docs/LATENT_WORLD_MODEL.md` §26) and once in the recorded 483-d `obs`, where it is present **by construction**
and was not noticed for days.

Before using any recorded vector as an encoder input — including one produced by another system — list its
fields and find the action-derived ones.

## 6. Actions, action-states and outcomes are different things

- **action**: what the agent commands — stick, buttons. A roll is an action. A block is an action.
- **action-state**: what routine Link is in — `actionFunc`, melee state, airborne.
- **outcome**: what the world does in response — position, HP, damage, kills, collisions.

Conflating them at the encoder or decoder level produces models that "predict" outcomes by echoing actions.
Keep them separate, and be explicit about which one a head predicts.

## 7. One seed is not a measurement

Six seeds of one identical config spanned walk trajectory error 1.38×–3.43× (`docs/LATENT_WORLD_MODEL.md` §25).
Headline numbers reported earlier in this project were seed 0 — the best of six on all three metrics at once.
Report a range, or say explicitly that it is a single seed.

## 8. The barometer is a ground-truth rollout, not a training loss

Anything claimed about the world model is validated by replaying **real action sequences from real start
states** and comparing decoded trajectories against ground truth, at multiple horizons
(`scripts/eval_dream_rollout.py`, `scripts/eval_event_windows.py`). A better training loss, a better tune
objective, or a better skill number is not a result on its own.

## 9. Ablate before you scale

Cheap ablation first, expensive training second. Every hour of training spent on an unvalidated headline
number is an hour that has to be thrown away, plus the time spent believing the conclusion it produced.

## 10. Report failure plainly and immediately

When a measurement kills a hypothesis — including one already reported to the user — say so in the next
message, state which earlier claim is now wrong, and do not soften it. Retracting costs one message.
Not retracting costs days.

## 11. An at-initialisation measurement does not predict a trained outcome

Measuring an architecture's properties with random weights tells you about its geometry at init, not about
where training lands. Joint training here runs a decode loss of ~300-1800 through the encoder, and that
pressure reshapes the latent far away from wherever it started.

**The incident.** Untrained on identical features at matched latent movement, the latent LayerNorm measured a
0.73 skill cost and the deep trunk another 0.36, so `shallow + tanh` was predicted to fix the encoder. Trained
under the real objective it scored skill **-2.023 against the deep+LayerNorm baseline's -0.776** -- 2.6x worse
on exactly the metric it was chosen to improve. The controls were right; the extrapolation from init to trained
was the error.

Corollary: when an architecture change is the hypothesis, the smoke test must TRAIN both arms and compare at
matched epochs. Nothing about init transfers.

## 12. Compare at matched epochs, and check the denominator before reading a ratio

`skill = 1 - pred/identity` divides by how much the latent moves. A change that makes the latent MORE static
shrinks the denominator and moves skill for reasons unrelated to the model getting better.

**The incident.** The transformer encoder showed a 3x better raw prediction error than the MLP at epoch 1
(0.0368 vs 0.1101) and a *worse* skill (-37.6 vs -16.2), purely because its identity error was 6x smaller
(0.00095 vs 0.00613). Reading either number alone would have given the opposite conclusion.

Always print `pred`, `identity` and the move ratio together, and compare arms at the same epoch count.

## 13. Follow the paper you adopted. Deviations are proposals, not edits.

When a method is adopted from a paper, the paper's premise is the plan. Changing it is a decision for the
person who chose it, and it is proposed before it is coded — never shipped inside a commit that claims to be
implementing the method.

**The incident.** OOCDM (Yu et al., ICML 2024) was adopted to replace the latent world model. Stage 0 (schema
+ census) and Stage 1 (per-field model at the full graph) were built as planned. Then, over six unrequested
experiments in one session, the following were added — **none of which appear in the paper**:

- a per-field **change gate** (a "did this field change" classifier gating the value head),
- per-field **`pos_weight` oversampling** of rare events,
- **binned direction + magnitude** heads for vector fields,
- per-field **firing-threshold calibration**.

All four came from `docs/LATENT_WORLD_MODEL.md` §3/§23/§29 — the *failed* line of work OOCDM was adopted to
escape. So the session re-ran the old approach's head-tuning inside the new architecture's shell.

Meanwhile **the paper's actual contribution was never run**: no masking, no CMI, no causal discovery, no
`Ĝ` vs `G₁` vs empty-graph comparison, and no shuffled-class control. What got trained was OOCDM's
*baseline* (full graph, no discovery), not OOCDM.

Cost: a session of compute, four architecture experiments scored against a floor that was too low (the
trivial `pos + k·vel` control, mandated by §1 and named in §28, was run last instead of first — and it beats
the model), and a codebase that had to be stripped back.

**The rules this produces.**

1. **The paper's method is the baseline; your idea is the experiment.** Run the paper's version first,
   completely, with its own controls, before proposing anything else.
2. **A deviation is proposed in a message, not in a commit.** If it is worth doing it is worth one sentence
   of agreement first.
3. **Deviations already present must be listed** in the module docstring, so the next reader knows what is
   the paper and what is ours. See the header of `zelda_rl/oocdm.py`.
4. **Finish the phase you are in before starting another.** Stage 1 still had two controls unrun
   (shuffled-class, seed variance) when Stage 2 was skipped entirely.
5. **One authorisation, one experiment.** "Start with a change gate" authorises a change gate — not the six
   follow-ups its results suggest. A fast iteration loop is a reason to check in more often, not less.

## 14. Read the whole runbook, and do not "fix" a documented fast path

**The incident.** Bringing up a 39-pod collection fleet. `docs/RUNPOD.md` line 80 states plainly:
*"Seed bootstrap (`bootstrap --via-seeds 0,2`): 4–48 s per pod; all 19 pods fully bootstrapped in ~3.5 min."*

I read the first 70 lines, used the `--via-seeds` example from the top-of-file summary, hit some transient SSH
errors at 8-way concurrency, and "fixed" it by throttling to **sequential batches of 4**. That turned a
3.5-minute job into an hour of billing with zero transitions collected. The retries were already handling the
transient errors; serialising was the bug.

Rules: read the whole runbook before operating a system you have not operated. When a documented path reports a
transient error, **re-run it**, do not redesign it. And if you find yourself building a batching wrapper around
a tool that already parallelises, stop.

## 15. On a fleet, the Mac's uplink is the enemy; never push per-pod what one pod can fan out

Home upload is ~0.7 MB/s (`RUNPOD.md` records this and line 118 names it as dominating bootstrap). Three
separate mistakes in one session, all the same shape:

- pushed **52 GB of `datasets/`** to every pod because `CODE_EXCLUDES` did not list it (pods *generate* data);
- pushed **47 MB of policies × 38 pods** from the Mac (~43 min) instead of one push plus pod→pod rsync;
- pulled savestates via S3 presigned URLs — **8–20 s per pod** — which is what the other two should have done.

Order of preference: S3 presigned pull (seconds) → one push + pod→pod fan-out → per-pod push from the Mac
(never, for anything large).

Also: when fanning out pod→pod, open **one** SSH to the seed and loop inside it (`xargs -P6`). One SSH per
target trips the seed's `MaxStartups` and every connection is reset.

## 16. Verify the artefact landed, not that the command ran

`push_policies` logged "push 9 mix policies (47 MB)" on every pod and the files were **not** where the pods
looked: `rsync -R` with a `/./` marker preserved the *absolute* path, so they arrived at
`/workspace/zelda-RL/Users/jordy/Documents/dev/zelda-RL/checkpoints/...`. Every launch then died on
`FileNotFoundError: checkpoint not found`.

After any push, `ls` the destination the consumer will actually read. A success log line is not evidence.

## 17. Make progress observable before starting a long remote job

There was no way to see fleet collection progress without hand-written ssh loops, so the first sign that
anything was wrong was a per-pod log tail. `python -m fleet collected --experiment E` now reports total
transitions, episodes and the outcome mix from the manifests, and warns about pods with no Dolphin running.

Before launching anything long and remote: know the single command that answers "is it working and how far
along is it", and run it once early.

## 18. Never diff against state you just overwrote

**The incident.** Bringing up 116 pods. `fleet create` only writes its state file at the end, so to start
bootstrapping early I rebuilt `runs/fleet/v5.json` from the live API — and then computed "which pods still need
bootstrapping" by diffing against **the file I had just overwritten with all 116**. It reported 7 instead of 78.
**78 pods sat idle and billing for ~25 minutes**, ~$40/hr of nothing.

Capture the "before" set into a variable BEFORE writing, or derive the answer from the pods themselves rather
than from bookkeeping. Which leads to:

## 19. After every phase, audit what the machines are actually doing

The 25 minutes went unnoticed because there was no way to see per-pod state without hand-writing an ssh
fan-out, so "bootstrap ran" was never checked against "every pod is working". `python -m fleet audit
--experiment E` now reports WORKING / READY-BUT-IDLE / BOOTSTRAP-INCOMPLETE per pod, the episode count, and the
$/hr being spent on pods doing nothing. Run it after create, after bootstrap, after launch — not when something
already looks wrong.

Corollary: a launch command reporting "started: N" is not evidence that N pods are working. It means N commands
were dispatched. Seven pods once reported "collector started" and produced zero episodes.

## 20. Rule 15 applies to EVERY large per-pod transfer, including ones you add yourself

`push_policies` was written to fix rule 16, and it pushed 47 MB from the Mac to each of 78 pods — 3.7 GB over a
~0.7 MB/s uplink, ~88 minutes, and it silently stalled the bootstrap while looking like it was progressing. The
function fixing one rule violated another written the same night.

Before adding any per-pod transfer, ask what it costs times the pod count. Over ~10 MB total it belongs in S3
with presigned GETs (measured: 47 MB of policies to 17 pods in 100 s, versus 1–2 pods per 100 s from the Mac).
`S3_PROFILE`-signed URLs need no credentials on the pod.

## 21. WHY THE SAME FAILURES REPEAT — and the only fix that works

Every failure on the 116-pod fleet was **already written down as a rule in this file**, and every one was
repeated anyway. That is the important finding, more than any individual bug:

| what happened | the rule that already existed |
|---|---|
| pushed 47 MB × 78 pods from the Mac, stalling bootstrap ~88 min | §15, written the same night |
| trusted "started: 48" as evidence 48 pods were working | §19, written the same night |
| threw a batching wrapper around a tool that already parallelises | §14 |
| diffed against a state file just overwritten | §18 (written *after*, but §1's "verify before building on it" covered it) |

**The mechanism: a rule in a document has to be REMEMBERED at the moment it matters.** These rules were read
once at session start and then competed with time pressure, a frustrated user, and 39 pods billing. Memory lost
every time. Prose does not run.

So the rule about rules:

**Encode every operational rule as a check that runs automatically and fails loudly. If it cannot be a check,
it will be broken.**

Applied:

- `fleet preflight --experiment E` tests every pod for dolphin / torch / **boto3** / xvfb / secrets / ≥9 mix
  policies / ≥10 savestates and prints *what each failure means*.
- **`fleet launch` runs preflight and SKIPS pods that would fail**, so "started: N" can no longer mean "N
  commands dispatched into the void". `--skip-preflight` to override deliberately.
- `fleet audit` uses **tmux session presence** for liveness, not `pgrep dolphin`.

## 22. Never build a control loop on a signal you have not measured

A "self-healing" daemon polled `pgrep -c dolphin-emu` every 40 s and relaunched anything reading zero. Booting
12 Dolphin instances takes longer than 40 s, so **it killed and relaunched freshly-started collectors in a
loop**, destroying work on every cycle until it was noticed by hand.

Before automating on a signal: run it once, by hand, on a machine in the state you are trying to detect, and on
a machine in the opposite state. Ten seconds of checking. `tmux has-session` is true the instant the collector
starts; `pgrep dolphin` is not.

And a control loop that can destroy work needs a bias toward inaction: require N consecutive observations,
interval longer than the startup it is watching, and never let it act on a single sample.

## 23. Inside a container, the tools that report the machine are lying to you

`free -g` said **1511 GB**. The cgroup limit was **175 GiB**. A 100-epoch run was launched on the strength of
the first number and SIGKILLed twenty minutes into its load — and the real figure had been printed by the
pod-create call itself (`mem=188GB`) and then ignored in favour of the more convenient one.

`free`, `nproc` and `/proc/meminfo` describe the HOST. The limits that apply are in `/sys/fs/cgroup/memory.max`
(v2) or `/sys/fs/cgroup/memory/memory.limit_in_bytes` (v1). This is now a check that runs before any load
(`wm_data.check_load_fits`), because a rule is not a check (§21).

Two more instruments that lied today, both cheap to get right:

- **`pgrep -f <pattern>` matches your own command line.** Running it over ssh, the remote shell's argv contains
  the pattern, so it reports a live process on an idle machine. Use `ps -eo cmd | grep "[p]attern"` — the
  bracket stops the grep matching itself — and corroborate with the thing you actually care about
  (`nvidia-smi` showing 0 MiB).
- **`pkill -f <pattern>` over ssh kills the ssh session, for the same reason, and the trainer survives.**
  `ssh pod "pkill -f train_jepa_latent"` returned 255 with no output: the remote shell's own argv contained
  the pattern, so pkill killed its parent before reaching the trainer. It looks exactly like a network
  failure. Bracket the pattern here too — `pkill -f '[t]rain_jepa_latent'` — and confirm with
  `nvidia-smi` at 0 MiB rather than with pkill's exit status.
- **`timeout` kills the shell, not the grandchild.** After `EXIT=124` a trainer was still holding the GPU and
  would have competed with the next launch. Kill the real process in the launcher's own trap.

## 24. A silent long-running job is indistinguishable from a hung one

`python … | grep …` redirected to a file **buffers**, so a healthy training run wrote an empty log for nine and
a half minutes while it was at 97% GPU. The conclusion drawn from the silence — that it was stuck — was wrong,
and acting on it threw away the work.

`python -u` and `grep --line-buffered`, always, for anything whose output you will read later. And any loop that
will run for more than a minute or two prints progress with an ETA: `load_many` read 103,716 files for ~25
minutes in total silence, which left RSS growth as the only evidence it was alive.

## 25. Pack the expensive preparation ONCE, and put it where a pod dying cannot take it

Loading v5 from `.npz` costs ~25 minutes of file opens and decompression, every run, and needs more host RAM
than the pod has. Packed to per-field `.npy` in the narrowest dtype each field admits it is 49 GiB, loads in
~2.5 minutes, and is the ONLY artefact a training run needs.

The rule is not "pack it" — it is **pack it once and put the result in S3 immediately**. `/workspace` is a pod
volume and is destroyed with the pod, which is precisely how a 20M-transition dataset was lost (§15-16). And
the packer and the trainer must share one implementation of the transformation (`zelda_rl/wm_data.py`), or the
packed path silently drifts from the in-memory one and numbers stop being comparable with nothing looking
wrong. Prove they agree instead of assuming: same data through both paths, bit-identical metrics.

## 26. "Go" approves the config you described. Changing it is a NEW decision that needs a NEW yes

A run was approved as `lr=3e-5` with the configuration just described to the user. It was then launched with an
extra eval horizon the user had never seen, which silently changed the eval subset for every horizon and made
the whole run's numbers incomparable to the sweep that chose the learning rate. That was killed and relaunched.
Then a diagnostic suggested a deeper training horizon, and it was killed and relaunched AGAIN with the horizon
doubled — which changes the objective and doubles the cost of a multi-hour job.

Neither change was asked for. Both were defended afterwards with real findings, and the findings being real is
exactly what makes this seductive: a good reason to change something is not authorisation to change it.

The rule:

- **Launch what was approved.** Not an improved version of it. If a better idea arrives between approval and
  launch, say it and wait — that costs one message.
- **A running job that the user approved is not yours to kill.** Report the problem, state the cost of
  continuing versus restarting, and let them choose. "I found a measurement bug" is a reason to ASK, not a
  mandate to act; so is a diagnostic that implies a better hyperparameter.
- **Evidence from a toy setup does not license a change to a real run.** The horizon change was justified from
  a dim-96 model, two epochs, one shard, on CPU. Two minutes into the relaunch, a diagnostic on the real model
  showed the latent norm had DOUBLED under the new horizon — the opposite of an improvement, and information
  that did not exist when the decision was made.
- **Every restart spends the user's money and their patience.** Two restarts here cost ~10 minutes of an H100.
  The data loss was zero, which is not the point: the user lost the ability to trust that what they approved is
  what ran.

If a change is genuinely required before the job can produce a valid result, say precisely that, name the cost
of each option, and stop talking until answered.

## 27. Every result is provenanced AS IT HAPPENS, or it is lost

This repository is private research; the papers will be written elsewhere, from **`PROVENANCE.md` alone**. The
failure mode is not losing the numbers — it is keeping the numbers and losing what produced them, so a claim
cannot be defended months later. That already happened on an earlier project, which is why the reference
template exists.

**The reference template is `../neural-simulators/PROVENANCE.md`.** Read it before adding a section here. Its
shape, which this repo now follows:

- **Evidence classes on every row.** **M** measured with a recoverable artefact · **Q** qualitative (a watched
  episode, a screenshot) · **I** inferred · **A** author-confirmed · **U** unverified. A **U** row is a promise,
  not a result, and `PROVENANCE.md` §5 is the list of them.
- **W&B is exported, never linked.** `scripts/export_wandb.py --history` writes
  `results/<run_id>/{run.json,config.json,summary.json,history.csv}` for every group, and `run.json` records
  **all 20 rank run-ids** so a fleet is recoverable and not just its aggregator. A claim cites the directory.
- **Artefacts are identified by md5**, not by filename: `scripts/make_samples.py` writes
  `artifacts/MANIFEST.json`. Checkpoints stay gitignored; the manifest is what lets a stray `.pt` be matched to
  a run later.
- **`samples/`** holds committed excerpts *with their field names*, so the state layout is legible without the
  loader.
- **`paper/CLAIMS.md`** is one row per claim a paper could make, with its class and its artefact. Fill the row
  when the measurement happens, not when the paper is written.

### What to write down at the moment it happens, because it is unrecoverable later

1. **The resume lineage.** Not "we trained Gohma" — *which* checkpoint each version started from, per rank, and
   what was widened. Thirteen Gohma attempts across three sessions were nearly guesswork.
2. **Wall clock and cost**, from W&B timestamps plus the fleet kill log, while the pods still exist.
3. **Step-axis discontinuities.** `global_step` restarts at 0 on a resumed fleet run; without that written down
   the curve is misread. (`deku-combat-gru-v2b`: wandb step *S* ≈ global update 30 + S/14,400.)
4. **Why a version exists.** Each numbered attempt is a stop, a review and a deliberate change. Record the
   change AND the result, including the ones that failed — `gohma-v1`'s "held C never fires" is worth more than
   its reward curve.
5. **What the number is NOT.** "10/10 slots" was one watched episode per slot: **Q**, not a rate. Say so in the
   same breath, or someone will cite it as one.
6. **Retractions stay in the file.** Three conclusions this project published were artefacts of the instrument
   rather than the model (scoring a gate on a metric the gate excludes; measuring a grid by re-fitting instead
   of reading `ck["grid"]`; a threshold rule derived from a normalisation bug). Each is recorded WITH its
   retraction, because the correction is the reusable part.

### Tracks are provenanced separately

`PROVENANCE.md` has an **RL** track and a **WM** track. They share a game, a RAM reader and
`zelda_rl/state_layout.py`, and nothing else. **A claim about one is not evidence about the other**, and a
single "Zelda results" section would invite exactly that error.

### The check

Before ending any session that produced a number: does `PROVENANCE.md` let someone else re-derive it without
asking you, and does `paper/CLAIMS.md` say what class it is? If not, the session is not finished. Regenerate the
inputs with:

```sh
.venv/bin/python scripts/export_wandb.py --history    # results/<run_id>/ for every W&B group
.venv/bin/python scripts/make_samples.py             # samples/ + artifacts/MANIFEST.json
```

## 28. On a machine running emulators, pin torch to ONE thread or it starves them

`torch.set_num_threads` is set in **27 scripts in this repo** and the pattern is not decoration. This Mac has 8
logical / **4 performance** cores; each Dolphin instance needs about 2 threads (emulation + the Null GPU
thread); torch defaults to taking **all 4 performance cores** for a forward pass. So every policy step steals
the cores the emulator needs, the VI counter stops advancing, and after the Null backend's 10-second
`stall_timeout` the env raises `emulation stalled (VI stuck at N)`.

**The incident.** Three scripts added in one session (`eval_fleet.py`, `measure_shield.py`, `wm_replay.py`) all
omitted it. The symptom looked like flaky hardware, so it was "fixed" twice by lowering the env count — 8 envs
to 4, then 4 to 3 — and once by adding stall-recovery that rebuilt the dead env and dropped its episodes. Three
runs were spent blaming the device for a one-line omission, and the recovery code was written to survive a bug
rather than to fix it.

```python
torch.set_num_threads(1)          # before building any env
```

`train_combat.py` exposes it as `--torch-threads`; the collectors hardcode 1. **Any new script that opens a
Dolphin sets it.** And when an emulator "stalls", check thread contention before you touch the env count:
`torch.get_num_threads()` against the core count is a five-second test, and lowering the env count hides the
problem while halving throughput.

## 29. A metric that moves with the sampling distribution cannot compare policies

**The incident.** `combat/success_rate` is episode-weighted over the curriculum's start slots. `gohma-band-v11`
showed the best trend of any version — `+0.175/update` overall, **`+0.270/update` over its final third** — and
that apparent momentum is why it was stopped at update 37 to try the inverted curriculum of v12. The per-slot
history says what was actually happening:

| slot | distance | first third | last third | slope |
|---|---|---|---|---|
| **7** | **38 u** | 7.32% | **14.00%** | **+0.298** |
| 3 | 188 u | 0.00% | 0.00% | 0 |
| 6 | 357 u | 0.00% | 0.00% | 0 |
| 2 | 368 u | 0.00% | 0.00% | 0 |
| 8 | 448 u | 0.00% | 0.00% | 0 |
| 5 | 472 u | 0.00% | 0.00% | 0 |
| 1 | 473 u | 0.00% | 0.00% | 0 |
| 4 | 541 u | 0.00% | 0.00% | 0 |

v11 was not learning the boss. It was learning the one start where Link begins on top of her, and the
aggregate climbed because that run weighted slot 7 at **50%**. Nine slots sat at exactly zero for 37 updates.

The same instrument then lied in the other direction: v14 and v15 read **flat on pooled success** while being
the first versions to win from *any* distance — v15 is non-zero on all six trained slots where v11 was
non-zero on one, and off the easy slot v15 averages ~3.3% against v11's ~0.13%, about **25× broader**. Three
version decisions were made from this number and all three were misread.

**The rule.** Any headline that averages over a distribution you control must be reported next to a
**mix-invariant** form of itself, and the mix-invariant form is what a version-to-version claim cites.
`zelda_rl/combat.band_metrics()` emits `combat/success_balanced` (unweighted mean of the per-slot rates),
`combat/success_hard` (the same with the easiest band dropped), `combat/slots_nonzero` (how many starts it
wins from at all — v11 scored 1, v15 scores 6) and `by_band/*` by engagement distance. The trainer's log line
carries `bal hard near mid far nz` beside `succ` so the misleading number never appears alone, and
`tests/test_band_metrics.py` **asserts** the invariance on v11's real skill profile: identical per-slot skill
under two mixes gives pooled rates 3.89 points apart and the same `success_balanced` to 1e-9.

Corollary, and the reason this matters beyond reporting: **PPO optimises expected return under the sampling
distribution, so the curriculum weights are part of the objective.** v16 spent 30% of its episodes on the slot
it already won 28% of the time and 13% on the one it won 0.5% of. If the headline is mix-invariant, the mix
itself becomes a free variable you can tune (`--slot-sampling deficit`).

## 30. A selector is not a selector until its statistic is shown to have resolution

**The incident.** The held-out evaluator ranked checkpoints to answer "which policy do we keep". It ranked on
**16-episode** held-out subsets. Pooled held-out rate over every v14 evaluation was **7/416 = 1.68%**, at which
a 16-episode subset reads 0.00% 76.2% of the time and **6.25% 20.9% of the time**. So the 6.25% that crowned
`gohma-kl_r9_u00040.pt` was *one lucky win in sixteen* — an outcome any checkpoint produces one time in five,
and across 26 evaluations several such rows were near-certain. Tested directly: the selected checkpoint's
2/32 against every other checkpoint's 5/384 gives **p = 0.094, not distinguishable**. Separating even a
genuinely 2× better checkpoint would need **~256 held-out episodes each**, 16× what it had.

Worse, the symptom was *seen and worked around*. The log had already shown `>>> NEW BEST on held-out slots:
0.00%` — every checkpoint scoring zero, so a strict `>` never fired and whichever arrived first stayed best
forever. The response was to add a `boss_hp_taken` tiebreak, not to conclude the statistic was unusable. §1
says run the control; the control ran, it failed, and a workaround shipped instead of a retraction.

**The rules.**

1. Before a statistic selects anything, compute **what it reads under the null**: the pooled rate, and the
   probability of each value its n can produce. If the winning value is a likely draw from the null, there is
   no selector — there is a lottery with a percentage printed on it.
2. **State the n the selector needs** and compare it to the n it gets. If the gap is 16×, the design is wrong,
   not under-powered.
3. Prefer a **graded** statistic over a rate when the rate is sparse. `boss_hp_taken` ranges 0.5–4.7 on 32
   episodes where the defeat rate cannot leave {0, 1/32, 2/32}.
4. **Select the run, not the checkpoint,** when the per-checkpoint n is unreachable. Training logs gave 2,629
   episodes in ten updates on six slots; the evaluator gave 32 held-out episodes per checkpoint.
5. A selector whose output nothing reads is not a selector. `.best.json` was written by the evaluator and read
   by **nothing** — not by the trainer, not by the relaunch, not by hand. Both resumes picked `latest.pt` by
   recency while a "best" file sat beside it. Wire the loop or delete the half of it that implies it exists.

## 31. Use the instrument you already have before you wait on a slower one

**The incident.** Asked whether v16 was improving, the answer was "~40 more minutes" — the time for the eval
fleet to produce enough 40-episode evaluations. Meanwhile the training run had already played **2,629
episodes** across six slots and logged every one, 65× the evidence, available immediately. The eval fleet is
the right instrument for a *held-out* claim and is slow by construction (40 episodes per checkpoint, one
checkpoint per 5 updates). It is the wrong instrument for "is it learning".

`scripts/train_compare.py` now answers the fast question from the training logs (`--exact`, straight from the
pods) or from W&B at matched updates; `scripts/eval_compare.py` answers the held-out question. Ask which
question is being asked before choosing which to wait for.

## 32. Pooling evidence is a measurement, and it has its own failure modes

Writing `scripts/eval_compare.py` — which only replaced inline analysis that had been retyped on every status
request — exposed **three** errors in that hand-pooling, all of which had moved p-values toward significance:

1. **Retracted rows were counted.** 2,831 episodes from the withdrawn round-based evaluator were in the pool.
   Directory markers alone could not stop it: `results/` was never in `CODE_EXCLUDES`, so every launch rsynced
   the Mac's withdrawn rows *onto* the pods, and pulling results back nested them one level deeper, outside
   the marker. Rows are now rejected **by content** — unequal per-slot counts, or fewer slots measured than
   the row lists, neither of which the fixed evaluator can produce.
2. **Duplicates.** The same evaluation existed in a committed copy, a re-pull and the pod's own jsonl; v14
   pooled to 1,551 episodes where it ran 880. Deduplicated on `(checkpoint, rank, ts)`.
3. **A subset posing as the baseline.** The quoted v14 baseline was 559 episodes of an 880-episode run,
   because the later evaluations lacked a `wandb_group` and fell into an unlabelled bucket. The corrected
   comparison moved v15 from p=0.122 to **p=0.087** — the same v15 data against the whole baseline.

**The rule.** The pooling step is code, with tests and markers, not something retyped per question. Mark
retracted data so it cannot be re-admitted (`.withdrawn`), label rows that predate a provenance field
(`.run_label`), deduplicate on an identity tuple, and print what was skipped and why on every run.

## 33. The same three operational rules broke again, and what finally stopped them

This session repeated §15 and §24 verbatim, hours after citing both to the user:

- **§15.** `fleet bootstrap` without `--assets-s3` and without usable seeds falls to an `else` branch that
  pushes the NAND and ten savestates **from the Mac** — 446 MB per pod at ~0.7 MB/s. All 462 MB were already
  in S3, with the Dolphin tarball and a prebuilt venv. Cost: ~25 min of a 19-pod fleet, ~$4, for an upload
  that was never needed. The flag was in `--help` the whole time.
- **§24.** The same bootstrap was piped through `| tail -18`, which buffers until the command exits, so its
  log was empty for 25 minutes while it ran.

Prose did not stop either one. What did:

- **`assets_in_s3()`** HEADs the asset keys once (cached) and `fleet bootstrap` now **takes the S3 path
  automatically** when they are present — there is no flag left to forget. If S3 is missing assets and no
  seeds are given it **refuses with exit 2**, printing the minutes the upload would cost and the three ways to
  fix it. `--push-from-mac` is the deliberate override.
- **`fleet ckpt`** puts an arbitrary training `--init-from` or `--kl-anchor` checkpoint on every pod by
  presigned GET and **verifies by `stat`-ing the path the trainer will open** (8.2 MB × 19 pods in 15 s).
- Long jobs write to a file with `-u` and are read with `grep`/`tail` afterwards. **Never pipe a long-running
  command into `tail` and expect to watch it.**

The pattern across §21 and this section is now unambiguous: **every rule in this file that is still prose will
be broken again.** When a rule is cited in a message, the next question is whether it exists as a check, and
if not, writing the check is part of the work.

## 34. Choose the metric by its INFORMATION, not by its name — and the headline is what the reward pays for

**The incident, which cost two runs and nearly a third.** Asked repeatedly "is it improving", the answer came
from `combat/success_rate` — boss defeated, a binary event occurring 1–10% of the time. On that signal, six
consecutive versions read "flat", "plateaued", "not learning", and two experiments were launched to fix the
plateau. The graded signal was saying the opposite the whole time. Boss damage, the HP actually taken off her
out of 10, over the same runs:

| version | `g_approach` | boss damage | slope |
|---|---|---|---|
| `gohma-v10` | — | 0.38 → 1.21 | **+0.003\*** |
| `gohma-band-v11` | — | 1.72 → 2.10 | **+0.015\*** |
| `gohma-band-v12` | — | 2.06 → 2.22 | **+0.008\*** |
| `gohma-band-v13` | — | 2.41 → 2.75 | **+0.007\*** |
| `gohma-kl-v14` | **0.0 (absent)** | 2.86 → 3.22 | **+0.012\*** |
| `gohma-kl-v15` | 0.5 | 3.17 → 3.32 | +0.004\* |
| `gohma-kl-v16` | 1.5 | 3.12 → 2.85 | **−0.017\*** |
| `gohma-kl-v17` | 0.5 + deficit sampling | 3.19 → 2.86 | **−0.022\*** |

`*` = slope distinguishable from zero. Return tells the same story: **−12.6 → +56.8** across v10–v15.

**The project had been improving steadily the entire time** — 0.38 → 3.32 HP, an 8.7× gain, every step
statistically solid — and it was reported as six failures because the instrument was a rare binary. A defeat
requires ten successful hits; damage counts each one. One of them can show a trend on a few hundred episodes
and the other cannot.

**Worse, the wrong metric then caused real damage.** `g_approach` was introduced in v15 on the strength of a
defeat-rate band analysis, and the dose-response above is monotone in its coefficient: v14 without it was the
last version learning, 0.5 halved the slope, 1.5 reversed it. **The term I added to fix the plateau is what
created it.** v16 and v17 are both regressions, confirmed on the mix-invariant per-slot form (v16 t=−4.75,
v17 t=−3.78), so neither is a curriculum artefact.

**The rules.**

1. **Rank candidate metrics by information per episode before choosing one.** A binary at rate *p* carries
   `p(1−p)` of variance per episode; a graded quantity over 0–10 carries far more. If the headline cannot
   move detectably over the run length you can afford, it is the wrong headline whatever its name.
2. **The headline must be the quantity the reward's largest terms pay for.** `g_boss_damage` is +4.0 per HP
   and `g_boss_defeat` +60 — damage is most of the achievable return, so damage is the headline.
   `combat/boss_damage_balanced` and `combat/boss_damage_hard` are emitted by
   `zelda_rl/combat.band_metrics()` and **lead the trainer's log line**, before any success rate.
3. **Report the trend WITH its standard error.** "Flat" and "rising" are claims about a slope, and a slope
   without an SE is an impression. `slope ± SE`, and say `|t|>2` or do not say "improving".
4. **A sparse metric cannot justify adding a reward term.** Before a new term ships, check the graded signal
   it is meant to help. Had boss damage been read in v15, `g_approach` would never have been added.
5. Everything in §29 still applies on top: the headline must be **mix-invariant** as well as informative.
   Episode-weighted boss damage falls when the curriculum moves to the hard slots at identical skill, which
   is why `boss_damage_balanced` is the unweighted mean of the per-slot values and is asserted mix-invariant
   in `tests/test_band_metrics.py`.

## 35. An ablation's WINNER needs the same statistical care as its measurements

§30 said a selector is not a selector until its statistic has resolution. That was written about choosing a
**checkpoint**. The identical error was then made twice more on the same day, choosing an **arm**:

**The incident.** A 12-arm grid (4 credit horizons × 3 learning rates) was run one pod per arm and ranked on
the **last-3-update mean**, i.e. ~40 episodes. The winner read **3.16** boss damage against a control's 2.40,
a +32% gap, and was scaled to the full fleet — where the same settings read **~2.5**. Confirmed properly as
two matched arms from one init, 7 pods vs 6: **+0.065 ± 0.117, t=+0.56**. The +32% was **entirely** the
upward bias of taking the maximum of twelve noisy estimates. The grid cost ~$7 and its *mechanism* finding
was sound; only its ranking was fiction.

**The rules.**

1. **The maximum of N noisy estimates is biased upward, and the bias grows with N.** A 12-arm grid read on 40
   episodes per arm will always produce an impressive-looking winner. Before quoting it, ask what the spread
   of the arms would look like if every arm were identical.
2. **Rank on the slope over all updates with its SE, never on a last-k mean.** A last-k mean throws away most
   of the data and keeps the noisiest part.
3. **A grid selects a REGION, not a cell.** Trust "long horizons need low LR" (monotone across 12 cells,
   2.7× effect); do not trust "9 s at 3e-5 is the best cell".
4. **Confirm the winner at matched n against a control, from the same init, before acting on it.** Budget for
   the confirmation when planning the grid — it is not optional, and here it reversed the conclusion.
5. **Pre-register the decision rule**: the winner must beat the runner-up by more than the combined standard
   error, or the result is a tie and the grid chose a region only.

## 36. The 2026-10-16 streak: eleven failures, one shape — a check that did not exist, or one that lied

This section exists because §21 and §33 both said "a rule that is only prose will be broken again" and then
**eleven** things broke in one session anyway. The money is the part worth remembering: **~30 of 35 pods sat
dead and billing at ~$16/hr**, the 17 sweep arms crashed at 08:59 and were not noticed until 09:2x, and that
is the *small* number next to the 88-minute bootstrap stall (§20) and the lost 34-pod fleet.

Each row names the check that now exists. **A row without a check is a row that will come back.**

| # | what failed | the money/time | the check that now stops it |
|---|---|---|---|
| 1 | Short episode (< one chunk) hit `W[-n:] = [...]*n` with n == 0; `W[-0:]` is `W[0:]`, so one episode wiped every weight collected. `keep` came back with **19** rows beside `obs` with **6890** → `IndexError: index 6538 ... size 19` | **17 of 22 sweep arms, ~$9/hr of dead pods** | weights appended per window in the same loop; `load_rvs_windows` asserts all six keys equal-length before returning; `merge_capped` asserts again after every Dyna merge; `tests/test_rvs_data.py` runs the n == 0 case |
| 2 | `tests/test_rvs_data.py` used pytest fixtures. **There is no pytest in this venv**, so the test written to catch #1 ran *nothing* and reported nothing | the fix shipped unverified | `tests/run_all.py` runs every `tests/*.py` and **fails on a test that collects nothing or imports pytest** |
| 3 | The loader printed `solved multiplier 0.88 -> winning play is 0.0% of the gradient`. `win_share_actual` counted weights `> 1.0`; with wins at 11.2% of steps and `--rvs-win-share 0.1` the multiplier is **below 1**, so every winning window failed the test | a fabricated number in every arm's log | the share is computed from the win **mask**, not from a threshold on the weight |
| 4 | 5 hybrid arms died on `No space left on device` writing a savestate, invisibly | 5 arms | `disk` is a `fleet preflight` check (≥4 GB), and `fleet launch` reclaims `runs/*/instances`, `wandb`, `runs/*.snaps` |
| 5 | Nothing reported pod liveness across experiments; "everything is dead" was discovered by the user, twice | ~30 pods × ~34 min | `scripts/fleet_health.py`: every experiment with live pods, **crash vs clean exit**, disk, process count, update count, log freshness, **non-zero exit** |
| 6 | `fleet_health.py` first called the 7 **finished** `gohma-s2` arms DEAD | cried wolf on correct work | a clean exit is matched by its own completion marker before any traceback scan |
| 7 | The whole fleet was lost with **every arm checkpoint on it** (RunPod balance exhausted), because `fleet sync` is manual | a session of training | `zelda_rl/ckpt_s3.py` uploads at `torch.save` time, daemon thread, and logs `CHECKPOINT BACKUP FAILED ... weights exist ONLY on this pod` |
| 8 | `fleet ckpt` **deduped by file size**, so a byte-identical reorder never reached the pods | two restarts ran old code | after any push, `grep` the pod for a string unique to the new code — §16, applied: the relaunch helper verifies the fix is present and **refuses to launch the pod if it is not** |
| 9 | `eval_fleet.py --ckpt` returned before the sinks were opened, so `--dump-steps` and `--per-episode-out` were silently inert | the v15 failure-anatomy run produced no rows | the `--ckpt` early return is below sink setup |
| 10 | `grep … | tail` on a long job buffers; a healthy run looked hung — **§24, in scripts written the same night** | twice | long jobs write to a file with `-u` and are read afterwards; never pipe one into `tail` |
| 11 | zsh `set -- $VAR` does not word-split, so three launches passed **empty args** | three launches with no slots | arrays, and the launcher echoes its own resolved argv before exec |

**The honest finding.** Rows 1, 3, 8 and 9 are all one error: **I trusted a success signal instead of the
artefact.** The loader said "6890 windows"; `keep` had 19. rsync exited 0; the file was not there. `--ckpt`
returned 0; no rows were written. The log said 0.0%; wins carried 10%. §16 is the oldest rule in this file
about exactly that, and it keeps being read as being about rsync specifically. It is not:

> **Every number a program prints about its own output is a claim, not evidence. Check the artefact: the
> array's shape, the file on disk, the row count, the string in the deployed file.**

Rows 2 and 6 are the second shape, and it is worse than having no check: **a check that cannot fail is a
check that lies.** The test for #1 passed by running nothing. The health script's first version failed on
seven pods that had done their job. So, before a check is trusted:

1. **Run it against the broken state** and watch it fail. If it cannot be made to fail, it is decoration.
2. **Run it against the healthy state** and watch it pass. A check that flags correct work gets ignored
   within a day, which is the same as not existing. (§22 said this about control loops; it is true of every
   check.)
3. Checks live where they execute — `tests/`, `fleet preflight`, `fleet_health.py` — **not in a new document.**
   This file gets five more lines, not a new MD. Prose does not run (§21), and a document nobody opens is
   worse than prose: it is prose that feels like work.

## 37. The question was already answered on the pods — four times asked, four times deferred

§36 is about code that failed. This one is about **me**, and it cost more than any of the bugs in it.

**The incident.** "Is there ANY signal? Has ANY formula done anything?" was asked four times over ~40
minutes. The answer was in every pod's log the whole time: 23 arms, their per-update boss damage, and five
replicates of the control to compare against. Each time, the reply was that a legitimate read needed
"~45 more minutes" of training. When the comparison was finally done **at matched updates** it took 90
seconds and said, unambiguously, **no effect at any coefficient** (|t| < 1 on all three, control spread sd
0.380 larger than every arm difference). Three of the four deferrals were wrong, and the fourth was only
right about the *confirmation*, not about the *read*.

Worse, the 40 minutes were not spent thinking — they went on three self-inflicted parse bugs:

- `grep 'upd '` matched only the repeating **header** line, so the first pass reported `n=0` for all 23 arms
  and looked like "no data yet";
- the `steps` column prints thousands separators (`566,400`), so a `(\S+)` field regex failed on every row;
- the throwaway script was named `signal.py`, which **shadowed the stdlib module** and broke `subprocess`
  inside `fleet.cli`.

**The rules.**

1. **A question asked twice is a script.** §31 already said "use the instrument you already have before you
   wait on a slower one" and it was cited in this very session while the slower instrument was being waited
   on anyway. `scripts/fleet_signal.py` now prints every arm's slope ± SE and the matched-update comparison
   against the controls, pooling FedAvg ranks into one row. It is the answer to "is anything working", and it
   takes seconds.
2. **"We need N more minutes" is a claim that must be checked against what is already on disk.** The honest
   form is: "the confirmation needs N minutes; here is what the data says NOW, with its n." Never the second
   half omitted.
3. **A throwaway script is still code.** Name it so it cannot shadow a stdlib module, and parse the format by
   looking at one real line first instead of guessing the regex.
4. **Report the null the moment it is significant enough to act on.** The sweep's null at 11 updates was
   decision-grade — it freed 23 pods — and 45 minutes of confirmation would not have changed it (§9: ablate
   before you scale; the corollary is stop paying for a scaled-up null).

### The other half: a design error that made half the grid worthless

The grid was 4 coefficients × 3 win-shares × 2 Dyna settings. **`--rvs-coef 0` disables RvS entirely, so
`win_share` and `dyna` are meaningless at that coefficient and all six `c=0` cells are the SAME config.**
Six of 24 pods ran one config, and by chance those were the pods that survived the ragged-buffer crash — so
the first status report after the crash was five replicates of the control, announced as "5 arms alive".

They turned out to be the most valuable arms in the sweep, because five replicates of one config are the
control spread, which is what killed every cell-level claim. That is luck, not design.

**The rule: before launching a grid, write out its cells and delete the degenerate ones.** A cell whose
other axes have no effect at that setting is a duplicate, and N duplicates are a control-spread measurement
you should decide to run deliberately, not discover afterwards (§35: a grid selects a region, and its
replicate spread is the thing that says whether any region is distinguishable at all).
