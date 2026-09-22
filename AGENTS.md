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
