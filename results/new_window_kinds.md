# New event-window kinds: baseline for `joint_v15_exitev`

`scripts/eval_event_windows.py` now covers nine window kinds. Command run:

```
.venv/bin/python -u scripts/eval_event_windows.py checkpoints/dynamics/joint_v15_exitev.pt \
  --horizon 40 --per-kind 20 --seeds 1 --out /tmp/ev_newkinds.json
```

Held-out slots 7 and 9, 81 episodes, 55 of them long enough for a 40-step window.

## Baseline, horizon 40

`detect` is *that kind's own event*, computed by one function (`event()`) for the dream, the real
trajectory and identity. `control` is the dream's rate on 40 random windows where the event did **not**
happen — a false-positive rate. `events` is distinct real occurrences, not window starts.

| kind | windows | events | detect dream | real | ident | control | mag dream | mag real | traj err / identity |
|---|---|---|---|---|---|---|---|---|---|
| hit | 8 | 14 | 12% | 100% | 0% | 0% | 0.12 HP | 1.25 HP | 1.11 |
| link_hit | 20 | 43 | 0% | 100% | 0% | 0% | 0.00 hearts | 0.93 hearts | 1.08 |
| walk | 20 | 99 | 0% | 100% | 0% | 3% | 54 units | 290 units | 1.04 |
| **kill** | 7 | 10 | 0% | 100% | 0% | 0% | 0.00 kills | 1.00 kills | 0.95 |
| **block** | 20 | 58 | 25% | 100% | 0% | **20%** | 0.25 events | 1.25 events | 1.09 |
| **lockon** | 20 | 57 | 15% | 100% | 0% | 0% | 2.0 locked steps | 13.7 | 1.09 |
| **jump** | 20 | 118 | 50% | 100% | 0% | 13% | 1.9 airborne steps | 7.0 | 1.12 |
| **roll** | 12 | 12 | 75% | 100% | 0% | 0% | 13.4 roll steps | 6.3 | 0.89 |
| **collision** | 20 | 45 | 15% | 100% | 0% | 3% | 1.9 wall steps | 11.8 | 0.92 |

Repeated at `--per-kind 40 --seeds 2` (H=40) and `--horizon 20 --per-kind 40 --seeds 2`; every number
above moves by less than its own uncertainty except `roll` magnitude (13.4 steps at H=40, 2.3 at H=20)
and `link_hit`/`kill` detection (0% → 10%/7% at H=20). The baseline run's full per-kind JSON is
`/tmp/ev_newkinds.json`; the two sweeps are reproducible with the flags named above.

`seed wrong` (the dream's own start frame disagreeing with the real one on the flag) is **0% for all four
boolean kinds**, so every miss below is a rollout failure, not a seed-decode failure.

## Which kinds are trustworthy

**Trustworthy (real 100%, identity 0%, ≥10 distinct events, control well below detect):**

- `jump` — 118 events. The strongest new kind: detect 50% against a 13% false-positive rate, and the first
  airborne step lands at step 10.6 against a real 10.9, so when the dream jumps it jumps at roughly the
  right moment. It under-holds: 1.9 airborne steps against 6.2.
- `lockon` — 57 events. detect 15–19% against a 0% control, so the signal is real but recall is poor, and
  the dream holds lock for 2.0 steps against a real 13.6. Lock-on acquisition is nearly absent from the
  dream even though it is a shaping term the reward pays for.
- `collision` — 45 events. detect 12–15% against 3–5% control. Real but weak; 1.9 wall-contact steps
  against 10–12. Wall contact is the one kind where the dream's trajectory beats identity (0.92), which is
  consistent with "the dream drifts less far, so it also hits fewer walls".
- `roll` — 12 events, exactly at the evidence floor. detect 75% at a 0–2% control, the highest detection of
  any kind. But the magnitude is **inverted**: 13.4 roll steps dreamed against 6.3 real at H=40. The dream
  enters the roll state and then stays in it far too long. Treat the direction as real and the number as
  provisional until there are more than 12 events.

**Not trustworthy — too little data, not a broken metric:**

- `kill` — only **10 distinct kills** exist in the held-out set at H=40 (7 non-overlapping windows). 26 of
  the 81 held-out episodes are shorter than 42 steps and they hold 13 of the 23 real defeats, so the
  horizon itself is what starves this kind. The measured 0.00 kills is consistent with every previous
  audit, but 7 windows is not evidence and the script prints a warning saying so.
- `hit` — same problem, pre-existing: 14 distinct hits, 8 non-overlapping windows. The old run reported
  n=20 for `hit` only because up to 40 window starts can point at the *same* HP drop; the deduplication
  added here exposes it.

**Trustworthy metric, worthless signal:**

- `block` — 58 events, real 100%, identity 0%, so the window kind itself is sound. But the dream's
  detection (18–25%) is **inside** its own false-positive rate on random non-block windows (19–23%). The
  dream books a block or dodge at the same rate whether or not one happened, so this column carries no
  information about the model. The cause is visible in the checkpoint's own metrics: `row_state` decodes at
  0.696, and a block is defined by a `lunge → pull_back` state cycle, so spurious cycles are cheap. Until
  `row_state` improves, `block` should be read as a false-positive monitor, not a capability measure.

## The block audit (this had never been done)

Blocks **do** occur in the data. Replaying the real `CombatTracker` over every recorded episode:

| set | episodes | lunges | blocks | dodges | lunge hits taken |
|---|---|---|---|---|---|
| held-out (slots 7, 9) | 81 | 168 | 10 | 48 | 55 |
| all collections | 398 | 899 | 84 | 280 | 352 |

So the block reward sitting at 0.00 in every run measured is **not** an absence in the data — the events
are there, at 84 blocks and 280 dodges across 899 lunge cycles. The window kind is therefore built, and it
counts blocks **and** dodges together (`ev["block"] or ev["dodge"]`), because blocks alone give only 10
held-out events, below the evidence floor. Block-only and dodge-only counts are in the table above if they
are ever needed separately.

## The `link_hit` bug: a reporting bug, now fixed

`link_hit` reported 0% real detection on windows selected for a real Link HP drop. The window logic was
never wrong. The printed table hard-coded `detect_baba` / `detect_baba_real` into the `detect d/r` column
for **every** row, and on `link_hit` windows the Baba loses no HP, so the column correctly printed 0%/0%
for Baba damage and was read as "the Link HP drop is not there". `detect_link_real` was in the JSON at
**1.00** the whole time.

Fixed by making each row's `detect` column its own event, via a single `event(rec, kind)` function applied
three times — to the dream, to the real window and to identity. That makes a broken window definition
visible as a real column below 100% instead of as a plausible-looking wrong number, and the script now
prints an explicit `WARNING <kind>: real detection N% < 95%` when it happens.

## Other changes made for discipline

- **Non-overlapping sampling.** One block generates up to H qualifying window starts, so 58 real blocks
  became 1075 "windows". Sampling now rejects any window within H steps of an already-chosen one in the
  same episode, so two rollouts can never be scored on the same real event, and the header prints distinct
  events, qualifying starts and windows scored as three separate numbers.
- **False-positive control.** Random windows are rolled once and each kind's `event()` is evaluated on the
  ones where the real event did not occur. This is what exposed `block` as noise; without it `block` reads
  as the second-best new kind.
- **Onset selection for the boolean kinds.** A window qualifies only if the flag is *off* at its start and
  turns on inside it. Without that, a window starting with Link already airborne is "detected" for free by
  a frozen world; with it, identity scores exactly 0% on all four, the same bar the HP and travel kinds face.
- **Symmetric flag reading.** The dream's flags come from the decoder heads thresholded exactly as
  `scripts/train_joint.py` scores them (`link_bg > 0`, `argmax link_action`, `argmax targeted`); the real
  flags come from `decode_targets` for the same fields. Prediction against its own training target, nothing
  else. `kill` and `block` are counted by running the reward's own `CombatTracker` (`DreamTracker` on the
  dream side) over both sequences, so they are the numbers the reward would actually pay.
- The `MAX_STEP_UNITS = 28` settling filter applies to every new kind.

## Bottom line

Of the six kinds asked for, four are usable now (`jump`, `lockon`, `collision`, `roll`), one is a sound
metric with no signal to report yet (`block` — and the audit shows the events exist, which is new
information), and one is starved by the held-out set rather than by the metric (`kill`: 10 events).
`jump` is the only new kind where the dream is clearly doing something right; `roll` fires reliably but
for far too long; `lockon` and `collision` fire a sixth of the time they should.
