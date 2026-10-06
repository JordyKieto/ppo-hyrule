# Eval rows from the ROUND-BASED evaluator — kept, with their defect

These were pulled off the two `zrl-gohma-eval` pods on 2026-10-05 before `scripts/eval_fleet.py` was fixed.
They are kept because a retraction is only useful next to the thing it retracts (AGENTS §27).

**The defect.** The evaluator round-robined the start slot **per round**, and a round ran `--num-envs`
episodes. With 12 envs and 5 slots, `--episodes-per-slot 8` actually ran 12 episodes on slot 1, 12 on slot 3,
12 on slot 5, 12 on slot 7 — and **none on slot 9** — while the row's `slots` field still listed all five.

**What is still usable.** The `by_slot` sub-rows are real measurements of the slots they name, at 12 episodes
each. The aggregate `defeat_rate` and `by_tier` are weighted by the wrong counts, and any claim of ten-slot
coverage is false. `v13_holdout.jsonl` is the worst case: all 48 episodes landed on slot 1.

Rows from the fixed, work-queue evaluator go to `v14_full_q.jsonl` so the two are never pooled.
