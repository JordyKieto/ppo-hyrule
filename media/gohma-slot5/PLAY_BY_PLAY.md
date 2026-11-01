# Play-by-play: v15 defeats Queen Gohma from the full-health start

Episode behind `gohma-slot5_s05_try04_crop.mp4` (md5 `a184dbd29233`, 197.7 s of video) and
`gohma-slot5_s05_try04.mp4` (md5 `3d772cab7cf7`). Policy `checkpoints/kept/gohma-kl-v15-r4_latest.pt`
(v15, update 40), sampled actions, 10 Hz control, 300 s cap, recorded 2026-11-01.

**Start (start 5, verified from the emulator in `runs/gohma_slot_audit.json`):** Gohma at **10 of 10 health**,
on the ceiling with her eye red, **347 units** horizontal from Link, **0 larvae** in the room, Link at 2.5 of
3 hearts with 26 seeds and Deku Nuts.

**Outcome:** `SUCCESS | killed 1/1 at [191.1] s | hearts lost 2.0 | return 231.1`.
Ten sword hits, eight knockdowns, eight larvae engaged and five killed outright, Link down to half a heart.

Machine-readable forms beside this file: `play_by_play.csv` and `play_by_play.json`, one row per HUD tick
(about 2 Hz, plus every tick on which an event fired), with her state, her health, horizontal distance,
larvae alive, lock-on, held item, seeds, return and the event dictionary.

---

## The shape of the fight: four damage windows separated by three long ceiling phases

Phases are segmented on whether she is reachable (falling, landing, stunned, or on the floor) or on the
ceiling where she cannot be hit.

| phase | from | to | duration | her health | sword hits | knockdowns | larvae killed |
|---|---|---|---|---|---|---|---|
| ceiling | 0.0 s | 0.4 s | 0.4 s | 10 | 0 | 0 | 0 |
| **REACHABLE** | 0.6 s | 10.1 s | **9.5 s** | **10 to 5** | **5** | 1 | 0 |
| ceiling | 12.1 s | 52.6 s | **40.5 s** | 5 | 0 | 0 | 2 |
| **REACHABLE** | 53.9 s | 67.4 s | 13.5 s | **5 to 4** | 1 | 3 | 0 |
| ceiling | 69.4 s | 110.3 s | **40.9 s** | 4 | 0 | 0 | 2 |
| **REACHABLE** | 111.4 s | 125.5 s | 14.1 s | **4 to 2** | 2 | 2 | 0 |
| ceiling | 127.5 s | 179.9 s | **52.4 s** | 2 | 0 | 0 | 1 |
| **REACHABLE** | 181.9 s | 191.0 s | 9.1 s | **2 to 0** | 2 | 2 | 0 |

**She is unreachable for 67% of the episode.** All ten points of damage happen inside 46 seconds of the 191,
spread over four windows. The three ceiling phases total 134 seconds in which her health does not move.

---

## Event totals

| event | count |
|---|---|
| sword hits on Gohma | 10 |
| knockdowns | 8 |
| stuns from a Deku Nut | **4** |
| stuns from a slingshot seed | **1** (the opening shot the start hands it) |
| seeds fired | **4**, from 26 carried |
| shots dropped while her eye was red | **3** |
| red-eye windows passed up entirely | 1 |
| Deku Nuts thrown | **13** |
| larvae hit / killed | 13 / 5 |
| times Link was hurt | 32 |
| lock-on acquisitions | 8 |

---

## What an analyst should look at, with the numbers that raise each question

1. **It does not bring her down; it waits for her.** One of the eight knockdowns came from a slingshot seed,
   and that was the opening shot at a red eye the start state hands it at 0.6 s. The other knockdowns are
   her own descent, after which nuts keep her down. The slingshot is the mechanism for *causing* a descent
   and it is used four times in 191 seconds out of 26 seeds carried, with **three shots dropped while her
   eye was red** and one red-eye window passed up. Compare the human corpus: 2.99 seeds per win, and a
   red-eye shot lands 92.9% of the time.
2. **The Deku Nut has replaced the slingshot, not supplemented it.** Thirteen nuts thrown for four stuns.
   Nuts only work while she is on the floor, so the nut is a tool for *extending* a window she is already in,
   and the policy is using it as its primary stun. Across every matched evaluation of this checkpoint, seed
   stuns are 1 per win on this start and 0 per win on the easy start.
3. **Lock-on is used 15% of ticks against the human's 63%.** Eight acquisitions across a three-minute fight.
4. **Larvae cost it time and health.** Thirteen larva hits across the ceiling phases, five killed. In the
   100-win human corpus, larva hits correlate **-0.598** with damage dealt, and time she spends on the
   ceiling **-0.559**. The policy spends the ceiling phases fighting adds rather than shooting her down,
   which is precisely the pattern the corpus says loses.
5. **It is taking a lot of damage.** 32 damage events, 2.5 hearts down to 0.5 at the low point. It never
   collects a heart, and it ends the fight at 1.5.
6. **The damage windows are efficient when they happen.** Five hits in the first 9.5 s window. The problem is
   not conversion once she is down, it is the 134 seconds in which she is not.

The direct implication for a reward change: the binding constraint in this episode is **time she spends
unreachable**, and the lever for that is the red-eye slingshot shot, which is the one thing the policy has
most clearly failed to learn.
