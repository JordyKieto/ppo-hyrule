# Reward provenance: which reward trained which reported policy

`zelda_rl/combat.py` is edited between experiments. A sentence in a paper that reads the *current* file
describes whatever the reward happens to be today, not what any reported policy was trained with. This file
pins the exact configuration per policy, resolved **from git** rather than from recollection, so a claim about
v15 stays a claim about v15 after the file moves on.

Regenerate with `scripts/reward_provenance.py`, which parses `CombatConfig`'s dataclass defaults and the
preset dictionaries out of each commit with `ast` and resolves `defaults + preset`. It does not import the
module, so it works on commits whose code no longer runs against the current package.

## The table

Every value is the **resolved** coefficient actually in force for that policy: the dataclass default, with the
named preset's overrides applied. A dot means the field did not exist at that commit.

| term | Deku Baba G150 | gohma v10 | gohma v11-v13 | gohma v14-v15 | gohma v16 | gohma v17-v18 | gohma-beh |
| commit | `01b231c` | `1317d4d` | `0ea8406` | `c9fd944` | `0887028` | `f3f3e37` | `WORKING` |
| preset | defaults | defaults | GOHMA_SIMPLE | GOHMA_SIMPLE | GOHMA_SIMPLE | GOHMA_SIMPLE | GOHMA_BEHAVIOUR |
|---|---|---|---|---|---|---|---|
| `g_boss_damage` | . | 4 | 4 | 4 | 4 | 4 | 20 |
| `g_boss_defeat` | . | 60 | 60 | 60 | 60 | 60 | 60 |
| `g_boss_hp_progress` | . | 2 | 2 | 2 | 2 | 2 | 2 |
| `g_seed_stun` | . | 15 | 15 | 15 | 15 | 15 | 5 |
| `g_seed_stun_cap` | . | 60 | 60 | 60 | 60 | 60 | 15 |
| `g_nut_stun` | . | 2 | 2 | 2 | 2 | 2 | 2 |
| `g_stun_hit` | . | 2 | 2 | 2 | 2 | 2 | 2 |
| `g_stun_on_convert` | . | . | on | on | on | on | on |
| `g_red_eye_seed` | . | 4 | 0 | 0 | 0 | 0 | 0 |
| `g_sling_ready` | . | 1 | 0 | 0 | 0 | 0 | 0 |
| `g_unfired_red_eye` | . | 1.5 | 0 | 0 | 0 | 0 | 0 |
| `g_approach` | . | . | . | 0.5 | 1.5 | 0 | 0 |
| `g_approach_cap` | . | . | . | 3 | 7 | 3 | 3 |
| `w_approach` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 5 |
| `w_approach_cap` | . | . | . | . | . | . | 10 |
| `g_spacing` | . | . | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| `g_sword_ready` | . | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 5 |
| `g_sword_ready_cap` | . | 1 | 1 | 1 | 1 | 1 | 10 |
| `g_larva_kill` | . | 2 | 2 | 2 | 2 | 2 | 2 |
| `g_grass_cut` | . | 0.2 | 0 | 0 | 0 | 0 | 5 |
| `g_grass_cut_cap` | . | 2 | 0 | 0 | 0 | 0 | 10 |
| `g_seed_pickup` | . | 0.05 | 0 | 0 | 0 | 0 | 5 |
| `g_nut_pickup` | . | 0.15 | 0 | 0 | 0 | 0 | 5 |
| `g_pickup_cap` | . | 6 | 0 | 0 | 0 | 0 | 10 |
| `w_lockon` | 0.3 | 0.3 | 0 | 0 | 0 | 0 | 5 |
| `w_lockon_cap` | . | . | . | . | . | . | 10 |
| `w_lock_step` | 0.005 | 0.005 | 0 | 0 | 0 | 0 | 5 |
| `lock_step_cap` | 1 | 1 | 1 | 1 | 1 | 1 | 10 |
| `g_link_heal` | . | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 5 |
| `g_link_heal_cap` | . | . | . | . | . | . | 10 |
| `g_facing` | . | . | . | . | . | . | 5 |
| `g_facing_cap` | . | . | . | . | . | . | 10 |
| `g_item_approach` | . | . | . | . | . | . | 5 |
| `g_item_approach_cap` | . | . | . | . | . | . | 10 |
| `g_nut_wasted` | . | . | . | . | . | . | 5 |
| `g_nut_miss` | . | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

## What the columns say, in words

- **Deku Baba G150** predates the boss entirely. No `g_*` term existed; it ran on the generic combat reward.
- **gohma v10** carries the full red-eye slingshot cluster: `g_red_eye_seed` 4, `g_sling_ready` 1,
  `g_unfired_red_eye` 1.5, plus pickups and grass. This is the configuration measured at 88% of total reward
  magnitude over 10,929 episodes.
- **v11 to v13** are the `simple` preset: that cluster zeroed, pickups and grass zeroed, lock-on zeroed, and
  `g_stun_on_convert` turned on so a stun pays only when it is cashed into damage.
- **v14 and v15** add `g_approach` at 0.5, cap 3. **This is the policy every headline number in the paper
  refers to.** Its reward is `simple` plus that one term.
- **v16** triples it to 1.5, cap 7. **v17 and v18** zero it again.
- **gohma-beh** is the 2026-11-01 experiment and **is not a reported result**. It is the only column with
  `g_boss_damage` 20, `g_seed_stun` 5 cap 15, and the eleven behaviour terms at 5.0 cap 10.

## The rule this file exists to enforce

**A paper sentence about a reward must name the policy and the commit, or quote this table.** The current
draft says the boss configuration "pays principally for damage dealt (+4.0 per health point) and the kill
(+60)". That is true of v10 through v18 and false of `gohma-beh`, where damage is 20. The sentence is correct
today only because the rescale went into a separate preset; it would have been silently wrong if the
coefficient had been edited in place.

Two things that are easy to get wrong and are worth stating:

1. **`g_approach` and `w_approach` are different terms.** `w_approach` is the generic approach reward, and
   the boss tracker never calls the generic step function, so it fired only as `approach_stunned`.
   `g_approach` is the boss-specific net-progress term added in v15. The table lists both.
2. **A cap of 0 means uncapped**, not disabled, for the three cap fields added on 2026-11-01
   (`w_lockon_cap`, `w_approach_cap`, `g_link_heal_cap`). They default to 0 so that every historical preset
   keeps its original unbounded behaviour. Resolving an old commit and a new one side by side will show the
   field as absent in the old column, which is the honest rendering.
