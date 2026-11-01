"""The five watched behaviours: the `behaviour` preset, and the three new terms it turns on.

Written BEFORE the fleet, because every shaping term this project has had to remove was farmable --
`g_sling_ready` milked for ~150 steps without starting the fight, `g_spacing`'s floor banking its whole cap
with zero boss damage, and `g_approach` banking its entire cap on step 1 from an infinite seed. AGENTS 9:
ablate before you scale. AGENTS 36 rule 1: break the code and watch the check fail, which is how the
first-step-banking case below was chosen.

Each test here fails without the corresponding change:
  * the decomp constants and `_read_items` do not exist at all;
  * `GOHMA_BEHAVIOUR` does not exist;
  * `item_approach`, `facing` and `nut_wasted` pay nothing;
  * `hearts_picked` / `stun_to_sword_steps` / `nuts_wasted` / `boss_live_steps` are not in the episode row.

    .venv/bin/python tests/test_gohma_behaviour_reward.py
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import (ACTORCAT_MISC, EN_ITEM00, GOHMA_BEHAVIOUR, GOHMA_SIMPLE, ITEM00_NUTS,  # noqa: E402
                             ITEM00_RECOVERY_HEART, ITEM00_RUPEE_RED, ITEM00_SEEDS, ITEM00_STICK,
                             ITEM00_WANTED, Collectible, CombatConfig, CombatState, Enemy, _read_items)
from zelda_rl.gohma import GOHMA_NO_CONVERT, GOHMA_STATE, GohmaTracker  # noqa: E402
from zelda_rl.obs import PLAY_STATE  # noqa: E402

UPD = 0x801E75F8
FN = {v: UPD + k for k, v in GOHMA_STATE.items()}
FACE_HER = 49152          # yaw whose forward is -x: Link at (+d,0,0) then faces a Gohma at the origin
FACE_AWAY = 16384         # forward +x: facing directly away from her

# The seven positive slingshot terms removed on measurement (shaping was 88% of reward magnitude over 10,929
# episodes). The `behaviour` preset must NOT restore any of them -- that decision is recorded in the preset's
# comment block and this is the check that it stays true if someone edits the dict.
RED_EYE_CLUSTER = ("g_red_eye_seed", "g_sling_red_eye", "g_sling_ready", "g_lock_red_eye", "g_elevated_seed",
                   "g_unfired_red_eye", "g_drop_red_eye_shot")


def st(dist=300.0, state="floor_walk", hp=10, yaw=0, link_hp=48, items=None, nuts=10, seeds=30,
       larva_dist=None):
    """Link `dist` units (+x) from a Gohma at the origin, her in `state`, with optional loot and a larva."""
    e = Enemy(0x1000, 0x28, hp, (0.0, 0.0, 0.0), fn=FN[state], yaw_to_player=0, shape_yaw=0, cat=9,
              params=0xFFFF, upd=UPD, focus=(0.0, 60.0, 0.0), yaw=0)
    enemies = {e.ptr: e}
    if larva_dist is not None:                     # a larva the opposite side of Link, so facing differs
        lv = Enemy(0x2000, 0x28, 2, (dist + larva_dist, 0.0, 0.0), fn=UPD - 0x0724, cat=5, params=0,
                   upd=UPD, focus=(dist + larva_dist, 10.0, 0.0))
        enemies[lv.ptr] = lv
    return CombatState(scene=10, room=0, hp=link_hp, hp_cap=48, sf1=0, pos=(dist, 0.0, 0.0), yaw=yaw,
                       targeted=0, arrow=0, enemies=enemies, held_item=0, seeds=seeds, nuts=nuts,
                       items=dict(items or {}))


def item(ptr, dist, kind=ITEM00_RECOVERY_HEART):
    """A collectible `dist` units from the origin along +x (so Link at x walks toward it by closing x)."""
    return Collectible(ptr, kind, (dist, 0.0, 0.0))


def roll(states, **cfg):
    """Feed `states` through a GohmaTracker and return (paid parts, episode row)."""
    t = GohmaTracker(CombatConfig(profile="gohma", **cfg))
    t.reset(states[0], max_steps=5000)
    prev = states[0]
    for cur in states[1:]:
        t.step(prev, cur)
        prev = cur
    return t.ep["r_parts"], t.ep


# ---------------------------------------------------------------------------------------------------------
def test_item00_constants_match_the_decomp():
    """AGENTS 3: never recall an offset or an enum value. These are read off the decomp and pinned here.

      include/actor_profile.h   ACTORCAT_SWITCH=0 .. MISC=8 .. BOSS=9 .. ACTORCAT_MAX=12
      src/code/z_en_item00.c    En_Item00_Profile = { ACTOR_EN_ITEM00, ACTORCAT_MISC, ... }
      include/z_en_item00.h     Item00Type: RUPEE_GREEN 0, BLUE 1, RED 2, RECOVERY_HEART 3, NUTS 0x0C,
                                STICK 0x0D, SEEDS 0x10
    """
    assert ACTORCAT_MISC == 8, ACTORCAT_MISC
    assert EN_ITEM00 == 0x0015, hex(EN_ITEM00)
    assert (ITEM00_RUPEE_RED, ITEM00_RECOVERY_HEART) == (0x02, 0x03)
    assert (ITEM00_NUTS, ITEM00_STICK, ITEM00_SEEDS) == (0x0C, 0x0D, 0x10)
    # A rupee is not loot worth crossing a boss room for; the four that are, are.
    assert ITEM00_RUPEE_RED not in ITEM00_WANTED
    assert ITEM00_WANTED == {ITEM00_RECOVERY_HEART, ITEM00_NUTS, ITEM00_SEEDS, ITEM00_STICK}


def test_read_items_reads_the_misc_list_and_masks_params():
    """The ARTEFACT, not the claim (AGENTS 36): build RAM by hand and read a collectible back out of it.

    Also pins the two things that would silently return nothing: the list index (MISC, not ENEMY) and the
    `& 0xFF` mask -- `EnItem00_Init` does `this->actor.params &= 0xFF`, but a state captured before Init has
    the un-masked spawn params, and a collectible flag in the high bits must not turn a heart into garbage.
    """
    ram = bytearray(0x400000)

    def w32(addr, v):
        struct.pack_into(">I", ram, addr & 0x3FFFFF, v)

    heart, nut = 0x80100000, 0x80110000
    w32(PLAY_STATE + 0x1C24 + 0x10 + ACTORCAT_MISC * 8, heart)       # MISC list head -> heart -> nut
    for a, kind, x in ((heart, ITEM00_RECOVERY_HEART | 0x3F00, 120.0), (nut, ITEM00_NUTS, -40.0)):
        struct.pack_into(">H", ram, a & 0x3FFFFF, EN_ITEM00)
        struct.pack_into(">H", ram, (a + 0x1C) & 0x3FFFFF, kind)
        struct.pack_into(">fff", ram, (a + 0x24) & 0x3FFFFF, x, 0.0, 7.0)
    w32(heart + 0x124, nut)
    w32(nut + 0x124, 0)
    # An enemy in the ENEMY list must NOT come back as a collectible.
    w32(PLAY_STATE + 0x1C24 + 0x10 + 5 * 8, 0x80120000)
    struct.pack_into(">H", ram, 0x80120000 & 0x3FFFFF, EN_ITEM00)

    got = _read_items(ram)
    assert len(got) == 2, got
    assert got[heart].kind == ITEM00_RECOVERY_HEART, hex(got[heart].kind)   # 0x3F00 flag masked off
    assert got[nut].kind == ITEM00_NUTS
    assert abs(got[heart].pos[0] - 120.0) < 1e-6 and abs(got[nut].pos[2] - 7.0) < 1e-6
    assert 0x80120000 not in got, "read the ENEMY list -- wrong actor category"


def test_preset_turns_on_each_of_the_five_behaviours():
    """One assertion per behaviour, so a dropped line in the preset names which behaviour it broke."""
    b, s = GOHMA_BEHAVIOUR, GOHMA_SIMPLE
    base = CombatConfig()
    cur = lambda k: s.get(k, getattr(base, k))
    # 1 items / hearts / grass: three re-enabled channels, a raised heal, one new approach term
    for k in ("g_grass_cut", "g_seed_pickup", "g_nut_pickup", "g_pickup_cap"):
        assert cur(k) == 0.0 and b[k] > 0, f"behaviour 1: {k} should go from zeroed to live"
    assert b["g_link_heal"] > cur("g_link_heal"), "behaviour 1: hearts pay more"
    assert b["g_item_approach"] > 0, "behaviour 1: nothing pays for WALKING toward loot"
    # 2 nut waste, as a state-gated term DISTINCT from the realised-waste g_nut_miss, which is left alone
    assert b["g_nut_wasted"] > 0, "behaviour 2"
    assert b.get("g_nut_miss", base.g_nut_miss) == base.g_nut_miss, "g_nut_miss must be left at its value"
    # 3 lock-on
    for k in ("w_lockon", "w_lock_step"):
        assert cur(k) == 0.0 and b[k] > 0, f"behaviour 3: {k} should go from zeroed to live"
    # 4 facing
    assert b["g_facing"] > 0, "behaviour 4"
    # 5 rush the stun: RAISED, never duplicated
    assert b["w_approach"] > cur("w_approach"), "behaviour 5: approach_stunned's weight"
    assert b["g_sword_ready"] > cur("g_sword_ready"), "behaviour 5: sword_ready_stunned's weight"
    # and the outcome terms move ONLY where the author asked, 2026-11-01. This guard used to assert that the
    # preset touched no outcome term at all; it fired when g_boss_damage was raised, which is the guard doing
    # its job. It is not deleted, it is narrowed to the two deliberate changes, so any OTHER outcome edit
    # still fails here.
    for k in ("g_boss_defeat", "g_boss_hp_progress", "g_death_penalty"):
        assert k not in b or b[k] == cur(k), f"{k} is an outcome term and must stay untouched"
    assert b["g_boss_damage"] == 20.0, "the author raised damage 4 -> 20 so the outcome dominates the shaping"
    assert "g_boss_damage_cap" not in b, "damage stays UNCAPPED"
    assert b["g_seed_stun"] == 5.0 and b["g_seed_stun_cap"] == 15.0, "seed stun cut from 15.0/60.0"

    # every behaviour term is on one scale: coefficient 5, cap 10. g_nut_wasted is the stated exception,
    # uncapped because a capped penalty makes later waste free.
    for k in ("g_grass_cut", "g_seed_pickup", "g_nut_pickup", "g_link_heal", "g_item_approach",
              "w_lockon", "w_lock_step", "g_facing", "w_approach", "g_sword_ready", "g_nut_wasted"):
        assert b[k] == 5.0, f"{k} should be at the common coefficient 5.0, got {b[k]}"
    for k in ("g_grass_cut_cap", "g_pickup_cap", "g_link_heal_cap", "g_item_approach_cap",
              "w_lockon_cap", "lock_step_cap", "g_facing_cap", "w_approach_cap", "g_sword_ready_cap"):
        assert b[k] == 10.0, f"{k} should be at the common cap 10.0, got {b[k]}"
    assert "g_nut_wasted_cap" not in b, "the waste PENALTY stays uncapped, deliberately"

    # the three caps added for this preset default to 0, which the tracker reads as unbounded, so every
    # historical preset keeps its original behaviour. This is what stops the rescale contaminating v15.
    for k in ("w_lockon_cap", "w_approach_cap", "g_link_heal_cap"):
        assert getattr(base, k) == 0.0, f"{k} must default to 0 (unbounded) so older presets are unchanged"


def test_no_positive_red_eye_term_is_restored():
    """The cluster was removed on a 10,929-episode measurement. Re-adding it is re-running a known failure."""
    cfg = CombatConfig(profile="gohma", **GOHMA_BEHAVIOUR)
    for k in RED_EYE_CLUSTER:
        assert getattr(cfg, k) == 0.0, f"{k} is back on at {getattr(cfg, k)} -- see the preset's comment"


# ---- behaviour 1: walking toward loot -------------------------------------------------------------------
def test_item_approach_pays_net_progress_and_cannot_be_farmed_by_oscillating():
    """Straight in and in-and-out must pay the SAME, and the total must be bounded by the start distance.

    The farmable alternative -- a signed per-step delta -- pays the oscillating path 3x here, because
    `_capped` drops negative values so the retreat half is free. That is the exact bug shape that cost this
    project `g_sling_ready` and `g_spacing`.
    """
    loot = {0xA: item(0xA, 0.0)}                    # loot at the origin; Link closes by reducing x
    straight = [st(d, items=loot) for d in (500.0, 400.0, 300.0, 200.0)]
    osc = [st(d, items=loot) for d in (500.0, 400.0, 500.0, 400.0, 500.0, 400.0, 300.0, 200.0)]
    p_straight = roll(straight, g_item_approach=0.3, g_item_approach_cap=99.0)[0]["item_approach"]
    p_osc = roll(osc, g_item_approach=0.3, g_item_approach_cap=99.0)[0]["item_approach"]
    assert abs(p_straight - 0.9) < 1e-6, p_straight                 # 300 units closed x 0.3/100
    assert abs(p_osc - p_straight) < 1e-6, (p_osc, p_straight)
    assert roll(osc, g_item_approach=0.3, g_item_approach_cap=99.0)[1]["item_approach_units"] <= 500.0 + 1e-6


def test_item_approach_does_not_bank_the_cap_on_the_first_step():
    """`_item_best` is SEEDED from where each item is first seen, never from infinity.

    Seeding `_d_best` to infinity made `g_approach`'s first step bank its entire cap with Link standing
    still. The identical bug is one line away here, and a new item appears MID-episode every time a larva
    dies, so it would fire repeatedly rather than once.
    """
    # Link never moves; the item appears on step 2. Nothing may be paid, ever.
    a = st(300.0, items={})
    b = st(300.0, items={0xA: item(0xA, 0.0)})
    parts, ep = roll([a, b, b, b], g_item_approach=0.3)
    assert parts.get("item_approach", 0.0) == 0.0, parts
    assert ep["item_approach_units"] == 0.0


def test_item_approach_ignores_a_rupee_and_is_capped():
    loot = {0xA: item(0xA, 0.0, kind=ITEM00_RUPEE_RED)}
    parts, _ = roll([st(d, items=loot) for d in (900.0, 100.0)], g_item_approach=0.3, g_item_approach_cap=99.0)
    assert parts.get("item_approach", 0.0) == 0.0, "a rupee is not in ITEM00_WANTED"
    loot = {0xB: item(0xB, 0.0, kind=ITEM00_SEEDS)}
    parts, _ = roll([st(d, items=loot) for d in (900.0, 100.0)], g_item_approach=0.3, g_item_approach_cap=1.5)
    assert abs(parts["item_approach"] - 1.5) < 1e-6, parts          # 800 units would pay 2.4; cap binds


def test_hearts_are_counted_once_per_heart_not_once_per_step():
    """A recovery heart raises Link's HP over several frames. Counting rising steps would quadruple it."""
    hp = [40, 44, 48, 48, 48, 44, 48]              # one heart over 2 steps, a hit, then another heal
    _, ep = roll([st(300.0, link_hp=h) for h in hp])
    assert ep["hearts_picked"] == 2, ep["hearts_picked"]
    assert ep["hp_healed"] == 12, ep["hp_healed"]


# ---- behaviour 2: wasted Deku Nuts ----------------------------------------------------------------------
def test_nut_wasted_fires_in_her_zero_conversion_states_and_not_on_the_floor():
    """Derived from the corpus, not invented: 71 impacts in ceiling_idle/ceiling_move, zero conversions."""
    assert "ceiling_idle" in GOHMA_NO_CONVERT and "ceiling_move" in GOHMA_NO_CONVERT
    assert "ceiling_eye_red" not in GOHMA_NO_CONVERT, "it converts 40.6% -- the point of the slingshot"
    for state, want in (("ceiling_move", 1), ("ceiling_idle", 1), ("floor_walk", 0), ("floor_eye_red", 0)):
        a = st(120.0, state=state, nuts=10)
        b = st(120.0, state=state, nuts=9)                          # one nut thrown, well inside nut range
        parts, ep = roll([a, b], g_nut_wasted=0.5)
        assert ep["nuts_thrown"] == 1
        assert ep["nuts_wasted"] == want, f"{state}: wasted {ep['nuts_wasted']}, want {want}"
        assert abs(parts.get("nut_wasted", 0.0) - (-0.5 * want)) < 1e-6, (state, parts)


def test_nut_wasted_fires_out_of_range_on_the_floor():
    """A nut at 190 units while she walked did nothing (live probe); the probe that worked was at 135."""
    cfg = dict(g_nut_wasted=0.5, g_nut_range=220.0)
    near = roll([st(150.0, nuts=10), st(150.0, nuts=9)], **cfg)[1]["nuts_wasted"]
    far = roll([st(600.0, nuts=10), st(600.0, nuts=9)], **cfg)[1]["nuts_wasted"]
    assert (near, far) == (0, 1), (near, far)


def test_nut_wasted_is_uncapped_so_later_waste_is_never_free():
    states = [st(120.0, state="ceiling_move", nuts=10 - i) for i in range(9)]
    parts, ep = roll(states, g_nut_wasted=0.5)
    assert ep["nuts_wasted"] == 8, ep["nuts_wasted"]
    assert abs(parts["nut_wasted"] - -4.0) < 1e-6, parts            # 8 x -0.5, no cap


def test_nuts_wasted_is_counted_even_with_the_coefficient_off():
    """The waste FRACTION has to be readable on the control arm, which runs g_nut_wasted 0."""
    parts, ep = roll([st(120.0, state="ceiling_move", nuts=10), st(120.0, state="ceiling_move", nuts=9)],
                     g_nut_wasted=0.0)
    assert ep["nuts_wasted"] == 1 and "nut_wasted" not in parts, (ep["nuts_wasted"], parts)


# ---- behaviour 4: facing her ----------------------------------------------------------------------------
def test_facing_pays_only_while_facing_her():
    toward = [st(300.0, yaw=FACE_HER)] * 6
    away = [st(300.0, yaw=FACE_AWAY)] * 6
    p_t, ep_t = roll(toward, g_facing=0.01, g_facing_cap=99.0)
    p_a, ep_a = roll(away, g_facing=0.01, g_facing_cap=99.0)
    assert abs(p_t["facing"] - 0.05) < 1e-9, p_t                    # 5 steps after reset
    assert p_a.get("facing", 0.0) == 0.0, p_a
    assert ep_t["facing_steps"] == 5 and ep_a["facing_steps"] == 0
    assert ep_t["boss_live_steps"] == ep_a["boss_live_steps"] == 5, "the denominator must not move with aim"
    assert abs(ep_t["facing_cos_sum"] / 5 - 1.0) < 1e-6
    assert abs(ep_a["facing_cos_sum"] / 5 + 1.0) < 1e-6


def test_facing_cannot_be_farmed_by_oscillating_in_and_out():
    """A per-step condition under a per-episode cap: leaving the condition strictly REDUCES paid steps."""
    held = [st(300.0, yaw=FACE_HER)] * 11
    flick = [st(300.0, yaw=FACE_HER if i % 2 else FACE_AWAY) for i in range(11)]
    p_held = roll(held, g_facing=0.01, g_facing_cap=99.0)[0]["facing"]
    p_flick = roll(flick, g_facing=0.01, g_facing_cap=99.0)[0]["facing"]
    assert p_flick < p_held, (p_flick, p_held)
    assert roll(held, g_facing=0.01, g_facing_cap=0.03)[0]["facing"] <= 0.03 + 1e-9


def test_facing_is_silent_while_she_is_in_a_zero_conversion_state():
    """Gate on her state: staring at her eye-shut ceiling crawl is 67% of the episode and buys nothing."""
    for state in sorted(GOHMA_NO_CONVERT):
        parts, ep = roll([st(300.0, yaw=FACE_HER, state=state)] * 6, g_facing=0.01, g_facing_cap=99.0)
        assert parts.get("facing", 0.0) == 0.0, (state, parts)
        assert ep["boss_live_steps"] == 0, state
    # ...but a RED ceiling eye is exactly the moment that matters, so it must still pay
    parts, _ = roll([st(300.0, yaw=FACE_HER, state="ceiling_eye_red")] * 6, g_facing=0.01, g_facing_cap=99.0)
    assert abs(parts["facing"] - 0.05) < 1e-9, parts


def test_facing_never_pays_for_a_larva():
    """Larva hits correlate -0.598 with damage dealt; a "nearest target" version would pay for the adds."""
    # A larva 50 units further out along +x, so Link facing AWAY from Gohma faces the larva squarely.
    parts, ep = roll([st(300.0, yaw=FACE_AWAY, larva_dist=50.0)] * 6, g_facing=0.01, g_facing_cap=99.0)
    assert parts.get("facing", 0.0) == 0.0, parts
    assert ep["facing_steps"] == 0


def test_facing_is_silent_beyond_range():
    parts, ep = roll([st(1200.0, yaw=FACE_HER)] * 6, g_facing=0.01, g_facing_cap=99.0, g_facing_range=600.0)
    assert parts.get("facing", 0.0) == 0.0 and ep["boss_live_steps"] == 0, (parts, ep["boss_live_steps"])


# ---- behaviour 5: the counter for a term that already exists --------------------------------------------
def test_stun_to_sword_counts_steps_and_separates_unreached_windows():
    """A window Link never reaches must count as a MISS, not as a slow arrival: two different failures."""
    reach = [st(300.0, state="floor_walk")] + [st(d, state="stunned") for d in (300.0, 200.0, 80.0, 80.0)]
    _, ep = roll(reach, w_approach=1.0)
    assert ep["stun_windows"] == 1 and ep["stun_reached"] == 1
    # onset is step 1 (floor_walk -> stunned); Link is inside the 90-unit sword range on step 3. So 2 steps.
    assert ep["stun_to_sword_steps"] == 2, ep["stun_to_sword_steps"]
    never = [st(300.0, state="floor_walk")] + [st(300.0, state="stunned")] * 4
    _, ep2 = roll(never, w_approach=1.0)
    assert ep2["stun_windows"] == 1 and ep2["stun_reached"] == 0
    assert ep2["stun_to_sword_steps"] == 0, "an unreached window must contribute no TIME"


def test_approach_stunned_already_exists_and_is_not_duplicated():
    """Behaviour 5 is a WEIGHT question, not a new term -- the regression guard against adding a second one."""
    states = [st(300.0, state="stunned")] + [st(d, state="stunned") for d in (250.0, 200.0, 150.0)]
    parts, _ = roll(states, w_approach=1.0)
    assert "approach_stunned" in parts and parts["approach_stunned"] > 0, parts
    dupes = [k for k in parts if "approach" in k and k not in ("approach", "approach_stunned", "item_approach")]
    assert not dupes, f"a second approach term appeared: {dupes}"
    # and doubling the coefficient doubles the payment, so the knob is the knob
    half = roll(states, w_approach=0.5)[0]["approach_stunned"]
    assert abs(parts["approach_stunned"] - 2 * half) < 1e-6, (parts["approach_stunned"], half)


# Named explicitly, NOT collected out of globals(): tests/run_all.py statically checks that every test_* it
# can see is referenced, and a comprehension over globals() is invisible to it -- which is how a test file in
# this repo once passed by executing nothing at all (AGENTS 36 row 2). The list is the reference.
TESTS = [
    test_item00_constants_match_the_decomp,
    test_read_items_reads_the_misc_list_and_masks_params,
    test_preset_turns_on_each_of_the_five_behaviours,
    test_no_positive_red_eye_term_is_restored,
    test_item_approach_pays_net_progress_and_cannot_be_farmed_by_oscillating,
    test_item_approach_does_not_bank_the_cap_on_the_first_step,
    test_item_approach_ignores_a_rupee_and_is_capped,
    test_hearts_are_counted_once_per_heart_not_once_per_step,
    test_nut_wasted_fires_in_her_zero_conversion_states_and_not_on_the_floor,
    test_nut_wasted_fires_out_of_range_on_the_floor,
    test_nut_wasted_is_uncapped_so_later_waste_is_never_free,
    test_nuts_wasted_is_counted_even_with_the_coefficient_off,
    test_facing_pays_only_while_facing_her,
    test_facing_cannot_be_farmed_by_oscillating_in_and_out,
    test_facing_is_silent_while_she_is_in_a_zero_conversion_state,
    test_facing_never_pays_for_a_larva,
    test_facing_is_silent_beyond_range,
    test_stun_to_sword_counts_steps_and_separates_unreached_windows,
    test_approach_stunned_already_exists_and_is_not_duplicated,
]


def main() -> int:
    fails = []
    for fn in TESTS:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            fails.append((fn.__name__, e))
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(TESTS) - len(fails)}/{len(TESTS)} passed")
    for name, e in fails:
        print(f"  {name}: {e}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
