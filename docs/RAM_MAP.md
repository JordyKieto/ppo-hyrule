# OoT RAM Map (Wii VC NACE → N64 NTSC 1.2)

A full account of what is in the game's RAM that this project reads or could read, and how sure we are of each field.

**Legend**
- ✅ **verified in this build**: read from live RAM or your savestates, with the evidence noted.
- 🔶 **decomp-derived**: struct layout from the OoT decompilation (zeldaret/oot). Offsets inside structs are the same across N64 versions; not yet checked here.
- ❌ **checked and wrong / not usable** in this build.

Conventions: N64 virtual addresses (`0x80xxxxxx`); `addr & 0x3FFFFF` = offset into the 4 MB N64 RAM block; all values big-endian. Offsets like `+0x34` are relative to the struct base named in the section.

---

## 1. Memory layout

| Item | Location | Status | Evidence |
|---|---|---|---|
| N64 RAM (4 MB) | Wii MEM1 offset `0xF64120` (Wii addr `0x80F64120`) | ✅ | exception vector `3C1A8000 275A2600 03400008` at +0; GDB stub read; mach reads |
| `osMemSize` | `0x80000318` = `0x00400000` | ✅ | 4 MB, no Expansion Pak |
| `osTvType` | `0x80000300` = 1 (NTSC) | ✅ | |
| Wii game code | MEM1 `+0x3180` = `"NACE"` | ✅ | used to identify MEM1 views |
| In a savestate | decompressed offset `0x17A14A5` | ✅ | `zelda_rl/savestate.py` |
| macOS host mappings | 4 views of MEM1 (fastmem arena + mirrors) | ✅ | `find_mem1_views`; all 4 agree |
| Linux host mapping | `/dev/shm/dolphin-emu.<pid>` (deleted, fd open), MEM1 at offset 0 | 🔶 Dolphin source + fake-process test | `zelda_rl/linuxmem.py` |

## 2. Global addresses (NTSC 1.2)

| Symbol | Address | Status | Evidence |
|---|---|---|---|
| `gSaveContext` | `0x8011AC80` | ✅ | `"ZELDAZ"` at +0x1C; health/rupees match gameplay |
| `gPlayState` (GlobalContext) | `0x801C8D60` | ✅ | player actor list head, scene id, camera pointer all consistent |
| VI counter | `0x8011DBB0` (u32) | ✅ | +60/s at 1×, +360/s uncapped; restored by savestates |
| Other 60 Hz counters | `0x8000A4CC`, `0x800FF4A8`, `0x8012879C` | ✅ tick at 60/s | not used |
| `GameState.frames` | `gPlayState + 0x9C` | ❌ | does not tick in this build |

---

## 3. SaveContext (`gSaveContext`, 0x1428 bytes)

| Offset | Type | Field | Status / observed |
|---|---|---|---|
| `0x00` | s32 | entranceIndex | ✅ 0x0BB (Link's house), 0x211 (Kokiri Forest, your slots) |
| `0x04` | s32 | linkAge (0 adult, 1 child) | ✅ 1 |
| `0x08` | s32 | cutsceneIndex | 🔶 |
| `0x0C` | u16 | dayTime (0x8000 = noon) | ✅ 0x8000 (static in Kokiri Forest / house) |
| `0x10` | s32 | nightFlag | 🔶 |
| `0x14` | s32 | totalDays | 🔶 |
| `0x18` | s32 | bgsDayCount | 🔶 |
| `0x1C` | char[6] | newf `"ZELDAZ"` | ✅ |
| `0x22` | u16 | deaths | ✅ 0 |
| `0x24` | u8[8] | playerName (OoT charset) | ✅ "LINK" |
| `0x2C` | s16 | n64ddFlag | ✅ 0 |
| `0x2E` | s16 | healthCapacity (16 per heart) | ✅ 0x30 |
| `0x30` | s16 | health | ✅ 0x30 |
| `0x32` | s8 | magicLevel | ✅ 0 |
| `0x33` | s8 | magic | ✅ 0 |
| **`0x34`** | **s16** | **rupees** | ✅ 0 → 1 → 6 across your slots; 150 in the attract-demo save |
| `0x36` | u16 | swordHealth (Biggoron) | ✅ 0 |
| `0x38` | u16 | naviTimer | ✅ ticks |
| `0x3A` | u8 | isMagicAcquired | 🔶 |
| `0x3C` | u8 | isDoubleMagicAcquired | 🔶 |
| `0x3E` | u8 | isDoubleDefenseAcquired | 🔶 |
| `0x3F` | u8 | bgsFlag | 🔶 |
| `0x40` | u16 | ocarinaGameRoundNum | 🔶 |
| `0x42` | ItemEquips | childEquips (buttonItems[4], cButtonSlots[3], equipment u16) | 🔶 |
| `0x4C` | ItemEquips | adultEquips | 🔶 |
| `0x66` | s16 | savedSceneId | 🔶 |
| `0x68` | u8[4] | equips.buttonItems (B, C-left, C-down, C-right) | ✅ B = 0xFF (no sword) |
| `0x6C` | u8[3] | equips.cButtonSlots | 🔶 |
| `0x70` | u16 | equips.equipment (currently equipped) | ✅ 0x1100 (Kokiri tunic + boots) |
| **`0x74`** | **u8[24]** | **inventory.items** (0xFF = empty) | ✅ all 0xFF (fresh file) |
| `0x8C` | s8[16] | inventory.ammo | 🔶 |
| **`0x9C`** | **u16** | **inventory.equipment** (owned; bits 0–3 swords, 4–7 shields, 8–11 tunics, 12–15 boots) | ✅ 0x1100 |
| `0xA0` | u32 | inventory.upgrades (quiver, bomb bag, strength, scale, wallet, …) | ✅ 0 |
| `0xA4` | u32 | inventory.questItems (medallions, songs, stones, heart pieces) | ✅ 0 |
| `0xA8` | u8[20] | inventory.dungeonItems (map/compass/boss key bits) | 🔶 |
| `0xBC` | s8[19] | inventory.dungeonKeys | 🔶 |
| `0xCF` | s8 | inventory.defenseHearts | 🔶 |
| `0xD0` | s16 | inventory.gsTokens | 🔶 |
| **`0xD4`** | **SavedSceneFlags[124]** (0x1C each: chest, swch, clear, collect, unk, rooms, floors) | **permanent per-scene flags** | ✅ readable; Kokiri Forest (85) all 0 in your slots |
| `0xE64` | FaroresWindData | Farore's Wind warp | 🔶 |
| `0xEB4` | s32[6] | gsFlags (Gold Skulltulas) | 🔶 |
| **`0xED4`** | **u16[14]** | **eventChkInf** (story events) | ✅ readable; 0 bits in all slots |
| **`0xEF0`** | **u16[4]** | **itemGetInf** | ✅ readable; 0 bits |
| **`0xEF8`** | **u16[30]** | **infTable** (NPC-talk / info flags) | ✅ readable; 2 bits set, identical in all slots |
| `0xF34` | u32[?] | worldMapAreaData | 🔶 |
| `0x1354` | s32 | fileNum | 🔶 |
| `0x135C` | s32 | gameMode | 🔶 |
| `0x1360` | s32 | sceneLayer | 🔶 |
| `0x1364`… | … | respawn data, nextTransition, HUD/minimap state, `buttonStatus[5]`, `forceRisingButtonAlphas` | 🔶 |

**Flags are the hook for "reach state X" goals.** A savestate target defines the goal bits (`goal & ~start`). Progress = how many of those bits are set now. For your slots 1→5 no flag differs, so only position and rupees matter for this task.

---

## 4. PlayState (`gPlayState`, 0x12518 bytes)

| Offset | Type | Field | Status |
|---|---|---|---|
| `0x0000` | GameState | state (gfxCtx, main/init/destroy, size, `input[4]`, heap, running, frames) | 🔶 layout; `frames` ❌ |
| **`0x0014`** | **Input[4] (0x14 each)** | **controller: cur.button u16, cur.stick_x s8, cur.stick_y s8** | ✅ `00007f00` with stick right; zeros in frozen slot 4 |
| **`0x00A4`** | **s16** | **sceneId** | ✅ 52 house, 85 Kokiri Forest, 81 Hyrule Field |
| `0x00C0` | View | main view | 🔶 |
| **`0x01E0`** | **Camera** | **mainCamera** | ✅ `cameraPtrs[0]` points here |
| `0x034C` | Camera[3] | subCameras | 🔶 |
| **`0x0790`** | **Camera*[4]** | **cameraPtrs** | ✅ [0] = 0x801C8F40 |
| `0x07A0` | s16 | activeCamId | ✅ 0 |
| `0x07A2` | s16 | nextCamId | 🔶 |
| `0x07A8` | LightContext | lights | 🔶 |
| `0x07B8` | FrameAdvanceContext | | 🔶 |
| `0x07C0` | CollisionContext | static + dynamic collision (floors, walls) | 🔶 |
| **`0x1C24`** | **ActorContext** | **actors** (§5) | ✅ |
| `0x1D64` | CutsceneContext | cutscene state | 🔶 |
| `0x1E08` | SramContext | | 🔶 |
| `0x1E18` | SkyboxContext | | 🔶 |
| `0x20D8` | MessageContext | text boxes: mode, text id, choices | 🔶; **dialog-state byte not yet identified** (214 candidate bytes differ in slot 1; live test pending) |
| `0x104F0`… | InterfaceContext | HUD, button alphas | 🔶 |
| `0x10A24`… | PauseContext | pause menu | 🔶 |
| `0x11D30`… | room context, transition actors, `numSetupActors`, … | | 🔶 |

## 5. ActorContext (`gPlayState + 0x1C24`)

| Offset | Type | Field | Status |
|---|---|---|---|
| `0x08` | u8 | total actors | 🔶 |
| **`0x0C`** | **{s32 length; Actor* head}[12]** | **actor lists by category** | ✅ |
| `0x6C` | TargetContext | Z-target state | 🔶 |
| `0x100`… | flags {swch, tempSwch, unk0, tempUnk0, chest, clear, tempClear, collect, tempCollect} | live scene flags | 🔶 |
| `0x138` | TitleCardContext | | 🔶 |

Categories (list index): 0 SWITCH, 1 BG, **2 PLAYER** ✅, 3 EXPLOSIVE, **4 NPC** ✅, 5 ENEMY, **6 PROP** ✅, **7 ITEMACTION** ✅, **8 MISC** ✅, 9 BOSS, 10 DOOR ✅, 11 CHEST.

In Kokiri Forest, your slot 1 has: BG 2, PLAYER 1, NPC 11, PROP 53, ITEMACTION 18, MISC 9, DOOR 2.

## 6. Actor (common header of every actor, 0x14C bytes)

| Offset | Type | Field | Status |
|---|---|---|---|
| **`0x000`** | **s16** | **id** (actor type) | ✅ |
| **`0x002`** | **u8** | **category** | ✅ matches the list it's in |
| `0x003` | s8 | room | 🔶 |
| **`0x004`** | **u32** | **flags** (bit0 targetable, bit2 hostile, bit3 friendly, bit6 update-when-culled? …) | ✅ readable; bit meanings 🔶 |
| `0x008` | PosRot | home (pos Vec3f 0x08, rot Vec3s 0x14) | 🔶 |
| **`0x01C`** | **s16** | **params** (type-specific variant) | ✅ EnItem00 rupee types |
| `0x01E` | s8 | objBankIndex | 🔶 |
| `0x01F` | s8 | targetMode | 🔶 |
| **`0x024`** | **Vec3f** | **world.pos** | ✅ |
| **`0x030`** | **Vec3s** | **world.rot** (y at 0x32 = yaw) | ✅ yaw changes while turning |
| `0x038` | PosRot | focus (look-at / target point) | 🔶 |
| `0x050` | Vec3f | scale | 🔶 |
| `0x05C` | Vec3f | velocity | 🔶 |
| `0x068` | f32 | speedXZ | 🔶 |
| `0x06C` | f32 | gravity | 🔶 |
| `0x088` | u16 | bgCheckFlags (on ground, touching wall, in water) | 🔶 |
| `0x08A` | s16 | yawTowardsPlayer | 🔶 |
| `0x08C` | f32 | xyzDistToPlayerSq | 🔶 |
| **`0x090`** | **f32** | **xzDistToPlayer** | ✅ equals our computed xz distance |
| `0x094` | f32 | yDistToPlayer | 🔶 |
| `0x098` | CollisionCheckInfo | damage table, `health` at **`0x0AF`** (u8) | ✅ readable (8 on NPCs and items); meaning 🔶 |
| `0x0B4` | ActorShape | shape.rot (visual rotation) | 🔶 |
| `0x10C` | u8 | isTargeted | 🔶 |
| `0x118` | Actor* | parent | 🔶 |
| `0x11C` | Actor* | child | 🔶 |
| `0x120` | Actor* | prev | 🔶 |
| **`0x124`** | **Actor*** | **next** | ✅ list walk works |

### Actor IDs seen (your slots, Kokiri Forest)

| id | Category | Actor (decomp name) | Notes |
|---|---|---|---|
| `0x0015` | 8 MISC | **En_Item00** collectible | ✅ params & 0xFF: 0 green (1), 1 blue (5), 2 red (20), 3 recovery heart, 0x13 purple (50), 0x14 orange (200) 🔶 for the last two |
| `0x0018` | 7 ITEMACTION | En_Elf (fairy / **Navi**) | ✅ ~4 xz units from Link |
| `0x0163` | 4 NPC | En_Ko (**Kokiri children**) | 🔶 name; params 0xFF00, 0xFF01, 0xFF0C = which Kokiri |
| `0x014E` / `0x014F` / `0x0151` | 6 PROP | forest props (grass / bushes / signs, TBD) | 🔶 identify live |
| `0x0039` | 6 PROP | TBD | 🔶 |
| `0x0000` | 2 PLAYER | Player (Link) | ✅ |

Mido, doors, the shop, the Deku Baba, and grass (`En_Kusa`) IDs will be pinned down with the live actor dump in the v2 observation work.

## 7. Player (Link, extends Actor)

| Offset | Type | Field | Status |
|---|---|---|---|
| `0x000`–`0x14B` | Actor | header above | ✅ |
| `0x14C`… | | age props, skeleton, held item, masks… | 🔶 |
| **`0x066C`** | **u32** | **stateFlags1** | ✅ slot 1 = `0x00018000` (Z-targeting + friendly focus on Mido), others 0 |
| `0x0670` | u32 | stateFlags2 | ✅ readable (e.g. 0x10000000, 0x00102002) |
| `0x0674` | u8 | stateFlags3 | ✅ readable |

stateFlags1 bits (🔶 decomp names; bit 15/16 ✅ consistent with slot 1):

| Bit | Meaning | Bit | Meaning |
|---|---|---|---|
| 0 | exiting scene | 16 | friendly actor focus (talk target) ✅ |
| 1 | swinging bottle | 17 | parallel (Z without target) |
| 3 | aiming first-person item | 18 | jumping |
| 4 | hostile lock-on | 19 | freefall |
| **6** | **TALKING** (text box) | 20 | first-person view |
| 7 | dead | 21 | climbing ladder |
| 10 | getting item | 22 | shielding |
| 11 | carrying actor (pot, bomb) | 23 | on horse |
| 12 | charging spin attack | 26 | damaged |
| 13 | hanging off ledge | 27 | in water |
| 14 | climbing ledge | 28 | item over head |
| **15** | **Z-targeting** ✅ | 29 | in cutscene |

## 8. Camera (`gPlayState + 0x1E0`)

| Offset | Type | Field | Status |
|---|---|---|---|
| **`0x50`** | **Vec3f** | **at** (look-at point) | ✅ ~35 units from Link (his head) |
| **`0x5C`** | **Vec3f** | **eye** (camera position) | ✅ 111–163 units from Link |
| `0x68` | Vec3f | up | 🔶 |
| `0x74` | Vec3f | eyeNext | 🔶 |
| `0x80`… | | direction / setting / mode / status | 🔶 |
| derived | | **camera yaw = atan2(at.z − eye.z, at.x − eye.x)** | ✅ computed; stick→world mapping calibration pending |

## 9. Scenes seen

| sceneId | Scene |
|---|---|
| 52 (0x34) | Link's House ✅ |
| 81 (0x51) | Hyrule Field ✅ (attract demo) |
| 85 (0x55) | Kokiri Forest ✅ (your slots) |

## 10. What the policy uses today vs next

| Data | v1 observation | v2 (in progress) |
|---|---|---|
| Link pos / yaw / velocity, rupees, health | ✅ | ✅ |
| Nearest rupee actors (4) | ✅ | superseded by nearest-16 actors of any type (id, category, rel pos, distance, flags, health, params) |
| Camera yaw | ❌ | ✅ + world-relative movement actions |
| Player stateFlags1 bits (talking, Z-target, carrying, climbing, in air, water, cutscene…) | ❌ | ✅ |
| Dialog open / text id | ❌ | stateFlags1 TALKING bit now; MessageContext byte once identified |
| Save progress: items, equipment, event/inf/scene flags | ❌ | ✅ as goal-flag progress (missing/extra bit counts) + sword/shield owned |

## 11. Combat fields (verified 2026-09-13, Deku Tree savestates + live recordings)

| Field | Address | Status / evidence |
|---|---|---|
| Lock-on target (`TargetContext.targetedActor`) | gPlayState **+0x1CCC** | ✅ equals the Baba's pointer exactly in Z-locked slots; 0 otherwise |
| Navi arrow target | gPlayState **+0x1CC8** | ✅ lockable enemy in range (set before lock-on) |
| Current room number | gPlayState **+0x11CBC** | 🔶 reads 0 in all Deku Tree slots (change on exit not yet observed; scene change was) |
| Link equipment in use | save ctx +0x68 (B item) = 59 (Kokiri Sword); +0x9C = `0x1111` (sword, Deku Shield, tunic, boots) | ✅ |
| Ammo | save ctx **+0x8C** (u8 per slot) | ✅ slot 1 changed 0 → 5 on a pickup (likely Deku Nuts; slot 0 = 1, likely a stick) |
| Link death | HP (save ctx +0x30) = 0 **and** stateFlags1 bit 7, same frame | ✅ |
| Link damaged / shield / Z-target | stateFlags1 bits 26 / 22 / 15 | ✅ live |
| Shield block | stateFlags1 bit 22 drops for 1–2 frames and returns, no HP loss | ✅ 7 of 7 blocks (too short for 10 Hz; see lunge cycles) |
| Controls (VC GC pad) | L held = Z-target, R held = shield, B = sword, A = action | ✅ pipe probe |
| Enemy HP init | new actors read HP 8 for one frame before initialising | ✅ trap |
| Death/reload despawn | all enemy actors removed at once | ✅ trap (not kills) |

### Deku Baba (`En_Dekubaba`, actor id 0x0055, category ENEMY)

| Field | Offset | Status |
|---|---|---|
| HP | +0xAF | ✅ 2 max; 1 per sword hit |
| Head direction | shape rotation y **+0xB6** | ✅ spans the full circle; world rotation (+0x32) is the fixed base |
| Yaw to player | +0x8A | ✅ |
| **Action routine (behaviour state)** | **+0x1B0** | ✅ `0x801EDF30` hidden, `0x801EDFDC`/`0x801EE444` emerge/retract, `0x801EEE94`/`0x801EE910` idle, `0x801EEDC8` prepare lunge, `0x801EEB40` lunge, `0x801EF220` pull back, `0x801EF318` hit, `0x801EF428` stunned, `0x801EF5EC`/`0x801EF950` dying; all resolve in all 10 slots |
| Action timer | +0x1B6 | ✅ countdown |
| Animation updater | +0x19C (`0x8008D3C4` loop / `0x8008D50C` one-shot) | ✅ |
| +0x154 float triple | | ❌ not the lunging head (≈220 from base in all normal states) |
| Focus point | +0x38 | ✅ within 35 units of the base |
