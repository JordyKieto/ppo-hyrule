# The world-model viewer

A local page that plays a recorded episode next to the world model's AUTOREGRESSIVE dream of it, so you can
see where the dream goes wrong and click any entity to compare every field it predicts.

No server, no install, no artifact. One command writes a self-contained `.html` you open in a browser.

## Run it

```sh
cd ~/Documents/dev/zelda-RL
.venv/bin/python scripts/wm_replay.py --html viz/out/dream.html --open
```

That picks the first episode with >= 160 frames, rolls `runs/wm_v3/ck_0085.pt` forward 200 steps from frame 20,
prints a divergence table in the terminal and writes the page. `--open` launches your browser; drop the flag
and open `viz/out/dream.html` by hand, or mail it to someone -- it is one file with the data inside it.

| flag | what it does |
|---|---|
| `--ckpt PATH` | which model. Default `runs/wm_v3/ck_0085.pt`; `runs/wm_v2/ck_0080.pt` is the better pre-fine-tune one |
| `--episode PATH` | a specific `.npz` under `datasets/transitions/v5/episodes/` |
| `--start N` | episode frame the rollout begins at (the model gets the 8 real frames before it as history) |
| `--horizon N` | how many autoregressive steps to roll |
| `--sample` | sample the discrete heads instead of argmax. A Baba's inner state is unobservable, so its next behaviour is genuinely stochastic; argmax is the default so a page is reproducible |
| `--json PATH` | write just the data, no page |
| `--frame N --entity link` | print one step's field table in the terminal and exit. `--entity 2` for actor row 2 |

Compare two checkpoints on the same episode:

```sh
E=datasets/transitions/v5/episodes/ep00001_G030_s04_success.npz
.venv/bin/python scripts/wm_replay.py --episode $E --ckpt runs/wm_v2/ck_0080.pt --html viz/out/v2.html
.venv/bin/python scripts/wm_replay.py --episode $E --ckpt runs/wm_v3/ck_0085.pt --html viz/out/v3.html
```

## What the right panel is

**Autoregressive, open loop.** Step 0 is the real state. After that the model sees ONLY ITS OWN OUTPUT: each
step it gets its predicted state plus the REAL action the player took at that frame, and predicts the next
state. It never invents an action and it never re-reads the truth. Error compounds, which is the point.

The divergence chart's dashed line is a control that does nothing at all -- Link frozen at the start position.
**If the solid line is above the dashed line the model is worse than assuming nothing moved.** For
`wm_v3/ck_0085.pt` it is, at every horizon, which is what `docs/WM_EXPERIMENT_REGIME.md` already records: that
checkpoint's one-step skill went from +0.188 teacher-forced to about -0.13 under autoregressive fine-tuning.

## Reading the picture

Both panels use the SAME colour for the same creature. The panel is told apart by its heading, never by
recolouring things.

* **Fill = what it is.** Link green, Deku Baba / Skulltula / Skullwalltula red, Gohma magenta, larva yellow,
  pickups blue, props grey. Names come from `zelda_rl.oomdp.ACTOR_CLASSES`, which was built from the
  decompilation's `actor_table.h`; an unidentified id shows as `actor 0xNNNN` rather than a guess.
* **Ring = what state it is in.** hidden, idle, winding up, attacking, hit, stunned, dying. A ring colour
  changing is a state change, on either panel.
* **Link's outline = what he is doing.** white for shield or a live sword swing, red when damaged, dashed in a
  cutscene. These come from the `sf1` flags the model also predicts, so the outline appears on both panels.
* **The spoke** is Link's facing. The model predicts yaw, so it has a heading too and both panels draw it.
* **Grey cells** are walls, from the accumulated wall map -- incomplete geometry (209 of 1024 cells), only
  where Link actually bumped into something.
* Faded diamond = present but 0 HP. A struck-through chip in the inspector roster means the model has that
  entity gone while the recording still has it.

## Controls

| | |
|---|---|
| `space` | play / pause |
| `←` `→` | step, `shift` for 10 |
| `r` / `End` | back to step 0 / jump to the end |
| `g` | ghost: true positions as outlines on the dream panel, with a line to where the model put each thing |
| `t` | trails: Link's last 40 steps on each panel |
| hover | compare that entity field by field |
| click | pin it, so it survives scrubbing. Click again to release |

## Caveats worth knowing

* **Angle diffs are wraparound-aware.** Yaw is a 16-bit angle, so a raw difference of 46,708 is really 103
  degrees. The inspector accounts for that; a naive read of the numbers would not.
* **`persistence` here is distance from the START position**, which is the right do-nothing control for an
  open-loop rollout but means skill is undefined (`nan`) on the early steps where Link has not moved yet.
* **The advance step is a port of `advance()` in `scripts/train_wm_tokens.py`.** If you change the head layout
  or the token fields there, change them here too. Keeping one copy would have meant refactoring the trainer,
  which is not something a viewer should force.
* Only fields the model actually predicts are shown for the dream. Nothing is filled in from the truth to make
  the panel look complete.
