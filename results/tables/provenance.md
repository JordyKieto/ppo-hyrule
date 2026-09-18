| Property | Value |
|---|---|
| Dataset | `datasets/transitions/deku-baba/20260918-134229` |
| Episodes / transitions | 168 / 109,817 |
| Wall clock | 17.6 min |
| Throughput | 104 steps/s |
| Decision rate | 10 Hz (6 VI/step) |
| Episode cap | 180 s |
| Action space | 63 discrete |
| Policy obs | v4, 483 dims |
| Start slots | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
| Sampling | sampled from the policy |
| Actors dropped | 0 |
| G5 | 15 eps, 20,782 trans — `checkpoints/inbox/deku-combat_r5_u00005.pt` |
| G30 | 22 eps, 20,211 trans — `checkpoints/inbox/deku-combat-gru-v2_u00030.pt` |
| G30 | 20 eps, 12,738 trans — `checkpoints/inbox/deku-combat_r2_u00030.pt` |
| G90 | 21 eps, 12,644 trans — `checkpoints/inbox/deku-combat-gru-v2b_u00060.pt` |
| G110 | 27 eps, 14,751 trans — `checkpoints/inbox/deku-combat-gru-v2b_u00080.pt` |
| G130 | 27 eps, 12,539 trans — `checkpoints/inbox/deku-combat-gru-v2b_u00100.pt` |
| G150 | 36 eps, 16,152 trans — `checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` |
| Tiers | 6 |
