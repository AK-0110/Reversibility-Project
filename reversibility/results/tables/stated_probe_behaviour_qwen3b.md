| quantity | value [95% CI] |
|---|---|
| stated answer, binary bal-acc (strict: parse failures wrong) | 0.699 [0.649, 0.746] |
| stated answer, 3-class bal-acc (strict) | 0.466 [0.421, 0.515] |
| stated parse failures | 0 / 389 |
| binary: probe vs stated bal-acc on the same 297 rows (0 parse failures dropped) | 0.984 vs 0.699 |
| binary: kappa(stated, probe) | 0.393 [0.290, 0.491] |
| three: probe vs stated bal-acc on the same 389 rows (0 parse failures dropped) | 0.993 vs 0.466 |
| three: kappa(stated, probe) | 0.213 [0.147, 0.281] |
| K: names the irreversible version (both orders; chance 0.5) | 0.590 [0.549, 0.639] |
| K both orders correct (chance 0.25) | 0.181 |
| K_single: right on each version's state alone | 0.181 [0.000, 0.458] |
| K* = 2K − 1 | 0.181 [0.042, 0.333] |
| A (brief): careful only in the irreversible version | 0.000 [0.000, 0.000] |
| A* = sensitivity | -0.042 [-0.125, 0.000] |
| A_out*: avoids the shortcut only where irreversible | 0.069 [0.000, 0.208] |
| Gap (brief) = K − A, ~0.5 under no knowledge | 0.590 [0.549, 0.639] |
| Gap* = K* − A* (headline) | 0.222 [0.069, 0.375] |
| Gap*_outcome = K* − A_out* | 0.111 [-0.125, 0.319] |
| McNemar p (caution irr vs rev) | 0.25 |
| careful 2×2 (both / irr only / rev only / neither) | 30 / 0 / 3 / 39 |
| careful prompt: Gap (brief), neutral K | 0.590 [0.549, 0.639] |
| careful prompt: Gap*, neutral K | 0.264 [0.139, 0.389] |
