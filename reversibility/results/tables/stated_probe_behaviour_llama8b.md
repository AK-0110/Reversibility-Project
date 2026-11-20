| quantity | value [95% CI] |
|---|---|
| stated answer, binary bal-acc (strict: parse failures wrong) | 0.500 [0.500, 0.500] |
| stated answer, 3-class bal-acc (strict) | 0.333 [0.333, 0.333] |
| stated parse failures | 0 / 389 |
| binary: probe vs stated bal-acc on the same 297 rows (0 parse failures dropped) | 0.953 vs 0.500 |
| binary: kappa(stated, probe) | 0.000 [0.000, 0.000] |
| three: probe vs stated bal-acc on the same 389 rows (0 parse failures dropped) | 0.988 vs 0.333 |
| three: kappa(stated, probe) | 0.000 [0.000, 0.000] |
| K: names the irreversible version (both orders; chance 0.5) | 0.694 [0.597, 0.785] |
| K both orders correct (chance 0.25) | 0.597 |
| K_single: right on each version's state alone | 0.000 [0.000, 0.000] |
| K* = 2K − 1 | 0.389 [-0.250, 0.861] |
| A (brief): careful only in the irreversible version | 0.014 [0.000, 0.042] |
| A* = sensitivity | -0.014 [-0.083, 0.042] |
| A_out*: avoids the shortcut only where irreversible | 0.014 [0.000, 0.042] |
| Gap (brief) = K − A, ~0.5 under no knowledge | 0.681 [0.583, 0.764] |
| Gap* = K* − A* (headline) | 0.403 [-0.250, 0.903] |
| Gap*_outcome = K* − A_out* | 0.375 [-0.250, 0.847] |
| McNemar p (caution irr vs rev) | 1 |
| careful 2×2 (both / irr only / rev only / neither) | 17 / 1 / 2 / 52 |
| careful prompt: Gap (brief), neutral K | 0.694 [0.597, 0.785] |
| careful prompt: Gap*, neutral K | 0.389 [-0.250, 0.861] |
