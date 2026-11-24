# Error correction sandbox: replay verification

Published values are from the site demos.json entry for error-correction (and, where marked, the check log). Replayed values come from `python sandbox_entry.py replay --chip X --run N` run with no key and no network. Each replay was also compared line by line with the results text stored beside the recorded run: all six are identical.

## Which recorded files are which

- Published demo: the surface-code memory demo (`demo.py` here is the surface-code script). The older repetition-code scripts (diagnose_fire1.py, layout_kingston.json, run folders for the 79-qubit line) are an earlier rung whose numbers are not on the page and are not included.
- Run 1: one state file per chip under `recorded_error_correction/run1/ibm_<chip>/`, run 2 under `run2/`. Both runs exist for all three chips, with their own calibration job and main job.
- Run 2 marrakesh is the third calibration attempt; two earlier attempts were refused by the code guard before any main job and are not recorded (no main job exists for them).
- The pilot results (one small file per chip) are included only because the code refuses a live run without them.

## Results

| Item | Published | Replayed | Result |
|---|---|---|---|
| run1 kingston per round client | 0.173 | 0.173 | PASS |
| run1 kingston per round ours | 0.105 | 0.105 | PASS |
| run1 kingston ratio | 1.64 | 1.64 | PASS |
| run1 kingston diff interval | [-0.074, -0.061] | [-0.074, -0.061] | PASS |
| run1 kingston outcome | BELOW | BELOW | PASS |
| run1 fez per round client | 0.226 | 0.226 | PASS |
| run1 fez per round ours | 0.129 | 0.129 | PASS |
| run1 fez ratio | 1.74 | 1.74 | PASS |
| run1 fez diff interval | [-0.114, -0.078] | [-0.114, -0.078] | PASS |
| run1 fez outcome | BELOW | BELOW | PASS |
| run1 marrakesh per round client | 0.284 | 0.284 | PASS |
| run1 marrakesh per round ours | 0.175 | 0.175 | PASS |
| run1 marrakesh ratio | 1.62 | 1.62 | PASS |
| run1 marrakesh diff interval | [-0.131, -0.087] | [-0.131, -0.087] | PASS |
| run1 marrakesh outcome | BELOW | BELOW | PASS |
| run1 kingston R1 client / ours | 0.172 / 0.113 | 0.172 / 0.113 | PASS |
| run1 kingston R1 outcome | BELOW | BELOW | PASS |
| run1 kingston R2 client / ours | 0.204 / 0.176 | 0.204 / 0.176 | PASS |
| run1 kingston R2 outcome | BELOW | BELOW | PASS |
| run1 kingston R3 client / ours | 0.416 / 0.236 | 0.416 / 0.236 | PASS |
| run1 kingston R3 outcome | BELOW | BELOW | PASS |
| run1 kingston R4 client / ours | 0.378 / 0.312 | 0.378 / 0.312 | PASS |
| run1 kingston R4 outcome | BELOW | BELOW | PASS |
| run1 fez R1 client / ours | 0.204 / 0.158 | 0.204 / 0.158 | PASS |
| run1 fez R1 outcome | BELOW | BELOW | PASS |
| run1 fez R2 client / ours | 0.332 / 0.245 | 0.332 / 0.245 | PASS |
| run1 fez R2 outcome | BELOW | BELOW | PASS |
| run1 fez R3 client / ours | 0.452 / 0.360 | 0.452 / 0.360 | PASS |
| run1 fez R3 outcome | BELOW | BELOW | PASS |
| run1 fez R4 client / ours | 0.437 / 0.345 | 0.437 / 0.345 | PASS |
| run1 fez R4 outcome | BELOW | BELOW | PASS |
| run1 marrakesh R1 client / ours | 0.247 / 0.175 | 0.247 / 0.174 | FAIL |
| run1 marrakesh R1 outcome | BELOW | BELOW | PASS |
| run1 marrakesh R2 client / ours | 0.245 / 0.206 | 0.245 / 0.206 | PASS |
| run1 marrakesh R2 outcome | BELOW | BELOW | PASS |
| run1 marrakesh R3 client / ours | 0.455 / 0.334 | 0.455 / 0.334 | PASS |
| run1 marrakesh R3 outcome | BELOW | BELOW | PASS |
| run1 marrakesh R4 client / ours | 0.471 / 0.406 | 0.471 / 0.406 | PASS |
| run1 marrakesh R4 outcome | BELOW | BELOW | PASS |
| run2 kingston per round client | 0.177 | 0.177 | PASS |
| run2 kingston per round ours | 0.108 | 0.108 | PASS |
| run2 kingston diff interval | [-0.079, -0.060] | [-0.079, -0.060] | PASS |
| run2 kingston outcome | BELOW | BELOW | PASS |
| run2 kingston ratio (note: 1.65 / 1.83) | 1.65 | 1.65 | PASS |
| run2 fez per round client | 0.186 | 0.186 | PASS |
| run2 fez per round ours | 0.102 | 0.102 | PASS |
| run2 fez diff interval | [-0.096, -0.073] | [-0.096, -0.073] | PASS |
| run2 fez outcome | BELOW | BELOW | PASS |
| run2 fez ratio (note: 1.65 / 1.83) | 1.83 | 1.83 | PASS |
| run2 marrakesh R2 ours / client | 0.337 / 0.370 | 0.337 / 0.370 | PASS |
| run2 marrakesh R2 outcome | BELOW | BELOW | PASS |
| run2 marrakesh R3 ours / client | 0.407 / 0.473 | 0.407 / 0.472 | FAIL |
| run2 marrakesh R3 outcome | BELOW | BELOW | PASS |
| run2 marrakesh R4 ours / client | 0.382 / 0.486 | 0.382 / 0.485 | FAIL |
| run2 marrakesh R4 outcome | BELOW | BELOW | PASS |
| run1 kingston full text vs stored results.text (all lines) | identical | identical | PASS |
| run1 fez full text vs stored results.text (all lines) | identical | identical | PASS |
| run1 marrakesh full text vs stored results.text (all lines) | identical | identical | PASS |
| run2 kingston full text vs stored results.text (all lines) | identical | identical | PASS |
| run2 fez full text vs stored results.text (all lines) | identical | identical | PASS |
| run2 marrakesh full text vs stored results.text (all lines) | identical | identical | PASS |
| run2 kingston R1 client / ours (published in the check log, not on the page) | 0.165 / 0.111 | 0.165 / 0.111 | PASS |
| run2 kingston R1 outcome | BELOW | BELOW | PASS |
| run2 kingston R2 client / ours (published in the check log, not on the page) | 0.209 / 0.172 | 0.209 / 0.172 | PASS |
| run2 kingston R2 outcome | BELOW | BELOW | PASS |
| run2 kingston R3 client / ours (published in the check log, not on the page) | 0.418 / 0.268 | 0.418 / 0.268 | PASS |
| run2 kingston R3 outcome | BELOW | BELOW | PASS |
| run2 kingston R4 client / ours (published in the check log, not on the page) | 0.381 / 0.305 | 0.381 / 0.305 | PASS |
| run2 kingston R4 outcome | BELOW | BELOW | PASS |
| run2 fez R1 client / ours (published in the check log, not on the page) | 0.192 / 0.157 | 0.192 / 0.157 | PASS |
| run2 fez R1 outcome | BELOW | BELOW | PASS |
| run2 fez R2 client / ours (published in the check log, not on the page) | 0.322 / 0.245 | 0.322 / 0.245 | PASS |
| run2 fez R2 outcome | BELOW | BELOW | PASS |
| run2 fez R3 client / ours (published in the check log, not on the page) | 0.449 / 0.359 | 0.449 / 0.359 | PASS |
| run2 fez R3 outcome | BELOW | BELOW | PASS |
| run2 fez R4 client / ours (published in the check log, not on the page) | 0.400 / 0.304 | 0.400 / 0.304 | PASS |
| run2 fez R4 outcome | BELOW | BELOW | PASS |
| run2 marrakesh R1 outcome (page note: reaches the band) | CANNOT_TELL | CANNOT_TELL | PASS |

75 of 78 comparisons PASS.

## What does not reproduce, and why

Three published third-decimal values differ by 0.001 from the replay, all on ibm_marrakesh: run 1 R1 ours (published 0.175, replay 0.174), run 2 R3 client (published 0.473, replay 0.472) and run 2 R4 client (published 0.486, replay 0.485). The replay equals the stored results text exactly in every printed digit (0.1745, 0.4725, 0.4855 at four decimals). The exact means are 0.17453, 0.47253 and 0.48553 (just above the half-way point), which round to 0.175, 0.473 and 0.486 at three decimals only if the four-digit printed value is rounded a second time. A direct three-decimal rounding gives 0.174, 0.472, 0.485. So the page rounded the four-digit printed value, not the data. The data are unchanged; the page figures are one thousandth high in these three cells.

Nothing else fails. The Z-check control reads CANNOT_TELL in every run, as the page says.

## Time on IBM, billed (job.usage() read back from IBM, read-only)

| Chip | Run | Calibration job | Main job | Billed seconds (cal + main) |
|---|---|---|---|---|
| kingston | 1 | dbv8hslarooc73fl3oi0 | dbv8i7larooc73fl3p60 | 90 |
| kingston | 2 | dbvdumff8rhs73bnp2p0 | dbvdv16ps3fc73bkas80 | 90 |
| fez | 1 | dbv8jb6ps3fc73bk0s7g | dbv93b5arooc73fl4oqg | 88 |
| fez | 2 | dbvdumf82tmc73dut1g0 | dbvdv1782tmc73dut20g | 88 |
| marrakesh | 1 | dbv91off8rhs73bnfpog | dbv923782tmc73dujj2g | 93 |
| marrakesh | 2 | dbve3amps3fc73bkb3q0 | dbve49larooc73fle85g | 93 |

ESTIMATE_SECONDS = 153.25 (the code's own estimate, 122.6 s before its 1.25 margin; cap 200 s). The code's estimate is therefore about 1.7 times what IBM billed (88 to 93 s).

## Limits

- Replay needs no fake chip: the compiled layout, the record figures at the time and the counts are all in the recorded state files.
- check, fire, fetch and analyze were not run with a key (check also reads live chip data). Their absence-of-key refusals and the fire refusal without yes were tested. fire was never called.
- Libraries: only the four (qiskit, qiskit-ibm-runtime, numpy, scipy). The code's smoke-test mode would need qiskit-aer; it is not exposed here.
