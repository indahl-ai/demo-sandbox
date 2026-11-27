# Simulation sandbox: verification status

Replay is offline (no key, no network). Metric: error against the exact answer, mean over steps 2 to 20, four copies per chip. Published values are from the demo page data; replayed values are from `replay(chip, run)`.

Run 1 is the first fired run, run 2 is the replicate. Marrakesh gate ratio is the mean RB/stated CZ error ratio printed by the analysis.

| chip | run | published client | replayed client | published ours | replayed ours | result |
|---|---|---|---|---|---|---|
| kingston | 1 | 0.065 | 0.0652 (0.065) | 0.020 | 0.0204 (0.020) | PASS |
| fez | 1 | 0.076 | 0.0763 (0.076) | 0.022 | 0.0219 (0.022) | PASS |
| marrakesh | 1 | 0.102 | 0.1021 (0.102) | 0.027 | 0.0268 (0.027) | PASS |
| kingston | 2 | 0.064 | 0.0637 (0.064) | 0.025 | 0.0250 (0.025) | PASS |
| fez | 2 | 0.076 | 0.0762 (0.076) | 0.021 | 0.0213 (0.021) | PASS |
| marrakesh | 2 | 0.103 | 0.1033 (0.103) | 0.077 | 0.0770 (0.077) | PASS |

Run 1 paired 99% intervals (ours minus client): kingston [-0.0491, -0.0406], fez [-0.0576, -0.0510], marrakesh [-0.0949, -0.0557]. All three replay to the published four digits: PASS. Outcome word BELOW on all six: PASS.

Measured/stated gate error ratios quoted on the page: kingston 2.13, fez 1.55, marrakesh 3.24 (run 1) and 4.25 (run 2). Replayed: 2.13, 1.55, 3.24 and 4.25: PASS. (Replayed, not published: kingston run 2 2.01, fez run 2 1.58.)

Ratios "3.2x, 3.5x, 3.8x closer" on the chips table: replay prints 3.2x, 3.5x, 3.8x (run 1): PASS.

## What does not reproduce
Nothing in the published tables differs.

## How the replay works, and its one stand-in
Recorded: the stated-record numbers, the benchmarking counts, the readout counts and the estimator values from each run. The frozen fade exponent per chip (0.18 kingston from an earlier job, 0.215 fez, 0.17 marrakesh from short pilot jobs) is recorded in `recorded_simulation/pilot/`. Stand-in: the chip's shape is rebuilt on a fake IBM backend of the same name to count gates per step. The result matches to the digits published, so the stand-in does not change any number.

## Billed time (job.usage(), read from IBM, read-only)
Job P 18 s plus main job 24 s (25 s on marrakesh): 42 s per run on kingston and fez, 43 s on marrakesh, both runs. The code's own estimate is 97 s per run (cap 200 s). Details in `recorded_simulation/usage.json`.

## Notes
- No library beyond the four in `requirements.txt` is used.
- `check`, `fire`, `fetch` need IBM_QUANTUM_TOKEN. `fire` was not run and no job was submitted while building this.
- The original planning-file gate in `fire` (a power ratio line in a plan file) is applied only when that file exists; it is not shipped here.

- 27 Nov 2026 (Sage): the measured-vs-stated ratios above (2.13 / 1.55 / 3.24, "mean RB/stated") were WITHDRAWN from the demo page. That fit counts the single-qubit gates inside the benchmark against the two-qubit gate and reads high, and the calibration never used it. The isolated figure it does use divides by (1 - 2e_sx) per sx where a two-qubit benchmark costs (1 - 1.6e_sx), so it may read low. No ratio is printed until a known-answer test settles it. The gains against the default run are unchanged.
