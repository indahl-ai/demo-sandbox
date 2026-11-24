# Replay status: logistics (QAOA max-cut) demonstration

Run 1 = the first recorded run (job IDs in `recorded_optimization/run1/<chip>/state.json`); run 2 = the replicate. Matched to the published tables by job ID and by value.
Replay is offline: `python sandbox_entry.py replay --chip kingston --run 1` (no key, no network, no IBM time). `python verify.py` regenerates the table below.

## Published against replayed (approximation ratio, 4 decimals)

| chip | run | published client | replayed client | published ours | replayed ours | O-S | O-S word | O-N word | result |
|---|---|---|---|---|---|---|---|---|---|
| kingston | 1 | 0.9691 | 0.9691 | 0.9836 | 0.9836 | +0.0045 | ABOVE | ABOVE | PASS |
| kingston | 1 (O minus S) | +0.0045 | - | - | +0.0045 |  |  |  | PASS |
| kingston | 2 | 0.9689 | 0.9689 | 0.9834 | 0.9834 | +0.0045 | ABOVE | ABOVE | PASS |
| fez | 1 | 0.9585 | 0.9585 | 0.9763 | 0.9763 | +0.0104 | ABOVE | ABOVE | PASS |
| fez | 1 (O minus S) | +0.0104 | - | - | +0.0104 |  |  |  | PASS |
| fez | 2 | 0.9564 | 0.9564 | 0.9776 | 0.9776 | +0.0138 | ABOVE | ABOVE | PASS |
| marrakesh | 1 | 0.9580 | 0.9580 | 0.9781 | 0.9781 | +0.0107 | ABOVE | ABOVE | PASS |
| marrakesh | 1 (O minus S) | +0.0107 | - | - | +0.0107 |  |  |  | PASS |
| marrakesh | 2 | 0.9334 | 0.9334 | 0.9536 | 0.9536 | +0.0121 | ABOVE | ABOVE | PASS |
ALL PASS

Also checked: the published statement "the gap to the noiseless cut shrank 1.75 to 1.92 times in run 1 and 1.4 to 1.95 times in run 2" recomputes from the replayed values as 1.88, 1.75, 1.92 (run 1) and 1.87, 1.95, 1.44 (run 2). PASS. The published "ours beat the stated-error correction on every chip in both runs" replays as ABOVE in all six runs. PASS.

## What does not reproduce

Nothing differed. All twelve published client and ours values, the three published run-1 O-minus-S values and all outcome words match to the published digits.

## Notes

- The analysis needs no backend, so no chip stand-in is used (the finance replay needs one). Everything it needs is in the recorded state file (layout, edges, angles, stated gate error saved at fire time, job-P benchmarking counts) and the two raw count files.
- Billed QPU seconds (job.usage() on each recorded job, read-only lookups, saved in `recorded_optimization/billed_usage.json`): kingston 14 + 34 = 48 s, fez 14 + 33 = 47 s, marrakesh 15 + 35 = 50 s, identical in both runs. The code's own estimate is 57.5 s per chip (before its x1.25 margin; refuses over 200 s).
- Live `fire` and `check` were not exercised here (no job may be submitted and no key was used for them). The code path is the unmodified demo path with house paths removed and the key taken only from IBM_QUANTUM_TOKEN. One unverified point: demo_v3 fire refuses with "PILOT NEEDED" if its model power ratio is under 3 and no pilot result exists; the recorded pilot result file is not in the source folder, so a live fire may require the pilot step first.
- No library beyond qiskit, qiskit-ibm-runtime, numpy, scipy. Live runs also use the stated-gate-error figures from the chip's live record.
