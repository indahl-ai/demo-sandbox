# Status: AI and machine learning demonstration replay

Re-verified 25 Nov 2026 against the current published page data (demos.json, entry `ai`) with `python verify_replay.py <path to demos.json>`: 54 of 54 comparisons PASS. Replay needs no key and no network. Command: `python sandbox.py ai_kernels replay --chip kingston|fez|marrakesh --run N`. Run 1 and run 2 are as in the page's job table (run 1 = jobs dbvffq..., dbvffu..., dbvfg2...; run 2 = dbvfov..., dbvfp3..., dbvfp7...).

| item | chip | run | published | replayed | result |
|---|---|---|---|---|---|
| self-score 8q | kingston | 1 | 0.788 | 0.788 | PASS |
| self-score 16q | kingston | 1 | 0.489 | 0.489 | PASS |
| self-score 24q | kingston | 1 | 0.240 | 0.240 | PASS |
| self-score 32q | kingston | 1 | 0.117 | 0.117 | PASS |
| self-score 40q | kingston | 1 | 0.079 | 0.079 | PASS |
| shot bill 8q | kingston | 1 | 1.06x | 1.06x | PASS |
| shot bill 16q | kingston | 1 | 1.40x | 1.40x | PASS |
| shot bill 24q | kingston | 1 | 2.19x | 2.19x | PASS |
| shot bill 32q | kingston | 1 | 3.43x | 3.43x | PASS |
| shot bill 40q | kingston | 1 | 3.60x | 3.60x | PASS |
| self-score 40q (replicate table) | kingston | 1 | 0.079 | 0.079 | PASS |
| shot bill 40q (replicate table) | kingston | 1 | 3.60x | 3.60x | PASS |
| readout qubits flagged | kingston | 1 | 11 | 11 | PASS |
| readout example q55 P(1|0) today / record | kingston | 1 | 0.0075 / 0.0010 | 0.0075 / 0.0010 | PASS |
| self-score 40q (replicate table) | kingston | 2 | 0.074 | 0.074 | PASS |
| shot bill 40q (replicate table) | kingston | 2 | 3.84x | 3.84x | PASS |
| readout qubits flagged | kingston | 2 | 15 | 15 | PASS |
| readout example q55 P(1|0) today / record | kingston | 2 | 0.0140 / 0.0010 | 0.0140 / 0.0010 | PASS |
| self-score 8q | fez | 1 | 0.768 | 0.768 | PASS |
| self-score 16q | fez | 1 | 0.448 | 0.448 | PASS |
| self-score 24q | fez | 1 | 0.200 | 0.200 | PASS |
| self-score 32q | fez | 1 | 0.057 | 0.057 | PASS |
| self-score 40q | fez | 1 | 0.034 | 0.034 | PASS |
| shot bill 8q | fez | 1 | 1.05x | 1.05x | PASS |
| shot bill 16q | fez | 1 | 1.32x | 1.32x | PASS |
| shot bill 24q | fez | 1 | 2.18x | 2.18x | PASS |
| shot bill 32q | fez | 1 | 4.82x | 4.82x | PASS |
| shot bill 40q | fez | 1 | 5.20x | 5.20x | PASS |
| self-score 40q (replicate table) | fez | 1 | 0.034 | 0.034 | PASS |
| shot bill 40q (replicate table) | fez | 1 | 5.20x | 5.20x | PASS |
| readout qubits flagged | fez | 1 | 5 | 5 | PASS |
| readout example q144 P(1|0) today / record | fez | 1 | 0.0050 / 0.0000 | 0.0050 / 0.0000 | PASS |
| self-score 40q (replicate table) | fez | 2 | 0.028 | 0.028 | PASS |
| shot bill 40q (replicate table) | fez | 2 | 6.48x | 6.48x | PASS |
| readout qubits flagged | fez | 2 | 4 | 4 | PASS |
| readout example q144 P(1|0) today / record | fez | 2 | 0.0095 / 0.0000 | 0.0095 / 0.0000 | PASS |
| self-score 8q | marrakesh | 1 | 0.736 | 0.736 | PASS |
| self-score 16q | marrakesh | 1 | 0.334 | 0.334 | PASS |
| self-score 24q | marrakesh | 1 | 0.092 | 0.092 | PASS |
| self-score 32q | marrakesh | 1 | 0.046 | 0.046 | PASS |
| self-score 40q | marrakesh | 1 | 0.012 | 0.012 | PASS |
| shot bill 8q | marrakesh | 1 | 1.04x | 1.04x | PASS |
| shot bill 16q | marrakesh | 1 | 1.69x | 1.69x | PASS |
| shot bill 24q | marrakesh | 1 | 4.09x | 4.09x | PASS |
| shot bill 32q | marrakesh | 1 | 5.06x | 5.06x | PASS |
| shot bill 40q | marrakesh | 1 | 10.99x | 10.99x | PASS |
| self-score 40q (replicate table) | marrakesh | 1 | 0.012 | 0.012 | PASS |
| shot bill 40q (replicate table) | marrakesh | 1 | 10.99x | 10.99x | PASS |
| readout qubits flagged | marrakesh | 1 | 19 | 19 | PASS |
| readout example q63 P(1|0) today / record | marrakesh | 1 | 0.0515 / 0.0010 | 0.0515 / 0.0010 | PASS |
| self-score 40q (replicate table) | marrakesh | 2 | 0.011 | 0.011 | PASS |
| shot bill 40q (replicate table) | marrakesh | 2 | 12.35x | 12.35x | PASS |
| readout qubits flagged | marrakesh | 2 | 18 | 18 | PASS |
| readout example q63 P(1|0) today / record | marrakesh | 2 | 0.0510 / 0.0010 | 0.0510 / 0.0010 | PASS |

PASS 54 of 54

Until 24 Nov 2026 the page carried 10.96x (marrakesh run 1, 40 qubits, in two tables) and 6.47x (fez run 2, 40 qubits). Those three cells did not match the replay (10.99x, 6.48x). The page now carries the replayed figures, as in the table above.

## How the shot bill is computed
The published shot bill uses a gate-only reading that includes IBM's single-qubit gate errors: S = readout x (CZ fade x single-qubit fade + uniform term), single-qubit fade = product over every sx and x gate of (1 - recorded error), binomial bill (S/A)(1-0.5A)/(1-0.5S). The single-qubit gate counts come from rebuilding the circuits on the stand-in chip.

## What the recorded files add
The analysis in the original script reads a live backend record at analysis time. That record is not saved with the runs. For each chip it was recovered read-only from IBM's calibration history (the record whose update time equals the stamp saved with each run) and saved into every recorded state file under the key `record` (symmetric and one-way readout errors, single-qubit and CZ gate errors). Both runs on a chip share one stamp. Replay is then fully offline. Billed QPU seconds per job are saved under `billed_qpu_seconds`.
Cross-checks that hold: the recovered kingston readout figures match the values printed in the original analysis file (for example q105), and the readout flags and gate-only readings published for kingston reproduce from it. The saved gate-only reading S was not recomputed from the recovered CZ errors.

## Other points
- Layouts: the circuits are rebuilt on a stand-in chip with the saved layout, and the replay refuses to proceed unless the CZ count and CZ pairs match the recorded ones (they match at every size on every chip and run). The single-qubit gate counts come from that rebuild.
- The page's readout table counts qubits flagged on the one-way figures (read against IBM's matching one-way record). The original v7 script printed the two-way comparison, later withdrawn; the sandbox script prints the one-way version the page states.
- MEASURED_USAGE_v7.json in the source folder labels run 1 and run 2 the other way round from the page. The sandbox follows the page. The billed seconds are the same either way: kingston 20 s, fez 20 s, marrakesh 21 s in both runs (confirmed from IBM's job records).
- ESTIMATE_SECONDS = 23.2 (the code's own estimate per chip; fire multiplies by 1.25 and refuses above a 60 s cap or the remaining time).
- No library beyond qiskit, qiskit-ibm-runtime, numpy and scipy is needed.
- fire needs `yes=True` / `--yes`. No job was submitted while building this.
