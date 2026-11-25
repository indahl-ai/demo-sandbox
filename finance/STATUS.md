# Finance sandbox: verification status

Replay is offline (no key, no network). Metric: error of the option price against the exact price 8.2543, 32 copies per chip. Published values are from the demo page data; replayed values are from `python sandbox.py finance replay --chip X --run N` (25 Nov 2026), four digits as printed, with the published digits in brackets. Where the printed fourth digit is a 5, the unrounded value from the same analysis is given too.

Run 1 is the first fired run, run 2 is the replicate. Marrakesh is the layout-fault control: its improvement is not a calibration result.

## Average error per copy (chips table, run 1, and the replicate table, run 2)

| chip | run | published default | replayed default | published calibrated | replayed calibrated | result |
|---|---|---|---|---|---|---|
| kingston | 1 | 0.532 | 0.5317 (0.532) | 0.209 | 0.2094 (0.209) | PASS |
| fez | 1 | 1.046 | 1.0457 (1.046) | 0.067 | 0.0674 (0.067) | PASS |
| marrakesh | 1 | 1.910 | 1.9101 (1.910) | 0.360 | 0.3602 (0.360) | PASS |
| kingston | 2 | 0.586 | 0.5858 (0.586) | 0.068 | 0.0684 (0.068) | PASS |
| fez | 2 | 1.079 | 1.0788 (1.079) | 0.069 | 0.0694 (0.069) | PASS |
| marrakesh | 2 | 1.854 | 1.8543 (1.854) | 0.357 | 0.3572 (0.357) | PASS |

Factors on the chips table (run 1): 2.5x, 15.5x, 5.3x. The RESULT box prints "Error 2.5x lower", "15.5x lower" and "5.3x lower": PASS.

## Pooled table (error of the pooled-shots price, percent of the exact price)

| chip | cell | published | replayed | result |
|---|---|---|---|---|
| kingston | run 1 default | 0.532 (6.4%) | 0.5317 (0.532, 6.4%) | PASS |
| kingston | run 1 calibrated | 0.211 (2.55%) | 0.2108 (0.211, 2.55%) | PASS |
| kingston | run 2 default | 0.510 (6.2%) | 0.5104 (0.510, 6.2%) | PASS |
| kingston | run 2 calibrated | 0.001 (0.01%) | 0.0010 (0.001, 0.01%) | PASS |
| fez | run 1 default | 0.693 (8.4%) | 0.6925 (unrounded 0.69249: 0.692, 8.4%) | FAIL |
| fez | run 1 calibrated | 0.030 (0.36%) | 0.0298 (0.030, 0.36%) | PASS |
| fez | run 2 default | 1.465 (17.8%) | 1.4654 (1.465, 17.8%) | PASS |
| fez | run 2 calibrated | 0.061 (0.75%) | 0.0615 (unrounded 0.06150: 0.061, 0.75%) | PASS |
| marrakesh | run 1 default | 1.909 (23.1%) | 1.9090 (1.909, 23.1%) | PASS |
| marrakesh | run 1 calibrated | 0.359 (4.34%) | 0.3586 (0.359, 4.34%) | PASS |
| marrakesh | run 2 default | 1.855 (22.5%) | 1.8548 (1.855, 22.5%) | PASS |
| marrakesh | run 2 calibrated | 0.359 (4.34%) | 0.3586 (0.359, 4.34%) | PASS |

## What does not reproduce
One pooled cell: fez run 1 default. The page shows 0.693; the analysis gives 0.69249, which is 0.692 to three digits. Cause: double rounding (the four-digit print 0.6925 was rounded again to three). Its percent, 8.4%, is unchanged. The figure is listed here as it stands and is not changed in this repository.

## How the replay works, and its one stand-in
Recorded: the stated-record gate errors, the benchmarking counts and the per-copy hit counts from each run, in `recorded/run1/` and `recorded/run2/`. The fade scale per chip was frozen from that chip's earlier job before these runs. Stand-in: the circuits are rebuilt on a fake IBM backend of the same name only to count depth and two-qubit gates.
