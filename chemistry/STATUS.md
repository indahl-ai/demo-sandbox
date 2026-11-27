# Replay verification, chemistry demonstration

Run 1 = the first column pair of the published chips table (job ids in recorded_chemistry/run1); run 2 = the replicate columns (run2).

| Quantity | Chip | Run | Published | Replayed | Result |
|---|---|---|---|---|---|
| client energy error, mHa | kingston | 1 | 12.8 | 12.8 | PASS |
| ours energy error, mHa | kingston | 1 | 3.5 | 3.5 | PASS |
| ratio text | kingston | 1 | 3.6x lower | 3.6x lower | PASS |
| client energy error, mHa | kingston | 2 | 12.8 | 12.8 | PASS |
| ours energy error, mHa | kingston | 2 | 2.6 | 2.6 | PASS |
| readout fix alone, mHa (page text) | kingston | 1 | 8.1 | 8.1 | PASS |
| client energy error, mHa | fez | 1 | 14.8 | 14.8 | PASS |
| ours energy error, mHa | fez | 1 | 0.9 | 0.9 | PASS |
| ratio text | fez | 1 | >5x lower | >5x lower | PASS |
| client energy error, mHa | fez | 2 | 15.3 | 15.3 | PASS |
| ours energy error, mHa | fez | 2 | 2.9 | 2.9 | PASS |
| readout fix alone, mHa (page text) | fez | 1 | 9.0 | 9.0 | PASS |
| client energy error, mHa | marrakesh | 1 | 17.8 | 17.8 | PASS |
| ours energy error, mHa | marrakesh | 1 | 4.6 | 4.6 | PASS |
| ratio text | marrakesh | 1 | 3.9x lower | 3.9x lower | PASS |
| client energy error, mHa | marrakesh | 2 | 17.0 | 17.0 | PASS |
| ours energy error, mHa | marrakesh | 2 | 2.2 | 2.2 | PASS |
| readout fix alone, mHa (page text) | marrakesh | 1 | 13.6 | 13.6 | PASS |
| outcome ours vs client, energy ruler | kingston | 1 | CONSISTENT | CONSISTENT | PASS |
| outcome ours vs client, bits-right ruler | kingston | 1 | BELOW | BELOW | PASS |
| outcome ours vs stated, energy | kingston | 1 | CONSISTENT | CONSISTENT | PASS |
| ours below stated by 1 to 5 mHa (page text) | kingston | 1 | 1 to 5 | 1.2 | PASS |
| outcome ours vs client, energy ruler | fez | 1 | BELOW | BELOW | PASS |
| outcome ours vs client, bits-right ruler | fez | 1 | BELOW | BELOW | PASS |
| outcome ours vs stated, energy | fez | 1 | CONSISTENT | CONSISTENT | PASS |
| ours below stated by 1 to 5 mHa (page text) | fez | 1 | 1 to 5 | 2.7 | PASS |
| outcome ours vs client, energy ruler | marrakesh | 1 | BELOW | BELOW | PASS |
| outcome ours vs client, bits-right ruler | marrakesh | 1 | BELOW | BELOW | PASS |
| outcome ours vs stated, energy | marrakesh | 1 | CONSISTENT | CONSISTENT | PASS |
| ours below stated by 1 to 5 mHa (page text) | marrakesh | 1 | 1 to 5 | 3.3 | PASS |
| outcome ours vs client, energy ruler | kingston | 2 | BELOW | BELOW | PASS |
| outcome ours vs client, bits-right ruler | kingston | 2 | BELOW | BELOW | PASS |
| outcome ours vs stated, energy | kingston | 2 | CONSISTENT | CONSISTENT | PASS |
| ours below stated by 1 to 5 mHa (page text) | kingston | 2 | 1 to 5 | 2.4 | PASS |
| outcome ours vs client, energy ruler | fez | 2 | BELOW | BELOW | PASS |
| outcome ours vs client, bits-right ruler | fez | 2 | BELOW | BELOW | PASS |
| outcome ours vs stated, energy | fez | 2 | CONSISTENT | CONSISTENT | PASS |
| ours below stated by 1 to 5 mHa (page text) | fez | 2 | 1 to 5 | 2.0 | PASS |
| outcome ours vs client, energy ruler | marrakesh | 2 | BELOW | BELOW | PASS |
| outcome ours vs client, bits-right ruler | marrakesh | 2 | BELOW | BELOW | PASS |
| outcome ours vs stated, energy | marrakesh | 2 | CONSISTENT | CONSISTENT | PASS |
| ours below stated by 1 to 5 mHa (page text) | marrakesh | 2 | 1 to 5 | 4.7 | PASS |
| all bits right vs stated (page text) | kingston | 1 | CONSISTENT | CONSISTENT (ours 0.836 vs 0.821) | PASS |
| all bits right vs stated (page text) | fez | 1 | BELOW | BELOW (ours 0.819 vs 0.782) | PASS |
| all bits right vs stated (page text) | marrakesh | 1 | BELOW | BELOW (ours 0.746 vs 0.706) | PASS |
| all bits right vs stated (page text) | kingston | 2 | BELOW | BELOW (ours 0.852 vs 0.823) | PASS |
| all bits right vs stated (page text) | fez | 2 | BELOW | BELOW (ours 0.788 vs 0.768) | PASS |
| all bits right vs stated (page text) | marrakesh | 2 | BELOW | BELOW (ours 0.788 vs 0.729) | PASS |

ALL PASS

- 27 Nov 2026 (Sage): the bits-right comparison against the stated-figure correction (BELOW or CONSISTENT) is no longer page text. Our measured gate figure comes from a benchmark fit that reads high, and a larger correction can score better on bits right without being more accurate. The energy comparison (1 to 5 mHa lower, inside the 10 mHa margin, not a shown difference) and the comparison against the default run stand.
