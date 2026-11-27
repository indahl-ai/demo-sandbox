"""NULL TEST (Sage's gate before push, 27 Nov 2026). Simulate the check on a stand-in chip whose noise model is built from that chip's OWN
published figures (AerSimulator.from_backend: 1q, 2q, readout, thermal). A fair method must read measured/stated near 1.0x on every pair,
and its 99% interval must cover the true figure about 99 times in 100. Usage: python tests/null_test.py [reps] [pairs]"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, 'my_circuit'))
import check_circuit as C
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKingston
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5; npairs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
be = FakeKingston(); sim = AerSimulator.from_backend(be); g = C.twoq_name(be)
pairs = [tuple(sorted(e)) for e in be.target[g]][:200]; pairs = sorted(set(pairs))[::max(1, len(set(pairs)) // npairs)][:npairs]
st2, _, _ = C.stated(be, g, {p: 1 for p in pairs}, sorted({q for p in pairs for q in p}))
ratios, cover, falses = {p: [] for p in pairs}, {p: 0 for p in pairs}, {p: 0 for p in pairs}
for r in range(reps):
    circs, meta = C.rb_circuits(be, pairs, seed=100 + r)
    res = sim.run(circs, shots=C.RB_SHOTS, seed_simulator=7 + r).result()
    counts = [res.get_counts(i) for i in range(len(circs))]
    fit = C.rb_fit(counts, meta); ci, _ = C.rb_ci(counts, meta, n_boot=200, seed=r)
    for p in pairs:
        k = '%d-%d' % p; s = st2[k]; ratios[p].append(fit[k] / s); cover[p] += ci[k][0] <= s <= ci[k][1]; falses[p] += (ci[k][0] / s > C.FLAG_HI) or (ci[k][1] / s < C.FLAG_LO)
print('NULL TEST on a FakeKingston noise model, %d reps, RB_SEQ=%d' % (reps, C.RB_SEQ))
allr = []
for p in pairs:
    k = '%d-%d' % p; allr += ratios[p]
    print('  %-8s stated %.5f   measured/stated mean %.2f  (min %.2f max %.2f)   99%% interval covered stated %d/%d'
          % (k, st2[k], np.mean(ratios[p]), min(ratios[p]), max(ratios[p]), cover[p], reps))
print('ALL PAIRS: mean ratio %.2f, median %.2f, coverage %d/%d, FALSE FLAGS %d/%d (a flag on a chip that matches its record)' % (np.mean(allr), np.median(allr), sum(cover.values()), reps * len(pairs), sum(falses.values()), reps * len(pairs)))
