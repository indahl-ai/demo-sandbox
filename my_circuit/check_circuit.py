"""check_circuit.py: check YOUR circuit's gates against the chip as it is today.

  python sandbox.py mycircuit plan    FILE.qasm --backend ibm_kingston     lay the circuit out, list its gates and IBM's stated figures. Submits nothing.
  python sandbox.py mycircuit fire    FILE.qasm --backend ibm_kingston --yes   one short check job on YOUR account (the gates your circuit uses)
  python sandbox.py mycircuit fetch   FILE.qasm --backend ibm_kingston     collect it
  python sandbox.py mycircuit analyze FILE.qasm --backend ibm_kingston     IBM's stated figure beside today's measurement, gate by gate

Your circuit never leaves this machine. It is laid out (transpiled) here, and only the check circuits go to IBM: short randomized-benchmarking
sequences on the two-qubit pairs your layout uses, and an all-0 / all-1 readout check on its qubits. Your key comes from IBM_QUANTUM_TOKEN.
Add --fake kingston|fez|marrakesh to `plan` to try it offline on a stand-in chip, with no key.

What it measures, and what it does not: randomized benchmarking gives the AVERAGE error per two-qubit gate on each pair, today. It does not
predict exactly how far your circuit's answer moves. The "signal kept" lines multiply (1 - 4e/3) over your circuit's two-qubit gates, a rough
guide that ignores single-qubit gates, idling and crosstalk.
"""
import os, sys, json, time, random, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RB_LENGTHS, RB_SEQ, RB_SHOTS, RO_SHOTS = [1, 4, 8, 16], 3, 300, 2000
MAX_PAIRS = int(os.environ.get('MYCIRCUIT_MAX_PAIRS', '10'))   # most-used pairs first; keeps the job near half a minute
FLAG_HI, FLAG_LO = 1.25, 0.80
TWOQ = ('cz', 'ecr', 'cx')


def load_circuit(path):
    from qiskit import qasm2, qasm3
    src = open(path).read()
    if 'OPENQASM 3' in src: return qasm3.loads(src)
    return qasm2.loads(src, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def backend(name, fake=None):
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
        return {'kingston': FakeKingston, 'fez': FakeFez, 'marrakesh': FakeMarrakesh}[fake](), None
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'finance'))
    if not os.environ.get('IBM_QUANTUM_TOKEN', '').strip():
        sys.exit('Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine), or add --fake kingston to plan offline')
    from qiskit_ibm_runtime import QiskitRuntimeService as S
    sys.path.insert(0, os.path.dirname(HERE)); import pick_instance
    tok = os.environ['IBM_QUANTUM_TOKEN'].strip()
    base = S(channel='ibm_quantum_platform', token=tok)
    svc = S(channel='ibm_quantum_platform', token=tok, instance=pick_instance.crn(base))
    return svc.backend(name), svc


def twoq_name(be):
    for g in TWOQ:
        if g in be.target.operation_names: return g
    sys.exit('this chip has no cz, ecr or cx gate in its target')


def layout(qc, be, opt):
    from qiskit import transpile
    return transpile(qc, backend=be, optimization_level=opt, seed_transpiler=7)


def used(tc):
    pairs, qubits = {}, set()
    for i in tc.data:
        idx = [tc.find_bit(x).index for x in i.qubits]
        if i.operation.name == 'measure': qubits.update(idx)
        if i.operation.name in TWOQ:
            p = tuple(sorted(idx)); pairs[p] = pairs.get(p, 0) + 1; qubits.update(idx)
    return pairs, sorted(qubits)


def stated(be, g, pairs, qubits):
    t = be.target; st2, ro = {}, {}
    for a, b in pairs:
        p = t[g].get((a, b)) or t[g].get((b, a))
        st2['%d-%d' % (a, b)] = None if p is None else p.error
    props = None
    try: props = be.properties()
    except Exception: pass
    for q in qubits:
        m = t['measure'].get((q,)) if 'measure' in t.operation_names else None
        p01 = p10 = None
        if props is not None:
            try: p01 = props.qubit_property(q, 'prob_meas1_prep0')[0]; p10 = props.qubit_property(q, 'prob_meas0_prep1')[0]
            except Exception: pass
        ro[str(q)] = {'two_way': None if m is None else m.error, 'p01': p01, 'p10': p10}
    when = None
    try: when = props.last_update_date.isoformat()
    except Exception: pass
    return st2, ro, when


def keep(pairs, errs):
    v = 1.0
    for p, n in pairs.items():
        e = errs.get('%d-%d' % p)
        if e is not None: v *= (1 - 4 * e / 3) ** n
    return v


def rb_circuits(be, chosen, seed=11):
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import random_clifford, Clifford
    rng = random.Random(seed); circs, meta = [], []
    for e in chosen:
        for m in RB_LENGTHS:
            for _ in range(RB_SEQ):
                qc = QuantumCircuit(2); comp = Clifford(QuantumCircuit(2))
                for _k in range(m):
                    c = random_clifford(2, seed=rng.randrange(10**9)); qc.compose(c.to_circuit(), inplace=True); comp = comp.compose(c)
                qc.compose(comp.adjoint().to_circuit(), inplace=True); qc.measure_all()
                tc = transpile(qc, backend=be, initial_layout=list(e), optimization_level=1, seed_transpiler=5)
                n2 = sum(v for k, v in tc.count_ops().items() if k in TWOQ)
                circs.append(tc); meta.append(['%d-%d' % e, m, n2])
    return circs, meta


def ro_circuits(be, qubits):
    from qiskit import QuantumCircuit, transpile
    out = []
    for ones in (False, True):
        qc = QuantumCircuit(len(qubits), len(qubits))
        if ones: qc.x(range(len(qubits)))
        qc.measure(range(len(qubits)), range(len(qubits)))
        out.append(transpile(qc, backend=be, initial_layout=qubits, optimization_level=0))
    return out


def rb_ci(counts, meta, n_boot=400, seed=3):
    """99% interval per pair: resample that pair's benchmarking circuits with replacement and refit (27 Nov: a flag needs an interval, Sage's rule)."""
    rng = np.random.default_rng(seed); out = {}
    for key in sorted(set(m[0] for m in meta)):
        idx = [i for i, m in enumerate(meta) if m[0] == key]; vals = []
        for _ in range(n_boot):
            pick = rng.choice(idx, size=len(idx), replace=True)
            if len(set(meta[i][2] for i in pick)) < 2: continue
            vals.append(rb_fit([counts[i] for i in pick], [meta[i] for i in pick])[key])
        out[key] = (float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))) if vals else (None, None)
    return out


def rb_fit(counts, meta):
    """per pair: P00 = 0.25 + A r^n over the transpiled two-qubit gate count n; error per gate = 0.75 (1 - r)."""
    out = {}
    for key in sorted(set(m[0] for m in meta)):
        xs = np.array([m[2] for c, m in zip(counts, meta) if m[0] == key], float)
        ys = np.array([c.get('00', 0) / sum(c.values()) for c, m in zip(counts, meta) if m[0] == key], float)
        best = None
        for A in np.linspace(0.55, 0.75, 41):
            r = np.linspace(0.90, 1.0, 2001)
            sse = ((0.25 + A * r[:, None] ** xs[None, :] - ys[None, :]) ** 2).sum(1); i = int(sse.argmin())
            if best is None or sse[i] < best[0]: best = (sse[i], r[i])
        out[key] = float(0.75 * (1 - best[1]))
    return out


def est_seconds(circs, shots): return 3.0 + sum(shots * (0.0006 + 1e-6 * c.depth()) for c in circs)


def state_path(path, name):
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:10]
    d = os.path.join(os.path.dirname(HERE), 'my_runs'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'mycircuit_%s_%s.json' % (name, h))


def counts_of(pub):
    d = pub.data
    reg = [getattr(d, a) for a in dir(d) if not a.startswith('_') and hasattr(getattr(d, a), 'get_counts')][0]
    return reg.get_counts()


def show_plan(name, tc, pairs, qubits, st2, chosen, when):
    print('YOUR CIRCUIT on %s, laid out here (nothing sent): %d qubits used, %d two-qubit gates on %d pairs, depth %d.'
          % (name, len(qubits), sum(pairs.values()), len(pairs), tc.depth()))
    if when: print("IBM's published figures dated %s." % when)
    print('\n  pair        uses   IBM stated error   checked')
    for p, n in sorted(pairs.items(), key=lambda x: -x[1]):
        k = '%d-%d' % p; s = st2.get(k)
        print('  %-10s %5d   %16s   %s' % (k, n, '-' if s is None else '%.5f' % s, 'yes' if p in chosen else 'no (MYCIRCUIT_MAX_PAIRS)'))
    print('\n  signal kept by the two-qubit gates, from IBM\'s stated figures: %.3f   (rough guide, see the note in the help)' % keep(pairs, st2))


def run(cmd, path, name, yes=False, fake=None, opt=1):
    qc = load_circuit(path)
    be, svc = backend(name, fake)
    g = twoq_name(be); tc = layout(qc, be, opt); pairs, qubits = used(tc)
    if not pairs: sys.exit('this circuit has no two-qubit gates after layout; there is nothing for the gate check to measure')
    chosen = [p for p, _ in sorted(pairs.items(), key=lambda x: -x[1])[:MAX_PAIRS]]
    st2, ro, when = stated(be, g, pairs, qubits)
    sp = state_path(path, name if not fake else 'fake_' + fake)

    if cmd == 'plan':
        show_plan(name if not fake else 'a stand-in for ibm_' + fake, tc, pairs, qubits, st2, chosen, when)
        rbc, _ = rb_circuits(be, chosen); roc = ro_circuits(be, qubits)
        sec = est_seconds(rbc, RB_SHOTS) + est_seconds(roc, RO_SHOTS)
        print('\nThe check: %d benchmarking circuits on %d pairs and a readout check on %d qubits, about %.0f s of IBM time (estimate).' % (len(rbc), len(chosen), len(qubits), sec))
        if svc is not None:
            try: print('Your remaining IBM time: %.0f s' % float(svc.usage().get('usage_remaining_seconds')))
            except Exception: pass
        print('Nothing was submitted. To run the check: python sandbox.py mycircuit fire %s --backend %s --yes' % (path, name))
        return

    if fake: sys.exit('--fake is for plan only')
    if cmd == 'fire':
        if not yes: sys.exit('This submits one job on your IBM account and uses your IBM time. Run plan first, then add --yes.')
        rbc, meta = rb_circuits(be, chosen); roc = ro_circuits(be, qubits)
        left = float(svc.usage().get('usage_remaining_seconds') or 0); need = est_seconds(rbc, RB_SHOTS) + est_seconds(roc, RO_SHOTS)
        if left < need + 30: sys.exit('refusing: %.0f s of IBM time left, the check needs about %.0f s' % (left, need))
        from qiskit_ibm_runtime import SamplerV2
        s = SamplerV2(mode=be)
        try: s.options.dynamical_decoupling.enable = False; s.options.twirling.enable_gates = False; s.options.twirling.enable_measure = False
        except Exception: pass
        j1 = s.run(rbc, shots=RB_SHOTS); j2 = s.run(roc, shots=RO_SHOTS)
        json.dump(dict(backend=name, circuit_sha=os.path.basename(sp), fired=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), rb_job=j1.job_id(), ro_job=j2.job_id(),
                       meta=meta, qubits=qubits, pairs={'%d-%d' % p: n for p, n in pairs.items()}, stated2=st2, stated_ro=ro, stated_when=when), open(sp, 'w'), indent=1)
        print('Submitted on your account: benchmarking job %s, readout job %s.\nNext: python sandbox.py mycircuit fetch %s --backend %s' % (j1.job_id(), j2.job_id(), path, name))
        return

    if not os.path.exists(sp): sys.exit('no check fired for this circuit on %s yet: run fire first' % name)
    st = json.load(open(sp))
    if cmd == 'fetch':
        out = {}
        for k in ('rb_job', 'ro_job'):
            j = svc.job(st[k]); s = str(j.status()); print('%s %s: %s' % (k[:2], st[k], s))
            if s.endswith('DONE'): out[k] = [counts_of(p) for p in j.result()]
        if len(out) == 2:
            st['rb_counts'], st['ro_counts'] = out['rb_job'], out['ro_job']; json.dump(st, open(sp, 'w'), indent=1)
            print('DONE. Next: python sandbox.py mycircuit analyze %s --backend %s' % (path, name))
        else: print('Not finished yet. Run fetch again in about 30 seconds.')
        return

    if cmd == 'analyze':
        if 'rb_counts' not in st: sys.exit('not fetched yet: run fetch until it reports DONE')
        analyze(st)


def analyze(st):
    meas = rb_fit(st['rb_counts'], st['meta']); ci = rb_ci(st['rb_counts'], st['meta']); s2 = st['stated2']; pairs = {tuple(map(int, k.split('-'))): n for k, n in st['pairs'].items()}
    print('YOUR CIRCUIT\'S GATES on %s, checked %s on your own account (jobs %s, %s).' % (st['backend'], st['fired'], st['rb_job'], st['ro_job']))
    if st.get('stated_when'): print("Against IBM's published figures dated %s." % st['stated_when'])
    print('\n  pair        uses   IBM stated   measured today (99% interval)       ratio (99% interval)')
    flags = 0
    for k in sorted(meas, key=lambda k: -st['pairs'][k]):
        s = s2.get(k); m = meas[k]; lo, hi = ci[k]; r = (m / s) if s else None
        # flagged only when the whole 99% interval sits past the margin: WORSE if lo/s > FLAG_HI, better if hi/s < FLAG_LO
        f = ''
        if s and lo is not None:
            if lo / s > FLAG_HI: f = '  WORSE than stated'
            elif hi / s < FLAG_LO: f = '  better than stated'
        flags += bool(f)
        iv = '' if lo is None else '[%.5f, %.5f]' % (lo, hi); rv = '' if (r is None or lo is None) else '%.2fx [%.2f, %.2f]' % (r, lo / s, hi / s)
        print('  %-10s %5d   %10s   %.5f %-20s   %s%s' % (k, st['pairs'][k], '-' if s is None else '%.5f' % s, m, iv, rv, f))
    c0, c1 = st['ro_counts']; n = len(st['qubits']); shots = sum(c0.values())
    print('\n  qubit   read 1 when 0: stated / today     read 0 when 1: stated / today')
    rflags = 0
    for i, q in enumerate(st['qubits']):
        p01 = sum(v for b, v in c0.items() if b[::-1][i] == '1') / shots; p10 = sum(v for b, v in c1.items() if b[::-1][i] == '0') / shots
        r = st['stated_ro'][str(q)]; a, b = r.get('p01'), r.get('p10')
        bad = (a is not None and p01 > max(2 * a, a + 0.01)) or (b is not None and p10 > max(2 * b, b + 0.01))
        rflags += bad
        print('  %5d   %8s / %.4f              %8s / %.4f%s' % (q, '-' if a is None else '%.4f' % a, p01, '-' if b is None else '%.4f' % b, p10, '   WORSE than stated' if bad else ''))
    mfull = {k: meas.get(k, s2.get(k)) for k in s2}
    ks, km = keep(pairs, s2), keep(pairs, mfull)
    print('\n  signal kept by your two-qubit gates:  from IBM\'s stated figures %.3f   from today\'s measurement %.3f' % (ks, km))
    print('  (pairs not checked keep IBM\'s stated figure in the second number)')
    print('\nRESULT: %d of %d checked pairs (whole 99%% interval past %.2fx or under %.2fx of IBM\'s figure) and %d of %d qubits read differently from IBM\'s published figures today.'
          % (flags, len(meas), FLAG_HI, FLAG_LO, rflags, n))
    print('Benchmarking gives the average error per gate on each pair. It does not predict exactly how far your circuit\'s answer moves.')
    print('The jobs are on your IBM account, so anyone you show this to can look them up.')
