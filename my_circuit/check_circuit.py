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
RB_LENGTHS, RB_SHOTS, RO_SHOTS = [1, 8, 24, 48], 300, 2000   # 27 Nov: up to 16 Cliffords could not resolve ~0.1% pairs (the fit hit r = 1); longer costs little IBM time
RB_SEQ = int(os.environ.get('MYCIRCUIT_RB_SEQ', '8'))   # 27 Nov (Sage): 3 per length gave only 10 distinct resamples; 8 per length
MAX_PAIRS = int(os.environ.get('MYCIRCUIT_MAX_PAIRS', '10'))   # most-used pairs first; keeps the job near half a minute
FLAG_HI, FLAG_LO = 1.25, 0.80   # flag margins, fixed 27 Nov 2026 13:5x UTC BEFORE the first live run (ibm_kingston, 14:03Z); never tuned on a result
TWOQ = ('cz', 'ecr', 'cx')
RESOLVE = 0.00005   # a 99% lower bound at or under this is treated as zero (the fit reached r = 1)


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


def oneq_factor(be, tc):
    """27 Nov (Sage's null test): the single-qubit gates inside each RB sequence also decay the signal. Without this the per-pair figure read
    about 1.3x high on a chip simulated at exactly its stated figures. Each physical 1q gate with stated error e1 multiplies the two-qubit
    polarization by (1 - 1.6 e1) (12 of the 15 two-qubit Paulis touch that qubit, and a 1q depolarizing parameter is 1 - 2 e1). Uses IBM's stated
    1q figures, so that part is not re-measured. rz is virtual and costs nothing."""
    t = be.target; f = 1.0
    for i in tc.data:
        name = i.operation.name
        if len(i.qubits) != 1 or name in ('rz', 'measure', 'barrier', 'delay') or name not in t.operation_names: continue
        q = tc.find_bit(i.qubits[0]).index; pr = t[name].get((q,))
        if pr is not None and pr.error: f *= (1 - 1.6 * pr.error)
    return f


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
                circs.append(tc); meta.append(['%d-%d' % e, m, n2, oneq_factor(be, tc)])
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
    """99% interval per pair. Stratified: resample sequences WITHIN each length (27 Nov, Sage), refit; returns (lo, hi) and the draws."""
    rng = np.random.default_rng(seed); out, draws = {}, {}
    for key in sorted(set(m[0] for m in meta)):
        by_len = {}
        for i, m in enumerate(meta):
            if m[0] == key: by_len.setdefault(m[1], []).append(i)
        vals = []
        for _ in range(n_boot):
            pick = [j for L in by_len.values() for j in rng.choice(L, size=len(L), replace=True)]
            vals.append(rb_fit([counts[i] for i in pick], [meta[i] for i in pick])[key])
        draws[key] = vals
        out[key] = (float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5)))
    return out, draws


def rb_fit(counts, meta):
    """per pair: P00 = 0.25 + A F1 r^n, n = transpiled two-qubit gate count, F1 = known 1q decay (oneq_factor); error per 2q gate = 0.75 (1 - r)."""
    out = {}
    for key in sorted(set(m[0] for m in meta)):
        xs = np.array([m[2] for c, m in zip(counts, meta) if m[0] == key], float)
        f1 = np.array([(m[3] if len(m) > 3 else 1.0) for c, m in zip(counts, meta) if m[0] == key], float)   # 1q decay, known from stated 1q figures
        ys = np.array([c.get('00', 0) / sum(c.values()) for c, m in zip(counts, meta) if m[0] == key], float)
        best = None
        for A in np.linspace(0.55, 0.80, 51):
            r = np.linspace(0.90, 1.0, 2001)
            sse = ((0.25 + A * f1[None, :] * r[:, None] ** xs[None, :] - ys[None, :]) ** 2).sum(1); i = int(sse.argmin())
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
    print('\n  signal kept by the two-qubit gates, from IBM\'s stated figures: %.3f   (depolarizing-model estimate, see the help)' % keep(pairs, st2))


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
    meas = rb_fit(st['rb_counts'], st['meta']); ci, draws = rb_ci(st['rb_counts'], st['meta']); s2 = st['stated2']; pairs = {tuple(map(int, k.split('-'))): n for k, n in st['pairs'].items()}
    print('YOUR CIRCUIT\'S GATES on %s, checked %s on your own account (jobs %s, %s).' % (st['backend'], st['fired'], st['rb_job'], st['ro_job']))
    if st.get('stated_when'): print("Against IBM's published figures dated %s." % st['stated_when'])
    print('\n  pair        uses   IBM stated   measured today (99% interval)       ratio (99% interval)')
    flags = 0
    for k in sorted(meas, key=lambda k: -st['pairs'][k]):
        s = s2.get(k); m = meas[k]; lo, hi = ci[k]; r = (m / s) if s else None
        # flagged only when the whole 99% interval sits above the margin (lo/s > FLAG_HI); FLAG_LO is kept for the record, no 'better' flag
        f = ''
        floor = lo is not None and lo <= RESOLVE   # interval reaches zero: this check cannot resolve the pair's error, so no flag either way
        if floor: f = '  below what this check resolves'
        elif s and lo is not None:
            if lo / s > FLAG_HI: f = '  WORSE than stated'   # 27 Nov (Sage): WORSE is the only flag; a low ratio prints with no flag
        flags += bool(f) and not floor
        iv = '' if lo is None else '[%.5f, %.5f]' % (lo, hi); rv = '' if (r is None or lo is None) else '%.2fx [%.2f, %.2f]' % (r, lo / s, hi / s)
        print('  %-10s %5d   %10s   %.5f %-20s   %s%s' % (k, st['pairs'][k], '-' if s is None else '%.5f' % s, m, iv, rv, f))
    c0, c1 = st['ro_counts']; n = len(st['qubits']); shots = sum(c0.values())
    print('\n  qubit   read 1 when 0: IBM stated / today (99% interval)      read 0 when 1: IBM stated / today (99% interval)')
    rflags = 0
    for i, q in enumerate(st['qubits']):
        k01 = sum(v for b, v in c0.items() if b[::-1][i] == '1'); k10 = sum(v for b, v in c1.items() if b[::-1][i] == '0')
        r = st['stated_ro'][str(q)]; a, b = r.get('p01'), r.get('p10')
        (l01, h01), (l10, h10) = clopper(k01, shots), clopper(k10, shots)
        # one-way against one-way (prob_meas1_prep0, prob_meas0_prep1); flagged only when the whole interval clears max(2x stated, stated + 0.01)
        bad = (a is not None and l01 > max(2 * a, a + 0.01)) or (b is not None and l10 > max(2 * b, b + 0.01))
        rflags += bad
        print('  %5d   %8s / %.4f [%.4f, %.4f]          %8s / %.4f [%.4f, %.4f]%s' % (q, '-' if a is None else '%.4f' % a, k01 / shots, l01, h01,
              '-' if b is None else '%.4f' % b, k10 / shots, l10, h10, '   WORSE than stated' if bad else ''))
    nb = min(len(v) for v in draws.values())
    km_draws = [keep(pairs, {k: (draws[k][j] if k in draws else s2.get(k)) for k in s2}) for j in range(nb)]
    ks, km = keep(pairs, s2), keep(pairs, {k: meas.get(k, s2.get(k)) for k in s2})
    print('\n  Depolarizing-model estimate of the signal your two-qubit gates keep (product of 1 - 4e/3 over the gates; ignores single-qubit error,')
    print('  readout, coherent error and crosstalk):  from IBM\'s stated figures %.3f   from today\'s measurement %.3f [%.3f, %.3f] (99%%)'
          % (ks, km, np.percentile(km_draws, 0.5), np.percentile(km_draws, 99.5)))
    print('  %d of %d pairs your circuit uses were measured. Any others keep IBM\'s stated figure in the second number.' % (len(meas), len(st['pairs'])))
    print('\nRESULT: %d of %d checked pairs (whole 99%% interval above %.2fx of IBM\'s figure, flagged WORSE) and %d of %d qubits read differently from IBM\'s published figures today.'
          % (flags, len(meas), FLAG_HI, rflags, n))
    print('Up to %d pairs are checked, each at 99%%, so across many pairs one flag can be chance. Pairs with an interval that straddles the margin are not flagged.' % MAX_PAIRS)
    print('Benchmarking gives the average error per gate on each pair, on the pairs measured, at the hour measured. It does not predict exactly how far your circuit\'s answer moves.')
    print('Your circuit stays on this machine. The check\'s results are on your IBM account and here, and nothing goes to Heuremen.')
    print('On hardware this check has read most pairs below IBM\'s figure. That is not yet explained, so a low ratio is not flagged.')
    print('The interval is labelled 99%. In a simulator test with known answers it covered the truth 92 times in 100. It covers sampling within this check, not change between checks; a repeat minutes later has differed by more.')
    print('The jobs are on your IBM account, so anyone you show this to can look them up.')


def clopper(k, n, conf=0.99):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi
