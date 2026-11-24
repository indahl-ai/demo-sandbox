#!/usr/bin/env python
"""OPTIMIZATION / LOGISTICS DEMO: the client's own QAOA (level-3 transpile, layout from the record), ONE layout, ONE set of circuits.
Three readings of the SAME shots:
  N = no readout correction (the tutorial)     R = correction with the record's readout figures (a careful client)
  O = correction with today's job-P table (ours)
Job P: all-0 / all-1 readout calibration on the layout's qubits, plus the readout check (qubits worse today than the record says).
Main job: the client's QAOA circuit, COPIES copies. Approximation ratio per reading with a 99% interval from the copies; outcome word for O minus R.
Modes: --test --dry --fire --fetch --analyze.   Env IBM_QUANTUM_TOKEN (your own key) / DEMO_INSTANCE.
Never prints a key or a CRN. Nothing here is committed or pushed. The static-ZZ fold (option c) is NOT built: see PLAN.md."""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'run')
FORGE = HERE
BACKEND = os.environ.get('DEMO_BACKEND', 'ibm_kingston')
N_NODES = 100
SHOTS = 10_000
COPIES = 12
P_SHOTS = 4_096
MAX_TOTAL_S = 200.0
SEC_PER_SHOT_CIRCUIT = 0.30e-3     # rate the earlier runs used (112 s planned for 36 x 10,000)
SEC_PER_JOB = 4.0
EST_MARGIN = 1.25                  # refuse on estimate x margin
GAP_MIN = 0.5                      # denominator guard: 1 - p10 - p01 < 0.5 drops a qubit's edges from every reading
BAND = 0.02                        # CONSISTENT = interval inside +-BAND
SEED = 20261118
TABLE_DRAWS = 200


def fp(name): return os.path.join(RUN, name)


# ------------------------------------------------------------ graph, angles, exact p = 1 (from the earlier optimization demo) ----
def build_graph(coupling_edges, n=N_NODES):
    """The tutorial's rule: every coupler with both ends < n, endpoints sorted, duplicates folded."""
    return sorted({(min(u, v), max(u, v)) for u, v in coupling_edges if u < n and v < n and u != v})


def cost_op(edges, n):
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_sparse_list([('ZZ', [u, v], 0.5) for u, v in edges], num_qubits=n)


def neighbours(edges):
    nb = {}
    for u, v in edges:
        nb.setdefault(u, set()).add(v)
        nb.setdefault(v, set()).add(u)
    return nb


def zz_p1(edges, gamma, beta, nb=None):
    """{(u, v): <Z_u Z_v>} at p = 1, exact, by light cone."""
    nb = nb or neighbours(edges)
    eset = set(edges)
    out = {}
    for u, v in edges:
        qs = sorted({u, v} | nb[u] | nb[v])
        k = len(qs)
        idx = {q: i for i, q in enumerate(qs)}
        loc = [e for e in eset if (e[0] in (u, v) or e[1] in (u, v))]
        bits = ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
        z = 1 - 2 * bits
        phase = np.zeros(2 ** k)
        for a, b in loc:
            phase += -0.5 * gamma * z[:, idx[a]] * z[:, idx[b]]
        psi = (np.exp(1j * phase) / math.sqrt(2 ** k)).reshape([2] * k)
        c, s = math.cos(beta), -1j * math.sin(beta)
        rx = np.array([[c, s], [s, c]])
        for q in (u, v):
            ax = k - 1 - idx[q]
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [ax])), 0, ax)
        p = np.abs(psi.reshape(-1)) ** 2
        out[(u, v)] = float(np.sum(p * z[:, idx[u]] * z[:, idx[v]]))
    return out


def expected_cut(edges, gamma, beta, nb=None):
    return float(sum((1 - x) / 2 for x in zz_p1(edges, gamma, beta, nb).values()))


def best_angles(edges, grid=48):
    from scipy.optimize import minimize
    nb = neighbours(edges)
    G = np.linspace(-math.pi, math.pi, grid, endpoint=False)
    B = np.linspace(-math.pi / 2, math.pi / 2, grid // 2, endpoint=False)
    _, g0, b0 = max((expected_cut(edges, g, b, nb), g, b) for g in G for b in B)
    r = minimize(lambda x: -expected_cut(edges, x[0], x[1], nb), [g0, b0], method='Nelder-Mead',
                 options=dict(xatol=1e-7, fatol=1e-9, maxiter=2000))
    return float(r.x[0]), float(r.x[1]), -float(r.fun)


def exact_maxcut(edges, n):
    from scipy.optimize import milp, LinearConstraint, Bounds
    m = len(edges)
    c = np.concatenate([np.zeros(n), -np.ones(m)])
    A = np.zeros((2 * m, n + m))
    lb, ub = np.full(2 * m, -np.inf), np.zeros(2 * m)
    for i, (u, v) in enumerate(edges):
        A[2 * i, [u, v, n + i]] = [-1, -1, 1]
        A[2 * i + 1, [u, v, n + i]] = [1, 1, 1]
        ub[2 * i + 1] = 2
    r = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(n + m), bounds=Bounds(0, 1))
    return int(round(-r.fun))


def angles_cached(edges, n):
    os.makedirs(RUN, exist_ok=True)
    pth = fp('angles_%d_%d.json' % (n, len(edges)))
    if os.path.exists(pth):
        j = json.load(open(pth))
        if [tuple(e) for e in j['edges']] == list(edges):
            return j['gamma'], j['beta'], j['ideal'], j['maxcut']
    g, b, ideal = best_angles(edges)
    mc = exact_maxcut(edges, n)
    json.dump(dict(edges=[list(e) for e in edges], gamma=g, beta=b, ideal=ideal, maxcut=mc), open(pth, 'w'))
    return g, b, ideal, mc


# ------------------------------------------------------------ circuits ------------------------------------------------------------
def build_logical(edges, n, gamma, beta):
    from qiskit.circuit.library import QAOAAnsatz
    qc = QAOAAnsatz(cost_operator=cost_op(edges, n), reps=1)
    qc.measure_all()
    par = {p.name: p for p in qc.parameters}
    return qc.assign_parameters({par['γ[0]']: gamma, par['β[0]']: beta})


def transpile_client(qc, backend):
    """The tutorial's pass manager: level 3, layout chosen by the transpiler from the backend's record. Fixed seed. -> circuit, layout."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    c = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=SEED).run(qc)
    return c, [int(x) for x in c.layout.final_index_layout()]


def twoq_info(c):
    n, edges = 0, set()
    for inst in c.data:
        if inst.operation.num_qubits == 2 and inst.operation.name not in ('barrier',):
            n += 1
            edges.add(tuple(sorted(c.find_bit(q).index for q in inst.qubits)))
    return n, sorted(edges)


def p_circuits(nphys, layout):
    """P0: prepare nothing, measure the layout's qubits (bit i = virtual node i). P1: X on each, measure."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    out = []
    for k in (0, 1):
        qr, cr = QuantumRegister(nphys, 'q'), ClassicalRegister(len(layout), 'c')
        c = QuantumCircuit(qr, cr, name='P%d' % k)
        if k:
            for q in layout:
                c.x(qr[q])
        for i, q in enumerate(layout):
            c.measure(qr[q], cr[i])
        out.append(c)
    return out


def sampler(b, tutorial=True):
    from qiskit_ibm_runtime import SamplerV2
    smp = SamplerV2(mode=b)
    if tutorial:      # the QAOA tutorial's sampling options, set explicitly; no readout mitigation
        smp.options.dynamical_decoupling.enable = True
        smp.options.dynamical_decoupling.sequence_type = 'XY4'
        smp.options.twirling.enable_gates = True
        smp.options.twirling.num_randomizations = 'auto'
    else:             # job P: plain readout, nothing added
        smp.options.dynamical_decoupling.enable = False
        smp.options.twirling.enable_gates = False
        smp.options.twirling.enable_measure = False
    return smp


def est_seconds(shot_circuits, jobs=1): return SEC_PER_SHOT_CIRCUIT * shot_circuits + SEC_PER_JOB * jobs


def est_total(n_copies=COPIES, shots=SHOTS, p_shots=P_SHOTS):
    return est_seconds(2 * p_shots) + est_seconds(n_copies * shots)


# ------------------------------------------------------------ readings ------------------------------------------------------------
def unpack(arr, n):
    """BitArray .array (shots x nbytes, uint8) -> shots x n of {0,1}, column j = classical bit j."""
    arr = np.asarray(arr, np.uint8)
    nb = arr.shape[1]
    return np.stack([(arr[:, nb - 1 - j // 8] >> (j % 8)) & 1 for j in range(n)], 1)


def bits_from_counts(counts, n):
    rows = []
    for k, v in counts.items():
        k = k.replace(' ', '')
        rows.append(np.tile(np.array([int(k[len(k) - 1 - i]) for i in range(n)], np.uint8), (v, 1)))
    return np.concatenate(rows, 0)


def marg_bits(bits, edges):
    z = 1 - 2 * bits.astype(np.int8)
    return z.mean(0), {(u, v): float(np.mean(z[:, u] * z[:, v])) for u, v in edges}


def correct(Z, ZZ, p10, p01):
    """Tensored per-qubit readout inversion on one- and two-qubit marginals (the model the vendor's mitigation solves)."""
    p10, p01 = np.asarray(p10, float), np.asarray(p01, float)
    g, c = 1 - p10 - p01, p01 - p10
    Zt = (np.asarray(Z) - c) / g
    return Zt, {(u, v): (m - g[u] * c[v] * Zt[u] - c[u] * g[v] * Zt[v] - c[u] * c[v]) / (g[u] * g[v]) for (u, v), m in ZZ.items()}


def ratio(ZZ, exact, kept):
    """Approximation ratio: measured cut over the noiseless QAOA cut, on the kept edges."""
    return float(sum((1 - ZZ[e]) / 2 for e in kept) / sum((1 - exact[e]) / 2 for e in kept))


def table_from_bits(b0, b1):
    """P bits -> {column: (p10, p01, k10, n0, k01, n1)}."""
    n0, n1 = b0.shape[0], b1.shape[0]
    k10, k01 = b0.sum(0), (1 - b1).sum(0)
    return {i: (k10[i] / n0, k01[i] / n1, int(k10[i]), n0, int(k01[i]), n1) for i in range(b0.shape[1])}


def cp99(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.005, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.995, k + 1, n - k))
    return lo, hi


def size_rule(s, m, lo, hi):
    """Record s against today's m with m's 99% Clopper-Pearson interval [lo, hi]: 'worse' if the record is below the interval and
    today's figure is over 2 x record + 0.01 (the QIA readout rule, worse direction only)."""
    return 'worse' if (s < lo and m > 2 * s + 0.01) else None


def readout_check(rec_v, ptab_v):
    """{virtual node: {par: (record, today, lo, hi)}} for nodes whose readout today is worse than the record says."""
    out = {}
    for i, (p10, p01, k10, n0, k01, n1) in ptab_v.items():
        fl = {}
        for nm, s, k, n in (('p10', rec_v[i][0], k10, n0), ('p01', rec_v[i][1], k01, n1)):
            lo, hi = cp99(k, n)
            if size_rule(s, k / n, lo, hi):
                fl[nm] = (s, k / n, lo, hi)
        if fl:
            out[i] = fl
    return out


def record_readout(props):
    st = {}
    for q, ps in enumerate(props.get('qubits', [])):
        v = {x['name']: x for x in ps}
        if 'prob_meas1_prep0' in v and 'prob_meas0_prep1' in v:
            st[q] = (float(v['prob_meas1_prep0']['value']), float(v['prob_meas0_prep1']['value']))
    return st


def virt(tab_phys, layout): return {i: tab_phys[q] for i, q in enumerate(layout)}


def guard_nodes(tab, n): return {i for i in range(n) if 1 - tab[i][0] - tab[i][1] < GAP_MIN}


def pv(tab_v, n): return np.array([tab_v[i][0] for i in range(n)]), np.array([tab_v[i][1] for i in range(n)])


def interval99(vals, sd_fixed=0.0):
    from scipy.stats import t
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return float('nan'), float('nan'), float('nan')
    m = float(v.mean())
    se = math.sqrt(float(v.var(ddof=1)) / n + float(sd_fixed or 0.0) ** 2)
    h = float(t.ppf(0.995, n - 1) * se)
    return m, m - h, m + h


def word(lo, hi):
    """Equivalence first: an interval wholly inside +-BAND is CONSISTENT (a real but tiny offset is not a finding); then ABOVE / BELOW."""
    if lo != lo:
        return 'CANNOT_TELL'
    if lo > -BAND and hi < BAND:
        return 'CONSISTENT'
    if lo > 0:
        return 'ABOVE'
    if hi < 0:
        return 'BELOW'
    return 'CANNOT_TELL'


def table_fixed_sd(ptab_v, meanZ, meanZZ, rec_v, exact, kept, n, draws=TABLE_DRAWS, seed=SEED):
    """Fixed-per-run error of P's table (its own shots): redraw each node's p10, p01 binomially; sd of O and of O - R over draws.
    The correction is linear in the marginals, so the copy-mean of the corrected ratio is the ratio of the corrected mean marginals."""
    rng = np.random.default_rng(seed)
    R = ratio(correct(meanZ, meanZZ, *pv(rec_v, n))[1], exact, kept)
    o, d = [], []
    for _ in range(draws):
        p10 = np.array([rng.binomial(ptab_v[i][3], ptab_v[i][0]) / ptab_v[i][3] for i in range(n)])
        p01 = np.array([rng.binomial(ptab_v[i][5], ptab_v[i][1]) / ptab_v[i][5] for i in range(n)])
        r = ratio(correct(meanZ, meanZZ, p10, p01)[1], exact, kept)
        o.append(r)
        d.append(r - R)
    return float(np.std(o, ddof=1)), float(np.std(d, ddof=1))


def three_readings(copy_bits, ptab_v, rec_v, edges, exact, n, same_edges=True):
    """copy_bits: list of shots x n bit arrays, one per copy. ptab_v: today's table per virtual node; rec_v: the record per virtual node.
    N, R, O read the SAME shots on the SAME kept edges (same_edges=False: N and R on all edges, O on the guard-kept edges: the earlier
    run's convention, used only for the cross-check against its numbers)."""
    gP, gR = guard_nodes(ptab_v, n), guard_nodes(rec_v, n)
    if same_edges:
        drop = gP | gR
        keptN = keptR = keptO = [e for e in edges if e[0] not in drop and e[1] not in drop]
    else:
        keptN = keptR = list(edges)
        keptO = [e for e in edges if e[0] not in gP and e[1] not in gP]
    p10P, p01P = pv(ptab_v, n)
    p10R, p01R = pv(rec_v, n)
    rN, rR, rO = [], [], []
    sZ, sZZ = np.zeros(n), {e: 0.0 for e in edges}
    for bits in copy_bits:
        Z, ZZ = marg_bits(bits, edges)
        sZ += Z / len(copy_bits)
        for e in edges:
            sZZ[e] += ZZ[e] / len(copy_bits)
        rN.append(ratio(ZZ, exact, keptN))
        rR.append(ratio(correct(Z, ZZ, p10R, p01R)[1], exact, keptR))
        rO.append(ratio(correct(Z, ZZ, p10P, p01P)[1], exact, keptO))
    sdO, sdD = table_fixed_sd(ptab_v, sZ, sZZ, rec_v, exact, keptO, n)
    diff = np.array(rO) - np.array(rR)
    return dict(N=interval99(rN), R=interval99(rR), O=interval99(rO, sdO), D=interval99(diff, sdD), kept=len(keptO), n_edges=len(edges),
                guard_P=sorted(gP), guard_R=sorted(gR), sd_table=(sdO, sdD), copies=len(copy_bits))


def report(res, ideal_over_max, label=''):
    f = lambda x: '%.4f (99%% interval %.4f to %.4f)' % x
    w = word(res['D'][1], res['D'][2])
    L = ['%sapproximation ratio = measured cut / noiseless QAOA cut (noiseless QAOA itself reaches %.3f of the exact max cut); %d copies; %d of %d edges kept in the O reading'
         % (label, ideal_over_max, res['copies'], res['kept'], res['n_edges']),
         '%s  N no readout correction (the tutorial):          %s' % (label, f(res['N'])),
         "%s  R correction with the record's readout figures:  %s" % (label, f(res['R'])),
         "%s  O correction with today's job-P table (ours):    %s  [P-table error sd %.5f included]" % (label, f(res['O']), res['sd_table'][0]),
         '%s  OUTCOME O minus R: %+.4f (99%% interval %+.4f to %+.4f, P-table error sd %.5f in quadrature): %s  (ABOVE = ours higher, BELOW = ours lower, CONSISTENT = inside +-%.2f)'
         % (label, res['D'][0], res['D'][1], res['D'][2], res['sd_table'][1], w, BAND),
         '%s  O minus N: %+.4f' % (label, res['O'][0] - res['N'][0]),
         '%s  denominator guard (1 - p10 - p01 < %.1f), virtual nodes: today %s, record %s' % (label, GAP_MIN, res['guard_P'] or 'none', res['guard_R'] or 'none')]
    return '\n'.join(L)


# ------------------------------------------------------------ backend, key, account -----------------------------------------------
def read_key():
    t = os.environ.get('IBM_QUANTUM_TOKEN', '').strip()
    if not t: sys.exit('no IBM Quantum key: set IBM_QUANTUM_TOKEN (your own key; it stays on this machine)')
    return t


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService as S
    tok, want = read_key(), os.environ.get('DEMO_INSTANCE', 'open-instance')
    base = S(channel='ibm_quantum_platform', token=tok)
    crn = None
    for i in base.instances():
        if want.lower() in str(i.get('name', '')).lower():
            crn = i['crn']
            break
    if crn is None:
        sys.exit("no instance named like '%s' on this key" % want)
    return S(channel='ibm_quantum_platform', token=tok, instance=crn)


def usage_remaining(svc):
    try:
        return float(svc.usage().get('usage_remaining_seconds'))
    except Exception as e:  # noqa: BLE001
        sys.exit('cannot read usage_remaining_seconds (%s); refusing' % type(e).__name__)


def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def props_of(b): return json.loads(json.dumps(b.properties().to_dict(), default=str))


def prepare(b, n):
    """Graph, angles, circuit, layout on backend b with n nodes."""
    edges = build_graph(b.coupling_map.get_edges(), n)
    gamma, beta, ideal, maxcut = angles_cached(edges, n)
    c, lay = transpile_client(build_logical(edges, n, gamma, beta), b)
    return dict(edges=edges, gamma=gamma, beta=beta, ideal=ideal, maxcut=maxcut, circ=c, layout=lay)


def save_raw(path, res):
    arrs = {}
    for i, pr in enumerate(res):
        for reg in pr.data:
            arrs['%s_%d' % (reg, i)] = np.asarray(getattr(pr.data, reg).array)
    np.savez_compressed(path, **arrs)


def refusals(tot, rem):
    why = []
    if tot * EST_MARGIN > MAX_TOTAL_S:
        why.append('estimate x%.2f = %.0f s is over the %.0f s limit' % (EST_MARGIN, tot * EST_MARGIN, MAX_TOTAL_S))
    if rem is not None and tot * EST_MARGIN > rem:
        why.append('estimate x%.2f = %.0f s is over the %.0f s remaining' % (EST_MARGIN, tot * EST_MARGIN, rem))
    return why


# ------------------------------------------------------------ modes ----------------------------------------------------------------
def mode_test():
    """Offline: Aer on the fake backend. 20-node QAOA on the fake chip's own coupler graph; readout-only noise from the fake record, with
    5 qubits of the layout PLANTED worse than the record says. Gate noise is not simulated (a noisy statevector run would not finish), so
    the ideal ratio is 1; the test checks that O recovers it, that N and R do not, and that the check flags exactly the plants."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    NT, SH = 20, 4000
    b = fake_backend()
    rec_phys = record_readout(props_of(b))
    x = prepare(b, NT)
    lay, edges = x['layout'], x['edges']
    print('[SIMULATION] fake kingston, %d nodes, %d edges; level 3: %d two-qubit gates, depth %d; layout %s'
          % (NT, len(edges), twoq_info(x['circ'])[0], x['circ'].depth(), lay))
    plant = [lay[i] for i in (2, 5, 9, 13, 17)]
    truth = dict(rec_phys)
    for q in plant:
        truth[q] = (0.06, 0.08)
    print('[SIMULATION] planted readout faults on physical qubits %s: truth p10 0.06 p01 0.08; the record says %s'
          % (plant, [tuple(round(v, 4) for v in rec_phys[q]) for q in plant]))
    nm = NoiseModel()
    for q, (p10, p01) in truth.items():
        nm.add_readout_error(ReadoutError([[1 - p10, p10], [p01, 1 - p01]]), [q])
    sim = AerSimulator(noise_model=nm)
    pb = [bits_from_counts(sim.run(c, shots=P_SHOTS, seed_simulator=11 + k).result().get_counts(), NT)
          for k, c in enumerate(p_circuits(b.num_qubits, lay))]
    ptab_v = table_from_bits(pb[0], pb[1])
    rec_v = virt(rec_phys, lay)
    fl = readout_check(rec_v, ptab_v)
    print('[SIMULATION] readout check flags physical %s (planted %s): %s' % (sorted(lay[i] for i in fl), sorted(plant),
          'MATCH' if sorted(lay[i] for i in fl) == sorted(plant) else 'MISMATCH'))
    exact = zz_p1(edges, x['gamma'], x['beta'])
    cb = [bits_from_counts(sim.run(x['circ'], shots=SH, seed_simulator=100 + t).result().get_counts(), NT) for t in range(COPIES)]
    res = three_readings(cb, ptab_v, rec_v, edges, exact, NT)
    print(report(res, x['ideal'] / x['maxcut'], '[SIMULATION] '))
    print('[SIMULATION] truth is 1.0000 (ideal circuit, readout noise only). Hardware adds gate noise, which lowers all three alike.')
    print('est QPU seconds, real run: %.1f (P %.1f + main %.1f)' % (est_total(), est_seconds(2 * P_SHOTS), est_seconds(COPIES * SHOTS)))


def mode_dry():
    svc = get_service()
    b = svc.backend(BACKEND)
    rem = usage_remaining(svc)
    print('live backend ok: %s, %d qubits; usage_remaining_seconds = %.0f' % (BACKEND, b.num_qubits, rem))
    t0 = time.time()
    x = prepare(b, N_NODES)
    n2, _ = twoq_info(x['circ'])
    print('graph: %d nodes %d edges; p=1 angles gamma %.6f beta %.6f; noiseless expected cut %.3f; exact max cut %d (noiseless QAOA = %.3f of max)'
          % (N_NODES, len(x['edges']), x['gamma'], x['beta'], x['ideal'], x['maxcut'], x['ideal'] / x['maxcut']))
    print("client's circuit (level 3, layout from the record): %d two-qubit gates, depth %d; built in %.0f s" % (n2, x['circ'].depth(), time.time() - t0))
    print('layout (virtual node -> physical qubit): %s' % x['layout'])
    p, m = est_seconds(2 * P_SHOTS), est_seconds(COPIES * SHOTS)
    tot = p + m
    print('job P: 2 circuits x %d shots, est %.1f s;  main: %d copies x %d shots, est %.1f s;  TOTAL est %.1f s (x%.2f = %.1f s; limit %.0f; remaining %.0f)'
          % (P_SHOTS, p, COPIES, SHOTS, m, tot, EST_MARGIN, tot * EST_MARGIN, MAX_TOTAL_S, rem))
    print('no-fire reasons: %s' % (refusals(tot, rem) or 'none'))


def mode_fire():
    svc = get_service()
    b = svc.backend(BACKEND)
    rem = usage_remaining(svc)
    tot = est_total()
    print('estimate %.1f s (limit %.0f), remaining %.0f s' % (tot, MAX_TOTAL_S, rem))
    why = refusals(tot, rem)
    if why:
        sys.exit('REFUSED: ' + '; '.join(why))
    if os.path.exists(fp('state.json')) and json.load(open(fp('state.json'))).get('job_main'):
        sys.exit('REFUSED: run/state.json already holds a fired main job; move run/ aside to fire again')
    x = prepare(b, N_NODES)
    from qiskit import qpy
    with open(fp('circuit.qpy'), 'wb') as f:
        qpy.dump([x['circ']], f)
    json.dump(props_of(b), open(fp('props_fire.json'), 'w'))
    st = dict(chip=BACKEND, layout=x['layout'], edges=[list(e) for e in x['edges']], gamma=x['gamma'], beta=x['beta'], ideal=x['ideal'],
              maxcut=x['maxcut'], copies=COPIES, shots=SHOTS, p_shots=P_SHOTS, est_s=tot, fired_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(st, open(fp('state.json'), 'w'), indent=1)
    jp = sampler(b, tutorial=False).run([(c, None, P_SHOTS) for c in p_circuits(b.num_qubits, x['layout'])])
    st['job_P'] = jp.job_id()
    json.dump(st, open(fp('state.json'), 'w'), indent=1)
    print('job P submitted')
    jm = sampler(b, tutorial=True).run([(x['circ'], None, SHOTS)] * COPIES)
    st['job_main'] = jm.job_id()
    json.dump(st, open(fp('state.json'), 'w'), indent=1)
    print('main job submitted (%d copies of the client circuit, tutorial options). P and main are independent. Next: --fetch when both are DONE, then --analyze.' % COPIES)


def mode_fetch():
    st = json.load(open(fp('state.json')))
    svc = get_service()
    for key, raw in (('job_P', 'raw_P.npz'), ('job_main', 'raw_main.npz')):
        j = svc.job(st[key])
        print(key, j.status())
        if 'DONE' not in str(j.status()):
            return print('not done: run --fetch again later')
        save_raw(fp(raw), j.result())
        st[key + '_usage'] = (j.metrics() or {}).get('usage')
    json.dump(st, open(fp('state.json'), 'w'), indent=1)
    print('saved raw_P.npz and raw_main.npz; usage: P %s, main %s' % (st.get('job_P_usage'), st.get('job_main_usage')))


def mode_analyze():
    st = json.load(open(fp('state.json')))
    n, lay = N_NODES, st['layout']
    edges = [tuple(e) for e in st['edges']]
    z = np.load(fp('raw_P.npz'))
    ptab_v = table_from_bits(unpack(z['c_0'], n), unpack(z['c_1'], n))
    rec_v = virt(record_readout(json.load(open(fp('props_fire.json')))), lay)
    fl = readout_check(rec_v, ptab_v)
    lines = ['readout check (today worse than the record): %d of %d layout qubits' % (len(fl), n)]
    for i in sorted(fl):
        for par, (s, m, lo, hi) in fl[i].items():
            lines.append('  q%d (node %d) %s: record %.4f, today %.4f (99%% interval %.4f to %.4f)' % (lay[i], i, par, s, m, lo, hi))
    zm = np.load(fp('raw_main.npz'))
    cb = [unpack(zm['meas_%d' % t], n) for t in range(st['copies'])]
    res = three_readings(cb, ptab_v, rec_v, edges, zz_p1(edges, st['gamma'], st['beta']), n)
    out = '\n'.join(lines) + '\n' + report(res, st['ideal'] / st['maxcut'])
    print(out)
    open(fp('results.text'), 'w').write(out + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for m in ('test', 'dry', 'fire', 'fetch', 'analyze'):
        ap.add_argument('--' + m, action='store_true')
    a = ap.parse_args()
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.fire: mode_fire()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
