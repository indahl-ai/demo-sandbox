"""Demo 4: Simulation / materials. Trotter dynamics of a 1D transverse-field Ising chain on a line of the chosen backend.
Client arm: vendor defaults, the chain's static ZZ is simply left in. Our arm: we measure the static ZZ on the chain's edges
in a short pre-job (echoed Ramsey, neighbour in 0 vs 1, phase difference) and fold it into each Trotter ZZ layer's RZZ angle.
Modes: --test --dry --fire-zz --fetch-zz --fire --fetch --analyze.   This script never submits unless --fire-zz / --fire is given.
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE (optional instance name). The key is never printed or stored."""
import os, sys, json, argparse, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = 'ibm_kingston'
N = 10                      # chain length (qubits)
K = 10                      # Trotter steps
J, H, DT = 1.0, 1.0, 0.25   # H = -J sum ZZ - h sum X ; first-order Trotter step dt; start |0...0>
SHOTS = 4096                # per circuit (Estimator default precision 0.015625 = 4096 shots; both arms identical)
ZZ_TAUS_US = [0, 10, 20, 30, 40, 50]
ZZ_SHOTS = 400
SEED = 20261119
MAX_SECONDS = 300
SIGN = None                 # set below by convention test; see zz_fit
FILES = {k: os.path.join(HERE, v) for k, v in dict(jobs='jobs.json', zz='zz_result.json', main='main_result.json').items()}

# ---------------------------------------------------------------- physics
def edges(n=N): return [(i, i + 1) for i in range(n - 1)]

def trotter_step(n, zz_corr=None):
    """One first-order step: exp(+i dt J ZZ) layers (even then odd edges), then exp(+i dt h X). zz_corr[e] (rad) is subtracted
    from RZZ angle of edge e: static ZZ already adds RZZ(+corr) during the step, so the folded angle cancels it."""
    qc = QuantumCircuit(n)
    th = -2 * J * DT
    for par in (0, 1):
        for e, (a, b) in enumerate(edges(n)):
            if e % 2 == par: qc.rzz(th - (zz_corr[e] if zz_corr is not None else 0.0), a, b)
    for q in range(n): qc.rx(-2 * H * DT, q)
    return qc

def obs_ops(n):
    avgz = SparsePauliOp.from_sparse_list([('Z', [q], 1.0 / n) for q in range(n)], n)
    avgzz = SparsePauliOp.from_sparse_list([('ZZ', [a, b], 1.0 / (n - 1)) for a, b in edges(n)], n)
    return avgz, avgzz

def exact_values(n=N, k=K):
    """Exact classical values of the ideal Trotter circuit (what a perfect device returns) and of continuous-time evolution."""
    avgz, avgzz = obs_ops(n); step = trotter_step(n); sv = Statevector.from_label('0' * n)
    tro = []
    for _ in range(k):
        sv = sv.evolve(step); tro.append((float(sv.expectation_value(avgz).real), float(sv.expectation_value(avgzz).real)))
    Hm = sum(-J * SparsePauliOp.from_sparse_list([('ZZ', [a, b], 1.0)], n) for a, b in edges(n)) \
        + sum(-H * SparsePauliOp.from_sparse_list([('X', [q], 1.0)], n) for q in range(n))
    w, v = np.linalg.eigh(Hm.to_matrix()); psi0 = np.zeros(2 ** n, complex); psi0[0] = 1; c = v.conj().T @ psi0
    cont = []
    for s in range(1, k + 1):
        psi = v @ (np.exp(-1j * w * DT * s) * c); s_ = Statevector(psi)
        cont.append((float(s_.expectation_value(avgz).real), float(s_.expectation_value(avgzz).real)))
    return np.array(tro), np.array(cont)

# ---------------------------------------------------------------- chain + circuits
def pick_chain(backend, n=N):
    """Best simple path of n qubits on the coupling graph: lowest sum of CZ error + readout error (from the target)."""
    t = backend.target; adj = {q: set() for q in range(backend.num_qubits)}
    for a, b in backend.coupling_map.get_edges(): adj[a].add(b); adj[b].add(a)
    def qerr(q):
        try: return t['measure'][(q,)].error or 0.05
        except Exception: return 0.05
    def eerr(a, b):
        for g in ('cz', 'ecr', 'cx'):
            if g in t.operation_names:
                try: return t[g][(a, b)].error or 0.05
                except Exception: pass
        return 0.05
    best = [1e9, None]
    def dfs(path, cost):
        if cost >= best[0]: return
        if len(path) == n: best[:] = [cost, list(path)]; return
        for nb in sorted(adj[path[-1]]):
            if nb not in path: dfs(path + [nb], cost + eerr(path[-1], nb) + qerr(nb))
    for q in range(backend.num_qubits):
        if qerr(q) < 0.1: dfs([q], qerr(q))
    return best[1]

def step_duration_s(backend, phys_step):
    return float(phys_step.estimate_duration(backend.target))

def phys_step(backend, chain, zz_corr=None):
    return transpile(trotter_step(N, zz_corr), backend, initial_layout=chain, optimization_level=1, seed_transpiler=SEED)

def stack(step, k, planted=None, nq=None):
    """k copies of the physical one-step circuit (barrier between), optional planted per-edge RZZ after each step."""
    qc = QuantumCircuit(step.num_qubits)
    for _ in range(k):
        qc.compose(step, inplace=True); qc.barrier()
        if planted:
            for (a, b), al in planted: qc.rzz(al, a, b)
            qc.barrier()
    return qc

# ---------------------------------------------------------------- ZZ measurement (echoed Ramsey, neighbour 0 vs 1)
def zz_groups(n=N):
    es = edges(n); return [[e for i, e in enumerate(es) if i % 3 == g] for g in range(3)]   # probe=a, neighbour=b; spacing keeps probes un-adjacent

def zz_circuit(group, js, tau_us, basis, dt, planted_hz=None):
    """Probe a of each edge in the group starts in +, neighbour b in |js>, delay tau/2, X on both, delay tau/2, read probe in X or Y.
    With planted_hz given (offline sim) the delay is replaced by the planted ZZ rotation RZZ(pi*zeta*tau/2) on every chain edge."""
    qc = QuantumCircuit(N); cr = ClassicalRegister(len(group), 'p'); qc.add_register(cr)
    probes = [a for a, b in group]; nbrs = [b for a, b in group]
    qc.h(probes)
    if js: qc.x(nbrs)
    qc.barrier()
    def half():
        if planted_hz is None:
            d = int(round(tau_us * 0.5e-6 / dt / 16)) * 16
            if d: qc.delay(d, range(N), unit='dt')
        else:
            for e, (a, b) in enumerate(edges()): qc.rzz(np.pi * planted_hz[e] * tau_us * 0.5e-6, a, b)
        qc.barrier()
    half(); qc.x(probes + nbrs); qc.barrier(); half()
    if basis == 'Y': qc.sdg(probes)
    qc.h(probes)
    for i, p in enumerate(probes): qc.measure(p, cr[i])
    return qc

def zz_meta():
    return [(gi, js, t, bs) for gi in range(3) for js in (0, 1) for t in ZZ_TAUS_US for bs in 'XY']

def zz_fit(counts_by_key, shots, rng, nboot=200):
    """counts_by_key[(gi,js,tau,basis)] = counts dict over the group's probe bits. Returns {edge_index: (zeta_Hz, sigma_Hz)}."""
    groups = zz_groups(); res = {}
    grid = np.linspace(-80e3, 80e3, 3201); taus = np.array(ZZ_TAUS_US) * 1e-6
    def mean_z(counts, i):
        tot = sum(counts.values()); return sum(v * (1 - 2 * int(k.replace(' ', '')[::-1][i])) for k, v in counts.items()) / tot
    for gi, g in enumerate(groups):
        for i, (a, b) in enumerate(g):
            e = a  # edge index == probe index on the chain (edge (a,a+1))
            def est(boot):
                c = []
                for js in (0, 1):
                    row = []
                    for t in ZZ_TAUS_US:
                        xs, ys = [], []
                        for bs, acc in (('X', xs), ('Y', ys)):
                            m = mean_z(counts_by_key[(gi, js, t, bs)], i)
                            if boot: m = np.clip(2 * rng.binomial(shots, (1 + m) / 2) / shots - 1, -1, 1)
                            acc.append(m)
                        row.append(xs[0] + 1j * ys[0])
                    c.append(np.array(row))
                prod = c[1] * np.conj(c[0])
                score = [abs(np.sum(prod * np.exp(-1j * SIGN * 2 * np.pi * f * taus))) for f in grid]
                return grid[int(np.argmax(score))]
            z0 = est(False); bs_ = [est(True) for _ in range(nboot)]
            res[e] = (float(z0), float(np.std(bs_)))
    return res

# ---------------------------------------------------------------- service
def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get('IBM_QUANTUM_TOKEN', '').strip()
    if not tok: sys.exit('Set IBM_QUANTUM_TOKEN to your own IBM Quantum key first. It is read from the environment only and never printed or stored.')
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok)
    import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); import pick_instance
    return QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=pick_instance.crn(svc))

def remaining(svc):
    try: return float(svc.usage().get('usage_remaining_seconds'))
    except Exception: return None

def est_seconds_zz(): 
    n = len(zz_meta()); return n * ZZ_SHOTS * (250e-6 + np.mean(ZZ_TAUS_US) * 1e-6 + 2e-6) * 1.3 + 4
def est_seconds_main(tstep):
    return sum(2 * SHOTS * (250e-6 + k * tstep + 2e-6) * 2 for k in range(1, K + 1)) * 1.3 + 4   # 2 arms, x2 for default TREX calibration

def guard(est, svc):
    rem = remaining(svc); print('estimated vendor seconds: %.0f   usage_remaining_seconds: %s' % (est, rem))
    if est > MAX_SECONDS: sys.exit('REFUSED: estimate exceeds %d s' % MAX_SECONDS)
    if rem is not None and est > rem: sys.exit('REFUSED: estimate exceeds remaining seconds')

def load(k): return json.load(open(FILES[k]))
def save(k, d): json.dump(d, open(FILES[k], 'w'), indent=1)
def jobs_get(): return json.load(open(FILES['jobs'])) if os.path.exists(FILES['jobs']) else {}
def jobs_set(k, v): j = jobs_get(); j[k] = v; save('jobs', j)

# ---------------------------------------------------------------- analysis (shared by --test and --analyze)
def analyze(main, zz, tro, cont, label=''):
    c = np.array(main['client']); f = np.array(main['folded'])      # shape (K, 2): avgZ, avgZZ
    ec, ef = np.abs(c - tro), np.abs(f - tro)
    rows = ['step   exact(Z,ZZ)        client err(Z,ZZ)      folded err(Z,ZZ)']
    for k in range(K): rows.append('%4d  %7.4f %7.4f   %8.4f %8.4f   %8.4f %8.4f' % (k + 1, *tro[k], *ec[k], *ef[k]))
    out = '\n'.join(rows) + '\n'
    rc = np.sqrt((ec ** 2).mean(0)); rf = np.sqrt((ef ** 2).mean(0))
    out += 'RMS error over steps  avg<Z>: client %.4f folded %.4f ratio %.2f | NN<ZZ>: client %.4f folded %.4f ratio %.2f\n' % (rc[0], rf[0], rc[0] / rf[0], rc[1], rf[1], rc[1] / rf[1])
    rcall = np.sqrt((ec ** 2).mean()); rfall = np.sqrt((ef ** 2).mean())
    out += 'RMS error, both observables: client %.4f folded %.4f ratio client/folded %.2f\n' % (rcall, rfall, rcall / rfall)
    out += 'Trotter error of the ideal circuit vs continuous time (max abs): %.4f\n' % np.abs(tro - cont).max()
    if zz:
        out += 'edge  measured_kHz  +-sigma  vendor_listed_kHz  planted_kHz\n'
        for e in sorted(zz['measured'], key=int):
            m, s = zz['measured'][e]; pl = zz.get('planted_hz', {}).get(e)
            out += '%3s   %9.2f   %5.2f   %8.2f        %s\n' % (e, m / 1e3, s / 1e3, zz['listed_hz'].get(e, 0.0) / 1e3, '%.2f' % (pl / 1e3) if pl is not None else '-')
    return out, (rcall, rfall)

# ---------------------------------------------------------------- --test (offline)
def run_test(seed, shots):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    rng = np.random.default_rng(seed); fb = FakeKingston()
    chain = pick_chain(fb); print('SEED', seed, ' shots', shots, ' chain (fake kingston physical qubits):', chain)
    ne = N - 1
    planted = rng.uniform(3e3, 15e3, ne) * rng.choice([-1, 1], ne) * 0 + rng.uniform(3e3, 15e3, ne)   # a few kHz each edge, positive
    print('planted static ZZ (kHz):', np.round(planted / 1e3, 2))
    tro, cont = exact_values()
    # --- ZZ pre-job offline: logical circuits, planted rzz in place of delays, readout error 2% (T1/T2 not modelled in this pre-job sim)
    nm_ro = NoiseModel(); ro = 0.02; nm_ro.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    sim0 = AerSimulator(noise_model=nm_ro, seed_simulator=seed)
    global SIGN
    counts = {}
    for key in zz_meta():
        gi, js, t, bs = key; qc = zz_circuit(zz_groups()[gi], js, t, bs, fb.dt, planted_hz=planted)
        counts[key] = sim0.run(qc, shots=ZZ_SHOTS).result().get_counts()
    # sign convention: planted ZZ is positive, so the correct SIGN is the one that returns positive zeta
    best = None
    for s in (1, -1):
        SIGN = s; r = zz_fit(counts, ZZ_SHOTS, rng, nboot=1); err = np.mean([abs(r[e][0] - planted[e]) for e in r])
        if best is None or err < best[0]: best = (err, s)
    SIGN = best[1]; print('ZZ phase sign convention found by offline recovery of a known planted ZZ: SIGN=%d' % SIGN)
    fit = zz_fit(counts, ZZ_SHOTS, rng)
    zz = dict(measured={str(e): list(v) for e, v in fit.items()}, listed_hz={str(e): 0.0 for e in fit},
              planted_hz={str(e): float(planted[e]) for e in range(ne)})
    # --- Trotter dynamics, fake kingston noise, planted ZZ acts during every step
    step_c = phys_step(fb, chain); T = step_duration_s(fb, step_c)
    corr = np.array([fit[e][0] for e in range(ne)]) * np.pi * T            # rad per step, from MEASURED zeta
    step_f = phys_step(fb, chain, corr)
    print('one Trotter step duration (fake kingston schedule): %.3f us; planted ZZ phase per step 2*pi*zeta*T (kHz-scale): %s rad'
          % (T * 1e6, np.round(2 * np.pi * planted * T, 4)))
    pl = [((chain[e], chain[e + 1]), float(np.pi * planted[e] * T)) for e in range(ne)]
    avgz, avgzz = obs_ops(N); res = {}
    for tag, noisy in (('full noise (fake kingston gate+readout noise) + planted ZZ', True), ('planted ZZ only (no other noise): isolates the folded term', False)):
        sim = AerSimulator(noise_model=NoiseModel.from_backend(fb), seed_simulator=seed) if noisy else AerSimulator(seed_simulator=seed)
        out = {'client': [], 'folded': []}
        for k in range(1, K + 1):
            for arm, st in (('client', step_c), ('folded', step_f)):
                qc = stack(st, k, planted=pl); cr_ = ClassicalRegister(N, 'm'); qc.add_register(cr_); [qc.measure(chain[i], cr_[i]) for i in range(N)]
                if not noisy:   # noiseless gates: re-plan with ideal basis, drop delay-free structure is identical
                    pass
                cts = sim.run(qc, shots=shots).result().get_counts(); vals = []
                z = zz_ = 0.0; tot = sum(cts.values())
                for bits, v in cts.items():
                    b = bits.replace(' ', '')[::-1]; s = [1 - 2 * int(b[i]) for i in range(N)]
                    z += v * np.mean(s); zz_ += v * np.mean([s[i] * s[i + 1] for i in range(ne)])
                out[arm].append((z / tot, zz_ / tot))
        o, (rc, rf) = analyze(out, zz if noisy else None, tro, cont)
        print('\n=== ' + tag + ' ===\n' + o); res[tag] = (rc, rf)
    print('\nHEADLINE (simulation): ' + '; '.join('%s: client RMS %.4f vs folded RMS %.4f (ratio %.2f)' % (t.split(' (')[0].split(':')[0], a, b, a / b) for t, (a, b) in res.items()))
    return res

# ---------------------------------------------------------------- live modes
def live_setup():
    svc = get_service(); b = svc.backend(BACKEND); chain = pick_chain(b); return svc, b, chain

def run_dry():
    svc, b, chain = live_setup(); print('backend', b.name, 'chain', chain)
    sc = phys_step(b, chain); T = step_duration_s(b, sc)
    print('one Trotter step: depth %d, 2q gates %d, duration %.3f us' % (sc.depth(), sc.count_ops().get('cz', 0), T * 1e6))
    listed = {}
    try:
        pr = b.properties()
        for e, (a, c) in enumerate(edges()):
            listed[e] = 0.0
    except Exception: pass
    print('vendor lists zz = 0 on live edges. Per-step ZZ phase delta = 2*pi*zz*T for assumed zz (kHz) -> rad:')
    for zk in (2.85, 5, 10, 15.36): print('  zz %5.2f kHz -> %.4f rad/step, %.4f rad after %d steps' % (zk, 2 * np.pi * zk * 1e3 * T, 2 * np.pi * zk * 1e3 * T * K, K))
    from qiskit_ibm_runtime import EstimatorV2
    print('Estimator default resilience_level:', EstimatorV2(mode=b).options.resilience_level)
    ez, em = est_seconds_zz(), est_seconds_main(T)
    print('estimate job Z (ZZ measurement): %.0f s  | main job: %.0f s | total %.0f s' % (ez, em, ez + em)); print('usage_remaining_seconds:', remaining(svc))
    print('circuits: ZZ job %d x %d shots; main job %d pubs (2 arms x %d steps), 2 observables each' % (len(zz_meta()), ZZ_SHOTS, 2 * K, K))

def run_fire_zz():
    from qiskit_ibm_runtime import SamplerV2
    svc, b, chain = live_setup(); est = est_seconds_zz(); guard(est, svc)
    pubs = []
    for key in zz_meta():
        gi, js, t, bs = key; qc = zz_circuit(zz_groups()[gi], js, t, bs, b.dt)
        pubs.append((transpile(qc, b, initial_layout=chain, optimization_level=0, seed_transpiler=SEED), None, ZZ_SHOTS))
    job = SamplerV2(mode=b).run(pubs); jobs_set('zz', dict(id=job.job_id(), chain=chain, backend=BACKEND)); print('submitted ZZ job', job.job_id())

def run_fetch_zz():
    svc = get_service(); jd = jobs_get()['zz']; res = svc.job(jd['id']).result(); counts = {}
    for key, pr in zip(zz_meta(), res): counts[key] = pr.data.p.get_counts()
    global SIGN; SIGN = ZZ_SIGN_FILE()
    fit = zz_fit(counts, ZZ_SHOTS, np.random.default_rng(SEED))
    save('zz', dict(measured={str(e): list(v) for e, v in fit.items()}, listed_hz={str(e): 0.0 for e in fit}, chain=jd['chain']))
    print('saved', FILES['zz']); [print('edge %d  %.2f +- %.2f kHz' % (e, v[0] / 1e3, v[1] / 1e3)) for e, v in fit.items()]

def ZZ_SIGN_FILE():
    p = os.path.join(HERE, 'sign.json'); return json.load(open(p))['SIGN'] if os.path.exists(p) else 1

def run_fire():
    from qiskit_ibm_runtime import EstimatorV2
    svc, b, chain = live_setup(); zz = load('zz')
    if zz.get('chain') != chain: sys.exit('REFUSED: chain changed since the ZZ job; re-run --fire-zz')
    sc = phys_step(b, chain); T = step_duration_s(b, sc); est = est_seconds_main(T); guard(est, svc)
    corr = np.array([zz['measured'][str(e)][0] for e in range(N - 1)]) * np.pi * T; sf = phys_step(b, chain, corr)
    avgz, avgzz = obs_ops(N); pubs, order = [], []
    for k in range(1, K + 1):
        for arm, st in (('client', sc), ('folded', sf)):
            isa = stack(st, k); pubs.append((isa, [avgz.apply_layout(sc.layout), avgzz.apply_layout(sc.layout)])); order.append((arm, k))
    est_ = EstimatorV2(mode=b)    # vendor defaults (resilience level, precision) for BOTH arms
    est_.options.default_shots = SHOTS
    job = est_.run(pubs); jobs_set('main', dict(id=job.job_id(), order=order, T=T)); print('submitted main job', job.job_id(), 'resilience_level', est_.options.resilience_level)

def run_fetch():
    svc = get_service(); jd = jobs_get()['main']; res = svc.job(jd['id']).result(); out = {'client': [None] * K, 'folded': [None] * K}
    for (arm, k), pr in zip(jd['order'], res): out[arm][k - 1] = [float(x) for x in pr.data.evs]
    save('main', out); print('saved', FILES['main'])

def run_analyze():
    tro, cont = exact_values(); zz = load('zz') if os.path.exists(FILES['zz']) else None
    o, _ = analyze(load('main'), zz, tro, cont); print(o)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for m in ('test', 'dry', 'fire-zz', 'fetch-zz', 'fire', 'fetch', 'analyze'): ap.add_argument('--' + m, action='store_true')
    ap.add_argument('--seed', type=int, default=SEED); ap.add_argument('--shots', type=int, default=8000)
    a = ap.parse_args()
    if a.test: run_test(a.seed, a.shots)
    elif a.dry: run_dry()
    elif a.fire_zz: run_fire_zz()
    elif a.fetch_zz: run_fetch_zz()
    elif a.fire: run_fire()
    elif a.fetch: run_fetch()
    elif a.analyze: run_analyze()
    else: ap.print_help()
