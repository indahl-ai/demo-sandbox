"""Demo 4 v2: Trotter dynamics of a 1D transverse-field Ising chain on a line of the live chip, four arms in ONE job.
ONE transpile (parametric RZZ angles); the arms differ ONLY in the bound RZZ angles; the script prints each arm's gate counts and
refuses to fire if any differ.
  CLIENT   = no fold
  FREE FIX = fold the LISTED static ZZ (zz_<a><b> in the live record, GHz)
  OURS     = fold our MEASURED static ZZ (fresh echoed-Ramsey ZZ job), over (T_step - CZ time on that edge)
  SIGN CTL = fold MINUS our measured ZZ (must lose)
Every run uses resilience_level 0, twirling OFF, dynamical decoupling OFF, transpile optimization_level 0; RZZ angles are asserted
exact in the transpiled circuit. Every parameter of our arm comes from a hardware job or the live record (SOURCES, checked before fire).
Modes: --test (code smoke test only, offline, no result) --dry --fire-zz --fetch-zz --fire --fetch --analyze.
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE (optional instance name). The key is never printed or stored."""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = 'ibm_kingston'
N, K, COPIES = 10, 20, 4          # chain qubits, Trotter steps, interleaved copies
J, H, DT = 1.0, 1.0, 0.25         # H = -J sum ZZ - h sum X ; start |0...0>
SHOTS = 1024                      # per pub per copy (4 arms x K steps x COPIES pubs)
ZZ_TAUS_US = [0, 10, 20, 30, 40, 50]; ZZ_SHOTS = 400
SEED = 20261119; MAX_SECONDS = 200
ZREF = 10e3                       # reference rate (Hz) for the on-chip sign control (explicit RZZ gates of known angle)
GRID = np.linspace(-80e3, 80e3, 3201)   # fit grid in Hz: allows negative values
assert GRID.min() < 0 < GRID.max()
ARMS = ['client', 'free', 'ours', 'signctl']
LABEL = {'client': 'CLIENT', 'free': 'FREE FIX', 'ours': 'OURS', 'signctl': 'SIGN CTL'}
T99 = {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250}
FILES = {k: os.path.join(HERE, v) for k, v in dict(jobs='jobs_v2.json', zz='zz_v2.json', main='main_v2.json').items()}

# ------------------------------------------------------------ physics
def edges(n=N): return [(i, i + 1) for i in range(n - 1)]
def obs_ops(n=N):
    return (SparsePauliOp.from_sparse_list([('Z', [q], 1.0 / n) for q in range(n)], n),
            SparsePauliOp.from_sparse_list([('ZZ', [a, b], 1.0 / (n - 1)) for a, b in edges(n)], n))

def step_param(n=N):
    pv = ParameterVector('th', n - 1); qc = QuantumCircuit(n)
    for par in (0, 1):
        for e, (a, b) in enumerate(edges(n)):
            if e % 2 == par: qc.rzz(pv[e], a, b)
    for q in range(n): qc.rx(-2 * H * DT, q)
    return qc, pv

def exact_values(n=N, k=K):
    avgz, avgzz = obs_ops(n); qc, pv = step_param(n)
    step = qc.assign_parameters({p: -2 * J * DT for p in pv}); sv = Statevector.from_label('0' * n); tro = []
    for _ in range(k):
        sv = sv.evolve(step); tro.append((float(sv.expectation_value(avgz).real), float(sv.expectation_value(avgzz).real)))
    return np.array(tro)

# ------------------------------------------------------------ chain, ONE transpile, durations
def pick_chain(backend, n=N):
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

def transpile_once(backend, chain):
    qc, pv = step_param(N)
    st = transpile(qc, backend, initial_layout=chain, optimization_level=0, seed_transpiler=SEED)
    return st, pv

def edge_times(backend, chain, st):
    """T_step (s), per-edge CZ count and CZ time, net time T_net[e] = T_step - n_cz_e * t_cz_e (the vendor's CZ calibration already
    absorbs the static ZZ during its own gate)."""
    T = float(st.estimate_duration(backend.target)); tg = backend.target; cnt = {}
    for ins in st.data:
        if ins.operation.name == 'cz':
            key = frozenset(st.find_bit(q).index for q in ins.qubits); cnt[key] = cnt.get(key, 0) + 1
    def tcz(a, b):
        for g in ('cz',):
            for pr in ((a, b), (b, a)):
                try: return float(tg[g][pr].duration)
                except Exception: pass
        return 0.0
    ncz, tc, net = [], [], []
    for e in range(N - 1):
        a, b = chain[e], chain[e + 1]; n_ = cnt.get(frozenset((a, b)), 0); t_ = tcz(a, b)
        ncz.append(n_); tc.append(t_); net.append(T - n_ * t_)
    return T, ncz, tc, np.array(net)

def listed_zz(backend, chain):
    """Listed static ZZ (Hz) per chain edge from the live record: general fields zz_<a><b>, unit GHz. None if absent."""
    try: gen = {x.name: x for x in backend.properties().general}
    except Exception: return None
    out = {}
    for e in range(N - 1):
        a, b = chain[e], chain[e + 1]; f = gen.get('zz_%d%d' % (a, b)) or gen.get('zz_%d%d' % (b, a))
        if f is None: return None
        out[e] = float(f.value) * (1e9 if str(f.unit) == 'GHz' else 1.0)
    return out

def arm_angles(arm, meas_hz, listed_hz, T_net):
    """RZZ angle per edge for an arm. Static ZZ adds RZZ(+pi*zeta*T_net) during a step; the fold subtracts it."""
    z = {'client': np.zeros(N - 1), 'free': np.array(listed_hz), 'ours': np.array(meas_hz), 'signctl': -np.array(meas_hz)}[arm]
    return -2 * J * DT - np.pi * z * T_net

def bound_step(st, pv, angles): return st.assign_parameters({pv[e]: float(angles[e]) for e in range(N - 1)})

def param_positions(st):
    """Instruction index of each edge's RZZ angle in the transpiled step; refuse if the transpiler changed/merged/duplicated any."""
    pos = {}
    for i, ins in enumerate(st.data):
        for p in ins.operation.params:
            if hasattr(p, 'parameters') and p.parameters:
                names = [x.name for x in p.parameters]
                if len(names) != 1 or str(p) != names[0]: sys.exit('REFUSED: transpiler altered a parameterized RZZ angle: %s' % p)
                pos.setdefault(int(names[0].split('[')[1][:-1]), []).append(i)
    if sorted(pos) != list(range(N - 1)) or any(len(v) != 1 for v in pos.values()): sys.exit('REFUSED: transpiler merged/duplicated/dropped an RZZ angle')
    return {e: v[0] for e, v in pos.items()}

def verify_angles(st, bound, angles, k, stacked):
    pos = param_positions(st); L = len(st.data) + 1
    for e, i in pos.items():
        for s_ in range(k):
            got = float(stacked.data[s_ * L + i].operation.params[0])
            if got != float(angles[e]): sys.exit('REFUSED: RZZ angle on edge %d step %d is %r, set %r' % (e, s_ + 1, got, float(angles[e])))
        if float(bound.data[i].operation.params[0]) != float(angles[e]): sys.exit('REFUSED: bound step angle differs on edge %d' % e)

def stack(step, k, planted=None):
    qc = QuantumCircuit(step.num_qubits)
    for _ in range(k):
        qc.compose(step, inplace=True); qc.barrier()
        if planted:
            for (a, b), al in planted: qc.rzz(al, a, b)
            qc.barrier()
    return qc

def gate_check(st, pv, meas, lst, Tn, k=K):
    """Refuse (exit) if any arm's gate counts differ from the CLIENT arm's."""
    rows = {}
    for arm in ARMS:
        ang = arm_angles(arm, meas, lst, Tn); bs = bound_step(st, pv, ang); c = stack(bs, k); verify_angles(st, bs, ang, k, c)
        rows[arm] = (dict(c.count_ops()), c.depth())
        print('  %-8s gate counts %s  depth %d' % (LABEL[arm], rows[arm][0], rows[arm][1]))
    if any(rows[a] != rows['client'] for a in ARMS): sys.exit('REFUSED: arms differ in gate counts/depth')
    print('  gate counts identical across all four arms; every RZZ angle in the transpiled circuit equals the angle set (exact)')
    return rows

# ------------------------------------------------------------ ZZ measurement (echoed Ramsey, neighbour 0 vs 1)
def zz_groups(n=N):
    es = edges(n); return [[e for i, e in enumerate(es) if i % 3 == g] for g in range(3)]

def zz_circuit(group, js, tau_us, basis, dt, planted_hz=None):
    """Probe a (in +) per edge, neighbour b in |js>; delay tau/2, X on both, delay tau/2, read probe in X or Y. Offline: each half is
    replaced by RZZ(pi*zeta*tau/2) on every edge; the two halves give RZZ(pi*zeta*tau) = exp(-i*2*pi*zeta/4*tau*ZZ), which makes the
    probe's phase differ by 2*pi*zeta*tau between neighbour 0 and 1 (checked: planted values are recovered at both signs)."""
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

def zz_control(group, js, tau_us, basis):
    """On-chip sign control: same echoed sequence, but each half is an explicit RZZ gate of known angle pi*ZREF*tau/2 on the group's edges."""
    qc = QuantumCircuit(N); cr = ClassicalRegister(len(group), 'p'); qc.add_register(cr)
    probes = [a for a, b in group]; nbrs = [b for a, b in group]; al = np.pi * ZREF * tau_us * 0.5e-6
    qc.h(probes)
    if js: qc.x(nbrs)
    qc.barrier()
    for _ in range(2):
        for a, b in group: qc.rzz(al, a, b)
        qc.barrier()
        if _ == 0: qc.x(probes + nbrs); qc.barrier()
    if basis == 'Y': qc.sdg(probes)
    qc.h(probes)
    for i, p in enumerate(probes): qc.measure(p, cr[i])
    return qc

def zz_meta():
    return [(kind, gi, js, t, bs) for kind in ('zz', 'ctl') for gi in range(3) for js in (0, 1) for t in ZZ_TAUS_US for bs in 'XY']

def zz_build(key, dt, planted_hz=None):
    kind, gi, js, t, bs = key
    return zz_control(zz_groups()[gi], js, t, bs) if kind == 'ctl' else zz_circuit(zz_groups()[gi], js, t, bs, dt, planted_hz=planted_hz)

def determine_sign(ctl_fit):
    """Phase-sign convention FROM THE CHIP: fit the explicit-RZZ control with SIGN=+1; the chip's own known-angle gates say which sign is right."""
    v = np.array([ctl_fit[e][0] for e in sorted(ctl_fit)])
    if not (np.all(v > 0.5 * ZREF) or np.all(v < -0.5 * ZREF)): return None
    return 1 if v[0] > 0 else -1

def zz_fit(counts, shots, rng, sign=1, nboot=200):
    taus = np.array(ZZ_TAUS_US) * 1e-6; res = {}
    def mz(cts, i):
        tot = sum(cts.values()); return sum(v * (1 - 2 * int(k.replace(' ', '')[::-1][i])) for k, v in cts.items()) / tot
    for gi, g in enumerate(zz_groups()):
        for i, (a, b) in enumerate(g):
            def est(boot):
                c = []
                for js in (0, 1):
                    row = []
                    for t in ZZ_TAUS_US:
                        v = []
                        for bs in 'XY':
                            m = mz(counts[(gi, js, t, bs)], i)
                            if boot: m = np.clip(2 * rng.binomial(shots, (1 + m) / 2) / shots - 1, -1, 1)
                            v.append(m)
                        row.append(v[0] + 1j * v[1])
                    c.append(np.array(row))
                prod = c[1] * np.conj(c[0])
                return GRID[int(np.argmax([abs(np.sum(prod * np.exp(-1j * sign * 2 * np.pi * f * taus))) for f in GRID]))]
            z0 = est(False); bs_ = [est(True) for _ in range(nboot)]; res[a] = (float(z0), float(np.std(bs_)))
    return res

# ------------------------------------------------------------ service / io
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

# rates calibrated on the 14-18 Nov run: ZZ job est 14 s took 11 s; main job 20 pubs x 4096 shots took 34 s (0.415 ms per pub-shot)
def est_zz(): return len(zz_meta()) * ZZ_SHOTS * (250e-6 + np.mean(ZZ_TAUS_US) * 1e-6 + 2e-6) * 1.3 + 4
def est_main(T): return sum(len(ARMS) * COPIES * SHOTS * (0.45e-3 + 2 * k * T) for k in range(1, K + 1)) * 1.1 + 4

def guard(est, svc):
    rem = remaining(svc); print('estimated seconds: %.0f   usage_remaining_seconds: %s' % (est, rem))
    if est > MAX_SECONDS: sys.exit('REFUSED: estimate exceeds %d s' % MAX_SECONDS)
    if rem is not None and est > rem: sys.exit('REFUSED: estimate exceeds remaining seconds')

def load(k): return json.load(open(FILES[k]))
def save(k, d): json.dump(d, open(FILES[k], 'w'), indent=1)
def jobs_get(): return json.load(open(FILES['jobs'])) if os.path.exists(FILES['jobs']) else {}
def jobs_set(k, v): j = jobs_get(); j[k] = v; save('jobs', j)
def live_setup():
    svc = get_service(); b = svc.backend(BACKEND); return svc, b, pick_chain(b)

# ------------------------------------------------------------ analysis
def t99(c): return T99.get(c - 1, 3.0)

def analyze(data, tro, zz=None, headline=True, seed=SEED, S=4000):
    """data[arm] = array (COPIES, steps, 2): avgZ and NN ZZ per copy. Every step carries a 99% interval (Student t over copies)."""
    rng = np.random.default_rng(seed); out = ''
    A = {a: np.array(data[a]) for a in ARMS}; C, Ks, _ = A['client'].shape; tc = t99(C)
    M = {a: A[a].mean(0) for a in ARMS}; SE = {a: A[a].std(0, ddof=1) / np.sqrt(C) for a in ARMS}
    # step-1 null: product state, ZZ fold cannot move it -> every pair of arms must read CONSISTENT
    fails, tot = 0, 0
    out += 'STEP 1 NULL (99%% intervals on arm differences; %d copies)\n' % C
    for i, an in enumerate(('avg<Z>', 'NN<ZZ>')):
        for ia in range(4):
            for ib in range(ia + 1, 4):
                a, b = ARMS[ia], ARMS[ib]; d = M[a][0, i] - M[b][0, i]; hw = tc * np.hypot(SE[a][0, i], SE[b][0, i]); bad = abs(d) > hw
                tot += 1; fails += bad
                out += '  %-7s %-8s - %-8s diff %+.4f +- %.4f %s\n' % (an, LABEL[a], LABEL[b], d, hw, 'EXCLUDES 0' if bad else 'consistent')
    out += 'STEP 1: %s (%d of %d comparisons exclude 0)\n\n' % ('CONSISTENT across all arms' if fails == 0 else 'NOT CONSISTENT', fails, tot)
    for i, an in enumerate(('avg<Z>', 'NN<ZZ>')):
        out += 'PER STEP %s, value +- 99%% half-width\nstep  exact   ' % an + ' '.join('%-15s' % LABEL[a] for a in ARMS) + '\n'
        for k in range(Ks): out += '%4d  %7.4f ' % (k + 1, tro[k, i]) + ' '.join('%7.4f+-%.4f ' % (M[a][k, i], tc * SE[a][k, i]) for a in ARMS) + '\n'
        out += '\n'
    if zz:
        out += 'ZZ per edge (kHz): edge  measured +-sigma  listed(live record)\n'
        for e in sorted(zz['measured'], key=int):
            m, s = zz['measured'][e]; l = zz.get('listed_hz', {}).get(e)
            out += '  %3s   %7.2f +-%.2f   %s\n' % (e, m / 1e3, s / 1e3, '%.2f' % (l / 1e3) if l is not None else 'n/a')
        out += '\n'
    if not headline: return out
    sl = slice(1, Ks)
    def rms_pt(a): return float(np.sqrt(((M[a][sl] - tro[sl]) ** 2).mean()))
    draws = {a: M[a][None] + SE[a][None] * rng.standard_t(max(C - 1, 1), size=(S, Ks, 2)) for a in ARMS}
    R = {a: np.sqrt(((draws[a][:, sl] - tro[None, sl]) ** 2).mean((1, 2))) for a in ARMS}
    ci = lambda x: np.percentile(x, [0.5, 99.5])
    out += 'RMS ERROR VS EXACT, steps 2-%d, both observables pooled, 99%% interval\n' % Ks
    for a in ARMS: lo, hi = ci(R[a]); out += '  %-8s %.4f  [%.4f, %.4f]\n' % (LABEL[a], rms_pt(a), lo, hi)
    d = R['ours'] - R['free']; lo, hi = ci(d); pt = rms_pt('ours') - rms_pt('free')
    out += 'OURS - FREE FIX: %+.4f  99%% interval [%+.4f, %+.4f]\n' % (pt, lo, hi)
    dc = R['ours'] - R['client']; l2, h2 = ci(dc); out += 'OURS - CLIENT:   %+.4f  99%% interval [%+.4f, %+.4f]\n' % (rms_pt('ours') - rms_pt('client'), l2, h2)
    ds = R['signctl'] - R['ours']; l3, h3 = ci(ds); out += 'SIGN CTL - OURS: %+.4f  99%% interval [%+.4f, %+.4f]  -> sign control %s\n' % (
        rms_pt('signctl') - rms_pt('ours'), l3, h3, 'LOSES as required' if l3 > 0 else 'does NOT separate from ours')
    word = 'BELOW' if hi < 0 else 'ABOVE' if lo > 0 else ('CONSISTENT' if (hi - lo) / 2 <= 0.005 else 'CANNOT_TELL')
    out += 'OUTCOME (OURS error relative to FREE FIX error; BELOW = ours lower; CONSISTENT needs interval half-width <= 0.005): %s\n' % word
    return out

# ------------------------------------------------------------ options, sources
def set_clean(opt):
    """No automatic changes: resilience 0, twirling off (gates and measure), dynamical decoupling off."""
    if hasattr(opt, 'resilience_level'): opt.resilience_level = 0
    opt.twirling.enable_gates = False; opt.twirling.enable_measure = False; opt.dynamical_decoupling.enable = False
    return opt

def options_line(opt):
    return 'resilience_level=%s twirling.enable_gates=%s twirling.enable_measure=%s dynamical_decoupling.enable=%s default_shots=%s' % (
        getattr(opt, 'resilience_level', 'n/a'), opt.twirling.enable_gates, opt.twirling.enable_measure, opt.dynamical_decoupling.enable, opt.default_shots)

def show_job_options(job, label):
    try:
        o = job.inputs.get('options', job.inputs); print('%s options read back from the job: %s' % (label, json.dumps(o, default=str)[:600]))
    except Exception as ex: print('%s: could not read options back from job (%s)' % (label, type(ex).__name__))

def sources_table(b, chain, zz, T, ncz, tc, lst):
    """SOURCES: parameter -> hardware job id or record stamp. Returns (rows, problems). Anything without a hardware source is a problem."""
    rows, bad = [], []
    try: stamp = str(b.properties().last_update_date)
    except Exception: stamp = None
    jid = (zz or {}).get('job_id')
    rows.append(('ZZ value per edge (9)', 'fresh ZZ job %s' % jid if jid else 'MISSING'))
    rows.append(('ZZ sign convention', 'on-chip RZZ control inside ZZ job %s' % (zz or {}).get('sign_job_id') if (zz or {}).get('sign') in (1, -1) else 'MISSING'))
    rows.append(('CZ gate time per edge', 'live record %s (target durations, %s ns)' % (stamp, sorted(set(round(t * 1e9, 1) for t in tc))) if stamp and all(t > 0 for t in tc) else 'MISSING'))
    rows.append(('CZ count per edge per step', 'from the transpiled circuit (structural): %s' % ncz))
    rows.append(('T_step', 'live record durations, scheduled transpiled step: %.3f us' % (T * 1e6)))
    rows.append(('FREE FIX listed zz', 'live record %s, fields zz_<a><b>' % stamp if lst and stamp else 'MISSING'))
    for r in rows:
        if 'MISSING' in r[1]: bad.append(r[0])
    return rows, bad

def check_sources(svc, b, chain, zz, T, ncz, tc, lst):
    rows, bad = sources_table(b, chain, zz, T, ncz, tc, lst)
    jobs = jobs_get()
    if zz:
        if jobs.get('zz', {}).get('id') != zz.get('job_id'): bad.append('zz_v2.json job id does not match jobs_v2.json')
        else:
            try:
                if 'DONE' not in str(svc.job(zz['job_id']).status()): bad.append('ZZ job not DONE')
            except Exception: bad.append('ZZ job not found on this instance')
        if sorted(zz.get('measured', {}), key=int) != [str(e) for e in range(N - 1)]: bad.append('measured ZZ missing edges')
    print('SOURCES (parameter -> hardware source):')
    for r in rows: print('  %-28s %s' % r)
    if bad: sys.exit('REFUSED: no hardware source for: %s' % '; '.join(bad))
    return rows

# ------------------------------------------------------------ modes
def run_dry():
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2
    svc, b, chain = live_setup(); print('backend', b.name, 'chain', chain)
    st, pv = transpile_once(b, chain); T, ncz, tc, Tn = edge_times(b, chain, st); param_positions(st)
    print('ONE transpile (optimization_level 0): one Trotter step depth %d, cz %d, T_step %.3f us' % (st.depth(), st.count_ops().get('cz', 0), T * 1e6))
    print('per edge: cz count %s ; cz time ns %s ; T_net us %s' % (ncz, np.round(np.array(tc) * 1e9, 1), np.round(Tn * 1e6, 3)))
    lst = listed_zz(b, chain)
    if lst is None: sys.exit('REFUSED: zz_<a><b> not found in live record for every chain edge')
    L = np.array([lst[e] for e in range(N - 1)])
    print('listed zz (kHz) per edge, from live record:', np.round(L / 1e3, 2))
    print('per-step ZZ phase 2*pi*zeta*T_net (rad) at the listed values:', np.round(2 * np.pi * L * Tn, 4))
    print('gate-count + exact-angle check (placeholder measured = 0.7 x listed, only so the arms carry different angles):')
    gate_check(st, pv, 0.7 * L, L, Tn)
    e_ = set_clean(EstimatorV2(mode=b).options); e2 = EstimatorV2(mode=b); set_clean(e2.options); e2.options.default_shots = SHOTS
    print('Estimator options that will be used:', options_line(e2.options))
    s2 = SamplerV2(mode=b); s2.options.dynamical_decoupling.enable = False; s2.options.twirling.enable_gates = False; s2.options.twirling.enable_measure = False
    print('Sampler (ZZ job) options:', 'dd=%s twirl_gates=%s twirl_measure=%s' % (s2.options.dynamical_decoupling.enable, s2.options.twirling.enable_gates, s2.options.twirling.enable_measure))
    rows, bad = sources_table(b, chain, None, T, ncz, tc, lst)
    print('SOURCES (pending items are filled by the fresh ZZ job; --fire refuses until they exist):')
    for r in rows: print('  %-28s %s' % r)
    ez, em = est_zz(), est_main(T)
    print('jobs: Z = %d circuits x %d shots (%d echoed-Ramsey + %d on-chip sign control); main = %d pubs (%d arms x %d steps x %d copies) x %d shots, 2 observables each'
          % (len(zz_meta()), ZZ_SHOTS, len(zz_meta()) // 2, len(zz_meta()) // 2, len(ARMS) * K * COPIES, len(ARMS), K, COPIES, SHOTS))
    print('estimated seconds: job Z %.0f, main %.0f, total %.0f (each job refused above %d)' % (ez, em, ez + em, MAX_SECONDS))
    print('usage_remaining_seconds:', remaining(svc))

def run_fire_zz():
    from qiskit_ibm_runtime import SamplerV2
    svc, b, chain = live_setup(); guard(est_zz(), svc); pubs = []
    for key in zz_meta():
        pubs.append((transpile(zz_build(key, b.dt), b, initial_layout=chain, optimization_level=0, seed_transpiler=SEED), None, ZZ_SHOTS))
    sm = SamplerV2(mode=b); sm.options.dynamical_decoupling.enable = False; sm.options.twirling.enable_gates = False; sm.options.twirling.enable_measure = False
    job = sm.run(pubs); jobs_set('zz', dict(id=job.job_id(), chain=chain, backend=BACKEND)); print('submitted ZZ job', job.job_id()); show_job_options(job, 'ZZ job')

def run_fetch_zz():
    svc = get_service(); jd = jobs_get()['zz']; res = svc.job(jd['id']).result(); cz, cc = {}, {}
    for key, pr in zip(zz_meta(), res): (cz if key[0] == 'zz' else cc)[key[1:]] = pr.data.p.get_counts()
    rng = np.random.default_rng(SEED); ctl = zz_fit(cc, ZZ_SHOTS, rng, sign=1); sign = determine_sign(ctl)
    print('on-chip sign control (explicit RZZ equal to +%.0f kHz), fitted with sign=+1 (kHz):' % (ZREF / 1e3), {e: round(v[0] / 1e3, 2) for e, v in ctl.items()})
    if sign is None: sys.exit('REFUSED: the on-chip sign control did not return a clean +-%.0f kHz on every edge; cannot fix the sign from hardware' % (ZREF / 1e3))
    fit = zz_fit(cz, ZZ_SHOTS, rng, sign=sign); b = svc.backend(BACKEND); lst = listed_zz(b, jd['chain'])
    save('zz', dict(measured={str(e): list(v) for e, v in fit.items()}, listed_hz={str(e): v for e, v in (lst or {}).items()}, chain=jd['chain'],
                    job_id=jd['id'], sign=sign, sign_job_id=jd['id'], control_fit_hz={str(e): list(v) for e, v in ctl.items()}))
    print('saved', FILES['zz'], '; sign from chip =', sign)
    for e, v in fit.items(): print('edge %d  measured %.2f +- %.2f kHz   listed %s' % (e, v[0] / 1e3, v[1] / 1e3, '%.2f' % (lst[e] / 1e3) if lst else 'n/a'))

def run_fire():
    from qiskit_ibm_runtime import EstimatorV2
    svc, b, chain = live_setup()
    if not os.path.exists(FILES['zz']): sys.exit('REFUSED: no fresh ZZ result (zz_v2.json); run --fire-zz and --fetch-zz first')
    zz = load('zz')
    if zz.get('chain') != chain: sys.exit('REFUSED: chain changed since the ZZ job; re-run --fire-zz')
    st, pv = transpile_once(b, chain); T, ncz, tc, Tn = edge_times(b, chain, st); guard(est_main(T), svc)
    lst = listed_zz(b, chain)
    if lst is None: sys.exit('REFUSED: listed zz not found in live record')
    rows = check_sources(svc, b, chain, zz, T, ncz, tc, lst)
    L = np.array([lst[e] for e in range(N - 1)]); Mz = np.array([zz['measured'][str(e)][0] for e in range(N - 1)])
    print('gate-count + exact-angle check before firing:'); gate_check(st, pv, Mz, L, Tn)
    avgz, avgzz = obs_ops(); ob = [avgz.apply_layout(st.layout), avgzz.apply_layout(st.layout)]
    steps = {a: bound_step(st, pv, arm_angles(a, Mz, L, Tn)) for a in ARMS}; pubs, order = [], []
    for c in range(COPIES):
        for k in range(1, K + 1):
            rot = ARMS[(k + c) % 4:] + ARMS[:(k + c) % 4]          # rotate arm order so no arm is always first
            for a in rot: pubs.append((stack(steps[a], k), ob)); order.append((a, c, k))
    e_ = EstimatorV2(mode=b); set_clean(e_.options); e_.options.default_shots = SHOTS
    ol = options_line(e_.options); print('Estimator options:', ol)
    if e_.options.resilience_level != 0 or e_.options.twirling.enable_gates or e_.options.twirling.enable_measure or e_.options.dynamical_decoupling.enable:
        sys.exit('REFUSED: clean options not set')
    job = e_.run(pubs); jobs_set('main', dict(id=job.job_id(), order=order, T=T, Tn=list(Tn), listed_hz=list(L), measured_hz=list(Mz), sign=zz['sign'],
                                              sources=rows, options=ol))
    print('submitted main job', job.job_id(), 'pubs', len(pubs)); show_job_options(job, 'Main job')

def run_fetch():
    svc = get_service(); jd = jobs_get()['main']; res = svc.job(jd['id']).result()
    out = {a: [[None] * K for _ in range(COPIES)] for a in ARMS}
    for (a, c, k), pr in zip(jd['order'], res): out[a][c][k - 1] = [float(x) for x in pr.data.evs]
    save('main', out); print('saved', FILES['main'])

def run_analyze():
    zz = load('zz') if os.path.exists(FILES['zz']) else None; print(analyze(load('main'), exact_values(), zz))

def run_test():
    """CODE SMOKE TEST ONLY. Offline, tiny, noiseless gates. Checks plumbing; produces no headline, no prediction, and sets no constant."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    rng = np.random.default_rng(SEED); fb = FakeKingston(); chain = pick_chain(fb); print('SMOKE TEST ONLY (code plumbing; no result). seed', SEED, 'chain', chain)
    st, pv = transpile_once(fb, chain); T, ncz, tc, Tn = edge_times(fb, chain, st); param_positions(st)
    print('T_step %.3f us; per-edge cz count %s; T_net/T_step %s' % (T * 1e6, ncz, np.round(Tn / T, 3)))
    planted = np.array([11.6, -7.4, 3.8, -5.35, 9.4, 4.7, -5.9, 11.7, -6.7]) * 1e3; lst = planted + 1.9e3
    sim0 = AerSimulator(seed_simulator=SEED); cz, cc = {}, {}
    for key in zz_meta():
        r = sim0.run(zz_build(key, fb.dt, planted_hz=planted), shots=ZZ_SHOTS).result().get_counts(); (cz if key[0] == 'zz' else cc)[key[1:]] = r
    ctl = zz_fit(cc, ZZ_SHOTS, rng, sign=1, nboot=5); sign = determine_sign(ctl); print('control-fit path: sign =', sign)
    fit = zz_fit(cz, ZZ_SHOTS, rng, sign=sign or 1, nboot=5); Mz = np.array([fit[e][0] for e in range(N - 1)])
    print('ZZ path executed; max |fit - planted| = %.2f kHz (plumbing check only)' % (np.abs(Mz - planted).max() / 1e3))
    gate_check(st, pv, Mz, lst, Tn, k=3)
    ks, cs, sh = 4, 2, 800; sim = AerSimulator(seed_simulator=SEED); pl = [((chain[e], chain[e + 1]), float(np.pi * planted[e] * Tn[e])) for e in range(N - 1)]
    data = {a: np.zeros((cs, ks, 2)) for a in ARMS}
    for a in ARMS:
        step = bound_step(st, pv, arm_angles(a, Mz, lst, Tn))
        for c in range(cs):
            for k in range(1, ks + 1):
                qc = stack(step, k, planted=pl); cr = ClassicalRegister(N, 'm'); qc.add_register(cr); [qc.measure(chain[i], cr[i]) for i in range(N)]
                cts = sim.run(qc, shots=sh).result().get_counts(); z = zz_ = 0.0
                for bits, v in cts.items():
                    s = [1 - 2 * int(bits.replace(' ', '')[::-1][i]) for i in range(N)]; z += v * np.mean(s); zz_ += v * np.mean([s[i] * s[i + 1] for i in range(N - 1)])
                data[a][c, k - 1] = (z / sh, zz_ / sh)
    analyze(data, exact_values(k=ks), headline=False); print('analysis path executed (headline suppressed in smoke test)')
    print('sources path: ', sources_table(fb, chain, None, T, ncz, tc, None)[1], '(missing items are expected offline)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for m in ('test', 'dry', 'fire-zz', 'fetch-zz', 'fire', 'fetch', 'analyze'): ap.add_argument('--' + m, action='store_true')
    a = ap.parse_args()
    if a.test: run_test()
    elif a.dry: run_dry()
    elif a.fire_zz: run_fire_zz()
    elif a.fetch_zz: run_fetch_zz()
    elif a.fire: run_fire()
    elif a.fetch: run_fetch()
    elif a.analyze: run_analyze()
    else: ap.print_help()
