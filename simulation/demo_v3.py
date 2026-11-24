"""Demo 4 v3: Trotter dynamics of the 10-qubit transverse-field Ising chain, the client's run, three readings of the SAME shots.
  N = raw (client)
  S = rescaled by the fade predicted from IBM's STATED per-edge CZ error, sx error and readout error (record stamp printed)
  O = rescaled by the fade predicted from OUR job-P measured CZ error per edge and measured readout (same model)
Product: IBM code Q17, real gate error versus IBM's stated figure (RB average infidelity). No fold, no ZZ angle change: the client circuits
are the CLIENT arm of v2, unchanged. Job P = short two-qubit RB on each of the 9 chain edges + all-0/all-1 readout on the chain (no
twirling, no DD, level 0). Main = the client circuits, 4 interleaved copies, resilience 0, runtime changes off, options read back.
FADE MODEL (fixed here before any v3 data; no parameter is fitted on the main shots):
  For observable ZZ on chain edge e at step k, walk the transpiled stacked circuit backward from the end. Cone = {the two qubits}.
  Every CZ touching the cone multiplies the signal by (1 - 4*err_cz(edge)/3) and adds both qubits to the cone; every sx on a cone qubit
  multiplies by (1 - 2*err_sx(q)). At the end, each qubit in the observable multiplies by (1 - p01 - p10) (stated: 1 - 2*readout error).
  Scored observable NN<ZZ> = mean over edges; its fade is the exact-value-weighted mean of the per-edge fades. <Z> is a control (cone of
  one qubit, plain mean), printed and never scored (Pack Q #44: it decays to ~0 by step 8).
  O differs from S ONLY in err_cz (RB per-edge, with the stated sx part of the RB circuits taken out) and readout (measured).
Modes: --test (code smoke test only, offline, no numbers) --dry --power --fire --fetch --analyze.
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE (optional). The key is never printed or stored."""
import os, sys, json, re, argparse, warnings, random
warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford
import demo_v2 as V

CHIPS = ('ibm_kingston', 'ibm_fez', 'ibm_marrakesh')
BACKEND = os.environ.get('DEMO_BACKEND', 'ibm_kingston')
if BACKEND not in CHIPS: sys.exit('DEMO_BACKEND must be one of %s' % (CHIPS,))
V.BACKEND = BACKEND; SHORT = BACKEND.replace('ibm_', '')
HERE = V.HERE
N, K, COPIES, SHOTS = V.N, V.K, V.COPIES, V.SHOTS
SEED = 20261119; MAX_SECONDS = 200
RB_LENGTHS, RB_SEQ, RB_SHOTS, RB_BOOT = [1, 4, 8, 16, 28], 4, 300, 100
RO_SHOTS = 2000
T99 = 5.841        # Student t, 99% two-sided, 3 dof (4 copies)
TQ = {2: 63.657, 3: 9.925, 4: 5.841}
def tq(c): return TQ.get(c, 2.576)
PILOT_STEPS, PILOT_COPIES, PILOT_RB_LENGTHS, PILOT_RB_SEQ, PILOT_MAX = [2, 6, 12, 20], 3, [1, 8, 16], 2, 30
EQUIV = 0.005      # CONSISTENT margin on the paired mean-abs-error difference (error units)
RHO_RANGE = (1.30, 1.82)   # measured/stated CZ error: Oct RB kingston 1.305 (finance PLAN), 19 Nov RB 1.56-1.82 (finance job dbv6lk7f8rhs73bncf00)
FILES = {k: os.path.join(HERE, v % SHORT) for k, v in dict(jobs='jobs_v3_%s.json', main='main_v3_%s.json', pdata='pdata_v3_%s.json', mainp='main_v3_pilot_%s.json', pdatap='pdata_v3_pilot_%s.json', scale='scale_v3_%s.json').items()}
# FROZEN effective light-cone exponent (fade = cone_fade ** SCALE). kingston: calibrated once on v2 CLIENT job dbv6u3nf8rhs73bncoe0 (an earlier,
# independent hardware job) at stated CZ error x RHO_REF (our Oct RB 1.305), see PLAN_v3.md. Never refitted on v3 shots. fez/marrakesh: no earlier job
# on the chip; the pilot on that chip is the calibration job (written to scale_v3_<chip>.json by --analyze --pilot), else PILOT NEEDED.
RHO_REF = {'ibm_kingston': 1.305, 'ibm_fez': 1.255, 'ibm_marrakesh': None}   # our Oct RB measured/stated (finance demo.py RB_RATIO); marrakesh none
SCALE_FROZEN = {'ibm_kingston': 0.18}   # calibrate_scale on v2 CLIENT at stated x 1.305 gave 0.18 (at stated x 1.0 it gives 0.21); frozen 19 Nov before any v3 shot

PLAN = os.path.join(HERE, 'PLAN_v3.md')
FIN = os.path.join(HERE, '..', '1_finance', 'state_v2.json')
V2MAIN = os.path.join(HERE, 'main_v2.json'); V2JOB = 'dbv6u3nf8rhs73bncoe0'

def load(k): return json.load(open(FILES[k]))
def save(k, d): json.dump(d, open(FILES[k], 'w'), indent=1)
def jobs_get(): return load('jobs') if os.path.exists(FILES['jobs']) else {}
def jobs_set(k, v): j = jobs_get(); j[k] = v; save('jobs', j)

# ------------------------------------------------------------ stated record
def stamp_of(b):
    try: return str(b.properties().last_update_date)
    except Exception: return None

def stated_record(b, chain):
    """STATED from the live record (target): per-edge CZ error (RB average infidelity), per-qubit sx error, per-qubit readout error."""
    t = b.target; cz, sx, ro = {}, {}, {}
    for e in range(N - 1):
        a, c = chain[e], chain[e + 1]; v = None
        for pr in ((a, c), (c, a)):
            try: v = float(t['cz'][pr].error); break
            except Exception: pass
        if v is None: sys.exit('REFUSED: no stated CZ error for edge %d-%d in the live record' % (a, c))
        cz[e] = v
    for q in chain:
        sx[q] = float(t['sx'][(q,)].error); ro[q] = float(t['measure'][(q,)].error)
    return dict(cz=cz, sx=sx, ro=ro)

# ------------------------------------------------------------ fade model
def exact_edge_values(n=N, k=K):
    """Exact (ideal) per-edge ZZ and per-qubit Z per step, from the ideal circuit (used only as weights)."""
    from qiskit.quantum_info import Statevector, SparsePauliOp
    qc, pv = V.step_param(n); step = qc.assign_parameters({p: -2 * V.J * V.DT for p in pv}); sv = Statevector.from_label('0' * n)
    ze, zq = [], []
    for _ in range(k):
        sv = sv.evolve(step)
        ze.append([float(sv.expectation_value(SparsePauliOp.from_sparse_list([('ZZ', [a, a + 1], 1.0)], n)).real) for a in range(n - 1)])
        zq.append([float(sv.expectation_value(SparsePauliOp.from_sparse_list([('Z', [q], 1.0)], n)).real) for q in range(n)])
    return np.array(ze), np.array(zq)

def cone_fades(st, chain, cz_err, sx_err):
    """Backward light-cone walk of the transpiled step, repeated K times. Returns G[support-key][k-1] = gate-part fade for
    k steps. keys: ('e', e) for ZZ on chain edge e, ('q', i) for Z on chain qubit i. cz_err[e] per chain edge (err of the CZ on that edge),
    sx_err[q] per physical qubit. Also returns the count of CZ and sx in the cone."""
    pos = {q: i for i, q in enumerate(chain)}; ops = []
    for ins in st.data:
        nm = ins.operation.name
        if nm in ('cz', 'sx', 'x'): ops.append((nm, tuple(st.find_bit(q).index for q in ins.qubits)))
    cz_of = {}
    for e in range(N - 1): cz_of[frozenset((chain[e], chain[e + 1]))] = e
    out, cnt = {}, {}
    for key, start in [(('e', e), {chain[e], chain[e + 1]}) for e in range(N - 1)] + [(('q', i), {chain[i]}) for i in range(N)]:
        cone = set(start); F = 1.0; ncz = nsx = 0; res = []; cc = []
        for _ in range(K):
            for nm, qs in reversed(ops):
                if nm == 'cz':
                    if qs[0] in cone or qs[1] in cone:
                        F *= 1 - 4 * cz_err[cz_of[frozenset(qs)]] / 3; cone |= set(qs); ncz += 1
                elif qs[0] in cone: F *= 1 - 2 * sx_err[qs[0]]; nsx += 1
            res.append(F); cc.append((ncz, nsx))
        out[key] = np.array(res); cnt[key] = cc
    return out, cnt

def model_fades(st, chain, cz_err, sx_err, ro_pm, ze, scale=1.0):
    """cz_err: list per edge; sx_err: dict per qubit; ro_pm: dict per qubit of (1 - p01 - p10). ze = exact per-edge ZZ [K,9].
    Returns (fade_NN[K], fade_Z[K], per-edge fades [K,9])."""
    G, _ = cone_fades(st, chain, cz_err, sx_err); Fe = np.zeros((K, N - 1))
    for e in range(N - 1): Fe[:, e] = G[('e', e)] * ro_pm[chain[e]] * ro_pm[chain[e + 1]]
    w = ze; fnn = (w * Fe).sum(1) / w.sum(1)
    fz = np.mean([G[('q', i)] * ro_pm[chain[i]] for i in range(N)], axis=0)
    return fnn ** scale, fz ** scale, Fe ** scale

def stated_model(b, chain, st, ze, scale=1.0):
    s = stated_record(b, chain)
    return model_fades(st, chain, [s['cz'][e] for e in range(N - 1)], s['sx'], {q: 1 - 2 * s['ro'][q] for q in chain}, ze, scale), s

def measured_cz_isolated(rb_err, k_sx, sx_pair):
    """RB per-CZ error includes the single-qubit gates inside the RB circuits (k_sx sx per CZ). The model counts sx separately at the
    STATED sx error, so that part is taken out: r_cz = r_RB / (1 - 2*err_sx)^k_sx, error = 0.75 (1 - r_cz)."""
    r = (1 - 4 * rb_err / 3) / (1 - 2 * float(np.mean(sx_pair))) ** k_sx
    return 0.75 * (1 - r)

def calibrate_scale(nn, ex, st, chain, cz_err, sx, ro_pm, ze):
    """Single exponent s in [0.02, 1.5] minimizing mean abs error of raw/fade^s vs exact over steps 2-K (copies pooled)."""
    base = model_fades(st, chain, cz_err, sx, ro_pm, ze)[0]; g = np.linspace(0.02, 1.5, 297)
    err = [np.mean(np.abs(nn / (base ** x)[None, :] - ex[None, :])[:, 1:]) for x in g]; return float(g[int(np.argmin(err))])

def get_scale():
    if SCALE_FROZEN.get(BACKEND): return SCALE_FROZEN[BACKEND]
    if os.path.exists(FILES['scale']): return json.load(open(FILES['scale']))['scale']
    return None

# ------------------------------------------------------------ job P: RB on chain edges + readout
def rb_circuits(b, chain, seed=11, lengths=None, nseq=None):
    lengths = lengths or RB_LENGTHS; nseq = nseq or RB_SEQ; rng = random.Random(seed); circs, meta = [], []
    for e in range(N - 1):
        for m in lengths:
            for s_ in range(nseq):
                qc = QuantumCircuit(2); comp = Clifford(QuantumCircuit(2))
                for _ in range(m):
                    c = random_clifford(2, seed=rng.randrange(10 ** 9)); qc.compose(c.to_circuit(), inplace=True); comp = comp.compose(c)
                qc.compose(comp.adjoint().to_circuit(), inplace=True); qc.measure_all()
                tc = transpile(qc, b, initial_layout=[chain[e], chain[e + 1]], optimization_level=0, seed_transpiler=5)
                ops = tc.count_ops(); ncz = int(ops.get('cz', 0)); nsx = int(ops.get('sx', 0) + ops.get('x', 0))
                circs.append(tc); meta.append((e, m, ncz, nsx))
    return circs, meta

def ro_circuits(b, chain):
    out = []
    for ones in (False, True):
        qc = QuantumCircuit(N, N)
        if ones: qc.x(range(N))
        qc.measure(range(N), range(N)); out.append(transpile(qc, b, initial_layout=chain, optimization_level=0, seed_transpiler=SEED))
    return out

def counts_of(pr):
    d = pr.data; reg = [getattr(d, a) for a in dir(d) if not a.startswith('_') and hasattr(getattr(d, a), 'get_counts')][0]; return reg.get_counts()

def rb_fit(counts_list, meta, rng, boot=RB_BOOT):
    """per edge: fit P00 = 0.25 + A r^ncz over the transpiled CZ count (A absorbs readout/prep); error per CZ = 0.75 (1 - r). Bootstrap sigma over shots."""
    Ag = np.linspace(0.45, 0.80, 36)[:, None, None]; rg = np.linspace(0.90, 1.0, 1001)[None, :, None]; out = {}
    for e in sorted(set(x[0] for x in meta)):
        xs, ps, ns = [], [], []
        for c, mt in zip(counts_list, meta):
            if mt[0] != e: continue
            tot = sum(c.values()); xs.append(mt[2]); ps.append(c.get('00', 0) / tot); ns.append(tot)
        xs, ps, ns = np.array(xs, float), np.array(ps), np.array(ns)
        def fit(y):
            sse = ((0.25 + Ag * rg ** xs[None, None, :] - y[None, None, :]) ** 2).sum(2); i = np.unravel_index(np.argmin(sse), sse.shape)
            return 0.75 * (1 - rg[0, i[1], 0])
        est = fit(ps); bs = [fit(rng.binomial(ns, ps) / ns) for _ in range(boot)]
        out[e] = (float(est), float(np.std(bs)))
    return out

def sx_per_cz(meta):
    k = {}
    for e in sorted(set(x[0] for x in meta)):
        nc = sum(m[2] for m in meta if m[0] == e); ns = sum(m[3] for m in meta if m[0] == e); k[e] = ns / max(nc, 1)
    return k

def ro_fit(cts_pair, chain):
    """per qubit p01 (read 1 given prepared 0) and p10 from the all-0 and all-1 circuits; returns (1 - p01 - p10) per qubit and the raw pair."""
    out = {}
    for i, q in enumerate(chain):
        p = []
        for cts, want in zip(cts_pair, (0, 1)):
            tot = sum(cts.values()); bad = sum(v for k_, v in cts.items() if int(k_.replace(' ', '')[::-1][i]) != want); p.append(bad / tot)
        out[q] = (p[0], p[1])
    return out

# ------------------------------------------------------------ estimates, guards
def est_P(circs): return 3.0 + sum(RB_SHOTS * (0.0006 + 1e-6 * c.depth()) for c in circs) + 2 * RO_SHOTS * 0.0008
def est_main(T): return sum(COPIES * SHOTS * (0.45e-3 + 2 * k * T) for k in range(1, K + 1)) * 1.1 + 4
def guard_total(est, svc):
    rem = V.remaining(svc); print('estimated seconds (job P + main): %.0f   usage_remaining_seconds: %s' % (est, rem))
    if est > MAX_SECONDS: sys.exit('REFUSED: total estimate exceeds %d s' % MAX_SECONDS)
    if rem is not None and est > rem: sys.exit('REFUSED: estimate exceeds remaining seconds')

def plan_ratio():
    if not os.path.exists(PLAN): return None
    pat = r'POWER RATIO:\s*([0-9]+(?:\.[0-9]+)?)' if BACKEND == 'ibm_kingston' else r'POWER RATIO \(%s\):\s*([0-9]+(?:\.[0-9]+)?)' % BACKEND
    r = re.search(pat, open(PLAN).read(), re.I); return float(r.group(1)) if r else None

# ------------------------------------------------------------ analysis (paired, three readings)
def mae_copies(raw, fade, tro):
    """raw [C,K]; rescaled raw/fade; mean abs error vs exact over steps 2-K, per copy."""
    return np.nanmean(np.abs(raw / fade[None, :] - tro[None, :])[:, 1:], axis=1)

def interval(x):
    x = np.asarray(x, float); c = len(x); m = x.mean(); hw = tq(c) * x.std(ddof=1) / np.sqrt(c) if c > 1 else np.nan; return m, m - hw, m + hw

def word(lo, hi):
    return 'ABOVE' if lo > 0 else 'BELOW' if hi < 0 else 'CONSISTENT' if (-EQUIV < lo and hi < EQUIV) else 'CANNOT_TELL'

def analyze3(raw, tro, fS, fO, stamp, label=''):
    """raw [C,K,2] (avgZ, NN<ZZ>) client copies. fS, fO = (fade_NN[K], fade_Z[K]). Scored: NN<ZZ>. Z printed as control."""
    C = raw.shape[0]; out = ''
    out += '%sSTATED record stamp: %s\n' % (label, stamp)
    nn = raw[:, :, 1]; ex = tro[:, 1]
    fN = np.ones(K)
    E = {'N': mae_copies(nn, fN, ex), 'S': mae_copies(nn, fS[0], ex), 'O': mae_copies(nn, fO[0], ex)}
    out += '%sNN<ZZ> (scored), mean abs error vs exact over steps 2-%d, value and 99%% interval over %d copies\n' % (label, K, C)
    for k_, nm in (('N', 'N raw (client)'), ('S', 'S fade from STATED errors'), ('O', 'O fade from OUR measured errors')):
        m, lo, hi = interval(E[k_]); out += '  %-34s %.4f  [%.4f, %.4f]\n' % (nm, m, lo, hi)
    dn = E['O'] - E['N']; m2, l2, h2 = interval(dn); wN = word(l2, h2)
    out += '%sHEADLINE ours vs the CLIENT raw run: O minus N, paired per copy (mean over steps 2-%d of err_O - err_N): %+.4f  99%% interval [%+.4f, %+.4f]\n' % (label, K, m2, l2, h2)
    out += '%sOUTCOME O vs N (BELOW = our calibrated run closer to exact; ABOVE = worse; CONSISTENT = inside +-%.3f): %s\n' % (label, EQUIV, wN)
    d = E['O'] - E['S']; m, lo, hi = interval(d); w = word(lo, hi)
    out += '%sBeside it, O minus S (vs the generic stated-error fade): %+.4f  99%% interval [%+.4f, %+.4f]\n' % (label, m, lo, hi)
    out += '%sOUTCOME O vs S: %s   (S minus N: %+.4f)\n' % (label, w, (E['S'] - E['N']).mean())
    w = wN
    out += '\nPER STEP NN<ZZ>, paired differences (O err - S err), mean over copies +- 99%% half-width\nstep  exact   raw     S-scaled  O-scaled  fadeS   fadeO   dErr(O-S)\n'
    for k in range(K):
        eo = np.abs(nn[:, k] / fO[0][k] - ex[k]); es = np.abs(nn[:, k] / fS[0][k] - ex[k]); dd = eo - es
        hw = tq(C) * dd.std(ddof=1) / np.sqrt(C) if C > 1 else float('nan')
        if np.isnan(nn[:, k]).any(): continue
        out += '%4d  %7.4f %7.4f %8.4f %9.4f  %6.4f  %6.4f  %+.4f+-%.4f\n' % (k + 1, ex[k], nn[:, k].mean(), (nn[:, k] / fS[0][k]).mean(), (nn[:, k] / fO[0][k]).mean(), fS[0][k], fO[0][k], dd.mean(), hw)
    z = raw[:, :, 0]; zex = tro[:, 0]; sl = slice(1, 8)
    out += '\nCONTROL <Z> (not scored; decays to ~0 by step 8), mean abs error steps 2-8: N %.4f  S %.4f  O %.4f\n' % tuple(
        np.nanmean(np.abs(z / f[None, :] - zex[None, :])[:, sl]) for f in (np.ones(K), fS[1], fO[1]))
    if np.isnan(nn).any():
        for nm_, d in (('O-N', E['O'] - E['N']), ('O-S', E['O'] - E['S'])):
            fl = tq(C) * d.std(ddof=1) / np.sqrt(C); out += 'PILOT: measured |%s| %.4f ; pilot own paired copy floor (99%% half-width over %d copies) %.4f ; ratio %.2f\n' % (nm_, abs(d.mean()), C, fl, abs(d.mean()) / max(fl, 1e-12))
    return out, w

# ------------------------------------------------------------ power (hardware numbers only)
def power(b, chain, st, ze, tro, quiet=False, scale=None):
    """Expected |O - S| paired error difference vs its paired floor, from OUR data only: v2 CLIENT per-copy per-step values (job V2JOB),
    the live stated record, and measured/stated ratio rho over RHO_RANGE (our RB runs). O fade at ratio rho = the model with err_cz*rho."""
    scale = scale if scale is not None else get_scale()
    if BACKEND != 'ibm_kingston' or scale is None or not os.path.exists(V2MAIN):
        l = ['%s: no earlier hardware job on this chip to calibrate the frozen fade exponent or give a copy floor -> PILOT NEEDED, POWER RATIO 0.00' % BACKEND]
        if not quiet: print(l[0])
        return 0.0, l
    d2 = np.array(json.load(open(V2MAIN))['client']); (fS, fZ, _), s = stated_model(b, chain, st, ze, scale); lines = ['frozen fade exponent %.3f (calibrated on v2 CLIENT, job %s)' % (scale, V2JOB)]
    ex = tro[:, 1]; nn = d2[:, :, 1]
    lines.append('record stamp %s ; stated CZ error per edge %s (mean %.5f, min %.5f, max %.5f)' % (
        stamp_of(b), np.round([s['cz'][e] for e in range(N - 1)], 5).tolist(), np.mean(list(s['cz'].values())), min(s['cz'].values()), max(s['cz'].values())))
    lines.append('v2 CLIENT NN<ZZ>/exact (hardware job %s), mean steps 2-%d: %.3f ; model S fade mean steps 2-%d: %.3f' % (V2JOB, K, (nn.mean(0)[1:] / ex[1:]).mean(), K, fS[1:].mean()))
    res = {}; resN = {}
    for rho in (RHO_RANGE[0], 1.56, RHO_RANGE[1]):
        fO = model_fades(st, chain, [s['cz'][e] * rho for e in range(N - 1)], s['sx'], {q: 1 - 2 * s['ro'][q] for q in chain}, ze, scale)[0]
        es = np.abs(nn / fS[None, :] - ex[None, :])[:, 1:].mean(1); eo = np.abs(nn / fO[None, :] - ex[None, :])[:, 1:].mean(1); d = eo - es
        sig = -d.mean(); floor = T99 * d.std(ddof=1) / np.sqrt(len(d)); eN = np.abs(nn - ex[None, :])[:, 1:].mean(1)
        dn = eo - eN; sigN = abs(dn.mean()); floorN = T99 * dn.std(ddof=1) / np.sqrt(len(dn)); resN[rho] = (sigN, floorN, dn.mean())
        res[rho] = (sig, floor, d.mean(), es.mean(), eo.mean(), eN.mean(), fO[-1], fO[1:].mean())
        lines.append('rho %.2f: mean abs error N %.4f  S %.4f  O %.4f ; O-S %+.4f ; benefit err_S - err_O %+.4f (|O-S| %.4f) ; paired floor (99%% half-width over the 4 v2 copies) %.4f ; ratio %.2f' % (
            rho, eN.mean(), es.mean(), eo.mean(), d.mean(), sig, abs(sig), floor, sig / floor if floor > 0 else float('inf')))
    # unpaired floor for reference: v2 client copy spread of the single-reading mean abs error
    eN = np.abs(nn - ex[None, :])[:, 1:].mean(1); lines.append('for reference: v2 CLIENT per-copy mean abs error %.4f, 99%% half-width over copies (unpaired) %.4f' % (eN.mean(), T99 * eN.std(ddof=1) / 2))
    for rho, (sg, fl, dm) in resN.items():
        lines.append('rho %.2f: O vs CLIENT raw: O-N %+.4f (|O-N| %.4f) ; paired floor %.4f ; ratio %.2f' % (rho, dm, sg, fl, sg / fl))
    ratioS = max(0.0, min(v[0] / v[1] for v in res.values()))
    ratio = min(sg / fl for sg, fl, _ in resN.values())
    lines.append('POWER RATIO O vs S (signed benefit / paired floor, smallest over rho, floored at 0): %.2f' % ratioS)
    lines.append('POWER RATIO (headline, |O - N| / paired floor on v2 copies, smallest over rho %.2f-%.2f): %.2f' % (RHO_RANGE[0], RHO_RANGE[1], ratio))
    if not quiet:
        for l in lines: print(l)
    return ratio, lines

# ------------------------------------------------------------ SOURCES
def sources_rows(b, chain, st, ncz_cone, pjob):
    stamp = stamp_of(b); fin = json.load(open(FIN)) if os.path.exists(FIN) else {}
    rows = [('stated CZ / sx / readout error', 'live record %s (RB average infidelity for CZ)' % stamp if stamp else 'MISSING'),
            ('measured CZ error per edge (9)', 'job P %s (RB, fired with the main job, applied in analysis)' % pjob if pjob else 'MISSING'),
            ('measured readout per chain qubit', 'job P %s (all-0 / all-1)' % pjob if pjob else 'MISSING'),
            ('measured/stated range 1.30-1.82 (power only)', 'kingston Oct RB 1.305 (fez 1.255); finance job P %s 1.56-1.82' % fin.get('job_p') if fin.get('job_p') else 'MISSING'),
            ('v2 CLIENT values, copy spread, fade exponent', ('kingston job %s (main_v2.json)' % V2JOB) if BACKEND == 'ibm_kingston' else ('pilot on %s (scale_v3_%s.json)' % (BACKEND, SHORT))),
            ('CZ / sx count in the cone', 'transpiled circuit (structural), cone model fixed in the script docstring'),
            ('chain, circuits, layout', 'v2 client arm, same chain and transpile (level 0, seed %d)' % SEED)]
    return rows, [r[0] for r in rows if 'MISSING' in r[1]]

def run_dry():
    svc, b, chain = V.live_setup(); print('backend', b.name, 'chain', chain, '; record stamp', stamp_of(b))
    st, pv = V.transpile_once(b, chain); T, ncz, tc, Tn = V.edge_times(b, chain, st); V.param_positions(st)
    s = stated_record(b, chain)
    print('client step: depth %d, cz %d, sx %d, T_step %.3f us ; cz per edge %s' % (st.depth(), st.count_ops().get('cz', 0), st.count_ops().get('sx', 0), T * 1e6, ncz))
    print('STATED CZ error per edge (%s record, RB average infidelity):' % BACKEND, np.round([s['cz'][e] for e in range(N - 1)], 5).tolist())
    print('STATED sx per qubit:', np.round([s['sx'][q] for q in chain], 5).tolist(), ' readout:', np.round([s['ro'][q] for q in chain], 4).tolist())
    rbc, meta = rb_circuits(b, chain); roc = ro_circuits(b, chain); eP, eM = est_P(rbc) , est_main(T)
    print('job P: %d RB circuits (9 edges x %d lengths x %d seqs, %d shots) + 2 readout circuits (%d shots); est %.0f s ; main: %d pubs (%d copies x %d steps) x %d shots, est %.0f s' % (
        len(rbc), len(RB_LENGTHS), RB_SEQ, RB_SHOTS, RO_SHOTS, eP, COPIES * K, COPIES, K, SHOTS, eM))
    print('RB circuits: mean cz %.1f max cz %d ; sx per cz %s' % (np.mean([m[2] for m in meta]), max(m[2] for m in meta), np.round([sx_per_cz(meta)[e] for e in range(N - 1)], 2).tolist()))
    print('TOTAL estimate %.0f s (limit %d) ; usage_remaining_seconds %s' % (eP + eM, MAX_SECONDS, V.remaining(svc)))
    e_ = V.set_clean(__import__('qiskit_ibm_runtime').EstimatorV2(mode=b).options); e2 = __import__('qiskit_ibm_runtime').EstimatorV2(mode=b); V.set_clean(e2.options); e2.options.default_shots = SHOTS
    print('Estimator options that will be used:', V.options_line(e2.options))
    rows, bad = sources_rows(b, chain, st, None, None); print('SOURCES (job P id fills in at fire):'); [print('  %-44s %s' % r) for r in rows]
    ze, _ = exact_edge_values(); print('chip %s power:' % BACKEND); ratio, _ = power(b, chain, st, ze, V.exact_values()); print('plan POWER RATIO line for %s:' % BACKEND, plan_ratio(), '; fade exponent frozen:', get_scale())

def run_power():
    svc, b, chain = V.live_setup(); st, pv = V.transpile_once(b, chain); ze, _ = exact_edge_values(); tro = V.exact_values()
    print('chip', BACKEND, 'stamp', stamp_of(b)); ratio, _ = power(b, chain, st, ze, tro)
    if BACKEND == 'ibm_kingston': print('--- alternative exponent calibrated at stated x 1.0 (0.21), for transparency'); power(b, chain, st, ze, tro, scale=0.21)

def run_fire(pilot=False):
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2
    r = plan_ratio()
    if not pilot and r is not None and r < 3: sys.exit('REFUSED: PLAN_v3.md POWER RATIO is %s (needs x >= 3)' % r)   # the planning-file gate applies only when that file exists
    if not pilot and get_scale() is None: sys.exit('REFUSED: no frozen fade exponent for %s (PILOT NEEDED first)' % BACKEND)
    svc, b, chain = V.live_setup(); st, pv = V.transpile_once(b, chain); T, ncz, tc, Tn = V.edge_times(b, chain, st); V.param_positions(st)
    pk = 'p' if pilot else ''
    rbc, meta = rb_circuits(b, chain, lengths=PILOT_RB_LENGTHS if pilot else None, nseq=PILOT_RB_SEQ if pilot else None); roc = ro_circuits(b, chain)
    nco, stp = (PILOT_COPIES, PILOT_STEPS) if pilot else (COPIES, list(range(1, K + 1)))
    est_m = sum(nco * SHOTS * (0.45e-3 + 2 * k * T) for k in stp) * 1.1 + 4 if pilot else est_main(T)
    est = est_P(rbc) + est_m
    if pilot and est > PILOT_MAX: sys.exit('REFUSED: pilot estimate %.0f s exceeds %d s' % (est, PILOT_MAX))
    guard_total(est, svc)
    s = stated_record(b, chain); stamp = stamp_of(b)
    rows, bad = sources_rows(b, chain, st, None, 'pending'); bad = [x for x in bad if not x.startswith('measured')]
    print('SOURCES:'); [print('  %-44s %s' % r) for r in rows]
    if bad or not stamp: sys.exit('REFUSED: no hardware source for: %s' % bad)
    print('STATED record stamp', stamp)
    avgz, avgzz = V.obs_ops(); ob = [avgz.apply_layout(st.layout), avgzz.apply_layout(st.layout)]
    step = V.bound_step(st, pv, V.arm_angles('client', np.zeros(N - 1), np.zeros(N - 1), Tn)); L = len(st.data) + 1
    for k in (1, K): V.verify_angles(st, step, np.full(N - 1, -2 * V.J * V.DT), k, V.stack(step, k))
    pubs, order = [], []
    for c in range(nco):
        ks = list(stp); ks = ks if c % 2 == 0 else ks[::-1]
        for k in ks: pubs.append((V.stack(step, k), ob)); order.append((c, k))
    ep = EstimatorV2(mode=b); V.set_clean(ep.options); ep.options.default_shots = SHOTS; ol = V.options_line(ep.options); print('Estimator options:', ol)
    if ep.options.resilience_level != 0 or ep.options.twirling.enable_gates or ep.options.twirling.enable_measure or ep.options.dynamical_decoupling.enable: sys.exit('REFUSED: clean options not set')
    sp = SamplerV2(mode=b); sp.options.dynamical_decoupling.enable = False; sp.options.twirling.enable_gates = False; sp.options.twirling.enable_measure = False
    jobs_set('stated' + pk, dict(stamp=stamp, cz={str(e): v for e, v in s['cz'].items()}, sx={str(q): v for q, v in s['sx'].items()}, ro={str(q): v for q, v in s['ro'].items()}, chain=chain))
    jp = sp.run([(c, None, RB_SHOTS) for c in rbc] + [(c, None, RO_SHOTS) for c in roc]); jobs_set('P' + pk, dict(id=jp.job_id(), meta=[list(m) for m in meta], n_rb=len(rbc)))
    print('submitted job P', jp.job_id()); V.show_job_options(jp, 'Job P')
    jm = ep.run(pubs); jobs_set('main' + pk, dict(id=jm.job_id(), order=order, options=ol, T=T)); print('submitted main job', jm.job_id(), 'pubs', len(pubs)); V.show_job_options(jm, 'Main job')

def run_fetch(pilot=False):
    pk = 'p' if pilot else ''; svc = V.get_service(); jd = jobs_get(); out = {}
    for key in ('P', 'main'):
        j = svc.job(jd[key + pk]['id']); print(key, j.status())
        if 'DONE' not in str(j.status()): return
        out[key] = j.result()
    n = jd['P' + pk]['n_rb']; cl = [counts_of(pr) for pr in out['P']]
    save('pdata' + pk, dict(rb=[dict(c) for c in cl[:n]], ro=[dict(c) for c in cl[n:]]))
    raw = np.full((PILOT_COPIES if pilot else COPIES, K, 2), np.nan)
    for (c, k), pr in zip(jd['main' + pk]['order'], out['main']): raw[c, k - 1] = [float(x) for x in pr.data.evs]
    save('main' + pk, raw.tolist()); print('saved', FILES['main' + pk], FILES['pdata' + pk])

def run_analyze(pilot=False):
    pk = 'p' if pilot else ''; jd = jobs_get(); pd_ = load('pdata' + pk); raw = np.array(load('main' + pk), dtype=float); stt = jd['stated' + pk]; chain = stt['chain']
    svc, b, _ = V.live_setup(); st, pv = V.transpile_once(b, chain); ze, _ = exact_edge_values(); tro = V.exact_values()
    meta = [tuple(m) for m in jd['P' + pk]['meta']]; rng = np.random.default_rng(SEED); fit = rb_fit(pd_['rb'], meta, rng); ro = ro_fit(pd_['ro'], chain)
    cz_s = [stt['cz'][str(e)] for e in range(N - 1)]; sx_s = {int(q): v for q, v in stt['sx'].items()}; ro_s = {int(q): v for q, v in stt['ro'].items()}
    ksx = sx_per_cz(meta); cz_o = [measured_cz_isolated(fit[e][0], ksx[e], [sx_s[chain[e]], sx_s[chain[e + 1]]]) for e in range(N - 1)]
    print('per edge CZ error: stated / OUR RB +- sigma / ratio   (record stamp %s)' % stt['stamp'])
    for e in range(N - 1): print('  edge %d (%d-%d)  %.5f   %.5f +- %.5f   %.2f   (isolated-CZ figure used by the model %.5f)' % (e, chain[e], chain[e + 1], cz_s[e], fit[e][0], fit[e][1], fit[e][0] / cz_s[e], cz_o[e]))
    print('mean RB/stated ratio %.2f ; mean isolated-CZ/stated %.2f' % (np.mean([fit[e][0] for e in range(N - 1)]) / np.mean(cz_s), np.mean(cz_o) / np.mean(cz_s)))
    print('readout per qubit: stated error / OUR (p01+p10)/2:', [(q, round(ro_s[q], 4), round(sum(ro[q]) / 2, 4)) for q in chain])
    sc = get_scale()
    if sc is None:
        if not pilot: sys.exit('REFUSED: no frozen fade exponent for %s; run --fire --pilot, --fetch --pilot, --analyze --pilot first' % BACKEND)
        nnp = raw[:, :, 1]; ok = ~np.isnan(nnp).any(0); ok[0] = False
        base = model_fades(st, chain, cz_o, sx_s, {q: 1 - sum(ro[q]) for q in chain}, ze)[0]; g = np.linspace(0.02, 1.5, 297)
        sc = float(g[int(np.argmin([np.mean(np.abs(nnp[:, ok] / (base ** x)[ok][None, :] - tro[ok, 1][None, :])) for x in g]))])
        json.dump(dict(scale=sc, source='pilot job %s on %s, calibrated at the pilot job-P measured CZ error' % (jd['mainp']['id'], BACKEND)), open(FILES['scale'], 'w'))
        print('PILOT CALIBRATION (in-sample for the pilot readings; frozen for the full run): fade exponent %.3f saved to %s' % (sc, FILES['scale']))
    print('fade exponent used (frozen): %.3f' % sc)
    fS = model_fades(st, chain, cz_s, sx_s, {q: 1 - 2 * ro_s[q] for q in chain}, ze, sc)[:2]
    fO = model_fades(st, chain, cz_o, sx_s, {q: 1 - sum(ro[q]) for q in chain}, ze, sc)[:2]
    print(analyze3(raw, tro, fS, fO, stt['stamp'])[0])

def run_test():
    """CODE SMOKE TEST ONLY. Offline, no result, no numbers kept. Checks the plumbing of the fade model, RB fit, readout fit and analysis."""
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    from qiskit_aer import AerSimulator
    fb = FakeKingston(); chain = V.pick_chain(fb); print('SMOKE TEST ONLY (code plumbing; no result). chain', chain)
    st, pv = V.transpile_once(fb, chain); V.param_positions(st); ze, _ = exact_edge_values(); tro = V.exact_values()
    (fS, fZ, Fe), s = stated_model(fb, chain, st, ze); assert fS.shape == (K,) and fZ.shape == (K,) and Fe.shape == (K, N - 1) and np.all(np.diff(fS) <= 1e-12)
    print('fade model path executed (shapes ok, monotone)')
    rbc, meta = rb_circuits(fb, chain); sim = AerSimulator(seed_simulator=SEED); keep = [i for i, m in enumerate(meta) if m[0] < 2 and m[1] <= 8]
    cl = [sim.run(rbc[i], shots=100).result().get_counts() for i in keep]; mt = [meta[i] for i in keep]
    fit = rb_fit(cl, mt, np.random.default_rng(SEED), boot=3); print('RB fit path executed on', len(keep), 'circuits; edges fitted', sorted(fit))
    roc = ro_circuits(fb, chain); cts = [sim.run(c, shots=100).result().get_counts() for c in roc]; r = ro_fit(cts, chain); print('readout fit path executed on', len(r), 'qubits')
    rng = np.random.default_rng(SEED); raw = np.stack([np.stack([tro[:, 0], tro[:, 1]], 1) * 0.9 + rng.normal(0, 0.01, (K, 2)) for _ in range(COPIES)])
    print(analyze3(raw, tro, (fS, fZ), (fS ** 1.3, fZ ** 1.3), 'offline', label='[SMOKE] ')[1], '(word from placeholder data; plumbing only)')
    print('plan ratio gate path:', plan_ratio(), '; est_P %.0f est_main %.0f (offline circuits)' % (est_P(rbc), est_main(float(st.estimate_duration(fb.target)))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for m in ('test', 'dry', 'power', 'fire', 'fetch', 'analyze', 'pilot'): ap.add_argument('--' + m, action='store_true')
    a = ap.parse_args()
    if a.test: run_test()
    elif a.dry: run_dry()
    elif a.power: run_power()
    elif a.fire: run_fire(a.pilot)
    elif a.fetch: run_fetch(a.pilot)
    elif a.analyze: run_analyze(a.pilot)
    else: ap.print_help()
