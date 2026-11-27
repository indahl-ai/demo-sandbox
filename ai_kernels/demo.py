#!/usr/bin/env python
"""Demo 6 (AI / machine learning): quantum-kernel SVM, ZZ feature map, fidelity kernel K(x,y) = P(all zeros) of U(x)^dagger U(y).
IBM code Q17 (gate error vs stated) + the readout audit (Q2/Q9/Q11, today's table vs the record).
Five readings of the SAME shots of the client's circuits (one set of circuits, identical for every arm):
  N  raw P(all zeros)                       (the client's kernel)
  S  readout-inverted with the RECORD's symmetric figure, fade rescaled at IBM's STATED per-edge CZ error
  O  readout-inverted with job P's per-qubit table, fade rescaled at OUR job-P RB per-edge CZ error
  (extras, attribution only)  R = O's readout only, no fade     G = O's fade only, raw readout
FADE MODEL (written before data; no parameter fitted on the main shots):
  P_meas(all zeros) = R_ro x [ F K + (1 - F) / 2^n ],   F = product over every CZ of the transpiled circuit of (1 - 4 e_pair / 3)
  (global-depolarizing fade; e_pair is our parallel-RB fit, which bills 1q gates to the CZ and includes crosstalk; it is not IBM's isolated convention).   # corrected 27 Nov 2026 (Sage)
  Estimate: invert readout on the full 2^n outcome distribution with tensor per-qubit confusion matrices, then K_hat = (P0 - (1-F)/2^n) / F.
  Same model for S and O; only the tables differ.
Task: ZZFeatureMap-style (reps 2, linear entanglement, H, P(2x), CX-P(2(pi-xi)(pi-xj))-CX), 8 features on 8 qubits, 16 train + 10 test points,
  labels from a fixed teacher in the same feature space; SVC(kernel='precomputed', C=10), split and C fixed in this file before any data.
Runtime changes OFF (twirling, DD; SamplerV2 has no resilience) in job P and main, same options, read back from each job. Transpile = client default
  (preset level 3, seed 7, layout = the client's own pick from the first circuit, reused for all): one circuit set, hash-asserted, shared by N, S, O.
Modes: --test (code smoke only) --dry --fire [--pilot] --fetch [--pilot] --analyze [--pilot].
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_BACKEND, DEMO_INSTANCE. Keys and CRNs never printed. Nothing here commits or pushes."""
import os, sys, json, math, time, argparse, random, hashlib, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford, Clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("DEMO_BACKEND", "ibm_kingston")
RUN = os.path.join(HERE, "run_" + BACKEND.replace("ibm_", ""))

# ------------------------------------------------------------ the client's task (fixed before data; ideal circuit only)
NQ, REPS, NTRAIN, NTEST = 8, 2, 16, 10
XSCALE = 0.18          # feature range [0, XSCALE]: keeps exact kernel entries well above shot noise (set from the ideal kernel only)
SEED = 20261121
SVM_C = 10.0
NKEEP = 2 ** NQ
# ------------------------------------------------------------ design (fixed before data)
CFG = {"full": dict(copies=8, shots=150, lengths=[1, 3, 6, 10, 15], seqs=3, rbshots=300, ro8shots=6000, roallshots=3000, ntrain=NTRAIN, ntest=NTEST, mainjobs="per_copy"),
       "pilot": dict(copies=4, shots=100, lengths=[1, 3, 6, 10], seqs=2, rbshots=200, ro8shots=2000, roallshots=1000, ntrain=12, ntest=0, mainjobs="one")}
MAX_SECONDS, MARGIN, PILOT_SECONDS = 200.0, 1.25, 30.0
SEC_PER_SHOT_CIRCUIT, SEC_PER_JOB = 0.30e-3, 4.0   # seconds per shot-circuit and per job, from earlier recorded jobs
RHOS = (1.30, 1.56, 1.82)   # measured/stated CZ error ratios used in the design
EQUIV_MAE, EQUIV_ACC = 0.005, 0.10   # CONSISTENT bands, set before data (0.005 = under a fifth of the per-entry shot-noise floor; 0.10 = one test point of ten)
NDRAW, NMC = 60, 400
ARMS = ("N", "S", "O", "R", "G")
LABEL = {"N": "raw (the client's kernel)", "S": "stated (record readout + stated CZ fade)", "O": "ours (job-P readout + job-P RB CZ fade)",
         "R": "extra: our readout only", "G": "extra: our CZ fade only"}


# ------------------------------------------------------------ service helpers (same as the other demos)
def read_key():
    tok = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if not tok: sys.exit("Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)")
    return tok


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService as S
    tok = read_key()
    base = S(channel="ibm_quantum_platform", token=tok)
    import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); import pick_instance
    crn = pick_instance.crn(base)
    return S(channel="ibm_quantum_platform", token=tok, instance=crn)


def usage_remaining(svc):
    try: return float(svc.usage().get("usage_remaining_seconds"))
    except Exception as e: sys.exit("cannot read usage_remaining_seconds (%s); refusing" % type(e).__name__)


def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
    return FakeFez() if "fez" in BACKEND else FakeMarrakesh() if "marrakesh" in BACKEND else FakeKingston()


def clean_sampler(be):
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=be); o = s.options
    o.dynamical_decoupling.enable = False; o.twirling.enable_gates = False; o.twirling.enable_measure = False
    return s


def assert_clean(s):
    o = s.options
    if o.twirling.enable_gates or o.twirling.enable_measure or o.dynamical_decoupling.enable: sys.exit("REFUSED: runtime changes are not all OFF")


def options_line(o):
    return "twirling.enable_gates=%s twirling.enable_measure=%s dynamical_decoupling.enable=%s resilience=none (SamplerV2 has no resilience) default_shots=%s" % (
        o.twirling.enable_gates, o.twirling.enable_measure, o.dynamical_decoupling.enable, o.default_shots)


def show_job_options(job, label):
    try:
        inp = job.inputs; o = inp.get("options", inp) if isinstance(inp, dict) else inp
        print("%s options read back from the job: %s" % (label, json.dumps(o, default=str)[:700]))
    except Exception as ex: print("%s: could not read options back from the job (%s)" % (label, type(ex).__name__))


def record_stamp(be):
    try: return str(be.properties().last_update_date)
    except Exception: return None


def fp(n): os.makedirs(RUN, exist_ok=True); return os.path.join(RUN, n)
def s_load(sfx): p = fp("state%s.json" % sfx); return json.load(open(p)) if os.path.exists(p) else {}
def s_save(sfx, d): cur = s_load(sfx); cur.update(d); json.dump(cur, open(fp("state%s.json" % sfx), "w"), indent=1)
def pair(a, b): return (a, b) if a < b else (b, a)


# ------------------------------------------------------------ feature map, data, exact kernel
def feature_circuit(x):
    qc = QuantumCircuit(NQ)
    for _ in range(REPS):
        qc.h(range(NQ))
        for i in range(NQ): qc.p(2 * x[i], i)
        for i in range(NQ - 1):
            qc.cx(i, i + 1); qc.p(2 * (math.pi - x[i]) * (math.pi - x[i + 1]), i + 1); qc.cx(i, i + 1)
    return qc


def states(X): return [Statevector(feature_circuit(x)).data for x in X]


def exact_matrix(X):
    S = np.array(states(X)); G = S.conj() @ S.T
    return np.abs(G) ** 2


def make_data():
    """Teacher labels in the same feature space, margin-filtered (the usual quantum-kernel benchmark construction): 300 candidates, keep the 13 most
    decisive per class around the median of f, shuffle. Built from the ideal circuit only; no hardware number enters."""
    rng = np.random.default_rng(SEED); n = NTRAIN + NTEST
    C = rng.uniform(0, XSCALE, (300, NQ)); Z = rng.uniform(0, XSCALE, (2, NQ)); w = rng.choice([-1.0, 1.0], 2) * rng.uniform(0.5, 1.5, 2)
    f = (np.abs(np.array(states(C)).conj() @ np.array(states(Z)).T) ** 2) @ w; m = np.median(f)
    pos = [i for i in np.argsort(-(f - m)) if f[i] > m][:n // 2]; neg = [i for i in np.argsort(f - m) if f[i] < m][:n // 2]
    idx = np.array(pos + neg); rng.shuffle(idx); X = C[idx]; y = (f[idx] > m).astype(int)
    if len(set(y[:NTRAIN])) < 2 or len(set(y[NTRAIN:])) < 2: sys.exit("REFUSED: a split has one class (fix SEED before any data)")
    return X, y


def entries(ntrain, ntest):
    """(a, b) off-diagonal entries actually run: train-train upper triangle, then test x train. Indices: train 0..ntrain-1, test ntrain..."""
    E = [(i, j) for i in range(ntrain) for j in range(i + 1, ntrain)]
    E += [(ntrain + t, j) for t in range(ntest) for j in range(ntrain)]
    return E


def kernel_circuit(x, y):
    qc = QuantumCircuit(NQ, NQ); qc.compose(feature_circuit(y), inplace=True); qc.compose(feature_circuit(x).inverse(), inplace=True)
    qc.measure(range(NQ), range(NQ)); return qc


# ------------------------------------------------------------ transpile (client default), counts per pair
BASIS_OK = {"cz", "sx", "rz", "x", "id", "measure", "barrier", "delay"}


def transpile_set(be, X, E):
    qcs = [kernel_circuit(X[a], X[b]) for a, b in E]
    pm0 = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=7)
    t0 = pm0.run(qcs[0]); lay = [int(q) for q in t0.layout.initial_index_layout(filter_ancillas=True)][:NQ]
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, initial_layout=lay, seed_transpiler=7)
    tcs = pm.run(qcs)
    for tc in tcs:
        bad = set(tc.count_ops()) - BASIS_OK
        if bad: sys.exit("REFUSED: transpiled circuit has ops %s" % sorted(bad))
    return tcs, lay


def cz_counts(tcs):
    """returns pairs (sorted list), NCZ (n_circuits, n_pairs)"""
    per = []
    for tc in tcs:
        d = {}
        for ci in tc.data:
            if ci.operation.name == "cz":
                k = pair(*(tc.find_bit(q).index for q in ci.qubits)); d[k] = d.get(k, 0) + 1
        per.append(d)
    pairs = sorted({k for d in per for k in d})
    return pairs, np.array([[d.get(p, 0) for p in pairs] for d in per], float)


def circuit_hash(tcs):
    h = hashlib.sha256()
    for tc in tcs:
        for ci in tc.data: h.update(("%s%s%s" % (ci.operation.name, [tc.find_bit(q).index for q in ci.qubits], [round(float(p), 9) for p in ci.operation.params if isinstance(p, (int, float))])).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------ record
def stated_errors(be, prs):
    out = {}
    for p in prs:
        for q in (p, p[::-1]):
            try:
                e = be.target["cz"][q].error
                if e is not None: out[p] = float(e); break
            except Exception: pass
        else: sys.exit("REFUSED: no stated CZ error in the record for pair %s" % (p,))
    return out


def record_readout(be):
    t = be.target; return np.array([t["measure"][(q,)].error if (q,) in t["measure"] and t["measure"][(q,)].error is not None else np.nan for q in range(be.num_qubits)])


# ------------------------------------------------------------ job P circuits
def colour_pairs(prs):
    cols = []
    for p in prs:
        for c in cols:
            if all(set(p).isdisjoint(q) for q in c): c.append(p); break
        else: cols.append([p])
    return cols


def rb_circuits(be, cols, cfg, seed=SEED):
    rng = random.Random(seed); circs, meta = [], []
    for k, prs in enumerate(cols):
        flat = [q for p in prs for q in p]
        for m in cfg["lengths"]:
            for s in range(cfg["seqs"]):
                qc = QuantumCircuit(2 * len(prs), 2 * len(prs))
                for i in range(len(prs)):
                    sub = QuantumCircuit(2); comp = Clifford(QuantumCircuit(2))
                    for _ in range(m):
                        c = random_clifford(2, seed=rng.randrange(10 ** 9)); sub.compose(c.to_circuit(), inplace=True); comp = comp.compose(c)
                    sub.compose(comp.adjoint().to_circuit(), inplace=True); qc.compose(sub, qubits=[2 * i, 2 * i + 1], inplace=True)
                qc.measure(range(2 * len(prs)), range(2 * len(prs)))
                from qiskit import transpile
                tc = transpile(qc, be, initial_layout=flat, optimization_level=0, seed_transpiler=SEED)
                cnt = {}
                for ci in tc.data:
                    if ci.operation.name == "cz":
                        key = pair(*(tc.find_bit(q).index for q in ci.qubits)); cnt[key] = cnt.get(key, 0) + 1
                circs.append(tc); meta.append(dict(col=k, m=m, s=s, ncz=[cnt.get(p, 0) for p in prs]))
    return circs, meta


def ro_circuits(be, lay):
    """8 layout qubits only (the main job's own measurement context) and all qubits (chip-wide audit)."""
    from qiskit import transpile
    out = []
    for ones in (False, True):
        qc = QuantumCircuit(NQ, NQ)
        if ones: qc.x(range(NQ))
        qc.measure(range(NQ), range(NQ)); out.append(transpile(qc, be, initial_layout=lay, optimization_level=0, seed_transpiler=SEED))
    n = be.num_qubits
    for ones in (False, True):
        qc = QuantumCircuit(n, n)
        if ones: qc.x(range(n))
        qc.measure(range(n), range(n)); out.append(transpile(qc, be, initial_layout=list(range(n)), optimization_level=0, seed_transpiler=SEED))
    return out


def counts_of(pub_result):
    d = pub_result.data
    reg = [getattr(d, a) for a in dir(d) if not a.startswith("_") and hasattr(getattr(d, a), "get_counts")][0]
    return reg.get_counts()


def dense(counts, nb=NQ):
    v = np.zeros(2 ** nb, np.int64)
    for s, c in counts.items(): v[int(s.replace(" ", ""), 2)] += c
    return v


def bit_matrix(nb): return ((np.arange(2 ** nb)[:, None] >> np.arange(nb)[None, :]) & 1).astype(int)   # bit q = clbit q


def p0(counts_arr_bits_zero): return counts_arr_bits_zero


def pair_k00(counts, nclb, npairs):
    """per pair survival counts of both qubits 0, from a counts dict over nclb clbits"""
    out = np.zeros(npairs, np.int64)
    for s, c in counts.items():
        s = s.replace(" ", "")[::-1]
        for i in range(npairs):
            if s[2 * i] == "0" and s[2 * i + 1] == "0": out[i] += c
    return out


def table_from_counts(c0, c1, nb, shots):
    """c0/c1: dense count vectors for prep 0 / prep 1. returns p01 (read 1 | prep 0), p10 (read 0 | prep 1), shots"""
    B = bit_matrix(nb)
    return (c0 @ B) / shots, (c1 @ (1 - B)) / shots


# ------------------------------------------------------------ RB fit
RGRID = np.linspace(0.95, 1.0, 1001)


def rb_fit_many(xs, Y):
    """xs (P,) CZ counts; Y (D,P) survival fractions. P00 = 0.25 + A r^n, A closed form per r (clipped 0.5-0.8); e = 0.75 (1-r)."""
    X = RGRID[:, None] ** xs[None, :]; den = (X ** 2).sum(1)
    A = np.clip(((Y - 0.25) @ X.T) / den, 0.5, 0.8)
    sse = ((0.25 + A[:, :, None] * X[None] - Y[:, None, :]) ** 2).sum(2)
    return np.maximum(0.75 * (1 - RGRID[np.argmin(sse, 1)]), 1e-4)


def measured_errors(cols, meta, K, shots, rng=None, draws=0):
    point, boot = {}, {}
    for k, prs in enumerate(cols):
        idx = [i for i, m in enumerate(meta) if m["col"] == k]
        for j, p in enumerate(prs):
            xs = np.array([meta[i]["ncz"][j] for i in idx], float); y = np.array([K[i][j] / shots for i in idx])
            point[p] = float(rb_fit_many(xs, y[None])[0])
            if draws: boot[p] = rb_fit_many(xs, rng.binomial(shots, np.clip(y, 0, 1)[None].repeat(draws, 0)) / shots)
    return point, boot


# ------------------------------------------------------------ estimators
def readout_weights(p01, p10):
    """w(x) = prod_q Ainv_q[0, x_q]; sum_x P(x) w(x) is the readout-inverted P(all zeros)."""
    B = bit_matrix(len(p01)); den = 1 - p01 - p10
    if np.any(den < 0.5): sys.exit("REFUSED: denominator guard 1-p01-p10 < 0.5")
    a, b = (1 - p10) / den, -p10 / den
    return np.prod(np.where(B == 0, a[None], b[None]), axis=1)


def fade_of(NCZ, errs, prs):
    e = np.array([errs[p] for p in prs]); return np.exp(NCZ @ np.log(1 - 4 * e / 3))


def estimates(cnt, shots, NCZ, prs, rec_ro, tabO, eS, eO):
    """cnt (C, N, 2^NQ). returns {arm: (C, N)} kernel estimates."""
    P = cnt / shots
    wS = readout_weights(rec_ro, rec_ro); wO = readout_weights(*tabO)
    FS, FO = fade_of(NCZ, eS, prs), fade_of(NCZ, eO, prs); off = lambda F: (1 - F) / NKEEP
    out = {"N": P[..., 0]}
    out["S"] = (P @ wS - off(FS)) / FS; out["O"] = (P @ wO - off(FO)) / FO
    out["R"] = P @ wO; out["G"] = (P[..., 0] - off(FO)) / FO
    return out


def to_matrix(vals, E, n):
    M = np.eye(n)
    for v, (a, b) in zip(vals, E): M[a, b] = M[b, a] = v
    return M


def tq(n):
    from scipy import stats
    return float(stats.t.ppf(0.995, n - 1))


def metrics(Kest, E, Kex, y, ntrain, ntest):
    """Kest: (C, N) estimates in entry order E. returns dict of per-copy arrays: mae, frob, acc"""
    from sklearn.svm import SVC
    n = ntrain + ntest; ex = np.array([Kex[a, b] for a, b in E]); C = Kest.shape[0]
    mae = np.abs(Kest - ex).mean(1); acc = np.zeros(C); frob = np.zeros(C)
    ti = [k for k, (a, b) in enumerate(E) if b < ntrain and a < ntrain]
    for c in range(C):
        M = to_matrix(Kest[c], E, n); frob[c] = np.sqrt(((M[:ntrain, :ntrain] - Kex[:ntrain, :ntrain]) ** 2).sum())
        if ntest:
            m = SVC(kernel="precomputed", C=SVM_C).fit(M[:ntrain, :ntrain], y[:ntrain]); acc[c] = (m.predict(M[ntrain:, :ntrain]) == y[ntrain:]).mean()
        else: acc[c] = np.nan
    return dict(mae=mae, frob=frob, acc=acc)


def word(lo, hi, band):
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "CONSISTENT" if (-band < lo and hi < band) else "CANNOT_TELL"


def paired(a, b, band, sdP=0.0):
    d = np.asarray(a) - np.asarray(b); n = len(d); m = d.mean()
    half = tq(n) * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    half = math.sqrt(half ** 2 + (2.576 * sdP) ** 2); lo, hi = m - half, m + half
    return m, lo, hi, word(lo, hi, band), half


def pooled_mae_boot(cnt, shots, NCZ, prs, rec_ro, eS, E, Kex, state, rng):
    """P-table uncertainty: redraw job P (RB survival, readout counts) NDRAW times; return per-draw (MAE_O - MAE_N, MAE_O - MAE_S) on the copy-pooled matrix."""
    ex = np.array([Kex[a, b] for a, b in E]); cols = [[tuple(p) for p in c] for c in state["cols"]]
    K = [np.array(k) for k in state["rb_k00"]]; _, boot = measured_errors(cols, state["meta"], K, state["cfg"]["rbshots"], rng, NDRAW)
    t = state["tab8"]; n0, n1 = t["n0"], t["n1"]
    pc = cnt.sum(0, keepdims=True); sp = shots * cnt.shape[0]; tabC = (np.array(t["p01"]), np.array(t["p10"]))
    N = pc[0, :, 0] / sp; Sr = estimates(pc, sp, NCZ, prs, rec_ro, tabC, eS, {p: float(np.mean(boot[p])) for p in prs})["S"][0]
    maeN, maeS = np.abs(N - ex).mean(), np.abs(Sr - ex).mean(); dON, dOS = [], []
    for d in range(NDRAW):
        p01 = rng.binomial(n0, np.array(t["p01"])) / n0; p10 = rng.binomial(n1, np.array(t["p10"])) / n1
        eO = {p: float(boot[p][d]) for p in prs}
        O = estimates(pc, sp, NCZ, prs, rec_ro, (p01, p10), eS, eO)["O"][0]
        m = np.abs(O - ex).mean(); dON.append(m - maeN); dOS.append(m - maeS)
    return float(np.std(dON, ddof=1)), float(np.std(dOS, ddof=1))


# ------------------------------------------------------------ design model (hardware-sourced ratios; a model never drops a design)
def model_power(NCZ, prs, stated, rec_ro8, Kex, E, cfg, rho, rng):
    ex = np.array([Kex[a, b] for a, b in E]); C, S = cfg["copies"], cfg["shots"]
    e = {p: stated[p] * rho for p in prs}; F = fade_of(NCZ, e, prs); Fs = fade_of(NCZ, stated, prs)
    R = float(np.prod(1 - rec_ro8)); Pn = R * (F * ex + (1 - F) / NKEEP)
    cnt = rng.binomial(S, np.clip(Pn, 0, 1)[None, None].repeat(NMC, 0).repeat(C, 1)) / S
    Nh = cnt; Oh = (cnt / R - (1 - F) / NKEEP) / F; Sh = (cnt / R - (1 - Fs) / NKEEP) / Fs
    mN, mO, mS = (np.abs(x - ex).mean(-1) for x in (Nh, Oh, Sh))      # (NMC, C)
    out = {}
    for nm, d in (("ON", mN - mO), ("OS", mS - mO)):
        hw = tq(C) * d.std(1, ddof=1) / math.sqrt(C); out[nm] = (float(abs(d.mean())), float(np.median(hw)), float(abs(d.mean()) / np.median(hw)))
    return out


# ------------------------------------------------------------ prepare (everything computed offline from the record; no hardware numbers invented)
def prepare(be, cfg):
    X, y = make_data(); E = entries(cfg["ntrain"], cfg["ntest"]); Xs = X[:cfg["ntrain"]] if not cfg["ntest"] else X
    tcs, lay = transpile_set(be, X, E); prs, NCZ = cz_counts(tcs); Kex = exact_matrix(X)
    return dict(X=X, y=y, E=E, tcs=tcs, lay=lay, prs=prs, NCZ=NCZ, Kex=Kex, cols=colour_pairs(prs), hash=circuit_hash(tcs))


def est_p(cfg, ncirc_rb):
    shotc = ncirc_rb * cfg["rbshots"] + 2 * cfg["ro8shots"] + 2 * cfg["roallshots"]
    return SEC_PER_SHOT_CIRCUIT * shotc + SEC_PER_JOB


def est_main(cfg, ncirc):
    jobs = cfg["copies"] if cfg["mainjobs"] == "per_copy" else 1
    return SEC_PER_SHOT_CIRCUIT * cfg["copies"] * ncirc * cfg["shots"] + SEC_PER_JOB * jobs


def sources_rows(be, x, pjob, stated, hashv):
    stamp = record_stamp(be)
    return [("stated CZ error per pair (RB average infidelity)", "live record %s, target cz error, %d pairs" % (stamp, len(stated)) if stamp and stated else "MISSING"),
            ("record readout per qubit (symmetric)", "live record %s, target measure error" % stamp if stamp else "MISSING"),
            ("measured CZ error per pair", "job P %s (parallel RB in colour layers, same fit and fade form as kingston 19 Nov finance jobs dbv6lk7f8rhs73bncf00/dbv6lkdarooc73fl1bn0)" % pjob if pjob else "MISSING"),
            ("today's readout table, layout qubits", "job P %s (all-0/all-1 on the 8 layout qubits only, the main job's own measurement context)" % pjob if pjob else "MISSING"),
            ("chip-wide readout audit (Q2/Q9/Q11)", "job P %s (all-0/all-1 on every qubit) vs the record" % pjob if pjob else "MISSING"),
            ("fade model, 1 - 4e/3 per CZ, global depolarizing offset", "written before data; no fitted parameter; counts per pair from the transpiled circuits (hash %s)" % hashv),
            ("design ratios measured/stated CZ error 1.30, 1.56, 1.82", "kingston RB Oct (RB_RATIO 1.305); earlier finance jobs"),
            ("readout record-vs-today gaps (design)", "earlier optimization and error-correction jobs"),
            ("layout, transpile, task data, SVM C", "client's own default (preset level 3, seed 7) and fixed task: not our parameters")]


def print_sources(rows):
    print("SOURCES (parameter -> hardware source):")
    for r in rows: print("  %-56s %s" % r)


def power_lines(x, stated, rec8, cfg, label=""):
    rng = np.random.default_rng(SEED); rows = []
    for rho in RHOS:
        r = model_power(x["NCZ"], x["prs"], stated, rec8, x["Kex"], x["E"], cfg, rho, rng); rows.append((rho, r))
        print("%smodel rho %.2f x stated: expected |O-N| %.4f MAE units over paired 99%% floor %.4f -> ratio %.1f;  |O-S| %.4f over %.4f -> ratio %.1f" % (label, rho, r["ON"][0], r["ON"][1], r["ON"][2], r["OS"][0], r["OS"][1], r["OS"][2]))
    return rows


# ------------------------------------------------------------ modes
def mode_test():
    from qiskit_aer import AerSimulator
    be = fake_backend(); cfg = dict(CFG["full"]); print("[SMOKE TEST ONLY: code paths on a fake backend; no number here is a result]")
    x = prepare(be, cfg); print("layout %s; %d entries; pairs %d; cz per circuit %d to %d; hash %s" % (x["lay"], len(x["E"]), len(x["prs"]), x["NCZ"].sum(1).min(), x["NCZ"].sum(1).max(), x["hash"]))
    sim = AerSimulator(method="matrix_product_state"); rng = np.random.default_rng(1); S = 4000; idx = list(range(0, len(x["E"]), 23))
    z = []
    for i in idx:
        c = sim.run(x["tcs"][i], shots=S, seed_simulator=3 + i).result().get_counts(); p = c.get("0" * NQ, 0) / S; e = x["Kex"][x["E"][i]]
        z.append((p - e) / math.sqrt(max(e * (1 - e), 1e-6) / S))
    print("transpiled circuits vs exact kernel (ideal MPS, bit order / layout check): max |z| = %.1f over %d entries -> %s" % (max(abs(np.array(z))), len(z), "OK" if max(abs(np.array(z))) < 4.5 else "MISMATCH"))
    stated = stated_errors(be, x["prs"]); rec = record_readout(be); rec8 = rec[x["lay"]]
    print("power code path:"); power_lines(x, stated, rec8, cfg, "  ")
    # RB code path
    circs, meta = rb_circuits(be, x["cols"], cfg); print("RB circuits %d (colour layers %s)" % (len(circs), [len(c) for c in x["cols"]]))
    K = []
    for i, tc in enumerate(circs[:len(circs)]):
        c = sim.run(tc, shots=cfg["rbshots"], seed_simulator=5 + i).result().get_counts(); K.append(pair_k00(c, tc.num_clbits, len(x["cols"][meta[i]["col"]])))
    pt, _ = measured_errors(x["cols"], meta, K, cfg["rbshots"], np.random.default_rng(2), 3); print("RB fit ran on ideal counts: %d pairs fitted (code path only)" % len(pt))
    # analysis path on synthetic counts (ideal MPS, 2 copies, subset) to run estimators/metrics/SVM
    sub = list(range(0, len(x["E"]), 5)); cnt = np.zeros((2, len(x["E"]), NKEEP), np.int64)
    for cp in range(2):
        for i in range(len(x["E"])):
            if i in sub or True:
                c = sim.run(x["tcs"][i], shots=50, seed_simulator=100 * cp + i).result().get_counts(); cnt[cp, i] = dense(c)
    eS = stated; eO = {p: v * 1.3 for p, v in stated.items()}; tab = (rec8 * 0.9, rec8 * 1.1)
    est = estimates(cnt, 50, x["NCZ"], x["prs"], rec8, tab, eS, eO)
    for a in ARMS: m = metrics(est[a], x["E"], x["Kex"], x["y"], NTRAIN, NTEST)
    print("estimators, metrics and SVM ran for arms %s" % ",".join(ARMS))
    print("options asserted off in code: %s" % ("clean_sampler needs a live backend (checked in --dry)"))
    print("smoke test done")


def mode_dry():
    svc = get_service(); u = usage_remaining(svc); be = svc.backend(BACKEND)
    print("backend %s operational=%s; usage_remaining_seconds=%.0f" % (BACKEND, be.status().operational, u))
    cfg = CFG["full"]; x = prepare(be, cfg); stamp = record_stamp(be)
    print("client's layout (default level 3 pick): %s; %d entries (%d train-train, %d test-train); %d native CZ pairs used; colour layers %s" % (x["lay"], len(x["E"]), len(entries(NTRAIN, 0)), NTEST * NTRAIN, len(x["prs"]), [len(c) for c in x["cols"]]))
    ncz = x["NCZ"].sum(1); print("CZ per circuit: min %d median %d max %d; depth max %d; transpile hash %s" % (ncz.min(), np.median(ncz), ncz.max(), max(tc.depth() for tc in x["tcs"]), x["hash"]))
    stated = stated_errors(be, x["prs"]); ev = np.array(list(stated.values())); rec = record_readout(be); rec8 = rec[x["lay"]]
    print("record stamp %s; stated CZ error over %d pairs: min %.5f median %.5f max %.5f; record readout on the 8 layout qubits: %s" % (stamp, len(ev), ev.min(), np.median(ev), ev.max(), np.round(rec8, 4).tolist()))
    print("exact kernel (ideal circuit, the target): off-diagonal mean %.3f, min %.4f, max %.3f; exact-kernel SVM test accuracy %.2f" % (
        np.mean([x["Kex"][a, b] for a, b in x["E"]]), min(x["Kex"][a, b] for a, b in x["E"]), max(x["Kex"][a, b] for a, b in x["E"]),
        metrics(np.array([[x["Kex"][a, b] for a, b in x["E"]]]), x["E"], x["Kex"], x["y"], NTRAIN, NTEST)["acc"][0]))
    for lab in ("P", "main"):
        s = clean_sampler(be); assert_clean(s); print("Sampler options for job %s: %s" % (lab, options_line(s.options)))
    rbc, meta = rb_circuits(be, x["cols"], cfg); ep, em = est_p(cfg, len(rbc)), est_main(cfg, len(x["E"]))
    print("job P: %d RB circuits x %d shots + readout (8 layout qubits 2 x %d shots, all %d qubits 2 x %d shots); main: %d copies x %d circuits x %d shots in %d jobs of %d PUBs" % (
        len(rbc), cfg["rbshots"], cfg["ro8shots"], be.num_qubits, cfg["roallshots"], cfg["copies"], len(x["E"]), cfg["shots"], cfg["copies"], len(x["E"])))
    tot = (ep + em) * MARGIN; print("estimated seconds: P %.1f + main %.1f = %.1f (x%.2f = %.1f; cap %d; remaining %.0f)" % (ep, em, ep + em, MARGIN, tot, MAX_SECONDS, u))
    pc = CFG["pilot"]; pshift = est_p(pc, len(rb_circuits(be, x["cols"], pc)[0])) + est_main(pc, len(entries(pc["ntrain"], 0)))
    print("pilot estimate (%d copies x %d entries x %d shots, 12 train points): %.1f s x%.2f = %.1f (cap %d)" % (pc["copies"], len(entries(pc["ntrain"], 0)), pc["shots"], pshift, MARGIN, pshift * MARGIN, PILOT_SECONDS))
    rows = power_lines(x, stated, rec8, cfg); pr = rows[0][1]["ON"][2]
    print("POWER RATIO: %.1f" % pr)
    print("(= expected |O-N| in mean-absolute-kernel-error units at rho 1.30, the low end of our kingston RB range, readout at the record (no gap assumed), over the paired 99%% half-width from %d copies x %d shots; %s)" % (cfg["copies"], cfg["shots"], "model passes 3" if pr >= 3 else "PILOT NEEDED (model under 3; a model number never drops a design)"))
    print("O vs S at rho 1.30: ratio %.1f" % rows[0][1]["OS"][2])
    print_sources(sources_rows(be, x, None, stated, x["hash"]))
    if tot > MAX_SECONDS or tot > u: print("WOULD REFUSE")


def submit_all(be, x, cfg, sfx, stated, pilot):
    rbc, meta = rb_circuits(be, x["cols"], cfg); roc = ro_circuits(be, x["lay"])
    s_save(sfx, dict(lay=x["lay"], E=[list(e) for e in x["E"]], prs=[list(p) for p in x["prs"]], cols=[[list(p) for p in c] for c in x["cols"]], meta=meta, cfg=cfg,
                     stated={"%d-%d" % p: v for p, v in stated.items()}, record_stamp=record_stamp(be), hash=x["hash"], fired_at=time.time(), pilot=pilot))
    sp = clean_sampler(be); assert_clean(sp)
    pubs = [(c, None, cfg["rbshots"]) for c in rbc] + [(roc[0], None, cfg["ro8shots"]), (roc[1], None, cfg["ro8shots"]), (roc[2], None, cfg["roallshots"]), (roc[3], None, cfg["roallshots"])]
    jp = sp.run(pubs); s_save(sfx, dict(job_p=jp.job_id(), p_options=options_line(sp.options))); print("job P submitted", jp.job_id()); show_job_options(jp, "P")
    rows = sources_rows(be, x, jp.job_id(), stated, x["hash"]); print_sources(rows)
    if any("MISSING" in r[1] for r in rows): sys.exit("REFUSED: no hardware source for a parameter")
    jobs = []
    groups = [[tc for _ in range(cfg["copies"]) for tc in x["tcs"]]] if cfg["mainjobs"] == "one" else [list(x["tcs"]) for _ in range(cfg["copies"])]
    for g in groups:
        sm = clean_sampler(be); assert_clean(sm); jm = sm.run([(tc, None, cfg["shots"]) for tc in g]); jobs.append(jm.job_id()); s_save(sfx, dict(jobs_main=jobs, main_options=options_line(sm.options)))
    print("main jobs submitted (%d)" % len(jobs)); show_job_options(jm, "main")
    s_save(sfx, dict(sources=rows))


def save_p(sfx, r, state):
    cfg = state["cfg"]; cols = [[tuple(p) for p in c] for c in state["cols"]]; nrb = len(state["meta"]); K = []
    for i in range(nrb): K.append(pair_k00(counts_of(r[i]), 2 * len(cols[state["meta"][i]["col"]]), len(cols[state["meta"][i]["col"]])).tolist())
    c8 = [dense(counts_of(r[nrb + i])) for i in range(2)]
    # chip-wide tables: counts dict -> per-qubit tallies directly (2^156 outcomes cannot be dense)
    n = None; tabs = []
    for i, shots in ((0, cfg["roallshots"]), (1, cfg["roallshots"])):
        cd = counts_of(r[nrb + 2 + i]); nb = len(next(iter(cd)).replace(" ", "")); n = nb
        tally = np.zeros(nb, np.int64)
        for s, c in cd.items():
            s = s.replace(" ", "")[::-1]; tally += np.array([int(ch) for ch in s], np.int64) * c
        tabs.append(tally)
    p01_all = tabs[0] / cfg["roallshots"]; p10_all = 1 - tabs[1] / cfg["roallshots"]
    p01, p10 = table_from_counts(c8[0], c8[1], NQ, cfg["ro8shots"])
    s_save(sfx, dict(rb_k00=K, tab8=dict(p01=p01.tolist(), p10=p10.tolist(), n0=cfg["ro8shots"], n1=cfg["ro8shots"]), taball=dict(p01=p01_all.tolist(), p10=p10_all.tolist(), n=cfg["roallshots"])))


def fetch(sfx):
    st = s_load(sfx); svc = get_service(); cfg = st["cfg"]; done = True
    if "rb_k00" not in st:
        j = svc.job(st["job_p"]); s = str(j.status()); print("job_p", s)
        if "DONE" in s.upper(): save_p(sfx, j.result(), st); show_job_options(j, "P"); print("saved job P")
        else: done = False
    if not os.path.exists(fp("raw_main%s.npz" % sfx)):
        parts = []
        for k, jid in enumerate(st["jobs_main"]):
            j = svc.job(jid); s = str(j.status()); print("main job %d" % k, s)
            if "DONE" not in s.upper(): done = False; continue
            r = j.result(); parts.append(np.array([dense(counts_of(pr)) for pr in r]))
        if done or len(parts) == len(st["jobs_main"]):
            flat = np.concatenate(parts) if len(parts) > 1 else parts[0]; N = len(st["E"])
            cnt = flat.reshape(cfg["copies"], N, NKEEP) if len(parts) == 1 else np.stack(parts)
            np.savez_compressed(fp("raw_main%s.npz" % sfx), cnt=cnt); show_job_options(j, "main"); print("saved raw_main")
        else: done = False
    return done


def analyze(sfx):
    st = s_load(sfx); cfg = st["cfg"]; svc = get_service(); be = svc.backend(BACKEND); pilot = bool(sfx)
    x = prepare(be, dict(ntrain=cfg["ntrain"], ntest=cfg["ntest"]))
    if x["hash"] != st["hash"]: print("WARNING: transpile hash today %s differs from fired %s (record moved); counts-per-pair from today's transpile are NOT the fired circuits" % (x["hash"], st["hash"]))
    cnt = np.load(fp("raw_main%s.npz" % sfx))["cnt"]; C, shots = cfg["copies"], cfg["shots"]; E = [tuple(e) for e in st["E"]]; prs = [tuple(p) for p in st["prs"]]
    cols = [[tuple(p) for p in c] for c in st["cols"]]; NCZ = x["NCZ"]
    stated = {tuple(int(v) for v in k.split("-")): e for k, e in st["stated"].items()}
    eO, _ = measured_errors(cols, st["meta"], [np.array(k) for k in st["rb_k00"]], cfg["rbshots"]); t8 = st["tab8"]; tabO = (np.array(t8["p01"]), np.array(t8["p10"]))
    rec = record_readout(be); rec8 = rec[st["lay"]]; ntr, nte = cfg["ntrain"], cfg["ntest"]; print("backend=%s record stamp fired %s; transpile hash %s; options main: %s" % (BACKEND, st.get("record_stamp"), st["hash"], st.get("main_options")))
    ra = st["taball"]; z = 2.576; flags = []
    for q in range(len(ra["p01"])):
        for nm, v in (("P(1|0)", ra["p01"][q]), ("P(0|1)", ra["p10"][q])):
            lo = v - z * math.sqrt(max(v * (1 - v), 1e-9) / ra["n"])
            if not np.isnan(rec[q]) and lo > rec[q]: flags.append((v / max(rec[q], 1e-4), q, nm, rec[q], v, lo))
    print("readout audit, IBM codes Q2/Q9/Q11 (today worse than the record beyond job P's 99%% bound): %d of %d qubits" % (len({f[1] for f in flags}), len(ra["p01"])))
    for r_, q, nm, rv, v, lo in sorted(flags, reverse=True)[:12]: print("  q%d %s: record %.4f, today %.4f (lower 99%% bound %.4f), %.1fx%s" % (q, nm, rv, v, lo, r_, "   <- in the kernel's layout" if q in st["lay"] else ""))
    inl = [(q, nm) for _, q, nm, *_ in flags if q in st["lay"]]; print("  layout qubits flagged: %s" % (inl or "none"))
    prsd = [p for p in prs]; print("pair: stated / RB-measured / ratio (job P):", [(p, round(stated[p], 5), round(eO[p], 5), round(eO[p] / stated[p], 2)) for p in prsd])
    est = estimates(cnt, shots, NCZ, prsd, rec8, tabO, stated, eO)
    ex = np.array([x["Kex"][a, b] for a, b in E]); print("exact kernel (ideal circuit, the target): off-diagonal mean %.3f; kernel entries run %d; %d copies x %d shots" % (ex.mean(), len(E), C, shots))
    mt = {a: metrics(est[a], E, x["Kex"], x["y"], ntr, nte) for a in ARMS}; sdON, sdOS = pooled_mae_boot(cnt, shots, NCZ, prsd, rec8, stated, E, x["Kex"], st, np.random.default_rng(SEED))
    if nte: ac_ex = metrics(np.array([ex]), E, x["Kex"], x["y"], ntr, nte)["acc"][0]; print("exact-kernel SVM test accuracy %.2f" % ac_ex)
    L = []
    for a in ARMS:
        m = mt[a]; f = lambda v: "%.4f (99%% interval %.4f to %.4f)" % ((v.mean(), v.mean() - tq(C) * v.std(ddof=1) / math.sqrt(C), v.mean() + tq(C) * v.std(ddof=1) / math.sqrt(C)))
        L.append("  %s %-42s mean abs kernel error %s; Frobenius (train block) %.3f%s" % (a, LABEL[a], f(m["mae"]), m["frob"].mean(), ("; test accuracy %.3f (99%% interval %.3f to %.3f)" % ((m["acc"].mean(),) + tuple(paired(m["acc"], np.zeros(C), 9)[1:3]))) if nte else ""))
    print("\n".join(L))
    for a, b, sd in (("O", "N", sdON), ("O", "S", sdOS)):
        mm, lo, hi, w, half = paired(mt[a]["mae"], mt[b]["mae"], EQUIV_MAE, sd)
        print("  OUTCOME %s minus %s, mean abs kernel error (paired by copy, 99%% interval %+.4f to %+.4f, P-table sd %.4f in quadrature; ABOVE = ours worse, BELOW = ours lower error, CONSISTENT = inside +-%.3f): %s" % (a, b, lo, hi, sd, EQUIV_MAE, w))
        pr = abs(mm) / half if half > 0 else float("nan"); print("    measured |%s-%s| %.4f over its own paired floor %.4f = %.1f" % (a, b, abs(mm), half, pr))
        if nte:
            mm, lo, hi, w, _ = paired(mt[a]["acc"], mt[b]["acc"], EQUIV_ACC); print("  OUTCOME %s minus %s, test accuracy (99%% interval %+.3f to %+.3f; ABOVE = ours higher; CONSISTENT = inside +-%.2f): %s" % (a, b, lo, hi, EQUIV_ACC, w))
    d = paired(mt["N"]["mae"], mt["O"]["mae"], EQUIV_MAE, sdON); print("  HEADLINE ours vs the client's raw run: mean abs kernel error %.4f vs %.4f" % (mt["O"]["mae"].mean(), mt["N"]["mae"].mean()))
    if pilot:
        pj = os.path.join(RUN, "pilot_result.json"); r = {"ratio_ON": abs(d[0]) / d[4], "ratio_OS": abs(paired(mt["S"]["mae"], mt["O"]["mae"], EQUIV_MAE, sdOS)[0]) / paired(mt["S"]["mae"], mt["O"]["mae"], EQUIV_MAE, sdOS)[4]}
        json.dump(r, open(pj, "w")); print("PILOT (hardware) POWER RATIO |O-N|/paired floor: %.1f ; |O-S|: %.1f -> saved pilot_result.json" % (r["ratio_ON"], r["ratio_OS"]))
    else:
        print("note: a CONSISTENT or CANNOT_TELL on O minus S is reported as it stands; the headline is O vs N.")


def mode_fire(pilot):
    sfx = "_pilot" if pilot else ""; cfg = CFG["pilot" if pilot else "full"]
    svc = get_service(); be = svc.backend(BACKEND); u = usage_remaining(svc); x = prepare(be, cfg); stated = stated_errors(be, x["prs"]); rec8 = record_readout(be)[x["lay"]]
    rbc, _ = rb_circuits(be, x["cols"], cfg); tot = est_p(cfg, len(rbc)) + est_main(cfg, len(x["E"])); cap = PILOT_SECONDS if pilot else MAX_SECONDS
    print("estimate P+main %.1f s (x%.2f = %.1f; cap %d; remaining %.0f)" % (tot, MARGIN, tot * MARGIN, cap, u))
    if tot * MARGIN > cap or tot * MARGIN > u: sys.exit("REFUSED: estimate above the cap or the remaining seconds (per chip cap %d s)" % cap)
    if s_load(sfx).get("jobs_main"): sys.exit("REFUSED: already fired on this chip; move run_%s/state%s.json aside to refire" % (BACKEND.replace("ibm_", ""), sfx))
    if not pilot:
        pr = power_lines(x, stated, rec8, cfg)[0][1]["ON"][2]; print("POWER RATIO (model): %.1f" % pr)
        if pr < 3:
            pj = os.path.join(RUN, "pilot_result.json"); r = json.load(open(pj))["ratio_ON"] if os.path.exists(pj) else None
            if r is None or r < 3: sys.exit("PILOT NEEDED: model POWER RATIO %.1f is under 3; run --fire --pilot first (pilot ratio %s)" % (pr, "none yet" if r is None else "%.1f" % r))
            print("pilot ratio %.1f (hardware) passes 3" % r)
    submit_all(be, x, cfg, sfx, stated, pilot)
    if pilot:
        t0 = time.time(); DEADLINE = 1800
        while time.time() - t0 < DEADLINE:
            time.sleep(20)
            if fetch(sfx): return analyze(sfx)
        sys.exit("pilot jobs not DONE after %d s; run --fetch --pilot then --analyze --pilot" % DEADLINE)
    print("Next: --fetch until everything is DONE, then --analyze.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in ("test", "dry", "fire", "fetch", "analyze", "pilot"): ap.add_argument("--" + m, action="store_true")
    a = ap.parse_args(); sfx = "_pilot" if a.pilot else ""
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.fire: mode_fire(a.pilot)
    elif a.fetch: fetch(sfx)
    elif a.analyze: analyze(sfx)
    else: ap.print_help()
