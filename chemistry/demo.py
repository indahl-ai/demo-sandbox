#!/usr/bin/env python
"""Demo 2: Chemistry / materials. Ground-state energy of H2 (minimal basis, 0.735 A, 2-qubit parity-reduced) by single-ancilla
iterative phase estimation (IPE): one ancilla, M mid-circuit readouts with reset and classical feedforward.
ONE layout (the client's default: transpile with no layout given, the vendor-record pick). Three readings of the SAME shots:
  N raw (what the client reads), R readout corrected with the vendor's RECORD figures, O corrected with today's measured table (ours).
Job P: fresh readout calibration (prep 0 / prep 1) on every qubit the circuit uses. Main job: COPIES copies of the client circuit.
Correction is model-based (the readout flip changes the feedforward, so a plain matrix inverse is not enough): see rec_probs.
Modes: --test --dry --fire --fetch --analyze.  --fire submits to IBM; nothing else does.
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE. Keys / CRNs are never printed."""
import os, sys, json, math, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.linalg import expm
from scipy import stats
from scipy.optimize import nnls
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate

HERE = os.path.dirname(os.path.abspath(__file__)); FORGE = HERE   # sandbox copy: no house paths
BACKEND = os.environ.get("DEMO_BACKEND", "ibm_kingston")
STATE = os.path.join(HERE, "state.json"); RESULT = os.path.join(HERE, "result.json")
M = 5                      # phase bits (mid-circuit ancilla readouts)
COPIES, SHOTS = 6, 4000    # main job: copies of the client circuit
P_SHOTS = 4000             # Job P shots per prep
MAX_SECONDS = 200          # refuse if the estimate is above this
EPS_FLOOR = 0.03           # model floor for the small excited-state weight
EQUIV_E, EQUIV_F = 0.01, 0.02   # CONSISTENT margins: energy error (Ha), fraction right
# H2, 0.735 A, STO-3G, parity mapping, 2-qubit reduction (electronic part; nuclear repulsion 0.7199689944 Ha not included)
HAM = [("II", -1.052373245772859), ("IZ", 0.39793742484318045), ("ZI", -0.39793742484318045),
       ("ZZ", -0.01128010425623538), ("XX", 0.18093119978423156)]
HMAT = SparsePauliOp.from_list(HAM).to_matrix()
W, V = np.linalg.eigh(HMAT); E0 = float(W[0])

def pick_time():
    """Evolution time t: ground-state phase phi0 = -E0 t/2pi lands exactly on a bin of M bits; other eigenphases stay clear of it."""
    best = None
    for k in range(1, 2 ** M):
        t = 2 * math.pi * k / (2 ** M) / (-E0) * -1 if False else 2 * math.pi * (k / 2 ** M) / (-E0)
        if not (1.0 < t < 3.8): continue
        ph = (-W * t / (2 * math.pi)) % 1.0
        sep = min(min(abs(p - ph[0]), 1 - abs(p - ph[0])) for p in ph[1:])
        if best is None or sep > best[0]: best = (sep, t, k)
    return best
_, T_EVOL, K_BIN = pick_time()
B0 = K_BIN                 # exact ground-state bin (phase = B0 / 2^M)
E_LO = E0 - 0.25           # energy window (E_LO, E_LO + 2pi/t] used to unwrap phase to energy
def phi_to_E(phi):
    w = 2 * math.pi / T_EVOL; e = -2 * math.pi * phi / T_EVOL
    return E_LO + ((e - E_LO) % w)
HF_STATE = int(np.argmax(np.abs(V[:, 0]) ** 2 * np.isin(np.arange(4), [1, 2])))   # |01> or |10>, the Hartree-Fock determinant
HF_OVERLAP = float(abs(V[HF_STATE, 0]) ** 2)

# ---------------------------------------------------------------- the client's circuit
def ipe_circuit():
    anc = QuantumRegister(1, "anc"); reg = QuantumRegister(2, "sys"); cr = ClassicalRegister(M, "ph")
    qc = QuantumCircuit(anc, reg, cr)
    for i in range(2):
        if (HF_STATE >> i) & 1: qc.x(reg[i])
    for k in range(M):
        p = M - 1 - k
        qc.reset(anc[0]); qc.h(anc[0])
        qc.append(UnitaryGate(expm(-1j * HMAT * T_EVOL * 2 ** p), label=f"U^{2**p}").control(1), [anc[0], reg[0], reg[1]])
        for j in range(k):
            with qc.if_test((cr[j], 1)): qc.p(-math.pi / 2 ** (k - j), anc[0])
        qc.h(anc[0]); qc.measure(anc[0], cr[k])
    return qc

def client_transpile(be):
    """Client default: no layout given, optimization level 3 (vendor-record layout)."""
    return transpile(ipe_circuit(), backend=be, optimization_level=3, seed_transpiler=7)

def physical_info(tc):
    """measured physical qubit per step, and the set of active physical qubits."""
    mq = {}; act = set()
    for ci in tc.data:
        qs = [tc.find_bit(q).index for q in ci.qubits]
        if ci.operation.name not in ("barrier", "delay"): act.update(qs)
        if ci.operation.name == "measure": mq[tc.find_bit(ci.clbits[0]).index] = qs[0]
        if ci.operation.name == "if_else":
            for blk in ci.operation.blocks:
                for cj in blk.data:
                    act.update(tc.find_bit(ci.qubits[blk.find_bit(q).index]).index for q in cj.qubits)
    return [mq[k] for k in range(M)], sorted(act)

def n2q(tc): return sum(v for g, v in tc.count_ops().items() if g in ("cz", "ecr", "cx"))

# ---------------------------------------------------------------- readout model and estimators
def rec_probs(phi, flips):
    """P(record string r | eigenphase phi) with the feedforward driven by the RAW recorded bits.
    phi: array (G,). flips: M tuples (p01, p10) = (P(read 1 | true 0), P(read 0 | true 1)) of the qubit read at step k.
    r integer = sum r_k 2^k (step 0 is the least significant phase bit). Returns (G, 2^M)."""
    phi = np.atleast_1d(np.asarray(phi, float)); G = len(phi); out = np.zeros((G, 2 ** M))
    prefixes = {(): np.ones(G)}
    for k in range(M):
        new = {}
        for pre, pr in prefixes.items():
            om = -math.pi * sum(b / 2 ** (k - j) for j, b in enumerate(pre))
            alpha = 2 * math.pi * 2 ** (M - 1 - k) * phi + om
            p0 = np.cos(alpha / 2) ** 2; a01, a10 = flips[k]
            r0 = p0 * (1 - a01) + (1 - p0) * a10
            new[pre + (0,)] = pr * r0; new[pre + (1,)] = pr * (1 - r0)
        prefixes = new
    for pre, pr in prefixes.items(): out[:, sum(b << j for j, b in enumerate(pre))] = pr
    return out

PHI_GRID = np.arange(4096) / 4096.0
def estimate(counts_vec, flips):
    """ML phase over a grid (small uniform floor for excited-state weight) -> energy; share of shots whose true phase bits are all right (non-negative deconvolution of the records through the readout model)."""
    P = rec_probs(PHI_GRID, flips); L = (1 - EPS_FLOOR) * P + EPS_FLOOR / 2 ** M
    ll = (np.log(L) * counts_vec[None, :]).sum(axis=1); phi = PHI_GRID[int(np.argmax(ll))]
    Pb = rec_probs(np.arange(2 ** M) / 2 ** M, flips)               # [b, r]: record distribution if the true bin were b
    q, _ = nnls(Pb.T, counts_vec / counts_vec.sum())                 # deconvolve: true-bin shares that reproduce the records
    frac = float(q[B0] / q.sum())
    return phi_to_E(phi), frac

def vec(counts):
    v = np.zeros(2 ** M)
    for s, n in counts.items(): v[int(s.replace(" ", ""), 2)] += n      # ph bits: string is c[M-1]..c[0], so int(s,2) = sum r_k 2^k
    return v

def readout_from_counts(c0, c1, qubits, shots):
    out = {}
    for i, q in enumerate(qubits):
        s01 = sum(v for b, v in c0.items() if b.replace(" ", "")[::-1][i] == "1"); s10 = sum(v for b, v in c1.items() if b.replace(" ", "")[::-1][i] == "0")
        out[q] = (s01 / shots, s10 / shots)
    return out

# ---------------------------------------------------------------- Job P circuits
def cal_circuits(be, qubits):
    out = []
    for ones in (False, True):
        qc = QuantumCircuit(len(qubits), len(qubits))
        if ones: qc.x(range(len(qubits)))
        qc.measure(range(len(qubits)), range(len(qubits)))
        out.append(transpile(qc, backend=be, initial_layout=list(qubits), optimization_level=0))
    return out

# ---------------------------------------------------------------- analysis (same shots, three readings)
def record_flips(record_err, mq):
    return [(record_err[q], record_err[q]) for q in mq]

def analyze_copies(copies, flip_sets):
    res = {}
    for name, fl in flip_sets.items():
        res[name] = [estimate(vec(c), fl) for c in copies]
    return res

def t99(n): return float(stats.t.ppf(0.995, n - 1))

def word(lo, hi, eq):
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "CONSISTENT" if (-eq < lo and hi < eq) else "CANNOT_TELL"

def report(res, header=""):
    L = [header + f"exact ground energy (electronic) {E0:.5f} Ha; phase bits {M}; t={T_EVOL:.4f}; exact bin {B0}/{2**M}; HF overlap {HF_OVERLAP:.4f}"]
    names = {"N": "raw (client, bits as read)", "R": "corrected with IBM record readout", "O": "corrected with today's table (ours)"}
    err = {}; fr = {}
    for a in "NRO":
        n = len(res[a]); e = np.array([abs(x[0] - E0) for x in res[a]]); f = np.array([x[1] for x in res[a]]); err[a], fr[a] = e, f
        he, hf = t99(n) * e.std(ddof=1) / math.sqrt(n), t99(n) * f.std(ddof=1) / math.sqrt(n)
        L.append(f"  {a} {names[a]}: energy error {e.mean()*1000:.1f} mHa (99% interval {max(e.mean()-he,0)*1000:.1f} to {(e.mean()+he)*1000:.1f}); all phase bits right {f.mean():.4f} (99% interval {f.mean()-hf:.4f} to {f.mean()+hf:.4f})")
    for lab, d, eq, unit in (("energy error", err["O"] - err["R"], EQUIV_E, "Ha"), ("fraction right", fr["O"] - fr["R"], EQUIV_F, "")):
        n = len(d); hh = t99(n) * d.std(ddof=1) / math.sqrt(n); lo, hi = d.mean() - hh, d.mean() + hh
        if lab == "fraction right":
            w = word(-hi, -lo, eq); note = "ABOVE = ours worse, BELOW = ours better"   # more right is better, so flip sign for the word
        else:
            w = word(lo, hi, eq); note = "ABOVE = ours worse, BELOW = ours better"
        L.append(f"  OUTCOME O minus R, {lab}: mean difference {d.mean():+.4f}{unit} (99% interval {lo:+.4f} to {hi:+.4f}; {note}; CONSISTENT inside +-{eq}): {w}")
    L.append(f"  HEADLINE vs R: energy error ours {err['O'].mean()*1000:.1f} mHa vs R {err['R'].mean()*1000:.1f} mHa (raw N {err['N'].mean()*1000:.1f}); all-bits-right ours {fr['O'].mean():.3f} vs R {fr['R'].mean():.3f} (raw {fr['N'].mean():.3f})")
    return "\n".join(L)

# ---------------------------------------------------------------- IBM helpers
def read_key():
    """Your IBM Quantum API key, from the IBM_QUANTUM_TOKEN environment variable only. Never printed, never stored."""
    t = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if not t: sys.exit("no IBM Quantum key: set IBM_QUANTUM_TOKEN (your own key; it stays on this machine)")
    return t

def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService as S
    tok = read_key()
    base = S(channel="ibm_quantum_platform", token=tok)
    import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); import pick_instance
    crn = pick_instance.crn(base)
    return S(channel="ibm_quantum_platform", token=tok, instance=crn)

def usage_remaining(svc):
    try: return float(svc.usage().get("usage_remaining_seconds"))
    except Exception as e: sys.exit(f"cannot read usage_remaining_seconds ({type(e).__name__}); refusing")

def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
    return FakeFez() if "fez" in BACKEND else FakeMarrakesh() if "marrakesh" in BACKEND else FakeKingston()

def record_readout(be, qubits):
    t = be.target; return {q: float(t["measure"][(q,)].error) for q in qubits}

def estimate_seconds(be, tc):
    try: dur = float(tc.estimate_duration(be.target))
    except Exception: dur = 60e-6
    dur = max(dur, 20e-6); per = dur + 250e-6      # default rep delay
    main = COPIES * SHOTS * per; calp = 2 * P_SHOTS * (per * 0 + 250e-6 + 3e-6)
    return main, calp, dur, main + calp + 3 * 2.0    # + about 2 s per-job overhead (3 jobs worth, conservative)

def setup(be):
    tc = client_transpile(be); mq, act = physical_info(tc)
    return tc, mq, act

# ---------------------------------------------------------------- modes
def aer_noise(be, act, mq, planted):
    from qiskit_aer.noise import NoiseModel, ReadoutError
    nm = NoiseModel.from_backend(be, readout_error=False); rec = record_readout(be, act)
    for q in act:
        p01, p10 = planted.get(q, (rec[q], rec[q]))
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q])
    return nm

def mode_test():
    from qiskit_aer import AerSimulator
    be = fake_backend(); tc, mq, act = setup(be); rec = record_readout(be, act)
    anc_q = mq[0]; planted = {anc_q: (0.02, 0.14)}
    print(f"[TEST: offline Aer, fake {BACKEND}, planted bad readout on the ancilla qubit {anc_q}: P(read1|0)=0.020 P(read0|1)=0.140; record says {rec[anc_q]:.4f}]")
    print(f"layout: ancilla measured on physical {sorted(set(mq))}, active qubits {act}, two-qubit gates {n2q(tc)}, record readout {{{', '.join(f'{q}:{rec[q]:.4f}' for q in act)}}}")
    # ideal check: noiseless IPE gives the exact bin
    ideal = AerSimulator().run(transpile(ipe_circuit(), AerSimulator(), optimization_level=0), shots=4000, seed_simulator=1).result().get_counts()
    v = vec(ideal); print(f"noiseless check: share on exact bin {v[B0]/v.sum():.4f} (HF overlap {HF_OVERLAP:.4f}); energy {estimate(v, [(0,0)]*M)[0]:.5f} vs exact {E0:.5f}")
    nm = aer_noise(be, act, mq, planted); sim = AerSimulator(noise_model=nm, method='statevector')
    ccs = cal_circuits(be, act)
    cc = [sim.run(c, shots=P_SHOTS, seed_simulator=104729 * (11 + i)).result().get_counts() for i, c in enumerate(ccs)]
    cal = readout_from_counts(cc[0], cc[1], act, P_SHOTS)
    print("Job P (simulated) measured readout, (P(1|0), P(0|1)): " + ", ".join(f"q{q} ({cal[q][0]:.4f},{cal[q][1]:.4f})" for q in act))
    copies = [sim.run(tc, shots=SHOTS, seed_simulator=104729 * (1000 + c)).result().get_counts() for c in range(COPIES)]
    flips = {"N": [(0.0, 0.0)] * M, "R": record_flips(rec, mq), "O": [cal[q] for q in mq]}
    res = analyze_copies(copies, flips)
    print(report(res, "SIMULATION. "))

def mode_dry():
    svc = get_service(); u = usage_remaining(svc); be = svc.backend(BACKEND); st = be.status()
    tc, mq, act = setup(be); main, calp, dur, tot = estimate_seconds(be, tc); rec = record_readout(be, act)
    print(f"backend {BACKEND} operational={st.operational} pending_jobs={st.pending_jobs}; usage_remaining_seconds={u:.0f}")
    print(f"client circuit: depth {tc.depth()}, two-qubit gates {n2q(tc)}, per-shot duration about {dur*1e6:.0f} us; M={M} mid-circuit readouts")
    print(f"layout (client default): ancilla measured on physical {sorted(set(mq))}; active qubits {act}; mid-circuit readout qubit per step {mq}")
    print("record readout error (IBM average flip): " + ", ".join(f"q{q} {rec[q]:.4f}" for q in act))
    print(f"estimated seconds: main {main:.1f} + Job P {calp:.1f} + overhead -> about {tot:.0f} s (cap {MAX_SECONDS})")
    if tot > MAX_SECONDS or tot > u: print("WOULD REFUSE")

def s_save(d):
    cur = json.load(open(STATE)) if os.path.exists(STATE) else {}; cur.update(d); json.dump(cur, open(STATE, "w"), indent=1)

def mode_fire():
    from qiskit_ibm_runtime import SamplerV2
    svc = get_service(); u = usage_remaining(svc); be = svc.backend(BACKEND)
    tc, mq, act = setup(be); main, calp, dur, tot = estimate_seconds(be, tc)
    if tot > MAX_SECONDS: sys.exit(f"REFUSED: estimate {tot:.0f} s above {MAX_SECONDS} s")
    if u < tot: sys.exit(f"REFUSED: usage_remaining_seconds {u:.0f} below estimate {tot:.0f}")
    if os.path.exists(STATE) and "job_main" in json.load(open(STATE)): sys.exit("REFUSED: state.json already has fired jobs; move it aside to refire")
    rec = record_readout(be, act); ccs = cal_circuits(be, act)
    s_save(dict(mq=mq, act=act, record={str(q): rec[q] for q in act}, backend=BACKEND, fired_at=time.time(), est_seconds=tot))
    jp = SamplerV2(mode=be).run(ccs, shots=P_SHOTS); s_save(dict(job_p=jp.job_id())); print("job P (readout calibration) submitted")
    jm = SamplerV2(mode=be).run([(tc,)] * COPIES, shots=SHOTS); s_save(dict(job_main=jm.job_id())); print("main job submitted; use --fetch then --analyze")

def bitcounts(pub):
    d = pub.data; reg = [getattr(d, a) for a in dir(d) if not a.startswith("_") and hasattr(getattr(d, a), "get_counts")][0]
    return reg.get_counts()

def mode_fetch():
    svc = get_service(); st = json.load(open(STATE)); out = {}
    for key, name in (("job_p", "P"), ("job_main", "main")):
        j = svc.job(st[key]); s = str(j.status()); print(f"job {name}: {s}")
        if "DONE" not in s.upper(): sys.exit("not finished; run --fetch again later")
        out[name] = [bitcounts(p) for p in j.result()]
        try: out[name + "_seconds"] = j.usage()
        except Exception: pass
    json.dump(out, open(RESULT, "w")); print("saved result.json; seconds used: " + str({k: v for k, v in out.items() if k.endswith("_seconds")}))

def mode_analyze():
    st = json.load(open(STATE)); r = json.load(open(RESULT)); act, mq = st["act"], st["mq"]
    cal = readout_from_counts(r["P"][0], r["P"][1], act, P_SHOTS); rec = {int(q): v for q, v in st["record"].items()}
    print("readout, record vs today (P(1|0), P(0|1)): " + ", ".join(f"q{q} record {rec[q]:.4f} / today ({cal[q][0]:.4f},{cal[q][1]:.4f})" for q in act))
    flips = {"N": [(0.0, 0.0)] * M, "R": record_flips(rec, mq), "O": [cal[q] for q in mq]}
    print(report(analyze_copies(r["main"], flips), "HARDWARE. "))
    print("Left out: the darkwave arm (see PLAN.md).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in ("test", "dry", "fire", "fetch", "analyze"): ap.add_argument("--" + m, action="store_true")
    a = ap.parse_args()
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.fire: mode_fire()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
