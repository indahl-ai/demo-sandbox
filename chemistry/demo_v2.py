#!/usr/bin/env python
"""Demo 2 v2 (chemistry, H2 by single-ancilla iterative phase estimation): the "ours" arm is OUR kick correction, not a readout table.
The register idles while the ancilla is read mid-circuit; each readout turns the waiting qubits' phase (our kick finding). We put a virtual
Rz on every register qubit right after each ancilla readout, size and sign from a SAME-SESSION hardware kick measurement (job K) on the
actual register qubits next to the actual ancilla qubit(s). Job K is never the job that applies it.

ONE layout, ONE transpile (the client's: level 3, record layout). The two circuits in the main job are that ISA circuit, byte for byte,
and the same circuit with Rz gates inserted after each ancilla readout. Nothing is re-transpiled, so nothing can merge, cancel or move our
rotations; the script asserts they survive unchanged. Runtime changes are OFF for K, P and main alike: twirling off (gates and measure),
dynamical decoupling off, no resilience (SamplerV2 has none). Options are printed from the object and read back from each job.

Arms (same shots read different ways, plus the corrected circuit):
  N    client circuit, raw bits (CLIENT DEFAULT: their circuit, their layout, bits as read)
  R    client circuit, readout corrected with the live record's figures
  O_ro client circuit, readout corrected with tonight's job-P table (readout only: the table without the kick)
  O    OUR circuit (kick-corrected), readout corrected with the same job-P table

Every parameter of our arm cites a hardware source (kick: job K id; readout table: job P id; sign: on-chip Rz control inside job K;
record figures: live record stamp). --fire refuses if any is missing.
Modes: --test (code smoke test only, no result) --dry --fire --fetch --analyze.
  --fire is staged: first call submits the calibration jobs K and P; after --fetch, the second call submits the main job.
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE. Keys and CRNs are never printed. Nothing here commits or pushes."""
import os, sys, json, math, time, argparse, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import RZGate

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import demo as v1                       # the v1 circuit, estimators and helpers; v1 files are not modified

FORGE = v1.FORGE; BACKEND = os.environ.get("DEMO_BACKEND", v1.BACKEND); v1.BACKEND = BACKEND; M = v1.M
CHIP = "" if BACKEND == "ibm_kingston" else "_" + BACKEND.replace("ibm_", "")
STATE = os.path.join(HERE, "state_v2.json"); KICKF = os.path.join(HERE, "kick_v2.json")
RESULT = os.path.join(HERE, "result_v2.json"); KRAW = os.path.join(HERE, "kick_raw_v2.npz")
COPIES, SHOTS, P_SHOTS = 6, 4000, 4000
K_SHOTS, K_NS, CTL_DEG = 4000, (8, 16), 5.0     # job K: shots per circuit, readout counts per circuit, on-chip control rotation per step
MAX_SECONDS = 200
EQUIV_E, EQUIV_F = 0.01, 0.02
HIST = "22 Oct dd-run, kingston job dbcuqa0c77vs738739g0 (results_dd.text R1, DD off, all spectators): median -3.243 deg per readout, interval -3.809 to -2.04; wired -2.561"
ARMS = ("N", "R", "O_ro", "O")
LABEL = {"N": "N    client default, raw", "R": "R    client circuit, record readout", "O_ro": "O_ro client circuit, job-P readout (readout only)",
         "O": "O    our circuit (kick-corrected), job-P readout"}

# ------------------------------------------------------------ options (identical for K, P and main)
def clean_sampler(be):
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=be); o = s.options
    o.dynamical_decoupling.enable = False; o.twirling.enable_gates = False; o.twirling.enable_measure = False
    return s

def options_line(o):
    return "twirling.enable_gates=%s twirling.enable_measure=%s dynamical_decoupling.enable=%s resilience=none (SamplerV2 has no resilience) default_shots=%s" % (
        o.twirling.enable_gates, o.twirling.enable_measure, o.dynamical_decoupling.enable, o.default_shots)

def assert_clean(s):
    o = s.options
    if o.twirling.enable_gates or o.twirling.enable_measure or o.dynamical_decoupling.enable: sys.exit("REFUSED: runtime changes are not all OFF")

def show_job_options(job, label):
    try:
        inp = job.inputs; o = inp.get("options", inp) if isinstance(inp, dict) else inp
        print("%s options read back from the job: %s" % (label, json.dumps(o, default=str)[:700]))
    except Exception as ex: print("%s: could not read options back from the job (%s)" % (label, type(ex).__name__))

# ------------------------------------------------------------ phase convention (pure algebra, no noise)
def convention_check():
    """Rz(theta) on |+> then the X / Y read-outs used below give phase = atan2(<Y>,<X>) = +theta. Exact algebra, produces no constant."""
    th = 0.37
    def ex(basis):
        qc = QuantumCircuit(1); qc.h(0); qc.rz(th, 0)
        if basis == "Y": qc.sdg(0)
        qc.h(0); p = Statevector(qc).probabilities([0]); return p[0] - p[1]
    got = math.atan2(ex("Y"), ex("X"))
    if abs(got - th) > 1e-9: sys.exit("REFUSED: phase convention check failed (%r vs %r)" % (got, th))
    return True

# ------------------------------------------------------------ job K: kick measurement on the actual qubits
def k_circuit(specs, mode, N, basis, L_dt, ctl=False):
    """Logical qubit 0 = the read (ancilla) qubit, 1.. = the register/spectator qubits, started on the equator. N steps, each fenced by
    barriers on all qubits. M: the ancilla does what it does in the IPE step minus the controlled-U: reset, H, measure. D: the ancilla
    idles the same scheduled length L. ctl: an explicit Rz(+CTL_DEG) on every spectator each step (on-chip sign control, D mode only)."""
    nq = 1 + len(specs); S = list(range(1, nq)); qc = QuantumCircuit(nq)
    regs = [ClassicalRegister(1, "m%d" % k) for k in range(N if mode == "M" else 0)]
    for r_ in regs: qc.add_register(r_)
    fin = ClassicalRegister(len(S), "fin"); qc.add_register(fin)
    qc.h(S); qc.barrier()
    for k in range(N):
        if mode == "M": qc.reset(0); qc.h(0); qc.measure(0, regs[k][0])
        else: qc.delay(L_dt, 0, unit="dt")
        if ctl: qc.rz(math.radians(CTL_DEG), S)
        qc.barrier()
    if basis == "Y": qc.sdg(S)
    qc.h(S); qc.measure(S, fin)
    return qc

def align_of(be):
    t = be.target; return math.lcm(int(getattr(t, "granularity", 1) or 1), int(getattr(t, "pulse_alignment", 1) or 1))

def barrier_steps(tc, width):
    bars = [t0 for ci, t0 in zip(tc.data, tc.op_start_times) if ci.operation.name == "barrier" and len(ci.qubits) == width]
    return np.diff(bars)

def step_len(be, r, specs):
    """Scheduled length (dt) of one M step on this read qubit, read off the transpiled schedule."""
    nq = 1 + len(specs); qc = k_circuit(specs, "M", 2, "X", None)
    t = transpile(qc, be, initial_layout=[r] + list(specs), optimization_level=0, scheduling_method="alap", seed_transpiler=7)
    d = barrier_steps(t, nq)
    if len(d) != 2 or d[0] != d[1]: sys.exit("REFUSED: M step lengths unequal on read qubit %d: %s" % (r, d))
    L = int(d[0]); al = align_of(be)
    if L % al: sys.exit("REFUSED: M step %d dt is not a multiple of the delay alignment %d on read qubit %d" % (L, al, r))
    return L

def k_plan(be, mq, act):
    """The K circuit list, transpiled (level 0, scheduled), with metadata. One block per distinct read qubit."""
    metas, circs, info = [], [], {}
    reads = sorted(set(mq))
    for ri, r in enumerate(reads):
        specs = [q for q in act if q != r]; L = step_len(be, r, specs); info[r] = dict(specs=specs, L_dt=L)
        def add(mode, N, basis, ctl=False):
            metas.append(dict(r=r, mode=mode, N=N, basis=basis, ctl=ctl))
            circs.append(transpile(k_circuit(specs, mode, N, basis, L, ctl), be, initial_layout=[r] + specs, optimization_level=0,
                                   scheduling_method="alap", seed_transpiler=7))
        for bs in "XY": add("D", 0, bs)
        for N in K_NS:
            for mode in ("D", "M"):
                for bs in "XY": add(mode, N, bs)
        if ri == 0:
            for bs in "XY": add("D", K_NS[0], bs, ctl=True)
    return circs, metas, info

def verify_k(circs, metas, info):
    """Every step (barrier to barrier) lasts exactly L in both M and D arms; no X or other gate on the ancilla in D beyond the delay."""
    for c, m in zip(circs, metas):
        L = info[m["r"]]["L_dt"]; w = 1 + len(info[m["r"]]["specs"])
        if m["N"]:
            d = barrier_steps(c, w)
            if not (len(d) == m["N"] and np.all(d == L)): sys.exit("REFUSED: K step lengths differ from L=%d in %s: %s" % (L, m, d[:4]))
    return "VERIFIED: every K step is exactly L dt (per read qubit) in the M, D and control circuits"

def est_k(be, metas, info):
    s = 0.0
    for m in metas: s += K_SHOTS * (250e-6 + m["N"] * info[m["r"]]["L_dt"] * be.target.dt + 5e-6)
    return s * 1.3 + 4

def unpack(arr, n):
    arr = np.asarray(arr, np.uint8); nb = arr.shape[1]
    return np.stack([(arr[:, nb - 1 - j // 8] >> (j % 8)) & 1 for j in range(n)], 1)

def phase_deg(bx, by):
    return np.degrees(np.arctan2((1 - 2 * by.astype(float)).mean(0), (1 - 2 * bx.astype(float)).mean(0)))

def wrap(d): return (np.asarray(d) + 180.0) % 360.0 - 180.0

def fit_kick(bits, metas, info, boot=300, seed=20261119):
    """bits[i] = shots x n_spec array for circuit i. Kick per readout step, per (read qubit, spectator): slope through the origin of
    wrap(phase_M - phase_D) against N, degrees, with a shot-bootstrap 99% interval. Also the on-chip control (must read +CTL_DEG)."""
    rng = np.random.default_rng(seed); ix = {(m["r"], m["mode"], m["N"], m["basis"], m["ctl"]): i for i, m in enumerate(metas)}
    def ph(r, mode, N, ctl, idx=None):
        a = []
        for bs in "XY":
            b = bits[ix[(r, mode, N, bs, ctl)]]
            if idx is not None: b = b[rng.integers(0, len(b), len(b))]
            a.append(b)
        return phase_deg(a[0], a[1])
    out, ctl_out = {}, {}
    def slope(r, idx):
        g = {N: wrap(ph(r, "M", N, False, idx) - ph(r, "D", N, False, idx)) for N in K_NS}
        return sum(N * g[N] for N in K_NS) / sum(N * N for N in K_NS)
    for r in info:
        base = slope(r, None); bs_ = np.array([slope(r, 1) for _ in range(boot)])
        for j, s in enumerate(info[r]["specs"]):
            lo, hi = np.percentile(bs_[:, j], [0.5, 99.5]); out["%d|%d" % (r, s)] = [float(base[j]), float(bs_[:, j].std()), float(lo), float(hi)]
    r0 = sorted(info)[0]; N0 = K_NS[0]
    gb = wrap(ph(r0, "D", N0, True) - ph(r0, "D", N0, False)) / N0
    gs = np.array([wrap(ph(r0, "D", N0, True, 1) - ph(r0, "D", N0, False, 1)) / N0 for _ in range(boot)])
    for j, s in enumerate(info[r0]["specs"]): ctl_out["%d|%d" % (r0, s)] = [float(gb[j]), float(gs[:, j].std())]
    return out, ctl_out

# ------------------------------------------------------------ our circuit: client's ISA circuit plus virtual Rz after each ancilla readout
def make_ours(tc, mq, act, kick_deg):
    """kick_deg[(r, s)] = measured kick per readout (degrees) on spectator s while r is read. The correction is Rz(-kick). Returns (circuit, inserted
    instruction indices, expected [(spectator, angle)])."""
    new = tc.copy_empty_like(); ins, exp = [], []
    for ci in tc.data:
        new.append(ci)
        if ci.operation.name == "measure":
            r = tc.find_bit(ci.qubits[0]).index; k = tc.find_bit(ci.clbits[0]).index
            if k < M and mq[k] == r:
                for s in act:
                    if s != r:
                        ang = -math.radians(kick_deg[(r, s)]); new.rz(ang, new.qubits[s]); ins.append(len(new.data) - 1); exp.append((s, ang))
    return new, ins, exp

def check_survive(tc, ours, ins, exp):
    """The client circuit is intact (same operations, qubits, clbits, parameters, in order) and each inserted Rz has exactly the set angle."""
    iset = set(ins); rest = [ci for i, ci in enumerate(ours.data) if i not in iset]
    if len(rest) != len(tc.data): sys.exit("REFUSED: our circuit lost or gained client operations")
    for a, b in zip(rest, tc.data):
        if a.operation.name != b.operation.name or [ours.find_bit(q).index for q in a.qubits] != [tc.find_bit(q).index for q in b.qubits] \
           or [ours.find_bit(c).index for c in a.clbits] != [tc.find_bit(c).index for c in b.clbits]:
            sys.exit("REFUSED: a client operation changed in our circuit")
        if a.operation.name not in ("if_else", "for_loop", "while_loop", "switch_case") and [float(x) for x in a.operation.params] != [float(x) for x in b.operation.params]:
            sys.exit("REFUSED: a client parameter changed in our circuit")
    if len(ins) != len(exp): sys.exit("REFUSED: insertion bookkeeping mismatch")
    for i, (s, ang) in zip(ins, exp):
        ci = ours.data[i]
        if ci.operation.name != "rz" or ours.find_bit(ci.qubits[0]).index != s or float(ci.operation.params[0]) != ang:
            sys.exit("REFUSED: a corrective rotation is not exactly as set (index %d)" % i)
    return "VERIFIED: client operations unchanged and in order; %d inserted Rz gates carry exactly the set angles (no re-transpile after insertion)" % len(ins)

# ------------------------------------------------------------ SOURCES (hardware only); the fire refuses on any gap
def record_stamp(be):
    try: return str(be.properties().last_update_date)
    except Exception: return None

def sources_rows(be, st, kick, ptab_done):
    rows, bad = [], []
    jk = (kick or {}).get("job_id"); jp = st.get("job_p") if ptab_done else None
    rows.append(("kick angle per (read qubit, register qubit), degrees per readout", "job K %s (same session, fired and read before main)" % jk if jk and kick.get("values") else "MISSING"))
    rows.append(("Rz sign (correction = minus the measured turn)", "on-chip +%.1f deg Rz control inside job K %s, fitted with the same code" % (CTL_DEG, jk) if jk and kick.get("control_ok") else "MISSING"))
    rows.append(("readout table per qubit, (P(1|0), P(0|1)), O_ro and O", "job P %s" % jp if jp else "MISSING"))
    stamp = record_stamp(be)
    rows.append(("record readout figures (R arm)", "live record, last_update_date %s" % stamp if stamp else "MISSING"))
    rows.append(("layout, read qubits per step, register qubits", "client's own level-3 transpile on the live record (their input, not ours): read %s, active %s" % (st.get("mq"), st.get("act"))))
    for r in rows:
        if "MISSING" in r[1]: bad.append(r[0])
    return rows, bad

def print_sources(rows):
    print("SOURCES (parameter -> hardware source):")
    for r in rows: print("  %-62s %s" % r)

# ------------------------------------------------------------ analysis
def t99(n): return float(stats.t.ppf(0.995, n - 1))

def word(lo, hi, eq):
    """Equivalence first (interval wholly inside +-eq is CONSISTENT), then ABOVE / BELOW / CANNOT_TELL."""
    if lo > -eq and hi < eq: return "CONSISTENT"
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "CANNOT_TELL"

def arm_results(client, ours, flips):
    """client / ours: lists of count dicts, one per copy. Returns {arm: [(energy, frac) per copy]}."""
    return {"N": [v1.estimate(v1.vec(c), flips["N"]) for c in client], "R": [v1.estimate(v1.vec(c), flips["R"]) for c in client],
            "O_ro": [v1.estimate(v1.vec(c), flips["O"]) for c in client], "O": [v1.estimate(v1.vec(c), flips["O"]) for c in ours]}

PILOT = False
def pilot_lines(res):
    """Rule 5 pilot: measured |O-S| per copy (O ours vs O_ro, same shots' readout table) and the pilot's own copy floor."""
    e = {a: np.array([abs(x[0] - v1.E0) for x in res[a]]) for a in ("O", "O_ro")}; f = {a: np.array([x[1] for x in res[a]]) for a in ("O", "O_ro")}
    n = len(e["O"]); dE = np.abs(e["O"] - e["O_ro"]); dF = np.abs(f["O"] - f["O_ro"])
    floorE = t99(n) * (e["O_ro"].std(ddof=1)) / math.sqrt(n); floorF = t99(n) * (f["O_ro"].std(ddof=1)) / math.sqrt(n)
    return ["PILOT (rule 5, %d copies, measured on hardware): mean |O-O_ro| energy %.2f mHa vs pilot copy floor (99%% half-width of O_ro copies) %.2f mHa -> ratio %.2f" % (n, dE.mean() * 1000, floorE * 1000, dE.mean() / max(floorE, 1e-12)),
            "PILOT: mean |O-O_ro| all-bits-right %.4f vs floor %.4f -> ratio %.2f" % (dF.mean(), floorF, dF.mean() / max(floorF, 1e-12))]

def report(res, header=""):
    L = [header + "exact ground energy (electronic) %.5f Ha; %d phase bits; t=%.4f; exact bin %d/%d; HF overlap %.4f; %d copies per circuit" % (
        v1.E0, M, v1.T_EVOL, v1.B0, 2 ** M, v1.HF_OVERLAP, len(res["N"]))]
    err, fr = {}, {}
    for a in ARMS:
        n = len(res[a]); e = np.array([abs(x[0] - v1.E0) for x in res[a]]); f = np.array([x[1] for x in res[a]]); err[a], fr[a] = e, f
        he, hf = t99(n) * e.std(ddof=1) / math.sqrt(n), t99(n) * f.std(ddof=1) / math.sqrt(n)
        L.append("  %s: energy error %.1f mHa (99%% interval %.1f to %.1f); all phase bits right %.4f (99%% interval %.4f to %.4f)" % (
            LABEL[a], e.mean() * 1000, max(e.mean() - he, 0) * 1000, (e.mean() + he) * 1000, f.mean(), f.mean() - hf, f.mean() + hf))
    for other, why in (("O_ro", "the kick's own contribution (same readout table)"), ("R", "vs the record-readout reading"), ("N", "vs the client default")):
        for lab, d, eq, unit in (("energy error", err["O"] - err[other], EQUIV_E, "Ha"), ("fraction right", fr["O"] - fr[other], EQUIV_F, "")):
            n = len(d); hh = t99(n) * d.std(ddof=1) / math.sqrt(n); lo, hi = d.mean() - hh, d.mean() + hh
            w = word(-hi, -lo, eq) if lab == "fraction right" else word(lo, hi, eq)      # ABOVE = ours worse, BELOW = ours better, both rows
            L.append("  OUTCOME O minus %s, %s [%s]: mean difference %+.4f%s (99%% interval %+.4f to %+.4f; paired by copy; ABOVE = ours worse, BELOW = ours better; CONSISTENT inside +-%s): %s" % (
                other, lab, why, d.mean(), unit, lo, hi, eq, w))
    if PILOT: L += pilot_lines(res)
    return "\n".join(L)

# ------------------------------------------------------------ state / io
def s_load(): return json.load(open(STATE)) if os.path.exists(STATE) else {}
def s_save(d): cur = s_load(); cur.update(d); json.dump(cur, open(STATE, "w"), indent=1)

def save_k_raw(res, metas, info):
    arrs = {}
    for i, (pr, m) in enumerate(zip(res, metas)):
        n = len(info[m["r"]]["specs"]); arrs["b%d" % i] = unpack(pr.data.fin.array, n)
    np.savez_compressed(KRAW, **arrs)

def setup(be):
    tc = v1.client_transpile(be); mq, act = v1.physical_info(tc); return tc, mq, act

def plan_seconds(be, tc, metas, info):
    main, calp, dur, _ = v1.estimate_seconds(be, tc)
    ek, ep, em = est_k(be, metas, info), calp + 4, 2 * main + 4
    return ek, ep, em, ek + ep + em

def fake_kick_table(mq, act, deg=1.0):
    return {(r, s): deg for r in set(mq) for s in act if s != r}

# ------------------------------------------------------------ modes
def mode_test():
    """CODE SMOKE TEST ONLY. Offline, fake chip, ideal Aer. Checks plumbing and the refusals. Produces no result, no prediction, sets no constant:
    the table below is a plumbing stand-in and is never printed as a number that means anything."""
    from qiskit_aer import AerSimulator
    be = v1.fake_backend(); print("SMOKE TEST ONLY (code plumbing; no result, no headline, no constant comes from this)")
    convention_check(); print("phase convention (exact algebra): OK")
    tc, mq, act = setup(be); print("layout built: read qubits %s, active %s, two-qubit gates %d" % (sorted(set(mq)), act, v1.n2q(tc)))
    circs, metas, info = k_plan(be, mq, act); print(verify_k(circs, metas, info)); print("K circuits: %d" % len(circs))
    ours, ins, exp = make_ours(tc, mq, act, fake_kick_table(mq, act)); print(check_survive(tc, ours, ins, exp))
    bad = False
    try:
        ours2, ins2, exp2 = make_ours(tc, mq, act, fake_kick_table(mq, act)); ours2.data[ins2[0]] = CircuitInstruction(RZGate(0.5), ours2.data[ins2[0]].qubits); check_survive(tc, ours2, ins2, exp2)
    except SystemExit: bad = True
    print("tamper check (changed Rz angle is refused): %s" % ("OK" if bad else "FAILED"))
    rows, miss = sources_rows(be, dict(mq=mq, act=act), None, False)
    print("source guard with nothing measured refuses: %s (missing %d rows)" % ("OK" if miss else "FAILED", len(miss)))
    sim = AerSimulator(seed_simulator=1); cl, ou = [], []
    for c in range(2):
        for circ, dst in ((tc, cl), (ours, ou)): dst.append(sim.run(circ, shots=300, seed_simulator=10 + c).result().get_counts())
    flips = {"N": [(0.0, 0.0)] * M, "R": v1.record_flips(v1.record_readout(be, act), mq), "O": [(0.01, 0.01)] * M}
    res = arm_results(cl, ou, flips); txt = report(res, "SMOKE ")
    print("analysis path executed on %d lines of output (numbers withheld in the smoke test)" % len(txt.splitlines()))
    print("estimated seconds path: K/P/main/total =", [round(x, 1) for x in plan_seconds(be, tc, metas, info)], "(offline fake durations)")

def mode_dry():
    from qiskit_ibm_runtime import SamplerV2
    convention_check(); svc = v1.get_service(); u = v1.usage_remaining(svc); be = svc.backend(BACKEND); stt = be.status()
    tc, mq, act = setup(be); rec = v1.record_readout(be, act)
    print("backend %s operational=%s pending_jobs=%s; usage_remaining_seconds=%.0f" % (BACKEND, stt.operational, stt.pending_jobs, u))
    print("client circuit (one ISA transpile, level 3, record layout): depth %d, two-qubit gates %d; M=%d mid-circuit readouts" % (tc.depth(), v1.n2q(tc), M))
    print("layout: read qubit per step %s; active qubits %s" % (mq, act))
    print("record readout error (live record average flip): " + ", ".join("q%d %.4f" % (q, rec[q]) for q in act))
    circs, metas, info = k_plan(be, mq, act); print(verify_k(circs, metas, info))
    for r in sorted(info): print("job K block: read qubit %d, register qubits %s, M step %d dt (%.2f us), %d circuits in K total" % (
        r, info[r]["specs"], info[r]["L_dt"], info[r]["L_dt"] * be.target.dt * 1e6, len(circs)))
    ours, ins, exp = make_ours(tc, mq, act, fake_kick_table(mq, act, 1.0))
    print("insertion plan (angles are placeholders so the survival check runs; real angles come from job K): %d Rz gates, %d client operations" % (len(ins), len(tc.data)))
    print(check_survive(tc, ours, ins, exp))
    for lab in ("K", "P", "main"):
        s = clean_sampler(be); assert_clean(s); print("Sampler options for job %s: %s" % (lab, options_line(s.options)))
    ek, ep, em, tot = plan_seconds(be, tc, metas, info)
    print("estimated seconds: K %.0f + P %.0f + main %.0f = %.0f (cap %d)" % (ek, ep, em, tot, MAX_SECONDS))
    rows, bad = sources_rows(be, dict(mq=mq, act=act), None, False)
    print("pending SOURCES (filled by jobs K and P; --fire refuses until they exist):"); print_sources(rows)
    print("cross-check on disk, not applied: " + HIST)
    if tot > MAX_SECONDS or tot > u: print("WOULD REFUSE (over cap or over remaining seconds)")

def fire_calibrations(svc, be, tc, mq, act):
    circs, metas, info = k_plan(be, mq, act); print(verify_k(circs, metas, info)); ek, ep, em, tot = plan_seconds(be, tc, metas, info)
    u = v1.usage_remaining(svc); print("estimated seconds K %.0f P %.0f main %.0f total %.0f; remaining %.0f" % (ek, ep, em, tot, u))
    if tot > MAX_SECONDS: sys.exit("REFUSED: total estimate %.0f s above %d s" % (tot, MAX_SECONDS))
    if tot > u: sys.exit("REFUSED: estimate above remaining seconds")
    rec = v1.record_readout(be, act)
    s_save(dict(mq=mq, act=act, record={str(q): rec[q] for q in act}, record_stamp=record_stamp(be), backend=BACKEND, est_seconds=tot,
                k_info={str(r): v for r, v in info.items()}, k_metas=metas, staged_at=time.time()))
    sk = clean_sampler(be); assert_clean(sk); jk = sk.run([(c, None, K_SHOTS) for c in circs]); s_save(dict(job_k=jk.job_id())); print("job K (kick measurement) submitted", jk.job_id()); show_job_options(jk, "K")
    sp = clean_sampler(be); assert_clean(sp); jp = sp.run(v1.cal_circuits(be, act), shots=P_SHOTS); s_save(dict(job_p=jp.job_id())); print("job P (readout calibration) submitted", jp.job_id()); show_job_options(jp, "P")
    print("Next: --fetch until both are DONE, then --fire again to submit the main job.")

def mode_fire():
    svc = v1.get_service(); be = svc.backend(BACKEND); st = s_load(); tc, mq, act = setup(be)
    if "job_k" not in st: return fire_calibrations(svc, be, tc, mq, act)
    if "job_main" in st: sys.exit("REFUSED: main job already fired; move state_v2.json aside to refire")
    if not (os.path.exists(KICKF) and os.path.exists(RESULT)): sys.exit("REFUSED: run --fetch first (jobs K and P must be DONE and saved)")
    kick = json.load(open(KICKF)); res = json.load(open(RESULT))
    if kick.get("job_id") != st["job_k"] or "P" not in res or res.get("P_job") != st["job_p"]: sys.exit("REFUSED: saved K or P results do not match the jobs in state_v2.json")
    if st["mq"] != mq or st["act"] != act: sys.exit("REFUSED: layout changed since K and P (the record updated); move state_v2.json aside and redo")
    for j in (st["job_k"], st["job_p"]):
        if "DONE" not in str(svc.job(j).status()).upper(): sys.exit("REFUSED: a calibration job is not DONE")
    rows, bad = sources_rows(be, st, kick, True); print_sources(rows)
    if bad: sys.exit("REFUSED: no hardware source for: %s" % "; ".join(bad))
    kd = {tuple(int(x) for x in k.split("|")): v[0] for k, v in kick["values"].items()}
    ours, ins, exp = make_ours(tc, mq, act, kd); print(check_survive(tc, ours, ins, exp))
    print("kick table applied (degrees per readout, read|register): " + ", ".join("%s %.2f" % (k, v[0]) for k, v in kick["values"].items()))
    u = v1.usage_remaining(svc); main = 2 * v1.estimate_seconds(be, tc)[0] + 4
    if main > MAX_SECONDS or main > u: sys.exit("REFUSED: main estimate %.0f s above cap or remaining" % main)
    order = []
    for c in range(COPIES): order += [("client", c), ("ours", c)] if c % 2 == 0 else [("ours", c), ("client", c)]
    pubs = [((tc if a == "client" else ours), None, SHOTS) for a, c in order]
    for (a, c), p in zip(order, pubs):                                  # re-assert on the exact objects submitted
        if a == "ours": check_survive(tc, p[0], ins, exp)
    sm = clean_sampler(be); assert_clean(sm); print("Sampler options for main: " + options_line(sm.options))
    jm = sm.run(pubs); s_save(dict(job_main=jm.job_id(), order=order, main_options=options_line(sm.options), n_rz=len(ins))); print("main job submitted", jm.job_id()); show_job_options(jm, "main")

def mode_fetch():
    svc = v1.get_service(); st = s_load(); res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    if "job_k" in st and not os.path.exists(KICKF):
        j = svc.job(st["job_k"]); s = str(j.status()); print("job K:", s)
        if "DONE" in s.upper():
            r = j.result(); metas, info = st["k_metas"], {int(k): v for k, v in st["k_info"].items()}
            save_k_raw(r, metas, info); raw = np.load(KRAW); bits = [raw["b%d" % i] for i in range(len(metas))]
            convention_check(); vals, ctl = fit_kick(bits, metas, info)
            okc = all(abs(v[0] - CTL_DEG) <= 0.2 * CTL_DEG for v in ctl.values())
            print("on-chip control (explicit Rz +%.1f deg per step) read back: %s" % (CTL_DEG, {k: "%.2f +- %.2f" % tuple(v) for k, v in ctl.items()}))
            print("control %s" % ("OK (sign and size recovered on hardware)" if okc else "FAILED: main will refuse"))
            json.dump(dict(job_id=st["job_k"], values=vals, control=ctl, control_ok=bool(okc), L_dt={str(r): v["L_dt"] for r, v in info.items()}), open(KICKF, "w"), indent=1)
            print("kick per readout, degrees (read|register: value, sigma, 99%% interval): " + "; ".join("%s: %.2f, %.2f, [%.2f, %.2f]" % (k, *v) for k, v in vals.items()))
            show_job_options(j, "K")
        else: print("job K not finished; run --fetch again later")
    if "job_p" in st and "P" not in res:
        j = svc.job(st["job_p"]); s = str(j.status()); print("job P:", s)
        if "DONE" in s.upper():
            res["P"] = [v1.bitcounts(p) for p in j.result()]; res["P_job"] = st["job_p"]; show_job_options(j, "P")
            try: res["P_seconds"] = j.usage()
            except Exception: pass
        else: print("job P not finished; run --fetch again later")
    if "job_main" in st and "main" not in res:
        j = svc.job(st["job_main"]); s = str(j.status()); print("job main:", s)
        if "DONE" in s.upper():
            r = j.result(); out = {"client": [None] * COPIES, "ours": [None] * COPIES}
            for (a, c), p in zip(st["order"], r): out[a][c] = v1.bitcounts(p)
            res["main"] = out; res["main_job"] = st["job_main"]; show_job_options(j, "main")
            try: res["main_seconds"] = j.usage()
            except Exception: pass
        else: print("job main not finished; run --fetch again later")
    json.dump(res, open(RESULT, "w")); print("saved result_v2.json (keys: %s)" % ", ".join(k for k in res if not k.endswith("seconds")))

def mode_analyze():
    st = s_load(); r = json.load(open(RESULT)); kick = json.load(open(KICKF)); act, mq = st["act"], st["mq"]
    cal = v1.readout_from_counts(r["P"][0], r["P"][1], act, P_SHOTS); rec = {int(q): v for q, v in st["record"].items()}
    print("HARDWARE. kick table from job %s (degrees per readout; read|register: value, sigma, 99%% interval):" % kick["job_id"])
    for k, v in kick["values"].items(): print("  %s: %.2f, %.2f, [%.2f, %.2f]" % (k, *v))
    print("  on-chip control: " + str({k: "%.2f +- %.2f" % tuple(v) for k, v in kick["control"].items()}) + "; cross-check, not applied: " + HIST)
    print("readout, record vs job P (P(1|0), P(0|1)): " + ", ".join("q%d record %.4f / P (%.4f,%.4f)" % (q, rec[q], *cal[q]) for q in act))
    flips = {"N": [(0.0, 0.0)] * M, "R": v1.record_flips(rec, mq), "O": [cal[q] for q in mq]}
    print(report(arm_results(r["main"]["client"], r["main"]["ours"], flips), "HARDWARE. "))
    print("sources: kick job %s, readout job P %s, main job %s, record stamp %s; options main: %s" % (kick["job_id"], r.get("P_job"), r.get("main_job"), st.get("record_stamp"), st.get("main_options")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in ("test", "dry", "fire", "fetch", "analyze", "pilot"): ap.add_argument("--" + m, action="store_true")
    a = ap.parse_args()
    if CHIP and not a.pilot:   # non-kingston full runs get their own files too
        STATE = os.path.join(HERE, "state_v2%s.json" % CHIP); KICKF = os.path.join(HERE, "kick_v2%s.json" % CHIP)
        RESULT = os.path.join(HERE, "result_v2%s.json" % CHIP); KRAW = os.path.join(HERE, "kick_raw_v2%s.npz" % CHIP)
    if a.pilot:      # rule 5 hardware pilot: <= 30 s, 3 copies per arm, own state files; same clean options
        PILOT = True; COPIES = v1.COPIES = 3; K_SHOTS = 1000; P_SHOTS = v1.P_SHOTS = 1000; MAX_SECONDS = 30
        STATE = os.path.join(HERE, "state_v2%s_pilot.json" % CHIP); KICKF = os.path.join(HERE, "kick_v2%s_pilot.json" % CHIP)
        RESULT = os.path.join(HERE, "result_v2%s_pilot.json" % CHIP); KRAW = os.path.join(HERE, "kick_raw_v2%s_pilot.npz" % CHIP)
        print("PILOT MODE: reopened for hardware (the 'global phase' verdict assumed a uniform kick on an ideal circuit); cap %d s" % MAX_SECONDS)
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.fire: mode_fire()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
