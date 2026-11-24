#!/usr/bin/env python
"""CHEMISTRY DEMO v3: H2 by single-ancilla IPE, the client's circuit and layout unchanged; our treatment is Q17, the REAL gate error.
Arms, all on the SAME shots of the SAME circuit (nothing is added to the circuit, so there is no corrective gate to alter):
  N    raw bits as read, no model                          (client default)
  S    per-bit visibility from the vendor's STATED CZ error (generic fix), job-P readout
  O    per-bit visibility from OUR measured CZ error (job-P RB on the edges the circuit uses), job-P readout
  RO   no gate weighting, job-P readout only (rung 2, readout audit; the v1 'O' arm)
  M3   plain per-qubit confusion-matrix inverse of the records (the IBM-standard readout fix), then ML phase (relabel check)
Estimator: maximum-likelihood phase over a grid, with the feedforward replayed and a per-step fringe visibility v_k =
prod over the CZs of step k of (1 - 4e/3) (two-qubit RB convention, e = error per CZ).  e stated vs e measured is the only difference S vs O.
Modes: --test --dry --power --fire [--pilot] --fetch --analyze [--pilot].  --fire submits; nothing else does. Never prints keys/CRNs."""
import os, sys, json, math, time, random, argparse, warnings, collections
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from scipy.optimize import nnls
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford, Clifford

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import demo as v1                       # v1 circuit, estimator pieces, helpers (untouched)

BACKEND, M = os.environ.get("DEMO_BACKEND", v1.BACKEND), v1.M
v1.BACKEND = BACKEND
CHIP = "" if BACKEND == "ibm_kingston" else "_" + BACKEND.replace("ibm_", "")   # per-chip state/run files
COPIES, SHOTS, P_SHOTS = 6, 4000, 4000
RB_LENGTHS, RB_SEQ, RB_SHOTS = [1, 4, 8, 16], 5, 400
MAX_SECONDS = 200
EQUIV_E, EQUIV_F = 0.01, 0.02
PILOT = False
SUFFIX = "_v3" + CHIP
FINANCE_RB_JOB = "dbv6lk7f8rhs73bncf00"      # finance v2 job P (19 Nov), RB on kingston edges 105-106 / 105-117: measured/stated 1.56 and 1.82
RB_RATIOS = [("Oct RB kingston ratio 1.305 (finance/demo.py RB_RATIO, our Oct RB 1.30-1.31)", 1.305),
             ("finance v2 job P %s, 19 Nov, mean of 1.56 and 1.82" % FINANCE_RB_JOB, 1.69)]
KICK_JOB = "dbcuqa0c77vs738739g0"
KICK_DEG, KICK_NOTE = 3.243, "22 Oct dd-run job %s R1: -3.243 deg per readout (dense); ~0.17 deg per readout at 10 reads" % KICK_JOB
PLAN2_KICK_RATIO = 0.0001 / 0.007       # PLAN_v2 exact algebra (ideal circuit, one register qubit given the full 3.243 deg): shift 0.0001 on all-bits-right vs floor 0.007

def paths():
    return (os.path.join(HERE, "state%s.json" % SUFFIX), os.path.join(HERE, "result%s.json" % SUFFIX))
def s_load():
    p = paths()[0]; return json.load(open(p)) if os.path.exists(p) else {}
def s_save(d):
    cur = s_load(); cur.update(d); json.dump(cur, open(paths()[0], "w"), indent=1)

# ------------------------------------------------------------ options (identical for job P and main)
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
    except Exception as ex: print("%s: could not read options back (%s)" % (label, type(ex).__name__))

# ------------------------------------------------------------ circuit facts
def cz_per_step(tc):
    """{step: {edge: count}} from the transpiled client circuit; step k ends at the k-th measure."""
    seg = 0; out = collections.defaultdict(collections.Counter)
    for ci in tc.data:
        n = ci.operation.name; qs = tuple(tc.find_bit(q).index for q in ci.qubits)
        if n in ("cz", "ecr", "cx"): out[seg][tuple(sorted(qs))] += 1
        if n == "measure": seg += 1
        if n == "if_else":
            for b in ci.operation.blocks:
                for cj in b.data:
                    if cj.operation.name in ("cz", "ecr", "cx"): sys.exit("REFUSED: two-qubit gate inside a feedforward block; per-step count would be wrong")
    return out
def circuit_edges(tc):
    return sorted({e for st in cz_per_step(tc).values() for e in st})
def stated_cz(be, edges):
    t = be.target; out = {}
    for a, b in edges:
        try: out[(a, b)] = float(t["cz"][(a, b)].error)
        except Exception: out[(a, b)] = float(t["cz"][(b, a)].error)
    return out
def step_vis(tc, errs):
    st = cz_per_step(tc); v = []
    for k in range(M):
        x = 1.0
        for e, n in st.get(k, {}).items(): x *= (1 - 4 * errs[e] / 3) ** n
        v.append(x)
    return v

# ------------------------------------------------------------ job P: RB on the used edges + readout table, one job, clean options
def rb_circuits(be, edges, seed=11):
    rng = random.Random(seed); circs, meta = [], []
    for e in edges:
        for m in RB_LENGTHS:
            for s in range(RB_SEQ):
                qc = QuantumCircuit(2); comp = Clifford(QuantumCircuit(2))
                for _ in range(m):
                    c = random_clifford(2, seed=rng.randrange(10 ** 9)); qc.compose(c.to_circuit(), inplace=True); comp = comp.compose(c)
                qc.compose(comp.adjoint().to_circuit(), inplace=True); qc.measure_all()
                tc = transpile(qc, backend=be, initial_layout=list(e), optimization_level=0, seed_transpiler=5)
                circs.append(tc); meta.append((e, m, v1.n2q(tc)))
    return circs, meta
def rb_fit(counts_list, meta):
    out = {}
    for e in sorted(set(x[0] for x in meta)):
        xs, ys = [], []
        for c, (ee, m, nz) in zip(counts_list, meta):
            if ee != e: continue
            xs.append(nz); ys.append(c.get("00", 0) / sum(c.values()))
        xs, ys = np.array(xs, float), np.array(ys); best = None
        for A in np.linspace(0.60, 0.80, 41):
            for r in np.linspace(0.90, 1.0, 1001):
                sse = ((0.25 + A * r ** xs - ys) ** 2).sum()
                if best is None or sse < best[0]: best = (sse, r)
        out[e] = 0.75 * (1 - best[1])
    return out
def job_p_pubs(be, edges, act):
    rbc, meta = rb_circuits(be, edges); cal = v1.cal_circuits(be, act)
    return [(c, None, RB_SHOTS) for c in rbc] + [(c, None, P_SHOTS) for c in cal], meta

# ------------------------------------------------------------ estimators with per-step visibility
def rec_probs_v(phi, flips, vis):
    phi = np.atleast_1d(np.asarray(phi, float)); G = len(phi); out = np.zeros((G, 2 ** M)); prefixes = {(): np.ones(G)}
    for k in range(M):
        new = {}
        for pre, pr in prefixes.items():
            om = -math.pi * sum(b / 2 ** (k - j) for j, b in enumerate(pre))
            alpha = 2 * math.pi * 2 ** (M - 1 - k) * phi + om
            p0 = 0.5 + (np.cos(alpha / 2) ** 2 - 0.5) * vis[k]; a01, a10 = flips[k]
            r0 = p0 * (1 - a01) + (1 - p0) * a10
            new[pre + (0,)] = pr * r0; new[pre + (1,)] = pr * (1 - r0)
        prefixes = new
    for pre, pr in prefixes.items(): out[:, sum(b << j for j, b in enumerate(pre))] = pr
    return out
def estimate_v(cv, flips, vis):
    P = rec_probs_v(v1.PHI_GRID, flips, vis); L = (1 - v1.EPS_FLOOR) * P + v1.EPS_FLOOR / 2 ** M
    phi = v1.PHI_GRID[int(np.argmax((np.log(L) * cv[None, :]).sum(axis=1)))]
    Pb = rec_probs_v(np.arange(2 ** M) / 2 ** M, flips, vis); q, _ = nnls(Pb.T, cv / cv.sum())
    return v1.phi_to_E(phi), float(q[v1.B0] / q.sum())
def estimate_m3(cv, flips):
    """Plain M3-style fix: invert the per-step confusion on the record string (no feedforward replay), then ML phase on the ideal model."""
    p = cv / cv.sum()
    for k in range(M):
        a01, a10 = flips[k]; C = np.array([[1 - a01, a10], [a01, 1 - a10]]); Ci = np.linalg.inv(C)
        p = p.reshape([2] * M)       # axis order: index bits are big-endian in reshape; step k is bit k (LSB first)
        ax = M - 1 - k; p = np.moveaxis(np.tensordot(Ci, np.moveaxis(p, ax, 0), axes=(1, 0)), 0, ax).reshape(-1)
    p = np.clip(p, 0, None); p /= p.sum()
    return estimate_v(p * 4000, [(0.0, 0.0)] * M, [1.0] * M)
def arms(copies, flips_p, vis_s, vis_o):
    """copies: list of count dicts. Returns {arm: [(energy, frac) per copy]}."""
    z = [(0.0, 0.0)] * M; one = [1.0] * M; vs = [v1.vec(c) for c in copies]
    return {"N": [estimate_v(v, z, one) for v in vs], "S": [estimate_v(v, flips_p, vis_s) for v in vs],
            "O": [estimate_v(v, flips_p, vis_o) for v in vs], "RO": [estimate_v(v, flips_p, one) for v in vs],
            "M3": [estimate_m3(v, flips_p) for v in vs]}

def t99(n): return float(stats.t.ppf(0.995, n - 1))
def word(lo, hi, eq):
    if lo > -eq and hi < eq: return "CONSISTENT"
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "CANNOT_TELL"
def hw(x): n = len(x); return t99(n) * np.std(x, ddof=1) / math.sqrt(n)
LAB = {"N": "N  raw (client)", "S": "S  stated CZ error weighting", "O": "O  measured CZ error weighting (ours)", "RO": "RO readout table only", "M3": "M3 plain confusion inverse"}

def report(res, header=""):
    n = len(res["N"]); L = [header + "exact ground energy (electronic) %.5f Ha; %d bits; exact bin %d/%d; %d copies" % (v1.E0, M, v1.B0, 2 ** M, n)]
    err = {a: np.array([abs(x[0] - v1.E0) for x in res[a]]) for a in res}; fr = {a: np.array([x[1] for x in res[a]]) for a in res}
    for a in res:
        L.append("  %s: energy error %.1f mHa (99%% interval %.1f to %.1f); all bits right %.4f (99%% interval %.4f to %.4f)" % (
            LAB[a], err[a].mean() * 1e3, max(err[a].mean() - hw(err[a]), 0) * 1e3, (err[a].mean() + hw(err[a])) * 1e3, fr[a].mean(), fr[a].mean() - hw(fr[a]), fr[a].mean() + hw(fr[a])))
    for other in ("S", "N", "RO", "M3"):
        for lab, d, eq, unit in (("energy error", err["O"] - err[other], EQUIV_E, "Ha"), ("fraction right", fr["O"] - fr[other], EQUIV_F, "")):
            lo, hi = d.mean() - hw(d), d.mean() + hw(d)
            w = word(-hi, -lo, eq) if lab == "fraction right" else word(lo, hi, eq)
            L.append("  OUTCOME O minus %s, %s: mean difference %+.4f%s (99%% interval %+.4f to %+.4f; paired by copy; ABOVE = ours worse, BELOW = ours better; CONSISTENT inside +-%s): %s" % (other, lab, d.mean(), unit, lo, hi, eq, w))
    dE = np.abs(err["O"] - err["S"]); dF = np.abs(fr["O"] - fr["S"])
    fE, fF = hw(err["S"]), hw(fr["S"])
    L.append("  PILOT/MEASURED |O-S|: energy %.2f mHa vs copy floor (99%% half-width of the S copies) %.2f mHa -> ratio %.2f; all-bits-right %.4f vs %.4f -> ratio %.2f" % (
        dE.mean() * 1e3, fE * 1e3, dE.mean() / max(fE, 1e-12), dF.mean(), fF, dF.mean() / max(fF, 1e-12)))
    return "\n".join(L)

# ------------------------------------------------------------ POWER from our own v1 hardware counts (--power)
def model_ratio(v1counts, flips_p, tc, stated, rho):
    vs = step_vis(tc, stated); vo = step_vis(tc, {e: stated[e] * rho for e in stated}); one = [1.0] * M
    cv = [v1.vec(c) for c in v1counts]
    eS = np.array([abs(estimate_v(v, flips_p, vs)[0] - v1.E0) for v in cv]); eO = np.array([abs(estimate_v(v, flips_p, vo)[0] - v1.E0) for v in cv])
    fS = np.array([estimate_v(v, flips_p, vs)[1] for v in cv]); fO = np.array([estimate_v(v, flips_p, vo)[1] for v in cv])
    sig_E, sig_F = np.abs(eO - eS).mean(), np.abs(fO - fS).mean()
    # paired floor: the v1 copies' own spread of the paired difference (and of the S estimate itself, the larger is used)
    fl_E = max(hw(eO - eS), hw(eS)); fl_F = max(hw(fO - fS), hw(fS))
    return sig_E, fl_E, sig_F, fl_F, vs, vo
def mode_power():
    if BACKEND != "ibm_kingston":
        be = v1.get_service().backend(BACKEND); tc, mq, act = v1.setup(be); edges = circuit_edges(tc); stated = stated_cz(be, edges); cz = cz_per_step(tc)
        print("POWER on %s: client layout read %s active %s; CZ edges %s; CZ per bit %s; stated CZ error %s" % (BACKEND, mq, act, edges, [sum(cz[k].values()) for k in range(M)], {e: round(v, 5) for e, v in stated.items()}))
        print("No v1 chemistry fire exists on %s (closet: only the 19 Nov kingston fire), so there is no hardware floor and no model ratio can be formed." % BACKEND)
        print("POWER RATIO: none computable on %s" % BACKEND); print("PILOT NEEDED: run --fire --pilot on %s (its own copy floor and measured |O-S|); kingston model ratio 2.78 is not transferable." % BACKEND); return
    st = json.load(open(os.path.join(HERE, "state.json"))); r = json.load(open(os.path.join(HERE, "result.json")))
    try: be = v1.get_service().backend(BACKEND); src = "live backend target"
    except SystemExit: raise
    except Exception: be = v1.fake_backend(); src = "fake backend target (offline)"
    tc, mq, act = v1.setup(be)
    if mq != st["mq"] or act != st["act"]: sys.exit("REFUSED: today's client layout differs from the v1 fire; power not comparable")
    edges = circuit_edges(tc); stated = stated_cz(be, edges); cz = cz_per_step(tc)
    print("POWER inputs: client layout read %s active %s (%s); CZ edges %s" % (mq, act, src, edges))
    print("CZs per phase bit (step k = record bit k; steps are NOT depth-ordered: each U^(2^p) is a general controlled unitary):")
    for k in range(M): print("  bit %d: %s" % (k, dict((str(e), n) for e, n in cz[k].items())))
    print("stated CZ error (target): " + ", ".join("%s %.5f" % (e, stated[e]) for e in edges))
    cal = v1.readout_from_counts(r["P"][0], r["P"][1], act, v1.P_SHOTS); flips_p = [cal[q] for q in mq]
    rec = {int(q): v for q, v in st["record"].items()}
    print("v1 data: job_main %s, job_p %s (19 Nov fire_log.text), 6 copies of 4000 shots (hardware counts, re-read with a model)" % (st["job_main"], st["job_p"]))
    rungs = {}
    for lab, rho in RB_RATIOS:
        sE, fE, sF, fF, vs, vo = model_ratio(r["main"], flips_p, tc, stated, rho)
        rE, rF = sE / fE, sF / fF; rungs.setdefault("1", []).append(max(rE, rF))
        print("RUNG 1 (Q17 gate-error weighting), measured/stated = %s:" % lab)
        print("   vis stated %s / measured %s; |O-S| energy %.3f mHa vs paired floor %.3f mHa -> %.2f ; all-bits-right %.5f vs %.5f -> %.2f" % (
            ["%.4f" % x for x in vs], ["%.4f" % x for x in vo], sE * 1e3, fE * 1e3, rE, sF, fF, rF))
    # rung 2: readout audit per bit, measured on the v1 hardware (job P vs record), same copies
    cv = [v1.vec(c) for c in r["main"]]; one = [1.0] * M; recf = v1.record_flips(rec, mq)
    eR = np.array([abs(estimate_v(v, recf, one)[0] - v1.E0) for v in cv]); eP = np.array([abs(estimate_v(v, flips_p, one)[0] - v1.E0) for v in cv])
    fR = np.array([estimate_v(v, recf, one)[1] for v in cv]); fP = np.array([estimate_v(v, flips_p, one)[1] for v in cv])
    e3 = np.array([abs(estimate_m3(v, flips_p)[0] - v1.E0) for v in cv]); f3 = np.array([estimate_m3(v, flips_p)[1] for v in cv])
    sE, sF = np.abs(eP - eR).mean(), np.abs(fP - fR).mean(); flE, flF = max(hw(eP - eR), hw(eR)), max(hw(fP - fR), hw(fR))
    r2 = max(sE / flE, sF / flF); rungs["2"] = [r2]
    print("RUNG 2 (per-bit readout audit, job-P table vs record, feedforward-aware likelihood; both readings re-read from v1 HARDWARE counts, not a model of the chip): |P-record| energy %.2f mHa vs floor %.2f mHa -> %.2f; all-bits-right %.4f vs %.4f -> %.2f" % (sE * 1e3, flE * 1e3, sE / flE, sF, flF, sF / flF))
    print("   M3 relabel check (same job-P table, plain confusion inverse, no feedforward replay): M3 energy error %.2f mHa, frac %.4f; feedforward-aware %.2f mHa, frac %.4f; |difference| %.2f mHa and %.4f -> %s" % (
        e3.mean() * 1e3, f3.mean(), eP.mean() * 1e3, fP.mean(), np.abs(e3 - eP).mean() * 1e3, np.abs(f3 - fP).mean(),
        "NOT just M3 (the feedforward replay changes the answer)" if (np.abs(e3 - eP).mean() > 0.001 or np.abs(f3 - fP).mean() > 0.005) else "equal to M3 within 1 mHa and 0.005: this rung IS M3 relabelled, not ours"))
    # rung 3: kick on a non-parity-symmetric register
    rungs["3"] = [PLAN2_KICK_RATIO]
    print("RUNG 3 (kick on a register where only one register qubit sits beside the ancilla): model ratio %.3f. Basis: PLAN_v2 exact algebra on the ideal circuit, one register qubit given the full %.3f deg per readout (%s): shift 0.0001 vs floor 0.007. Ideal-circuit number: it cannot drop the rung (rule 5)." % (PLAN2_KICK_RATIO, KICK_DEG, KICK_NOTE))
    best = max(rungs, key=lambda k: max(rungs[k])); ratio = max(rungs[best])
    names = {"1": "Q17 real gate error weighting", "2": "per-bit readout audit", "3": "kick on a non-parity-symmetric register"}
    print("")
    for k in "123":
        tag = ("rejected on the model, kept for the pilot" if k != best else "chosen")
        print("RUNG %s %s: model ratio %s (%s)%s" % (k, names[k], " / ".join("%.2f" % x for x in rungs[k]), tag, "  PILOT NEEDED" if max(rungs[k]) < 3 else ""))
    print("POWER RATIO: %.2f" % ratio)
    if ratio < 3: print("PILOT NEEDED: model ratio under 3 on every rung; a model number cannot drop a design (rule 5). Fire --pilot (<= 30 s) and read the measured |O-S| against the pilot's own floor.")

# ------------------------------------------------------------ seconds
def plan_seconds(be, tc, nP):
    main, calp, dur, _ = v1.estimate_seconds(be, tc)       # uses v1.COPIES / v1.SHOTS
    rb = sum(RB_SHOTS * (0.0004 + 1e-6 * 40) for _ in range(nP))
    return rb + calp + 4, main + 3, rb + calp + main + 7
def mode_test():
    from qiskit_aer import AerSimulator
    be = v1.fake_backend(); tc, mq, act = v1.setup(be); edges = circuit_edges(tc); stated = stated_cz(be, edges)
    print("[SMOKE TEST: offline code-path check only, fake %s, no number here is a result]" % BACKEND)
    print("layout read %s active %s; CZ edges %s; CZ per step %s" % (mq, act, edges, [sum(cz_per_step(tc).get(k, {}).values()) for k in range(M)]))
    pubs, meta = job_p_pubs(be, edges, act); nrb = len(meta); sim = AerSimulator()
    cl = [sim.run(p[0], shots=100, seed_simulator=3 + i).result().get_counts() for i, p in enumerate(pubs)]
    meas = rb_fit(cl[:nrb], meta); print("RB fit path ran on %d circuits for edges %s" % (nrb, sorted(meas)))
    cal = v1.readout_from_counts(cl[nrb], cl[nrb + 1], act, 100); flips = [cal[q] for q in mq]
    vs, vo = step_vis(tc, stated), step_vis(tc, meas)
    ideal = sim.run(transpile(v1.ipe_circuit(), sim, optimization_level=0), shots=400, seed_simulator=1).result().get_counts()
    res = arms([ideal, ideal, ideal], flips, vs, vo); out = report(res, "SMOKE. ")
    print("analysis path executed on %d lines (numbers withheld)" % len(out.splitlines()))
    print("options guard:", end=" ")
    class O: pass
    try: assert_clean(type("S", (), {"options": type("o", (), {"twirling": type("t", (), {"enable_gates": True, "enable_measure": False})(), "dynamical_decoupling": type("d", (), {"enable": False})()})()})()); print("FAILED to refuse")
    except SystemExit: print("a dirty option is refused: OK")
    v1.COPIES, v1.SHOTS = COPIES, SHOTS; print("seconds path: P / main / total = %s" % [round(x, 1) for x in plan_seconds(be, tc, nrb)])

def mode_dry():
    svc = v1.get_service(); u = v1.usage_remaining(svc); be = svc.backend(BACKEND); s = be.status()
    tc, mq, act = v1.setup(be); edges = circuit_edges(tc); stated = stated_cz(be, edges); pubs, meta = job_p_pubs(be, edges, act)
    print("backend %s operational=%s pending_jobs=%s; usage_remaining_seconds=%.0f" % (BACKEND, s.operational, s.pending_jobs, u))
    print("client circuit (level-3 transpile on the live record, used UNCHANGED for both arms): depth %d, CZs %d; read qubit per step %s; active %s" % (tc.depth(), v1.n2q(tc), mq, act))
    print("CZ edges %s; stated error %s; RB circuits %d (level 0 after layout, no barriers merged)" % (edges, {e: round(v, 5) for e, v in stated.items()}, len(meta)))
    print("corrective gates: none added; the main circuit is the client's, so 'unchanged' holds by construction (asserted at fire by gate-count equality of the submitted object)")
    for lab in ("P", "main"):
        sp = clean_sampler(be); assert_clean(sp); print("Sampler options for job %s: %s" % (lab, options_line(sp.options)))
    P, Mn, tot = plan_seconds(be, tc, len(meta)); print("estimated seconds: P %.0f + main %.0f = %.0f (cap %d%s)" % (P, Mn, tot, MAX_SECONDS, ", pilot" if PILOT else ""))
    if tot > MAX_SECONDS or tot > u: print("WOULD REFUSE")

def mode_fire():
    svc = v1.get_service(); be = svc.backend(BACKEND); u = v1.usage_remaining(svc); tc, mq, act = v1.setup(be)
    edges = circuit_edges(tc); stated = stated_cz(be, edges); pubs, meta = job_p_pubs(be, edges, act)
    P, Mn, tot = plan_seconds(be, tc, len(meta)); print("estimated seconds P %.0f main %.0f total %.0f; cap %d; remaining %.0f" % (P, Mn, tot, MAX_SECONDS, u))
    if tot > MAX_SECONDS or tot > u: sys.exit("REFUSED: estimate above cap or remaining seconds")
    if "job_main" in s_load(): sys.exit("REFUSED: %s already holds fired jobs; move it aside to refire" % os.path.basename(paths()[0]))
    rec = v1.record_readout(be, act); n0 = len(tc.data); ops0 = tc.count_ops()
    s_save(dict(mq=mq, act=act, edges=[list(e) for e in edges], stated={"%d-%d" % e: v for e, v in stated.items()}, record={str(q): rec[q] for q in act},
                meta=[[list(e), m, nz] for e, m, nz in meta], n_rb=len(meta), pilot=PILOT, copies=COPIES, fired_at=time.time(), est_seconds=tot,
                stamp=str(getattr(be.properties(), "last_update_date", "?")) if hasattr(be, "properties") else "?"))
    print("SOURCES: stated CZ error <- live record target (stamp %s); measured CZ error <- job P RB (filled at fire, applied only at --analyze); readout table <- job P; layout <- client's own transpile (their input)" % s_load()["stamp"])
    sp = clean_sampler(be); assert_clean(sp); jp = sp.run(pubs); s_save(dict(job_p=jp.job_id(), p_options=options_line(sp.options))); print("job P (RB on used edges + readout table) submitted", jp.job_id()); show_job_options(jp, "P")
    assert len(tc.data) == n0 and tc.count_ops() == ops0, "client circuit changed"
    sm = clean_sampler(be); assert_clean(sm); jm = sm.run([(tc, None, SHOTS)] * COPIES)
    s_save(dict(job_main=jm.job_id(), main_options=options_line(sm.options))); print("main job submitted", jm.job_id()); show_job_options(jm, "main")
    print("Same clean options for P and main:", s_load()["p_options"] == s_load()["main_options"])
    print("next: --fetch%s until DONE, then --analyze%s" % (" --pilot" if PILOT else "", " --pilot" if PILOT else ""))

def mode_fetch():
    svc = v1.get_service(); st = s_load(); out = {}
    for key, name in (("job_p", "P"), ("job_main", "main")):
        j = svc.job(st[key]); s = str(j.status()); print("job %s: %s" % (name, s))
        if "DONE" not in s.upper(): sys.exit("not finished; run --fetch again later")
        out[name] = [v1.bitcounts(p) for p in j.result()]
        try: out[name + "_seconds"] = j.usage()
        except Exception: pass
    out["P_job"], out["main_job"] = st["job_p"], st["job_main"]; json.dump(out, open(paths()[1], "w")); print("saved %s; seconds used: %s" % (os.path.basename(paths()[1]), {k: v for k, v in out.items() if k.endswith("_seconds")}))

def mode_analyze():
    st = s_load(); r = json.load(open(paths()[1])); act, mq = st["act"], st["mq"]; nrb = st["n_rb"]
    meta = [(tuple(e), m, nz) for e, m, nz in st["meta"]]; meas = rb_fit(r["P"][:nrb], meta)
    stated = {tuple(int(x) for x in k.split("-")): v for k, v in st["stated"].items()}
    cal = v1.readout_from_counts(r["P"][nrb], r["P"][nrb + 1], act, P_SHOTS); flips = [cal[q] for q in mq]
    be = v1.get_service().backend(BACKEND); tc, mq2, act2 = v1.setup(be)
    if mq2 != mq or act2 != act: sys.exit("REFUSED: today's client layout differs from the fired one; rebuild from the saved state is not possible")
    vs, vo = step_vis(tc, stated), step_vis(tc, meas)
    print("pair: stated / RB-measured / ratio: " + ", ".join("%s %.5f / %.5f / %.2f" % (e, stated[e], meas[e], meas[e] / stated[e]) for e in sorted(stated)))
    print("per-bit visibility stated %s measured %s" % (["%.4f" % x for x in vs], ["%.4f" % x for x in vo]))
    print("readout, record vs job P (P(1|0), P(0|1)): " + ", ".join("q%d record %.4f / P (%.4f,%.4f)" % (q, st["record"][str(q)], *cal[q]) for q in act))
    print(report(arms(r["main"], flips, vs, vo), "HARDWARE%s. " % (" PILOT" if st.get("pilot") else "")))
    print("sources: job P %s (RB + readout), main job %s, stated <- target stamp %s; options main: %s" % (r["P_job"], r["main_job"], st.get("stamp"), st.get("main_options")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in ("test", "dry", "power", "fire", "fetch", "analyze", "pilot"): ap.add_argument("--" + m, action="store_true")
    a = ap.parse_args()
    if a.pilot:      # rule 5 pilot: <= 30 s, 3 copies, shorter RB, own state files, same clean options for P and main
        PILOT = True; SUFFIX = "_v3" + CHIP + "_pilot"; COPIES = 3; P_SHOTS = 1000; RB_SEQ, RB_SHOTS = 3, 300; MAX_SECONDS = 30
        print("PILOT MODE (rule 5): both arms on the same shots, runtime changes off, cap %d s" % MAX_SECONDS)
    v1.COPIES, v1.SHOTS, v1.P_SHOTS = COPIES, SHOTS, P_SHOTS
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.power: mode_power()
    elif a.fire: mode_fire()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
