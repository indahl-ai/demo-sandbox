#!/usr/bin/env python
"""Demo 3 v3 (logistics, p=1 QAOA max-cut on the chip's own coupler graph), ladder rung 1, IBM code Q17:
the client's measured cut is shrunk by gate error. Three arms on the SAME shots of the client's circuit:
  N  raw (client)          S  cut rescaled by a fade predicted from IBM's STATED per-edge CZ error (record, stamp printed)
  O  cut rescaled by the same fade built from OUR per-edge CZ error (job P: parallel RB in the circuit's own colour layers)
FADE MODEL (written before data, no parameter fitted on the main shots):
  <Z_u Z_v> measured = F_uv x <Z_u Z_v> ideal,  F_uv = product over every CZ in the p=1 light cone of edge (u,v) of r_g,
  light cone = the 2 CZ of every graph edge sharing a node with (u,v) (its own included),  r_g = 1 - 4 e_g / 3  (2-qubit depolarizing
  parameter for an average infidelity e_g).  Corrected <ZZ> = measured / F.  Same model for S and O; only e_g differs.
Readout: NOT corrected in any arm (identical for all three). Runtime changes OFF (twirling, DD; SamplerV2 has no resilience) in job P and main,
same options, read back from each job. Level 0 after the client's layout; every RZZ angle asserted exact; 2 CZ per edge asserted.
Modes: --test (code smoke only) --dry --fire [--pilot] --fetch [--pilot] --analyze [--pilot].
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE. Keys and CRNs never printed. Nothing here commits or pushes."""
import os, sys, json, math, time, argparse, random, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford, Clifford

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import demo as v1
import demo_v2 as v2

BACKEND = v1.BACKEND
RUN = os.path.join(HERE, "run_v3_" + BACKEND.replace("ibm_", ""))   # per-chip state and raw files
SEED = 20261120
CFG = {"full": dict(copies=12, shots=10_000, lengths=[1, 3, 6, 10, 15], seqs=6, rbshots=500),
       "pilot": dict(copies=4, shots=8_000, lengths=[1, 3, 6, 10, 15], seqs=4, rbshots=400)}
MAX_SECONDS, MARGIN, PILOT_SECONDS = 200.0, 1.25, 30.0
SEC_PER_SHOT_CIRCUIT, SEC_PER_JOB = 0.30e-3, 4.0
BAND = 0.001            # CONSISTENT = paired interval wholly inside +-0.001 of ratio (2.5x the v1 99% half-width 0.0004); set before data
FLOOR_V1 = 0.0004       # v1 fire_log.text, 12 copies: N 0.9344-0.9352, R 0.9498-0.9507 -> 99% half-width about 0.0004
RHOS = (1.30, 1.56, 1.82)   # measured/stated CZ error: kingston RB Oct 1.305; 19 Nov 1.56 and 1.82
NDRAW = 100


def fp(n): os.makedirs(RUN, exist_ok=True); return os.path.join(RUN, n)
def s_load(sfx): p = fp("state%s.json" % sfx); return json.load(open(p)) if os.path.exists(p) else {}
def s_save(sfx, d): cur = s_load(sfx); cur.update(d); json.dump(cur, open(fp("state%s.json" % sfx), "w"), indent=1)
def pair(a, b): return (a, b) if a < b else (b, a)


# ------------------------------------------------------------ record, fade model
def stated_errors(be, pairs):
    """IBM's stated CZ error per physical pair (RB average infidelity) from the live target; refuses on a gap."""
    out = {}
    for p in pairs:
        for q in (p, p[::-1]):
            try:
                e = be.target["cz"][q].error
                if e is not None: out[p] = float(e); break
            except Exception: pass
        else: sys.exit("REFUSED: no stated CZ error in the record for pair %s" % (p,))
    return out


def edge_pairs(edges, lay): return [pair(lay[u], lay[v]) for u, v in edges]


def cone(edges, lay):
    """per graph edge: list of physical pairs whose 2 CZ lie in its light cone (every edge sharing a node with it, its own included)."""
    inc = {}
    for i, (u, v) in enumerate(edges): inc.setdefault(u, []).append(i); inc.setdefault(v, []).append(i)
    pp = edge_pairs(edges, lay); out = []
    for u, v in edges: out.append([pp[j] for j in sorted(set(inc[u]) | set(inc[v]))])
    return out


def fade(cn, errs, ncz=2):
    """F per edge = prod r_g^ncz, r_g = 1 - 4 e / 3."""
    return np.array([np.prod([(1 - 4 * errs[p] / 3) ** ncz for p in c]) for c in cn])


BAD_E = 0.05    # pre-set: an edge whose light cone holds a pair the record states above 5% is dropped from EVERY arm (the record calls it unusable)


def keep_mask(cn, stated): return np.array([all(stated[p] <= BAD_E for p in c) for c in cn])


def ratio_of(zz, exact_cut_sum, F=None, keep=None):
    z = zz if F is None else zz / F
    k = np.ones(len(z), bool) if keep is None else keep
    return float(np.sum((1 - z[k]) / 2) / exact_cut_sum)


def power_rows(stated, cn, edges, gamma, beta):
    """Expected paired |O - S| (in ratio) if the chip's per-CZ error is rho x stated on every edge; ZZ0 = exact noiseless p=1 value (algebra only)."""
    zz0 = np.array([v1.zz_p1(edges, gamma, beta)[e] for e in edges]); keep = keep_mask(cn, stated); den = float(np.sum((1 - zz0[keep]) / 2)); rows = []
    FS = fade(cn, stated)
    for rho in RHOS:
        FO = fade(cn, {p: rho * e for p, e in stated.items()})
        zmeas = zz0 * FO
        rO, rS, rN = ratio_of(zmeas, den, FO, keep), ratio_of(zmeas, den, FS, keep), ratio_of(zmeas, den, None, keep)
        rows.append((rho, abs(rO - rS), abs(rO - rN)))
    return rows, float(np.mean(1 - FS))


# ------------------------------------------------------------ job P: parallel RB in the circuit's own colour layers
def colour_pairs(edges, lay):
    pp = dict(zip(edges, edge_pairs(edges, lay)))
    return [[pp[e] for e in col] for col in v2.edge_colour(edges)]


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
                tc = transpile(qc, be, initial_layout=flat, optimization_level=0, seed_transpiler=SEED)
                cnt = {}
                for ci in tc.data:
                    if ci.operation.name == "cz":
                        key = pair(*(tc.find_bit(q).index for q in ci.qubits)); cnt[key] = cnt.get(key, 0) + 1
                circs.append(tc); meta.append(dict(col=k, m=m, s=s, ncz=[cnt.get(p, 0) for p in prs]))
    return circs, meta


def p00_from_bits(bits, npairs): return np.array([int(np.sum((bits[:, 2 * i] == 0) & (bits[:, 2 * i + 1] == 0))) for i in range(npairs)])


RGRID = np.linspace(0.95, 1.0, 1001)
XCACHE = {}


def rb_fit_many(xs, Y):
    """xs: (P,) CZ counts. Y: (D,P) survival fractions. Model P00 = 0.25 + A r^n, A closed-form per r (clipped 0.5-0.8). Returns e per draw = 0.75 (1-r)."""
    X = RGRID[:, None] ** xs[None, :]; den = (X ** 2).sum(1)
    A = np.clip(((Y - 0.25) @ X.T) / den, 0.5, 0.8)                                 # (D,R)
    sse = ((0.25 + A[:, :, None] * X[None] - Y[:, None, :]) ** 2).sum(2)
    r = RGRID[np.argmin(sse, 1)]
    return np.maximum(0.75 * (1 - r), 1e-4)


def measured_errors(cols, meta, K, shots, rng=None, draws=0):
    """K: (ncirc, ...) list of per-circuit k00 arrays over that colour's pairs. Returns {pair: e} and, if draws, {pair: array(draws)}."""
    point, boot = {}, {}
    for k, prs in enumerate(cols):
        idx = [i for i, m in enumerate(meta) if m["col"] == k]
        for j, p in enumerate(prs):
            xs = np.array([meta[i]["ncz"][j] for i in idx], float); y = np.array([K[i][j] / shots for i in idx])
            point[p] = float(rb_fit_many(xs, y[None])[0])
            if draws: boot[p] = rb_fit_many(xs, rng.binomial(shots, np.clip(y, 0, 1)[None].repeat(draws, 0)) / shots)
    return point, boot


# ------------------------------------------------------------ the main circuit
def prepare(be, n):
    x = v2.setup(be, n); edges, lay = x["edges"], x["layout"]
    ang = [x["gamma"]] * len(edges); bound = v2.bind(x["st"], x["pv"], ang); v2.verify_angles(x["st"], bound, ang)
    cnt = {}
    for ci in bound.data:
        if ci.operation.name == "cz":
            key = pair(*(bound.find_bit(q).index for q in ci.qubits)); cnt[key] = cnt.get(key, 0) + 1
    if any(cnt.get(p, 0) != 2 for p in edge_pairs(edges, lay)): sys.exit("REFUSED: not 2 CZ on every graph edge")
    x["bound"] = bound; x["cn"] = cone(edges, lay); x["cols"] = colour_pairs(edges, lay)
    return x


def est_p(ncol, cfg): return SEC_PER_SHOT_CIRCUIT * ncol * len(cfg["lengths"]) * cfg["seqs"] * cfg["rbshots"] + SEC_PER_JOB
def est_main(cfg): return SEC_PER_SHOT_CIRCUIT * cfg["copies"] * cfg["shots"] + SEC_PER_JOB


def choose_n(be, cfg, nmax=100):
    for n in range(nmax, 9, -10):
        ed = v2.graph_for(be, n); tot = est_p(len(v2.edge_colour(ed)), cfg) + est_main(cfg)
        if tot * MARGIN <= MAX_SECONDS: return n, tot
    sys.exit("REFUSED: no size fits under %d s" % MAX_SECONDS)


def sources_rows(be, x, pjob, stated):
    stamp = v2.record_stamp(be)
    return [("stated CZ error per edge (RB average infidelity)", "live record %s, target cz error, %d edges" % (stamp, len(stated)) if stamp and stated else "MISSING"),
            ("measured CZ error per edge", "job P %s (parallel RB in the circuit's colour layers)" % pjob if pjob else "MISSING"),
            ("fade model (light cone, r = 1 - 4e/3)", "written before data; no fitted parameter; counts per edge from the transpiled circuit"),
            ("readout", "not corrected in any arm"),
            ("layout, graph, angles", "client's own (level-3 layout on the record; exact p=1 optimum): not our parameters")]


def print_sources(rows):
    print("SOURCES (parameter -> hardware source):")
    for r in rows: print("  %-50s %s" % r)


def model_power(be, x):
    stated = stated_errors(be, sorted({p for c in x["cn"] for p in c}))
    rows, mean_fade = power_rows(stated, x["cn"], x["edges"], x["gamma"], x["beta"])
    return stated, rows, mean_fade


# ------------------------------------------------------------ analysis
def word(lo, hi):
    if lo != lo: return "CANNOT_TELL"
    if lo > -BAND and hi < BAND: return "CONSISTENT"
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "CANNOT_TELL"


def arms_analysis(copy_bits, edges, exact, cn, stated, cols, meta, K, rbshots):
    keep = keep_mask(cn, stated); exact_sum = float(sum((1 - exact[e]) / 2 for e, k in zip(edges, keep) if k))
    ZZ = np.array([[v1.marg_bits(b, edges)[1][e] for e in edges] for b in copy_bits])             # copies x edges
    meas, boot = measured_errors(cols, meta, K, rbshots, np.random.default_rng(SEED), NDRAW)
    FS, FO = fade(cn, stated), fade(cn, meas)
    rN = np.array([ratio_of(z, exact_sum, None, keep) for z in ZZ]); rS = np.array([ratio_of(z, exact_sum, FS, keep) for z in ZZ]); rO = np.array([ratio_of(z, exact_sum, FO, keep) for z in ZZ])
    zm = ZZ.mean(0); od = []
    for d in range(NDRAW):
        FOd = fade(cn, {p: boot[p][d] for p in boot}); od.append(ratio_of(zm, exact_sum, FOd, keep))
    sd = float(np.std(od, ddof=1))
    return dict(N=rN, S=rS, O=rO, sd_rb=sd, meas=meas, FS=FS, FO=FO, copies=len(rN), kept=int(keep.sum()), n_edges=len(keep))


def report(res, ideal_over_max, label=""):
    f = lambda v, sdf=0.0: "%.4f (99%% interval %.4f to %.4f)" % v1.interval99(v, sdf)
    L = ["%sapproximation ratio = measured cut / noiseless QAOA cut (noiseless QAOA itself reaches %.3f of the exact max cut); %d copies; %d of %d edges kept (edges whose light cone holds a pair the record states above %.0f%% are dropped from every arm)" % (label, ideal_over_max, res["copies"], res["kept"], res["n_edges"], BAD_E * 100),
         "%s  N raw (client):                                  %s" % (label, f(res["N"])),
         "%s  S fade from IBM's stated CZ error:               %s" % (label, f(res["S"])),
         "%s  O fade from our measured CZ error (job P):       %s  [job-P RB error sd %.5f included]" % (label, f(res["O"], res["sd_rb"]), res["sd_rb"])]
    out = {}
    for b, nm in (("S", "O minus S"), ("N", "O minus N")):
        d = res["O"] - res[b]; m, lo, hi = v1.interval99(d, res["sd_rb"]); out[b] = (m, lo, hi)
        L.append("%s  OUTCOME %s: %+.4f (99%% interval %+.4f to %+.4f, paired by copy, RB error in quadrature; ABOVE = ours higher, BELOW = ours lower, CONSISTENT = inside +-%.3f): %s" % (label, nm, m, lo, hi, BAND, word(lo, hi)))
    ms = np.array([res["meas"][p] / s for p, s in zip(res["meas"], [res["stated_of"][p] for p in res["meas"]])])
    L.append("%s  measured/stated CZ error over %d pairs: median %.2f (min %.2f max %.2f)" % (label, len(ms), np.median(ms), ms.min(), ms.max()))
    return "\n".join(L), out


def analyze(sfx):
    st = s_load(sfx); cfg = CFG["pilot" if sfx else "full"]; edges = [tuple(e) for e in st["edges"]]; lay = st["layout"]
    cn = cone(edges, lay); cols = [[tuple(p) for p in c] for c in st["cols"]]; meta = st["meta"]; stated = {tuple(map(int, k.split("-"))): v for k, v in st["stated"].items()}
    zp = np.load(fp("raw_P%s.npz" % sfx)); K = [zp["k_%d" % i] for i in range(len(meta))]
    zm = np.load(fp("raw_main%s.npz" % sfx)); bits = [zm["c_%d" % c] for c in range(cfg["copies"])]
    exact = v1.zz_p1(edges, st["gamma"], st["beta"])
    res = arms_analysis(bits, edges, exact, cn, stated, cols, meta, K, cfg["rbshots"]); res["stated_of"] = stated
    txt, out = report(res, st["ideal"] / st["maxcut"], "PILOT " if sfx else "HARDWARE ")
    extra = ""
    if sfx:
        d = res["O"] - res["S"]; fl = float(np.hypot(stats_half(d), res["sd_rb"] * 5.841)); m = abs(float(d.mean()))
        dN = res["O"] - res["N"]; flN = float(np.hypot(stats_half(dN), res["sd_rb"] * 5.841)); mN = abs(float(dN.mean()))
        extra = "\nPILOT RATIO O-S: measured |O-S| %.5f over the pilot's own copy floor %.5f (99%% half-width of the paired copies plus RB error) = %.2f\nPILOT RATIO O-N: |O-N| %.5f over floor %.5f = %.2f" % (m, fl, m / fl, mN, flN, mN / flN)
        json.dump(dict(ratio_OS=m / fl, abs_OS=m, floor_OS=fl, job_p=st.get("job_p"), job_main=st.get("job_main")), open(fp("pilot_result.json"), "w"))
    txt += extra + "\nsources: record stamp %s, job P %s, main job %s; options main: %s" % (st.get("record_stamp"), st.get("job_p"), st.get("job_main"), st.get("main_options"))
    print(txt); open(fp("results%s.text" % sfx), "w").write(txt + "\n")


def stats_half(d):
    from scipy import stats
    n = len(d); return float(stats.t.ppf(0.995, n - 1) * np.std(d, ddof=1) / math.sqrt(n))


# ------------------------------------------------------------ io
def save_main(sfx, res, n, cfg):
    np.savez_compressed(fp("raw_main%s.npz" % sfx), **{"c_%d" % c: v2.unpack(res[c].data.c.array, n) for c in range(cfg["copies"])})


def save_p(sfx, res, cols, meta):
    arrs = {}
    for i, m in enumerate(meta): arrs["k_%d" % i] = p00_from_bits(v2.unpack(res[i].data.c.array, 2 * len(cols[m["col"]])), len(cols[m["col"]]))
    np.savez_compressed(fp("raw_P%s.npz" % sfx), **arrs)


def mode_test():
    """CODE SMOKE TEST ONLY. Offline, small graph, fake chip, ideal Aer. Plumbing only: no result, no headline, no constant."""
    from qiskit_aer import AerSimulator
    be = v1.fake_backend(); n = 12; print("SMOKE TEST ONLY (code plumbing; no result, no headline, no constant comes from this)")
    x = prepare(be, n); edges, lay = x["edges"], x["layout"]; cfg = dict(copies=3, shots=300, lengths=[1, 3], seqs=2, rbshots=200)
    print("graph %d nodes %d edges; level-0 CZ %d; colour layers %s" % (n, len(edges), x["bound"].count_ops().get("cz", 0), [len(c) for c in x["cols"]]))
    stated, rows, _ = model_power(be, x); print("light-cone and power path executed (%d rho rows; numbers withheld)" % len(rows))
    sim = AerSimulator(seed_simulator=3); circs, meta = rb_circuits(be, x["cols"], cfg)
    K = []; ok = True
    for i, c in enumerate(circs):
        r = sim.run(c, shots=cfg["rbshots"], seed_simulator=i).result().get_counts(); bits = v1.bits_from_counts(r, 2 * len(x["cols"][meta[i]["col"]]))
        k = p00_from_bits(bits, len(x["cols"][meta[i]["col"]])); K.append(k); ok &= bool((k == cfg["rbshots"]).all())
    print("RB circuits %d; ideal-run sequences return to |00> on every pair (Clifford inverse correct): %s" % (len(circs), "OK" if ok else "FAILED"))
    bits = [v1.bits_from_counts(sim.run(x["bound"], shots=cfg["shots"], seed_simulator=7 + c).result().get_counts(), n) for c in range(cfg["copies"])]
    exact = v1.zz_p1(edges, x["gamma"], x["beta"]); res = arms_analysis(bits, edges, exact, x["cn"], stated, x["cols"], meta, K, cfg["rbshots"]); res["stated_of"] = stated
    txt, _ = report(res, x["ideal"] / x["maxcut"], "SMOKE "); print("analysis path executed on %d lines of output (numbers withheld)" % len(txt.splitlines()))
    rows = sources_rows(be, x, None, stated); print("source guard with job P missing refuses: %s" % ("OK" if any("MISSING" in r[1] for r in rows) else "FAILED"))
    print("estimates full: P %.1f + main %.1f; pilot: P %.1f + main %.1f" % (est_p(len(x["cols"]), CFG["full"]), est_main(CFG["full"]), est_p(len(x["cols"]), CFG["pilot"]), est_main(CFG["pilot"])))


def mode_dry():
    svc = v1.get_service(); u = v1.usage_remaining(svc); be = svc.backend(BACKEND); print("backend %s operational=%s; usage_remaining_seconds=%.0f" % (BACKEND, be.status().operational, u))
    cfg = CFG["full"]; n, tot = choose_n(be, cfg); x = prepare(be, n); edges, lay = x["edges"], x["layout"]
    print("size chosen by the 200 s rule: %d nodes (%s), %d edges; gamma %.6f beta %.6f; noiseless cut %.3f of max cut %d" % (n, "full graph" if n == 100 else "SUB-GRAPH of the client's 100", len(edges), x["gamma"], x["beta"], x["ideal"], x["maxcut"]))
    print("level-0 circuit after the client's layout: depth %d, cz %d (2 per edge asserted), every RZZ angle exact; colour layers %s" % (x["bound"].depth(), x["bound"].count_ops().get("cz", 0), [len(c) for c in x["cols"]]))
    stamp = v2.record_stamp(be); stated, rows, mean_fade = model_power(be, x)
    ev = np.array(list(stated.values())); print("record stamp %s; stated CZ error (RB average infidelity) over %d pairs: min %.5f median %.5f max %.5f; model fade under stated errors: mean %.4f of <ZZ>" % (stamp, len(ev), ev.min(), np.median(ev), ev.max(), mean_fade))
    circs, meta = rb_circuits(be, x["cols"], cfg); print("job P: %d parallel-RB circuits (%d colour layers x %d lengths x %d seqs) x %d shots; CZ per pair per circuit %d to %d" % (
        len(circs), len(x["cols"]), len(cfg["lengths"]), cfg["seqs"], cfg["rbshots"], min(min(m["ncz"]) for m in meta), max(max(m["ncz"]) for m in meta)))
    for lab in ("P", "main"):
        s = v2.clean_sampler(be); v2.assert_clean(s); print("Sampler options for job %s: %s" % (lab, v2.options_line(s.options)))
    ep, em = est_p(len(x["cols"]), cfg), est_main(cfg); print("estimated seconds: P %.0f + main %.0f = %.0f (x%.2f = %.0f; cap %d; remaining %.0f)" % (ep, em, ep + em, MARGIN, (ep + em) * MARGIN, MAX_SECONDS, u))
    pc = CFG["pilot"]; print("pilot estimate: P %.0f + main %.0f = %.0f s (cap %d)" % (est_p(len(x["cols"]), pc), est_main(pc), est_p(len(x["cols"]), pc) + est_main(pc), PILOT_SECONDS))
    print("edges kept for every arm: %d of %d (light cone free of any pair stated above %.0f%%; pairs stated above that: %s)" % (keep_mask(x["cn"], stated).sum(), len(edges), BAD_E * 100, sorted(p for p, e in stated.items() if e > BAD_E)))
    for rho, os_, on in rows: print("model: if the chip's CZ error is %.2f x stated: expected |O-S| %.5f, |O-N| %.5f (ratio of the cut)" % (rho, os_, on))
    pr = rows[0][1] / FLOOR_V1
    print("POWER RATIO: %.2f" % pr)
    print("(= expected |O-S| at rho 1.30, the low end of our kingston RB range, over the v1 12-copy 99%% half-width %.4f; %s)" % (FLOOR_V1, "PILOT NEEDED (model under 3; a model number never drops a design)" if pr < 3 else "model passes 3"))
    print_sources(sources_rows(be, x, None, stated))
    if (ep + em) * MARGIN > MAX_SECONDS or (ep + em) * MARGIN > u: print("WOULD REFUSE")


def submit_both(be, x, cfg, sfx, stated, pilot):
    n = x["n"]; edges, lay = x["edges"], x["layout"]
    circs, meta = rb_circuits(be, x["cols"], cfg)
    s_save(sfx, dict(n=n, layout=lay, edges=[list(e) for e in edges], gamma=x["gamma"], beta=x["beta"], ideal=x["ideal"], maxcut=x["maxcut"], cols=[[list(p) for p in c] for c in x["cols"]],
                     meta=meta, stated={"%d-%d" % p: v for p, v in stated.items()}, record_stamp=v2.record_stamp(be), cfg=cfg, fired_at=time.time(), pilot=pilot))
    sp = v2.clean_sampler(be); v2.assert_clean(sp); jp = sp.run([(c, None, cfg["rbshots"]) for c in circs]); s_save(sfx, dict(job_p=jp.job_id())); print("job P (parallel RB) submitted", jp.job_id()); v2.show_job_options(jp, "P")
    rows = sources_rows(be, x, jp.job_id(), stated); print_sources(rows)
    if any("MISSING" in r[1] for r in rows): sys.exit("REFUSED: no hardware source for a parameter")
    sm = v2.clean_sampler(be); v2.assert_clean(sm); ol = v2.options_line(sm.options); print("Sampler options for main: " + ol)
    jm = sm.run([(x["bound"], None, cfg["shots"])] * cfg["copies"]); s_save(sfx, dict(job_main=jm.job_id(), main_options=ol, sources=rows)); print("main job submitted", jm.job_id()); v2.show_job_options(jm, "main")


def fetch(sfx, quiet=False):
    st = s_load(sfx); svc = v1.get_service(); cfg = CFG["pilot" if sfx else "full"]; cols = [[tuple(p) for p in c] for c in st["cols"]]
    done = True
    for key, raw in (("job_p", "raw_P%s.npz" % sfx), ("job_main", "raw_main%s.npz" % sfx)):
        if os.path.exists(fp(raw)): continue
        j = svc.job(st[key]); s = str(j.status()); print(key, s)
        if "DONE" not in s.upper(): done = False; continue
        r = j.result(); save_p(sfx, r, cols, st["meta"]) if key == "job_p" else save_main(sfx, r, st["n"], cfg); v2.show_job_options(j, key); print("saved", raw)
    return done


def mode_fire(pilot):
    sfx = "_pilot" if pilot else ""; cfg = CFG["pilot" if pilot else "full"]
    svc = v1.get_service(); be = svc.backend(BACKEND); u = v1.usage_remaining(svc); n, tot = choose_n(be, cfg)
    if pilot and tot > PILOT_SECONDS: sys.exit("REFUSED: pilot estimate %.0f s over %.0f s" % (tot, PILOT_SECONDS))
    x = prepare(be, n); stated, rows, _ = model_power(be, x); pr = rows[0][1] / FLOOR_V1
    print("estimate P+main %.1f s (x%.2f = %.1f; cap %d; remaining %.0f)" % (tot, MARGIN, tot * MARGIN, PILOT_SECONDS if pilot else MAX_SECONDS, u))
    if tot * MARGIN > MAX_SECONDS or tot * MARGIN > u: sys.exit("REFUSED: estimate above the cap or the remaining seconds")
    if s_load(sfx).get("job_main"): sys.exit("REFUSED: already fired; move run_v3/state%s.json aside to refire" % sfx)
    if not pilot:
        print("POWER RATIO (model): %.2f" % pr)
        if pr < 3:
            pj = os.path.join(RUN, "pilot_result.json"); r = json.load(open(pj))["ratio_OS"] if os.path.exists(pj) else None
            if r is None or r < 3: sys.exit("PILOT NEEDED: model POWER RATIO %.2f is under 3; run --fire --pilot first (pilot ratio %s)" % (pr, "none yet" if r is None else "%.2f" % r))
            print("pilot ratio %.2f (hardware) passes 3" % r)
    submit_both(be, x, cfg, sfx, stated, pilot)
    if pilot:
        t0 = time.time(); DEADLINE = 1800
        while time.time() - t0 < DEADLINE:
            time.sleep(20)
            if fetch(sfx): return analyze(sfx)
        sys.exit("pilot jobs not DONE after %d s; run --fetch --pilot then --analyze --pilot" % DEADLINE)
    print("Next: --fetch until both are DONE, then --analyze.")


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
