"""Demo 6 v4 "X" (AI kernels): the diagonal pilot. Rule 5 hardware pilot, per chip.

X marks the spot: the self-score K(x, x) has a known answer of exactly 1 and runs the same CZ layout as every off-diagonal score
(checked at compile: 54 CZ on the same 7 pairs, kingston 19 Nov). Its shortfall from 1 on the chip is the fade (gate leak + readout
together), measured on the circuit itself. The client's level-3 compile cancels K(x, x) to 0 CZ, so the diagonal is compiled from a
parameterized template (same layout, same seed) with the angles bound after compile; the script asserts the CZ count and pairs match.

Stages (each <= 30 s of QPU, x1.25 margin):
  1. job D: K(x, x) for the 12 pilot train points, COPIES_D copies x SHOTS_D shots. Fetched and saved BEFORE stage 2 is submitted
     (a correction is never measured and applied in the same job).
  2. the existing demo.py pilot unchanged: job P (RB + readout) and main (4 copies x 66 entries x 100 shots), client default compile.
Arm D (written before data): K_D = (P0 - (1 - A)/256) / A on the raw all-zeros share, A = mean raw P0 over job D's diagonal.
Readings beside it: N (client raw), S (stated), O (our RB fade), from demo.py's estimators unchanged.
Modes: --dry (compile + estimate, no submit), --fire --pilot, --fetch --pilot, --analyze --pilot. Keys / CRNs never printed.
"""
import os, sys, json, math, time, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import demo as m
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

m.RUN = RUN = os.path.join(HERE, "run_v4_" + m.BACKEND.replace("ibm_", ""))
COPIES_D, SHOTS_D = 4, 250
STAGE_CAP = 30.0
SFX = "_pilot"
EQUIV_MAE = m.EQUIV_MAE


def pfeature(v):
    qc = QuantumCircuit(m.NQ)
    for _ in range(m.REPS):
        qc.h(range(m.NQ))
        for i in range(m.NQ): qc.p(2 * v[i], i)
        for i in range(m.NQ - 1):
            qc.cx(i, i + 1); qc.p(2 * (math.pi - v[i]) * (math.pi - v[i + 1]), i + 1); qc.cx(i, i + 1)
    return qc


def template(be, lay):
    a, b = ParameterVector("a", m.NQ), ParameterVector("b", m.NQ)
    qc = QuantumCircuit(m.NQ, m.NQ); qc.compose(pfeature(b), inplace=True); qc.compose(pfeature(a).inverse(), inplace=True)
    qc.measure(range(m.NQ), range(m.NQ))
    t = generate_preset_pass_manager(optimization_level=3, backend=be, initial_layout=lay, seed_transpiler=7).run(qc)
    return t, a, b


def diag_circuits(be, x):
    t, a, b = template(be, x["lay"])
    prs_t, N_t = m.cz_counts([t])
    if prs_t != x["prs"] or not np.all(N_t[0] == x["NCZ"][0]) or not np.all(x["NCZ"] == x["NCZ"][0]):
        sys.exit("REFUSED: template CZ layout differs from the client's off-diagonal circuits (template %s %s; client per-entry totals %s)" % (prs_t, N_t[0].tolist(), sorted(set(x["NCZ"].sum(1).tolist()))))
    n = x["cfg_ntrain"]
    circs = [t.assign_parameters({**dict(zip(a, x["X"][i])), **dict(zip(b, x["X"][i]))}) for i in range(n)]
    for c in circs:
        bad = set(c.count_ops()) - m.BASIS_OK
        if bad: sys.exit("REFUSED: bound diagonal circuit has ops %s" % sorted(bad))
        if sum(1 for ci in c.data if ci.operation.name == "cz") != int(x["NCZ"][0].sum()): sys.exit("REFUSED: a bound diagonal circuit lost CZ gates")
    return circs, int(x["NCZ"][0].sum()), prs_t


def est_d(n): return m.SEC_PER_SHOT_CIRCUIT * n * COPIES_D * SHOTS_D + m.SEC_PER_JOB


def setup():
    svc = m.get_service(); be = svc.backend(m.BACKEND); cfg = m.CFG["pilot"]
    x = m.prepare(be, cfg); x["cfg_ntrain"] = cfg["ntrain"]
    return svc, be, cfg, x


def mode_dry():
    svc, be, cfg, x = setup(); u = m.usage_remaining(svc)
    circs, ncz, prs = diag_circuits(be, x)
    rbc, _ = m.rb_circuits(be, x["cols"], cfg); e2 = m.est_p(cfg, len(rbc)) + m.est_main(cfg, len(x["E"])); e1 = est_d(len(circs))
    print("DRY %s: layout %s; client off-diagonal entries %d, every one %d CZ on pairs %s; template diagonal %d circuits, %d CZ each (client compile of K(x,x): cancelled)" % (m.BACKEND, x["lay"], len(x["E"]), ncz, prs, len(circs), ncz))
    print("estimate: stage 1 job D %.1f s (x%.2f = %.1f), stage 2 P+main %.1f s (x%.2f = %.1f); cap %d per stage; remaining %.0f" % (e1, m.MARGIN, e1 * m.MARGIN, e2, m.MARGIN, e2 * m.MARGIN, STAGE_CAP, u))
    if e1 * m.MARGIN > STAGE_CAP or e2 * m.MARGIN > STAGE_CAP or (e1 + e2) * m.MARGIN > u: print("WOULD REFUSE")


def wait_done(svc, jid, label, deadline=1800):
    t0 = time.time()
    while time.time() - t0 < deadline:
        s = str(svc.job(jid).status())
        if "DONE" in s.upper(): return
        if any(w in s.upper() for w in ("ERROR", "CANCEL", "FAIL")): sys.exit("job %s %s: %s" % (label, jid, s))
        time.sleep(20)
    sys.exit("job %s %s not DONE after %d s; rerun --fire --pilot (it resumes)" % (label, jid, deadline))


def mode_fire():
    svc, be, cfg, x = setup(); u = m.usage_remaining(svc); st = m.s_load(SFX)
    circs, ncz, prs = diag_circuits(be, x)
    rbc, _ = m.rb_circuits(be, x["cols"], cfg); e2 = m.est_p(cfg, len(rbc)) + m.est_main(cfg, len(x["E"])); e1 = est_d(len(circs))
    print("PILOT X %s estimate: job D %.1f s, then P+main %.1f s (x%.2f; cap %d per stage; remaining %.0f)" % (m.BACKEND, e1, e2, m.MARGIN, STAGE_CAP, u))
    if e1 * m.MARGIN > STAGE_CAP or e2 * m.MARGIN > STAGE_CAP or (e1 + e2) * m.MARGIN > u: sys.exit("REFUSED: a stage is over %d s or the remaining seconds" % STAGE_CAP)
    if not os.path.exists(m.fp("raw_D.npz")):
        if "job_d" not in st:
            sd = m.clean_sampler(be); m.assert_clean(sd)
            jd = sd.run([(c, None, SHOTS_D) for _ in range(COPIES_D) for c in circs])
            m.s_save(SFX, dict(job_d=jd.job_id(), d_options=m.options_line(sd.options), d_ncz=ncz, d_hash=x["hash"], d_record=m.record_stamp(be), d_at=time.time()))
            print("job D submitted", jd.job_id()); m.show_job_options(jd, "D"); st = m.s_load(SFX)
        wait_done(svc, st["job_d"], "D")
        r = svc.job(st["job_d"]).result(); n = len(circs)
        cnt = np.array([m.dense(m.counts_of(r[k])) for k in range(len(r))]).reshape(COPIES_D, n, m.NKEEP)
        np.savez_compressed(m.fp("raw_D.npz"), cnt=cnt); print("saved raw_D (job D DONE before stage 2 is submitted)")
    if m.s_load(SFX).get("jobs_main"): print("stage 2 already submitted")
    else:
        stated = m.stated_errors(be, x["prs"]); m.submit_all(be, x, cfg, SFX, stated, True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if m.fetch(SFX): return mode_analyze()
        time.sleep(20)
    sys.exit("stage 2 not DONE after 1800 s; run --fetch --pilot then --analyze --pilot")


def mode_analyze():
    svc, be, cfg, x = setup(); st = m.s_load(SFX)
    if x["hash"] != st["hash"] or st.get("d_hash") != st["hash"]: print("WARNING: transpile hash moved since the fire (fired %s, job D %s, today %s)" % (st["hash"], st.get("d_hash"), x["hash"]))
    D = np.load(m.fp("raw_D.npz"))["cnt"] / SHOTS_D; P0d = D[..., 0]            # (COPIES_D, n)
    A = float(P0d.mean()); nD = P0d.size * SHOTS_D
    cnt = np.load(m.fp("raw_main%s.npz" % SFX))["cnt"]; C, shots = cfg["copies"], cfg["shots"]; E = [tuple(e) for e in st["E"]]; prs = [tuple(p) for p in st["prs"]]
    cols = [[tuple(p) for p in c] for c in st["cols"]]
    stated = {tuple(int(v) for v in k.split("-")): e for k, e in st["stated"].items()}
    eO, _ = m.measured_errors(cols, st["meta"], [np.array(k) for k in st["rb_k00"]], cfg["rbshots"]); t8 = st["tab8"]
    rec8 = m.record_readout(be)[st["lay"]]
    est = m.estimates(cnt, shots, x["NCZ"], prs, rec8, (np.array(t8["p01"]), np.array(t8["p10"])), stated, eO)
    Kd = lambda a: (cnt[..., 0] / shots - (1 - a) / m.NKEEP) / a
    est["D"] = Kd(A)
    ex = np.array([x["Kex"][a, b] for a, b in E])
    FS = float(m.fade_of(x["NCZ"][:1], stated, prs)[0]); FO = float(m.fade_of(x["NCZ"][:1], eO, prs)[0])
    print("HARDWARE PILOT X, %s; job D %s (earlier, saved before stage 2), job P %s, main %s" % (m.BACKEND, st["job_d"], st["job_p"], st["jobs_main"]))
    print("diagonal K(x,x) raw all-zeros share (true value 1): mean A = %.4f over %d points x %d copies (per-point %.3f to %.3f)" % (A, P0d.shape[1], P0d.shape[0], P0d.mean(0).min(), P0d.mean(0).max()))
    print("fade each arm divides by, same 54-CZ circuit: D (measured, incl. readout) %.4f; S stated-CZ fade %.4f; O our-RB fade %.4f (S and O also invert readout first)" % (A, FS, FO))
    print("per-pair CZ error stated / our RB:", [("%d-%d" % p, round(stated[p], 5), round(eO[p], 5), round(eO[p] / stated[p], 2)) for p in prs])
    mt = {a: m.metrics(est[a], E, x["Kex"], x["y"], cfg["ntrain"], 0) for a in ("N", "S", "O", "D")}
    for a, lab in (("N", "client raw"), ("S", "stated fix"), ("O", "our RB-fade fix"), ("D", "X: diagonal-measured fade")):
        v = mt[a]["mae"]; bias = (est[a] - ex).mean(1)
        print("  %s %-26s mean abs kernel error %.4f (99%% interval %.4f to %.4f); bias %+.4f" % (a, lab, v.mean(), v.mean() - m.tq(C) * v.std(ddof=1) / math.sqrt(C), v.mean() + m.tq(C) * v.std(ddof=1) / math.sqrt(C), bias.mean()))
    rng = np.random.default_rng(m.SEED); draws = rng.binomial(nD, A, m.NDRAW) / nD
    pooledN = np.abs(cnt[..., 0].sum(0) / (shots * C) - ex).mean()
    sdD = float(np.std([np.abs((cnt[..., 0].sum(0) / (shots * C) - (1 - a) / m.NKEEP) / a - ex).mean() for a in draws], ddof=1))
    for a, b in (("D", "N"), ("D", "S"), ("D", "O")):
        mm, lo, hi, w, half = m.paired(mt[a]["mae"], mt[b]["mae"], EQUIV_MAE, sdD)
        print("  OUTCOME %s minus %s, mean abs kernel error (paired by copy, job-D sd %.4f in quadrature): %+.4f (99%% interval %+.4f to %+.4f): %s (BELOW = X lower error; CONSISTENT inside +-%.3f)" % (a, b, sdD, mm, lo, hi, w, EQUIV_MAE))
        print("    PILOT RATIO |%s-%s| %.4f over its paired floor %.4f = %.1f" % (a, b, abs(mm), half, abs(mm) / half if half > 0 else float("nan")))
    print("note: pilot = 12 train points, no test split, so no accuracy reading; the headline is D vs N, D vs S beside it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("dry", "fire", "fetch", "analyze", "pilot"): ap.add_argument("--" + k, action="store_true")
    a = ap.parse_args()
    if a.dry: mode_dry()
    elif a.fire:
        if not a.pilot: sys.exit("REFUSED: demo_v4.py is a pilot only; use --fire --pilot")
        mode_fire()
    elif a.fetch: print("stage 2 done" if m.fetch(SFX) else "stage 2 not done")
    elif a.analyze: mode_analyze()
    else: ap.print_help()
