"""Demo 6 v5 "THE SPLIT" (AI kernels): two known-answer anchors on the same 54-CZ mirror. Rule 5 hardware pilot, per chip.

Las Ecuaciones / Pack Q #27: D = S + P x crossing; fit the shared part S and the scaling part P separately, because an offset divided
by a strength manufactures a trend. The v4 pilot measured P (the diagonal, K = 1) and divided the chip's all-zeros floor S by it:
the overshoot on all three chips (19 Nov). v5 measures S too, with a floor anchor of known exact K near 0.

Job A (EARLIER than the main, saved before stage 2 is submitted), all compiled once from the v4 parameterized template (two
parameter vectors, angles bound after compile, 54 CZ asserted on every bound circuit):
  - the diagonal: K(x, x) for the 12 pilot train points (exact 1);
  - the floor: 12 anchor pairs (a, b) of quarter-grid feature vectors whose exact K is below 0.01 (exact value printed, used).
Stage 2: demo.py's pilot unchanged (job P, then main 4 copies x 66 entries x 100 shots, client default compile).
Arms (written before data):
  N   client raw; S stated fix; O our RB-fade fix (demo.py estimators, unchanged);
  D   the published diagonal rescale (Hubregtsen et al. 2021): the CLIENT'S FREE FIX, never ours;
  X   THE SPLIT: K = (P0 - c) / (A - c), with A = mean raw P0 on the diagonal and c = mean over anchors of (P0_anchor - A K_a)/(1 - K_a)
      (the floor, after removing the anchors' small exact K at slope A; c solves both anchors exactly when K_a = 0).
Headline: X vs D (the bar a client gets for free), with X vs N and X vs S beside it.
Prior art to declare: learning a linear noisy-to-ideal map from known-answer circuits is Clifford data regression (Czarnik et al.
2021). What v5 adds is the split ruler on the kernel's own mirror circuit, both anchors in an earlier job, no fit on scored shots.
Modes: --dry, --fire --pilot, --analyze --pilot. Keys / CRNs never printed.
"""
import os, sys, json, math, time, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import demo_v4 as v4
m = v4.m
from qiskit.quantum_info import Statevector

m.RUN = RUN = os.path.join(HERE, "run_v5_" + m.BACKEND.replace("ibm_", ""))
COPIES_A, SHOTS_A = 4, 250
N_ANCHOR, K_ANCHOR_MAX, ANCHOR_SEED = 12, 0.01, 20261119
SFX = "_pilot"


def anchors():
    """12 pairs of quarter-grid feature vectors with exact K < 0.01, chosen by a fixed seed before any data; exact K from the ideal circuit."""
    rng = np.random.default_rng(ANCHOR_SEED); grid = np.array([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]); out = []
    while len(out) < N_ANCHOR:
        a, b = rng.choice(grid, m.NQ), rng.choice(grid, m.NQ)
        k = abs(np.vdot(Statevector(m.feature_circuit(a)).data, Statevector(m.feature_circuit(b)).data)) ** 2
        if k < K_ANCHOR_MAX: out.append((a, b, float(k)))
    return out


def anchor_circuits(be, x):
    t, pa, pb = v4.template(be, x["lay"])
    prs_t, N_t = m.cz_counts([t]); ncz = int(x["NCZ"][0].sum())
    if prs_t != x["prs"] or not np.all(N_t[0] == x["NCZ"][0]) or not np.all(x["NCZ"] == x["NCZ"][0]): sys.exit("REFUSED: template CZ layout differs from the client's entries")
    n = x["cfg_ntrain"]; diag = [t.assign_parameters({**dict(zip(pa, x["X"][i])), **dict(zip(pb, x["X"][i]))}) for i in range(n)]
    an = anchors(); flo = [t.assign_parameters({**dict(zip(pa, a)), **dict(zip(pb, b))}) for a, b, _ in an]
    for c in diag + flo:
        if set(c.count_ops()) - m.BASIS_OK: sys.exit("REFUSED: bound anchor circuit has ops %s" % sorted(set(c.count_ops()) - m.BASIS_OK))
        if sum(1 for ci in c.data if ci.operation.name == "cz") != ncz: sys.exit("REFUSED: a bound anchor circuit lost CZ gates")
    return diag, flo, [k for *_, k in an], ncz, prs_t


def est_a(n): return m.SEC_PER_SHOT_CIRCUIT * n * COPIES_A * SHOTS_A + m.SEC_PER_JOB


def mode_dry():
    svc, be, cfg, x = v4.setup(); u = m.usage_remaining(svc)
    diag, flo, ka, ncz, prs = anchor_circuits(be, x)
    rbc, _ = m.rb_circuits(be, x["cols"], cfg); e2 = m.est_p(cfg, len(rbc)) + m.est_main(cfg, len(x["E"])); e1 = est_a(len(diag) + len(flo))
    print("DRY %s: every client entry and every anchor %d CZ on %s; %d diagonal + %d floor anchors (exact K %.4f to %.4f)" % (m.BACKEND, ncz, prs, len(diag), len(flo), min(ka), max(ka)))
    print("estimate: job A %.1f s (x%.2f = %.1f), P+main %.1f s (x%.2f = %.1f); cap %d per stage; remaining %.0f" % (e1, m.MARGIN, e1 * m.MARGIN, e2, m.MARGIN, e2 * m.MARGIN, v4.STAGE_CAP, u))
    if e1 * m.MARGIN > v4.STAGE_CAP or e2 * m.MARGIN > v4.STAGE_CAP or (e1 + e2) * m.MARGIN > u: print("WOULD REFUSE")


def mode_fire():
    svc, be, cfg, x = v4.setup(); u = m.usage_remaining(svc); st = m.s_load(SFX)
    diag, flo, ka, ncz, prs = anchor_circuits(be, x)
    rbc, _ = m.rb_circuits(be, x["cols"], cfg); e2 = m.est_p(cfg, len(rbc)) + m.est_main(cfg, len(x["E"])); e1 = est_a(len(diag) + len(flo))
    print("PILOT SPLIT %s estimate: job A %.1f s, then P+main %.1f s (x%.2f; cap %d per stage; remaining %.0f)" % (m.BACKEND, e1, e2, m.MARGIN, v4.STAGE_CAP, u))
    if e1 * m.MARGIN > v4.STAGE_CAP or e2 * m.MARGIN > v4.STAGE_CAP or (e1 + e2) * m.MARGIN > u: sys.exit("REFUSED: a stage is over the cap or the remaining seconds")
    if not os.path.exists(m.fp("raw_A.npz")):
        if "job_a" not in st:
            sa = m.clean_sampler(be); m.assert_clean(sa); circs = diag + flo
            ja = sa.run([(c, None, SHOTS_A) for _ in range(COPIES_A) for c in circs])
            m.s_save(SFX, dict(job_a=ja.job_id(), a_options=m.options_line(sa.options), a_ncz=ncz, a_hash=x["hash"], a_record=m.record_stamp(be), anchor_k=ka, n_diag=len(diag), n_floor=len(flo)))
            print("job A submitted", ja.job_id()); m.show_job_options(ja, "A"); st = m.s_load(SFX)
        v4.wait_done(svc, st["job_a"], "A")
        r = svc.job(st["job_a"]).result(); n = len(diag) + len(flo)
        cnt = np.array([m.dense(m.counts_of(r[k])) for k in range(len(r))]).reshape(COPIES_A, n, m.NKEEP)
        np.savez_compressed(m.fp("raw_A.npz"), cnt=cnt); print("saved raw_A (job A DONE before stage 2 is submitted)")
    if not m.s_load(SFX).get("jobs_main"):
        stated = m.stated_errors(be, x["prs"]); m.submit_all(be, x, cfg, SFX, stated, True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if m.fetch(SFX): return mode_analyze()
        time.sleep(20)
    sys.exit("stage 2 not DONE after 1800 s; run --analyze --pilot once it is")


def mode_analyze():
    svc, be, cfg, x = v4.setup(); st = m.s_load(SFX)
    if x["hash"] != st["hash"] or st.get("a_hash") != st["hash"]: print("WARNING: transpile hash moved (fired %s, job A %s, today %s)" % (st["hash"], st.get("a_hash"), x["hash"]))
    Acnt = np.load(m.fp("raw_A.npz"))["cnt"] / SHOTS_A; nd = st["n_diag"]; ka = np.array(st["anchor_k"])
    Pd, Pf = Acnt[:, :nd, 0], Acnt[:, nd:, 0]
    A = float(Pd.mean()); c = float(((Pf.mean(0) - A * ka) / (1 - ka)).mean())
    cnt = np.load(m.fp("raw_main%s.npz" % SFX))["cnt"]; C, shots = cfg["copies"], cfg["shots"]; E = [tuple(e) for e in st["E"]]; prs = [tuple(p) for p in st["prs"]]
    cols = [[tuple(p) for p in cc] for cc in st["cols"]]; stated = {tuple(int(v) for v in k.split("-")): e for k, e in st["stated"].items()}
    eO, _ = m.measured_errors(cols, st["meta"], [np.array(k) for k in st["rb_k00"]], cfg["rbshots"]); t8 = st["tab8"]; rec8 = m.record_readout(be)[st["lay"]]
    est = m.estimates(cnt, shots, x["NCZ"], prs, rec8, (np.array(t8["p01"]), np.array(t8["p10"])), stated, eO)
    P0 = cnt[..., 0] / shots
    est["D"] = (P0 - (1 - A) / m.NKEEP) / A
    est["X"] = (P0 - c) / (A - c)
    ex = np.array([x["Kex"][a, b] for a, b in E])
    print("HARDWARE PILOT SPLIT, %s; job A %s (earlier, saved before stage 2), job P %s, main %s" % (m.BACKEND, st["job_a"], st["job_p"], st["jobs_main"]))
    print("anchors from job A: diagonal A = %.4f (exact 1; %d points x %d copies); floor c = %.4f (from %d anchors, exact K %.4f-%.4f; raw P0 %.4f-%.4f); uniform-noise term (1-A)/256 = %.4f" % (
        A, nd, Pd.shape[0], c, Pf.shape[1], ka.min(), ka.max(), Pf.mean(0).min(), Pf.mean(0).max(), (1 - A) / m.NKEEP))
    mt = {a: m.metrics(est[a], E, x["Kex"], x["y"], cfg["ntrain"], 0) for a in ("N", "S", "O", "D", "X")}
    for a, lab in (("N", "client raw"), ("S", "stated fix"), ("O", "our RB-fade fix"), ("D", "diagonal rescale (client's free fix)"), ("X", "THE SPLIT (two anchors)")):
        vv = mt[a]["mae"]; bias = (est[a] - ex).mean(1); h = m.tq(C) * vv.std(ddof=1) / math.sqrt(C)
        print("  %s %-36s mean abs kernel error %.4f (99%% interval %.4f to %.4f); bias %+.4f" % (a, lab, vv.mean(), vv.mean() - h, vv.mean() + h, bias.mean()))
    rng = np.random.default_rng(m.SEED); nA = Pd.size * SHOTS_A; nF = Pf.size * SHOTS_A; pooled = P0.sum(0) / C
    dr = [(rng.binomial(nA, A) / nA, rng.binomial(nF, max(c, 1e-6)) / nF) for _ in range(m.NDRAW)]
    sdX = float(np.std([np.abs((pooled - cc) / (aa - cc) - ex).mean() for aa, cc in dr], ddof=1))
    for a, b in (("X", "D"), ("X", "N"), ("X", "S")):
        mm, lo, hi, w, half = m.paired(mt[a]["mae"], mt[b]["mae"], m.EQUIV_MAE, sdX)
        print("  OUTCOME %s minus %s, mean abs kernel error (paired by copy, job-A sd %.4f in quadrature): %+.4f (99%% interval %+.4f to %+.4f): %s (BELOW = THE SPLIT lower error; CONSISTENT inside +-%.3f)" % (a, b, sdX, mm, lo, hi, w, m.EQUIV_MAE))
        print("    PILOT RATIO |%s-%s| %.4f over its paired floor %.4f = %.1f" % (a, b, abs(mm), half, abs(mm) / half if half > 0 else float("nan")))
    print("note: pilot = 12 train points, no test split, no accuracy reading. Headline: X vs D (the client's free fix).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("dry", "fire", "analyze", "pilot"): ap.add_argument("--" + k, action="store_true")
    a = ap.parse_args()
    if a.dry: mode_dry()
    elif a.fire:
        if not a.pilot: sys.exit("REFUSED: demo_v5.py is a pilot only; use --fire --pilot")
        mode_fire()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
