"""Demo 6 v7 (AI kernels): KERNEL AUDIT + SHOT BILL AT SCALE. Hardware only, per chip.

The client: an AI / ML team running IBM's showcase fidelity kernel (ZZ feature map, linear entanglement, 2 reps) and planning to scale
it up. The self-score K(x, x) has a known answer of exactly 1 at ANY size, so it is a known-answer probe at 8, 16, 24, 32 and 40 qubits
with no simulator. For each size N:
  A(N)  measured: mean raw all-zeros share of K(x, x) for 12 points drawn like the client's data (features in [0, 0.18]), compiled from a
        parameterized template with the client's default (level 3, seed 7, layout chosen by the compiler), angles bound after compile,
        CZ count asserted on every bound circuit;
  S(N)  what IBM's record implies for the same circuit, written before data: prod_i (1 - readout_error_i) over the layout x prod over
        every CZ (1 - 4 e_pair / 3), e_pair = the record's CZ error, plus the uniform term (1 - F) / 2^N;
  BILL  shots the client needs per kernel entry for a given precision scale as 1 / A^2, so the bill against plan is (S(N) / A(N))^2.
Plus the readout audit: all-0 and all-1 on the union of the layout qubits, each qubit's P(1|0) and P(0|1) against the record's figure,
flagged where today's lower 99% bound sits above the record (IBM codes Q2/Q9/Q11).
One job per chip (all sizes + audit), IBM's runtime changes off and read back. Modes: --dry, --fire, --fetch, --analyze.
"""
import os, sys, json, math, time, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import demo as m
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

m.RUN = RUN = os.path.join(HERE, "run_v7_" + m.BACKEND.replace("ibm_", ""))
SIZES = (8, 16, 24, 32, 40)
NPTS, COPIES, SHOTS, RO_SHOTS = 12, 4, 250, 2000
REPS, XSCALE, SEED = 2, 0.18, 20261119
CAP = 60.0


def pfeature(v, n):
    qc = QuantumCircuit(n)
    for _ in range(REPS):
        qc.h(range(n))
        for i in range(n): qc.p(2 * v[i], i)
        for i in range(n - 1):
            qc.cx(i, i + 1); qc.p(2 * (math.pi - v[i]) * (math.pi - v[i + 1]), i + 1); qc.cx(i, i + 1)
    return qc


def template(be, n):
    a, b = ParameterVector("a", n), ParameterVector("b", n)
    qc = QuantumCircuit(n, n); qc.compose(pfeature(b, n), inplace=True); qc.compose(pfeature(a, n).inverse(), inplace=True); qc.measure(range(n), range(n))
    t = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=7).run(qc)
    lay = [int(q) for q in t.layout.initial_index_layout(filter_ancillas=True)][:n]
    return t, a, b, lay


def build(be):
    rng = np.random.default_rng(SEED); sets = []
    for n in SIZES:
        t, a, b, lay = template(be, n); prs, N = m.cz_counts([t]); ncz = int(N[0].sum())
        X = rng.uniform(0, XSCALE, (NPTS, n))
        circs = [t.assign_parameters({**dict(zip(a, x)), **dict(zip(b, x))}) for x in X]
        for c in circs:
            if set(c.count_ops()) - m.BASIS_OK: sys.exit("REFUSED: N=%d bound circuit has ops %s" % (n, sorted(set(c.count_ops()) - m.BASIS_OK)))
            if sum(1 for ci in c.data if ci.operation.name == "cz") != ncz: sys.exit("REFUSED: N=%d a bound self-score circuit lost CZ gates" % n)
        sets.append(dict(n=n, circs=circs, lay=lay, prs=prs, ncz_pair=N[0].tolist(), ncz=ncz))
    union = sorted({q for s in sets for q in s["lay"]})
    return sets, union


def ro_circs(be, union):
    out = []
    for flip in (False, True):
        qc = QuantumCircuit(be.num_qubits, len(union))
        if flip: qc.x(union)
        qc.measure(union, range(len(union))); out.append(qc)
    return out


def stated_selfscore(be, s):
    rec = m.record_readout(be); ro = float(np.prod([1 - rec[q] for q in s["lay"]]))
    st = m.stated_errors(be, s["prs"]); F = float(np.prod([(1 - 4 * st[p] / 3) ** k for p, k in zip(s["prs"], s["ncz_pair"])]))
    return ro * (F + (1 - F) / 2 ** s["n"]), ro, F


def est(sets): return m.SEC_PER_SHOT_CIRCUIT * (len(sets) * NPTS * COPIES * SHOTS + 2 * RO_SHOTS) + m.SEC_PER_JOB


def mode_dry():
    svc = m.get_service(); be = svc.backend(m.BACKEND); u = m.usage_remaining(svc); sets, union = build(be)
    for s in sets:
        S, ro, F = stated_selfscore(be, s)
        print("N=%2d layout %s... %d CZ on %d pairs; record-implied self-score S = %.4f (readout %.4f x gate fade %.4f)" % (s["n"], s["lay"][:4], s["ncz"], len(s["prs"]), S, ro, F))
    e = est(sets); print("audit union %d qubits; estimate %.1f s (x%.2f = %.1f; cap %d; remaining %.0f)" % (len(union), e, m.MARGIN, e * m.MARGIN, CAP, u))
    if e * m.MARGIN > CAP or e * m.MARGIN > u: print("WOULD REFUSE")


def mode_fire():
    svc = m.get_service(); be = svc.backend(m.BACKEND); u = m.usage_remaining(svc); sets, union = build(be); e = est(sets)
    if e * m.MARGIN > CAP or e * m.MARGIN > u: sys.exit("REFUSED: estimate %.1f x%.2f over the cap %d or the remaining seconds" % (e, m.MARGIN, CAP))
    if m.s_load("").get("job"): sys.exit("REFUSED: already fired on this chip; move %s aside to refire" % RUN)
    pubs = [(c, None, SHOTS) for s in sets for _ in range(COPIES) for c in s["circs"]] + [(c, None, RO_SHOTS) for c in ro_circs(be, union)]
    stated = {str(s["n"]): stated_selfscore(be, s) for s in sets}
    sp = m.clean_sampler(be); m.assert_clean(sp); j = sp.run(pubs)
    m.s_save("", dict(job=j.job_id(), options=m.options_line(sp.options), record_stamp=m.record_stamp(be), record=record_snapshot(be), union=union, stated=stated,
                      sets=[dict(n=s["n"], lay=s["lay"], prs=[list(p) for p in s["prs"]], ncz=s["ncz"]) for s in sets], fired_at=time.time()))
    print("job submitted", j.job_id(), "(%d PUBs)" % len(pubs)); m.show_job_options(j, "v7")


def zeros_share(counts, n):
    tot = sum(counts.values()); return sum(c for k, c in counts.items() if set(k.replace(" ", "")) == {"0"}) / tot


def mode_fetch():
    st = m.s_load(""); svc = m.get_service(); j = svc.job(st["job"]); s = str(j.status()); print("job", s)
    if "DONE" not in s.upper(): return False
    r = j.result(); k = 0; out = {}
    for ss in st["sets"]:
        vals = np.zeros((COPIES, NPTS))
        for c in range(COPIES):
            for i in range(NPTS): vals[c, i] = zeros_share(m.counts_of(r[k]), ss["n"]); k += 1
        out[str(ss["n"])] = vals.tolist()
    ro = []
    for flip in (0, 1):
        cd = m.counts_of(r[k]); k += 1; nb = len(st["union"]); tally = np.zeros(nb)
        for key, c in cd.items():
            b = key.replace(" ", "")[::-1]; tally += np.array([int(ch) for ch in b]) * c
        ro.append((tally / RO_SHOTS).tolist())
    try: billed = j.usage()
    except Exception: billed = None
    m.s_save("", dict(selfscore=out, ro_p10=ro[0], ro_p11=ro[1], billed_qpu_seconds=billed)); m.show_job_options(j, "v7"); print("saved"); return True


def record_snapshot(be):
    """The calibration record as served now: readout (symmetric and one-way), single-qubit gate errors, CZ errors. Saved with the run at fire time."""
    t = be.target; n = be.num_qubits; nan = float("nan"); props = None
    try: props = be.properties()
    except Exception: pass
    def one(q, key):
        try: return float(props.qubit_property(q, key)[0])
        except Exception: return None
    def gerr(g, q):
        try: e = t[g][q].error; return None if e is None else float(e)
        except Exception: return None
    ge = {"sx": {str(q): gerr("sx", (q,)) for q in range(n)}, "x": {str(q): gerr("x", (q,)) for q in range(n)},
          "cz": {"%d,%d" % k: gerr("cz", k) for k in t["cz"]}}
    ro = [float(t["measure"][(q,)].error) if (q,) in t["measure"] and t["measure"][(q,)].error is not None else None for q in range(n)]
    return dict(stamp=m.record_stamp(be), n=n, readout_error=ro, prob_meas0_prep1=[one(q, "prob_meas0_prep1") for q in range(n)],
                prob_meas1_prep0=[one(q, "prob_meas1_prep0") for q in range(n)], gate_error=ge, source="record served by the backend at fire time")


def onequbit_fade(be, ss, rec):
    """Single-qubit gate fade of the compiled circuit: product over every sx and x gate of (1 - error in the record), on the saved layout.
    The circuit is rebuilt on the saved layout (same seed, same level) to count the gates per qubit."""
    n = ss["n"]; a, b = ParameterVector("a", n), ParameterVector("b", n)
    qc = QuantumCircuit(n, n); qc.compose(pfeature(b, n), inplace=True); qc.compose(pfeature(a, n).inverse(), inplace=True); qc.measure(range(n), range(n))
    t = generate_preset_pass_manager(optimization_level=3, backend=be, initial_layout=ss["lay"], seed_transpiler=7).run(qc)
    prs, N = m.cz_counts([t])
    if int(N[0].sum()) != ss["ncz"] or sorted(prs) != sorted(tuple(p) for p in ss["prs"]): sys.exit("REFUSED: the rebuilt circuit at N=%d does not match the recorded CZ count and pairs" % n)
    f = 1.0
    for ci in t.data:
        nm = ci.operation.name
        if nm in ("sx", "x"):
            e = rec["gate_error"][nm].get(str(t.find_bit(ci.qubits[0]).index))
            if e is not None: f *= (1 - e)
    return f


def mode_analyze():
    st = m.s_load(""); svc = m.get_service(); be = svc.backend(m.BACKEND)
    rec = st.get("record") or record_snapshot(be)
    print("KERNEL AUDIT + SHOT BILL, %s; job %s; record stamp at fire %s; options %s" % (m.BACKEND, st["job"], st["record_stamp"], st["options"]))
    if not st.get("record"): print("NOTE: this run has no saved record; the record as served now is used (it may differ from the one in force at fire time).")
    if st.get("billed_qpu_seconds") is not None: print("Run time billed by IBM for this job: %s s" % st["billed_qpu_seconds"])
    rows = []; Z = 2.576
    print("\nMEASURED FADE BY SIZE (self-score of a point against itself; the ideal circuit gives 1 at any size; 99% interval)")
    print("  N   CZ   measured self-score A (99% interval)   record-implied S (IBM's record)   A minus S (99% interval): word")
    for ss in st["sets"]:
        v = np.array(st["selfscore"][str(ss["n"])]); pc = v.mean(1); A = pc.mean(); h = m.tq(COPIES) * pc.std(ddof=1) / math.sqrt(COPIES)
        S, ro, F = st["stated"][str(ss["n"])]; lo, hi = A - S - h, A - S + h; w = m.word(lo, hi, 0.01)
        f1 = onequbit_fade(be, ss, rec); F2 = F * f1; S1 = ro * (F2 + (1 - F2) / 2 ** ss["n"])
        rows.append((ss["n"], A, h, S, S1))
        print("  %2d  %3d   %.3f (%.4f to %.4f)                  %.4f                                %+.4f (%+.4f to %+.4f): %s" % (ss["n"], ss["ncz"], A, A - h, A + h, S, A - S, lo, hi, w))
    print("\nTHE SHOT BILL: shots each kernel entry needs, as a multiple of what a plan built on IBM's published gate and readout errors would budget")
    print("  (IBM's two-qubit and single-qubit gate errors and readout errors; binomial bill at a typical score of 0.5, defined after the data)")
    print("  Qubits   record-implied S   measured A   shot bill")
    for n, A, h, S, S1 in rows:
        K = 0.5; bill = (S1 / A) * (1 - A * K) / (1 - S1 * K)
        print("  %5d    %.3f              %.3f        %.2fx" % (n, S1, A, bill))
    union = st["union"]; p10 = np.array(st["ro_p10"]); p01 = 1 - np.array(st["ro_p11"]); flags = []
    for i, q in enumerate(union):
        for nm, val, key in (("P(1|0)", p10[i], "prob_meas1_prep0"), ("P(0|1)", p01[i], "prob_meas0_prep1")):
            r = rec[key][q]
            if r is None: continue
            lb = val - Z * math.sqrt(max(val * (1 - val), 1e-9) / RO_SHOTS)
            if lb > r: flags.append((q, nm, r, val))
    print("\nREADOUT AUDIT: the one-way error measured now against IBM's matching one-way figure for the same qubit (flagged where the lower 99% bound sits above the record)")
    print("  %d of the %d layout qubits used at any size are flagged" % (len({f[0] for f in flags}), len(union)))
    print("  flagged qubits, largest ratio first (a record of 0.0000 is a figure stated as zero; no ratio quoted):")
    for q, nm, r, val in sorted(flags, key=lambda f: -(f[3] / max(f[2], 1e-4) if f[2] > 0 else 1e9))[:10]:
        print("    q%d %s: today %.4f vs record %.4f%s" % (q, nm, val, r, "" if r == 0 else "  (%.1fx)" % (val / r)))
    print("  (The gate-only reading S and the record's readout figures are IBM's own record at fire time; the bill compares shots for equal kernel precision.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("dry", "fire", "fetch", "analyze"): ap.add_argument("--" + k, action="store_true")
    a = ap.parse_args()
    if a.dry: mode_dry()
    elif a.fire: mode_fire()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
