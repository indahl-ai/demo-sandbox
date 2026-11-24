#!/usr/bin/env python
"""Demo 3 v2 (logistics, p=1 QAOA max-cut on the chip's own coupler graph): the "ours" arm is OUR measured static ZZ folded into the RZZ angles.
ONE layout, ONE parametric transpile (optimization level 0 after the client's layout). The arms differ ONLY in the bound RZZ angles:
  CLIENT    tutorial angles (gamma on every edge), their layout; run under our clean options so the angles are the only difference
  FREE FIX  fold the ZZ the vendor LISTS (record general fields zz_<a><b>, GHz)
  OURS      fold our MEASURED static ZZ (job Z: echoed Ramsey, neighbour 0 vs 1, sign-resolving fit, negative values allowed)
Fold: angle_e = gamma - pi * zeta_e * T_net_e, T_net_e = circuit time before readout minus the edge's own CZ time (n_cz_e * t_cz_e).
The sign convention comes from the chip: an explicit RZZ of known sign inside job Z is fitted with the same code.
Readout: the job-P table (separate job, same options) is applied the same way to every arm. The raw (uncorrected) ratio is printed too.
Runtime changes are OFF for Z, P and main alike (twirling off for gates and measure, dynamical decoupling off; SamplerV2 has no resilience);
options are printed from the object and read back from each job. Every RZZ angle is asserted exact in the transpiled circuits.
Every parameter of our arm cites a hardware source (SOURCES, enforced: --fire refuses on any gap).
Size: the largest node count (100 down in steps of 10) whose total estimate x1.25 stays under 200 s (job Z + job P + main).
Modes: --test (code smoke test only, no result) --dry --fire (staged: Z and P, then after --fetch the main job) --fetch --analyze.
Env: IBM_QUANTUM_TOKEN (your own key), DEMO_INSTANCE. Keys and CRNs never printed. Nothing here commits or pushes."""
import os, sys, json, math, time, argparse, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import demo as v1                      # v1 graph, exact p=1, readout correction helpers; v1 files are not modified

RUN = os.path.join(HERE, "run_v2")
FORGE = v1.FORGE; BACKEND = v1.BACKEND; SEED = 20261119
SHOTS, COPIES, P_SHOTS = 10_000, 8, 4_096
ZZ_TAUS_US = [0, 10, 20, 30, 40, 50]; ZZ_SHOTS = 400; ZREF = 10e3; CTL_GROUPS = 2
GRID = np.linspace(-80e3, 80e3, 3201); assert GRID.min() < 0 < GRID.max()      # negative values allowed
MAX_SECONDS, MARGIN = 200.0, 1.25
BAND = 0.01                                                                   # CONSISTENT = interval wholly inside +-BAND of ratio
SEC_PER_SHOT_CIRCUIT, SEC_PER_JOB = 0.30e-3, 4.0                              # rates the 18-19 Nov runs used
ARMS = ("client", "free", "ours"); LABEL = {"client": "CLIENT (tutorial angles, default layout)", "free": "FREE FIX (fold listed zz)", "ours": "OURS (fold measured zz)"}
HIST = None

def fp(n): return os.path.join(RUN, n)

# ------------------------------------------------------------ options (identical for Z, P and main)
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

# ------------------------------------------------------------ graph, angles (math only), edge colouring
def graph_for(be, n): return v1.build_graph(be.coupling_map.get_edges(), n)

def angles_for(edges, n):
    """Exact p=1 light-cone optimum (pure math). Reads v1's cache when it matches; otherwise computes into run_v2."""
    for d in (v1.RUN, RUN):
        p = os.path.join(d, "angles_%d_%d.json" % (n, len(edges)))
        if os.path.exists(p):
            j = json.load(open(p))
            if [tuple(e) for e in j["edges"]] == list(edges): return j["gamma"], j["beta"], j["ideal"], j["maxcut"]
    g, b, ideal = v1.best_angles(edges); mc = v1.exact_maxcut(edges, n)
    json.dump(dict(edges=[list(e) for e in edges], gamma=g, beta=b, ideal=ideal, maxcut=mc), open(os.path.join(RUN, "angles_%d_%d.json" % (n, len(edges))), "w"))
    return g, b, ideal, mc

def edge_colour(edges):
    """Greedy proper edge colouring (cost layers run in parallel)."""
    cols, used = [], {}
    for e in edges:
        c = 0
        while any(x in used.get(c, set()) for x in e): c += 1
        used.setdefault(c, set()).update(e)
        while len(cols) <= c: cols.append([])
        cols[c].append(e)
    return cols

def zz_groups(edges, adj):
    """Probe edges measured together: no two share a qubit or are joined by a coupler (induced matchings), greedy colouring of the conflict graph.
    adj: logical adjacency {node: set(node)} from the chip's couplers among the layout qubits."""
    def conflict(e, f): return bool(set(e) & set(f)) or any(y in adj[x] for x in e for y in f)
    order = sorted(range(len(edges)), key=lambda i: -sum(conflict(edges[i], edges[j]) for j in range(len(edges)) if j != i))
    groups = []
    for i in order:
        for g in groups:
            if not any(conflict(edges[i], edges[j]) for j in g): g.append(i); break
        else: groups.append([i])
    return [sorted(g) for g in groups]

# ------------------------------------------------------------ circuits: ONE parametric transpile, level 0 after the layout
def build_param(edges, n, beta):
    pv = ParameterVector("th", len(edges)); idx = {e: i for i, e in enumerate(edges)}
    qc = QuantumCircuit(n, n); qc.h(range(n))
    for col in edge_colour(edges):
        for e in col: qc.rzz(pv[idx[e]], e[0], e[1])
    qc.rx(2 * beta, range(n)); qc.measure(range(n), range(n))
    return qc, pv

def transpile_once(be, qc, layout):
    return transpile(qc, be, initial_layout=list(layout), optimization_level=0, seed_transpiler=SEED)

def param_positions(st, ne):
    pos = {}
    for i, ci in enumerate(st.data):
        for p in ci.operation.params:
            if hasattr(p, "parameters") and p.parameters:
                names = [x.name for x in p.parameters]
                if len(names) != 1 or str(p) != names[0]: sys.exit("REFUSED: transpiler altered a parameterized RZZ angle: %s" % p)
                pos.setdefault(int(names[0].split("[")[1][:-1]), []).append(i)
    if sorted(pos) != list(range(ne)) or any(len(v) != 1 for v in pos.values()): sys.exit("REFUSED: transpiler merged, duplicated or dropped an RZZ angle")
    return {e: v[0] for e, v in pos.items()}

def bind(st, pv, angles): return st.assign_parameters({pv[e]: float(angles[e]) for e in range(len(pv))})

def verify_angles(st, bound, angles):
    pos = param_positions(st, len(angles))
    for e, i in pos.items():
        if float(bound.data[i].operation.params[0]) != float(angles[e]): sys.exit("REFUSED: bound angle differs on edge %d" % e)

def edge_times(be, st, pv, layout, edges):
    """T_circ: scheduled time before readout (the static coupling acts for all of it). Per edge: CZ count and CZ time from the transpiled
    circuit and the record's durations; T_net = T_circ - n_cz * t_cz (the vendor's CZ calibration absorbs the static ZZ during its own gate)."""
    tg = be.target; T_all = float(bind(st, pv, np.zeros(len(pv))).estimate_duration(tg)); tm = max(float(tg["measure"][(q,)].duration) for q in layout)
    T = T_all - tm; cnt = {}
    for ci in st.data:
        if ci.operation.name == "cz": k = frozenset(st.find_bit(q).index for q in ci.qubits); cnt[k] = cnt.get(k, 0) + 1
    def tcz(a, b):
        for pr in ((a, b), (b, a)):
            try: return float(tg["cz"][pr].duration)
            except Exception: pass
        return 0.0
    ncz, tc, net = [], [], []
    for u, v in edges:
        a, b = layout[u], layout[v]; n_ = cnt.get(frozenset((a, b)), 0); t_ = tcz(a, b); ncz.append(n_); tc.append(t_); net.append(T - n_ * t_)
    return T, ncz, tc, np.array(net)

def listed_zz(be, layout, edges):
    """Listed static ZZ (Hz) per graph edge from the live record's GENERAL fields zz_<a><b> (unit GHz). Refuses on a missing or ambiguous name."""
    try: gen = {x.name: x for x in be.properties().general}
    except Exception: return None
    chip = {tuple(sorted(e)) for e in be.coupling_map.get_edges()}; out = []
    for u, v in edges:
        a, b = layout[u], layout[v]; names = ["zz_%d%d" % (a, b), "zz_%d%d" % (b, a)]
        for nm in names:                                     # concatenated digits can collide: another chip coupler must not spell the same name
            clash = [pr for pr in chip if pr != tuple(sorted((a, b))) and nm in ("zz_%d%d" % pr, "zz_%d%d" % pr[::-1])]
            if clash and nm in gen: sys.exit("REFUSED: record field %s is ambiguous (%s)" % (nm, clash))
        f = next((gen[nm] for nm in names if nm in gen), None)
        if f is None: return None
        out.append(float(f.value) * (1e9 if str(f.unit) == "GHz" else 1.0))
    return np.array(out)

def arm_angles(arm, gamma, meas, listed, Tn):
    z = {"client": np.zeros(len(Tn)), "free": listed, "ours": meas}[arm]
    return gamma - np.pi * z * Tn

# ------------------------------------------------------------ job Z: static ZZ by echoed Ramsey (neighbour 0 vs 1) + on-chip sign control
def zz_circuit(n, group_edges, js, tau_us, basis, dt, act):
    qc = QuantumCircuit(n); cr = ClassicalRegister(len(group_edges), "p"); qc.add_register(cr)
    probes = [a for a, b in group_edges]; nbrs = [b for a, b in group_edges]
    qc.h(probes)
    if js: qc.x(nbrs)
    qc.barrier()
    def half():
        d = int(round(tau_us * 0.5e-6 / dt / 16)) * 16
        if d: qc.delay(d, range(n), unit="dt")
        qc.barrier()
    half(); qc.x(probes + nbrs); qc.barrier(); half()
    if basis == "Y": qc.sdg(probes)
    qc.h(probes)
    for i, p in enumerate(probes): qc.measure(p, cr[i])
    return qc

def zz_control(n, group_edges, js, tau_us, basis):
    """Same echoed sequence, each half an explicit RZZ of known sign and size pi*ZREF*tau/2 on the group's edges."""
    qc = QuantumCircuit(n); cr = ClassicalRegister(len(group_edges), "p"); qc.add_register(cr)
    probes = [a for a, b in group_edges]; nbrs = [b for a, b in group_edges]; al = np.pi * ZREF * tau_us * 0.5e-6
    qc.h(probes)
    if js: qc.x(nbrs)
    qc.barrier()
    for k in range(2):
        for a, b in group_edges: qc.rzz(al, a, b)
        qc.barrier()
        if k == 0: qc.x(probes + nbrs); qc.barrier()
    if basis == "Y": qc.sdg(probes)
    qc.h(probes)
    for i, p in enumerate(probes): qc.measure(p, cr[i])
    return qc

def zz_meta(groups):
    return [(kind, gi, js, t, bs) for kind in ("zz", "ctl") for gi in (range(len(groups)) if kind == "zz" else range(min(CTL_GROUPS, len(groups))))
            for js in (0, 1) for t in ZZ_TAUS_US for bs in "XY"]

def zz_build(key, n, edges, groups, dt, act):
    kind, gi, js, t, bs = key; ge = [edges[i] for i in groups[gi]]
    return zz_control(n, ge, js, t, bs) if kind == "ctl" else zz_circuit(n, ge, js, t, bs, dt, act)

def mean_z(counts, nbits):
    tot = sum(counts.values()); m = np.zeros(nbits)
    for k, v in counts.items():
        k = k.replace(" ", "")[::-1]
        for i in range(nbits): m[i] += v * (1 - 2 * int(k[i]))
    return m / tot

def zz_fit(counts, groups, sign, rng, nboot=200):
    """counts[(gi, js, tau, basis)] -> counts over the group's probe bits. Returns {edge index: (zeta_Hz, sigma_Hz)}; signed, negative allowed."""
    taus = np.array(ZZ_TAUS_US) * 1e-6; E = np.exp(-1j * sign * 2 * np.pi * np.outer(GRID, taus)); res = {}
    for gi, g in enumerate(groups):
        if not any(k[0] == gi for k in counts): continue
        mz = {k: mean_z(c, len(g)) for k, c in counts.items() if k[0] == gi}
        for i, ei in enumerate(g):
            M = np.array([[[mz[(gi, js, t, bs)][i] for bs in "XY"] for t in ZZ_TAUS_US] for js in (0, 1)])          # (2, T, 2)
            def est(m):
                c = m[..., 0] + 1j * m[..., 1]; prod = c[..., 1, :] * np.conj(c[..., 0, :])
                return GRID[np.argmax(np.abs(prod @ E.T), axis=-1)]
            bt = np.clip(2 * rng.binomial(ZZ_SHOTS, (1 + M) / 2, size=(nboot,) + M.shape) / ZZ_SHOTS - 1, -1, 1)
            res[ei] = (float(est(M)), float(np.std(est(bt))))
    return res

def determine_sign(ctl):
    v = np.array([ctl[e][0] for e in sorted(ctl)])
    if not (np.all(v > 0.5 * ZREF) or np.all(v < -0.5 * ZREF)): return None
    return 1 if v[0] > 0 else -1

# ------------------------------------------------------------ seconds
def est_zz(ng):
    nc = 2 * (ng + min(CTL_GROUPS, ng)) * len(ZZ_TAUS_US) * 2
    return nc * ZZ_SHOTS * (250e-6 + np.mean(ZZ_TAUS_US) * 1e-6 + 2e-6) * 1.3 + 4
def est_p(): return SEC_PER_SHOT_CIRCUIT * 2 * P_SHOTS + SEC_PER_JOB
def est_main(): return SEC_PER_SHOT_CIRCUIT * len(ARMS) * COPIES * SHOTS + SEC_PER_JOB

def adjacency(be, layout):
    inv = {q: i for i, q in enumerate(layout)}; adj = {i: set() for i in range(len(layout))}
    for a, b in be.coupling_map.get_edges():
        if a in inv and b in inv: adj[inv[a]].add(inv[b]); adj[inv[b]].add(inv[a])
    return adj

def choose_size(be, nmax=100):
    """Largest n (nmax down in steps of 10) whose total estimate x MARGIN is under MAX_SECONDS. Layout for the estimate: the identity on q0..n-1
    (the graph is the coupler map on those qubits); the real layout is the client's transpile and is checked against it later."""
    for n in range(nmax, 9, -10):
        ed = graph_for(be, n); lay = list(range(n)); ng = len(zz_groups(ed, adjacency(be, lay)))
        tot = est_zz(ng) + est_p() + est_main()
        if tot * MARGIN <= MAX_SECONDS: return n, ng, tot
    sys.exit("REFUSED: no size fits under %d s" % MAX_SECONDS)

# ------------------------------------------------------------ readings (same P table for every arm)
def arm_ratios(bits_by_arm, ptab_v, rec_v, edges, exact, n):
    gP, gR = v1.guard_nodes(ptab_v, n), v1.guard_nodes(rec_v, n); drop = gP | gR
    kept = [e for e in edges if e[0] not in drop and e[1] not in drop]; p10, p01 = v1.pv(ptab_v, n)
    out = {}
    for a, copies in bits_by_arm.items():
        raw, cor = [], []
        for bits in copies:
            Z, ZZ = v1.marg_bits(bits, edges); raw.append(v1.ratio(ZZ, exact, kept)); cor.append(v1.ratio(v1.correct(Z, ZZ, p10, p01)[1], exact, kept))
        out[a] = dict(raw=np.array(raw), cor=np.array(cor))
    return out, kept, sorted(gP), sorted(gR)

def t99(n): return float(stats.t.ppf(0.995, n - 1))

def interval(v):
    n = len(v); m = float(np.mean(v)); h = t99(n) * float(np.std(v, ddof=1)) / math.sqrt(n); return m, m - h, m + h

def word(lo, hi):
    if lo > -BAND and hi < BAND: return "CONSISTENT"
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "CANNOT_TELL"

def report(res, kept, ne, ideal_over_max, gP, gR, label=""):
    f = lambda t: "%.4f (99%% interval %.4f to %.4f)" % t
    L = ["%sapproximation ratio = measured cut / noiseless QAOA cut (noiseless QAOA itself reaches %.3f of the exact max cut); %d copies per arm; %d of %d edges kept for every arm" % (
        label, ideal_over_max, len(res["client"]["cor"]), kept, ne)]
    for a in ARMS: L.append("%s  %-42s readout-corrected with job P: %s   raw: %s" % (label, LABEL[a], f(interval(res[a]["cor"])), f(interval(res[a]["raw"]))))
    for a, b, nm in (("ours", "free", "OURS vs FREE FIX"), ("ours", "client", "OURS vs CLIENT")):
        d = res[a]["cor"] - res[b]["cor"]; m, lo, hi = interval(d)
        L.append("%s  OUTCOME %s: difference in ratio %+.4f (99%% interval %+.4f to %+.4f; paired by copy; ABOVE = ours higher, BELOW = ours lower, CONSISTENT = interval inside +-%.2f): %s" % (label, nm, m, lo, hi, BAND, word(lo, hi)))
    L.append("%s  denominator guard (1 - p10 - p01 < %.1f), nodes dropped: today %s, record %s. The same job-P table is applied to every arm; its own shot error is common to the arms." % (label, v1.GAP_MIN, gP or "none", gR or "none"))
    return "\n".join(L)

# ------------------------------------------------------------ sources (hardware only); the fire refuses on any gap
def record_stamp(be):
    try: return str(be.properties().last_update_date)
    except Exception: return None

def sources_rows(be, zz, pjob, ncz, tc, T, lst, layout):
    rows, bad = [], []; stamp = record_stamp(be); jz = (zz or {}).get("job_id")
    rows.append(("ZZ value per edge", "job Z %s (fresh, separate from main)" % jz if jz and zz.get("measured") else "MISSING"))
    rows.append(("ZZ sign convention", "on-chip explicit-RZZ control inside job Z %s" % jz if jz and zz.get("sign") in (1, -1) else "MISSING"))
    rows.append(("readout table per qubit (every arm)", "job P %s" % pjob if pjob else "MISSING"))
    rows.append(("CZ time per edge", "live record %s (target durations ns %s)" % (stamp, sorted(set(round(t * 1e9, 1) for t in tc))) if stamp and tc and all(t > 0 for t in tc) else "MISSING"))
    rows.append(("CZ count per edge", "transpiled circuit (structural), values %s" % sorted(set(ncz)) if ncz else "MISSING"))
    rows.append(("circuit time before readout", "live record durations, scheduled circuit: %.3f us" % (T * 1e6) if T else "MISSING"))
    rows.append(("FREE FIX listed zz", "live record %s, general fields zz_<a><b>" % stamp if lst is not None and stamp else "MISSING"))
    rows.append(("layout, graph, tutorial angles", "client's own (level-3 layout on the record; exact p=1 optimum): not our parameters"))
    for r in rows:
        if "MISSING" in r[1]: bad.append(r[0])
    return rows, bad

def print_sources(rows):
    print("SOURCES (parameter -> hardware source):")
    for r in rows: print("  %-38s %s" % r)

# ------------------------------------------------------------ io
def s_load(): return json.load(open(fp("state.json"))) if os.path.exists(fp("state.json")) else {}
def s_save(d): cur = s_load(); cur.update(d); json.dump(cur, open(fp("state.json"), "w"), indent=1)

def unpack(arr, n): return v1.unpack(arr, n)

def setup(be, n):
    edges = graph_for(be, n); gamma, beta, ideal, maxcut = angles_for(edges, n)
    c, lay = v1.transpile_client(v1.build_logical(edges, n, gamma, beta), be)           # the client's level-3 pick: used for the LAYOUT only
    qc, pv = build_param(edges, n, beta); st = transpile_once(be, qc, lay)
    nsw = sum(1 for ci in st.data if ci.operation.name == "swap")
    n2 = sum(1 for ci in st.data if ci.operation.name == "cz")
    if n2 != 2 * len(edges) or nsw: sys.exit("REFUSED: level-0 circuit needs routing (cz %d, swaps %d, edges %d); layout not native" % (n2, nsw, len(edges)))
    return dict(n=n, edges=edges, gamma=gamma, beta=beta, ideal=ideal, maxcut=maxcut, layout=lay, st=st, pv=pv, client_circ=c)

def gate_check(x, listed, meas, Tn):
    rows = {}
    for a in ARMS:
        ang = arm_angles(a, x["gamma"], meas, listed, Tn); b = bind(x["st"], x["pv"], ang); verify_angles(x["st"], b, ang)
        rows[a] = (dict(b.count_ops()), b.depth()); print("  %-42s gate counts %s depth %d" % (LABEL[a], rows[a][0], rows[a][1]))
    if any(rows[a] != rows["client"] for a in ARMS): sys.exit("REFUSED: arms differ in gate counts or depth")
    print("  gate counts and depth identical across arms; every bound RZZ angle in the transpiled circuit equals the angle set (exact)")

# ------------------------------------------------------------ modes
def mode_test():
    """CODE SMOKE TEST ONLY. Offline, small graph, fake chip, ideal Aer. Plumbing and refusals only: no result, no headline, no constant."""
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    be = v1.fake_backend(); n = 12; print("SMOKE TEST ONLY (code plumbing; no result, no headline, no constant comes from this)")
    edges = graph_for(be, n); gamma, beta, ideal, maxcut = angles_for(edges, n); lay = list(range(n))
    # the manual cost layer is the same unitary as the QAOAAnsatz one (exact algebra check on 8 nodes)
    e8 = graph_for(be, 8); g8, b8, _, _ = angles_for(e8, 8); q1, p1 = build_param(e8, 8, b8); q1.remove_final_measurements(); q1 = q1.assign_parameters({p: g8 for p in p1})
    q2 = v1.build_logical(e8, 8, g8, b8); q2.remove_final_measurements()
    ok = abs(abs(np.vdot(Statevector(q1).data, Statevector(q2).data)) - 1) < 1e-9; print("manual QAOA circuit equals the tutorial ansatz (exact algebra, 8 nodes): %s" % ("OK" if ok else "FAILED"))
    qc, pv = build_param(edges, n, beta); st = transpile_once(be, qc, lay); pos = param_positions(st, len(edges)); print("level-0 transpile: %d edges, every RZZ angle appears once" % len(pos))
    x = dict(n=n, edges=edges, gamma=gamma, beta=beta, st=st, pv=pv); T, ncz, tc, Tn = edge_times(be, st, pv, lay, edges)
    L = np.full(len(edges), 1.0e3); Mz = np.full(len(edges), -2.0e3); gate_check(x, L, Mz, Tn)
    groups = zz_groups(edges, adjacency(be, lay)); print("ZZ groups: %d (largest %d edges); circuits in job Z: %d" % (len(groups), max(map(len, groups)), len(zz_meta(groups))))
    sim = AerSimulator(seed_simulator=3); rng = np.random.default_rng(SEED); cc = {}
    for key in zz_meta(groups):
        if key[0] == "ctl": cc[key[1:]] = sim.run(zz_build(key, n, edges, groups, be.dt, lay), shots=ZZ_SHOTS).result().get_counts()
    ctl = zz_fit(cc, groups, 1, rng, nboot=5); sign = determine_sign(ctl); print("control-fit path executed; sign resolved: %s" % (sign in (1, -1)))
    exact = v1.zz_p1(edges, gamma, beta); bits = {}
    for a in ARMS:
        b = bind(st, pv, arm_angles(a, gamma, Mz, L, Tn))
        bits[a] = [v1.bits_from_counts(sim.run(b, shots=300, seed_simulator=7 + c).result().get_counts(), n) for c in range(3)]
    tab = {i: (0.01, 0.01, 10, 4096, 10, 4096) for i in range(n)}; res, kept, gP, gR = arm_ratios(bits, tab, tab, edges, exact, n)
    txt = report(res, len(kept), len(edges), ideal / maxcut, gP, gR, "SMOKE "); print("analysis path executed on %d lines of output (numbers withheld in the smoke test)" % len(txt.splitlines()))
    zz = dict(job_id=None, measured={}, sign=None); rows, bad = sources_rows(be, zz, None, ncz, tc, T, None, lay)
    print("source guard with nothing measured refuses: %s (missing %d rows)" % ("OK" if bad else "FAILED", len(bad)))
    print("size rule path: estimates Z/P/main = %s" % [round(v, 1) for v in (est_zz(len(groups)), est_p(), est_main())])

def mode_dry():
    svc = v1.get_service(); u = v1.usage_remaining(svc); be = svc.backend(BACKEND); print("backend %s operational=%s; usage_remaining_seconds=%.0f" % (BACKEND, be.status().operational, u))
    n, ng, tot = choose_size(be); print("size chosen by the 200 s rule: %d nodes, %d probe groups, total estimate %.0f s (x%.2f = %.0f)" % (n, ng, tot, MARGIN, tot * MARGIN))
    x = setup(be, n); edges, lay = x["edges"], x["layout"]; print("graph: %d nodes, %d edges; tutorial gamma %.6f beta %.6f; noiseless cut %.3f of max cut %d" % (n, len(edges), x["gamma"], x["beta"], x["ideal"], x["maxcut"]))
    print("layout (client's level-3 pick on the record): %s%s" % (lay[:12], " ..." if len(lay) > 12 else "")); print("ONE transpile (optimization level 0 after the layout): depth %d, cz %d" % (x["st"].depth(), x["st"].count_ops().get("cz", 0)))
    T, ncz, tc, Tn = edge_times(be, x["st"], x["pv"], lay, edges); lst = listed_zz(be, lay, edges)
    if lst is None: sys.exit("REFUSED: zz_<a><b> not found in the live record for every edge")
    print("circuit time before readout %.3f us; cz per edge %s; cz time %s ns; T_net %.3f to %.3f us" % (T * 1e6, sorted(set(ncz)), sorted(set(round(t * 1e9, 1) for t in tc)), Tn.min() * 1e6, Tn.max() * 1e6))
    print("listed zz (kHz) from the live record general fields: min %.2f median %.2f max %.2f over %d edges; per-circuit fold angle pi*zeta*T_net (rad): median %.4f" % (
        lst.min() / 1e3, np.median(lst) / 1e3, lst.max() / 1e3, len(lst), np.median(np.pi * lst * Tn)))
    print("gate-count and exact-angle check (placeholder measured = 0.7 x listed, only so the arms carry different angles):"); gate_check(x, lst, 0.7 * lst, Tn)
    groups = zz_groups(edges, adjacency(be, lay)); print("job Z: %d probe groups (largest %d edges), %d circuits x %d shots (%d echoed-Ramsey, %d on-chip sign control)" % (
        len(groups), max(map(len, groups)), len(zz_meta(groups)), ZZ_SHOTS, sum(1 for k in zz_meta(groups) if k[0] == "zz"), sum(1 for k in zz_meta(groups) if k[0] == "ctl")))
    for lab in ("Z", "P", "main"):
        s = clean_sampler(be); assert_clean(s); print("Sampler options for job %s: %s" % (lab, options_line(s.options)))
    ez, ep, em = est_zz(len(groups)), est_p(), est_main(); print("estimated seconds: Z %.0f + P %.0f + main %.0f = %.0f (x%.2f = %.0f; cap %d; remaining %.0f)" % (ez, ep, em, ez + ep + em, MARGIN, (ez + ep + em) * MARGIN, MAX_SECONDS, u))
    rows, bad = sources_rows(be, None, None, ncz, tc, T, lst, lay); print("pending SOURCES (filled by jobs Z and P; --fire refuses until they exist):"); print_sources(rows)
    if (ez + ep + em) * MARGIN > MAX_SECONDS or (ez + ep + em) * MARGIN > u: print("WOULD REFUSE")

def mode_fire():
    svc = v1.get_service(); be = svc.backend(BACKEND); st = s_load(); u = v1.usage_remaining(svc)
    n, ng, tot = choose_size(be); x = setup(be, n); edges, lay = x["edges"], x["layout"]; groups = zz_groups(edges, adjacency(be, lay))
    if "job_z" not in st:
        ez, ep = est_zz(len(groups)), est_p(); print("estimate Z %.0f + P %.0f + main %.0f = %.0f (x%.2f = %.0f; cap %d; remaining %.0f)" % (ez, ep, est_main(), tot, MARGIN, tot * MARGIN, MAX_SECONDS, u))
        if tot * MARGIN > MAX_SECONDS or tot * MARGIN > u: sys.exit("REFUSED: estimate x%.2f above the cap or the remaining seconds" % MARGIN)
        s_save(dict(n=n, layout=lay, edges=[list(e) for e in edges], gamma=x["gamma"], beta=x["beta"], ideal=x["ideal"], maxcut=x["maxcut"], groups=groups,
                    record_stamp=record_stamp(be), est_seconds=tot, staged_at=time.time()))
        json.dump(v1.props_of(be), open(fp("props_fire.json"), "w"))
        circs = [(transpile(zz_build(k, n, edges, groups, be.dt, lay), be, initial_layout=lay, optimization_level=0, seed_transpiler=SEED), None, ZZ_SHOTS) for k in zz_meta(groups)]
        sz = clean_sampler(be); assert_clean(sz); jz = sz.run(circs); s_save(dict(job_z=jz.job_id())); print("job Z (static ZZ) submitted", jz.job_id()); show_job_options(jz, "Z")
        sp = clean_sampler(be); assert_clean(sp); jp = sp.run([(c, None, P_SHOTS) for c in v1.p_circuits(be.num_qubits, lay)]); s_save(dict(job_p=jp.job_id())); print("job P (readout calibration) submitted", jp.job_id()); show_job_options(jp, "P")
        return print("Next: --fetch until both are DONE, then --fire again to submit the main job.")
    if "job_main" in st: sys.exit("REFUSED: main job already fired; move run_v2/state.json aside to refire")
    if not (os.path.exists(fp("zz.json")) and os.path.exists(fp("raw_P.npz"))): sys.exit("REFUSED: run --fetch first (jobs Z and P must be DONE and saved)")
    zz = json.load(open(fp("zz.json")))
    if zz.get("job_id") != st["job_z"] or st["n"] != n or st["layout"] != lay or [tuple(e) for e in st["edges"]] != edges: sys.exit("REFUSED: size, layout or graph changed since Z and P (the record updated); move run_v2/state.json aside and redo")
    for j in (st["job_z"], st["job_p"]):
        if "DONE" not in str(svc.job(j).status()).upper(): sys.exit("REFUSED: a calibration job is not DONE")
    T, ncz, tc, Tn = edge_times(be, x["st"], x["pv"], lay, edges); lst = listed_zz(be, lay, edges)
    rows, bad = sources_rows(be, zz, st["job_p"], ncz, tc, T, lst, lay); print_sources(rows)
    if bad: sys.exit("REFUSED: no hardware source for: %s" % "; ".join(bad))
    if sorted(map(int, zz["measured"])) != list(range(len(edges))): sys.exit("REFUSED: measured ZZ missing edges")
    Mz = np.array([zz["measured"][str(e)][0] for e in range(len(edges))])
    print("gate-count and exact-angle check before firing:"); gate_check(x, lst, Mz, Tn)
    if tot * MARGIN > u: sys.exit("REFUSED: estimate above remaining seconds")
    bound = {a: bind(x["st"], x["pv"], arm_angles(a, x["gamma"], Mz, lst, Tn)) for a in ARMS}; order, pubs = [], []
    for c in range(COPIES):
        rot = list(ARMS[c % 3:] + ARMS[:c % 3])
        for a in rot: order.append((a, c)); pubs.append((bound[a], None, SHOTS))
    sm = clean_sampler(be); assert_clean(sm); ol = options_line(sm.options); print("Sampler options for main: " + ol)
    print("measured zz (kHz): min %.2f median %.2f max %.2f; sign from chip %d" % (Mz.min() / 1e3, np.median(Mz) / 1e3, Mz.max() / 1e3, zz["sign"]))
    jm = sm.run(pubs); s_save(dict(job_main=jm.job_id(), order=order, main_options=ol, T=T, Tn=list(Tn), listed_hz=list(lst), measured_hz=list(Mz), sources=rows))
    print("main job submitted", jm.job_id(), "pubs", len(pubs)); show_job_options(jm, "main")

def mode_fetch():
    svc = v1.get_service(); st = s_load(); n = st["n"]; edges = [tuple(e) for e in st["edges"]]; groups = st["groups"]
    if "job_z" in st and not os.path.exists(fp("zz.json")):
        j = svc.job(st["job_z"]); s = str(j.status()); print("job Z:", s)
        if "DONE" in s.upper():
            res = j.result(); cz, cc = {}, {}
            for key, pr in zip(zz_meta(groups), res): (cz if key[0] == "zz" else cc)[key[1:]] = pr.data.p.get_counts()
            rng = np.random.default_rng(SEED); ctl = zz_fit(cc, groups, 1, rng); sign = determine_sign(ctl)
            print("on-chip control (explicit RZZ equal to +%.0f kHz), fitted with sign=+1 (kHz, first 6): %s" % (ZREF / 1e3, {e: round(v[0] / 1e3, 2) for e, v in list(ctl.items())[:6]}))
            if sign is None: sys.exit("REFUSED: the on-chip sign control did not return a clean +-%.0f kHz on every control edge; sign cannot be fixed from hardware" % (ZREF / 1e3))
            fit = zz_fit(cz, groups, sign, rng); be = svc.backend(BACKEND); lst = listed_zz(be, st["layout"], edges)
            json.dump(dict(job_id=st["job_z"], sign=sign, measured={str(e): list(v) for e, v in fit.items()}, listed_hz={str(e): float(v) for e, v in enumerate(lst)} if lst is not None else {},
                           control_fit_hz={str(e): list(v) for e, v in ctl.items()}), open(fp("zz.json"), "w"), indent=1)
            vals = np.array([fit[e][0] for e in range(len(edges))]); print("saved zz.json; sign from chip = %d; measured zz (kHz): min %.2f median %.2f max %.2f; negative on %d edges; median sigma %.2f" % (
                sign, vals.min() / 1e3, np.median(vals) / 1e3, vals.max() / 1e3, int((vals < 0).sum()), np.median([v[1] for v in fit.values()]) / 1e3)); show_job_options(j, "Z")
        else: print("job Z not finished; run --fetch again later")
    if "job_p" in st and not os.path.exists(fp("raw_P.npz")):
        j = svc.job(st["job_p"]); s = str(j.status()); print("job P:", s)
        if "DONE" in s.upper():
            r = j.result(); np.savez_compressed(fp("raw_P.npz"), c_0=unpack(r[0].data.c.array, n), c_1=unpack(r[1].data.c.array, n)); show_job_options(j, "P"); print("saved raw_P.npz")
        else: print("job P not finished; run --fetch again later")
    if "job_main" in st and not os.path.exists(fp("raw_main.npz")):
        j = svc.job(st["job_main"]); s = str(j.status()); print("job main:", s)
        if "DONE" in s.upper():
            r = j.result(); arrs = {}
            for (a, c), pr in zip(st["order"], r): arrs["%s_%d" % (a, c)] = unpack(pr.data.c.array, n)
            np.savez_compressed(fp("raw_main.npz"), **arrs); show_job_options(j, "main"); print("saved raw_main.npz")
        else: print("job main not finished; run --fetch again later")

def mode_analyze():
    st = s_load(); n, lay = st["n"], st["layout"]; edges = [tuple(e) for e in st["edges"]]; zz = json.load(open(fp("zz.json")))
    z = np.load(fp("raw_P.npz")); ptab = v1.table_from_bits(z["c_0"], z["c_1"])
    rec_v = v1.virt(v1.record_readout(json.load(open(fp("props_fire.json")))), lay)
    zm = np.load(fp("raw_main.npz")); bits = {a: [zm["%s_%d" % (a, c)] for c in range(COPIES)] for a in ARMS}
    exact = v1.zz_p1(edges, st["gamma"], st["beta"]); res, kept, gP, gR = arm_ratios(bits, ptab, rec_v, edges, exact, n)
    vals = np.array([zz["measured"][str(e)][0] for e in range(len(edges))]); lst = np.array(st["listed_hz"])
    print("HARDWARE. job Z %s: measured zz (kHz) min %.2f median %.2f max %.2f (negative on %d edges); listed (live record) median %.2f; sign from chip %d" % (
        zz["job_id"], vals.min() / 1e3, np.median(vals) / 1e3, vals.max() / 1e3, int((vals < 0).sum()), np.median(lst) / 1e3, zz["sign"]))
    print("zz per edge (kHz) measured +-sigma / listed, first 8: " + "; ".join("%d: %.2f+-%.2f / %.2f" % (e, zz["measured"][str(e)][0] / 1e3, zz["measured"][str(e)][1] / 1e3, lst[e] / 1e3) for e in range(8)))
    print(report(res, len(kept), len(edges), st["ideal"] / st["maxcut"], gP, gR, "HARDWARE. "))
    print("sources: zz job %s, readout job P %s, main job %s, record stamp %s; options main: %s" % (zz["job_id"], st["job_p"], st["job_main"], st.get("record_stamp"), st.get("main_options")))

# ------------------------------------------------------------ PILOT (rule 5: only the chip can drop a treatment)
PILOT_COPIES, PILOT_SHOTS, PILOT_CAP = 4, 6_000, 30.0

def mode_pilot():
    """--fire --pilot: hardware pilot on DEMO_BACKEND, runtime changes off, two stages each <= 30 s of QPU (job Z, then a 3-arm main, same shots per copy).
    State in run_v2/pilot_<chip>/. Prints the measured |OURS-FREE FIX| and |OURS-CLIENT| over the pilot's own copy floor (99% half-width of the paired copies).
    No readout correction (identical for every arm). Blocks with a 30-minute deadline per stage."""
    global RUN
    RUN = os.path.join(RUN, "pilot_" + BACKEND.replace("ibm_", "")); os.makedirs(RUN, exist_ok=True)
    svc = v1.get_service(); be = svc.backend(BACKEND); st = s_load(); u = v1.usage_remaining(svc)
    n, ng, tot = choose_size(be); x = setup(be, n); edges, lay = x["edges"], x["layout"]; groups = zz_groups(edges, adjacency(be, lay))
    ez = est_zz(len(groups)); em = SEC_PER_SHOT_CIRCUIT * len(ARMS) * PILOT_COPIES * PILOT_SHOTS + SEC_PER_JOB
    print("PILOT %s estimate: job Z %.0f s, then main %.0f s (cap %.0f each; remaining %.0f)" % (BACKEND, ez, em, PILOT_CAP, u))
    if ez > PILOT_CAP or em > PILOT_CAP or (ez + em) * MARGIN > u: sys.exit("REFUSED: a pilot stage is over %.0f s or the remaining seconds" % PILOT_CAP)
    def wait(label):
        t0 = time.time()
        while time.time() - t0 < 1800:
            time.sleep(20); mode_fetch()
            if label == "z" and os.path.exists(fp("zz.json")): return
            if label == "main" and os.path.exists(fp("raw_main.npz")): return
        sys.exit("pilot stage %s not DONE after 1800 s" % label)
    if "job_z" not in st:
        s_save(dict(n=n, layout=lay, edges=[list(e) for e in edges], gamma=x["gamma"], beta=x["beta"], ideal=x["ideal"], maxcut=x["maxcut"], groups=groups, record_stamp=record_stamp(be), pilot=True))
        circs = [(transpile(zz_build(k, n, edges, groups, be.dt, lay), be, initial_layout=lay, optimization_level=0, seed_transpiler=SEED), None, ZZ_SHOTS) for k in zz_meta(groups)]
        sz = clean_sampler(be); assert_clean(sz); jz = sz.run(circs); s_save(dict(job_z=jz.job_id())); print("pilot job Z submitted", jz.job_id()); show_job_options(jz, "Z")
    if not os.path.exists(fp("zz.json")): wait("z")
    zz = json.load(open(fp("zz.json"))); st = s_load()
    T, ncz, tc, Tn = edge_times(be, x["st"], x["pv"], lay, edges); lst = listed_zz(be, lay, edges)
    if lst is None: sys.exit("REFUSED: zz_<a><b> not in the live record for every edge")
    Mz = np.array([zz["measured"][str(e)][0] for e in range(len(edges))]); gate_check(x, lst, Mz, Tn)
    if "job_main" not in st:
        bound = {a: bind(x["st"], x["pv"], arm_angles(a, x["gamma"], Mz, lst, Tn)) for a in ARMS}; order, pubs = [], []
        for c in range(PILOT_COPIES):
            for a in list(ARMS[c % 3:] + ARMS[:c % 3]): order.append((a, c)); pubs.append((bound[a], None, PILOT_SHOTS))
        sm = clean_sampler(be); assert_clean(sm); ol = options_line(sm.options); print("Sampler options for pilot main: " + ol)
        jm = sm.run(pubs); s_save(dict(job_main=jm.job_id(), order=order, main_options=ol, Tn=list(Tn), listed_hz=list(lst), measured_hz=list(Mz))); print("pilot main submitted", jm.job_id()); show_job_options(jm, "main")
    if not os.path.exists(fp("raw_main.npz")): wait("main")
    st = s_load(); zm = np.load(fp("raw_main.npz")); exact = v1.zz_p1(edges, st["gamma"], st["beta"]); r = {}
    for a in ARMS: r[a] = np.array([v1.ratio(v1.marg_bits(zm["%s_%d" % (a, c)], edges)[1], exact, edges) for c in range(PILOT_COPIES)])
    print("PILOT raw ratio vs noiseless (no readout correction, same for every arm), %d copies x %d shots: " % (PILOT_COPIES, PILOT_SHOTS) + "; ".join("%s %.4f" % (a, r[a].mean()) for a in ARMS))
    for b, nm in (("free", "OURS minus FREE FIX"), ("client", "OURS minus CLIENT")):
        d = r["ours"] - r[b]; half = t99(PILOT_COPIES) * float(d.std(ddof=1)) / math.sqrt(PILOT_COPIES)
        print("PILOT RATIO %s: measured |diff| %.5f (signed %+.5f) over the pilot's own copy floor %.5f (99%% half-width of the paired copies) = %.2f" % (nm, abs(d.mean()), d.mean(), half, abs(d.mean()) / half))
    print("sources: zz job %s, main job %s, record stamp %s, options %s" % (zz["job_id"], st["job_main"], st.get("record_stamp"), st.get("main_options")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in ("test", "dry", "fire", "fetch", "analyze", "pilot"): ap.add_argument("--" + m, action="store_true")
    a = ap.parse_args()
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.fire and a.pilot: mode_pilot()
    elif a.fire: mode_fire()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
