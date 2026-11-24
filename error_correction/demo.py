#!/usr/bin/env python
"""Error correction: a distance-3 rotated surface-code patch on ibm_kingston (DEMO_BACKEND), the code's OWN default compile vs ours.
Why this rung: in the phase-flip repetition code the client's own default compile left the data waiting in the Z basis, where the readout
turn costs nothing, so our correction had nothing to fix . In a surface code the data are entangled under X-type AND Z-type
checks, so a turn on a waiting data qubit is a phase flip that the X-type checks read, whatever the compiler does.
The client's run: the textbook patch (9 data, 4 X-type and 4 Z-type checks), X-basis memory (data start in |+>), R rounds of CNOT checks
with a barrier around each readout, then the data read in the X basis; each round count compiled ONCE at level 3 with no layout given
(the client default, routing included). After every readout each data qubit carries one placeholder rz(p) between barriers; the
compiler routes it with its data qubit, so it lands on whichever physical qubit holds that data qubit after that readout.
Arms, the SAME compiled circuit with the placeholders bound (nothing re-compiled after binding):
  CE   the client: every placeholder 0 (rz(0) is no pulse and no time)
  OE   ours: placeholder = -turn of the physical qubit holding that data qubit through that readout (job K, this session, earlier job)
  OEm  the sign test (pilot only): +turn
Pilot (--pilot, done 19 Nov on three chips): R = 3, raw X-check detections (scored) and Z-check detections (control).
Full run: R = 1, 2, 3, 4, arms CE and OE. Readings: X and Z detections per R, and logical error (the same matching decoder, uniform
weights, on both arms) per R and per round from the fit across R. The full fire refuses unless this chip's own pilot showed the turn real.
Modes: --test (code smoke test, never a number) --dry --fire --fire-main --fetch --analyze, each with or without --pilot. Only
--fire / --fire-main submit. Env: IBM_QUANTUM_TOKEN (your own IBM Quantum key; required for every mode that talks to IBM), DEMO_INSTANCE, DEMO_BACKEND (default
ibm_kingston), DEMO_FAKE=1 with --dry. Keys / CRNs are never printed."""
import os, sys, json, math, time, argparse, warnings
from functools import lru_cache
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit import Parameter

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("DEMO_BACKEND", "ibm_kingston")
PILOT = "--pilot" in sys.argv
RUN_BASE = os.path.join(HERE, "run" if BACKEND == "ibm_kingston" else "run_" + BACKEND)
PILOT_DIR = RUN_BASE + "_pilot"                    # where this chip's recorded pilot result is read from (the sandbox points it at ./recorded)
RUN = RUN_BASE + ("_pilot" if PILOT else "")       # pilot state: run_pilot/ (kingston), run_<backend>_pilot/ (others)
DIST = 3
if PILOT:
    R_LIST = (3,); ARMS = ("CE", "OE", "OEm"); COPIES, SHOTS = 12, 700; MAX_SECONDS = 30.0
else:
    R_LIST = (1, 2, 3, 4); ARMS = ("CE", "OE"); COPIES, SHOTS = 16, 2000; MAX_SECONDS = 200.0
CAL_SHOTS = 800
K_STEPS = (1, 2, 4)                       # job K: readouts per circuit
SEED_T = 7                                # the client compile's transpiler seed (fixed so --dry and --fire compile alike on one target)
BAND_DET = 0.002                          # CONSISTENT: detection difference inside +-BAND_DET (set before data)
BAND_LOG = 0.002                          # CONSISTENT: logical error (per R, and per round) difference inside +-BAND_LOG (set before data)
POWER_MIN = 3.0                           # the full fire needs this chip's pilot ratio at least this
EST_MARGIN = 1.25
SEC_PER_SHOT_MIN = 0.375e-3; REP_DELAY = 250e-6; SEC_PER_JOB = 4.0
TABLE_AGE_CAP_MIN = 60.0; WAIT_S = 1800; POLL_S = 20
COH_MIN = 0.10
SEED = 20261119; SEED_GAP = 1000003

def fp(n): return os.path.join(RUN, n)

# ---------------------------------------------------------------- the patch
def patch():
    """Rotated distance-3 patch: data on a 3x3 grid; checks on plaquettes (X where i+j even, Z where odd); X boundaries top/bottom,
    Z boundaries left/right. Returns (data index by grid point, list of (type, [corner grid points in NW,NE,SW,SE order]))."""
    data = {(i, j): k for k, (i, j) in enumerate([(i, j) for i in range(DIST) for j in range(DIST)])}
    stabs = []
    for i in range(-1, DIST):
        for j in range(-1, DIST):
            c = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]; n = sum(x in data for x in c); t = "X" if (i + j) % 2 == 0 else "Z"
            if n == 4 or (n == 2 and ((t == "X" and i in (-1, DIST - 1)) or (t == "Z" and j in (-1, DIST - 1)))): stabs.append((t, c))
    return data, stabs
DATA_IDX, STABS = patch(); ND = len(DATA_IDX); NA = len(STABS); NQ = ND + NA
SUPPORT = [[DATA_IDX[x] for x in c if x in DATA_IDX] for _, c in STABS]
XS = [a for a, (t, _) in enumerate(STABS) if t == "X"]; ZS = [a for a, (t, _) in enumerate(STABS) if t == "Z"]

def logical_x():
    """The logical X the memory reads: a weight-DIST data column with even overlap with every Z check."""
    for c in range(DIST):
        L = [DATA_IDX[(i, c)] for i in range(DIST)]
        if all(len(set(L) & set(SUPPORT[a])) % 2 == 0 for a in ZS): return L
    raise RuntimeError("no logical X column")
LOGICAL = logical_x()

def marker(r, j): return f"m{r}_{j}"

def sc_circuit(R):
    """The client's memory. Clbits: round r check a -> r*NA + a; final data j -> R*NA + j."""
    D = list(range(ND)); A = list(range(ND, NQ))
    qc = QuantumCircuit(NQ, R * NA + ND)
    qc.h(D); qc.barrier()
    for r in range(R):
        for a, (t, _) in zip(A, STABS):
            if t == "X": qc.h(a)
        for step in range(4):                       # X checks in Z order (NW,NE,SW,SE), Z checks in N order (NW,SW,NE,SE)
            for a, (t, c) in zip(A, STABS):
                g = c[(0, 1, 2, 3)[step] if t == "X" else (0, 2, 1, 3)[step]]
                if g in DATA_IDX: qc.cx(a, DATA_IDX[g]) if t == "X" else qc.cx(DATA_IDX[g], a)
        for a, (t, _) in zip(A, STABS):
            if t == "X": qc.h(a)
        qc.barrier()
        for k, a in enumerate(A): qc.measure(a, r * NA + k)
        for a in A: qc.reset(a)
        qc.barrier()
        for j in D: qc.rz(Parameter(marker(r, j)), j)
        qc.barrier()
    qc.h(D)
    for j in D: qc.measure(j, R * NA + j)
    return qc

def compile_client(be, R):
    """The client default compile (level 3, no layout), once per round count. Returns the compiled circuit and where everything landed."""
    tc = transpile(sc_circuit(R), backend=be, optimization_level=3, seed_transpiler=SEED_T)
    return tc, landing(tc, R)

def compile_all(be):
    out = [compile_client(be, R) for R in R_LIST]
    return [t for t, _ in out], [l for _, l in out]

def landing(tc, R):
    mk, bad = {}, []
    for ci in tc.data:
        for p in ci.operation.params:
            ps = getattr(p, "parameters", None)
            if not ps: continue
            for prm in ps:
                ok = ci.operation.name == "rz" and len(ci.qubits) == 1 and str(p) == prm.name
                if not ok: bad.append(f"R={R}: placeholder {prm.name} merged or moved by the compiler ({ci.operation.name} {p})")
                mk.setdefault(prm.name, []).append(tc.find_bit(ci.qubits[0]).index)
    meas = {tc.find_bit(ci.clbits[0]).index: tc.find_bit(ci.qubits[0]).index for ci in tc.data if ci.operation.name == "measure"}
    S = [[meas[r * NA + a] for a in range(NA)] for r in range(R)]
    H = []
    for r in range(R):
        h = []
        for j in range(ND):
            q = mk.get(marker(r, j), [])
            if len(q) != 1: bad.append(f"R={R}: placeholder {marker(r, j)} found {len(q)} times"); h.append(None)
            else: h.append(q[0])
        H.append(h)
        if set(x for x in h if x is not None) & set(S[r]): bad.append(f"R={R} round {r}: a data placeholder sits on a qubit read in that round")
    final = [meas[R * NA + j] for j in range(ND)]
    qubits = sorted(set(sum(S, [])) | set(x for h in H for x in h if x is not None) | set(final))
    return dict(R=R, S=S, H=H, final=final, qubits=qubits, n2q=n2q(tc), bad=bad)

def n2q(tc): return sum(v for g, v in tc.count_ops().items() if g in ("cz", "ecr", "cx"))

def key_of(land, r): return (tuple(sorted(land["S"][r])), tuple(land["H"][r]))
def keystr(k): return json.dumps([list(k[0]), list(k[1])])

def k_keys(lands):
    """One job-K setting per distinct (readout set, data set) across all rounds of all compiled circuits."""
    keys = []
    for land in lands:
        for r in range(land["R"]):
            k = key_of(land, r)
            if k not in keys: keys.append(k)
    return keys

def union_qubits(lands): return sorted(set(q for l in lands for q in l["qubits"]))

def k_circuit(nq, S, H, N, basis, plant=None):
    """Job K on the device width: the data-holding qubits H on the equator while the round's readout set S is read and reset N times
    (exactly that round's readout), then H read in X or Y. Gives each H qubit's turn per readout and its coherence."""
    qc = QuantumCircuit(nq, N * len(S) + len(H))
    qc.h(list(H)); qc.barrier()
    for k in range(N):
        for i, q in enumerate(S): qc.measure(q, k * len(S) + i)
        for q in S: qc.reset(q)
        qc.barrier()
        if plant is not None:
            for q in H: qc.rz(math.radians(plant.get(q, 0.0)), q)
            qc.barrier()
    if basis == "Y": qc.sdg(list(H))
    qc.h(list(H))
    for j, q in enumerate(H): qc.measure(q, N * len(S) + j)
    return qc

def p_circuit(nq, qs, ones):
    qc = QuantumCircuit(nq, len(qs))
    if ones: qc.x(list(qs))
    for k, q in enumerate(qs): qc.measure(q, k)
    return qc

def tp0(qc, be): return transpile(qc, backend=be, optimization_level=0, initial_layout=list(range(qc.num_qubits)))

def build_cal(be, lands, plant=None):
    nq = be.num_qubits; qs = union_qubits(lands)
    cal = [tp0(p_circuit(nq, qs, False), be), tp0(p_circuit(nq, qs, True), be)]
    for S, H in k_keys(lands):
        for N in K_STEPS:
            for b in ("X", "Y"): cal.append(tp0(k_circuit(nq, S, H, N, b, plant), be))
    return cal

def cal_index(lands):
    idx = {}; n = 2
    for key in k_keys(lands):
        for N in K_STEPS:
            for b in ("X", "Y"): idx[(key, N, b)] = n; n += 1
    return idx

# ---------------------------------------------------------------- arms
def arm_values(arm, land, turn, plant=None):
    """Placeholder values (radians). turn: {round: {phys: deg}}. plant (smoke test only) adds the fake chip's turn."""
    v = {}
    for r, h in enumerate(land["H"]):
        for j, q in enumerate(h):
            t = turn[r][q]; c = 0.0 if arm == "CE" else -t if arm == "OE" else +t
            v[marker(r, j)] = math.radians(c + (plant.get(q, 0.0) if plant else 0.0))
    return v

def bind(tc, vals):
    return tc.assign_parameters({p: vals[p.name] for p in tc.parameters})

def check_bound(tc, land, turn):
    """Refusal reasons: placeholders merged/moved, routing put data on a read qubit, or a bound arm that differs from the client's
    compiled circuit in anything but our placeholder angles."""
    bad = list(land["bad"])
    base = bind(tc, arm_values("CE", land, turn))
    for a in ARMS[1:]:
        b = bind(tc, arm_values(a, land, turn)); want = arm_values(a, land, turn)
        if len(b.data) != len(base.data): bad.append(f"R={land['R']} {a}: instruction count differs"); continue
        for x, y, z in zip(base.data, b.data, tc.data):
            if x.operation.name != y.operation.name or [base.find_bit(q).index for q in x.qubits] != [b.find_bit(q).index for q in y.qubits]:
                bad.append(f"R={land['R']} {a}: gate list differs from the client's"); break
            pn = [getattr(p, "parameters", set()) for p in z.operation.params]
            if any(pn):
                nm = next(iter(pn[0])).name
                if abs(float(y.operation.params[0]) - want[nm]) > 1e-9: bad.append(f"R={land['R']} {a}: {nm} bound to the wrong angle"); break
            elif [float(p) for p in x.operation.params if isinstance(p, (int, float))] != [float(p) for p in y.operation.params if isinstance(p, (int, float))]:
                bad.append(f"R={land['R']} {a}: a client angle changed"); break
    return bad

def check_all(tcs, lands, turns): return [w for tc, l, t in zip(tcs, lands, turns) for w in check_bound(tc, l, t)]

# ---------------------------------------------------------------- record, bits, tables
def record_figures(be, qs):
    t = be.target
    return dict(readout={str(q): float(t["measure"][(q,)].error) for q in qs},
                t_meas={str(q): float(t["measure"][(q,)].duration or 2.3e-6) for q in qs},
                t_reset={str(q): float(t["reset"][(q,)].duration or 2.3e-6) if ("reset" in t.operation_names and (q,) in t["reset"]) else 2.3e-6 for q in qs})

def to_bits(strings): return np.array([[c == "1" for c in s.replace(" ", "")[::-1]] for s in strings], np.int8)
def pack(bits): return np.packbits(bits, axis=1, bitorder="little")
def unpack(arr, n): return np.unpackbits(arr, axis=1, count=n, bitorder="little").astype(np.int8)
def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
def near(x, ref): return x + 2 * np.pi * np.round((ref - x) / (2 * np.pi))

def make_table(bits, lands, cal_job):
    qs = union_qubits(lands)
    ro = {str(q): (float(bits[0][:, k].mean()), float(1 - bits[1][:, k].mean())) for k, q in enumerate(qs)}
    idx = cal_index(lands); per_key = {}
    for key in k_keys(lands):
        S, H = key; ph, co = {}, {}
        for N in K_STEPS:
            e = {}
            for b in ("X", "Y"):
                bb = bits[idx[(key, N, b)]][:, N * len(S): N * len(S) + len(H)]
                a01 = np.array([ro[str(q)][0] for q in H]); a10 = np.array([ro[str(q)][1] for q in H])
                p1t = np.clip((bb.mean(axis=0) - a01) / np.maximum(1 - a01 - a10, 0.5), 0, 1); e[b] = 1 - 2 * p1t
            ph[N] = np.arctan2(e["Y"], e["X"]); co[N] = np.hypot(e["X"], e["Y"])
        tk, ck = {}, {}
        for j, q in enumerate(H):
            p = [ph[N][j] for N in K_STEPS]; u = [p[0]]; d1 = wrap(p[1] - p[0]); u.append(u[0] + d1)
            u.append(u[1] + near(wrap(p[2] - p[1]), 2 * d1))
            tk[str(q)] = round(math.degrees(np.polyfit(np.array(K_STEPS, float), u, 1)[0]), 4); ck[str(q)] = float(co[4][j])
        per_key[keystr(key)] = dict(turn=tk, coh4=ck)
    by_R = {}
    for land in lands:
        turn, coh = {}, {}
        for r in range(land["R"]):
            kk = per_key[keystr(key_of(land, r))]; turn[str(r)] = kk["turn"]; coh[str(r)] = kk["coh4"]
        by_R[str(land["R"])] = dict(turn=turn, coh4=coh)
    return dict(readout=ro, keys=per_key, by_R=by_R, source=cal_job)

def turn_of(tab, R): return {int(r): {int(q): v for q, v in d.items()} for r, d in tab["by_R"][str(R)]["turn"].items()}

def provenance(tab, lands):
    rows, bad = [], []
    for land in lands:
        R = land["R"]; tb = tab["by_R"][str(R)]
        for r, h in enumerate(land["H"]):
            for j, q in enumerate(h):
                t = tb["turn"][str(r)].get(str(q)); c = tb["coh4"][str(r)].get(str(q), 0.0)
                rows.append(dict(setting=f"R={R}: turn on data {j} at q{q} through readout {r}, deg", value=t, source=tab["source"],
                                 how="job K, slope over 1/2/4 readouts of that round's readout set"))
                if t is None or not np.isfinite(t): bad.append(f"R={R} round {r} data {j} q{q}: no turn")
                if c < COH_MIN: bad.append(f"R={R} round {r} data {j} q{q}: coherence after 4 readouts {c:.3f} below {COH_MIN}")
    if not tab.get("source"): bad.append("calibration job id missing")
    return rows, bad

# ---------------------------------------------------------------- readings
def detectors(bits, R):
    """X-basis memory. X detectors: round 0 against +1, rounds 1..R-1 against the round before, and a final layer from the data's X
    readout. Z detectors (control): rounds 1..R-1 against the round before (none when R = 1). Returns (X detector bits, Z detector bits)."""
    m = bits[:, : R * NA].reshape(-1, R, NA); fin = bits[:, R * NA: R * NA + ND]
    xd = [m[:, 0, XS]] + [m[:, r, XS] ^ m[:, r - 1, XS] for r in range(1, R)]
    xd.append(np.stack([fin[:, SUPPORT[a]].sum(axis=1) % 2 for a in XS], axis=1) ^ m[:, R - 1, XS])
    zd = [m[:, r, ZS] ^ m[:, r - 1, ZS] for r in range(1, R)]
    return np.concatenate(xd, axis=1), (np.concatenate(zd, axis=1) if zd else np.zeros((len(bits), 0), np.int8))

@lru_cache(maxsize=None)
def matching_graph(R):
    """Space-time graph for phase flips: node (layer, X check) for layers 0..R, plus one boundary node. Space edge per data qubit
    (between its two X checks, or to the boundary if it has one), time edge per X check between layers (a check readout error).
    Uniform weights. Returns (all-pairs distance, parity of the logical flip along a shortest path)."""
    nx = len(XS); n = nx * (R + 1) + 1; B = n - 1; INF = 10 ** 6
    dist = np.full((n, n), INF, int); par = np.zeros((n, n), np.int8)
    np.fill_diagonal(dist, 0)
    def edge(u, v, f):
        if dist[u, v] > 1: dist[u, v] = dist[v, u] = 1; par[u, v] = par[v, u] = f
    for L in range(R + 1):
        for j in range(ND):
            ch = [k for k, a in enumerate(XS) if j in SUPPORT[a]]; f = int(j in LOGICAL)
            if len(ch) == 2: edge(L * nx + ch[0], L * nx + ch[1], f)
            elif len(ch) == 1: edge(L * nx + ch[0], B, f)
        if L < R:
            for k in range(nx): edge(L * nx + k, (L + 1) * nx + k, 0)
    for k in range(n):                                   # Floyd-Warshall, carrying the parity of the path
        nd = dist[:, k:k + 1] + dist[k:k + 1, :]; better = nd < dist
        par = np.where(better, par[:, k:k + 1] ^ par[k:k + 1, :], par); dist = np.where(better, nd, dist)
    return dist, par

@lru_cache(maxsize=None)
def _match(R, defects):
    """Exact minimum-weight matching of the defects, each to another defect or to the boundary. Returns the logical parity."""
    dist, par = matching_graph(R); B = dist.shape[0] - 1; d = list(defects)
    if len(d) > 16:                                       # greedy fallback for rare large defect sets
        rest = list(d); f = 0
        while rest:
            u = rest.pop(0); best = (dist[u, B], None)
            for v in rest:
                if dist[u, v] < best[0]: best = (dist[u, v], v)
            if best[1] is None: f ^= par[u, B]
            else: f ^= par[u, best[1]]; rest.remove(best[1])
        return int(f)
    @lru_cache(maxsize=None)
    def go(mask):
        if mask == 0: return (0, 0)
        i = (mask & -mask).bit_length() - 1; m2 = mask & ~(1 << i); u = d[i]
        c, f = go(m2); best = (c + dist[u, B], f ^ par[u, B])
        mm = m2
        while mm:
            jb = mm & -mm; j = jb.bit_length() - 1; mm &= ~jb
            c, f = go(m2 & ~jb); cand = (c + dist[u, d[j]], f ^ par[u, d[j]])
            if cand[0] < best[0]: best = cand
        return best
    return int(go((1 << len(d)) - 1)[1])

def logical_fail(bits, R):
    """Per shot: 1 if the decoded logical X disagrees with +1."""
    xd, _ = detectors(bits, R); fin = bits[:, R * NA: R * NA + ND]
    obs = fin[:, LOGICAL].sum(axis=1) % 2
    pred = np.array([_match(R, tuple(np.flatnonzero(row))) for row in xd], np.int8)
    return (obs ^ pred).astype(np.int8)

def eps_per_round(pl, Rs):
    """Logical error per round from 1 - 2 p_L(R) = A (1 - 2 eps)^R, fitted across R (the fit takes out preparation and readout)."""
    y = np.log(np.clip(1 - 2 * np.asarray(pl, float), 1e-6, None)); s = np.polyfit(np.asarray(Rs, float), y, 1)[0]
    return (1 - math.exp(s)) / 2

def t99(n): return float(stats.t.ppf(0.995, n - 1))
def word(lo, hi, eq): return "ABOVE" if lo > eq else "BELOW" if hi < -eq else "CONSISTENT" if (-eq < lo and hi < eq) else "CANNOT_TELL"
def ci(x): x = np.asarray(x, float); h = t99(len(x)) * x.std(ddof=1) / math.sqrt(len(x)); return x.mean(), x.mean() - h, x.mean() + h

NAME = {"CE": "client, its own default compile", "OE": "ours, rz(-turn) per data qubit per readout", "OEm": "sign test, rz(+turn)"}

def pilot_report(copies, land, header=""):
    R = land["R"]; dx = {a: [] for a in ARMS}; dz = {a: [] for a in ARMS}
    for cp in copies:
        for a in ARMS: x, z = detectors(cp[(R, a)], R); dx[a].append(x.mean()); dz[a].append(z.mean())
    L = [header + f"distance-3 surface-code patch, X-basis memory, R = {R}, {len(copies)} copies x {SHOTS} shots; {land['n2q']} two-qubit gates (routing included)"]
    for a in ARMS:
        m, lo, hi = ci(dx[a]); mz, loz, hiz = ci(dz[a])
        L.append(f"  {a:4s} {NAME[a]}: X detection {m:.4f} ({lo:.4f} to {hi:.4f}); Z detection {mz:.4f} ({loz:.4f} to {hiz:.4f})")
    res = {}
    for a in ARMS[1:]:
        m, lo, hi = ci(np.array(dx[a]) - np.array(dx["CE"])); res[a] = (m, lo, hi)
        L.append(f"  OUTCOME {a} minus CE, X detectors: {m:+.5f} (99% interval {lo:+.5f} to {hi:+.5f}): {word(lo, hi, BAND_DET)}  (BELOW = fewer detections than the client)")
        m2, lo2, hi2 = ci(np.array(dz[a]) - np.array(dz["CE"]))
        L.append(f"  control {a} minus CE, Z detectors: {m2:+.5f} (99% interval {lo2:+.5f} to {hi2:+.5f}): {word(lo2, hi2, BAND_DET)}  (should be CONSISTENT: rz commutes with Z checks)")
    m, lo, hi = res["OE"]; floor = (hi - lo) / 2; ratio = abs(m) / floor if floor > 0 else float("inf")
    real = "turn" if hi < -BAND_DET else None; sign_ok = res["OE"][0] < res["OEm"][0]
    L.append(f"  sign: rz(-turn) {'beats' if sign_ok else 'does NOT beat'} rz(+turn) (OE {res['OE'][0]:+.5f} vs OEm {res['OEm'][0]:+.5f} against CE)")
    L.append(f"  PART THE CHIP SHOWS REAL: {'turn' if real else 'NONE (ours is not BELOW the client)'}")
    L.append(f"  POWER RATIO (chip, pilot): {ratio:.1f}  (= |OE minus CE| / 99% half-width of the paired copy difference)")
    if os.path.isdir(RUN): json.dump(dict(real=real, ratio=ratio, sign_ok=bool(sign_ok), outcomes={a: list(v) for a, v in res.items()}), open(fp("pilot_result.json"), "w"), indent=1)
    return "\n".join(L)

def full_report(copies, lands, header=""):
    Rs = [l["R"] for l in lands]; n = len(copies)
    dx = {(R, a): [] for R in Rs for a in ARMS}; dz = dict((k, []) for k in dx); pl = dict((k, []) for k in dx)
    for cp in copies:
        for R in Rs:
            for a in ARMS:
                b = cp[(R, a)]; x, z = detectors(b, R)
                dx[(R, a)].append(x.mean()); dz[(R, a)].append(z.mean() if z.size else np.nan); pl[(R, a)].append(logical_fail(b, R).mean())
    L = [header + f"distance-3 surface-code patch, X-basis memory, R = {', '.join(map(str, Rs))}; {n} copies x {SHOTS} shots per arm per R; "
         + "two-qubit gates (routing included) " + ", ".join(f"R={l['R']}: {l['n2q']}" for l in lands)]
    L.append("decoder: the same exact minimum-weight matching on both arms (uniform weights, space-time graph of the X checks); logical = data column " + str(LOGICAL))
    for R in Rs:
        L.append(f"R = {R}:")
        for a in ARMS:
            m, lo, hi = ci(dx[(R, a)]); mp, lop, hip = ci(pl[(R, a)])
            zs = "" if R == 1 else "; Z detection %.4f (%.4f to %.4f)" % ci(dz[(R, a)])
            L.append(f"  {a:3s} {NAME[a]}: X detection {m:.4f} ({lo:.4f} to {hi:.4f}){zs}; logical error {mp:.4f} ({lop:.4f} to {hip:.4f})")
        m, lo, hi = ci(np.array(dx[(R, 'OE')]) - np.array(dx[(R, 'CE')]))
        L.append(f"  OUTCOME OE minus CE, X detectors: {m:+.5f} (99% interval {lo:+.5f} to {hi:+.5f}): {word(lo, hi, BAND_DET)}")
        if R > 1:
            m, lo, hi = ci(np.array(dz[(R, 'OE')]) - np.array(dz[(R, 'CE')]))
            L.append(f"  control OE minus CE, Z detectors: {m:+.5f} (99% interval {lo:+.5f} to {hi:+.5f}): {word(lo, hi, BAND_DET)}  (should be CONSISTENT)")
        m, lo, hi = ci(np.array(pl[(R, 'OE')]) - np.array(pl[(R, 'CE')]))
        L.append(f"  OUTCOME OE minus CE, logical error at R = {R}: {m:+.5f} (99% interval {lo:+.5f} to {hi:+.5f}): {word(lo, hi, BAND_LOG)}")
    ep = {a: [eps_per_round([pl[(R, a)][c] for R in Rs], Rs) for c in range(n)] for a in ARMS}
    for a in ARMS:
        m, lo, hi = ci(ep[a]); L.append(f"logical error per round, {a} ({NAME[a]}): {m:.4f} (99% interval {lo:.4f} to {hi:.4f}), fit across R per copy")
    m, lo, hi = ci(np.array(ep["OE"]) - np.array(ep["CE"])); w = word(lo, hi, BAND_LOG)
    L.append(f"OUTCOME OE minus CE, logical error per round: {m:+.5f} (99% interval {lo:+.5f} to {hi:+.5f}): {w}  (BELOW = ours fails less often)")
    mc = float(np.mean(ep["CE"])); mo = float(np.mean(ep["OE"]))
    if w == "BELOW" and mo > 0: L.append(f"  ratio, client over ours, per round: {mc/mo:.2f}x (means of the per-copy fits; the outcome word above is the result)")
    return "\n".join(L)

def audit_lines(tab, rec, n):
    L = ["readout today worse than the record (P(1|0) or P(0|1) above the record figure, beyond job P's 99% interval):"]
    for q in sorted(rec["readout"], key=int):
        r = rec["readout"][q]
        for lab, p in zip(("P(1|0)", "P(0|1)"), tab["readout"][q]):
            k = round(p * n); lo = float(stats.beta.ppf(0.005, k, n - k + 1)) if k else 0.0
            if lo > r: L.append(f"  q{q} {lab}: record {r:.4f}, today {p:.4f} (lower 99% bound {lo:.4f}), {p/max(r,1e-9):.1f}x")
    return L

def table_lines(tab):
    allv = [v for k in tab["keys"].values() for v in k["turn"].values()]
    big = sorted({(q, v) for k in tab["keys"].values() for q, v in k["turn"].items()}, key=lambda kv: -abs(kv[1]))[:6]
    return [f"job K ({tab['source']}): {len(tab['keys'])} readout settings; turn per readout on the data-holding qubits, median {np.median(allv):+.2f} deg; largest "
            + ", ".join(f"q{q} {v:+.2f}" for q, v in big)]

# ---------------------------------------------------------------- seconds
def seconds(be, tcs, lands, rec):
    rate = lambda d: max(d + REP_DELAY, SEC_PER_SHOT_MIN)
    tm = max(rec["t_meas"].values()); main = 0.0
    for tc, land in zip(tcs, lands):
        try: dur = float(tc.estimate_duration(be.target, unit="s"))
        except Exception: dur = land["R"] * (40 * 7e-8 + 2 * tm) + tm
        main += COPIES * len(ARMS) * SHOTS * rate(dur)
    cal = 2 * CAL_SHOTS * rate(tm)
    for S, H in k_keys(lands):
        tw = max(rec["t_meas"][str(q)] + rec["t_reset"][str(q)] for q in S)
        cal += sum(2 * CAL_SHOTS * rate(N * tw + tm) for N in K_STEPS)
    tot = main + cal + 2 * SEC_PER_JOB
    return dict(main=main, cal=cal, total=tot, with_margin=tot * EST_MARGIN)

# ---------------------------------------------------------------- IBM helpers (as demo 5)
def read_key():
    tok = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if not tok: sys.exit("Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)")
    return tok

def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService as Svc
    tok = read_key(); want = os.environ.get("DEMO_INSTANCE", "open-instance")
    base = Svc(channel="ibm_quantum_platform", token=tok); crn = None
    for i in base.instances():
        if want.lower() in str(i.get("name", "")).lower(): crn = i["crn"]; break
    if crn is None: sys.exit(f"no instance named like '{want}' on this key")
    return Svc(channel="ibm_quantum_platform", token=tok, instance=crn)

def usage_remaining(svc):
    try: return float(svc.usage().get("usage_remaining_seconds"))
    except Exception as e: sys.exit(f"cannot read usage_remaining_seconds ({type(e).__name__}); refusing")

def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
    return FakeFez() if "fez" in BACKEND else FakeMarrakesh() if "marrakesh" in BACKEND else FakeKingston()

def sampler(be):
    """IBM's software corrections off on every arm (no mitigation in SamplerV2; twirling and DD off). These are the client defaults."""
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=be)
    s.options.dynamical_decoupling.enable = False
    s.options.twirling.enable_gates = False; s.options.twirling.enable_measure = False
    if s.options.dynamical_decoupling.enable or s.options.twirling.enable_gates or s.options.twirling.enable_measure:
        sys.exit("REFUSED: sampler corrections did not switch off")
    return s

def options_of(job):
    try: o = job.inputs.get("options", {}) or {}
    except Exception: return ["options could not be read back from the job"]
    on = []
    if (o.get("dynamical_decoupling") or {}).get("enable"): on.append("dynamical decoupling")
    tw = o.get("twirling") or {}
    if tw.get("enable_gates") or tw.get("enable_measure"): on.append("twirling")
    return on

def s_load(): return json.load(open(fp("state.json"))) if os.path.exists(fp("state.json")) else {}
def s_save(d):
    os.makedirs(RUN, exist_ok=True); cur = s_load(); cur.update(d); json.dump(cur, open(fp("state.json"), "w"), indent=1)

def pub_bits(pub):
    d = pub.data; reg = [getattr(d, a) for a in dir(d) if not a.startswith("_") and hasattr(getattr(d, a), "get_bitstrings")][0]
    return to_bits(reg.get_bitstrings())

def test_turns(lands):
    """Distinct angles used only to prove the binding keeps every angle (dry run)."""
    return [{r: {q: round(-3.0 + 0.37 * k * (-1) ** k + r, 4) for k, q in enumerate(h)} for r, h in enumerate(l["H"])} for l in lands]

def order():
    """Every (copy, R, arm); arm order alternates by copy, R order rotates by copy (drift lands on both arms alike)."""
    out = []
    for c in range(COPIES):
        Rs = list(R_LIST[c % len(R_LIST):] + R_LIST[: c % len(R_LIST)])
        for R in Rs:
            for a in (ARMS if c % 2 == 0 else ARMS[::-1]): out.append((c, R, a))
    return out

def regroup(bits, od):
    copies = [dict() for _ in range(max(o[0] for o in od) + 1)]
    for (c, R, a), b in zip(od, bits): copies[c][(R, a)] = b
    return copies

def save_npz(name, bits):
    os.makedirs(RUN, exist_ok=True)
    np.savez_compressed(fp(name), **{f"p{i}": pack(b) for i, b in enumerate(bits)}, nbits=np.array([b.shape[1] for b in bits]))
def load_npz(name):
    z = np.load(fp(name)); nb = z["nbits"]; return [unpack(z[f"p{i}"], int(nb[i])) for i in range(len(nb))]

def pilot_gate():
    """Rule 5: the full fire on this chip runs only if this chip's own pilot showed the turn real with ratio >= POWER_MIN."""
    p = os.path.join(PILOT_DIR, "pilot_result.json")
    if not os.path.exists(p): return [f"no pilot result for {BACKEND} ({os.path.basename(RUN_BASE)}_pilot/pilot_result.json); fire --pilot first"]
    pr = json.load(open(p)); why = []
    if pr.get("real") != "turn": why.append(f"{BACKEND} pilot did not show the turn real")
    if not (pr.get("ratio", 0) >= POWER_MIN): why.append(f"{BACKEND} pilot ratio {pr.get('ratio')} below {POWER_MIN}")
    return why

# ---------------------------------------------------------------- modes
def print_build(tcs, lands, rec, sec):
    print(f"client default compile (level 3, no layout, seed {SEED_T}), one per round count; qubits {union_qubits(lands)}")
    for tc, land in zip(tcs, lands):
        print(f"  R = {land['R']}: {land['n2q']} two-qubit gates, depth {tc.depth()}")
        for r in range(land["R"]): print(f"    readout {r}: read {sorted(land['S'][r])}; data held on {land['H'][r]}")
        print(f"    final data read on {land['final']}")
    print(f"job-K settings (distinct readout/data sets across all compiles): {len(k_keys(lands))}")
    print("record readout error on the patch qubits: median %.4f, worst " % np.median(list(rec["readout"].values()))
          + ", ".join(f"q{q} {v:.4f}" for q, v in sorted(rec["readout"].items(), key=lambda kv: -kv[1])[:5]))
    print(f"estimated seconds: main {sec['main']:.1f} + calibration {sec['cal']:.1f} + jobs {2*SEC_PER_JOB:.0f} = {sec['total']:.1f}; x{EST_MARGIN} = {sec['with_margin']:.1f} (cap {MAX_SECONDS:.0f})")

def mode_dry():
    fake = os.environ.get("DEMO_FAKE") == "1"
    if fake: be = fake_backend(); u = None; print(f"FAKE DRY (code check on the fake {BACKEND}; not the chip) {'PILOT' if PILOT else 'FULL'}")
    else:
        svc = get_service(); u = usage_remaining(svc); be = svc.backend(BACKEND); st = be.status()
        print(f"backend {BACKEND} operational={st.operational} pending_jobs={st.pending_jobs}; usage_remaining_seconds={u:.0f}; {'PILOT' if PILOT else 'FULL'}")
    tcs, lands = compile_all(be); rec = record_figures(be, union_qubits(lands)); sec = seconds(be, tcs, lands, rec)
    print_build(tcs, lands, rec, sec)
    cal = build_cal(be, lands); print(f"calibration job: {len(cal)} circuits (job P 2, job K {len(cal)-2}); main job: {len(order())} circuits")
    why = check_all(tcs, lands, test_turns(lands))
    print("check with test angles (placeholders standalone, arms identical to the client's compile except our angles): " + ("PASS" if not why else "FAIL: " + "; ".join(why[:5])))
    if not fake: sampler(be); print("sampler: dynamical decoupling off, twirling off")
    if not PILOT: why += pilot_gate()
    if sec["with_margin"] > MAX_SECONDS: why.append(f"estimate x margin {sec['with_margin']:.0f} s above {MAX_SECONDS:.0f}")
    if u is not None and sec["with_margin"] > u: why.append(f"usage_remaining_seconds {u:.0f} below {sec['with_margin']:.0f}")
    if "job_main" in s_load(): why.append("state already holds a fired main job")
    print("no-fire reasons: " + ("none" if not why else "; ".join(why[:8])))

def mode_fire():
    why = [] if PILOT else pilot_gate()
    if why: sys.exit("REFUSED: " + "; ".join(why))
    svc = get_service(); u = usage_remaining(svc); be = svc.backend(BACKEND)
    if "job_cal" in s_load(): sys.exit("REFUSED: state already holds a calibration job; move the run folder aside to refire")
    tcs, lands = compile_all(be); rec = record_figures(be, union_qubits(lands)); sec = seconds(be, tcs, lands, rec)
    why = check_all(tcs, lands, test_turns(lands))
    if sec["with_margin"] > MAX_SECONDS: why.append(f"estimate x margin {sec['with_margin']:.0f} s above {MAX_SECONDS:.0f}")
    if sec["with_margin"] > u: why.append(f"usage_remaining_seconds {u:.0f} below {sec['with_margin']:.0f}")
    if why: sys.exit("REFUSED: " + "; ".join(why[:8]))
    os.makedirs(RUN, exist_ok=True)
    with open(fp("client_compiled.qpy"), "wb") as f: qpy.dump(tcs, f)     # the exact compiled circuits both arms bind into
    s_save(dict(backend=BACKEND, lands=lands, record=rec, est=sec, fired_at=time.time(), pilot=PILOT, R_list=list(R_LIST)))
    jc = sampler(be).run([(c,) for c in build_cal(be, lands)], shots=CAL_SHOTS); s_save(dict(job_cal=jc.job_id()))
    print(f"calibration job (job P readout table + job K turn table) submitted; waiting up to {WAIT_S//60} min")
    t0 = time.time()
    while time.time() - t0 < WAIT_S:
        s = str(svc.job(jc.job_id()).status())
        if "DONE" in s.upper(): s_save(dict(cal_done_at=time.time())); break
        if any(x in s.upper() for x in ("ERROR", "CANCEL")): sys.exit(f"calibration job {s}")
        time.sleep(POLL_S)
    else: sys.exit("calibration job not finished; run --fire-main when it is DONE (the table-age cap still applies)")
    submit_main(svc, be, s_load())

def submit_main(svc, be, st):
    if "job_main" in st: sys.exit("REFUSED: state already holds a fired main job")
    lands = st["lands"]; j = svc.job(st["job_cal"]); s = str(j.status())
    if "DONE" not in s.upper(): sys.exit(f"calibration job {s}; run --fire-main when it is DONE")
    bits = [pub_bits(p) for p in j.result()]; save_npz("cal_bits.npz", bits)
    tab = make_table(bits, lands, st["job_cal"]); json.dump(tab, open(fp("table.json"), "w"), indent=1)
    rows, why = provenance(tab, lands); json.dump(rows, open(fp("provenance.json"), "w"), indent=1)
    why += [f"calibration job ran with {x} on" for x in options_of(j)]
    age = (time.time() - (st.get("cal_done_at") or time.time())) / 60
    if age > TABLE_AGE_CAP_MIN: why.append(f"table {age:.0f} min old, cap {TABLE_AGE_CAP_MIN:.0f}")
    with open(fp("client_compiled.qpy"), "rb") as f: tcs = qpy.load(f)
    turns = [turn_of(tab, l["R"]) for l in lands]; why += check_all(tcs, lands, turns)
    print("\n".join(table_lines(tab)))
    if why: sys.exit("REFUSED main job: " + "; ".join(why[:8]))
    byR = {l["R"]: (tc, l, t) for tc, l, t in zip(tcs, lands, turns)}
    circs = [bind(byR[R][0], arm_values(a, byR[R][1], byR[R][2])) for _, R, a in order()]
    jm = sampler(be).run([(c,) for c in circs], shots=SHOTS)
    s_save(dict(job_main=jm.job_id(), order=order(), main_submitted_at=time.time()))
    print(f"main job submitted ({len(circs)} circuits, every turn from {st['job_cal']}). Next: --fetch, then --analyze (same --pilot flag).")

def mode_fire_main():
    if not PILOT:
        why = pilot_gate()
        if why: sys.exit("REFUSED: " + "; ".join(why))
    svc = get_service(); be = svc.backend(BACKEND); submit_main(svc, be, s_load())

def mode_fetch():
    svc = get_service(); st = s_load(); j = svc.job(st["job_main"]); s = str(j.status()); print(f"main job: {s}")
    if "DONE" not in s.upper(): sys.exit("not finished; run --fetch again later")
    save_npz("main_bits.npz", [pub_bits(p) for p in j.result()])
    use = {}
    for k in ("job_cal", "job_main"):
        try: use[k] = svc.job(st[k]).usage()
        except Exception: pass
    s_save(dict(usage=use)); print(f"saved main_bits.npz; usage {use}")

def mode_analyze():
    st = s_load(); lands = st["lands"]; rec = st["record"]; tab = json.load(open(fp("table.json")))
    copies = regroup(load_npz("main_bits.npz"), [tuple(x) for x in st["order"]])
    kind = "HARDWARE PILOT" if PILOT else "HARDWARE FULL RUN"
    lines = [f"{kind}, {BACKEND}; calibration job {st['job_cal']}, main job {st['job_main']}"] + table_lines(tab)
    lines += audit_lines(tab, rec, CAL_SHOTS)
    lines.append(pilot_report(copies, lands[0], kind + ". ") if PILOT else full_report(copies, lands, kind + ". "))
    out = "\n".join(lines); print(out); open(fp("results.text"), "w").write(out + "\n")

def mode_test():
    """SMOKE TEST of the code path only, never a number: Aer on the fake chip with planted turns on the data-holding qubits."""
    global COPIES, SHOTS, CAL_SHOTS
    COPIES = int(os.environ.get("DEMO_TEST_COPIES", "3")); SHOTS = int(os.environ.get("DEMO_TEST_SHOTS", "60")); CAL_SHOTS = int(os.environ.get("DEMO_TEST_CAL", "300"))
    from qiskit_aer import AerSimulator
    be = fake_backend(); tcs, lands = compile_all(be); rec = record_figures(be, union_qubits(lands))
    print(f"[SMOKE TEST of the code only, not a number: Aer, fake {BACKEND}, ideal gates; planted turns stand in for the chip; {'PILOT' if PILOT else 'FULL'}]")
    print_build(tcs, lands, rec, seconds(be, tcs, lands, rec))
    why = check_all(tcs, lands, test_turns(lands)); print("check with test angles: " + ("PASS" if not why else "FAIL " + "; ".join(why[:3])))
    sim = AerSimulator(method="matrix_product_state")
    run = lambda c, shots, s: to_bits(sim.run(c, shots=shots, seed_simulator=s, memory=True).result().get_memory())
    for tc, land in zip(tcs, lands):
        zero = {r: {q: 0.0 for q in h} for r, h in enumerate(land["H"])}; R = land["R"]
        b0 = run(bind(tc, arm_values("CE", land, zero)), 100, SEED); x0, z0 = detectors(b0, R)
        print(f"R = {R} ideal client circuit, no plant: X detections {x0.mean():.4f}, Z detections {z0.mean() if z0.size else 0:.4f}, logical failures {logical_fail(b0, R).mean():.4f}  (all must be 0)")
    land = lands[-1]; tc = tcs[-1]; R = land["R"]; nfail = 0; nfire = 0
    for r0 in range(R):
        for j in range(ND):
            v = arm_values("CE", land, {r: {q: 0.0 for q in h} for r, h in enumerate(land["H"])}); v[marker(r0, j)] = math.pi
            b = run(bind(tc, v), 4, SEED + 7 * j + r0); xd, _ = detectors(b, R); nfire += int(xd.sum(axis=1).min() > 0); nfail += int(logical_fail(b, R).sum())
    print(f"decoder check, R = {R}: one deliberate phase flip (rz(pi)) on each data qubit after each readout, {R*ND} cases: detected in {nfire}, decoded wrong in {nfail} shots  (must be {R*ND} and 0)")
    pq = sorted({q for l in lands for h in l["H"] for q in h}); plant = {q: (-35.0, -8.0, -20.0, -3.0, -50.0)[k % 5] for k, q in enumerate(pq)}
    cal = build_cal(be, lands, plant)
    bits = [run(c, CAL_SHOTS, SEED + SEED_GAP * (1 + i)) for i, c in enumerate(cal)]
    tab = make_table(bits, lands, "smoke-test"); turns = [turn_of(tab, l["R"]) for l in lands]
    print("planted turn per readout (last R, round 0 data): " + ", ".join(f"q{q} {plant[q]:+.1f}" for q in lands[-1]["H"][0][:8]))
    print("job K recovered (last R, round 0):               " + ", ".join(f"q{q} {turns[-1][0][q]:+.1f}" for q in lands[-1]["H"][0][:8]))
    rows, pw = provenance(tab, lands); print(f"provenance: {len(rows)} settings, refusals: {pw if pw else 'none'}")
    print("check with the measured table: " + ("PASS" if not check_all(tcs, lands, turns) else "FAIL"))
    byR = {l["R"]: (tc, l, t) for tc, l, t in zip(tcs, lands, turns)}
    mb = [run(bind(byR[R][0], arm_values(a, byR[R][1], byR[R][2], plant)), SHOTS, SEED + SEED_GAP * (100 + n)) for n, (c, R, a) in enumerate(order())]
    copies = regroup(mb, order())
    print(pilot_report(copies, lands[0], "SMOKE TEST (not a number). ") if PILOT else full_report(copies, lands, "SMOKE TEST (not a number). "))
    print("[direction check, code only: with planted turns, OE should show fewer X detections and fewer logical failures than CE, Z unchanged]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in ("test", "dry", "fire", "fire-main", "fetch", "analyze", "pilot"): ap.add_argument("--" + m, action="store_true")
    a = ap.parse_args()
    if a.test: mode_test()
    elif a.dry: mode_dry()
    elif a.fire: mode_fire()
    elif a.fire_main: mode_fire_main()
    elif a.fetch: mode_fetch()
    elif a.analyze: mode_analyze()
    else: ap.print_help()
