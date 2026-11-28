"""owner_check.py: an independent record of an IBM machine's readout and T1, for the people who own or rent one.

  python sandbox.py owner history --family heron_r2       our published readings of ibm_fez, ibm_kingston and ibm_marrakesh. No key, no network.
  python sandbox.py owner check   --backend ibm_cleveland  qubit count, the age of IBM's current record, the IBM time a run uses. Submits nothing.
  python sandbox.py owner fire    --backend ibm_cleveland --yes   our calibration check on YOUR machine, on YOUR account (one job)
  python sandbox.py owner analyze --backend ibm_cleveland  collects the job and prints the same table as `history`, for your machine

The measurement is the one behind every calibration reading we publish (https://quantumidleaudit.com/qia/cal/<job>.json), ported unchanged:
one job, every usable qubit at once, physical layout, no optimization. C0: measure at rest (1,000 shots). C1: X, then measure (1,000 shots).
T1: X, a delay of 15, 40, 80, 160 or 320 us, then measure (500 shots each). P(1|0) and P(0|1) get exact (Clopper-Pearson) 99.9% intervals.
IBM's readout figure is itself a count (out of 1,024 or 4,096 shots, read from the record), so it gets its own exact 99.9% interval, and a reading
is called only when the two intervals do not overlap. T1 = B + A exp(-d/T1) with B fixed at the measured P(1|0), a 99.99% profile interval,
called only beyond a 25% margin. A run whose median measured/stated readout or T1 falls outside [0.5, 2.0] calls nothing (known-good seat).
IBM's record is saved when the job is sent, and the record in force when the job ran is pulled by time at analysis.

Your key is read from IBM_QUANTUM_TOKEN, used only to reach your own IBM account, never printed and never written to disk. Results are saved in
./my_runs/ on this machine and on your IBM account. Nothing goes to Heuremen."""
import os, sys, json, time, gzip, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FAMILIES = {f: os.path.join(HERE, 'recorded', f + '.jsonl.gz') for f in ('heron_r2', 'nighthawk_r2', 'heron_r3')}   # nighthawk_r2: ibm_phoenix, ibm_chicago
PUBLISHED = 'https://quantumidleaudit.com/qia/cal/%s.json'
SRO, ST1 = 1000, 500                                     # shots: readout circuits, each T1 delay (qia_cal/cal.py)
BILLED_S = 4                                             # IBM billed 4 s for every one of the 2,171 calibration jobs in our record as of 27 Nov 2026 (120 and 156 qubits)
SPARE_S = 30                                             # fire refuses with less than BILLED_S + SPARE_S left, as mycircuit does
TAG = 'owner-calibration-check'

# ---- the analysis, ported unchanged from qia_cal/callib.py ------------------------------------------------------------------------------------
DCHI = 15.137                                            # chi2(1 dof) 99.99%
DELAYS_US = [0, 15, 40, 80, 160, 320]
GRID = np.logspace(0, 3.5, 700)                          # 1 us .. 3162 us
MARGIN = 1.25


def clopper(k, n):
    """Exact (Clopper-Pearson) 99.9% interval. Returns estimate, low, high."""
    from scipy.stats import beta
    k = np.asarray(k, float); n = float(n); a = 0.001
    lo = np.where(k > 0, beta.ppf(a / 2, k, n - k + 1), 0.0); hi = np.where(k < n, beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
    return k / n, lo, hi


def side(stated, lo, hi):
    if stated is None or not np.isfinite(stated): return 'CANNOT_TELL'
    return 'ABOVE' if stated > hi else ('BELOW' if stated < lo else 'CONSISTENT')


def off_grid(stated, n_ibm):
    """True when IBM's figure is not k/n_ibm. side_counts then compares against the point figure."""
    return abs(stated * n_ibm - round(stated * n_ibm)) > 1e-6


def side_counts(stated, lo, hi, n_ibm=4096):
    """IBM's readout figure is k/n_ibm: it gets its own exact 99.9% interval, and the call is made only when the two do not overlap."""
    if stated is None or not np.isfinite(stated): return 'CANNOT_TELL'
    k = stated * n_ibm
    if off_grid(stated, n_ibm): return side(stated, lo, hi)
    _, ilo, ihi = clopper(np.array([round(k)]), n_ibm); ilo, ihi = float(ilo[0]), float(ihi[0])
    return 'ABOVE' if ilo > hi else ('BELOW' if ihi < lo else 'CONSISTENT')


def side_margin(stated, lo, hi, m=MARGIN):
    """IBM publishes no T1 fit uncertainty: call only beyond a 25% margin past our interval."""
    if stated is None or not np.isfinite(stated): return 'CANNOT_TELL'
    if lo <= stated <= hi: return 'CONSISTENT'
    return 'ABOVE' if stated > hi * m else ('BELOW' if stated < lo / m else 'CANNOT_TELL')


def t1_fit(p1, n, B):
    d = np.asarray(DELAYS_US, float); p1 = np.asarray(p1, float); n = np.asarray(n, float); B = np.asarray(B, float)
    E = np.exp(-d[None, :] / GRID[:, None]); y = p1 - B[:, None]
    def prof(w):
        num = np.einsum('qd,gd,qd->qg', w, E, y); den = np.einsum('qd,gd->qg', w, E * E); A = np.clip(num / den, 0, None)
        r = y[:, None, :] - A[:, :, None] * E[None, :, :]; return np.einsum('qd,qgd->qg', w, r * r), A
    pc = np.clip(p1, 1e-3, 1 - 1e-3); chi, A = prof(n / (pc * (1 - pc)))
    i = chi.argmin(1); m = np.clip(B[:, None] + A[np.arange(len(i)), i][:, None] * E[i], 1e-3, 1 - 1e-3)
    chi, A = prof(n / (m * (1 - m)))
    i = chi.argmin(1); best = GRID[i]; ok = chi <= chi.min(1, keepdims=True) + DCHI
    lo = np.array([GRID[np.where(row)[0].min()] for row in ok]); hi = np.array([GRID[np.where(row)[0].max()] for row in ok])
    edge = (lo <= GRID[0] * 1.001) | (hi >= GRID[-1] * 0.999) | (best < 5.0)
    return best, lo, hi, edge


def ibm_readout_shots(stated):
    """The shot count behind IBM's readout figures, read from the record: 1,024 when every figure is a multiple of 1/1024, else 4,096."""
    v = [x for s in stated.values() for x in (s.get('p10'), s.get('p01')) if x is not None and np.isfinite(x)]
    return 1024 if v and all(abs(x * 1024 - round(x * 1024)) < 1e-6 for x in v) else 4096


def analyse(c0_ones, c1_zeros, t1_ones, shots_ro, shots_t1, stated, n_ibm=None):
    Q = len(c0_ones); p10, l10, h10 = clopper(c0_ones, shots_ro); p01, l01, h01 = clopper(c1_zeros, shots_ro)
    n_ibm = n_ibm or ibm_readout_shots(stated)
    p1 = np.column_stack([1 - p01] + [np.asarray(t1_ones)[:, j] / shots_t1 for j in range(np.asarray(t1_ones).shape[1])])
    n = np.array([shots_ro] + [shots_t1] * (p1.shape[1] - 1)); best, lo, hi, edge = t1_fit(p1, n, p10)
    rows = []
    for q in range(Q):
        s = stated[q]
        rows.append(dict(p10=round(float(p10[q]), 4), p10_ci=[round(float(l10[q]), 4), round(float(h10[q]), 4)], ibm_p10=s.get('p10'), p10_outcome=side_counts(s.get('p10'), l10[q], h10[q], n_ibm),
                         p01=round(float(p01[q]), 4), p01_ci=[round(float(l01[q]), 4), round(float(h01[q]), 4)], ibm_p01=s.get('p01'), p01_outcome=side_counts(s.get('p01'), l01[q], h01[q], n_ibm),
                         t1=round(float(best[q]), 1), t1_ci=[round(float(lo[q]), 1), round(float(hi[q]), 1)], ibm_t1=s.get('t1'),
                         t1_outcome='CANNOT_TELL' if edge[q] else side_margin(s.get('t1'), lo[q], hi[q])))
    ro_ratio = [((r['p10'] + r['p01']) / 2) / ((r['ibm_p10'] + r['ibm_p01']) / 2) for r in rows if r['ibm_p10'] and r['ibm_p01']]
    t1_ratio = [r['t1'] / r['ibm_t1'] for r in rows if r['ibm_t1'] and r['t1_outcome'] != 'CANNOT_TELL']
    mro = float(np.median(ro_ratio)) if ro_ratio else float('nan'); mt1 = float(np.median(t1_ratio)) if t1_ratio else float('nan')
    kg = bool(0.5 <= mro <= 2.0 and 0.5 <= mt1 <= 2.0)
    if not kg:
        for r in rows: r['p10_outcome'] = r['p01_outcome'] = r['t1_outcome'] = 'CANNOT_TELL'
    return rows, dict(ibm_readout_shots=n_ibm, known_good='passed' if kg else 'failed', median_readout_measured_over_stated=round(mro, 3), median_t1_measured_over_stated=round(mt1, 3))


# ---- counting rules for the table (tests/test_owner.py) ---------------------------------------------------------------------------------------
LETTER = {'ABOVE': 'A', 'BELOW': 'B', 'CONSISTENT': 'C', 'CANNOT_TELL': 'T'}


def readout_off(out_p10, out_p01):
    """A qubit reading reads off IBM's record, worse than stated, when either readout quantity is BELOW: IBM's figure, with its own interval,
    sits wholly under our interval. Outcomes are named from IBM's side, so BELOW for a readout error means more error than IBM states."""
    return out_p10 == 'B' or out_p01 == 'B'


def tenfold(lo, ibm, out):
    """An off reading whose interval's low end is at least 10x IBM's figure. IBM figures of 0 have no ratio and are counted apart."""
    return out == 'B' and ibm is not None and ibm > 0 and lo >= 10 * ibm


def excess(lo, ibm):
    """How far above IBM's figure the reading is at the least: the interval's low end minus IBM's figure. Ranks the worst readings."""
    return lo - ibm


def parse_utc(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')) if s else None


def age_hours(run_start, ibm_date):
    t0, t1 = parse_utc(run_start), parse_utc(ibm_date)
    return None if (t0 is None or t1 is None) else (t0 - t1).total_seconds() / 3600


def t1_counts(t1_out):
    """T1 scatters both ways, so it is counted apart: A = IBM states a longer T1 than measured (worse), B = shorter (better)."""
    return t1_out.count('A'), t1_out.count('B')


def summarize(runs):
    """runs: rows in the extract's shape (tools/build_owner_history.py). Returns the counts the table prints, for one chip."""
    s = dict(runs=len(runs), readings=0, off=0, tenfold=0, zero_off=0, t1_worse=0, t1_better=0, t1_called=0, failed_seat=0)
    per_q, ages, worst = {}, [], []
    for r in runs:
        s['failed_seat'] += r.get('known_good') == 'failed'
        for i, q in enumerate(r['q']):
            s['readings'] += 1; per_q.setdefault(q, 0)
            o10, o01 = r['p10_out'][i], r['p01_out'][i]
            if readout_off(o10, o01): s['off'] += 1; per_q[q] += 1
            ten = zero = False
            for k, o in (('p10', o10), ('p01', o01)):
                lo, ibm = r[k + '_lo'][i], r['ibm_' + k][i]
                ten |= tenfold(lo, ibm, o); zero |= (o == 'B' and ibm == 0)
                if o == 'B': worst.append((excess(lo, ibm), r['chip'], q, r['window'][0], k, r[k][i], lo, r[k + '_hi'][i], ibm, r['file']))
            s['tenfold'] += ten; s['zero_off'] += zero
            a = [age_hours(r['window'][0], r[k + '_date'][i]) for k in ('p10', 'p01')]
            a = [x for x in a if x is not None]
            if a: ages.append(max(a))
        w, b = t1_counts(r['t1_out']); s['t1_worse'] += w; s['t1_better'] += b; s['t1_called'] += len(r['t1_out']) - r['t1_out'].count('T')
    s['per_qubit'] = per_q; s['ages'] = ages; s['worst'] = sorted(worst, key=lambda x: -x[0])
    return s


# ---- printing ---------------------------------------------------------------------------------------------------------------------------------
def pct(a, b): return 100.0 * a / b if b else 0.0


def c(n): return '{:,}'.format(n)


def hours(h):
    return '%.1f h' % h if h < 48 else '%.1f days' % (h / 24)


HERON_R2_CHIPS = ('ibm_fez', 'ibm_kingston', 'ibm_marrakesh')


def heron_r2_rows(n):
    """Heron r2 (156 qubits): eight rows of 16 qubits, joined by four bridge qubits after each of the first seven rows. Row r holds qubits
    20r..20r+15, its bridges 20r+16..20r+19 sit under columns 3, 7, 11, 15 (even r) or 1, 5, 9, 13 (odd r). Checked against IBM's coupling map."""
    if n != 156: return None
    out = []
    for r in range(8):
        out.append([(c, r * 20 + c) for c in range(16)])
        if r < 7: out.append([(c, r * 20 + 16 + j) for j, c in enumerate((3, 7, 11, 15) if r % 2 == 0 else (1, 5, 9, 13))])
    return out


def cell(k, n):
    if k == 0: return '   .'
    p = 100.0 * k / n
    return '  <1' if p < 1 else '%4d' % round(p)


def qubit_map(per_q, n_runs, n_qubits):
    rows = heron_r2_rows(n_qubits)
    if rows is not None: print('    (laid out as on the chip)')
    if rows is None:                                      # another layout: by qubit number, sixteen to a line
        qs = sorted(per_q); rows = [[(c, q) for c, q in enumerate(qs[i:i + 16])] for i in range(0, len(qs), 16)]
        print('    (by qubit number, sixteen to a line, not the chip\'s layout)')
    for row in rows:
        line = [' ' * 4] * 16
        for c, q in row: line[c] = cell(per_q.get(q, 0), n_runs)
        print('    ' + ''.join(line).rstrip())


def report(by_chip, title, source_note, one_run=False):
    """Print the table for each chip. by_chip: {chip: [runs]}."""
    print(title); print(source_note); print()
    totals = {}
    for chip, runs in by_chip.items():
        s = summarize(runs); totals[chip] = s
        nq = len({q for r in runs for q in r['q']})
        first, last = min(r['window'][0] for r in runs), max(r['window'][0] for r in runs)
        print('=' * 100)
        print('%s: %s run%s, %s qubit readings, %s to %s' % (chip, c(s['runs']), '' if s['runs'] == 1 else 's', c(s['readings']), first[:16] + 'Z', last[:16] + 'Z'))
        print('  readout read off IBM\'s record, worse than stated: %s of %s qubit readings (%.1f%%)' % (c(s['off']), c(s['readings']), pct(s['off'], s['readings'])))
        off_q = sum(1 for v in s['per_qubit'].values() if v)
        print('  distinct qubits that read off at least once: %d of %d' % (off_q, nq))
        print('  off readings whose whole interval sits at 10x IBM\'s figure or more: %s   (plus %s off where IBM states 0)' % (c(s['tenfold']), c(s['zero_off'])))
        if s['ages']:
            a = np.array(s['ages'])
            print('  age of IBM\'s readout record when the run started: median %s, 90th percentile %s, oldest %s'
                  % (hours(float(np.median(a))), hours(float(np.percentile(a, 90))), hours(float(a.max()))))
        print('  T1, counted apart (it scatters both ways): IBM states a longer T1 than measured in %s readings, a shorter one in %s (of %s called)'
              % (c(s['t1_worse']), c(s['t1_better']), c(s['t1_called'])))
        if s['failed_seat']: print('  %d run%s failed the known-good seat and call%s nothing' % (s['failed_seat'], '' if s['failed_seat'] == 1 else 's', 's' if s['failed_seat'] == 1 else ''))
        print('\n  worst single readings (by how far the interval\'s low end sits above IBM\'s figure):')
        print('    %-14s %5s  %-17s %-7s %-26s %-12s %s' % ('chip', 'qubit', 'run start (UTC)', 'reads', 'measured (99.9% interval)', 'IBM figure', 'published file'))
        for x in s['worst'][:10]:
            _, ch, q, t, k, m, lo, hi, ibm, f = x
            print('    %-14s %5d  %-17s %-7s %.4f [%.4f, %.4f]   %-12.6g %s' % (ch, q, t[:16] + 'Z', 'P(1|0)' if k == 'p10' else 'P(0|1)', m, lo, hi, ibm, PUBLISHED % f if not one_run else f))
        if not s['worst']: print('    none')
        print('\n  qubit map: % of runs in which each qubit\'s readout read off IBM\'s record ("." never, "<1" under 1%)' if not one_run else
              '\n  qubit map: qubits whose readout read off IBM\'s record in this run ("100"), "." where it did not')
        qubit_map(s['per_qubit'], s['runs'], nq if chip in HERON_R2_CHIPS else -1)   # the drawn layout is checked against Heron r2 only
        print()
    return totals


def load_family(family):
    p = FAMILIES.get(family)
    if not p or not os.path.exists(p): sys.exit('no recorded history for %s in this copy. Families: %s' % (family, ', '.join(sorted(FAMILIES))))
    runs = [json.loads(l) for l in gzip.open(p, 'rt') if l.strip()]
    by = {}
    for r in runs: by.setdefault(r['chip'], []).append(r)
    return by



def strong_counts(by):
    """27 Nov 2026 (v1.5): the two cross-chip counts quoted in owner pitches, under the published rule, runs with a failed known-good seat skipped.
    tenx: runs where at least one qubit's readout was called off with its interval's LOW end at 10x IBM's nonzero figure or more
          (tenx_1pct: the same, counting only IBM figures of 1% or more).
    near_total: distinct (chip, qubit) with at least one off readout reading whose interval's low end is above 0.5 while IBM's figure was under 5%."""
    n = tenx = tenx1 = 0; near = set()
    for chip, runs in by.items():
        for r in runs:
            if r.get('known_good') == 'failed': continue
            n += 1; t = t1 = False
            for i, q in enumerate(r['q']):
                for k in ('p10', 'p01'):
                    if r[k + '_out'][i] != 'B': continue
                    ib = r['ibm_' + k][i]
                    if ib and r[k + '_lo'][i] >= 10 * ib:
                        t = True
                        if ib >= 0.01: t1 = True
                    if ib is not None and ib < 0.05 and r[k + '_lo'][i] > 0.5: near.add((chip, q))
            tenx += t; tenx1 += t1
    return dict(runs=n, tenx=tenx, tenx_1pct=tenx1, near_total=len(near))

def history(family):
    by = load_family(family)
    t = report(by, 'RECORDED HISTORY, %s: our published calibration readings of the public %s chips. Nothing is sent anywhere and no key is used.'
               % (family, ', '.join(sorted(by))),
               'Every reading is a job we ran and published. Each row names its file at %s.\n'
               'Readout only in the headline counts. A reading counts as off when IBM\'s figure, given its own exact 99.9%% interval\n'
               '(IBM\'s figures are counts out of 1,024 or 4,096 shots), and our exact 99.9%% interval do not overlap, with ours higher.' % (PUBLISHED % '<file>'))
    off = sum(s['off'] for s in t.values()); n = sum(s['readings'] for s in t.values()); runs = sum(s['runs'] for s in t.values())
    print('RESULT: on %s runs across %d public chips of this family, %s of %s qubit readings (%.1f%%) read off IBM\'s readout record, worse than stated.'
          % (c(runs), len(t), c(off), c(n), pct(off, n)))
    sc = strong_counts(by)
    print('In %s of %s runs (failed seats skipped), at least one qubit\'s readout was at least 10 times worse than IBM\'s figure (interval low end; %s runs counting only IBM figures of 1%% or more).' % (c(sc['tenx']), c(sc['runs']), c(sc['tenx_1pct'])))
    print('%d qubits read wrong more than half the time (interval low end above 0.5, one prepared state) on at least one reading while IBM\'s figure was under 5%%.' % sc['near_total'])
    print('Every one of those readings sat under a record IBM was publishing at the time. Without an independent record of a machine, none of this shows.')
    print('This shows the public machines on the days we ran them. It says nothing about any other machine until that machine is measured.')
    print('To measure your own: python sandbox.py owner check --backend <your machine>')


# ---- live: their key, their machine -----------------------------------------------------------------------------------------------------------
def service():
    if not os.environ.get('IBM_QUANTUM_TOKEN', '').strip():
        sys.exit('Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)')
    from qiskit_ibm_runtime import QiskitRuntimeService as S
    sys.path.insert(0, ROOT); import pick_instance
    tok = os.environ['IBM_QUANTUM_TOKEN'].strip()
    base = S(channel='ibm_quantum_platform', token=tok)
    return S(channel='ibm_quantum_platform', token=tok, instance=pick_instance.crn(base))


def usable(be):
    t = be.target
    return [q for q in range(be.num_qubits) if t.instruction_supported('x', (q,)) and t.instruction_supported('measure', (q,))]


def circuits(be, qs):
    """C0, C1 and the five T1 delays, every usable qubit at once, physical (trivial) layout, optimization 0 (qia_cal/cal.py)."""
    from qiskit import QuantumCircuit, transpile
    dt_s = be.target.dt
    def base(x, d_us):
        qc = QuantumCircuit(be.num_qubits, len(qs))
        if x:
            for q in qs: qc.x(q)
        if d_us:
            n = int(round(d_us * 1e-6 / dt_s / 16)) * 16
            for q in qs: qc.delay(n, q, unit='dt')
        qc.barrier()
        for i, q in enumerate(qs): qc.measure(q, i)
        return qc
    raw = [base(False, 0), base(True, 0)] + [base(True, d) for d in DELAYS_US[1:]]
    return [transpile(c, be, optimization_level=0, initial_layout=list(range(be.num_qubits)), layout_method='trivial') for c in raw]


def pubs(be, qs):
    return [(c, None, SRO if i < 2 else ST1) for i, c in enumerate(circuits(be, qs))]


def utc16(s):
    """IBM's stamps carry their own offset. Kept in UTC with the zone, never bare."""
    s = str(s or '')
    try:
        t = dt.datetime.fromisoformat(s.replace(' ', 'T'))
        return t.astimezone(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ') if t.tzinfo else s[:16]
    except Exception: return s[:16]


def stated_of(props, qs):
    by = {}
    for i, ql in enumerate(props['qubits']):
        by[i] = {p['name']: (p['value'], utc16(p.get('date', ''))) for p in ql}
    out, dates = {}, {}
    for k, q in enumerate(qs):
        d = by.get(q, {}); g = lambda n: d.get(n, (None, ''))[0]
        out[k] = dict(p10=g('prob_meas1_prep0'), p01=g('prob_meas0_prep1'), t1=g('T1'))
        dates[k] = dict(p10=d.get('prob_meas1_prep0', (0, ''))[1], p01=d.get('prob_meas0_prep1', (0, ''))[1], t1=d.get('T1', (0, ''))[1])
    return out, dates


def counts_from_result(res, n_ro=SRO):
    """Per pub, the number of 1s per qubit (clbit k = qubit qs[k]). Returns c0 ones, c1 zeros, T1 ones (Q x 5)."""
    cnt = []
    for pub in res:
        reg = getattr(pub.data, 'c', None) or getattr(pub.data, 'meas'); bits = reg.get_bitstrings()
        a = np.array([[ch == '1' for ch in b[::-1]] for b in bits], bool)
        cnt.append(a.sum(0))
    return cnt[0], n_ro - cnt[1], np.column_stack(cnt[2:])


def result_file(backend, job_id, win, sent, qs, rows, kg, dates, c0, c1z, t1):
    """The result in the same shape as our published files (/qia/cal/<job>.json)."""
    per = []
    for k, q in enumerate(qs):
        x = dict(qubit=q, **rows[k]); x.update(ibm_p10_date=dates[k]['p10'], ibm_p01_date=dates[k]['p01'], ibm_t1_date=dates[k]['t1']); per.append(x)
    return dict(processor=backend, job_id=job_id, window_utc=win, record_used='IBM record in force at the job start (%s)' % win[0], record_saved_at_send=sent,
                rule='qia_cal PLAN.md with Notes 1 and 3', interval_levels=dict(readout=0.999, t1=0.9999, readout_method='exact Clopper-Pearson', t1_method='profile, delta chi2 <= 15.14'),
                dates_utc=True, shots=dict(readout=SRO, t1=ST1), delays_us=DELAYS_US, **kg, qubits=per,
                raw=dict(c0_ones=[int(v) for v in c0], c1_zeros=[int(v) for v in c1z], t1_ones=[[int(v) for v in row] for row in t1]))


def as_run(res):
    """A result file in the extract's shape, so `analyze` prints the same table as `history`."""
    Q = res['qubits']; r = dict(chip=res['processor'], job=res['job_id'], file=res['job_id'], window=res['window_utc'], known_good=res['known_good'], q=[v['qubit'] for v in Q])
    for k in ('p10', 'p01'):
        r[k] = [v[k] for v in Q]; r[k + '_lo'] = [v[k + '_ci'][0] for v in Q]; r[k + '_hi'] = [v[k + '_ci'][1] for v in Q]
        r['ibm_' + k] = [v['ibm_' + k] for v in Q]; r[k + '_date'] = [v.get('ibm_%s_date' % k) or None for v in Q]
        r[k + '_out'] = ''.join(LETTER[v[k + '_outcome']] for v in Q)
    r['t1_out'] = ''.join(LETTER[v['t1_outcome']] for v in Q)
    return r


def state_path(backend):
    d = os.path.join(ROOT, 'my_runs'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'owner_%s.json' % backend)


def remaining(svc):
    try:
        v = svc.usage().get('usage_remaining_seconds')
        return None if v is None else float(v)
    except Exception: return None


def record_age(props, qs):
    now = dt.datetime.now(dt.timezone.utc); _, dates = stated_of(props, qs); a = []
    for k in dates:
        h = [age_hours(now.strftime('%Y-%m-%dT%H:%MZ'), dates[k][x]) for x in ('p10', 'p01')]
        h = [x for x in h if x is not None]
        if h: a.append(max(h))
    return a


def check(backend):
    svc = service(); be = svc.backend(backend); qs = usable(be)
    props = be.properties(refresh=True).to_dict(); a = record_age(props, qs); left = remaining(svc)
    print('YOUR MACHINE %s: %d qubits, %d usable for the check (X and measure available).' % (backend, be.num_qubits, len(qs)))
    if a: print('IBM\'s current readout record: median age %s, oldest qubit %s.' % (hours(float(np.median(a))), hours(max(a))))
    print('The check: one job of 7 circuits (2 readout circuits at %d shots, 5 T1 delays at %d shots), every usable qubit at once.' % (SRO, ST1))
    print('IBM time: about %d s (estimate). IBM billed %d s for every one of the 2,171 calibration jobs in our record as of 27 Nov 2026. Queue time is not billed.' % (BILLED_S, BILLED_S))
    print('Your remaining IBM time: %s' % ('%.0f s' % left if left is not None else 'not reported by IBM for this instance'))
    if left is None: print('fire refuses when remaining time is not reported. If your plan covers about %d s, add --time-unknown-ok to fire.' % BILLED_S)
    if left is not None and left < BILLED_S + SPARE_S: print('Too little time left: fire refuses below %d s.' % (BILLED_S + SPARE_S))
    print('\nNothing was submitted. To run it: python sandbox.py owner fire --backend %s --yes' % backend)


def fire(backend, yes=False, time_unknown_ok=False):
    if not yes: sys.exit('This submits jobs on your IBM account and uses your IBM time. Run "check" first, then add --yes to fire.')
    svc = service(); be = svc.backend(backend); qs = usable(be); left = remaining(svc)
    if left is None and not time_unknown_ok:   # fail closed (27 Nov 2026, Sage): dedicated and contract instances may not report remaining time
        sys.exit('refusing: IBM does not report remaining time for this instance. If your plan has time for about %d s, add --time-unknown-ok to fire.' % BILLED_S)
    if left is None: print('WARNING: IBM reports no remaining time for this instance. Proceeding on --time-unknown-ok. The check uses about %d s.' % BILLED_S)
    if left is not None and left < BILLED_S + SPARE_S:
        sys.exit('refusing: %.0f s of IBM time left, the check needs about %d s plus %d s to spare' % (left, BILLED_S, SPARE_S))
    from qiskit_ibm_runtime import SamplerV2
    p = pubs(be, qs); props = be.properties(refresh=True).to_dict()
    sm = SamplerV2(mode=be); sm.options.environment.job_tags = [TAG]
    j = sm.run(p); jid = j.job_id(); now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print('Submitted on your account: job %s.' % jid)
    if os.path.exists(state_path(backend)):
        print('Note: this replaces the saved local state of your previous check (job %s). That job stays on your IBM account.' % json.load(open(state_path(backend))).get('job_id'))
    json.dump(dict(backend=backend, job_id=jid, sent=now, qubits=qs, props_at_send=props), open(state_path(backend), 'w'), default=str)
    print('%d qubits.\nNext: python sandbox.py owner analyze --backend %s   (run again every 30 s or so until it prints the table)'
          % (len(qs), backend))


def analyze(backend):
    sp = state_path(backend)
    if not os.path.exists(sp): sys.exit('no check fired on %s from this copy yet: run fire first' % backend)
    st = json.load(open(sp)); svc = service(); j = svc.job(st['job_id']); s = str(j.status())
    if not s.endswith('DONE'):
        if s.endswith('ERROR') or s.endswith('CANCELLED'): sys.exit('job %s ended %s on IBM\'s side; nothing to analyze' % (st['job_id'], s))
        print('job %s: %s. Not finished yet. Run analyze again in about 30 seconds.' % (st['job_id'], s)); return
    m = j.metrics(); ts = m.get('timestamps') or {}
    if not ts.get('running') or not ts.get('finished'): sys.exit('job %s: IBM returned no run timestamps yet. Run analyze again in a minute.' % st['job_id'])
    def _z(v):
        t = v if isinstance(v, dt.datetime) else dt.datetime.fromisoformat(str(v).replace('Z', '+00:00'))
        return (t.astimezone(dt.timezone.utc) if t.tzinfo else t).strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
    win = [_z(ts['running']), _z(ts['finished'])]
    try: run_props = svc.backend(backend).properties(datetime=dt.datetime.fromisoformat(win[0].replace('Z', '+00:00'))).to_dict()
    except Exception: run_props = None
    if not run_props:
        run_props = st['props_at_send']; print('NOTE: IBM did not return the record in force at the job start, so the record saved at send is used.')
    qs = st['qubits']; stated, dates = stated_of(run_props, qs)
    c0, c1z, t1 = counts_from_result(j.result())
    rows, kg = analyse(c0, c1z, t1, SRO, ST1, stated)
    res = result_file(backend, st['job_id'], win, st['sent'], qs, rows, kg, dates, c0, c1z, t1)
    out = os.path.join(ROOT, 'my_runs', 'res_%s.json' % st['job_id']); json.dump(res, open(out, 'w'))
    show_own(res, out)


def point_figure_count(qubits, n_ibm):
    """Readout figures that are not k/n_ibm: those are compared against IBM's point figure, without an interval of IBM's own."""
    return sum(1 for q in qubits for k in ('ibm_p10', 'ibm_p01')
               if q.get(k) is not None and np.isfinite(q[k]) and off_grid(q[k], n_ibm))


def point_figure_line(qubits, n_ibm):
    npf = point_figure_count(qubits, n_ibm)
    if not npf: return None
    return '%d of IBM\'s readout figures were compared as point figures (not a count out of %d shots), so they carry no interval of IBM\'s own.' % (npf, n_ibm)


def show_own(res, path=None):
    r = as_run(res)
    t = report({res['processor']: [r]}, 'YOUR MACHINE %s, job %s on your own account: our calibration check against IBM\'s record in force at the job start.'
               % (res['processor'], res['job_id']),
               'Known-good seat: %s (median measured/stated readout %.3f, T1 %.3f). IBM\'s readout figures read as counts out of %d shots%s.'
               % (res['known_good'], res['median_readout_measured_over_stated'], res['median_t1_measured_over_stated'], res['ibm_readout_shots'],
                  ' (inferred from the figures: a figure IBM counted on 1,024 shots would get too narrow an interval here and be called off more readily)' if res['ibm_readout_shots'] == 4096 else ''), one_run=True)
    s = t[res['processor']]
    print('RESULT: %d of %d qubits on %s read off IBM\'s readout record in this run, worse than stated (%.1f%%).' % (s['off'], s['readings'], res['processor'], pct(s['off'], s['readings'])))
    pl = point_figure_line(res['qubits'], res['ibm_readout_shots']) if res.get('known_good') != 'failed' else None
    if pl: print(pl)
    print('One run is a few minutes of one machine. The record IBM publishes changes, and so does the machine: a record of your own needs repeat runs.')
    if path: print('Saved: %s  (the same shape as our published files. The job ID opens the raw counts on your IBM account)' % path)
    print('Nothing goes to Heuremen.')


def run(cmd, family='heron_r2', backend=None, yes=False, time_unknown_ok=False):
    if cmd == 'history': return history(family)
    if not backend: sys.exit('name your machine: --backend <name>, for example --backend ibm_cleveland')
    if cmd == 'check': return check(backend)
    if cmd == 'fire': return fire(backend, yes=yes, time_unknown_ok=time_unknown_ok)
    if cmd == 'analyze': return analyze(backend)
