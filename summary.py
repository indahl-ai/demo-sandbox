"""summary.py: a short RESULT box printed after a replay or an analysis.

It only re-reads lines the demonstration already printed above it (values, intervals, outcome words) and restates them in plain
words. It computes nothing new, and when a line it needs is missing it says so instead of guessing. The full output stays above."""
import json, os, re

W = 78
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = 'Default run (raw counts, no error correction: the client\'s settings):'


def _f(pat, text, n=1):
    m = re.search(pat, text, re.M)
    return m.groups() if m and n > 1 else (m.group(1) if m else None)


def _box(title, lines, above):
    """25 Nov 2026: the footer once said job IDs and settings were printed above for every demonstration, while four printed none.
    It now names only what the text above the box actually holds."""
    has = ['its interval']
    if re.search(r'\b[a-z0-9]{20}\b', above): has.append('job IDs')
    if 'twirling' in above: has.append('IBM options as read back')
    held = has[0] if len(has) == 1 else ', '.join(has[:-1]) + ' and ' + has[-1]
    out = ['', '=' * W, ' RESULT  ' + title, '-' * W]
    out += ['  ' + l for l in lines]
    out += ['-' * W, '  Every figure above this box, with %s, is the full record.' % held, '=' * W]
    return '\n'.join(out)


def _flat(d, pre=''):
    """{'dd': {'enable': False}} -> 'dynamical_decoupling.enable=False' (dd is the IBM short name for dynamical decoupling)."""
    def walk(d, pre):
        out = []
        for k, v in d.items():
            k = {'dd': 'dynamical_decoupling'}.get(k, k)
            out += walk(v, pre + k + '.') if isinstance(v, dict) else ['%s%s=%s' % (pre, k, v)]
        return out
    return ' '.join(walk(d, pre))


def recorded(demo, chip, run):
    """Job IDs and IBM options exactly as the recorded files hold them, for a replay. Prints nothing the files do not contain."""
    def j(*p): return json.load(open(os.path.join(HERE, *p)))
    n = 'ibm_' + chip
    try:
        if demo == 'finance':
            d = j('recorded', 'run%d' % run, 'state_v3_%s.json' % n); ob = d.get('options_readback') or {}
            jobs = [('job P', d.get('job_p'), _flat(ob['p']) if 'p' in ob else None), ('main job', d.get('job_main'), _flat(ob['main']) if 'main' in ob else None)]
        elif demo == 'chemistry':
            d = j('recorded_chemistry', 'run%d' % run, 'state_v3_%s.json' % n)
            jobs = [('job P', d.get('job_p'), d.get('p_options')), ('main job', d.get('job_main'), d.get('main_options'))]
        elif demo == 'simulation':
            d = j('recorded_simulation', 'run%d' % run, 'jobs_v3_%s.json' % chip)
            jobs = [('job P', d['P'].get('id'), d['P'].get('options')), ('main job', d['main'].get('id'), d['main'].get('options'))]
        elif demo == 'error_correction':
            d = j('recorded_error_correction', 'run%d' % run, n, 'state.json')
            jobs = [('calibration job', d.get('job_cal'), None), ('main job', d.get('job_main'), None)]
        elif demo == 'optimization':
            d = j('recorded_optimization', 'run%d' % run, chip, 'state.json')
            jobs = [('job P', d.get('job_p'), d.get('p_options')), ('main job', d.get('job_main'), d.get('main_options'))]
        elif demo == 'ai_kernels':
            d = j('recorded_ai_kernels', 'run%d' % run, 'state_v7_%s.json' % n)
            jobs = [('job', d.get('job'), d.get('options'))]
        else: return []
    except Exception: return []
    out = ['RECORDED JOBS, %s run %d (as saved at the run):' % (n, run)]
    for name, jid, opt in jobs:
        if not jid: continue
        out.append('  %s %s; IBM options read back: %s' % (name, jid, opt) if opt else '  %s %s; IBM options: not in the recorded file' % (name, jid))
    return out if len(out) > 1 else []


def finance(t):
    ex = _f(r'exact price ([\d.]+)', t)
    n = _f(r'N no fade \(client\): price ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\), mean abs error ([\d.]+)', t, 4)
    o = _f(r'O fade at our measured CZ error: price ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\), mean abs error ([\d.]+)', t, 4)
    w = _f(r'OUTCOME ours minus CLIENT N \(headline; error difference, 99% interval (\S+) to (\S+);.*\): (\w+)', t, 3)
    if not (ex and n and o and w): return None
    e = float(ex); pn = 100 * float(n[3]) / e; po = 100 * float(o[3]) / e
    return ['Option price. Exact answer: %s' % ex,
            # 25 Nov: the percent is the AVERAGE ERROR per copy as a share of the price (what the page shows), never the gap of the
            # averaged price: "7.9974, 12.7% off" read as wrong arithmetic (7.9974 is 3.1% from 8.2543). Both now say what they are.
            DEFAULT,
            '  average error per copy %s (%.1f%% of the price); averaged price %s (99%% interval %s to %s)' % (n[3], pn, n[0], n[1], n[2]),
            'Calibrated run (ours):',
            '  average error per copy %s (%.1f%% of the price); averaged price %s (99%% interval %s to %s)' % (o[3], po, o[0], o[1], o[2]),
            'Error %.1fx lower. Outcome against the default run: %s (BELOW = ours closer to the exact price),' % (float(n[3]) / float(o[3]), w[2]),
            '  99%% interval of the error difference %s to %s.' % (w[0], w[1])]


def chemistry(t):
    c = _f(r'^Client \(default layout, bits as read\):\s+([\d.]+) mHa\s+\(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    o = _f(r'^Ours \(calibrated with the measured gate error\):\s+([\d.]+) mHa\s+\(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    w = _f(r'^Outcome, ours against the client:\s+(\w+)\s+\((.*) lower\)', t, 2)
    if not (c and o and w): return None
    return ['Molecular energy error (lower is better), in thousandths of a hartree',
            DEFAULT, '  %s  (99%% interval %s to %s)' % c,
            'Calibrated run (ours):', '  %s  (99%% interval %s to %s)' % o,
            'Calibrated error is %s lower. Outcome against the default run: %s.' % (w[1], w[0])]


def optimization(t):
    n = _f(r'N raw \(client\):\s+([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    o = _f(r'O fade from our measured CZ error \(job P\):\s+([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    w = _f(r'OUTCOME O minus N: (\S+) \(99% interval (\S+) to ([^,]+),.*\): (\w+)', t, 4)
    if not (n and o and w): return None
    return ['Schedule quality: approximation ratio (1 = the noiseless answer; higher is better)',
            DEFAULT, '  %s  (99%% interval %s to %s), %.1f%% short' % (n + (100 * (1 - float(n[0])),)),
            'Calibrated run (ours):', '  %s  (99%% interval %s to %s), %.1f%% short' % (o + (100 * (1 - float(o[0])),)),
            'Outcome against the default run: %s (ABOVE = ours higher), difference %s,' % (w[3], w[0]),
            '  99%% interval %s to %s.' % (w[1], w[2])]


def simulation(t):
    if 'HEADLINE ours vs the CLIENT' in t:          # a live analysis
        n = _f(r'N raw \(client\)\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]', t, 3)
        o = _f(r'O fade from OUR measured errors\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]', t, 3)
        w = _f(r'OUTCOME O vs N .*?: (\w+)', t)
    else:                                            # a replay
        n = _f(r'default \(client, values read raw\)\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]', t, 3)
        o = _f(r'calibrated \(ours, gate error measured\)\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]', t, 3)
        w = _f(r'outcome: (\w+)', t)
    if not (n and o and w): return None
    return ['Simulation error against the exact answer (lower is better)',
            DEFAULT, '  %s  (99%% interval %s to %s)' % n,
            'Calibrated run (ours):', '  %s  (99%% interval %s to %s)' % o,
            'Outcome against the default run: %s (BELOW = ours closer to exact).' % w]


def error_correction(t):
    c = _f(r'logical error per round, CE \(client[^)]*\): ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    o = _f(r'logical error per round, OE \(ours[^:]*: ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    w = _f(r'OUTCOME OE minus CE, logical error per round: (\S+) \(99% interval (\S+) to (\S+)\): (\w+)', t, 4)
    r = _f(r'ratio, client over ours, per round: ([\d.]+x)', t)
    if not (c and o and w): return None
    return ['Logical qubit failures per round (lower is better)',
            DEFAULT, '  %s  (99%% interval %s to %s)' % c,
            'Calibrated layout (ours):', '  %s  (99%% interval %s to %s)' % o,
            ('Failures per round %s lower. ' % r if r else '') + 'Outcome: %s (BELOW = ours fails less often),' % w[3],
            '  99%% interval of the difference %s to %s.' % (w[1], w[2])]


def ai_kernels(t):
    rows = re.findall(r'^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+x)\s*$', t, re.M)
    fl = _f(r'(\d+) of the (\d+) layout qubits used at any size are flagged', t, 2)
    if not rows: return None
    q, s, a, x = rows[-1]
    out = ['Shot budget for a kernel entry, against a plan built on IBM\'s published figures',
           'At %s qubits the plan expects a score of %s; the chip delivered %s.' % (q, s, a),
           'Shot bill: %s the shots the published-figure plan budgets (%s).' % (x, ', '.join('%s qubits %s' % (r[0], r[3]) for r in rows))]
    if fl: out.append('Readout audit: %s of the %s qubits used read worse than IBM\'s record (list above).' % fl)
    return out


FUN = {'finance': finance, 'chemistry': chemistry, 'optimization': optimization, 'simulation': simulation,
       'error_correction': error_correction, 'ai_kernels': ai_kernels}


def render(demo, chip, what, text, run=None):
    """run: the recorded run number for a replay; its job IDs and IBM options are printed from the recorded files before the box."""
    try: lines = FUN[demo](text)
    except Exception: lines = None
    title = '%s, ibm_%s, %s' % (demo.replace('_', ' '), chip, what)
    if not lines: lines = ['(No summary: the lines this box reads were not found. Read the output above.)']
    rec = '\n'.join(recorded(demo, chip, run)) if run else ''
    return ('\n' + rec if rec else '') + _box(title, lines, text + rec)
