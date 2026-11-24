"""summary.py: a short RESULT box printed after a replay or an analysis.

It only re-reads lines the demonstration already printed above it (values, intervals, outcome words) and restates them in plain
words. It computes nothing new, and when a line it needs is missing it says so instead of guessing. The full output stays above."""
import re

W = 78


def _f(pat, text, n=1):
    m = re.search(pat, text, re.M)
    return m.groups() if m and n > 1 else (m.group(1) if m else None)


def _box(title, lines):
    out = ['', '=' * W, ' RESULT  ' + title, '-' * W]
    out += ['  ' + l for l in lines]
    out += ['-' * W, '  Every figure above this box, with its interval, job IDs and settings, is the full record.', '=' * W]
    return '\n'.join(out)


def finance(t):
    ex = _f(r'exact price ([\d.]+)', t)
    n = _f(r'N no fade \(client\): price ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\), mean abs error ([\d.]+)', t, 4)
    o = _f(r'O fade at our measured CZ error: price ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\), mean abs error ([\d.]+)', t, 4)
    w = _f(r'OUTCOME ours minus CLIENT N \(headline; error difference, 99% interval (\S+) to (\S+);.*\): (\w+)', t, 3)
    if not (ex and n and o and w): return None
    e = float(ex); pn = 100 * float(n[3]) / e; po = 100 * float(o[3]) / e
    return ['Option price. Exact answer: %s' % ex,
            'Default run (the client\'s settings):  %s  (99%% interval %s to %s), %.1f%% off' % (n[0], n[1], n[2], pn),
            'Calibrated run (ours):                %s  (99%% interval %s to %s), %.1f%% off' % (o[0], o[1], o[2], po),
            'Outcome against the default run: %s (BELOW = ours closer to the exact price),' % w[2],
            '  99%% interval of the error difference %s to %s.' % (w[0], w[1])]


def chemistry(t):
    c = _f(r'^Client \(default layout, bits as read\):\s+([\d.]+) mHa\s+\(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    o = _f(r'^Ours \(calibrated with the measured gate error\):\s+([\d.]+) mHa\s+\(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    w = _f(r'^Outcome, ours against the client:\s+(\w+)\s+\((.*) lower\)', t, 2)
    if not (c and o and w): return None
    return ['Molecular energy error (lower is better), in thousandths of a hartree',
            'Default run (the client\'s settings):  %s  (99%% interval %s to %s)' % c,
            'Calibrated run (ours):                %s  (99%% interval %s to %s)' % o,
            'Calibrated error is %s lower. Outcome against the default run: %s.' % (w[1], w[0])]


def optimization(t):
    n = _f(r'N raw \(client\):\s+([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    o = _f(r'O fade from our measured CZ error \(job P\):\s+([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    w = _f(r'OUTCOME O minus N: (\S+) \(99% interval (\S+) to ([^,]+),.*\): (\w+)', t, 4)
    if not (n and o and w): return None
    return ['Schedule quality: approximation ratio (1 = the noiseless answer; higher is better)',
            'Default run (the client\'s settings):  %s  (99%% interval %s to %s), %.1f%% short' % (n + (100 * (1 - float(n[0])),)),
            'Calibrated run (ours):                %s  (99%% interval %s to %s), %.1f%% short' % (o + (100 * (1 - float(o[0])),)),
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
            'Default run (the client\'s settings):  %s  (99%% interval %s to %s)' % n,
            'Calibrated run (ours):                %s  (99%% interval %s to %s)' % o,
            'Outcome against the default run: %s (BELOW = ours closer to exact).' % w]


def error_correction(t):
    c = _f(r'logical error per round, CE \(client[^)]*\): ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    o = _f(r'logical error per round, OE \(ours[^:]*: ([\d.]+) \(99% interval ([\d.]+) to ([\d.]+)\)', t, 3)
    w = _f(r'OUTCOME OE minus CE, logical error per round: (\S+) \(99% interval (\S+) to (\S+)\): (\w+)', t, 4)
    r = _f(r'ratio, client over ours, per round: ([\d.]+x)', t)
    if not (c and o and w): return None
    return ['Logical qubit failures per round (lower is better)',
            'Default compile (the client\'s settings): %s  (99%% interval %s to %s)' % c,
            'Calibrated layout (ours):                %s  (99%% interval %s to %s)' % o,
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


def render(demo, chip, what, text):
    try: lines = FUN[demo](text)
    except Exception: lines = None
    title = '%s, ibm_%s, %s' % (demo.replace('_', ' '), chip, what)
    if not lines: lines = ['(No summary: the lines this box reads were not found. Read the output above.)']
    return _box(title, lines)
