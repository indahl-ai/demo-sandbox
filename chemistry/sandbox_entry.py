"""sandbox_entry.py: the Quantum Idle Audit chemistry demonstration (H2 ground-state energy by phase estimation with mid-circuit readout).

  replay(chip, run)  re-analyzes the recorded hardware counts offline (no key, no network) and prints the page's headline lines
  check(chip)        needs YOUR IBM key in IBM_QUANTUM_TOKEN: confirms it works, shows your remaining time and the estimate; submits nothing
  fire(chip)         submits the two jobs on YOUR account (refuses over the cap or over your remaining time)
  fetch(chip)        collects the finished jobs
  analyze(chip)      the same comparison on YOUR run

The key is read from the IBM_QUANTUM_TOKEN environment variable only, never printed, never written to disk."""
import os, sys, json, shutil, io, contextlib, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDED = os.path.join(ROOT, 'recorded_chemistry')
MY_RUNS = os.path.join(ROOT, 'my_runs')
sys.path.insert(0, HERE)
CHIPS = {'kingston': 'ibm_kingston', 'fez': 'ibm_fez', 'marrakesh': 'ibm_marrakesh'}

# The code's own estimate of QPU seconds for one chip (written into every recorded state file as est_seconds).
ESTIMATE_SECONDS = 23.504
# What IBM billed (job.usage(), read once, see recorded_chemistry/billed_usage.json): 9 s readout/benchmarking job + 9 s main job.
MEASURED_SECONDS = {'run%d' % r: {c: 18 for c in CHIPS} for r in (1, 2)}


def _load(chip):
    if chip not in CHIPS: sys.exit('chip must be one of: ' + ', '.join(CHIPS))
    os.environ['DEMO_BACKEND'] = CHIPS[chip]            # demo.py reads the chip name at import
    for m in ('demo', 'demo_v2', 'demo_v3'): sys.modules.pop(m, None)
    import demo as D, demo_v3 as V3
    return D, V3


def _paths(V3, state, result):
    V3.paths = lambda: (state, result)                  # the scripts read their state and result files through this one function


def _vis_from_facts(facts, errs):
    """Per-phase-bit visibility, the same formula as demo_v3.step_vis, from the saved per-bit gate counts of the circuit as submitted."""
    out = []
    for k in range(5):
        x = 1.0
        for e, n in facts['cz_per_step'].get(str(k), {}).items():
            a, b = (int(t) for t in e.split('-')); x *= (1 - 4 * errs[(a, b)] / 3) ** n
        out.append(x)
    return out


def _tail(V3, D, st, r, vs, vo, flips, cal, meas, stated, act, mq, print_report=True):
    print('pair: stated / RB-measured / ratio: ' + ', '.join('%s %.5f / %.5f / %.2f' % (e, stated[e], meas[e], meas[e] / stated[e]) for e in sorted(stated)))
    print('per-bit visibility stated %s measured %s' % (['%.4f' % x for x in vs], ['%.4f' % x for x in vo]))
    print('readout, record vs job P (P(1|0), P(0|1)): ' + ', '.join('q%d record %.4f / P (%.4f,%.4f)' % (q, st['record'][str(q)], *cal[q]) for q in act))
    res = V3.arms(r['main'], flips, vs, vo)
    print(V3.report(res, 'HARDWARE. '))
    text, nums = _headline(res, V3); print(text)
    return nums


def _headline(res, V3):
    import numpy as np
    D = V3.v1; E0 = D.E0
    err = {a: np.array([abs(x[0] - E0) for x in res[a]]) for a in res}
    def iv(a): h = V3.hw(err[a]); return err[a].mean() * 1e3, max(err[a].mean() - h, 0) * 1e3, (err[a].mean() + h) * 1e3
    def out(other):
        d = err['O'] - err[other]; lo, hi = d.mean() - V3.hw(d), d.mean() + V3.hw(d); return V3.word(lo, hi, V3.EQUIV_E)
    n, o, s, ro = iv('N'), iv('O'), iv('S'), iv('RO')
    fr = {a: np.array([x[1] for x in res[a]]) for a in res}
    dF = fr['O'] - fr['S']; fword = V3.word(-(dF.mean() + V3.hw(dF)), -(dF.mean() - V3.hw(dF)), V3.EQUIV_F)
    ratio = round(n[0], 1) / round(o[0], 1);   # the page divides the two numbers it shows
    dN = fr['O'] - fr['N']; fwordN = V3.word(-(dN.mean() + V3.hw(dN)), -(dN.mean() - V3.hw(dN)), V3.EQUIV_F)
    ratio_txt = ('%.0fx' % ratio) if ratio >= 10 else ('%.1fx' % ratio)
    L = ['', '=== HEADLINE (as on the demonstration page): energy error in mHa, lower is better ===',
         'Client (default layout, bits as read):          %.1f mHa  (99%% interval %.1f to %.1f)' % n,
         'Ours (calibrated with the measured gate error):  %.1f mHa  (99%% interval %.1f to %.1f)' % o,
         'Outcome, ours against the client:               %s   (%s lower)' % (out('N'), ratio_txt),
         'Readout fix alone (no gate correction):          %.1f mHa  (99%% interval %.1f to %.1f)' % ro,
         'Correction from IBM stated gate error:           %.1f mHa  (99%% interval %.1f to %.1f)' % s,
         'Outcome, ours against the stated-error correction (energy): %s' % out('S'),
         'All five phase bits right, ours %.3f vs stated-error correction %.3f vs client %.3f; ours against stated: %s' % (fr['O'].mean(), fr['S'].mean(), fr['N'].mean(), fword)]
    return '\n'.join(L), {'client': n[0], 'ours': o[0], 'readout_only': ro[0], 'stated': s[0], 'outcome_vs_client': out('N'), 'outcome_vs_stated': out('S'), 'ratio_text': ratio_txt + ' lower', 'frac_outcome_vs_stated': fword, 'frac_outcome_vs_client': fwordN, 'ours_minus_stated_mHa': o[0] - s[0], 'frac_ours': fr['O'].mean(), 'frac_stated': fr['S'].mean()}


def replay(chip, run=2, quiet=False):
    """Offline re-analysis of a recorded run. Returns a dict of the headline numbers."""
    D, V3 = _load(chip)
    src_s = os.path.join(RECORDED, 'run%d' % run, 'state_v3_%s.json' % CHIPS[chip])
    src_r = os.path.join(RECORDED, 'run%d' % run, 'result_v3_%s.json' % CHIPS[chip])
    if not os.path.exists(src_s) or not os.path.exists(src_r): sys.exit('no recorded run: ' + src_s)
    st = json.load(open(src_s)); r = json.load(open(src_r))               # the recorded files are only read
    facts = json.load(open(os.path.join(RECORDED, 'circuit_facts.json')))['runs']['run%d_%s' % (run, chip)]
    act, mq, nrb = st['act'], st['mq'], st['n_rb']
    if facts['mq'] != mq or facts['act'] != act: sys.exit('recorded circuit facts do not match the recorded state')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet else sys.stdout):
        print('REPLAY of recorded run %d on %s. Nothing is sent anywhere; no key is used.' % (run, CHIPS[chip]))
        print('Recorded hardware counts, recorded gate-error benchmarking and readout data, and the recorded per-bit gate counts of the submitted circuit.')
        if chip == 'marrakesh': print('NOTE: marrakesh is our layout-fault control: the default layout puts the circuit on weak qubits. Its improvement is not a fade-calibration result.')
        meta = [(tuple(e), m, nz) for e, m, nz in st['meta']]; meas = V3.rb_fit(r['P'][:nrb], meta)
        stated = {tuple(int(x) for x in k.split('-')): v for k, v in st['stated'].items()}
        cal = D.readout_from_counts(r['P'][nrb], r['P'][nrb + 1], act, V3.P_SHOTS); flips = [cal[q] for q in mq]
        nums = _tail(V3, D, st, r, _vis_from_facts(facts, stated), _vis_from_facts(facts, meas), flips, cal, meas, stated, act, mq)
    return nums


def _live(chip):
    if not os.environ.get('IBM_QUANTUM_TOKEN', '').strip():
        sys.exit('Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)')
    D, V3 = _load(chip); os.makedirs(MY_RUNS, exist_ok=True)
    _paths(V3, os.path.join(MY_RUNS, 'state_v3_%s.json' % CHIPS[chip]), os.path.join(MY_RUNS, 'result_v3_%s.json' % CHIPS[chip]))
    return D, V3


def check(chip):
    D, V3 = _live(chip); V3.mode_dry()
    print('\nNothing was submitted. The code estimates about %.0f s for this demonstration; IBM billed %d s for each recorded chip run.' % (ESTIMATE_SECONDS, 18))


def fire(chip):
    D, V3 = _live(chip); V3.mode_fire()


def fetch(chip):
    D, V3 = _live(chip); V3.mode_fetch()


def analyze(chip):
    D, V3 = _live(chip); V3.mode_analyze()           # prints the full comparison; the headline lines follow
    st = V3.s_load(); r = json.load(open(V3.paths()[1])); act, mq, nrb = st['act'], st['mq'], st['n_rb']
    meta = [(tuple(e), m, nz) for e, m, nz in st['meta']]; meas = V3.rb_fit(r['P'][:nrb], meta)
    stated = {tuple(int(x) for x in k.split('-')): v for k, v in st['stated'].items()}
    cal = D.readout_from_counts(r['P'][nrb], r['P'][nrb + 1], act, V3.P_SHOTS); flips = [cal[q] for q in mq]
    tc, mq2, act2 = D.setup(D.get_service().backend(CHIPS[chip]))
    res = V3.arms(r['main'], flips, V3.step_vis(tc, stated), V3.step_vis(tc, meas)); print(_headline(res, V3)[0])
