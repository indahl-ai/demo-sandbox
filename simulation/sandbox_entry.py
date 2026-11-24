"""sandbox_entry.py: the simulation demonstration (Trotter dynamics of a 10-qubit Ising chain on IBM hardware), as a module.

  replay(chip, run)  re-analyzes the recorded hardware data. No key, no network, no IBM time.
  check(chip)        with YOUR IBM_QUANTUM_TOKEN: shows your remaining IBM time and what a live run would use. Submits nothing.
  fire(chip)         submits the two jobs (job P, then the main job) on YOUR account. Refuses over the estimate cap or your remaining time.
  fetch(chip)        collects the finished jobs into ./my_runs/<chip>/
  analyze(chip)      the same comparison on YOUR run (offline; the chip's shape is rebuilt on a stand-in chip)

The key is read from the IBM_QUANTUM_TOKEN environment variable only, never printed, never written to a file."""
import os, sys, io, json, shutil, tempfile, importlib, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDED = os.path.join(os.path.dirname(HERE), 'recorded_simulation')
MINE = os.path.join(HERE, 'my_runs')
CHIPS = {'kingston': 'ibm_kingston', 'fez': 'ibm_fez', 'marrakesh': 'ibm_marrakesh'}

# The code's own estimate of QPU seconds for one live run (job P + main job), as `check` and `fire` compute it (cap: 200 s).
ESTIMATE_SECONDS = {'kingston': 97, 'fez': 97, 'marrakesh': 97}     # 96.6 to 96.8 s computed by the script's estimate functions
# What IBM billed for the recorded jobs (job.usage(), job P + main job), from recorded_simulation/usage.json.
MEASURED_SECONDS = {'run1': {'kingston': 42, 'fez': 42, 'marrakesh': 43}, 'run2': {'kingston': 42, 'fez': 42, 'marrakesh': 43}}


def _load(chip):
    """Import the demo scripts fresh for one chip (the script reads the chip name at import)."""
    if chip not in CHIPS: sys.exit('chip must be one of %s' % sorted(CHIPS))
    os.environ['DEMO_BACKEND'] = CHIPS[chip]
    for m in ('demo_v3', 'demo_v2', 'demo'): sys.modules.pop(m, None)
    if HERE not in sys.path: sys.path.insert(0, HERE)
    V = importlib.import_module('demo_v2'); V3 = importlib.import_module('demo_v3')
    return V, V3


def _fake(chip):
    from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
    return {'kingston': FakeKingston, 'fez': FakeFez, 'marrakesh': FakeMarrakesh}[chip]()


def _offline(V, chip):
    V.live_setup = lambda: (None, _fake(chip), None)   # analysis needs only the chip's shape, to rebuild the circuit structure


def _point(V3, d):
    for k in V3.FILES: V3.FILES[k] = os.path.join(d, os.path.basename(V3.FILES[k]))


def _need_key():
    if not os.environ.get('IBM_QUANTUM_TOKEN', '').strip():
        sys.exit('Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)')


def _mine(chip):
    V, V3 = _load(chip); d = os.path.join(MINE, chip); os.makedirs(d, exist_ok=True); _point(V3, d)
    sc = os.path.join(RECORDED, 'pilot', 'scale_v3_%s.json' % chip)       # frozen fade exponent for chips without an earlier job
    if os.path.exists(sc) and not os.path.exists(V3.FILES['scale']): shutil.copyfile(sc, V3.FILES['scale'])
    return V, V3


def _summary(V3, raw, tro, fS, fO):
    nn = raw[:, :, 1]; ex = tro[:, 1]; C = raw.shape[0]
    eN = V3.mae_copies(nn, np.ones(V3.K), ex); eO = V3.mae_copies(nn, fO[0], ex); eS = V3.mae_copies(nn, fS[0], ex)
    mN, lN, hN = V3.interval(eN); mO, lO, hO = V3.interval(eO); d, ld, hd = V3.interval(eO - eN); w = V3.word(ld, hd)
    lines = ['Error against the exact answer (mean over steps 2 to %d), value and 99%% interval over %d copies:' % (V3.K, C),
             '  default (client, values read raw)       %.3f  [%.4f, %.4f]' % (mN, lN, hN),
             '  calibrated (ours, gate error measured)  %.3f  [%.4f, %.4f]   %.1fx closer' % (mO, lO, hO, mN / mO),
             '  ours minus client, paired: %+.4f  99%% interval [%+.4f, %+.4f]' % (d, ld, hd),
             '  outcome: %s  (BELOW = our calibrated run is closer to exact)' % w]
    return dict(client=mN, ours=mO, stated_fade=V3.interval(eS)[0], diff=d, diff_lo=ld, diff_hi=hd, outcome=w, text='\n'.join(lines))


def analyze(chip, run=None, quiet=False):
    """Analyze the run saved in ./my_runs/<chip>/ (or, with run, a recorded run from a private copy). Offline."""
    if run is None: V, V3 = _mine(chip)
    else:
        V, V3 = _load(chip); src = os.path.join(RECORDED, 'run%d' % run)
        tmp = tempfile.mkdtemp(prefix='replay_'); _point(V3, tmp)
        for f in ('jobs', 'main', 'pdata'): shutil.copyfile(os.path.join(src, '%s_v3_%s.json' % (f, chip)), V3.FILES[f])
        if chip != 'kingston': shutil.copyfile(os.path.join(RECORDED, 'pilot', 'scale_v3_%s.json' % chip), V3.FILES['scale'])   # recorded files are never opened for writing
    _offline(V, chip); box = {}; orig = V3.analyze3
    def tap(raw, tro, fS, fO, stamp, label=''):
        box.update(_summary(V3, raw, tro, fS, fO)); return orig(raw, tro, fS, fO, stamp, label)
    V3.analyze3 = tap
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): V3.run_analyze()
    if run is not None: shutil.rmtree(tmp, ignore_errors=True)
    box['full'] = buf.getvalue()
    if not quiet: print(box['full'])
    return box


def replay(chip='kingston', run=2, full=False):
    """Re-analyze a recorded run. No key, no network. Prints the headline, then (full=True) the whole analysis."""
    if not os.path.isdir(os.path.join(RECORDED, 'run%d' % run)): sys.exit('no recorded run %s' % run)
    print('REPLAY of recorded run %d on %s. Nothing is sent anywhere; no key is used.' % (run, CHIPS[chip]))
    print('Recorded hardware readings and benchmarking data; the circuits are rebuilt on a stand-in chip only to count gates per step.')
    if chip == 'marrakesh': print('NOTE: marrakesh reads a layout fault more than a calibrated fade; read its improvement with that in mind.')
    r = analyze(chip, run, quiet=True)
    print(r['text'])
    if full: print('\n' + r['full'])
    return r


def _estimate(V, V3, chip):
    be = _fake(chip); chain = V.pick_chain(be); st, pv = V.transpile_once(be, chain); T = float(st.estimate_duration(be.target))
    rbc, meta = V3.rb_circuits(be, chain); return V3.est_P(rbc) + V3.est_main(T)


def check(chip='kingston'):
    """Free: confirms the key, shows remaining IBM time and the estimate for a live run. Submits nothing."""
    _need_key(); V, V3 = _mine(chip)
    svc = V.get_service(); b = svc.backend(CHIPS[chip]); chain = V.pick_chain(b); st, pv = V.transpile_once(b, chain); T = float(st.estimate_duration(b.target))
    rbc, meta = V3.rb_circuits(b, chain); est = V3.est_P(rbc) + V3.est_main(T); rem = V.remaining(svc)
    print('backend %s, chain %s, record stamp %s' % (b.name, chain, V3.stamp_of(b)))
    print('estimated QPU seconds for a live run (job P + main job): %.0f   cap %d   your usage_remaining_seconds: %s' % (est, V3.MAX_SECONDS, rem))
    print('IBM billed 42 to 43 s for each recorded run on a chip; the estimate is the cautious one that fire checks against.')
    print('Nothing was submitted.')
    return est, rem


def fire(chip='kingston', yes=False):
    if not yes: sys.exit('This submits jobs on your IBM account and uses your IBM time. Run check first, then fire(chip, yes=True).')
    _need_key(); V, V3 = _mine(chip); V3.run_fire(False)


def fetch(chip='kingston'):
    _need_key(); V, V3 = _mine(chip); V3.run_fetch(False)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='python sandbox_entry.py replay|check|fire|fetch|analyze [--chip kingston|fez|marrakesh] [--run 1|2] [--full] [--yes]')
    ap.add_argument('cmd', choices=('replay', 'check', 'fire', 'fetch', 'analyze')); ap.add_argument('--chip', choices=sorted(CHIPS), default='kingston')
    ap.add_argument('--run', type=int, choices=(1, 2), default=2); ap.add_argument('--full', action='store_true'); ap.add_argument('--yes', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'replay': replay(a.chip, a.run, a.full)
    elif a.cmd == 'check': check(a.chip)
    elif a.cmd == 'fire': fire(a.chip, a.yes)
    elif a.cmd == 'fetch': fetch(a.chip)
    else: analyze(a.chip)
