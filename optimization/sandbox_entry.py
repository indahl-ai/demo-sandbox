"""sandbox_entry.py: the logistics (QAOA max-cut) demonstration, entry points for the sandbox.

  replay(chip, run)  re-analyzes the recorded hardware counts offline: no key, no network, no IBM time
  check(chip)        needs your key; confirms it works and shows the estimate; submits nothing
  fire(chip)         needs your key; submits two jobs on YOUR IBM account (refuses over the cap or over your remaining time)
  fetch(chip)        collects your finished jobs
  analyze(chip)      the same comparison on YOUR run

The key is read only from the IBM_QUANTUM_TOKEN environment variable, never printed, never written to disk."""
import os, sys, json, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RECORDED = os.path.join(os.path.dirname(HERE), 'recorded_optimization')
MINE = os.path.join(os.path.dirname(HERE), 'my_runs_optimization')
CHIPS = {'kingston': 'ibm_kingston', 'fez': 'ibm_fez', 'marrakesh': 'ibm_marrakesh'}

# The code's own estimate (demo_v3.est_p + est_main on the full configuration, before its 1.25 margin) and what IBM billed
# for the recorded jobs (job.usage() on each recorded job id: job P + main job). Per chip, identical in run 1 and run 2.
ESTIMATE_SECONDS = {'kingston': None, 'fez': None, 'marrakesh': None}     # filled in below from the recorded colour layers
MEASURED_SECONDS = {
    'kingston':  {'run1': {'P': 14, 'main': 34, 'total': 48}, 'run2': {'P': 14, 'main': 34, 'total': 48}},
    'fez':       {'run1': {'P': 14, 'main': 33, 'total': 47}, 'run2': {'P': 14, 'main': 33, 'total': 47}},
    'marrakesh': {'run1': {'P': 15, 'main': 35, 'total': 50}, 'run2': {'P': 15, 'main': 35, 'total': 50}},
}


def _load(chip):
    os.environ['DEMO_BACKEND'] = CHIPS[chip]
    import demo as D, demo_v2 as V2, demo_v3 as V3
    for m in (D, V2, V3): m.BACKEND = CHIPS[chip]
    return D, V2, V3


def _est(chip):
    _, _, V3 = _load(chip)
    st = json.load(open(os.path.join(RECORDED, 'run2', chip, 'state.json')))
    return V3.est_p(len(st['cols']), V3.CFG['full']) + V3.est_main(V3.CFG['full'])


for _c in CHIPS:
    ESTIMATE_SECONDS[_c] = round(_est(_c), 1)


def _need_key():
    if not os.environ.get('IBM_QUANTUM_TOKEN', '').strip():
        sys.exit('Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)')


def replay(chip, run=2):
    D, V2, V3 = _load(chip)
    src = os.path.join(RECORDED, 'run%d' % run, chip)
    if not os.path.exists(os.path.join(src, 'state.json')): sys.exit('no recorded run: ' + src)
    work = tempfile.mkdtemp(prefix='replay_')
    try:
        for f in ('state.json', 'raw_main.npz', 'raw_P.npz'): shutil.copyfile(os.path.join(src, f), os.path.join(work, f))   # recorded files are never opened for writing
        V3.RUN = work
        print('REPLAY of recorded run %d on %s. Nothing is sent anywhere; no key is used.' % (run, CHIPS[chip]))
        print('Recorded hardware counts (main job and the randomized-benchmarking job); no chip stand-in is needed for this analysis.')
        print('Headline: approximation ratio, client (N raw) against ours (O, fade from our measured gate error); S uses IBM\'s stated gate error.')
        if chip == 'marrakesh': print('NOTE: marrakesh read about 0.025 lower in run 2 across the whole chip; the ordering held.')
        V3.analyze('')
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _live(chip):
    _need_key()
    D, V2, V3 = _load(chip)
    V3.RUN = os.path.join(MINE, 'run_v3_' + chip)
    os.makedirs(V3.RUN, exist_ok=True)
    return D, V2, V3


def check(chip):
    D, V2, V3 = _live(chip); V3.mode_dry()
    print('\nNothing was submitted. Estimate for this chip: about %.0f s (the code refuses over %d s after a x%.2f margin).' % (ESTIMATE_SECONDS[chip], V3.MAX_SECONDS, V3.MARGIN))


def fire(chip, pilot=False):
    D, V2, V3 = _live(chip); V3.mode_fire(pilot)


def fetch(chip, pilot=False):
    D, V2, V3 = _live(chip); V3.fetch('_pilot' if pilot else '')


def analyze(chip, pilot=False):
    D, V2, V3 = _live(chip); V3.analyze('_pilot' if pilot else '')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['replay', 'check', 'fire', 'fetch', 'analyze'])
    ap.add_argument('--chip', choices=sorted(CHIPS), default='kingston'); ap.add_argument('--run', type=int, choices=(1, 2), default=2)
    a = ap.parse_args()
    replay(a.chip, a.run) if a.cmd == 'replay' else globals()[a.cmd](a.chip)
