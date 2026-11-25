#!/usr/bin/env python3
"""sandbox.py: the Quantum Idle Audit demonstrations, run in a space that belongs to whoever starts them.

  python3 sandbox.py [DEMO] COMMAND [options]      DEMO is finance (the default), chemistry, optimization, simulation, error_correction or ai_kernels.


  python3 sandbox.py replay [--run 1|2] [--chip kingston|fez|marrakesh]
        Re-computes our published result from the recorded hardware runs. No key, no network, no IBM time.
        Prints the default (client) fit beside the fit that uses the measured gate error, with the exact price and the intervals.
  python3 sandbox.py check  [--chip ...]
        With YOUR IBM Quantum key: confirms the key works, shows your remaining IBM time and what a live run would use. Submits nothing.
  python3 sandbox.py fire   [--chip ...] --yes
        Runs the demonstration live on your own IBM account. IBM billed 140 to 143 s per chip run in the recorded runs; refuses with less than 200 s remaining.
  python3 sandbox.py fetch  [--chip ...]       collect the finished jobs
  python3 sandbox.py analyze [--chip ...]      the same comparison, on YOUR run

Your key is read from the IBM_QUANTUM_TOKEN environment variable, used only to talk to IBM, never printed, never written to disk, and never sent anywhere
else. Your runs are saved in ./my_runs/ on this machine. The recorded runs in ./recorded/ are read-only references and are never changed."""
import os, sys, shutil, argparse
import io, logging, contextlib
logging.getLogger('qiskit_ibm_runtime').setLevel(logging.ERROR)   # IBM's library prints long WARNING lines on every call; they read as errors to a newcomer
import summary


import re
_HINT = re.compile(r'(run|use|rerun|[Nn]ext:|then|To run it live:)')
_FLAG = re.compile(r'(?:python )?(?:sandbox_entry\.py |demo(?:_v\d)?\.py )?--?(fire|fetch|analyze|dry)\b((?: --pilot)?)(?: --chip \w+)?(?: --yes)?')


def _hints(text, demo, chip):
    """The demonstration scripts print next-step hints in their own flag form (--fetch, --analyze). Rewrite those hint lines, and only
    those, into the sandbox command a reader types (24 Nov 2026: a cold run followed a hint to a file that is not in the repository)."""
    if not demo: return text
    def one(line):
        if not _HINT.search(line) or '--' not in line and 'sandbox_entry.py' not in line: return line
        def sub(m):
            cmd = 'check' if m.group(1) == 'dry' else m.group(1)
            return 'python sandbox.py %s %s --chip %s%s%s' % (demo, cmd, chip, m.group(2), ' --yes' if cmd == 'fire' else '')
        return _FLAG.sub(sub, line)
    return '\n'.join(one(l) for l in text.split('\n')).replace(' (same --pilot flag)', '')


class _Hints(io.TextIOBase):
    def __init__(self, a, demo, chip): self.a, self.demo, self.chip, self.pend = a, demo, chip, ''
    def write(self, x):
        self.pend += x
        if '\n' in self.pend:
            done, self.pend = self.pend.rsplit('\n', 1)
            self.a.write(_hints(done, self.demo, self.chip) + '\n')
        return len(x)
    def flush(self):
        if self.pend: self.a.write(_hints(self.pend, self.demo, self.chip)); self.pend = ''
        self.a.flush()


class _Tee(io.TextIOBase):
    def __init__(self, a): self.a, self.buf = a, io.StringIO()
    def write(self, x): self.a.write(x); self.buf.write(x); return len(x)
    def flush(self): self.a.flush()


def _with_summary(demo, chip, what, fn, run=None):
    """Run fn with its output shown as usual, then print a RESULT box that restates the printed headline in plain words.
    For a replay (run given) the recorded job IDs and IBM options are printed from the recorded files just before the box."""
    t = _Tee(sys.stdout)
    with contextlib.redirect_stdout(t): r = fn()
    print(summary.render(demo, chip, what, t.buf.getvalue(), run=run))
    return r

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'finance'))
CHIPS = {'kingston': 'ibm_kingston', 'fez': 'ibm_fez', 'marrakesh': 'ibm_marrakesh'}


def load(chip):
    os.environ['DEMO_BACKEND'] = CHIPS[chip]            # demo.py reads the chip name at import
    import demo as D, demo_v2 as V2, demo_v3 as V3
    return D, V2, V3


def use_state(V2, V3, path):
    V3.STATE3 = path; V2.STATE2 = path                  # the scripts read and write their state through these two names


def cmd_replay(a):
    D, V2, V3 = load(a.chip)
    src = os.path.join(HERE, 'recorded', 'run%d' % a.run, 'state_v3_%s.json' % CHIPS[a.chip])
    if not os.path.exists(src): sys.exit('no recorded run: ' + src)
    work = os.path.join(HERE, '.replay_tmp'); os.makedirs(work, exist_ok=True)
    try:                                                 # 25 Nov 2026: a crashed replay left its copy of the recorded state behind; removed in every case now
        tmp = os.path.join(work, os.path.basename(src)); shutil.copyfile(src, tmp)   # the recorded file itself is never opened for writing
        use_state(V2, V3, tmp)
        class Offline:                                   # the analysis only needs the chip's shape to rebuild the circuits, so a fake chip stands in
            def backend(self, name): return D.fake_backend()
        V3.get_service = lambda: Offline()
        print('REPLAY of recorded run %d on %s. Nothing is sent anywhere; no key is used.' % (a.run, CHIPS[a.chip]))
        print('Recorded hardware counts and benchmarking data; the circuits are rebuilt on a stand-in chip only to count depth and two-qubit gates.')
        if a.chip == 'marrakesh': print('NOTE: marrakesh is our layout-fault control. Its fade scale sits on the edge of the search range, so its improvement is not a calibration result.')
        V3.mode_analyze()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def live(a):
    if not os.environ.get('IBM_QUANTUM_TOKEN', '').strip():
        sys.exit('Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)')
    D, V2, V3 = load(a.chip)
    mine = os.path.join(HERE, 'my_runs'); os.makedirs(mine, exist_ok=True)
    use_state(V2, V3, os.path.join(mine, 'state_v3_%s.json' % CHIPS[a.chip]))
    return D, V2, V3


def cmd_check(a):
    D, V2, V3 = live(a); V3.mode_dry()
    print('\nNothing was submitted. To run it live: python3 sandbox.py fire --chip %s --yes' % a.chip)


def cmd_fire(a):
    if not a.yes: sys.exit('This submits jobs on your IBM account and uses your IBM time. Run "check" first, then add --yes to fire.')
    D, V2, V3 = live(a); V3.mode_fire()


def cmd_fetch(a):
    D, V2, V3 = live(a); V2.mode_fetch()


def cmd_analyze(a):
    D, V2, V3 = live(a); V3.mode_analyze()


OTHER = ('chemistry', 'optimization', 'simulation', 'error_correction', 'ai_kernels')


def other_demo(demo, argv):
    """Every demonstration other than finance ships its own sandbox_entry module (replay, check, fire, fetch, analyze)."""
    ap = argparse.ArgumentParser(prog='sandbox.py ' + demo)
    ap.add_argument('cmd', choices=['replay', 'check', 'fire', 'fetch', 'analyze'])
    ap.add_argument('--chip', choices=sorted(CHIPS), default='kingston'); ap.add_argument('--run', type=int, choices=(1, 2), default=2)
    ap.add_argument('--yes', action='store_true')
    ap.add_argument('--pilot', action='store_true', help='optimization only: the short pilot run the code asks for when the day\'s power ratio is under 3')
    a = ap.parse_args(argv)
    folder = os.path.join(HERE, demo)
    if not os.path.isdir(folder): sys.exit('this demonstration is not in this copy: ' + demo)
    sys.path.insert(0, folder); os.chdir(folder)
    import sandbox_entry as E
    if a.cmd == 'fire' and not a.yes: sys.exit('This submits jobs on your IBM account and uses your IBM time. Run "check" first, then add --yes to fire.')
    if a.cmd == 'replay': return _with_summary(demo, a.chip, 'recorded run %d (replay)' % a.run, lambda: E.replay(a.chip, a.run), run=a.run)
    if a.cmd == 'analyze' and not a.pilot: return _with_summary(demo, a.chip, 'your live run', lambda: E.analyze(a.chip))
    if a.pilot:
        if demo != 'optimization' or a.cmd not in ('fire', 'fetch', 'analyze'): sys.exit('--pilot applies to optimization fire, fetch and analyze only')
        return getattr(E, a.cmd)(a.chip, pilot=True)
    fn = getattr(E, a.cmd)
    import inspect
    if a.cmd == 'fire' and 'yes' in inspect.signature(fn).parameters: return fn(a.chip, yes=a.yes)   # 24 Nov 2026: --yes was never passed on, so fire refused for ai_kernels, simulation and error_correction
    return fn(a.chip)


def chip_of(argv):
    """The chip as argparse reads it: `--chip fez` or `--chip=fez` (25 Nov 2026: the second form left the hints naming kingston)."""
    chip = 'kingston'
    for i, x in enumerate(argv):
        if x == '--chip' and i + 1 < len(argv): chip = argv[i + 1]
        elif x.startswith('--chip='): chip = x.split('=', 1)[1]
    return chip


def main():
    demo = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in OTHER + ('finance',) else ('finance' if len(sys.argv) > 1 and sys.argv[1] in ('replay', 'check', 'fire', 'fetch', 'analyze') else None)
    chip = chip_of(sys.argv)
    out = _Hints(sys.stdout, demo, chip)
    try:
        with contextlib.redirect_stdout(out): return _main()
    except SystemExit as e:
        out.flush()
        if isinstance(e.code, str): raise SystemExit(_hints(e.code, demo, chip))
        raise
    finally:
        out.flush()


def _main():
    if len(sys.argv) > 1 and sys.argv[1] in OTHER: return other_demo(sys.argv[1], sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'finance': sys.argv.pop(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('replay', cmd_replay), ('check', cmd_check), ('fire', cmd_fire), ('fetch', cmd_fetch), ('analyze', cmd_analyze)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument('--chip', choices=sorted(CHIPS), default='kingston')
        if name == 'replay': p.add_argument('--run', type=int, choices=(1, 2), default=2)
        if name == 'fire': p.add_argument('--yes', action='store_true')
    a = ap.parse_args()
    if a.cmd in ('replay', 'analyze'):
        return _with_summary('finance', a.chip, ('recorded run %d (replay)' % a.run) if a.cmd == 'replay' else 'your live run', lambda: a.fn(a),
                             run=a.run if a.cmd == 'replay' else None)
    a.fn(a)


if __name__ == '__main__':
    main()
