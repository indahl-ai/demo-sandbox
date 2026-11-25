"""tests/test_cli.py: every command for every demonstration reaches its code with the options the user typed. Submits nothing.

Run: python tests/test_cli.py     (also runs before every push from this copy: .git/hooks/pre-push)
24 Nov 2026: written after a cold run found that `fire --yes` never passed --yes to three demonstrations, so fire always refused."""
import os, sys, importlib, inspect
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.pop('IBM_QUANTUM_TOKEN', None)
import sandbox
fails = []


def run(argv):
    sys.argv = ['sandbox.py'] + argv
    try: sandbox.main(); return None
    except SystemExit as e: return str(e)


for demo in sandbox.OTHER:
    folder = os.path.join(HERE, demo); sys.path.insert(0, folder); cwd = os.getcwd()
    sys.modules.pop('sandbox_entry', None); os.chdir(folder); E = importlib.import_module('sandbox_entry'); os.chdir(cwd)
    seen = {}
    for name in ('replay', 'check', 'fire', 'fetch', 'analyze'):
        orig = getattr(E, name)
        def fake(*a, _n=name, **k): seen[_n] = (a, k)
        fake.__signature__ = inspect.signature(orig); setattr(E, name, fake)
    sandbox.summary.render = lambda *a, **k: ''
    run([demo, 'replay', '--chip', 'fez', '--run', '1'])
    if seen.get('replay', ((None, None), {}))[0][:2] != ('fez', 1): fails.append('%s replay got %s' % (demo, seen.get('replay')))
    for c in ('check', 'fetch', 'analyze'):
        run([demo, c, '--chip', 'marrakesh'])
        if c not in seen or seen[c][0][0] != 'marrakesh': fails.append('%s %s got %s' % (demo, c, seen.get(c)))
    seen.pop('fire', None); msg = run([demo, 'fire', '--chip', 'fez'])
    if 'fire' in seen and seen['fire'][1].get('yes') is not False: fails.append('%s fire without --yes reached the code with yes=%s' % (demo, seen['fire'][1].get('yes')))
    if 'fire' in seen and 'yes' not in seen['fire'][1] and not msg: fails.append('%s fire without --yes did not refuse' % demo)
    seen.pop('fire', None); run([demo, 'fire', '--chip', 'fez', '--yes'])
    if 'fire' not in seen: fails.append('%s fire --yes never reached the code' % demo)
    elif 'yes' in inspect.signature(getattr(E, 'fire')).parameters and seen['fire'][1].get('yes') is not True: fails.append('%s fire --yes arrived as yes=%s' % (demo, seen['fire'][1].get('yes')))
    sys.path.remove(folder)

# finance: fire must refuse without --yes before touching IBM, and must ask for a key with --yes
m = run(['finance', 'fire', '--chip', 'fez'])
if not m or 'add --yes' not in m: fails.append('finance fire without --yes: %r' % m)
m = run(['finance', 'fire', '--chip', 'fez', '--yes'])
if not m or 'IBM_QUANTUM_TOKEN' not in m: fails.append('finance fire --yes without a key: %r' % m)

# 25 Nov: the finance RESULT box once printed "7.9974 ... 12.7% off" (the percent was the average error, the number beside it the averaged
# price, 3.1% off). Every percent in the box must sit beside the error it is a share of, and equal error / exact price.
import re, importlib as _il
_S = _il.import_module('summary')
_t = """exact price 8.2543
  N no fade (client): price 7.9974 (99% interval 7.4620 to 8.5329), mean abs error 1.0457; pooled
  O fade at our measured CZ error: price 8.2827 (99% interval 8.2455 to 8.3200), mean abs error 0.0674; pooled
  OUTCOME ours minus CLIENT N (headline; error difference, 99% interval -1.1785 to -0.7781; BELOW = ours better): BELOW"""
_box = '\n'.join(_S.finance(_t) or [])
for _err, _pct in re.findall(r'average error per copy ([\d.]+) \(([\d.]+)% of the price\)', _box) or [('x', 'x')]:
    if _err == 'x' or abs(100 * float(_err) / 8.2543 - float(_pct)) > 0.05: fails.append('finance box percent not error/exact: %r' % _box)
if '% off' in _box or '15.5x lower' not in _box: fails.append('finance box wording: %r' % _box)

print('\n'.join(fails) if fails else 'test_cli: every command reaches its code with the options typed (6 demonstrations)')
sys.exit(1 if fails else 0)
