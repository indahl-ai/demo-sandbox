"""tools/make_notebook.py: writes QIA_demonstrations.ipynb, the notebook that runs in Google Colab and in a hosted sandbox.

The replay cell is saved with the real output of a finance replay (run here, at build time), so the result shows before anything runs.
Run from the repository root:  python tools/make_notebook.py"""
import subprocess, sys, os, nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = 'https://github.com/indahl-ai/demo-sandbox'
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# Quantum Idle Audit demonstrations

Six demonstrations, each the same circuit on the same IBM chip run two ways: the client's default settings, and calibrated with the gate error measured on the qubits the circuit uses.

**1. Replay** a recorded hardware result. No key, no IBM time. The finance replay below is already printed: select **Runtime > Run all** (Colab) or run the cells in order to recompute it here.
**2. Live run** on your own IBM Quantum account (optional), further down.

The code is the public repository [github.com/indahl-ai/demo-sandbox](%s). Nothing is sent to Heuremen. Heuremen is not affiliated with IBM.""" % REPO),
code("""# Setup: about one minute the first time. Downloads the public repository and installs four libraries (qiskit, qiskit-ibm-runtime, numpy, scipy).
import os, sys, subprocess
if not os.path.exists('sandbox.py'):
    if os.path.exists('/workspace/demo/sandbox.py'): os.chdir('/workspace/demo')
    else:
        if not os.path.exists('demo-sandbox'): subprocess.run(['git', 'clone', '-q', '%s'], check=True)
        os.chdir('demo-sandbox')
PY = os.path.abspath('.venv/bin/python') if os.path.exists('.venv/bin/python') else sys.executable
if PY == sys.executable: subprocess.run([PY, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', '-r', 'requirements.txt'], check=True)
def sandbox(*args):
    p = subprocess.Popen([PY, 'sandbox.py', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end='')
    p.wait()
print('Ready in', os.getcwd())""" % REPO),
md("""## Choose a demonstration
`finance`, `chemistry`, `optimization`, `simulation`, `error_correction` or `ai_kernels`. Chips: `kingston`, `fez`, `marrakesh`. Recorded runs: `1` or `2`."""),
code("""DEMO = 'finance'
CHIP = 'fez'
RUN = 1"""),
md("""## 1. Replay a recorded result (no key, no IBM time)
Re-analyzes the recorded hardware counts and prints the full output, then a RESULT box that restates the headline in plain words."""),
code("""sandbox(DEMO, 'replay', '--chip', CHIP, '--run', str(RUN))"""),
md("""## 2. Live run on your own IBM Quantum account (optional)
Set `LIVE = True` in the next cell and run it. Enter the API key when asked. It is held in this session only: not shown, not printed, not saved. If the key has more than one IBM instance, `check` lists them with the time left on each: put the name in `INSTANCE` and run the cell again."""),
code("""LIVE = False          # change to True to use your own IBM Quantum account
INSTANCE = ''          # only if asked, for example 'my-instance'
if LIVE:
    from getpass import getpass
    if not os.environ.get('IBM_QUANTUM_TOKEN'): os.environ['IBM_QUANTUM_TOKEN'] = getpass('IBM Quantum API key: ')
    if INSTANCE: os.environ['DEMO_INSTANCE'] = INSTANCE
    sandbox(DEMO, 'check', '--chip', CHIP)   # remaining IBM time and the estimate. Submits nothing.
else:
    print('Live run is off. Set LIVE = True in this cell and run it again to use your own IBM Quantum account.')"""),
md("""### Fire
Submits the jobs on your IBM account and uses your IBM time (see the estimate above). Change `False` to `True` to confirm."""),
code("""USE_MY_IBM_TIME = False
if LIVE and USE_MY_IBM_TIME: sandbox(DEMO, 'fire', '--chip', CHIP, '--yes')
else: print('Not submitted. Set LIVE = True above and USE_MY_IBM_TIME = True here to submit.')"""),
md("""### Collect and compare
Checks the jobs every 30 seconds and prints the comparison when they finish (usually a few minutes, longer if the chip has a queue)."""),
code("""import time
for i in range(60 if LIVE and USE_MY_IBM_TIME else 0):
    out = subprocess.run([PY, 'sandbox.py', DEMO, 'fetch', '--chip', CHIP], capture_output=True, text=True).stdout
    print(time.strftime('%H:%M:%S'), out.strip().splitlines()[-1] if out.strip() else '')
    res = subprocess.run([PY, 'sandbox.py', DEMO, 'analyze', '--chip', CHIP], capture_output=True, text=True)
    if ' RESULT ' in res.stdout and 'No summary' not in res.stdout:
        print(res.stdout); break
    time.sleep(30)
else:
    print('Nothing to collect yet.' if not (LIVE and USE_MY_IBM_TIME) else 'Not finished after 30 minutes. Run this cell again later.')"""),
md("""Delete the session when finished (Colab: **Runtime > Disconnect and delete runtime**). Everything in it, including your live run, is removed."""),
]

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'},
               'colab': {'provenance': [], 'name': 'QIA_demonstrations.ipynb'}}
# saved output for the replay cell: the real finance replay, run now
r = subprocess.run([sys.executable, 'sandbox.py', 'finance', 'replay', '--chip', 'fez', '--run', '1'], cwd=ROOT, capture_output=True, text=True)
if r.returncode != 0 or ' RESULT ' not in r.stdout: sys.exit('replay failed; notebook not written:\n' + r.stdout[-800:] + r.stderr[-800:])
nb.cells[5].outputs = [nbf.v4.new_output('stream', name='stdout', text=r.stdout)]
nb.cells[5].execution_count = 1
nbf.write(nb, os.path.join(ROOT, 'QIA_demonstrations.ipynb'))
print('wrote QIA_demonstrations.ipynb with the saved finance replay (%d lines)' % len(r.stdout.splitlines()))
