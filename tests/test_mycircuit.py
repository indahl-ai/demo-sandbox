"""Offline tests for my_circuit/check_circuit.py: plan on a stand-in chip, and the RB fit recovers a known gate error from synthetic counts."""
import os, sys, subprocess, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'my_circuit')); import check_circuit as C

def test_plan_offline():
    r = subprocess.run([sys.executable, 'sandbox.py', 'mycircuit', 'plan', 'my_circuit/examples/ghz_8.qasm', '--fake', 'kingston'], cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert 'Nothing was submitted' in r.stdout and '7 two-qubit gates' in r.stdout

def test_rb_fit_recovers_error():
    rng = np.random.default_rng(1); true = 0.004; r = 1 - true / 0.75; meta, counts = [], []
    for n in (2, 8, 16, 32, 6, 14, 30, 60):
        p = 0.25 + 0.7 * r ** n; k = rng.binomial(300, p); counts.append({'00': int(k), '11': 300 - int(k)}); meta.append(['3-4', 0, n])
    got = C.rb_fit(counts, meta)['3-4']
    assert abs(got - true) / true < 0.35, got

def test_analyze_prints_result(capsys):
    meta = [['0-1', 0, n] for n in (2, 8, 16, 32)]; r = 1 - 0.003 / 0.75
    st = dict(backend='ibm_test', fired='2026-11-27T00:00:00Z', rb_job='j1', ro_job='j2', meta=meta, qubits=[0, 1], pairs={'0-1': 3},
              stated2={'0-1': 0.0015}, stated_ro={'0': {'p01': 0.001, 'p10': 0.01}, '1': {'p01': 0.001, 'p10': 0.01}}, stated_when=None,
              rb_counts=[{'00': int(300 * (0.25 + 0.7 * r ** n)), '11': 300 - int(300 * (0.25 + 0.7 * r ** n))} for n in (2, 8, 16, 32)],
              ro_counts=[{'00': 1990, '01': 10}, {'11': 1900, '10': 100}])
    C.analyze(st); out = capsys.readouterr().out
    assert 'WORSE than stated' in out and 'RESULT:' in out and 'does not predict exactly' in out
