"""Offline tests for owner/owner_check.py: the counting rules, the port of our calibration analysis against one published run, and a dry
smoke test of the circuits on a stand-in chip (Aer). Nothing is sent to IBM.

The published run: ibm_marrakesh, job dc4t0smq4uuc73a9e63g (https://quantumidleaudit.com/qia/cal/dc4t0smq4uuc73a9e63g.json), copied to
tests/fixtures/. Its qubit 29 read P(1|0) 0.587 [0.5348, 0.6379] against IBM's 0.00048828125."""
import os, sys, json, subprocess
import numpy as np
import pytest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'owner')); import owner_check as O
FIX = os.path.join(HERE, 'tests', 'fixtures', 'res_dc4t0smq4uuc73a9e63g.json')


def run_row(p10_out, p01_out, t1_out='CCCC', ibm=None, lo=None, dates=None, start='2026-11-27T12:00:00Z', kg='passed', chip='ibm_x'):
    n = len(p10_out); d = dates or ['2026-11-27T10:00Z'] * n; ibm = ibm or [0.01] * n; lo = lo or [0.0] * n
    return dict(chip=chip, job='j', file='j', window=[start, start], known_good=kg, q=list(range(n)),
                p10=[0.5] * n, p10_lo=list(lo), p10_hi=[0.9] * n, ibm_p10=list(ibm), p10_date=d, p10_out=p10_out,
                p01=[0.5] * n, p01_lo=list(lo), p01_hi=[0.9] * n, ibm_p01=list(ibm), p01_date=d, p01_out=p01_out, t1_out=t1_out)


# ---- counting rules -----------------------------------------------------------------------------------------------------------------------------
def test_readout_off_is_either_quantity_below():
    assert O.readout_off('B', 'C') and O.readout_off('C', 'B') and O.readout_off('B', 'B')
    assert not O.readout_off('A', 'A') and not O.readout_off('C', 'T')   # ABOVE = IBM states MORE error than measured: never counted as worse


def test_tenfold_needs_an_off_reading_and_a_nonzero_figure():
    assert O.tenfold(0.10, 0.01, 'B') and not O.tenfold(0.099, 0.01, 'B')
    assert not O.tenfold(0.5, 0.01, 'C')          # a low end at 50x that is not a call does not count
    assert not O.tenfold(0.5, 0.0, 'B')           # IBM's 0 has no ratio; counted apart
    assert not O.tenfold(0.5, None, 'B')


def test_age_and_excess():
    assert O.age_hours('2026-11-27T12:00:00Z', '2026-11-27T10:30Z') == 1.5
    assert O.age_hours('2026-11-27T12:00:00Z', None) is None
    assert abs(O.excess(0.5348, 0.00048828125) - 0.53431171875) < 1e-12


def test_t1_counted_apart_both_ways():
    assert O.t1_counts('AABCT') == (2, 1)


def test_summarize_counts_qubit_readings_once():
    r = run_row('BBCC', 'BCAC', t1_out='ABTC', ibm=(0.01, 0.0, 0.01, 0.01), lo=(0.2, 0.3, 0.0, 0.0))
    s = O.summarize([r, run_row('CCCC', 'CCCC')])
    assert s['runs'] == 2 and s['readings'] == 8
    assert s['off'] == 2                            # qubit 0 (both quantities off) counts once; qubit 2's ABOVE is not worse
    assert s['tenfold'] == 1 and s['zero_off'] == 1  # qubit 0 at 20x; qubit 1 off against IBM's 0
    assert sum(1 for v in s['per_qubit'].values() if v) == 2
    assert (s['t1_worse'], s['t1_better'], s['t1_called']) == (1, 1, 7)
    assert s['ages'] == [2.0] * 8
    assert [w[2] for w in s['worst']] == [1, 0, 0]  # ranked by low end minus IBM's figure: 0.3, then 0.19 twice


def test_ibm_figure_gets_its_own_interval():
    # 1 in 1,000 against IBM's 0 of 4,096 is not a call; a clear gap is
    _, lo, hi = O.clopper(np.array([1]), 1000)
    assert O.side_counts(0.0, lo[0], hi[0], 4096) == 'CONSISTENT'
    _, lo, hi = O.clopper(np.array([587]), 1000)
    assert O.side_counts(0.00048828125, lo[0], hi[0], 4096) == 'BELOW'
    assert O.ibm_readout_shots({0: dict(p10=3 / 1024, p01=5 / 1024)}) == 1024 and O.ibm_readout_shots({0: dict(p10=3 / 4096, p01=5 / 1024)}) == 4096


def test_heron_r2_map_covers_every_qubit_once():
    rows = O.heron_r2_rows(156); qs = [q for row in rows for _, q in row]
    assert sorted(qs) == list(range(156))
    assert O.heron_r2_rows(133) is None


# ---- the port reproduces a published run ------------------------------------------------------------------------------------------------------
def test_port_reproduces_published_marrakesh_run():
    pub = json.load(open(FIX)); raw = pub['raw']
    stated = {k: dict(p10=q['ibm_p10'], p01=q['ibm_p01'], t1=q['ibm_t1']) for k, q in enumerate(pub['qubits'])}
    rows, kg = O.analyse(np.array(raw['c0_ones']), np.array(raw['c1_zeros']), np.array(raw['t1_ones']), O.SRO, O.ST1, stated)
    q29 = rows[29]
    assert q29['p10'] == 0.587 and q29['p10_ci'] == [0.5348, 0.6379] and q29['ibm_p10'] == 0.00048828125 and q29['p10_outcome'] == 'BELOW'
    for r, q in zip(rows, pub['qubits']):
        for k in ('p10', 'p10_ci', 'p10_outcome', 'p01', 'p01_ci', 'p01_outcome', 't1', 't1_ci', 't1_outcome'):
            assert r[k] == q[k], (q['qubit'], k, r[k], q[k])
    assert kg == {k: pub[k] for k in kg}
    dates = {k: dict(p10=q['ibm_p10_date'], p01=q['ibm_p01_date'], t1=q['ibm_t1_date']) for k, q in enumerate(pub['qubits'])}
    res = O.result_file(pub['processor'], pub['job_id'], pub['window_utc'], pub['record_saved_at_send'], [q['qubit'] for q in pub['qubits']], rows, kg, dates,
                        np.array(raw['c0_ones']), np.array(raw['c1_zeros']), np.array(raw['t1_ones']))
    assert set(res) == set(pub) and res['qubits'] == pub['qubits'] and res['raw'] == pub['raw']


def test_analyze_table_on_published_run(capsys):
    O.show_own(json.load(open(FIX))); out = capsys.readouterr().out
    assert 'RESULT:' in out and 'ibm_marrakesh' in out and '0.5870 [0.5348, 0.6379]' in out and 'Nothing goes to Heuremen' in out


def test_history_runs_offline():
    env = dict(os.environ); env.pop('IBM_QUANTUM_TOKEN', None)
    r = subprocess.run([sys.executable, 'sandbox.py', 'owner', 'history', '--family', 'heron_r2'], cwd=HERE, capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr
    assert 'ibm_kingston: 400 runs, 62,400 qubit readings' in r.stdout and '2,198 of 62,400' in r.stdout
    assert 'distinct qubits that read off at least once: 156 of 156' in r.stdout   # every marrakesh qubit
    assert 'RESULT:' in r.stdout and 'drift' not in r.stdout.lower()


def test_fire_refuses_without_yes_and_without_key():
    env = dict(os.environ); env.pop('IBM_QUANTUM_TOKEN', None)
    r = subprocess.run([sys.executable, 'sandbox.py', 'owner', 'fire', '--backend', 'ibm_cleveland'], cwd=HERE, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0 and 'add --yes' in r.stderr
    r = subprocess.run([sys.executable, 'sandbox.py', 'owner', 'check', '--backend', 'ibm_cleveland'], cwd=HERE, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0 and 'IBM_QUANTUM_TOKEN' in r.stderr


# ---- dry smoke test of the circuits on a stand-in chip ----------------------------------------------------------------------------------------
def test_circuits_on_fake_kingston_with_aer():
    pytest.importorskip('qiskit_aer')
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    from qiskit_ibm_runtime import SamplerV2
    be = FakeKingston(); qs = O.usable(be)
    assert len(qs) == 156
    circs = O.circuits(be, qs)
    assert len(circs) == 7 and all(c.num_qubits == 156 for c in circs)
    assert [c.layout.initial_index_layout()[:5] for c in circs[:1]] == [[0, 1, 2, 3, 4]]   # trivial layout: virtual qubit k on physical k
    assert circs[2].count_ops().get('delay', 0) == 156 and 'delay' not in circs[1].count_ops()
    sub = qs[:6]                                     # Aer drops idle qubits, so six measured qubits simulate quickly
    res = SamplerV2(mode=be).run(O.pubs(be, sub)).result()
    c0, c1z, t1 = O.counts_from_result(res)
    assert c0.shape == (6,) and t1.shape == (6, 5)
    props = be.properties().to_dict(); stated, dates = O.stated_of(props, sub)
    rows, kg = O.analyse(c0, c1z, t1, O.SRO, O.ST1, stated)
    assert len(rows) == 6 and all(0 <= r['p10'] <= 1 for r in rows)
    assert all(r['t1_ci'][0] <= r['t1'] <= r['t1_ci'][1] for r in rows)


def test_check_and_fire_refusal_with_a_stand_in_account(monkeypatch, capsys):
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    class Svc:
        def __init__(self, left): self.left = left
        def backend(self, name): return FakeKingston()
        def usage(self): return {'usage_remaining_seconds': self.left}
    monkeypatch.setattr(O, 'service', lambda: Svc(10))
    O.check('ibm_cleveland'); out = capsys.readouterr().out
    assert '156 qubits' in out and 'about 4 s' in out and 'Nothing was submitted' in out and 'fire refuses below 34 s' in out
    with pytest.raises(SystemExit) as e: O.fire('ibm_cleveland', yes=True)
    assert 'refusing' in str(e.value)


def test_fire_refuses_when_remaining_time_unknown(monkeypatch):
    class B:
        num_qubits = 2
    class Svc:
        def backend(self, n): return B()
    monkeypatch.setattr(O, 'service', lambda: Svc())
    monkeypatch.setattr(O, 'usable', lambda be: [0, 1])
    monkeypatch.setattr(O, 'remaining', lambda svc: None)
    with pytest.raises(SystemExit) as e: O.fire('ibm_cleveland', yes=True)
    assert 'time-unknown-ok' in str(e.value)


def test_utc_handles_eastern_dst_and_offsets():
    sys.path.insert(0, os.path.join(HERE, 'tools')); import build_owner_history as B
    assert B.utc('2026-10-31 12:00:00') == '2026-10-31T16:00Z'
    assert B.utc('2026-11-02 12:00:00') == '2026-11-02T17:00Z'
    assert B.utc('2026-11-27 01:12:52-05:00') == '2026-11-27T06:12Z'
    assert B.utc('2026-11-27T06:12Z') == '2026-11-27T06:12Z'


def test_owner_fire_parses_time_unknown_ok():
    env = dict(os.environ); env.pop('IBM_QUANTUM_TOKEN', None)
    r = subprocess.run([sys.executable, 'sandbox.py', 'owner', 'fire', '--backend', 'ibm_cleveland', '--time-unknown-ok'], cwd=HERE, capture_output=True, text=True, timeout=120, env=env)
    assert 'unrecognized arguments' not in r.stderr and 'add --yes' in r.stderr


def test_point_figure_fallback_is_counted():
    qs = [dict(ibm_p10=0.0012345, ibm_p01=10 / 4096), dict(ibm_p10=3 / 4096, ibm_p01=None)]
    assert O.point_figure_count(qs, 4096) == 1
    assert O.side_counts(0.0012345, 0.01, 0.02) == 'BELOW'   # point figure below our interval: called, with no interval for IBM
    assert O.off_grid(0.0012345, 4096) and not O.off_grid(10 / 4096, 4096)
    assert O.point_figure_line(qs, 4096) == "1 of IBM's readout figures were compared as point figures (not a count out of 4096 shots), so they carry no interval of IBM's own."
    assert O.point_figure_line([dict(ibm_p10=3 / 4096, ibm_p01=5 / 4096)], 4096) is None


def test_nighthawk_history_runs_offline():
    env = dict(os.environ); env.pop('IBM_QUANTUM_TOKEN', None)
    r = subprocess.run([sys.executable, 'sandbox.py', 'owner', 'history', '--family', 'nighthawk_r2'], cwd=HERE, capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr
    assert 'ibm_phoenix:' in r.stdout and 'ibm_chicago:' in r.stdout and 'RESULT:' in r.stdout
    assert 'not the chip\'s layout' in r.stdout and 'laid out as on the chip' not in r.stdout
