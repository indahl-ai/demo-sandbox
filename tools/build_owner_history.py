"""tools/build_owner_history.py: writes owner/recorded/heron_r2.jsonl.gz, the read-only extract `sandbox.py owner history` replays.

Source: our published calibration result files, one per job, as served at https://quantumidleaudit.com/qia/cal/<file>.json (file = job ID,
or job ID + "n3" for a run re-read under PLAN Note 3). Point it at a folder holding those files, named res_<file>.json:

    python tools/build_owner_history.py /path/to/folder

What it keeps, per run: job ID, the published file name, the run window, the shot count behind IBM's readout figures, the run's known-good result,
and per qubit: P(1|0) and P(0|1) with their 99.9% intervals, IBM's figure and its date, the published outcome, and the published T1 outcome.
Rules applied here, and only here:
  1. A job re-read under Note 3 (file <job>n3) replaces its first reading. The first reading's intervals for IBM's figure were drawn too narrow.
  2. IBM's dates in files written before 18 Nov 2026 carry no zone. They are IBM's US Eastern stamps (the saved records show -04:00 / -05:00)
     and are converted to UTC here (EDT until 1 Nov 2026 06:00 UTC, EST after).
Nothing else is changed. Numbers are copied as published."""
import os, sys, json, glob, gzip, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY_CHIPS = {'heron_r2': ('ibm_fez', 'ibm_kingston', 'ibm_marrakesh'), 'nighthawk_r2': ('ibm_phoenix', 'ibm_chicago')}
CHIPS = FAMILY_CHIPS['heron_r2']
OUT = {'ABOVE': 'A', 'BELOW': 'B', 'CONSISTENT': 'C', 'CANNOT_TELL': 'T'}
DST_END = dt.datetime(2026, 11, 1, 6, 0, tzinfo=dt.timezone.utc)


def utc(s):
    """IBM's date as UTC 'YYYY-MM-DDTHH:MMZ'. Bare stamps (pre 18 Nov 2026 files) are US Eastern."""
    if not s: return None
    if s.endswith('Z'): return s
    t = dt.datetime.fromisoformat(s.replace(' ', 'T'))
    if t.tzinfo: return t.astimezone(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    u = (t + dt.timedelta(hours=4)).replace(tzinfo=dt.timezone.utc)
    if u >= DST_END: u += dt.timedelta(hours=1)
    return u.strftime('%Y-%m-%dT%H:%MZ')


def main(src, family='heron_r2'):
    chips = FAMILY_CHIPS[family]
    files = sorted(glob.glob(os.path.join(src, 'res_*.json')))
    runs = {}
    for p in files:
        x = json.load(open(p))
        if x.get('processor') not in chips: continue
        name = os.path.basename(p)[4:-5]; n3 = name.endswith('n3')
        if x['job_id'] in runs and not n3: continue          # rule 1: the Note 3 re-read wins
        runs[x['job_id']] = (name, x)
    lines = []
    for jid, (name, x) in sorted(runs.items(), key=lambda kv: (kv[1][1]['processor'], kv[1][1]['window_utc'][0])):
        Q = x['qubits']
        r = dict(chip=x['processor'], job=jid, file=name, window=x['window_utc'], n_ibm=x.get('ibm_readout_shots'), known_good=x.get('known_good'),
                 q=[v['qubit'] for v in Q])
        for k in ('p10', 'p01'):
            r[k] = [v[k] for v in Q]; r[k + '_lo'] = [v[k + '_ci'][0] for v in Q]; r[k + '_hi'] = [v[k + '_ci'][1] for v in Q]
            r['ibm_' + k] = [v['ibm_' + k] for v in Q]; r[k + '_date'] = [utc(v.get('ibm_%s_date' % k)) for v in Q]
            r[k + '_out'] = ''.join(OUT[v[k + '_outcome']] for v in Q)
        r['t1_out'] = ''.join(OUT[v['t1_outcome']] for v in Q)
        lines.append(json.dumps(r, separators=(',', ':')))
    d = os.path.join(ROOT, 'owner', 'recorded'); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, '%s.jsonl.gz' % family)
    with gzip.GzipFile(out, 'wb', mtime=0) as f: f.write(('\n'.join(lines) + '\n').encode())
    print('wrote %s: %d runs, %d bytes' % (out, len(lines), os.path.getsize(out)))


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3): sys.exit(__doc__)
    main(sys.argv[1], *(sys.argv[2:3]))
