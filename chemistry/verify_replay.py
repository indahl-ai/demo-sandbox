"""verify_replay.py: replays every recorded chip and run and compares with the published table (demos.json). Writes STATUS_table.md next to this file (the table in STATUS.md came from it).
Usage: python verify_replay.py PATH_TO_demos.json"""
import os, sys, json
import sandbox_entry as S

pub = None
data = json.load(open(sys.argv[1])); L = data if isinstance(data, list) else data.get('demos', data)
pub = [e for e in L if isinstance(e, dict) and e.get('slug') == 'chemistry'][0]
chips1 = {r[0]: r for r in pub['chips']}; rep = {r[0]: r for r in pub['replicate']['rows']}
vs_note = {'kingston': 8.1, 'fez': 9.0, 'marrakesh': 13.6}      # run 1 readout fix alone, quoted in the page text
rows = []; allpass = True
def chk(label, chip, run, published, got, fmt='%.1f', exact=False):
    global allpass
    g = (fmt % got) if not isinstance(got, str) else got
    ok = (str(published) == g)
    allpass &= ok; rows.append((label, chip, run, published, g, 'PASS' if ok else 'FAIL'))
res = {}
for run in (1, 2):
    for chip in S.CHIPS:
        res[(run, chip)] = S.replay(chip, run, quiet=True)
for chip in S.CHIPS:
    r1, r2 = res[(1, chip)], res[(2, chip)]
    chk('client energy error, mHa', chip, 1, chips1[chip][1], r1['client']); chk('ours energy error, mHa', chip, 1, chips1[chip][2], r1['ours'])
    chk('ratio text', chip, 1, chips1[chip][3], r1['ratio_text'])
    chk('client energy error, mHa', chip, 2, rep[chip][1 + 2], r2['client']); chk('ours energy error, mHa', chip, 2, rep[chip][1 + 3], r2['ours'])
    chk('readout fix alone, mHa (page text)', chip, 1, vs_note[chip], r1['readout_only'])
for run in (1, 2):
    for chip in S.CHIPS:
        r = res[(run, chip)]
        chk('outcome ours vs client, energy ruler', chip, run, 'CONSISTENT' if (run, chip) == (1, 'kingston') else 'BELOW', r['outcome_vs_client'], fmt='%s')   # page text since 24 Nov 2026
        chk('outcome ours vs client, bits-right ruler', chip, run, 'BELOW', r['frac_outcome_vs_client'], fmt='%s')
        chk('outcome ours vs stated, energy', chip, run, 'CONSISTENT', r['outcome_vs_stated'], fmt='%s')
        gap = -r['ours_minus_stated_mHa']; ok = 1.0 <= round(gap) <= 5 or 0.5 <= gap <= 5.5
        allpass &= ok; rows.append(('ours below stated by 1 to 5 mHa (page text)', chip, run, '1 to 5', '%.1f' % gap, 'PASS' if ok else 'FAIL'))
outside = {(1, 'fez'), (1, 'marrakesh'), (2, 'kingston'), (2, 'fez'), (2, 'marrakesh')}   # page text since 24 Nov 2026: all three chips in run 2
for run in (1, 2):
    for chip in S.CHIPS:
        r = res[(run, chip)]; want = 'BELOW' if (run, chip) in outside else 'CONSISTENT'
        w = r['frac_outcome_vs_stated']; ok = (w == want)
        want = want if want == 'BELOW' else 'CONSISTENT'
        rows.append(('all bits right vs stated (page text)', chip, run, want, '%s (ours %.3f vs %.3f)' % (w, r['frac_ours'], r['frac_stated']), 'PASS' if ok else 'FAIL')); allpass &= ok
out = ['# Replay verification, chemistry demonstration', '', 'Run 1 = the first column pair of the published chips table (job ids in recorded_chemistry/run1); run 2 = the replicate columns (run2).', '',
       '| Quantity | Chip | Run | Published | Replayed | Result |', '|---|---|---|---|---|---|']
out += ['| %s | %s | %d | %s | %s | %s |' % r for r in rows]
out += ['', 'ALL PASS' if allpass else 'SOME FAIL: see rows marked FAIL']
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STATUS_table.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
