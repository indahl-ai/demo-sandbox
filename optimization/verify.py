"""verify.py: replays all six recorded runs and compares them with the published table. Prints a markdown table."""
import io, re, json, contextlib
import sandbox_entry as S
PUB = {  # chip: (run1 client, run1 ours, run2 client, run2 ours)  from the published replicate table
 'kingston': ('0.9691', '0.9836', '0.9689', '0.9834'), 'fez': ('0.9585', '0.9763', '0.9564', '0.9776'), 'marrakesh': ('0.9580', '0.9781', '0.9334', '0.9536')}
PUB_OS_RUN1 = {'kingston': '+0.0045', 'fez': '+0.0104', 'marrakesh': '+0.0107'}
rows, ok = [], True
for chip in ('kingston', 'fez', 'marrakesh'):
    for run in (1, 2):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): S.replay(chip, run)
        t = buf.getvalue()
        n = re.search(r'N raw \(client\):\s+([0-9.]+)', t).group(1); o = re.search(r'O fade from our measured CZ error \(job P\):\s+([0-9.]+)', t).group(1)
        os_ = re.search(r'O minus S: ([+-][0-9.]+).*?: (\w+)\n', t).group(1, 2); on = re.search(r'O minus N: ([+-][0-9.]+).*?: (\w+)\n', t).group(1, 2)
        pn, po = PUB[chip][0 if run == 1 else 2], PUB[chip][1 if run == 1 else 3]
        res = 'PASS' if (n, o) == (pn, po) else 'FAIL'
        ok &= res == 'PASS'
        rows.append((chip, run, pn, n, po, o, os_[0], os_[1], on[1], res))
        if run == 1:
            c = 'PASS' if os_[0] == PUB_OS_RUN1[chip] else 'FAIL'; ok &= c == 'PASS'
            rows.append((chip, '1 (O minus S)', PUB_OS_RUN1[chip], '-', '-', os_[0], '', '', '', c))
print('| chip | run | published client | replayed client | published ours | replayed ours | O-S | O-S word | O-N word | result |\n|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print('| ' + ' | '.join(str(x) for x in r) + ' |')
print('ALL PASS' if ok else 'SOME FAIL')
