"""verify_replay.py: runs replay for every chip and both runs and compares the printed numbers with the published page data (demos.json, entry 'ai').
Usage: python verify_replay.py path/to/demos.json     Writes nothing; prints a markdown table."""
import sys, re, io, json, contextlib
sys.dont_write_bytecode = True
import sandbox_entry as S
d = [e for e in json.load(open(sys.argv[1])) if e.get("slug") == "ai"][0] if isinstance(json.load(open(sys.argv[1])), list) else None
if d is None: d = [e for e in json.load(open(sys.argv[1]))["demos"] if e["slug"] == "ai"][0]
cols = {"kingston": 1, "fez": 2, "marrakesh": 3}
rows = []
def cap(chip, run):
    b = io.StringIO()
    with contextlib.redirect_stdout(b): S.replay(chip, run)
    return b.getvalue()
def fade_bill(txt):
    f, bl = {}, {}
    sec = None
    for ln in txt.splitlines():
        if ln.startswith("MEASURED FADE"): sec = "f"; continue
        if ln.startswith("THE SHOT BILL"): sec = "b"; continue
        if ln.startswith("READOUT"): sec = None
        if sec == "f":
            mm = re.match(r"\s+(\d+)\s+\d+\s+(\d\.\d{3}) \(", ln)
            if mm: f[mm.group(1)] = mm.group(2)
        if sec == "b":
            mm = re.match(r"\s+(\d+)\s+\S+\s+\S+\s+(\d+\.\d\d)x", ln)
            if mm: bl[mm.group(1)] = mm.group(2) + "x"
    return f, bl
def readout(txt):
    n = int(re.search(r"\s(\d+) of the \d+ layout qubits", txt).group(1))
    ex = {(int(a), b): (c, e) for a, b, c, e in re.findall(r"q(\d+) (P\(\d\|\d\)): today ([\d.]+) vs record ([\d.]+)", txt)}
    return n, ex
out = {(c, r): cap(c, r) for c in cols for r in (1, 2)}
def add(item, chip, run, pub, rep):
    rows.append((item, chip, run, pub, rep, "PASS" if pub == rep else "FAIL"))
for c in cols:
    for r in (1, 2):
        f, bl = fade_bill(out[(c, r)])
        if r == 1:
            for row in d["fade"]["rows"]: add("self-score %sq" % row[0], c, r, row[cols[c]], f[row[0]])
            for row in d["bill"]["rows"]: add("shot bill %sq" % row[0], c, r, row[cols[c]], bl[row[0]])
        rp = [x for x in d["replicate"]["rows"] if x[0] == "ibm_" + c][0]
        add("self-score 40q (replicate table)", c, r, rp[1 + (r - 1)], f["40"])
        add("shot bill 40q (replicate table)", c, r, rp[3 + (r - 1)], bl["40"])
        n, ex = readout(out[(c, r)])
        rr = [x for x in d["readout"]["rows"] if x[0] == "ibm_" + c][0]
        add("readout qubits flagged", c, r, rr[1].split(" / ")[r - 1], str(n))
        mm = re.match(r"q(\d+): ([\d.]+) / ([\d.]+) vs (?:([\d.]+)|a figure stated as zero)", rr[2]); q = int(mm.group(1))
        vals = d_v = None
        # the page quotes the one-way P(1|0) of that qubit in each run, against the record figure
        pub_today = mm.group(1 + r + 0) if False else (mm.group(2) if r == 1 else mm.group(3)); pub_rec = mm.group(4) or "0.0000"
        got = ex.get((q, "P(1|0)"))
        add("readout example q%d P(1|0) today / record" % q, c, r, "%s / %s" % (pub_today, pub_rec), "%s / %s" % got if got else "not flagged")
print("| item | chip | run | published | replayed | result |\n|---|---|---|---|---|---|")
for x in rows: print("| %s | %s | %d | %s | %s | %s |" % x)
print("\nPASS %d of %d" % (sum(1 for x in rows if x[-1] == "PASS"), len(rows)))
