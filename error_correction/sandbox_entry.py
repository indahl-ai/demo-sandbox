#!/usr/bin/env python3
"""sandbox_entry.py: the error correction demonstration (distance-3 surface-code memory, the client's own default compile vs ours).

  replay(chip, run)   re-analyzes the recorded hardware counts OFFLINE (no key, no network, no IBM time) and prints the default (client)
                      arm beside the calibrated arm, with 99% intervals and the outcome word.
  check(chip)         with YOUR IBM key: confirms the key, shows your remaining IBM time and what a live run would use. Submits nothing.
  fire(chip, yes)     runs it live on your own IBM account (refuses without yes=True, over the 200 s cap, or over your remaining time).
  fetch(chip)         collects the finished main job.
  analyze(chip)       the same comparison, on YOUR run.

Your key is read only from the IBM_QUANTUM_TOKEN environment variable, never printed, never written to disk. The recorded runs are read-only
references; replay works on a temporary copy. Your own runs are saved under ../my_runs/error_correction/<chip>/.
Command line:  python sandbox_entry.py replay|check|fire|fetch|analyze [--chip kingston|fez|marrakesh] [--run 1|2] [--yes]"""
import os, sys, io, re, shutil, tempfile, argparse, importlib, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDED = os.path.join(ROOT, "recorded_error_correction")
MY_RUNS = os.environ.get("SANDBOX_MY_RUNS", os.path.join(ROOT, "my_runs", "error_correction"))
CHIPS = {"kingston": "ibm_kingston", "fez": "ibm_fez", "marrakesh": "ibm_marrakesh"}
sys.path.insert(0, HERE)

# The code's own estimate for one live run on one chip: calibration job plus main job plus job overhead, then the code's x1.25 margin.
# (est in every recorded state file: main 96.0 s + calibration 18.6 s + 2 jobs x 4 s = 122.6 s; x1.25 = 153.25 s; cap 200 s.)
ESTIMATE_SECONDS = 153.25
ESTIMATE_SECONDS_BEFORE_MARGIN = 122.6
CAP_SECONDS = 200.0
# What IBM BILLED for the recorded jobs (job.usage() read back from IBM, read-only): calibration job + main job, per chip and run.
MEASURED_SECONDS = {
    "kingston":  {1: 16 + 74, 2: 16 + 74},
    "fez":       {1: 16 + 72, 2: 16 + 72},
    "marrakesh": {1: 16 + 77, 2: 16 + 77},
}
MEASURED_JOBS = {   # (calibration job id, main job id), billed seconds in MEASURED_SECONDS
    "kingston":  {1: ("dbv8hslarooc73fl3oi0", "dbv8i7larooc73fl3p60"), 2: ("dbvdumff8rhs73bnp2p0", "dbvdv16ps3fc73bkas80")},
    "fez":       {1: ("dbv8jb6ps3fc73bk0s7g", "dbv93b5arooc73fl4oqg"), 2: ("dbvdumf82tmc73dut1g0", "dbvdv1782tmc73dut20g")},
    "marrakesh": {1: ("dbv91off8rhs73bnfpog", "dbv923782tmc73dujj2g"), 2: ("dbve3amps3fc73bkb3q0", "dbve49larooc73fle85g")},
}


def _load(chip):
    if chip not in CHIPS: sys.exit("chip must be one of: " + ", ".join(CHIPS))
    os.environ["DEMO_BACKEND"] = CHIPS[chip]              # demo.py reads the chip name at import
    import demo as D
    return importlib.reload(D)


def _headline(text, chip, run):
    keep = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("logical error per round,") or s.startswith("OUTCOME OE minus CE, logical error per round") or s.startswith("ratio, client over ours"):
            keep.append(s)
    out = ["", "=" * 78, "HEADLINE, %s, %s: logical error per round (lower is better)" % (CHIPS[chip], ("recorded run %s" % run) if run else "your live run")]
    out += ["  " + k for k in keep]
    out.append("  CE is the client's own default compile; OE is the same compiled circuit with our measured correction. 99% intervals over 16 copies.")
    out.append("=" * 78)
    return "\n".join(out)


def _run_analyze(D):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): D.mode_analyze()
    return buf.getvalue()


def replay(chip="kingston", run=2):
    D = _load(chip)
    src = os.path.join(RECORDED, "run%d" % run, CHIPS[chip])
    if not os.path.exists(os.path.join(src, "state.json")): sys.exit("no recorded run: " + src)
    work = tempfile.mkdtemp(prefix="replay_")
    try:
        for n in ("state.json", "table.json", "main_bits.npz", "cal_bits.npz"): shutil.copyfile(os.path.join(src, n), os.path.join(work, n))
        D.RUN = work                                      # the recorded folder itself is never opened for writing
        print("REPLAY of recorded run %d on %s. Nothing is sent anywhere; no key is used." % (run, CHIPS[chip]))
        print("Recorded hardware counts (two jobs per chip: the calibration job and the main job); the decoder and the intervals are recomputed here.")
        if chip == "marrakesh" and run == 2:
            print("NOTE: marrakesh run 2 ran after IBM recalibrated; the client arm read worse that day, so its per-round size is not comparable to run 1.")
        text = _run_analyze(D)
        print(_headline(text, chip, run)); print(text)
        return text
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _live(chip):
    if not os.environ.get("IBM_QUANTUM_TOKEN", "").strip():
        sys.exit("Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)")
    D = _load(chip)
    mine = os.path.join(MY_RUNS, CHIPS[chip]); os.makedirs(mine, exist_ok=True)
    D.RUN = mine; D.RUN_BASE = mine
    D.PILOT_DIR = os.path.join(RECORDED, "pilot", CHIPS[chip])     # the recorded pilot for this chip stands in for the pilot gate
    return D


def check(chip="kingston"):
    D = _live(chip); os.environ["DEMO_FAKE"] = "0"; D.mode_dry()
    print("\nNothing was submitted. The code's estimate for a live run is %.0f s with margin (cap %.0f s); IBM billed %d to %d s for our recorded runs."
          % (ESTIMATE_SECONDS, CAP_SECONDS, min(min(v.values()) for v in MEASURED_SECONDS.values()), max(max(v.values()) for v in MEASURED_SECONDS.values())))
    print("To run it live: python sandbox.py error_correction fire --chip %s --yes" % chip)


def fire(chip="kingston", yes=False):
    if not yes: sys.exit('This submits jobs on your IBM account and uses your IBM time. Run "check" first, then fire with yes=True (--yes).')
    D = _live(chip); D.mode_fire()


def fetch(chip="kingston"):
    D = _live(chip); D.mode_fetch()


def analyze(chip="kingston"):
    D = _live(chip); text = _run_analyze(D); print(_headline(text, chip, None)); print(text); return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("replay", "check", "fire", "fetch", "analyze"))
    ap.add_argument("--chip", choices=sorted(CHIPS), default="kingston")
    ap.add_argument("--run", type=int, choices=(1, 2), default=2)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "replay": replay(a.chip, a.run)
    elif a.cmd == "check": check(a.chip)
    elif a.cmd == "fire": fire(a.chip, a.yes)
    elif a.cmd == "fetch": fetch(a.chip)
    else: analyze(a.chip)


if __name__ == "__main__":
    main()
