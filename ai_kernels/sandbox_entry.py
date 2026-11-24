#!/usr/bin/env python3
"""sandbox_entry.py: the Quantum Idle Audit AI and machine learning demonstration (quantum kernels, shot bill at 40 qubits).

  replay(chip, run)   re-analyzes our recorded hardware counts. No key, no network, no IBM time. Prints the fade by size, the shot bill and the readout audit.
  check(chip)         with YOUR IBM key: confirms the key, shows your remaining IBM time and what a live run would use. Submits nothing.
  fire(chip, yes)     runs the audit live on your own IBM account (one job). Refuses unless yes=True and the estimate fits the cap and your remaining time.
  fetch(chip)         collects the finished job.
  analyze(chip)       the same tables, on YOUR run.

Your key is read only from the IBM_QUANTUM_TOKEN environment variable, used only to talk to IBM, never printed, never written to disk.
Your runs are saved in ./my_runs/. The recorded runs in ../recorded_ai_kernels/ are read-only references and are never changed.
Command line:  python sandbox_entry.py replay --run 1 --chip kingston     (also: check, fire --yes, fetch, analyze)"""
import os, sys, json, shutil, argparse, importlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RECORDED = os.path.join(os.path.dirname(HERE), "recorded_ai_kernels")
CHIPS = ("kingston", "fez", "marrakesh")


def _name(chip):
    c = str(chip).lower().replace("ibm_", "")
    if c not in CHIPS: sys.exit("chip must be one of: " + ", ".join(CHIPS))
    return c


def _load(chip):
    os.environ["DEMO_BACKEND"] = "ibm_" + _name(chip)         # demo.py reads the chip name at import
    import demo as m
    importlib.reload(m)
    import demo_v7 as v7
    importlib.reload(v7)
    return m, v7


def _recorded(chip, run):
    return os.path.join(RECORDED, "run%d" % int(run), "state_v7_ibm_%s.json" % _name(chip))


def _estimate():
    os.environ.setdefault("DEMO_BACKEND", "ibm_kingston")
    import demo as m, demo_v7 as v7
    return round(m.SEC_PER_SHOT_CIRCUIT * (len(v7.SIZES) * v7.NPTS * v7.COPIES * v7.SHOTS + 2 * v7.RO_SHOTS) + m.SEC_PER_JOB, 1)


def _measured():
    out = {}
    for c in CHIPS:
        out[c] = {}
        for r in (1, 2):
            p = _recorded(c, r)
            if os.path.exists(p): out[c][r] = json.load(open(p)).get("billed_qpu_seconds")
    return out


ESTIMATE_SECONDS = _estimate()      # the code's own estimate for one chip: seconds of IBM time (about 23 s); fire adds a 1.25 margin and refuses above a 60 s cap
MEASURED_SECONDS = _measured()      # QPU seconds IBM billed for each recorded job, per chip and run (read from IBM's job records)


def replay(chip="kingston", run=2):
    m, v7 = _load(chip)
    src = _recorded(chip, run)
    if not os.path.exists(src): sys.exit("no recorded run: " + src)
    work = os.path.join(HERE, ".replay_tmp"); os.makedirs(work, exist_ok=True)
    shutil.copyfile(src, os.path.join(work, "state.json"))    # the recorded file itself is never opened for writing
    m.RUN = work
    class Offline:                                           # the analysis only needs the chip's shape to rebuild the circuits, so a stand-in chip is used
        def backend(self, name): return m.fake_backend()
    m.get_service = lambda: Offline()
    print("REPLAY of recorded run %d on ibm_%s. Nothing is sent anywhere; no key is used." % (int(run), _name(chip)))
    print("Recorded hardware counts and the IBM calibration record in force at the time; circuits are rebuilt on a stand-in chip only to count single-qubit gates.")
    try: v7.mode_analyze()
    finally: shutil.rmtree(work, ignore_errors=True)


def _live(chip):
    if not os.environ.get("IBM_QUANTUM_TOKEN", "").strip():
        sys.exit("Set your own IBM Quantum key first:  export IBM_QUANTUM_TOKEN=...   (it stays on this machine)")
    m, v7 = _load(chip)
    mine = os.path.join(HERE, "my_runs", "run_v7_" + _name(chip)); os.makedirs(mine, exist_ok=True)
    m.RUN = mine
    return m, v7


def check(chip="kingston"):
    m, v7 = _live(chip); v7.mode_dry()
    print("\nNothing was submitted. Estimate for one chip: about %.0f s of IBM time." % ESTIMATE_SECONDS)


def fire(chip="kingston", yes=False):
    if not yes: sys.exit("This submits a job on your IBM account and uses your IBM time. Run check first, then fire with yes=True (--yes).")
    m, v7 = _live(chip); v7.mode_fire()


def fetch(chip="kingston"):
    m, v7 = _live(chip); return v7.mode_fetch()


def analyze(chip="kingston"):
    m, v7 = _live(chip); v7.mode_analyze()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("replay", "check", "fire", "fetch", "analyze"))
    ap.add_argument("--chip", choices=CHIPS, default="kingston"); ap.add_argument("--run", type=int, choices=(1, 2), default=2); ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "replay": replay(a.chip, a.run)
    elif a.cmd == "fire": fire(a.chip, a.yes)
    else: {"check": check, "fetch": fetch, "analyze": analyze}[a.cmd](a.chip)


if __name__ == "__main__":
    main()
