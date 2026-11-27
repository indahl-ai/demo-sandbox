# Quantum Idle Audit demonstrations, run by you

Quantum Idle Audit, Heuremen LLC. https://quantumidleaudit.com  
One click on RunPod, or any machine with Python 3.11 or newer. License: LICENSE.txt.

Six demonstrations, each the same circuit on the same IBM chip run two ways: the client's default settings, and calibrated with the gate error measured on the qubits the circuit uses. Each one can be replayed from our recorded hardware runs with no key, or run live on your own IBM account.

| `<demo>` | What it measures | IBM time per chip, live run (billed on our recorded runs) |
|---|---|---|
| `finance` | option price error against the exact price | about 140 s |
| `chemistry` | molecular energy error (H2) | about 20 s |
| `optimization` | schedule quality (approximation ratio) | about 50 s |
| `simulation` | error against the exact simulated answer | about 42 s |
| `error_correction` | logical qubit failures per round | about 90 s |
| `ai_kernels` | shot budget against a plan built on IBM's published figures, and a readout audit | about 20 s |

Every command prints its full output, then a short RESULT box that restates the headline in plain words. The box only repeats figures printed above it.

## Step 1: start
On RunPod the template does this for you. Setup takes about a minute after the pod starts and ends with `SETUP COMPLETE` in `/root/setup.log`.

    cd /workspace/demo && source .venv/bin/activate

Elsewhere: `git clone https://github.com/indahl-ai/demo-sandbox && cd demo-sandbox && bash start.sh && source .venv/bin/activate`

Releases are tagged. The public RunPod template also clones release tag `v1.4`. On RunPod, open the pod's web terminal (Connect, then Start Web Terminal) and run `cd /workspace/demo && .venv/bin/python sandbox.py owner history`, or the owner check with `--backend` set to your machine. The notebook (`QIA_demonstrations.ipynb`, for Colab and hosted sandboxes) clones release tag `v1.4`, a fixed commit, never the moving `main`. To use the same code here: `git clone --branch v1.4 https://github.com/indahl-ai/demo-sandbox`.

## Step 2: replay a recorded result. No key, no network, no IBM time.
    python sandbox.py chemistry replay
    python sandbox.py finance replay --chip fez --run 1

Options: `--chip kingston`, `fez` or `marrakesh`, and `--run 1` or `2`. Without options it replays ibm_kingston, run 2. Each folder holds a `STATUS.md` listing every published figure beside the figure the replay prints.

## Step 3: run it live on your own IBM account (optional)
    read -s -p "IBM Quantum API key: " IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN; echo
    python sandbox.py chemistry check --chip fez            # remaining IBM time and the estimate; submits nothing
    python sandbox.py chemistry fire --chip fez --yes       # submits the jobs on your account
    python sandbox.py chemistry fetch --chip fez            # repeat every 30 s or so until it says DONE
    python sandbox.py chemistry analyze --chip fez          # the same comparison, on your run

`read -s` asks for the key without showing it and keeps it out of the shell history. Finance can instead read the key from a file: `export DEMO_KEYFILE=/path/to/keyfile` (a file holding only the key). Keep that file outside the repository folder and never commit it. If your key has more than one IBM instance, `check` lists them and stops: choose one with `export DEMO_INSTANCE="<instance name>"` and run `check` again. Each `fire` refuses if your remaining IBM time is too low for the run, and says how much it needs. Optimization may ask for a short pilot first on days when IBM's record leaves too little signal; the message gives the exact commands (`--pilot`).

A live run is saved on this machine under the repository folder (`analyze` prints the path). Each new terminal or SSH session needs the key entered again (and `DEMO_INSTANCE` set again, if used).

## What leaves this machine
On your own machine, in Colab, or on a RunPod pod you start yourself: only calls to IBM, made with your key, plus package downloads at setup. Your key is read from the environment variable, never printed and never written to a file. Nothing is sent to Heuremen. To stop everything, delete the container.

On a sandbox hosted by QIA, the container is run by the operator, and the private link opens a root session in it. Use replay only there, or an IBM key you revoke afterwards.

## For AI agents and scripts
Only with the account owner's approval: a pod and a live run spend on their RunPod and IBM accounts. Deploy the public RunPod template `kpyzbrz9gs` on a CPU pod (`cpu3c-2-4` is enough) through RunPod's API, with your SSH public key in the pod's `PUBLIC_KEY` environment variable. Wait until `/root/setup.log` contains `SETUP COMPLETE`, then run the commands above over SSH. Pass the IBM key over SSH standard input into `IBM_QUANTUM_TOKEN` rather than on a command line. Terminate the pod when done.

## What the replay is, exactly
It re-analyzes the recorded hardware counts and the recorded calibration data. The only stand-in is the chip's shape: circuits are rebuilt on a fake IBM chip to count gates, which the fit needs. Every published number comes out the same to the digits published, with three exceptions, error-correction cells that differ by 0.001 from a double rounding (a four-digit print rounded again to three), listed in `error_correction/STATUS.md`. A pooled finance cell with the same cause was corrected on the page on 25 Nov 2026 (`finance/STATUS.md`). Each `STATUS.md` lists every published figure beside the replayed one.

## Read the marrakesh finance runs differently
On ibm_marrakesh the finance fade scale sits on the edge of its search range: the error there is a layout fault, not a fade. It is the layout-fault control, and its improvement is not a calibration result.

## What it does not claim
Each demonstration shows one circuit on three IBM chips in two recorded runs. A live run shows that day's chip, so its figures vary around the published ones. It does not say your own circuit will improve by the same factor.

## Checks
`python tests/test_cli.py` drives every command of every demonstration with the IBM calls replaced, and fails if an option is lost on the way. It submits nothing.

`python -m pytest tests/test_owner.py` checks the owner counting rules, re-reads one published ibm_marrakesh run (job `dc4t0smq4uuc73a9e63g`, in `tests/fixtures/`) from its raw counts and requires every published figure back to the digit, and builds the check circuits on a stand-in chip. It submits nothing.

## For machine owners

An independent record of a dedicated IBM machine's readout and T1, measured on the owner's own account and set beside the record IBM publishes.

### The record so far, on the public Heron r2 and Nighthawk r2 chips. No key, no network.

    python sandbox.py owner history --family heron_r2
    python sandbox.py owner history --family nighthawk_r2

Replays the calibration readings Heuremen has published for ibm_fez, ibm_kingston and ibm_marrakesh since 30 Oct 2026 (Heron r2), or for ibm_phoenix and ibm_chicago (Nighthawk r2, whose map is printed by qubit number), from a read-only extract in `owner/recorded/`. Each reading names its published file, `https://quantumidleaudit.com/qia/cal/<file>.json`. Per chip it prints the runs and qubit readings, the readout readings that read off IBM's record (worse than stated), the distinct qubits that did so at least once, the readings at 10x IBM's figure or more, the age of IBM's readout record when each run started, the ten worst single readings, and a qubit map. The headline counts are readout only. T1 is printed apart, both ways.

A reading counts as off when IBM's figure and the measured value do not overlap. Both get an exact 99.9% interval, since IBM's readout figures are themselves counts out of 1,024 or 4,096 shots. Where a run was re-read under a later rule (published as `<job>n3`), the re-read replaces the first reading.

### The same check on the owner's machine

    read -s -p "IBM Quantum API key: " IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN; echo
    python sandbox.py owner check   --backend ibm_cleveland          # qubit count, age of IBM's current record, IBM time, time left. Submits nothing.
    python sandbox.py owner fire    --backend ibm_cleveland --yes    # submits one job on the account
    python sandbox.py owner analyze --backend ibm_cleveland          # repeat every 30 s or so until it prints the table

Any machine the key can see works. The job is the one behind every published reading: every usable qubit at once, physical layout, measured at rest and after X (1,000 shots each), and after X and a delay of 15, 40, 80, 160 or 320 us (500 shots each). IBM billed 4 s for each of the 2,166 such jobs on record. `fire` won't run without `--yes`, and it refuses with less than 34 s of IBM time left. IBM's record is saved when the job is sent, and the record in force when the job ran is pulled at analysis. `analyze` prints the same table as `history` and saves the result in `my_runs/`, in the same shape as the published files. The key is read from `IBM_QUANTUM_TOKEN`, never printed and never written to a file. Nothing goes to Heuremen.

One run reads one hour of one machine. A record of a machine needs repeat runs.

## Check your own circuit

Measure the gates your own circuit uses, on the chip as it is today, on your own IBM account, and set them beside IBM's published figures.

### What leaves your machine, and what doesn't

- **Stays on your machine:** the circuit. The `.qasm` file is read and laid out (transpiled) locally. It is never submitted.
- **Goes to IBM, on your key:** only check circuits. That's short two-qubit randomized-benchmarking sequences on the pairs your layout uses (lengths 1, 8, 24 and 48 Cliffords, eight sequences each, 300 shots), plus a readout check on your circuit's qubits (all 0 and all 1, 2,000 shots each). IBM sees which qubits and pairs were checked, as it does for any job.
- **Your key:** read from `IBM_QUANTUM_TOKEN` and used only to reach your own IBM account.
- **Results:** saved locally in `my_runs/` and on your IBM account. Nothing goes to Heuremen.

### The four commands

```bash
export IBM_QUANTUM_TOKEN=...        # your own key, kept on this machine

python sandbox.py mycircuit plan    my_circuit.qasm --backend ibm_kingston         # lay it out, list its pairs and IBM's figures, estimate IBM time. Submits nothing.
python sandbox.py mycircuit fire    my_circuit.qasm --backend ibm_kingston --yes   # submits the check: one benchmarking job and one readout job
python sandbox.py mycircuit fetch   my_circuit.qasm --backend ibm_kingston         # collects them. If not finished, run again in about 30 seconds.
python sandbox.py mycircuit analyze my_circuit.qasm --backend ibm_kingston         # IBM's figure beside today's measurement, pair by pair and qubit by qubit
```

`fire` won't run without `--yes`, and it refuses if your account doesn't have the estimated IBM time plus 30 seconds to spare. `plan` prints the estimate first.

### How long it takes

About 55 seconds of IBM time (estimate) for a circuit that uses 7 pairs on 8 qubits (224 benchmarking circuits plus the readout check), plus queue time. Only the 10 most-used pairs are checked by default. Set `MYCIRCUIT_MAX_PAIRS` to change that. Unchecked pairs are marked in `plan`.

### Try it offline first

```bash
python sandbox.py mycircuit plan my_circuit.qasm --backend ibm_kingston --fake kingston
```

`--fake kingston`, `--fake fez` or `--fake marrakesh` lays the circuit out on a stand-in chip with no key and nothing sent. The stand-in's figures are a stored snapshot, not today's. `--fake` works with `plan` only.

### Export your circuit from Qiskit

```python
from qiskit import qasm2, qasm3

open("my_circuit.qasm", "w").write(qasm2.dumps(qc))    # OpenQASM 2
# or
open("my_circuit.qasm", "w").write(qasm3.dumps(qc))    # OpenQASM 3
```

Both versions load. Reading an OpenQASM 3 file needs `pip install qiskit-qasm3-import`. The circuit needs at least one two-qubit gate after layout, since the check measures two-qubit gates.

### Reading the result

The first line is the answer: how many of the pairs your circuit uses read WORSE than IBM's published error, and how many qubits read worse than stated. A pair is flagged only when its whole interval sits above 1.25x of IBM's figure. WORSE is the only flag: on hardware this check has read most pairs below IBM's figure, which is not yet explained, so a low ratio is shown but never flagged. The second line gives the signal your two-qubit gates keep, by IBM's figures and by today's measurement.

Benchmarking gives the average error per gate on each pair. It does not predict exactly how far your circuit's answer moves. The signal-kept line is a depolarizing-model estimate: it multiplies (1 - 4e/3) over your two-qubit gates and ignores single-qubit error, readout, idling, coherent error and crosstalk.

The interval is labelled 99%. In a simulator test with known answers (`tests/null_test.py`, a chip simulated at exactly IBM's figures) the check read 1.03x on average, never flagged WORSE in 100 tries, and its interval covered the truth 92 times in 100. It covers sampling within one check, not change between checks: a repeat minutes later has differed by more. Single-qubit error inside the benchmark sequences is taken from IBM's stated single-qubit figures, not re-measured.
