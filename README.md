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

Releases are tagged. The notebook (`QIA_demonstrations.ipynb`, for Colab and hosted sandboxes) clones release tag `v1.1`, a fixed commit, never the moving `main`. To use the same code here: `git clone --branch v1.1 https://github.com/indahl-ai/demo-sandbox`.

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
