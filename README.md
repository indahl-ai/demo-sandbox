# The finance demonstration, run by you

Quantum Idle Audit, Heuremen LLC. Option pricing by amplitude estimation on IBM hardware.

## What this shows
The same circuit, the same shots, fitted two ways: the default way (gates assumed good) and with the gate error measured on the qubits the circuit uses.
The result is the error in the option price against the exact price (8.2543). Lower is better.

## Step 1: replay our recorded runs. No key, no network, no IBM time.
    python3 sandbox.py replay                      # run 2, ibm_kingston
    python3 sandbox.py replay --chip fez --run 1
The recorded hardware runs (every IBM job ID is in the state files under `recorded/`) are re-analyzed on this machine.
You should see the default price error and ours, with 99% intervals. Change any setting in `finance/` and run it again.

## Step 2: run it on your own IBM account (optional)
    export IBM_QUANTUM_TOKEN=<your own IBM Quantum key>
    python3 sandbox.py check --chip kingston       # shows your remaining IBM time and what a run would use; submits nothing
    python3 sandbox.py fire  --chip kingston --yes # submits two jobs; refuses if the estimate is over 400 s or over your remaining time
    python3 sandbox.py fetch --chip kingston       # when they finish
    python3 sandbox.py analyze --chip kingston     # the same comparison, on your run

## What leaves this machine
Only calls to IBM, made with your key. Your key is read from the environment variable, never printed and never written to a file.
Nothing is sent to us. Your runs are saved in `my_runs/` here and nowhere else. To stop everything, delete the container.

## What it costs you
Replay: nothing. A live run: about 370 seconds (a little over 6 minutes) of IBM QPU time on one chip. `check` prints the estimate for your chip and your remaining time before you commit, and `fire` refuses if your remaining time is less than the estimate.
On RunPod, a small CPU pod is enough; no GPU is used.

## What it does not claim
It shows one circuit, one option, on three IBM chips, in two recorded runs. It does not say your own circuit will improve by the same factor.
