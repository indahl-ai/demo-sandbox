#!/bin/bash
# start.sh: run once when the container starts (RunPod template start command). Makes sure Python 3.11 or newer is available, installs the
# four libraries into ./.venv, and prints how to begin. Nothing here sends anything anywhere except package downloads.
set -e
cd "$(dirname "$0")"
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --quiet --no-cache-dir --root-user-action=ignore -r requirements.txt
else
  python3 -m pip install --quiet --no-cache-dir --root-user-action=ignore uv      # uv fetches its own Python 3.12; the image's older Python is left alone
  uv venv --python 3.12 .venv
  uv pip install --quiet --python .venv/bin/python -r requirements.txt
fi
echo
if [ -n "${JUPYTER_PASSWORD:-}" ]; then   # hosted sandboxes only: the base CPU image has no JupyterLab, so it runs from this environment
  .venv/bin/python -m pip install --quiet --no-cache-dir --root-user-action=ignore jupyterlab ipykernel 2>/dev/null || uv pip install --quiet --python .venv/bin/python jupyterlab ipykernel
  nohup .venv/bin/jupyter lab --allow-root --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token="$JUPYTER_PASSWORD" --ServerApp.allow_origin='*' \
        --ServerApp.root_dir=/workspace --ServerApp.preferred_dir=/workspace/demo > /root/jupyter.log 2>&1 &
  echo "JupyterLab started on port 8888"
fi
echo "SETUP COMPLETE. Quantum Idle Audit demonstrations are ready in /workspace/demo"
echo "  cd /workspace/demo && source .venv/bin/activate"
echo "  Replay a recorded result (no key, no IBM time):   python sandbox.py <demo> replay"
echo "      <demo> = finance, chemistry, optimization, simulation, error_correction or ai_kernels"
echo "  Live run on your own IBM account: see README.md or https://quantumidleaudit.com/runpod-sandbox"
echo "Everything stays in this container. Read README.md first."
