"""tests/test_notebook.py: the notebook's code cells run in a pod-like environment (25 Nov 2026).

Cause it guards: the hosted sandbox runs the notebook on the .venv kernel that start.sh builds with uv, and a uv venv has no pip.
The setup cell called pip whenever the kernel was the .venv Python, so the first cell failed on the first human test. The grant script
only checked that JupyterLab answered, and the Colab test ran on a Python with pip. This test builds a uv venv (no pip), installs the
requirements with uv, and runs every code cell of QIA_demonstrations.ipynb in it, from the repository root, as the pod does."""
import json, os, shutil, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(shutil.which('uv'), 'uv not installed')
class NotebookInPodVenv(unittest.TestCase):
    def test_cells_run_on_pipless_venv(self):
        d = os.path.realpath(tempfile.mkdtemp())   # /var -> /private/var on macOS, else sys.executable never equals the cwd path
        try:
            shutil.copytree(ROOT, os.path.join(d, 'demo'), ignore=shutil.ignore_patterns('.venv', '.git', '__pycache__'))
            demo = os.path.join(d, 'demo')
            subprocess.run(['uv', 'venv', '-q', '--python', '3.12', '.venv'], cwd=demo, check=True)
            subprocess.run(['uv', 'pip', 'install', '-q', '--python', '.venv/bin/python', '-r', 'requirements.txt'], cwd=demo, check=True)
            py = os.path.join(demo, '.venv', 'bin', 'python')
            self.assertNotEqual(subprocess.run([py, '-m', 'pip', '--version'], capture_output=True).returncode, 0, 'venv should have no pip, like the pod')
            nb = json.load(open(os.path.join(demo, 'QIA_demonstrations.ipynb')))
            src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
            r = subprocess.run([py, '-c', src], cwd=demo, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn('RESULT', r.stdout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
