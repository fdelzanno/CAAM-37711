import subprocess
import sys
from pathlib import Path

from common import RESULTS_JSON

SRC = Path(__file__).resolve().parent
SCRIPTS = ["p4.py", "p5_classification.py", "p5_regression.py"]

RESULTS_JSON.unlink(missing_ok=True)
for script in SCRIPTS:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=SRC)
