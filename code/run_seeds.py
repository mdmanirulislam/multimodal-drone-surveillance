"""Run the remaining training seeds while asking Windows not to idle-sleep (released when this process exits)."""
import ctypes
import subprocess
import sys
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

logs = Path(__file__).resolve().parents[1] / "logs"
for s in sys.argv[1:]:
    with open(logs / f"train_seed{s}.log", "w") as f:
        rc = subprocess.call([sys.executable, "train_audio.py", "--seed", s], stdout=f, stderr=subprocess.STDOUT)
    if rc:
        sys.exit(rc)
print("ALL DONE")
