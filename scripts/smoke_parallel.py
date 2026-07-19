"""E2E smoke: three concurrent say.py calls with distinct session tags.

Expected: audio plays serialized (the speech lock), and while one speaks the
hub shows the other two queued, then drains them into the history in order.
Each fake session gets its own session_id so the hub treats them as distinct
speakers even though they share this script's parent.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SAY = str(ROOT / "say.py")

SESSIONS = [
    ("Alfa, smoke", "aaaaaaaa-0000-0000-0000-000000000001"),
    ("Bravo, smoke", "bbbbbbbb-0000-0000-0000-000000000002"),
    ("Charlie, smoke", "cccccccc-0000-0000-0000-000000000003"),
]

procs = []
for i, (tag, sid) in enumerate(SESSIONS):
    env = dict(os.environ, CLAUDE_CODE_SESSION_ID=sid)
    procs.append(subprocess.Popen(
        [PY, SAY, f"{tag}: probando la cola, numero {i + 1}."], env=env))
    time.sleep(0.4)  # stagger so the queue order is deterministic

for p in procs:
    p.wait()

print("done: the hub should have cycled 3 speakers, history newest-first")
