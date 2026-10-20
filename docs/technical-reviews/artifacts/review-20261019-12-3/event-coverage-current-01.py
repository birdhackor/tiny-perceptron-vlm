"""核對短聲音完整落框的條件與位移的幾何關係。"""

import hashlib
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SIZE = 400
SAMPLES = 1600


def containing_frames(start, length, hop, frame_starts):
    return [
        origin for origin in frame_starts
        if origin <= start and start + length <= origin + SIZE
    ]


def phase_covered(phase, length, hop):
    candidates = [index * hop for index in range(-8, 9)]
    return bool(containing_frames(phase, length, hop, candidates))


finite_examples = []
for start, length in [(399, 100), (399, 400), (320, 300), (319, 300), (400, 400)]:
    by_hop = {}
    for hop in (160, 400):
        frame_starts = list(range(0, SAMPLES - SIZE + 1, hop))
        by_hop[str(hop)] = containing_frames(start, length, hop, frame_starts)
    finite_examples.append({
        "event_start": start,
        "event_length": length,
        "event_inclusive_end": start + length - 1,
        "fully_containing_frame_starts_by_hop": by_hop,
    })

phase_enumeration = []
for length in (100, 240, 241, 300, 400, 401):
    by_hop = {}
    for hop in (160, 400):
        covered = sum(phase_covered(phase, length, hop) for phase in range(hop))
        expected = min(hop, max(0, SIZE - length + 1))
        if covered != expected:
            raise AssertionError("相位枚舉與完整包含條件不符")
        by_hop[str(hop)] = {
            "phase_count": hop,
            "covered_phase_count": covered,
            "derived_covered_phase_count": expected,
            "coverage_fraction": str(Fraction(covered, hop)),
        }
    phase_enumeration.append({"event_length": length, "by_hop": by_hop})

previous_probe = ROOT / "docs/technical-reviews/artifacts/review-20261019-12-3/verify_framing.py"
command = [sys.executable, str(previous_probe)]
completed = subprocess.run(
    command, cwd=ROOT, capture_output=True, text=True, check=True,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
)
probe = json.loads(completed.stdout)
print(json.dumps({
    "environment": probe["environment"],
    "rerun": {
        "command_argv": command,
        "code_sha256": hashlib.sha256(previous_probe.read_bytes()).hexdigest(),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    },
    "finite_examples": finite_examples,
    "phase_enumeration": phase_enumeration,
    "derivation": {
        "containment": "事件 [s,s+D-1] 完整位於框 [kH,kH+L-1] 的充要條件為 s+D-L <= kH <= s。",
        "relative_phase": "對錄音內部的週期框格，令 r=s mod H，最近且不晚於事件起點的框最可能容納全事件；條件是 r+D<=L。",
        "phase_count": "在 r=0,...,H-1 的整數相位中，可完整容納的相位數為 min(H,max(0,L-D+1))。",
        "comparison_scope": "相位比例只是在內部週期格、起點均勻遍歷相位時的幾何比例，並非錄音事件的實測機率，也不保證每一個起點改用較小位移都更好；有限錄音仍受頭尾邊界限制。",
    },
}, ensure_ascii=False, indent=2))
