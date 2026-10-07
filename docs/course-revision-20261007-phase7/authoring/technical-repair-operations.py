"""Author CPU checks for four technical repairs; never reviewer judgments."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/course-revision-20261007-phase7/authoring"
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, record):
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with (BASE / name).open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def guard():
    source = ROOT / "docs/natural-assistant/v4/TRAINING.md"
    fence = next(x for x in re.findall(r"```bash\n(.*?)\n```", source.read_text(), re.S) if x.startswith("PENDING_CHECKPOINTS="))
    rows = []
    for n, expected in [(0, "1039,2077"), (1000, "1039,2077"), (1100, "2077"), (2076, "2077"), (-1, None), (2077, None), (2078, None)]:
        with tempfile.TemporaryDirectory(prefix="phase7-resume-guard-") as directory:
            root = Path(directory)
            python = root / ".venv-natural/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            progress = root / "outputs/natural-my-v4/train/adapter/training.json"
            progress.parent.mkdir(parents=True)
            progress.write_text(json.dumps({"completed_steps": n}))
            stub = root / "scripts/natural_assistant.py"
            stub.parent.mkdir()
            stub.write_text('import json,sys\nfrom pathlib import Path\nPath("train-marker.json").write_text(json.dumps(sys.argv[1:]))\n')
            result = subprocess.run(["bash", "--noprofile", "--norc", "-c", fence], cwd=root, text=True, capture_output=True)
            marker = root / "train-marker.json"
            args = json.loads(marker.read_text()) if marker.exists() else None
            observed = args[args.index("--checkpoint-steps") + 1] if args else None
            passed = (result.returncode == 0 and observed == expected) if expected is not None else (result.returncode != 0 and args is None)
            rows.append({"completed_steps": n, "expected_pending_checkpoints": expected, "expected_train_invoked": expected is not None, "observed_pending_checkpoints": observed, "train_stub_invoked": args is not None, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": passed})
    record = {"kind": "author actual Bash guard CPU check", "command": ".venv/bin/python docs/course-revision-20261007-phase7/authoring/technical-repair-operations.py guard", "scope": "Exact published fence in a temporary filesystem; real progress-reading Python and Bash; only the training program replaced by a marker stub. No model, GPU, remote service, or training.", "python": sys.version, "bash": subprocess.check_output(["bash", "--version"], text=True).splitlines()[0], "source_sha256": digest(source), "fence": fence, "fence_sha256": hashlib.sha256(fence.encode()).hexdigest(), "records": rows, "passed": all(x["passed"] for x in rows)}
    save("technical-resume-guard-check.json", record)
    print(json.dumps({"passed": record["passed"], "cases": len(rows)}), flush=True)
    assert record["passed"]


def kernels():
    from scripts.check_notebooks import run_kernel
    from scripts.export_course import checked_notebook
    import torch

    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    selected = [x for x in index if x["id"] in {"10.8", "20.8", "20.13"}]
    assert len(selected) == 3
    rows = []
    for item in selected:
        source = ROOT / item["notebook"]
        started = time.perf_counter()
        executed = Path(run_kernel(source, ROOT / "outputs/notebooks", 120))
        checked_notebook(json.loads(source.read_text()), executed)
        nb = json.loads(executed.read_text())
        row = {"page_id": item["id"], "source_notebook_sha256": digest(source), "executed_path": str(executed.relative_to(ROOT)), "executed_sha256": digest(executed), "python_from_kernel": nb["metadata"]["language_info"]["version"], "torch_in_venv": torch.__version__, "seconds": round(time.perf_counter() - started, 3), "status": "executed_without_errors_and_source_matched"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    save("technical-repair-kernel-checks.json", {"kind": "author three fresh CPU kernels, not reviewer judgments", "command": ".venv/bin/python docs/course-revision-20261007-phase7/authoring/technical-repair-operations.py kernels", "device": "CPU", "scope": "Three edited notebook sources; no model-capability retraining.", "records": rows, "completed": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("guard", "kernels"))
    args = parser.parse_args()
    {"guard": guard, "kernels": kernels}[args.action]()
