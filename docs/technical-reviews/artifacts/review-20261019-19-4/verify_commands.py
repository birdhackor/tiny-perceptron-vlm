"""解析配方並從兩種工作目錄執行教材短程式；不執行長訓練。"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from scripts.selftrained import train

source = (ROOT / "docs/course-review-20261019/freeze-02/sources/19.4.md").read_text()
parsed = []
for block in re.findall(r"```bash\n(.*?)```", source, re.S):
    parts = shlex.split(block.replace("\\\n", " "))
    split = parts.index("--")
    args = train.parser().parse_args(parts[split + 1:] + ["--records", "data/placeholder.jsonl", "--asset-dir", "data/placeholder"])
    parsed.append({"command": block, "wrapper_argv": parts[:split], "trainer_options": vars(args)})
code = re.search(r"```python\n(.*?)```", source, re.S).group(1)
outputs = {}
for directory in [ROOT, ROOT / "notebooks/19"]:
    assert directory.is_dir(), "教材執行目錄不存在"
    command = [sys.executable, "-c", code]
    p = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=True)
    outputs[str(directory.relative_to(ROOT))] = {"command": command, "stdout": p.stdout, "stderr": p.stderr, "exit_code": p.returncode}

paths = ["docs/selftrained/v2-jobs/release-final-four-exports.json", ".github/workflows/selftrained-assistant.yml"]
checks = {}
with tempfile.TemporaryDirectory(prefix="commands-", dir=ARTIFACT) as temporary:
    work = Path(temporary)
    occupied = work / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("保留既有內容\n")
    command = ["git", "clone", str(ROOT), str(occupied)]
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checks["existing_clone_destination"] = {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    assert p.returncode != 0, "Git 未拒絕已存在且非空的 clone 目的地"
    command = [sys.executable, str(ROOT / "scripts/selftrained/train_local_stage.py"), "--manifest", str(ROOT / "docs/selftrained/v2-manifest.json"), "--manifest-sha256", "0" * 64, "--data-root", str(work), "--", "--output-dir", str(work / "training"), "--stage", "pretrain", "--steps", "1"]
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checks["manifest_pin_rejection"] = {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "no_output_directory": not (work / "training").exists()}
    assert p.returncode != 0 and not (work / "training").exists(), "manifest 指紋錯誤未在訓練開始前阻擋"

out = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-4/verify_commands.py",
    "environment": {"python": sys.version, "torch": str(train.torch.__version__), "device": "cpu"},
    "command_parse": parsed, "cwd_executions": outputs, "negative_checks": checks,
    "release": json.loads((ROOT / paths[0]).read_text()),
    "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    "attempt_naming": next(line.strip() for line in (ROOT / paths[1]).read_text().splitlines() if "SELFTRAINED_RUN:" in line),
    "scope": "長訓練配方只核對解析結果與來源限制；實際執行的是教材離線讀檔範例及兩個訓練前的拒絕條件。",
}
(ARTIFACT / "command-checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "核對完成", "parsed_commands": len(parsed), "cwd_executions": list(outputs), "lines": [len(r["stdout"].splitlines()) for r in outputs.values()]}, ensure_ascii=False))
