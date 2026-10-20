"""複查未變主張、依賴文件差異及本次小節執行；不訓練或載入權重。"""

import ast
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.training import choose_device, load_checkpoint


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def section(raw, identifier):
    pattern = rb"(?m)^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)"
    return re.search(pattern, raw, re.S)[0]


def device_default(path):
    tree = ast.parse((ROOT / path).read_bytes())
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--device"]
    if not found:
        return None
    assert len(found) == 1
    return ast.literal_eval(next(k.value for k in found[0].keywords if k.arg == "default"))


def main():
    report_path = "docs/course-review-20261019/reviews/rv1019_technical_9-3/technical-9-3-original-01.json"
    raw_report = (ROOT / report_path).read_bytes()
    assert sha(raw_report) == "d74b7875f719ec086543ce5d1701c94b6a885a42a69fb1fd71c8e4e7c99e24c5"
    previous = json.loads(raw_report)
    assert previous["issues"] == []
    source_checks = []
    for source in previous["sources"]:
        if source["kind"] != "repository_code":
            continue
        current_hash = sha((ROOT / source["path"]).read_bytes())
        same = current_hash == source["sha256"]
        assert same or source["path"] == "course/training.md"
        source_checks.append({"source_id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current_hash, "unchanged": same})
    artifact_checks = []
    for artifact in previous["artifacts"]:
        current_hash = sha((ROOT / artifact["path"]).read_bytes())
        assert current_hash == artifact["sha256"]
        artifact_checks.append({"path": artifact["path"], "sha256": current_hash, "unchanged": True})
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/9.3.md").read_bytes()
    assert section((ROOT / "course/chapters/09.md").read_bytes(), "9.3") == frozen
    freeze = (ROOT / "docs/course-review-20261019/freeze-06.json").read_bytes()
    assert sha(freeze) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
    page = next(p for p in json.loads(freeze)["pages"] if p["page_id"] == "9.3")
    assert page["source_sha256"] == sha(frozen) and page["figures_sha256"] == {}
    old = run(["git", "show", "5ef56bc756f3:course/training.md"])["stdout"].encode()
    target = run(["git", "show", "6a7aa1de5227:course/training.md"])["stdout"].encode()
    current = (ROOT / "course/training.md").read_bytes()
    assert sha(old) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
    assert target == current
    assert sha(current) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert section(old, "T.5") == section(current, "T.5")
    training_diff = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
    command = ".venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda"
    assert command.encode() in section(current, "T.5")
    cli = []
    defaults = {}
    for path, expected in (("scripts/train.py", "auto"), ("scripts/evaluate.py", "auto"), ("scripts/infer.py", "auto"), ("scripts/course_experiments/run.py", "cpu"), ("scripts/quantize.py", None), ("scripts/prepare_data.py", None), ("scripts/fetch_training_assets.py", None)):
        defaults[path] = device_default(path)
        assert defaults[path] == expected
        argv = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"] if path.endswith("/run.py") else [str(ROOT / ".venv/bin/python"), path, "--help"]
        receipt = run(argv)
        assert ("--device" in receipt["stdout"]) == (expected is not None)
        receipt["path"] = path
        cli.append(receipt)
    device_checks = []
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
            observed = str(choose_device())
        assert observed == expected
        device_checks.append({"cuda_available": cuda, "mps_available": mps, "observed": observed})
    assert str(choose_device("cpu")) == "cpu"
    assert inspect.signature(load_checkpoint).parameters["device"].default == "cpu"
    quantize_tree = ast.parse((ROOT / "scripts/quantize.py").read_bytes())
    loads = [node for node in ast.walk(quantize_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_checkpoint"]
    assert len(loads) == 1 and len(loads[0].args) == 1 and not loads[0].keywords
    curl = ["curl", "--fail", "--location", "--silent", "--show-error", "https://docs.pytorch.org/docs/2.14/notes/randomness.html"]
    fetched = subprocess.run(curl, capture_output=True, check=True)
    class Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
        def handle_data(self, value):
            self.parts.append(value)
    parser = Text()
    parser.feed(fetched.stdout.decode())
    text = " ".join(" ".join(parser.parts).split())
    excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
    assert excerpt in text
    run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T103230401233Z"
    files = {}
    for name in ("execution.json", "extraction.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (run_dir / name).read_bytes()
        files[name] = {"sha256": sha(raw), "content": raw.decode()}
    assert json.loads(files["execution.json"]["content"])["exit_code"] == 0
    assert files["stdout.txt"]["content"].splitlines()[-1] == "可核對事實 4"
    assert files["stderr.txt"]["content"] == ""
    extra_paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "tiny_perceptron/training.py"]
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-3/recheck-current-01.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "original_report": {"path": report_path, "sha256": sha(raw_report)},
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "original_issues": [],
        "source_sha256": sha(frozen),
        "training_diff": training_diff,
        "training_section_unchanged": {"selector": "T.5", "sha256": sha(section(current, "T.5")), "unchanged": True, "cited_command": command},
        "cli_help": cli,
        "device_defaults": defaults,
        "device_selection": {"method": "mock availability flags; no accelerator tensors", "cases": device_checks, "explicit_cpu": "cpu", "quantize_load_default": "cpu"},
        "additional_repository_sources": [{"path": path, "sha256": sha((ROOT / path).read_bytes())} for path in extra_paths],
        "reproducibility_source": {"url": curl[-1], "version": "PyTorch 2.14 documentation", "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": curl, "exit_code": fetched.returncode, "original_sha256": sha(fetched.stdout), "locator": "Reproducibility / opening paragraph", "excerpt": excerpt, "full_text_saved": False},
        "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/09.md#9.3\" --execute", "original_run_directory": str(run_dir.relative_to(ROOT)), "files": files},
        "scope": "沿用已核對且雜湊未變的原始證據，補查依賴文件差異。CLI僅跑help，裝置選擇只測分支，未訓練或載入權重；未做跨電腦或量化品質實驗。",
        "assertions": "全部通過",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
