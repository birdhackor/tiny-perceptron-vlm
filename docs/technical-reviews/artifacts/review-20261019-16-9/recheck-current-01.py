"""核對教材引用的現行版本、裝置入口及已封存證據；不執行訓練。"""

import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_16-9/technical-16-9-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headers) if h[0].startswith(("## " + identifier + " ").encode()))
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def call(argv):
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return process.stdout, {"command_argv": argv, "exit_code": process.returncode, "stdout": process.stdout.decode(), "stderr": process.stderr.decode()}


def device_arguments(path):
    tree = ast.parse(path.read_text())
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--device":
                result.append({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    return result


def main():
    original = json.loads(ORIGINAL.read_bytes())
    assert sha(ORIGINAL.read_bytes()) == "7cb580eecef63c9569d0c86abef37dce6b2b39f41714605aaa34a02d9bb9d925"
    previous_audit = json.loads((BASE / "audit-results.json").read_bytes())
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/16.9.md"
    current_section = section((ROOT / "course/chapters/16.md").read_bytes(), "16.9")
    assert current_section == frozen.read_bytes()
    assert sha(current_section) == original["source_sha256"]
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-9/recheck-current-01.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform()},
        "source_sha256": sha(current_section),
        "previous_issues": original["issues"],
        "source_matches_current_section": True,
        "scope": "現行檔、既有證據hash、CLI與裝置分支核對；沒有CUDA訓練、模型推論或下載資料包。",
    }
    old_files = [ORIGINAL, *(ROOT / a["path"] for a in original["artifacts"]), BASE / "self-check-01.json"]
    result["preserved_original_files"] = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in old_files}
    for item in original["artifacts"]:
        assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"]
    result["prior_local_input_checks"] = []
    for name, recorded in previous_audit["local_inputs"].items():
        actual = sha((ROOT / name).read_bytes())
        unchanged = actual == recorded
        if name == "course/training.md":
            assert actual == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
        else:
            assert unchanged
        result["prior_local_input_checks"].append({"path": name, "prior_sha256": recorded, "current_sha256": actual, "unchanged": unchanged})
    before, before_receipt = call(["git", "show", "5ef56bc756f3:course/training.md"])
    after, after_receipt = call(["git", "show", "6a7aa1de5227:course/training.md"])
    current = (ROOT / "course/training.md").read_bytes()
    assert sha(before) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
    assert after == current
    _, diff_receipt = call(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
    assert section(before, "T.8") == section(current, "T.8")
    command = ".venv/bin/python -m scripts.course_experiments.run --experiment flash_probe --device cuda"
    assert command.encode() in section(current, "T.8")
    result["training_revision_check"] = {"before_sha256": sha(before), "current_sha256": sha(current), "current_equals_diff_target": True, "T8_bytes_unchanged": True, "T8_sha256": sha(section(current, "T.8")), "explicit_cuda_command": command, "diff_receipt": diff_receipt, "git_show_commands": [before_receipt["command_argv"], after_receipt["command_argv"]]}
    freeze_path = ROOT / "docs/course-review-20261019/freeze-06.json"
    assert sha(freeze_path.read_bytes()) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
    page = next(item for item in json.loads(freeze_path.read_bytes())["pages"] if item["page_id"] == "16.9")
    assert page["source_sha256"] == original["source_sha256"]
    assert page["figures_sha256"] == original["figure_sha256"]
    result["freeze"] = {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path.read_bytes()), "page": page}
    result["figure_checks"] = []
    receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
    for item in receipt["records"]:
        if item["page_id"] != "16.9":
            continue
        svg = ROOT / item["figure"]
        png = ROOT / item["png"]["path"]
        assert sha(svg.read_bytes()) == original["figure_sha256"][item["figure"]]
        assert sha(png.read_bytes()) == item["png"]["sha256"]
        result["figure_checks"].append({"figure": item["figure"], "svg_sha256": sha(svg.read_bytes()), "png": item["png"], "visual_inspection": "本回合已由審閱者以view_image重看指定PNG；數字、箭頭順序、灰色起點與彩色新增尺度均同原判斷。"})
    sys.path.insert(0, str(ROOT))
    runtime = importlib.import_module("tiny_perceptron.training")
    selection = []
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
            observed = str(runtime.choose_device())
            explicit = str(runtime.choose_device("cpu"))
        assert observed == expected and explicit == "cpu"
        selection.append({"mock_cuda_available": cuda, "mock_mps_available": mps, "auto_device": observed, "explicit_cpu": explicit})
    result["auto_device_branch_checks"] = selection
    runner = importlib.import_module("scripts.course_experiments.run")
    result["runner_argument_checks"] = []
    for extras, expected in (([], "cpu"), (["--device", "cuda"], "cuda")):
        argv = ["run", "--experiment", "flash_probe", *extras]
        stdout = io.StringIO()
        fake_result = {"experiment_id": "flash_probe", "elapsed_seconds": 0, "evidence_status": "parser-only", "artifacts": []}
        with patch.object(sys, "argv", argv), patch.object(runner, "execute", return_value=fake_result) as execution, contextlib.redirect_stdout(stdout):
            runner.main()
        observed = execution.call_args.args[1]
        assert observed == expected
        result["runner_argument_checks"].append({"argv": argv, "observed_device": observed, "execute_replaced_with_mock": True, "stdout": stdout.getvalue(), "scope": "只跑真實parser，不執行experiment或建立輸出目錄。"})
    _, listing = call([sys.executable, "-B", "-m", "scripts.course_experiments.run", "--list-assets", "flash_probe"])
    result["readonly_entrypoint"] = listing
    paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "tiny_perceptron/training.py"]
    result["additional_source_hashes"] = {name: sha((ROOT / name).read_bytes()) for name in paths}
    result["cli_device_declarations"] = {name: device_arguments(ROOT / name) for name in paths[:6]}
    for name in paths[:3]:
        assert result["cli_device_declarations"][name] == [{"default": "auto"}]
    for name in paths[3:6]:
        assert result["cli_device_declarations"][name] == []
    result["no_device_cli_help"] = []
    for name in paths[3:6]:
        _, help_receipt = call([sys.executable, "-B", name, "--help"])
        assert "--device" not in help_receipt["stdout"]
        result["no_device_cli_help"].append(help_receipt)
    assert runtime.load_checkpoint.__defaults__[0] == "cpu"
    result["checkpoint_default_device"] = "cpu"
    url = "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/randomness.md"
    raw, download = call(["curl", "-fLsS", "--max-time", "30", url])
    text = raw.decode()
    assert "different platforms" in text and "identical seeds" in text
    result["reproducibility_original"] = {"url": url, "version": "5c4886908584029761b579af026dcfb627c84070", "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": download["command_argv"], "exit_code": download["exit_code"], "sha256": sha(raw), "full_text_saved": False, "excerpt": "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.", "inspection_note": "已讀Reproducibility開頭與Controlling sources of randomness，跨平台甚至CPU/GPU即使相同seed也不保證完全相同；不據此宣稱本課特定三題的失敗率。"}
    result["quantization_example_logic"] = {"kind": "derivation", "before": "red", "after": "red", "expected_label": "blue", "same_answer": True, "before_is_correct": False, "after_is_correct": False, "details": "保持前後答案與答對目標是不同判準；量化不是以標籤進行再訓練，沒有把原錯誤答案修正成blue的契約。此為邏輯例，不是假稱跑過的checkpoint結果。"}
    result["result"] = "本節、既有證據與現行T.8重跑主張均核對成立；裝置說明及結果限制的改句亦有直接依據。"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
