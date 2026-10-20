"""保存來源差異、程式重跑與裝置說明的複查證據。"""

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.quantization import quantize_symmetric
from tiny_perceptron.training import choose_device


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(argv):
    result = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    artifact_dir = Path(__file__).parent
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_18-1/technical-18-1-original-01.json"
    original = json.loads(original_path.read_bytes())
    assert digest(original_path.read_bytes()) == "4f76c0153c727129f72fea9daeed1bf9e2d98bb00223d8d0391164e9212f2404"
    artifacts = [
        {"path": item["path"], "sha256": digest((ROOT / item["path"]).read_bytes()), "unchanged": digest((ROOT / item["path"]).read_bytes()) == item["sha256"]}
        for item in original["artifacts"]
    ]
    assert all(item["unchanged"] for item in artifacts), "原始證據的內容已變動"
    repo_sources = []
    for item in original["sources"]:
        if item["kind"] == "repository_code":
            actual = digest((ROOT / item["path"]).read_bytes())
            repo_sources.append({"path": item["path"], "previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": item["sha256"] == actual})
    assert [item["path"] for item in repo_sources if not item["unchanged"]] == ["course/training.md"], "引用來源出現額外變動"
    old_doc = subprocess.check_output(["git", "show", "5ef56bc756f3:course/training.md"], cwd=ROOT)
    current_doc = (ROOT / "course/training.md").read_bytes()
    assert digest(old_doc) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
    assert current_doc == subprocess.check_output(["git", "show", "6a7aa1de5227:course/training.md"], cwd=ROOT)
    diff = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
    fixed_sections = []
    for section in ("T.4", "T.5", "T.8"):
        pattern = rb"(?ms)^## " + re.escape(section.encode()) + rb" .*?(?=^## |\Z)"
        old = re.search(pattern, old_doc).group()
        new = re.search(pattern, current_doc).group()
        assert old == new, "教師固定配方有實質差異"
        fixed_sections.append({"heading": new.splitlines()[0].decode(), "previous_sha256": digest(old), "current_sha256": digest(new), "identical_bytes": True})
    section_run = Path(sys.argv[1])
    extraction = json.loads((section_run / "extraction.json").read_bytes())
    execution = json.loads((section_run / "execution.json").read_bytes())
    environment = json.loads((section_run / "environment.json").read_bytes())
    section_text = (section_run / "section.md").read_bytes()
    assert section_text == (ROOT / "docs/course-review-20261019/freeze-01/sources/18.1.md").read_bytes()
    for item in execution["artifacts"]:
        assert digest((section_run / item["path"]).read_bytes()) == item["sha256"], "小節執行證據的雜湊不符"
    section_output = {"extraction": extraction, "execution": execution, "environment": environment, "stdout": (section_run / "stdout.txt").read_text(), "stderr": (section_run / "stderr.txt").read_text()}
    assert execution["exit_code"] == 0 and not section_output["stderr"]
    probe = run([str(ROOT / ".venv/bin/python"), str(artifact_dir / "verify.py")])
    old_probe = json.loads((artifact_dir / "verification.json").read_bytes())
    new_probe = json.loads(probe["stdout"])
    expected_probe = json.loads(json.dumps(old_probe))
    expected_probe["input_sha256"]["course/training.md"] = digest(current_doc)
    assert new_probe == expected_probe, "補充核對結果出現額外變動"
    help_outputs = []
    cli_paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/course_experiments/run.py", "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"]
    for name in cli_paths:
        out = run([str(ROOT / ".venv/bin/python"), name, "--help"])
        expected = name in cli_paths[:4]
        assert ("--device" in out["stdout"]) == expected, "裝置命令列介面與說明不符"
        help_outputs.append(out)
    device_routes = []
    for cuda_available, mps_available, expected in ((True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        with patch("torch.cuda.is_available", return_value=cuda_available), patch("torch.backends.mps.is_available", return_value=mps_available):
            observed = str(choose_device())
        assert observed == expected, "自動裝置選擇順序不符"
        device_routes.append({"cuda_available": cuda_available, "mps_available": mps_available, "observed": observed, "scope": "mock可用性旗標，只驗證選擇分支，未在該硬體執行tensor運算。"})
    tree = ast.parse((ROOT / "scripts/course_experiments/run.py").read_text())
    device_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--device"]
    default = next(ast.literal_eval(keyword.value) for keyword in device_calls[0].keywords if keyword.arg == "default")
    assert default == "cpu", "固定實驗預設裝置不符"
    weights = torch.tensor([[0.0], [2.0]])
    integers, scale = quantize_symmetric(weights, bits=4)
    restored = integers.float() * scale
    values = torch.tensor([[1.0]])
    candidates = ["blue", "green"]
    before = candidates[(values @ weights.T).argmax().item()]
    after = candidates[(values @ restored.T).argmax().item()]
    assert before == after == "green", "量化反例與預期不符"
    spec = importlib.util.spec_from_file_location("original_probe", artifact_dir / "verify.py")
    original_probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_probe)
    url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
    argv = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    raw = subprocess.check_output(argv)
    parser = original_probe.VisibleText()
    parser.feed(raw.decode())
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
    assert excerpt in text, "官方可重現性原文摘錄不符"
    input_paths = cli_paths + ["tiny_perceptron/training.py", "tiny_perceptron/quantization.py", "docs/course-review-20261019/freeze-06.json"]
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + str(section_run.relative_to(ROOT)),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
        "result": "全部斷言通過；沒有重新訓練或下載權重。",
        "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": digest(original_path.read_bytes()), "issues": original["issues"]},
        "previous_artifacts": artifacts,
        "repository_sources": repo_sources,
        "training_diff": diff,
        "fixed_teacher_sections": fixed_sections,
        "section_execution": section_output,
        "probe_execution": probe,
        "probe_observations_equal_except_training_input_sha256": True,
        "cli_help": help_outputs,
        "device_routes": device_routes,
        "experiment_device_default": default,
        "quantization_counterexample": {"candidates": candidates, "weights": weights.tolist(), "integers": integers.tolist(), "scale": scale.tolist(), "restored": restored.tolist(), "before": before, "after": after, "scope": "人工兩類Linear算例；只反證量化會強制改答blue的說法，不代表實際模型量化品質。"},
        "reproducibility_source": {"url": url, "version": "PyTorch 2.14 documentation", "accessed_at": datetime.now(timezone.utc).isoformat(), "command_argv": argv, "exit_code": 0, "original_sha256": digest(raw), "original_bytes": len(raw), "locator": "Reproducibility / opening limitations", "excerpt": excerpt, "retention": "全文在記憶體中讀取；只保存短摘錄與收據。"},
        "input_sha256": {name: digest((ROOT / name).read_bytes()) for name in input_paths},
        "scope": "重看指定PNG；只讀先前明確引用的圖收據與本次自行產生的小節執行目錄，未讀其他角色報告。固定教師配方與所有頁面主張沿用經雜湊確認未變的直接證據。",
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
