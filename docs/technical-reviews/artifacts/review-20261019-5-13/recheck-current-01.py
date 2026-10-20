"""複查既有證據的延續性、參考文件變動與實際程式入口。"""

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear
from tiny_perceptron.training import choose_device

BASE = Path(__file__).resolve().parent
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_5-13/technical-5-13-original-01.json"
EXPECTED_ORIGINAL = "c61552f5970d58ff552c527186b9647733c94c409d609fa175330c9f911ce4e5"
SOURCE_SHA = "b605ca3c39976c1d3b5da3efbf1ee74d7e0416c17ac6071ff65a36c59fdf6aa0"
FREEZE = "docs/course-review-20261019/freeze-06.json"
FREEZE_SHA = "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
TRAINING_SHA = "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return digest((ROOT / path).read_bytes())


def run(argv):
    completed = subprocess.run(argv, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=90)
    return {"argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


assert file_sha(ORIGINAL) == EXPECTED_ORIGINAL
assert file_sha(FREEZE) == FREEZE_SHA
assert file_sha("course/training.md") == TRAINING_SHA
original = json.loads((ROOT / ORIGINAL).read_bytes())
assert original["issues"] == []
page = next(item for item in json.loads((ROOT / FREEZE).read_bytes())["pages"] if item["page_id"] == "5.13")
assert page["source_sha256"] == SOURCE_SHA and page["figures_sha256"] == {}
source = (ROOT / page["snapshot"]).read_bytes()
assert digest(source) == SOURCE_SHA

sources = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        current = file_sha(item["path"])
        sources.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": current, "unchanged": current == item["sha256"]})
assert [item["path"] for item in sources if not item["unchanged"]] == ["course/training.md"]
artifacts = []
for item in original["artifacts"]:
    current = file_sha(item["path"])
    assert current == item["sha256"], item["path"]
    artifacts.append({"id": item["id"], "path": item["path"], "sha256": current, "unchanged": True})

old_training = run(["git", "show", "5ef56bc756f3:course/training.md"])
new_training = run(["git", "show", "6a7aa1de5227:course/training.md"])
diff = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
assert old_training["exit_code"] == new_training["exit_code"] == diff["exit_code"] == 0
assert digest(old_training["stdout"].encode()) == next(item["previous_sha256"] for item in sources if item["path"] == "course/training.md")
assert digest(new_training["stdout"].encode()) == TRAINING_SHA
reproduction = ".venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda"
assert reproduction in old_training["stdout"] and reproduction in new_training["stdout"]

scripts = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "scripts/course_experiments/run.py"]
commands = [[sys.executable, "-m", "scripts.course_experiments.run", "--help"] if path.endswith("course_experiments/run.py") else [sys.executable, path, "--help"] for path in scripts]
with ThreadPoolExecutor(max_workers=4) as pool:
    helps = list(pool.map(run, commands))
assert all(item["exit_code"] == 0 for item in helps)
cli = []
for path, execution in zip(scripts, helps, strict=True):
    tree = ast.parse((ROOT / path).read_text())
    defaults = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and any(isinstance(arg, ast.Constant) and arg.value == "--device" for arg in node.args):
            defaults.append(ast.literal_eval(next(keyword.value for keyword in node.keywords if keyword.arg == "default")))
    expected = ["cpu"] if path.endswith("course_experiments/run.py") else ["auto"] if path in scripts[:3] else []
    assert defaults == expected
    assert ("--device" in execution["stdout"]) == bool(expected)
    cli.append({"path": path, "sha256": file_sha(path), "device_defaults_from_ast": defaults, "help_execution": execution})

device_branches = []
for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        observed = str(choose_device())
        assert observed == expected
        assert str(choose_device("cpu")) == "cpu"
        device_branches.append({"mocked_cuda_available": cuda, "mocked_mps_available": mps, "observed_auto_selection": observed, "explicit_cpu": "cpu"})

section = run([sys.executable, "docs/review-tools/section_facts.py", "course/chapters/05.md#5.13", "--execute"])
assert section["exit_code"] == 0
section_summary = json.loads(section["stdout"])
assert section_summary["source_sha256"] == SOURCE_SHA
assert section_summary["python_fences"] == 1 and section_summary["svg_references"] == 0
run_directory = Path(section_summary["artifact_directory"])
section_files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"):
    raw = (run_directory / name).read_bytes()
    section_files[name] = {"sha256": digest(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
assert section_files["execution.json"]["content"]["exit_code"] == 0
assert section_files["stdout.txt"]["sha256"] == file_sha("docs/technical-reviews/artifacts/review-20261019-5-13/section-stdout.txt")
assert section_files["stderr.txt"]["content"] == ""
assert section_files["environment.json"]["content"]["guard_events"] == []
parameters = {str(width): TinyLM(ModelConfig(width=width)).description()["parameters"] for width in (8, 16, 32)}
assert parameters == {"8": 6104, "16": 13744, "32": 33632}

layer = torch.nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[0.1, 0.1], [1.0, 1.0]]))
    x = torch.tensor([[1.0, 1.0]])
    float_logits = layer(x)
    quantized_logits = QuantizedLinear(layer, bits=4)(x)
assert int(float_logits.argmax()) == int(quantized_logits.argmax()) == 1
quantization_probe = {"scope": "兩候選的無訓練機制算例；不是attributes checkpoint的測試或量化品質評估。", "candidate_names": ["blue", "other"], "input": x.tolist(), "weights": layer.weight.detach().tolist(), "float_logits": float_logits.tolist(), "int4_logits": quantized_logits.tolist(), "float_prediction": "other", "quantized_prediction": "other", "conclusion": "原模型答錯時，量化不必將答案修正成blue；比較前後答案一致與否並非要求兩者必答對。"}

url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
curl = run(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url])
assert curl["exit_code"] == 0
plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", curl["stdout"])))
excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
assert excerpt in plain and "even when using identical seeds" in plain
external = {"url": url, "version": "PyTorch 2.14 documentation; Reproducibility; Created/Last Updated 2026-05-14", "accessed_on": "2026-10-20", "checked_original": True, "locator": "Reproducibility opening paragraphs; Controlling sources of randomness", "curl_argv": curl["argv"], "exit_code": 0, "response_sha256": digest(curl["stdout"].encode()), "excerpt": excerpt, "inspection_note": "官方說明不保證跨平台、版本或CPU/GPU相同seed得到完全相同結果；這支持『可能不同』，不證明任兩台電腦必有差異。"}

code_paths = scripts + ["tiny_perceptron/training.py", "tiny_perceptron/quantization.py", "tiny_perceptron/model.py"]
code_receipts = []
for path in code_paths:
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    selected = {"choose_device", "seed_everything", "load_checkpoint"} if path.endswith("training.py") else {"quantize_symmetric", "QuantizedLinear"} if path.endswith("quantization.py") else {"main"} if path.endswith(("infer.py", "quantize.py", "course_experiments/run.py")) else set()
    code_receipts.append({"path": path, "sha256": file_sha(path), "excerpts": {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected}})

changes = [
    {"location": "course/training.md 的「Python 怎麼寫」", "statement": "train/evaluate/infer預設auto；固定實驗入口預設CPU；quantize固定CPU；準備資料和下載工具沒有device參數。", "status": "verified", "evidence": "七個入口的--help、argparse AST、choose_device分支、load_checkpoint預設device=cpu與quantize呼叫已核對。分支探測以mock提供可用狀態，不聲稱實測GPU/MPS。", "impact_on_page": "重做指令明寫--device cuda，沒有受到預設CPU的影響。"},
    {"location": "course/training.md 的多模態三題回答後提醒", "statement": "同一固定配方在不同電腦可能產生不同訓練結果，應用完整驗證題判讀。", "status": "verified", "evidence": "PyTorch Reproducibility原文支持跨平台與CPU/GPU不保證完全重現；seed_everything只設seed，沒有提供跨機器一致性的契約。", "impact_on_page": "新增限制與本節既有單seed及小資料限制一致，沒有變更文字實驗配置。"},
    {"location": "course/training.md 的量化前後blue回答比較", "statement": "原模型沒有答blue時，量化模型也不必答blue；示範比較的是前後答案。", "status": "verified", "evidence": "QuantizedLinear只以round/clamp與scale近似權重；沒有答案修復目標。兩候選機制算例的原版與int4均答other，支持不存在必答blue保證。", "impact_on_page": "不涉及本節四格模型、資料或loss結果，重做指令未變。"},
]
continuity = []
for claim in original["claims"]:
    affected = claim["id"] == "c41"
    continuity.append({"claim_id": claim["id"], "status": "verified", "reason": "重讀現行固定配方命令，核對其與兩端Git版本相同；重新執行CLI help並讀解析/dispatch實作。" if affected else "主張原文未變；引用的repo來源SHA與先前核對相同，外部定位與原artifact仍可沿用；本次小節程式輸出亦相同。", "new_evidence": ["training_diff", "cli", "reproduction_command"] if affected else ["source_hash_checks", "artifact_hash_checks", "section_execution"]})
receipt = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-13/recheck-current-01.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu; 模擬裝置可用性僅用於選擇邏輯", "platform": platform.platform()}, "original_report": {"path": ORIGINAL, "sha256": EXPECTED_ORIGINAL}, "source_freeze": {"path": FREEZE, "sha256": FREEZE_SHA}, "source_sha256": SOURCE_SHA, "figure_sha256": {}, "source_hash_checks": sources, "artifact_hash_checks": artifacts, "previous_issues": original["issues"], "resolved_issue_ids": [], "training_versions": {"before": "5ef56bc756f3", "after": "6a7aa1de5227", "before_sha256": digest(old_training["stdout"].encode()), "after_sha256": digest(new_training["stdout"].encode())}, "training_diff": diff, "reproduction_command": {"text": reproduction, "present_before": True, "present_current": True, "explicit_device": "cuda"}, "cli": cli, "device_branch_probe": device_branches, "section_execution": {"invocation": section, "summary": section_summary, "copied_outputs": section_files}, "parameters": parameters, "quantization_probe": quantization_probe, "external_original": external, "code_receipts": code_receipts, "dependency_change_checks": changes, "claim_continuity": continuity, "result": "全部原主張維持核實；改動的參考文件未影響重做指令；重新執行小節輸出與原次相同。", "limitations": ["沒有重跑訓練、GPU/MPS運算或實際attributes量化模型推論。", "只查核參考文件指定差異及本節用到的重做入口，不宣稱重審整份訓練教材。", "原始實測loss沿用先前未變的complete_run紀錄及證據。"]}
with BASE.joinpath("recheck-current-01.json").open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": receipt["result"], "claims": len(continuity), "prior_issues": len(original["issues"]), "changed_dependencies": [item["path"] for item in sources if not item["unchanged"]], "dependency_change_checks": len(changes), "section_exit_code": section["exit_code"]}, ensure_ascii=False))
