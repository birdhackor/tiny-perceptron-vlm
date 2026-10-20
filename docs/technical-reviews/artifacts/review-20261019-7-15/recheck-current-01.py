"""核對來源變更、固定配方與入口裝置設定；不訓練模型。"""
import ast
import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import run
from tiny_perceptron.training import choose_device, load_checkpoint

ARTIFACTS = ROOT / "docs/technical-reviews/artifacts/review-20261019-7-15"
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_7-15/technical-7-15-original-01.json"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_bytes())

def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

original = read_json(ROOT / ORIGINAL)
assert digest((ROOT / ORIGINAL).read_bytes()) == "247a175afb9d57268bbbe7eb5b19b01e9f2cd32d6fe75d14161e9895e4efbd23"
assert original["issues"] == []
sources = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    current = digest((ROOT / source["path"]).read_bytes())
    changed = current != source["sha256"]
    if source["path"] == "course/training.md":
        assert current == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    else:
        assert not changed, "另有引用的實作或資料檔變更"
    sources.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current, "changed": changed})
preserved = []
for artifact in original["artifacts"]:
    current = digest((ROOT / artifact["path"]).read_bytes())
    assert current == artifact["sha256"], "原始證據必須保持相同"
    preserved.append({"path": artifact["path"], "sha256": current})

old_training = subprocess.check_output(["git", "show", "5ef56bc756f3:course/training.md"], cwd=ROOT)
new_training = (ROOT / "course/training.md").read_bytes()
pinned_new_training = subprocess.check_output(["git", "show", "6a7aa1de5227:course/training.md"], cwd=ROOT)
assert digest(old_training) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
assert new_training == pinned_new_training

def recipe(raw):
    start = raw.index(b'<details id="fixed-sft-recipe"')
    end = raw.index(b'</details>', start) + len(b'</details>')
    return raw[start:end]

assert recipe(old_training) == recipe(new_training), "本頁引用的固定配方必須重新比較"
freeze_path = "docs/course-review-20261019/freeze-06.json"
assert digest((ROOT / freeze_path).read_bytes()) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
freeze = read_json(ROOT / freeze_path)
page = next(item for item in freeze["pages"] if item["page_id"] == "7.15")
frozen = (ROOT / page["snapshot"]).read_bytes()
raw = (ROOT / "course/chapters/07.md").read_bytes()
start = re.search(rb"(?m)^## 7\.15 ", raw).start()
next_header = re.search(rb"(?m)^## ", raw[start + 1:])
end = start + 1 + next_header.start() if next_header else len(raw)
assert raw[start:end] == frozen
assert digest(frozen) == original["source_sha256"] == page["source_sha256"]
assert page["figures_sha256"] == {}

old_audit = read_json(ARTIFACTS / "audit-output.json")
new_audit = read_json(ARTIFACTS / "audit-output-current-01.json")
assert new_audit["checks_passed"] is True
fields = ["section_stdout", "exercise", "data_regeneration", "arithmetic_population", "evaluations", "baseline_evaluation_identical", "baseline_checkpoint", "training", "digit_with_eos_example", "record_environment", "cli_help", "cli_list_assets_stdout"]
for field in fields:
    assert old_audit[field] == new_audit[field], "重算結果必須與保留證據逐項比較"

class DispatchObserved(Exception):
    pass

dispatches = []
for flags, expected in [([], "cpu"), (["--device", "cuda"], "cuda")]:
    def capture(experiment_id, device, output, dependencies, assets, **kwargs):
        dispatches.append({"argv": ["--experiment", "sft_ablation"] + flags, "experiment_id": experiment_id, "device": device, "output": str(output.relative_to(ROOT)), "dependencies": str(dependencies.relative_to(ROOT))})
        raise DispatchObserved
    with patch.object(sys, "argv", ["run.py", "--experiment", "sft_ablation"] + flags), patch.object(run, "execute", capture):
        try:
            run.main()
        except DispatchObserved:
            pass
    assert dispatches[-1]["device"] == expected

selection = []
for cuda_available, mps_available, expected in [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]:
    with patch.object(torch.cuda, "is_available", return_value=cuda_available), patch.object(torch.backends.mps, "is_available", return_value=mps_available):
        observed = str(choose_device())
    assert observed == expected
    selection.append({"cuda_available_mocked": cuda_available, "mps_available_mocked": mps_available, "selected": observed})
assert inspect.signature(load_checkpoint).parameters["device"].default == "cpu"

cli_paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "tiny_perceptron/training.py"]
code_evidence = []
for path in cli_paths:
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    arguments = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
    device_arguments = [node for node in arguments if any(isinstance(arg, ast.Constant) and arg.value == "--device" for arg in node.args)]
    if path in ("scripts/train.py", "scripts/evaluate.py", "scripts/infer.py"):
        assert len(device_arguments) == 1
        assert any(kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value == "auto" for kw in device_arguments[0].keywords)
    elif path != "tiny_perceptron/training.py":
        assert not device_arguments
    selected = []
    for line in text.splitlines():
        if "choose_device(args.device)" in line or '"--device"' in line or "load_checkpoint(args.checkpoint)" in line:
            selected.append(line.strip())
    code_evidence.append({"path": path, "sha256": digest((ROOT / path).read_bytes()), "device_argument_count": len(device_arguments), "selected_source": selected})
help_outputs = []
for path in ("scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"):
    result = command([sys.executable, path, "--help"])
    assert "--device" not in result["stdout"]
    help_outputs.append(result)

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-15/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "checks_passed": True,
    "original_report": {"path": ORIGINAL, "sha256": digest((ROOT / ORIGINAL).read_bytes())},
    "original_issues": [],
    "resolved_issue_ids": [],
    "source_inventory": sources,
    "preserved_artifacts": preserved,
    "diff": command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"]),
    "fixed_recipe": {"identical": True, "previous_sha256": digest(recipe(old_training)), "current_sha256": digest(recipe(new_training)), "locator": 'details id="fixed-sft-recipe"'},
    "current_page": {"source_sha256": digest(frozen), "matches_frozen": True, "figure_sha256": {}},
    "recomputed_result_fields_identical": fields,
    "all_claims_retained": [{"claim_id": item["id"], "kind": item["kind"], "status": "verified", "basis": "來源與既有證據未變；本回合重算資料、計分和程式結果，並核對所引固定配方逐位元相同。"} for item in original["claims"]],
    "dispatch_probe": {"training_invoked": False, "execute_was_intercepted": True, "dispatches": dispatches},
    "device_selection_probe": {"hardware_availability_mocked": True, "results": selection, "hardware_execution_tested": False},
    "load_checkpoint_default_device": "cpu",
    "cli_code_evidence": code_evidence,
    "choose_device_source": inspect.getsource(choose_device),
    "cli_help_receipts": help_outputs,
    "changed_passages": [
        {"locator": "Python 怎麼寫", "relevant_claim_ids": ["c14"], "status": "verified", "details": "限定 train/evaluate/infer 的 auto 裝置說明符合現行參數和 choose_device；run 預設 CPU，明寫 --device cuda 時傳入 CUDA；quantize 預設 CPU，三個指定工具皆無 --device。固定 SFT 配方未改，仍明寫 CUDA。"},
        {"locator": "圖音聯合推論的預期答案段落", "relevant_claim_ids": [], "status": "outside_scope", "details": "新增跨電腦訓練結果及三題示例的限制，位於其他配方，不在本頁或本頁引用的 fixed-sft-recipe 內；沒有更改本頁實驗結果或其使用方式。未在本次複查重審該配方的品質。"},
        {"locator": "量化後預期 blue 的段落", "relevant_claim_ids": [], "status": "outside_scope", "details": "新增單題量化前後比較的限制，位於量化練習，不在本頁或本頁引用的 fixed-sft-recipe 內；沒有更改本頁主張。未在本次複查重新載入量化模型。"},
    ],
    "independence": "只讀自己原報告、其明確引用的證據及本回合新產生的工具輸出檔；未讀其他角色或先前審閱紀錄。",
    "limitations": "未載入權重重跑 GPU 訓練、生成、logits/loss 或 L4 計時；入口裝置探針以 mock 驗證分支，沒有宣稱實際執行 CUDA/MPS。其他配方新增的內容不作獨立裁決。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
