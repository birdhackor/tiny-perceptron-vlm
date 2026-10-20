"""複查教材依賴、保全原始證據，並重跑本節示例。"""
import ast
import contextlib
import hashlib
import html
import io
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.training import choose_device

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def run(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)
    return {"command_argv": argv, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def code_run(code):
    namespace = {}
    output = io.StringIO()
    error = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(code, "section-example", "exec"), namespace)
        except Exception as e:
            error = type(e).__name__
    return {"stdout": output.getvalue(), "exception": error, "scores": namespace.get("scores")}

original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_7-13/technical-7-13-current-01.json"
original_raw = original_path.read_bytes()
assert sha(original_raw) == "89139319f818a7d3babd9e0af9240d0a69753cb332fa76988964edccced9ab18"
original = json.loads(original_raw)
assert original["issues"] == []
old_artifacts = []
for artifact in original["artifacts"]:
    raw = (ROOT / artifact["path"]).read_bytes()
    actual = sha(raw)
    assert actual == artifact["sha256"]
    old_artifacts.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_check = A / "self-check-current-01.txt"
old_artifacts.append({"path": str(old_check.relative_to(ROOT)), "sha256": sha(old_check.read_bytes()), "unchanged": True})
source_checks = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = sha((ROOT / source["path"]).read_bytes())
    source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"],
                          "current_sha256": actual, "unchanged": actual == source["sha256"]})
assert [s["path"] for s in source_checks if not s["unchanged"]] == ["course/training.md"]
training = ROOT / "course/training.md"
assert sha(training.read_bytes()) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
diff = run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
old_training = run(["git", "show", "ecf6dec71280:course/training.md"])
new_training = run(["git", "show", "c6b9339d2595:course/training.md"])
assert all(c["exit_code"] == 0 for c in [diff, old_training, new_training])
assert sha(old_training["stdout"].encode()) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert new_training["stdout"].encode() == training.read_bytes()
recipe_pattern = r'<details id="fixed-sft-recipe" markdown="1">.*?</details>'
old_recipe = re.search(recipe_pattern, old_training["stdout"], re.S)[0]
current_recipe = re.search(recipe_pattern, training.read_text(), re.S)[0]
assert old_recipe == current_recipe
snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/7.13.md"
chapter = (ROOT / "course/chapters/07.md").read_bytes()
start = re.search(rb"(?m)^## 7\.13 ", chapter).start()
end_match = re.search(rb"(?m)^## ", chapter[start + 1:])
end = start + 1 + end_match.start() if end_match else len(chapter)
section = chapter[start:end]
assert section == snapshot.read_bytes()
assert sha(section) == original["source_sha256"]
freeze = ROOT / "docs/course-review-20261019/freeze-07.json"
assert sha(freeze.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
section_command = run([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/07.md#7.13", "--execute"])
assert section_command["exit_code"] == 0
run_summary = json.loads(section_command["stdout"])
assert run_summary["source_sha256"] == original["source_sha256"]
assert run_summary["svg_references"] == 0
run_dir = Path(run_summary["artifact_directory"])
copied_files = []
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"]:
    raw = (run_dir / name).read_bytes()
    copied_files.append({"name": name, "sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()})
code = re.search(r"```python\n(.*?)```", section.decode(), re.S)[1]
correct = code.replace("3}", "2}")
fixed = correct.replace('and not scores["1+1內容正確"]', 'and scores["1+1內容正確"]')
renamed = correct.replace('reply = \'{"answer": 2}\'', 'reply = \'{"result": 2}\'')
variants = {"original": code_run(code), "answer_two_original_assertion": code_run(correct),
            "answer_two_both_true_assertion": code_run(fixed), "renamed_result": code_run(renamed)}
previous_verification = json.loads((A / "verification.json").read_text())
assert variants == previous_verification["example_variants"]
record = json.loads((ROOT / "docs/course-experiments/results/sft.json").read_bytes())
tok = ByteTokenizer()
test = record["results"]["after"]["test"]
recomputed = []
for sample in test["samples"]:
    ids = sample["generated_ids"]
    raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    exact = raw == tok.encode(sample["expected"])
    eos = tok.eos_id in ids
    assert exact is sample["exact"] and eos is sample["eos"]
    recomputed.append({"expected": sample["expected"], "exact": exact, "eos": eos})
assert sum(s["exact"] for s in recomputed) == test["matches"] == 5
assert len(recomputed) == test["records"] == 10
assert round(record["elapsed_seconds"], 1) == 31.7
assert record["results"]["training"]["steps"] == 900
assert record["results"]["pretrain_then_sft"]["after_sft"]["test"]["matches"] == 7

files = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py", "scripts/prepare_data.py",
         "scripts/fetch_training_assets.py", "scripts/course_experiments/run.py", "tiny_perceptron/training.py"]
cli_commands = []
source_inspection = []
for name in files:
    body = (ROOT / name).read_text()
    tree = ast.parse(body)
    options = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if "--device" in flags:
                default = next((ast.literal_eval(k.value) for k in node.keywords if k.arg == "default"), None)
                options.append({"flags": flags, "default": default, "source": ast.get_source_segment(body, node)})
    source_inspection.append({"path": name, "sha256": sha((ROOT / name).read_bytes()), "device_options": options,
                              "device_calls": [line for line in body.splitlines() if any(x in line for x in ("choose_device(", "load_checkpoint(", 'map_location="cpu"', 'return torch.device(', 'torch.cuda.is_available()', 'torch.backends.mps.is_available()'))]})
    if name.startswith("scripts/"):
        result = run([str(ROOT / ".venv/bin/python"), name, "--help"])
        assert result["exit_code"] == 0
        cli_commands.append(result)
lookup = {s["path"]: s for s in source_inspection}
for name in ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py"]:
    assert lookup[name]["device_options"][0]["default"] == "auto"
assert lookup["scripts/course_experiments/run.py"]["device_options"][0]["default"] == "cpu"
for name in ["scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"]:
    assert lookup[name]["device_options"] == []
    help_result = next(c for c in cli_commands if c["command_argv"][1] == name)
    assert "--device" not in help_result["stdout"]
selection = []
for cuda, mps, requested, expected in [(True, True, "auto", "cuda"), (False, True, "auto", "mps"),
                                        (False, False, "auto", "cpu"), (True, True, "cpu", "cpu")]:
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        observed = str(choose_device(requested))
    assert observed == expected
    selection.append({"mock_cuda_available": cuda, "mock_mps_available": mps, "requested": requested, "expected": expected, "observed": observed})

pytorch_receipt_path = A / "recheck-evidence-current-01.json"
pytorch_receipt_sha = sha(pytorch_receipt_path.read_bytes())
pytorch = json.loads(pytorch_receipt_path.read_text())["pytorch_original"]

import random
import scripts.train as train_module
import scripts.evaluate as evaluate_module
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import split_records
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

torch.set_num_threads(1)
random.seed(42)
torch.manual_seed(42)
modal_input_checks = []
choice_original = random.choice

class InputSpy:
    def __init__(self):
        self.inputs = []

    def __call__(self, ids, labels, image=None, waveform=None):
        self.inputs.append({"ids": ids.tolist(), "answer": tok.decode(labels[labels != -100].tolist()),
                            "image_shape": list(image.shape) if image is not None else None,
                            "waveform_shape": list(waveform.shape) if waveform is not None else None})
        return {"logits": torch.zeros(len(ids), tok.vocab_size), "labels": labels, "auxiliary": torch.tensor(0.0)}

for task in ("vision", "audio", "joint"):
    args = train_module.parser().parse_args(["--task", task, "--checkpoint", "checkpoints/attributes.pt", "--train", "--steps", "500"])
    assert args.data is None
    assert train_module.prepare_examples(args, tok) is None
    spy = InputSpy()
    with patch.object(train_module.random, "choice", wraps=choice_original) as chosen:
        losses = [float(train_module.modal_loss(spy, task, tok, "cpu")) for _ in range(2)]
        assert chosen.call_count == 6
        draw_choices = [list(call.args[0]) for call in chosen.call_args_list]
    modal_input_checks.append({"task": task, "data_argument": None, "prepared_examples": None,
                               "random_choice_calls_for_two_examples": chosen.call_count,
                               "choice_populations": draw_choices, "inputs": spy.inputs, "losses": losses,
                               "scope": "實際呼叫modal_loss建構輸入，使用spy模型與均勻logits；沒有訓練或測量模型品質。"})

model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=1, heads=1)))
rejection = None
with patch.object(evaluate_module, "load_checkpoint", return_value=(model, {"format_version": "multimodal-v1"})), patch.object(sys, "argv", ["evaluate.py", "in-memory-placeholder.pt", "--device", "cpu"]):
    try:
        evaluate_module.main()
    except ValueError as error:
        rejection = {"type": type(error).__name__, "message": str(error)}
assert rejection is not None and "TinyLM" in rejection["message"]
rejection["scope"] = "使用實際MultiModalLM物件，mock checkpoint loader；觸發evaluate.main型別守衛，未寫模型檔或評估報告。"

parts = split_records(generate_records("attributes-sft"), seed=42)
vqa_path = ROOT / "docs/course-experiments/results/vqa.json"
vqa = json.loads(vqa_path.read_text())
for name, rows in parts.items():
    assert len(rows) == vqa["results"]["text_data"][name]["count"]
    assert sha(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()) == vqa["results"]["text_data"][name]["sha256"]
expected_questions = [(row["messages"][:-1], row["messages"][-1]["content"]) for row in parts["test"]]
vqa_variants = []
for name in ("projector_only", "partial", "all", "all_replay"):
    variant = vqa["results"]["variants"][name]
    before, after = variant["text_before"], variant["text_after"]
    for phase in (before, after):
        questions = [(sample["messages"], sample["expected"]) for sample in phase["samples"]]
        assert questions == expected_questions
        recomputed_count = 0
        for sample in phase["samples"]:
            ids = sample["generated_ids"]
            raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact = raw == tok.encode(sample["expected"])
            assert exact is sample["exact"]
            recomputed_count += exact
        assert recomputed_count == phase["matches"]
        assert phase["records"] == len(phase["samples"]) == 10
    vqa_variants.append({"variant": name, "same_heldout_questions_before_after": True,
                         "test_records": 10, "before_matches": before["matches"], "after_matches": after["matches"],
                         "text_probe_is_training_diagnostic": variant["text_probe_is_training_diagnostic"]})

new_source_nodes = {
    "scripts/train.py": {"main", "modal_loss", "prepare_examples"},
    "scripts/evaluate.py": {"main"},
    "scripts/course_experiments/modalities.py": {"run_vqa", "_loss_fn"},
    "tiny_perceptron/multimodal.py": {"MultiModalLM"},
}
new_source_inspection = []
for name, identifiers in new_source_nodes.items():
    text = (ROOT / name).read_text()
    parsed = ast.parse(text)
    selected = [{"identifier": node.name, "source": ast.get_source_segment(text, node)}
                for node in ast.walk(parsed) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in identifiers]
    new_source_inspection.append({"path": name, "sha256": sha(text.encode()), "selected_code": selected})
modalities_hash = sha((ROOT / "scripts/course_experiments/modalities.py").read_bytes())
assert modalities_hash == vqa["code_sha256"]["scripts/course_experiments/modalities.py"]
extra_hash_paths = ["scripts/infer_modal.py", "docs/course-experiments/results/vqa.json", "course/chapters/11.md"]
new_source_hashes = [{"path": name, "sha256": sha((ROOT / name).read_bytes())} for name in extra_hash_paths]
modal_dependency_checks = {
    "modal_inputs": modal_input_checks, "evaluate_multimodal_rejection": rejection,
    "source_inspection": new_source_inspection, "extra_source_hashes": new_source_hashes,
    "vqa": {"path": str(vqa_path.relative_to(ROOT)), "sha256": sha(vqa_path.read_bytes()),
            "recorded_device": vqa["device"], "recorded_seed": vqa["seed"],
            "text_split_counts": {name: len(rows) for name, rows in parts.items()},
            "text_test_sha256": vqa["results"]["text_data"]["test"]["sha256"],
            "variants": vqa_variants, "recorded_code_matches_current": True,
            "scope": "四條延續已訓練projector底座的支線做文字前後測；direct_vqa支線沒有text_before/text_after，不把此主張外推給它。只重算保存紀錄，未重跑VQA訓練。"},
    "local_validation_inspection": "已讀train.main完整流程：這三條沒有--data的路線以modal_loss即時建構每筆例子；report保存訓練history而非固定驗證集。infer_modal的三條命令各生成單題答案，沒有在多模態訓練前後對照純文字題。",
    "linked_explanation": "已讀course/chapters/11.md的『新圖文能力進步時，舊文字能力有沒有退步？』，明確解釋四版text_before/text_after與同一批十題測試；training.md的固定實驗標題存在。"
}

for old in old_artifacts:
    assert sha((ROOT / old["path"]).read_bytes()) == old["sha256"]
assert sha(original_path.read_bytes()) == sha(original_raw)
evidence = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-13/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "previous_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_raw), "issues": original["issues"]},
    "source_sha256": sha(section), "source_freeze_sha256": sha(freeze.read_bytes()),
    "previous_source_hash_checks": source_checks, "previous_artifact_hash_checks": old_artifacts,
    "training_diff": diff, "training_versions": {"old_revision": "ecf6dec71280", "new_revision": "c6b9339d2595", "old_sha256": sha(old_training["stdout"].encode()), "current_sha256": sha(training.read_bytes())},
    "fixed_sft_recipe": {"unchanged": old_recipe == current_recipe, "sha256": sha(current_recipe.encode()), "current_text": current_recipe},
    "section_execution": {"command_receipt": section_command, "copied_from": str(run_dir.relative_to(ROOT)), "copied_files": copied_files},
    "example_variants": variants,
    "sample_recomputation": {"samples": recomputed, "matches": 5, "records": 10, "eos_count": sum(s["eos"] for s in recomputed), "pretrain_sft_matches": 7, "steps": 900, "elapsed_rounded": 31.7},
    "cli_commands": cli_commands, "source_inspection": source_inspection, "mocked_device_selection": selection,
    "pytorch_original": pytorch,
    "pytorch_receipt_reused_from": {"path": str(pytorch_receipt_path.relative_to(ROOT)), "sha256": pytorch_receipt_sha},
    "modal_dependency_checks": modal_dependency_checks,
    "scope": "不重跑訓練、量化匯出或實際下載；裝置選擇以mock可用性測試，不代表在實體CUDA/MPS裝置執行。舊證據中training.md的hash是歷史收據，此次以current_sha256及未變的fixed-sft-recipe作現行依據。",
    "result": "原報告及舊artifact保持原樣；所有頁面主張仍有支持，固定SFT配方未改動；新的多模態資料與前後測說明均有程式及原始實驗支持。"
}
with (A / "recheck-evidence-current-02.json").open("x", encoding="utf-8") as target:
    json.dump(evidence, target, ensure_ascii=False, indent=2)
    target.write("\n")
print(json.dumps({"result": evidence["result"], "section_exit_code": section_command["exit_code"], "changed_repo_sources": [s["path"] for s in source_checks if not s["unchanged"]], "test": evidence["sample_recomputation"]}, ensure_ascii=False))
