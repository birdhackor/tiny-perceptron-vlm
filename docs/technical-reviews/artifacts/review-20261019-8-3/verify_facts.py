"""重算對話目標、資料切分、抽樣曝光量與保存的逐題成績；不重新訓練。"""

import ast
import hashlib
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _style_record
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def function_text(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


tok = ByteTokenizer()
result_path = ROOT / "docs/course-experiments/results/style.json"
recorded = json.loads(result_path.read_text())
results = recorded["results"]
paths = [
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/text.py",
    "scripts/course_experiments/behavior.py",
    "scripts/course_experiments/run.py",
    "course/training.md",
    "docs/course-experiments/results/style.json",
]
report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-3/verify_facts.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
    "repository_head": git("rev-parse", "HEAD").decode().strip(),
    "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in paths},
    "historical_environment": {
        k: recorded[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale")
    },
    "scope": "只執行確定性算例與保存結果的獨立核算；未重跑 GPU 訓練或載入其權重。",
}

examples = {}
for name, answer in {"concise": "4", "vivid": "4，像把兩組積木合在一起再數。", "exercise": "4，共有四個"}.items():
    messages = [{"role": "user", "content": "2+2=?"}, {"role": "assistant", "content": answer}]
    x, y = render_chat(messages, tok)
    selected = y[y != -100].tolist()
    assert selected == tok.encode(answer) + [tok.eos_id], "有效目標必須恰為回答 bytes 與 EOS"
    assert len(x) == len(y) == len(answer.encode()) + 9, "對話位置數須包含角色、提問及位移"
    examples[name] = {
        "answer": answer,
        "characters": [{"character": c, "utf8_bytes": len(c.encode())} for c in answer],
        "answer_bytes": len(answer.encode()),
        "positions": len(x),
        "effective_targets": int((y != -100).sum()),
        "input_ids": x.tolist(),
        "target_ids": y.tolist(),
        "supervised_ids": selected,
    }
assert (examples["concise"]["positions"], examples["concise"]["effective_targets"]) == (10, 2)
assert (examples["vivid"]["positions"], examples["vivid"]["effective_targets"]) == (52, 44)
assert examples["exercise"]["effective_targets"] == 17
report["rendered_examples"] = examples

logits = torch.tensor([[[0.5, -0.5], [0.25, 1.25], [-0.75, 0.75]]], dtype=torch.float64, requires_grad=True)
labels = torch.tensor([[-100, 1, 0]])
loss = masked_loss(logits, labels)
expected = (-torch.log_softmax(logits, -1)[0, 1, 1] - torch.log_softmax(logits, -1)[0, 2, 0]) / 2
assert torch.allclose(loss, expected, atol=1e-12, rtol=0), "平均代價只計未忽略的標籤"
loss.backward()
assert torch.equal(logits.grad[0, 0], torch.zeros(2, dtype=torch.float64)), "被忽略位置的 logits 梯度須為零"
report["masked_loss"] = {
    "labels": labels.tolist(), "loss": loss.item(), "manual_nll": expected.item(),
    "absolute_error": abs(loss.item() - expected.item()), "ignored_position_gradient": logits.grad[0, 0].tolist(),
    "tolerance": 1e-12,
}

parts = split_records(arithmetic_records(), seed=recorded["seed"])
split_summary = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    summary = {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": sha(raw)}
    for key in summary:
        assert summary[key] == results["arithmetic_data"][split][key], "重建的加法切分須符合保存 manifest"
    split_summary[split] = summary
assert "2+2=?" not in {r["messages"][0]["content"] for r in parts["train"]}, "範例提問須未參與訓練"
assert len(parts["train"]) == 49 and len(parts["test"]) == 7
assert all(not ({r["family"] for r in parts[a]} & {r["family"] for r in parts[b]})
           for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
report["arithmetic_splits"] = split_summary

behavior_path = "scripts/course_experiments/behavior.py"
current_behavior = (ROOT / behavior_path).read_text()
historical = None
attempts = []
for commit in git("log", "--format=%H", "--", behavior_path).decode().splitlines():
    raw = git("show", commit + ":" + behavior_path)
    digest = sha(raw)
    attempts.append({"commit": commit, "sha256": digest})
    if digest == recorded["code_sha256"][behavior_path]:
        historical = raw.decode()
        historical_commit = commit
        break
assert historical is not None, "需找到與實驗收據雜湊相符的原始碼"
core_functions = {}
for name in ["_style_record", "_style_evaluations", "run_style"]:
    old = function_text(historical, name)
    new = function_text(current_behavior, name)
    assert old == new, "影響本次實驗的資料與訓練流程須與保存版本相同"
    core_functions[name] = {"sha256": sha(old.encode()), "same_as_current": True, "source": old}
report["historical_behavior_source"] = {
    "lookup_command": "git log --format=%H -- scripts/course_experiments/behavior.py; git show <commit>:scripts/course_experiments/behavior.py",
    "matched_commit": historical_commit, "matched_full_file_sha256": sha(historical.encode()),
    "recorded_full_file_sha256": recorded["code_sha256"][behavior_path], "inspected_candidates": attempts,
    "functions": core_functions,
    "note": "全檔現行雜湊與實驗收據不同；此處核對的資料建構、共用底座複製及無前綴訓練函式逐字相同。成績另由保存的 token IDs 獨立核算。",
}
for path in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"]:
    assert report["source_sha256"][path] == recorded["code_sha256"][path], "共用資料與訓練程式須符合實驗收據版本"
fit_source = inspect.getsource(__import__("scripts.course_experiments.common", fromlist=["fit_lm"]).fit_lm)
assert "batch_size=16" in fit_source and "copy.deepcopy(content)" in core_functions["run_style"]["source"]

def exposure(rows, steps):
    ex = text_examples(rows, "sft", max_length=128)
    lengths = [int((y != -100).sum()) for _, y in ex]
    sampler = random.Random(recorded["seed"])
    sampled = [value for _ in range(steps) for value in sampler.choices(lengths, k=16)]
    return {"steps": steps, "batch_size": 16, "presentations": len(sampled),
            "effective_targets": sum(sampled), "mean_targets": sum(sampled) / len(sampled),
            "extra_answer_digits": sum(value - min(lengths) for value in sampled),
            "target_lengths": sorted(set(lengths))}

base_exposure = exposure(parts["train"], 1000)
assert base_exposure["effective_targets"] == results["content_training"]["effective_tokens"] == 36931
assert records_sha256(parts["train"]) == results["content_training"]["records_sha256"]
report["base_exposure"] = base_exposure
report["style_runs"] = {}
for style in ["concise", "vivid"]:
    run = results["default_style_runs"][style]
    styled = {split: [_style_record(row, style, False) for row in rows] for split, rows in parts.items()}
    for split, rows in styled.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        assert sha(raw) == run["data"][split]["sha256"], "風格資料須符合保存 manifest"
        assert all("style=" not in r["messages"][0]["content"] for r in rows), "提問不得藏有 style 提示"
    assert records_sha256(styled["train"]) == run["training"]["records_sha256"]
    assert run["training"]["steps"] == 450
    counts = exposure(styled["train"], 450)
    assert counts["effective_targets"] == run["training"]["effective_tokens"], "抽樣重算須符合累計有效目標數"
    assert counts["target_lengths"] == ([2, 3] if style == "concise" else [44, 45])
    samples = run["after"]["test"]["samples"]
    checked = []
    for row, sample in zip(styled["test"], samples, strict=True):
        prompt = row["messages"][0]
        assert sample["messages"] == [prompt] and sample["expected"] == row["messages"][1]["content"]
        text = sample["generated"]
        assert sample["generated_ids"] == tok.encode(text) + [tok.eos_id], "可見生成須對應保存的原始 token IDs"
        correct = text.split("，", 1)[0].strip() == str(row["a"] + row["b"])
        style_correct = text.strip().isdigit() if style == "concise" else text.endswith("，像把兩組積木合在一起再數。")
        assert correct == sample["content_correct"] and style_correct == sample["style_correct"]
        checked.append({"question": prompt["content"], "correct_sum": row["a"] + row["b"],
                        "generated": text, "content_correct": correct, "style_correct": style_correct})
    assert len(checked) == 7 and sum(s["style_correct"] for s in checked) == 7
    assert sum(s["content_correct"] for s in checked) == 0
    assert run["after"]["test"]["rubric"][style]["style_correct"] == 7
    assert run["after"]["test"]["rubric"][style]["content_correct"] == 0
    assert checked[0]["question"] == "2+2=?"
    assert checked[0]["generated"] == ("3" if style == "concise" else "3，像把兩組積木合在一起再數。")
    report["style_runs"][style] = {"exposure": counts, "samples": checked,
        "checkpoint": run["training"]["checkpoint"], "records_sha256": run["training"]["records_sha256"],
        "style_correct": 7, "content_correct": 0, "test_records": 7}

base_samples = results["content_evaluation"]["test"]["samples"]
base_checked = []
for row, sample in zip(parts["test"], base_samples, strict=True):
    assert sample["messages"] == [row["messages"][0]]
    assert sample["expected"] == str(row["a"] + row["b"])
    assert sample["generated_ids"] == tok.encode(sample["generated"]) + [tok.eos_id]
    correct = sample["generated"] == str(row["a"] + row["b"])
    base_checked.append({"question": row["messages"][0]["content"], "expected": sample["expected"],
                         "generated": sample["generated"], "correct": correct})
assert len(base_checked) == 7 and not any(s["correct"] for s in base_checked)
report["base_test_samples"] = base_checked

short = report["style_runs"]["concise"]["exposure"]
long = report["style_runs"]["vivid"]["exposure"]
assert (short["effective_targets"], long["effective_targets"]) == (16591, 318991)
assert long["effective_targets"] - short["effective_targets"] == 42 * 7200
assert short["extra_answer_digits"] == long["extra_answer_digits"] == 16591 - 2 * 7200
assert round(short["mean_targets"], 1) == 2.3 and round(long["mean_targets"], 1) == 44.3
report["exposure_derivation"] = {
    "presentations": 450 * 16, "concise_mean": 16591 / 7200, "vivid_mean": 318991 / 7200,
    "fixed_suffix_extra_targets_per_presentation": 42, "fixed_suffix_extra_targets_total": 42 * 7200,
    "two_digit_answer_presentations": short["extra_answer_digits"], "vivid_to_concise_targets_ratio": 318991 / 16591,
    "note": "有效目標含 EOS，重複抽中的資料重複計數；此量不是完整回答筆數，也不代表梯度或品質按比例變化。",
}
training_text = (ROOT / "course/training.md").read_text()
assert "--experiment style --device cuda" in training_text
assert "content.pt" in training_text
report["training_entry"] = {
    "path": "course/training.md", "heading": "把風格、指令遵循與安全拆成不同目標",
    "command_present": True, "content_checkpoint_mentioned": True,
    "style_checkpoint_names_from_code_and_result": [report["style_runs"][s]["checkpoint"] for s in ["concise", "vivid"]],
    "result_paths_checked": ["results.default_style_runs.concise.training.effective_tokens",
        "results.default_style_runs.vivid.training.effective_tokens",
        "results.default_style_runs.concise.after.test.samples", "results.default_style_runs.vivid.after.test.samples"],
}
report["result"] = "全部確定性核算與保存紀錄的逐題核對通過"
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
