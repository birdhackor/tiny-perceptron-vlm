"""核對凍結教材、配對資料與已保存生成紀錄；不重新訓練模型。"""

import ast
import copy
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256, split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import generate
from types import SimpleNamespace


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def jsonl_hash(rows):
    return digest("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())


def main():
    result_path = ROOT / "docs/course-experiments/results/style.json"
    raw = result_path.read_bytes()
    receipt = json.loads(raw)
    saved = receipt["results"]
    code_paths = [
        "scripts/course_experiments/behavior.py",
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/text.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/data.py",
    ]
    provenance = {}
    for name in code_paths:
        actual = digest((ROOT / name).read_bytes())
        provenance[name] = {
            "current_sha256": actual,
            "recorded_sha256": receipt["code_sha256"][name],
            "current_matches_recorded": actual == receipt["code_sha256"][name],
        }
        if name != "scripts/course_experiments/behavior.py":
            require(provenance[name]["current_matches_recorded"], "執行程式與實驗紀錄的 SHA-256 不符：" + name)

    old_name = "scripts/course_experiments/behavior.py"
    revision = "6c84f3fb1af8d7f316bdfe3e40f95064357456f5"
    argv = ["git", "show", revision + ":" + old_name]
    historical_raw = subprocess.check_output(argv, cwd=ROOT)
    require(digest(historical_raw) == receipt["code_sha256"][old_name], "找到的程式版本與實驗紀錄的 SHA-256 不符")
    historical = historical_raw.decode()
    parsed = ast.parse(historical)
    selected = {"_conversation", "_style_record", "_style_metrics"}
    namespace = {"json": json}
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), old_name, "exec"), namespace)
    snippets = {
        node.name: ast.get_source_segment(historical, node)
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in selected | {"run_style", "_style_evaluations"}
    }
    provenance[old_name].update(
        inspected_revision=revision,
        historical_sha256=digest(historical_raw),
        historical_command_argv=argv,
        inspected_functions=snippets,
    )

    freeze_path = ROOT / "docs/course-review-20261019/freeze-01/sources/8.4.md"
    section = freeze_path.read_bytes()
    require(digest(section) == "7e6172cef153b4a1a91062d399817892131f7301bb4742ddb42a5318bb7f9f8d", "凍結原文 SHA-256 不符")
    full = (ROOT / "course/chapters/08.md").read_bytes()
    section_current = re.search(rb"(?ms)^## 8\.4 .*?(?=^## |\Z)", full).group()
    require(section_current == section, "實際執行的原文與凍結原文不同")
    fence = re.search(rb"```python\n(.*?)```", section, re.S).group(1).decode()
    original_namespace = {}
    capture = io.StringIO()
    with redirect_stdout(capture):
        exec(compile(fence, "frozen-section", "exec"), original_namespace)
    rows = original_namespace["records"]
    expected_rows = [
        {"提問": "style=concise; 1+1=?", "回答": "2"},
        {"提問": "style=vivid; 1+1=?", "回答": "2，像把兩組積木合在一起再數。"},
        {"提問": "style=concise; 2+2=?", "回答": "4"},
        {"提問": "style=vivid; 2+2=?", "回答": "4，像把兩組積木合在一起再數。"},
    ]
    require(rows == expected_rows, "配對程式的輸出不符")
    require('train' not in fence and 'torch' not in fence, "示範程式不應訓練模型")
    altered = fence.replace('["concise", "vivid"]', '["concise"]')
    concise_namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(altered, "concise-exercise", "exec"), concise_namespace)
    require(concise_namespace["records"] == expected_rows[::2], "練習的兩筆資料不符")
    value = json.loads('{"answer": 4}')
    require(set(value) == {"answer"} and type(value["answer"]) is int and value["answer"] == 4, "JSON 算例不符")

    arithmetic = split_records(arithmetic_records(), seed=receipt["seed"])
    style_record = namespace["_style_record"]
    conditional = {
        split: [style_record(row, style, True) for row in part for style in ("concise", "vivid", "json")]
        for split, part in arithmetic.items()
    }
    dates = []
    for day in range(1, 25):
        date = f"2026-10-{day:02d}"
        dates.extend([
            namespace["_conversation"](f"task=date;date={date};confirm", "已確認" + date + "。", "date-" + date, style="clarification"),
            namespace["_conversation"](f"task=date;id={day};date=?;confirm", "請提供日期。", "date-" + date, style="clarification"),
        ])
    date_parts = split_records(dates, seed=receipt["seed"])
    conditional = {split: part + date_parts[split] for split, part in conditional.items()}
    data_checks = {}
    for split in arithmetic:
        for label, part, manifest in [
            ("arithmetic", arithmetic[split], saved["arithmetic_data"][split]),
            ("conditional", conditional[split], saved["data"][split]),
        ]:
            computed = {"records": len(part), "sha256": jsonl_hash(part)}
            require(computed["records"] == manifest["records"] and computed["sha256"] == manifest["sha256"], "重建資料與保存紀錄不符：" + label + "/" + split)
            data_checks[label + "/" + split] = computed
    train_families = {row["family"] for row in arithmetic["train"]}
    test_families = {row["family"] for row in arithmetic["test"]}
    require(not train_families & test_families and "2+2" not in train_families, "測試加法題不可出現在訓練題家族")
    require(records_sha256(conditional["train"]) == saved["training"]["records_sha256"], "訓練輸入雜湊不符")
    require(saved["training"]["steps"] == 1000 and receipt["step_scale"] == 1.0, "保存紀錄的更新步數不符")
    require(saved["training"]["records"] == len(conditional["train"]) == 185, "訓練筆數不符")
    require(len(arithmetic["train"]) == 49, "加法訓練題數不符")
    require([len(arithmetic[p]) for p in ["validation", "test"]] == [8, 7], "加法留出題數不符")
    require([len(date_parts[p]) for p in ["validation", "test"]] == [4, 6], "日期留出題數不符")

    tok = ByteTokenizer()
    score_checks = {}
    all_samples = {}
    for split in ["validation", "test"]:
        report = saved["after"][split]
        samples = report["samples"]
        require(len(samples) == len(conditional[split]), "逐題生成數量不符")
        reconstructed = namespace["_style_metrics"](copy.deepcopy(report), conditional[split])
        require(reconstructed["rubric"] == report["rubric"], "評分程式重算結果不符")
        own = {style: {"records": 0, "style_correct": 0, "content_correct": 0, "eos": 0} for style in ["concise", "vivid", "json"]}
        selected_samples = []
        for row, sample in zip(conditional[split], samples, strict=True):
            require(sample["messages"] == row["messages"][:-1], "生成的輸入與配對資料不同")
            require(sample["expected"] == row["messages"][-1]["content"], "生成的目標與配對資料不同")
            ids = sample["generated_ids"]
            before_eos = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            require(tok.decode(before_eos) == sample["generated"], "token 解碼與可見文字不同")
            if "a" not in row:
                continue
            require(ids[-1] == tok.eos_id and tok.eos_id not in ids[:-1], "加法生成未在第一個 EOS 停止")
            require(all(8 <= token < 264 for token in before_eos), "加法可見文字前存在隱藏控制 token")
            text = sample["generated"]
            style = row["style"]
            answer = row["a"] + row["b"]
            if style == "json":
                obj = json.loads(text)
                style_ok = isinstance(obj, dict) and len(obj) == 1 and "answer" in obj and type(obj["answer"]) is int
                correct = style_ok and obj["answer"] == answer
            elif style == "vivid":
                style_ok = "像把兩組積木合在一起再數" in text
                correct = text.partition("，")[0].strip() == str(answer)
            else:
                style_ok = bool(re.fullmatch(r"[0-9]+", text.strip()))
                correct = text.strip() == str(answer)
            require(style_ok == sample["style_correct"] and correct == sample["content_correct"], "獨立重算的逐題分數不符")
            own[style]["records"] += 1
            own[style]["style_correct"] += int(style_ok)
            own[style]["content_correct"] += int(correct)
            own[style]["eos"] += int(sample["eos"])
            selected_samples.append({"input": sample["messages"][0]["content"], "expected": sample["expected"], "generated": text, "eos": sample["eos"], "style_correct": style_ok, "content_correct": correct})
        for style, count in own.items():
            for key in ["records", "style_correct", "content_correct"]:
                require(count[key] == report["rubric"][style][key], "獨立加總的分數不符")
        score_checks[split] = own
        all_samples[split] = selected_samples

    target = [sample for sample in all_samples["test"] if sample["input"].endswith("; 2+2=?")]
    require([s["generated"] for s in target] == ["3", "3，像把兩組積木合在一起再數。", '{"answer": 3}'], "正文表格的逐題回答不符")
    for style in ["concise", "vivid", "json"]:
        require(score_checks["test"][style] == {"records": 7, "style_correct": 7, "content_correct": 0, "eos": 7}, "測試表格的計數不符")
        probe = saved["prompt_only_comparison_same_weights"][style]["samples"]
        samples = [s for s in saved["after"]["test"]["samples"] if s.get("style") == style]
        require([s["generated_ids"] for s in probe] == [s["generated_ids"] for s in samples], "同參數提示對照的生成不同")

    class FixedScores(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(max_length=128)
            self.prefixes = []

        def forward(self, ids, cache=None):
            self.prefixes.append(ids.tolist())
            logits = torch.full((1, ids.shape[1], 264), -9.0)
            logits[0, -1, 59 if len(self.prefixes) == 1 else tok.eos_id] = 9.0
            return {"logits": logits, "cache": None}

    probe_model = FixedScores()
    parameter_before = probe_model.marker.detach().clone()
    generated = generate(probe_model, torch.tensor([[tok.bos_id]]), max_new_tokens=8, eos_id=tok.eos_id)
    require(generated.tolist() == [[tok.bos_id, 59, tok.eos_id]], "最高分生成與 EOS 停止不符")
    require(probe_model.prefixes == [[[tok.bos_id]], [[tok.bos_id, 59]]], "生成的 token 未接回前文")
    require(torch.equal(probe_model.marker.detach(), parameter_before), "生成改動了模型參數")
    require(probe_model.training, "生成沒有還原模型模式")
    greedy_probe = {"highest_score_tokens": [59, tok.eos_id], "generated": generated.tolist(), "prefixes_seen": probe_model.prefixes, "parameters_unchanged": True, "eos_stops_generation": True, "scope": "固定 logits 的 CPU 模型驗證專案 generate 函式控制流程；不是重新執行已保存的訓練模型。"}

    short = {"1+1=?": "2", "2+2=?": "4，像把兩組積木合在一起再數。"}
    require(len(short) == 2, "題目查表算例不符")
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-4/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
        "source_sha256": digest(section),
        "experiment_source": {"path": str(result_path.relative_to(ROOT)), "sha256": digest(raw), "recorded_revision": receipt["revision"], "device": receipt["device"], "gpu": receipt["gpu"], "python": receipt["python_version"], "torch": receipt["torch_version"], "seed": receipt["seed"], "step_scale": receipt["step_scale"]},
        "code_provenance": provenance,
        "greedy_generation_probe": greedy_probe,
        "section_demo": {"stdout": capture.getvalue(), "records": rows, "concise_exercise": concise_namespace["records"], "json_example": value},
        "reconstructed_data": data_checks,
        "training_record": {"steps": saved["training"]["steps"], "records": saved["training"]["records"], "effective_tokens": saved["training"]["effective_tokens"], "records_sha256": saved["training"]["records_sha256"], "checkpoint": saved["checkpoint"], "content_checkpoint": saved["content_checkpoint"]},
        "date_examples": dates[:2],
        "training_test_arithmetic_families_disjoint": not train_families & test_families,
        "scores_recomputed": score_checks,
        "saved_arithmetic_generations": all_samples,
        "target_generations": target,
        "prompt_only_same_weights_outputs_match": True,
        "question_only_counterexample": {"lookup": short, "details": "每個題目只有一種目標時，直接按題目查表即可擬合兩筆資料，完全不用讀風格條件；這只是可行解的存在性，不預測最佳化會選哪個解。"},
        "result": "全部核對成立",
        "limits": "核對保存的單次 seed=42 CUDA 實驗紀錄、資料雜湊與生成 token；沒有下載權重、重新生成或重新訓練，不能外推新風格、自由比喻、其他 seed 或其他模型的能力。",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
