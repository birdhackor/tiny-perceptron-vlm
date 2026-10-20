"""核對加權算例、既有生成紀錄、目標遮罩與平均代價的分母。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import loss_sum


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_excerpt(path, name):
    text = path.read_text()
    tree = ast.parse(text)
    item = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    lines = text.splitlines()
    return "\n".join(lines[item.lineno - 1 : item.end_lineno])


source = ROOT / "docs/course-review-20261019/freeze-01/sources/8.7.md"
result_path = ROOT / "docs/course-experiments/results/style.json"
record = json.loads(result_path.read_bytes())
tok = ByteTokenizer()
code = source.read_text().split("```python\n", 1)[1].split("```", 1)[0]
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(source), "exec"), namespace)

scores = []
for weight in [0.2, 2.0, 0.5]:
    totals = [weight * row["風格"] + row["正確"] for row in namespace["answers"]]
    scores.append(
        {
            "style_weight": weight,
            "totals": totals,
            "display": [round(t, 1) for t in totals],
            "short_minus_wrong": totals[0] - totals[1],
        }
    )

empirical = {}
for style in ["concise", "vivid"]:
    run = record["results"]["default_style_runs"][style]
    test = run["after"]["test"]
    samples = []
    for sample in test["samples"]:
        question = sample["messages"][0]["content"]
        left, right = map(int, question.removesuffix("=?").split("+"))
        expected_number = str(left + right)
        expected = expected_number + ("，像把兩組積木合在一起再數。" if style == "vivid" else "")
        x, y = render_chat(
            sample["messages"] + [{"role": "assistant", "content": expected}], tok
        )
        effective = y[y != -100]
        samples.append(
            {
                "question": question,
                "expected": expected,
                "recorded_expected_matches_arithmetic": expected == sample["expected"],
                "generated": sample["generated"],
                "decoded_generated_ids": tok.decode([i for i in sample["generated_ids"] if i != tok.eos_id]),
                "content_correct_recomputed": sample["generated"].split("，", 1)[0].strip() == expected_number,
                "recorded_content_correct": sample["content_correct"],
                "fixed_suffix_present": sample["generated"].endswith("，像把兩組積木合在一起再數。"),
                "first_target": int(effective[0]),
                "generated_first_token": sample["generated_ids"][0],
                "first_target_wrong": int(effective[0]) != sample["generated_ids"][0],
                "effective_tokens_recomputed": len(effective),
                "target_labels_match_answer_and_eos": effective.tolist() == tok.encode(expected) + [tok.eos_id],
                "input_length": len(x),
                "labels_dtype": str(y.dtype),
                "target_last_id": int(effective[-1]),
            }
        )
    recomputed_count = sum(s["effective_tokens_recomputed"] for s in samples)
    nll = test["nll_sum"] / recomputed_count
    empirical[style] = {
        "recorded_summary": {k: v for k, v in test.items() if k != "samples"},
        "training": {k: run["training"][k] for k in ["steps", "effective_tokens", "records", "records_sha256"]},
        "samples": samples,
        "recomputed_effective_tokens": recomputed_count,
        "correct_answers": sum(s["content_correct_recomputed"] for s in samples),
        "fixed_suffix_count": sum(s["fixed_suffix_present"] for s in samples),
        "recomputed_nll": nll,
        "recorded_nll_absolute_difference": abs(test["nll"] - nll),
        "round_sum_2": round(test["nll_sum"], 2),
        "round_nll_2": round(nll, 2),
        "round_nll_5": round(nll, 5),
    }

vivid = empirical["vivid"]
count = len(vivid["samples"])
suffix_bytes = len("，像把兩組積木合在一起再數。".encode())
digit_lower_bound = count * math.log(2)
remaining_upper_bound = vivid["recorded_summary"]["nll_sum"] - digit_lower_bound
bound = {
    "all_first_targets_wrong": all(s["first_target_wrong"] for s in vivid["samples"]),
    "single_byte_correct_numbers": all(len(tok.encode(s["expected"].split("，")[0])) == 1 for s in vivid["samples"]),
    "fixed_suffix_bytes_per_example": suffix_bytes,
    "fixed_suffix_target_count": suffix_bytes * count,
    "eos_target_count": count,
    "first_digit_nll_sum_lower_bound": digit_lower_bound,
    "first_digit_mean_nll_lower_bound": math.log(2),
    "fixed_suffix_plus_eos_nll_sum_upper_bound": remaining_upper_bound,
    "fixed_suffix_plus_eos_mean_nll_upper_bound": remaining_upper_bound / (vivid["recomputed_effective_tokens"] - count),
    "fixed_suffix_only_mean_nll_upper_bound": remaining_upper_bound / (suffix_bytes * count),
    "overall_vivid_mean": vivid["recomputed_nll"],
    "derivation": "greedy 的第一個輸出不同於正確目標，故正確目標機率至多為 1/2，該位置的 NLL 至少為 ln(2)。七題皆如此，數字位置合計至少為 7*ln(2)。NLL 非負，總代價扣除此下界後，是其餘位置代價的上界；再除以固定句位置數可得其平均上界。這是同一積木模型內部的分組界線，不能把兩支模型的全部平均差歸因於單一因素。",
}

probabilities = torch.tensor(
    [[[0.5, 0.25, 0.25], [0.2, 0.3, 0.5], [0.125, 0.75, 0.125], [0.5, 0.25, 0.25]]],
    dtype=torch.float64,
)
labels = torch.tensor([[0, -100, 1, 2]], dtype=torch.long)
total, effective_count = loss_sum(probabilities.log(), labels)
expected_total = -math.log(0.5) - math.log(0.75) - math.log(0.25)
masked_example = {
    "probabilities": probabilities.tolist(),
    "labels": labels.tolist(),
    "logits_shape": list(probabilities.shape),
    "logits_dtype": str(probabilities.dtype),
    "labels_dtype": str(labels.dtype),
    "expected_sum": expected_total,
    "observed_sum": total.item(),
    "sum_absolute_error": abs(total.item() - expected_total),
    "expected_count": 3,
    "observed_count": effective_count.item(),
    "observed_mean": (total / effective_count).item(),
}

code_sources = {}
for path, names in {
    "tiny_perceptron/data.py": ["ByteTokenizer", "render_chat", "pad_batch"],
    "tiny_perceptron/model.py": ["loss_sum", "generate"],
    "scripts/course_experiments/common.py": ["text_examples", "_nll", "evaluate_lm"],
    "scripts/course_experiments/behavior.py": ["_style_record", "_style_metrics", "run_style"],
}.items():
    full = ROOT / path
    code_sources[path] = {
        "sha256": digest(full),
        "recorded_sha256": record["code_sha256"][path],
        "matches_recorded_code": digest(full) == record["code_sha256"][path],
        "excerpts": {name: function_excerpt(full, name) for name in names},
    }

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-7/verify_score_tradeoff.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "inputs": {
        str(source.relative_to(ROOT)): digest(source),
        str(result_path.relative_to(ROOT)): digest(result_path),
    },
    "arithmetic_example": {"2+2": 2 + 2, "five_is_correct": 5 == 2 + 2},
    "frozen_code_stdout": stdout.getvalue(),
    "weight_cases": scores,
    "unchanged_answer_rows": namespace["answers"],
    "historical_environment": {k: record[k] for k in ["revision", "seed", "device", "gpu", "python_version", "torch_version", "step_scale", "evidence_status", "status"]},
    "historical_split": record["results"]["arithmetic_data"],
    "empirical_record_checks": empirical,
    "low_suffix_cost_bound": bound,
    "loss_sum_masked_example": masked_example,
    "repository_sources": code_sources,
    "limitations": [
        "讀取既有完整實驗紀錄並重算算術、生成判分與目標分母，沒有重訓或重跑歷史 GPU 權重。",
        "固定句的代價以非負性與 greedy 錯答建立上界，沒有宣稱量到各 token 的實際 NLL。",
        "現行風格資料生成器與歷史紀錄的全檔 SHA 不同；以逐題 expected/generated 原文直接核對答案，歷史平均代價與 greedy 路徑所需的共用、資料與模型實作均與紀錄 SHA 一致。",
        "不支持通用比喻品質、理解改善、其他題材或其他 seed 的品質結論。",
    ],
}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
