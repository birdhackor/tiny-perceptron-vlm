"""核對教材算例、遮罩、既有實驗逐題計數與原始來源；不進行訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_bytes())


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1
        if tag in ("p", "div", "section", "h1", "h2", "h3", "h4"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignored -= 1
        if tag in ("p", "div", "section", "h1", "h2", "h3", "h4"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def original_sources():
    settings = [
        (
            "llama2",
            "https://arxiv.org/html/2307.09288v2",
            "arXiv:2307.09288v2, 2023-07-19",
            "Safety RLHF / Measure of False Refusal",
            "With more safety data mixed in model tuning, the false-refusal rate becomes larger on both datasets",
            "增加安全資料後，正常題及邊界題的錯誤拒絕率都可能增加。論文同時說一般 helpfulness 題的錯誤拒絕仍罕見，不能用它聲稱模型實際拒絕所有題。",
        ),
        (
            "instructgpt",
            "https://arxiv.org/html/2203.02155v1",
            "arXiv:2203.02155v1, 2022-03-04",
            "Results / We can minimize performance regressions on public NLP datasets",
            "During RLHF fine-tuning, we observe performance regressions compared to GPT-3 on certain public NLP datasets",
            "偏好微調可能使先前任務退步；混入預訓練更新在該論文設定可減輕退步。這不是本頁手寫配方或風格轉 JSON 失敗的實驗結果。",
        ),
        (
            "dpo_paper",
            "https://arxiv.org/html/2305.18290v3",
            "arXiv:2305.18290v3, 2024-07-29",
            "Direct Preference Optimization / equation (7), What does the DPO update do?, Experimental Evaluation",
            "the DPO update increases the relative log probability of preferred to dispreferred responses",
            "式中策略以 x 為條件，資料分布只決定訓練期望的權重；配方中拒絕資料的占比不是每一題輸出的拒絕機率。梯度更新共用參數，未教行為並無不變保證。候選分數與生成另行評估。",
        ),
        (
            "python_none",
            "https://docs.python.org/3.13/library/constants.html#None",
            "Python 3.13 documentation, accessed 2026-10-20",
            "Built-in Constants / None",
            "An object frequently used to represent the absence of a value",
            "None 可以表示尚無數值；本頁明確約定其意義為尚未量測，與整數零分開。",
        ),
        (
            "python_items",
            "https://docs.python.org/3.13/library/stdtypes.html#dict.items",
            "Python 3.13 documentation, accessed 2026-10-20",
            "Mapping Types / dict.items",
            "Return a new view of the dictionary’s items",
            "items 提供 key、value 成對的 view；本頁迴圈逐對解包。",
        ),
        (
            "python_expressions",
            "https://docs.python.org/3.13/reference/expressions.html",
            "Python 3.13 documentation, accessed 2026-10-20",
            "Dictionary displays; Conditional expressions",
            "The expression x if C else y first evaluates the condition",
            "字典 comprehension 產生各 key/value；條件式依條件決定回傳哪一邊。此處是 value is None，故零不會被當成未測。",
        ),
    ]
    receipts = []
    for identifier, url, version, locator, excerpt, note in settings:
        command = ["curl", "-L", "--fail", "--max-time", "45", "-sS", url]
        completed = subprocess.run(command, capture_output=True, check=True)
        parser = PlainText()
        parser.feed(completed.stdout.decode("utf-8"))
        plain = re.sub(r"\s+", " ", "".join(parser.parts))
        assert excerpt in plain, f"原始來源未找到預期摘錄：{url}"
        assert len(excerpt.split()) <= 25
        if identifier.startswith("python_"):
            assert "Python 3.13" in plain
        if identifier in ("llama2", "instructgpt", "dpo_paper"):
            assert version.split(",")[0] in plain
        receipts.append(
            {
                "id": identifier,
                "url": url,
                "version": version,
                "locator": locator,
                "accessed_on": datetime.now(UTC).date().isoformat(),
                "command_argv": command,
                "curl_exit_code": completed.returncode,
                "response_bytes": len(completed.stdout),
                "response_sha256": digest(completed.stdout),
                "excerpt": excerpt,
                "inspection_note": note,
                "checked_original": True,
                "full_text_storage": "完整回應只在記憶體中解析；僅保存短摘錄與雜湊。",
            }
        )
    return receipts


def preference_summary(evaluation):
    samples = evaluation["samples"]
    relative = 0
    absolute = 0
    rows = []
    for row in samples:
        margin = row["policy_chosen_logp"] - row["policy_rejected_logp"]
        relative_margin = margin - row["reference_margin"]
        assert math.isclose(margin, row["policy_margin"], abs_tol=1e-10)
        assert math.isclose(relative_margin, row["relative_margin"], abs_tol=1e-10)
        relative += int(relative_margin > 0)
        absolute += int(margin > 0)
        rows.append(row)
    assert len(samples) == evaluation["records"]
    assert relative == evaluation["relative_preference_improved"]
    assert absolute == evaluation["chosen_higher_absolute_probability"]
    return {"records": len(samples), "relative_positive": relative, "chosen_first": absolute, "samples": rows}


def generation_summary(evaluation):
    tok = ByteTokenizer()
    samples = evaluation["samples"]
    count = 0
    for sample in samples:
        ids = sample["generated_ids"]
        raw = ids[: ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        exact = raw == tok.encode(sample["expected"])
        assert exact == sample["exact"]
        count += int(exact)
    assert count == evaluation["matches"]
    assert len(samples) == evaluation["records"]
    assert evaluation["exact_match"] == count / len(samples)
    return {"records": len(samples), "matches": count, "samples": samples}


def main():
    torch.set_num_threads(1)
    dpo = read_json("docs/course-experiments/results/dpo.json")
    style = read_json("docs/course-experiments/results/style.json")
    source = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.9.md").read_text()
    match = re.search(r"```python\n(.*?)```", source, re.S)
    namespace = {}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(match[1], "frozen-section", "exec"), namespace)
    recipe, metrics = namespace["recipe"], namespace["metrics"]
    assert sum(recipe.values()) == 100
    assert len(metrics) == 6 and all(value is None for value in metrics.values())
    assert capture.getvalue().count("尚未量測") == 6
    changed = dict(recipe)
    changed["應拒絕題"] += changed["正常題完成"]
    changed["正常題完成"] = 0
    assert sum(changed.values()) == 100 and changed["應拒絕題"] == 40
    assert all(value is None for value in metrics.values())
    zero_display = "尚未量測" if 0 is None else 0
    assert zero_display == 0

    examples = [
        render_chat([{"role": "user", "content": "題目"}, {"role": "assistant", "content": text}])
        for text in ("A", "BCD")
    ]
    x, labels, valid = pad_batch(examples)
    tok = ByteTokenizer()
    assert [int((y != IGNORE).sum()) for y in labels] == [2, 4]
    assert all(int(y[y != IGNORE][-1]) == tok.eos_id for y in labels)
    assert (labels[~valid] == IGNORE).all()
    logits = torch.zeros((*labels.shape, tok.vocab_size), dtype=torch.float64)
    scores = sequence_log_probability(logits, labels)
    expected = -torch.tensor([2, 4], dtype=torch.float64) * math.log(tok.vocab_size)
    assert torch.allclose(scores, expected, rtol=0, atol=1e-12)

    runs = {
        name: {
            "beta": run["beta"],
            "training": {k: v for k, v in run["training"].items() if k != "history"},
            "arithmetic_test": generation_summary(run["arithmetic"]["test"]),
            "preference_test": preference_summary(run["preference"]["test"]),
        }
        for name, run in dpo["results"]["runs"].items()
    }
    baseline = generation_summary(style["results"]["content_evaluation"]["test"])
    assert baseline["matches"] == 0 and baseline["records"] == 7
    baseline_prompts = [sample["messages"] for sample in baseline["samples"]]
    for run in runs.values():
        assert run["arithmetic_test"]["matches"] == 0 and run["arithmetic_test"]["records"] == 7
        assert [sample["messages"] for sample in run["arithmetic_test"]["samples"]] == baseline_prompts
    formatted = dpo["results"]["format_only"]
    format_summary = {
        "preference_test": preference_summary(formatted["preference"]["test"]),
        "arithmetic_test": generation_summary(formatted["arithmetic"]["test"]),
    }
    assert format_summary["preference_test"]["relative_positive"] == 7
    assert format_summary["arithmetic_test"]["matches"] == 0
    natural = dpo["results"]["ultrafeedback_pilot"]
    natural_summary = {
        "source_records": natural["source_records"],
        "data": natural["data"],
        "sft_training": {k: v for k, v in natural["sft_training"].items() if k != "history"},
        "training": {k: v for k, v in natural["training"].items() if k != "history"},
        "evaluation": {name: preference_summary(report) for name, report in natural["evaluation"].items()},
        "keys": list(natural),
    }
    assert natural["sft_training"]["steps"] == natural["training"]["steps"] == 80
    assert natural_summary["evaluation"]["validation"]["relative_positive"] == 6
    assert natural_summary["evaluation"]["validation"]["chosen_first"] == 7
    assert natural_summary["evaluation"]["test"]["relative_positive"] == 8
    assert natural_summary["evaluation"]["test"]["chosen_first"] == 4
    model = TinyLM(ModelConfig(width=32, layers=1, max_length=256))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == natural["sft_training"]["parameters"] == 37728

    paths = [
        "docs/course-experiments/results/dpo.json",
        "docs/course-experiments/results/style.json",
        "scripts/course_experiments/behavior.py",
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/text.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/alignment.py",
        "tiny_perceptron/model.py",
        "course/training.md",
    ]
    provenance = {}
    for name, record in (("dpo", dpo), ("style", style)):
        command = ["git", "show", record["revision"] + ":scripts/course_experiments/behavior.py"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        provenance[name] = {
            "recorded_revision": record["revision"],
            "recorded_environment": {k: record[k] for k in ("python_version", "torch_version", "device", "gpu", "seed")},
            "recorded_behavior_sha256": record["code_sha256"]["scripts/course_experiments/behavior.py"],
            "current_behavior_sha256": digest((ROOT / "scripts/course_experiments/behavior.py").read_bytes()),
            "historical_source_command": command,
            "historical_source_exit_code": completed.returncode,
            "historical_source_stderr": completed.stderr,
            "scope": "本次核對保存的結果與目前程式；未重跑原 GPU 訓練，舊 revision 無法取得該程式。",
        }
    behavior = (ROOT / "scripts/course_experiments/behavior.py").read_text()
    tree = ast.parse(behavior)
    selected = ("_pair_examples", "_dpo_train", "_preference_evaluate", "_natural_dpo_pilot", "run_dpo")
    excerpts = {
        node.name: ast.get_source_segment(behavior, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    }
    print(json.dumps({
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "input_sha256": {path: digest((ROOT / path).read_bytes()) for path in paths},
        "recipe": {"original": recipe, "total": sum(recipe.values()), "stdout": capture.getvalue(), "metric_names": list(metrics), "all_none": True, "zero_display": zero_display, "changed": changed, "changed_total": sum(changed.values()), "changed_metrics_all_none": all(v is None for v in metrics.values())},
        "target_arithmetic": {"thirty_pairs_hundred_targets_each_side": 30 * 2 * 100, "thirty_pairs_ten_targets_each_side": 30 * 2 * 10},
        "mask_probe": {"input": x.tolist(), "labels": labels.tolist(), "valid": valid.tolist(), "effective_targets": [2, 4], "uniform_logp_observed": scores.tolist(), "uniform_logp_expected": expected.tolist(), "absolute_tolerance": 1e-12},
        "baseline": baseline,
        "runs": runs,
        "format_only": format_summary,
        "natural": natural_summary,
        "architecture_probe": {"width": 32, "layers": 1, "max_length": 256, "parameters": parameters},
        "code_excerpts": excerpts,
        "provenance_limits": provenance,
        "original_sources": original_sources(),
        "result": "全部斷言通過；只核對紀錄與機制，不重新量測訓練品質。",
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
