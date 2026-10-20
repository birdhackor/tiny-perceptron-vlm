import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from scripts.course_experiments.compression import _evaluate, _example, _prompt
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    result = subprocess.run(
        ["curl", "-L", "--fail", "--max-time", "40", "-sS", url],
        capture_output=True,
        check=True,
    )
    return result.stdout


def visible(raw):
    text = raw.decode("utf-8")
    text = re.sub(r"<(script|style)\b[^>]*>[\s\S]*?</\1>", " ", text)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def gold(question):
    fields = question.split(";")
    attrs = dict(item.split("=") for item in fields[:-1])
    return {
        "describe": attrs["shape"],
        "shape?": attrs["shape"],
        "color?": attrs["color"],
        "pitch?": attrs["pitch"],
        "joint?": attrs["shape"] + "," + attrs["pitch"],
    }[fields[-1]]


def main():
    torch.set_num_threads(1)
    tok = ByteTokenizer()
    path = ROOT / "docs/course-experiments/results/distillation.json"
    run = json.loads(path.read_bytes())
    attrs = run["results"]["tasks"]["attributes"]
    source = ROOT / "course/chapters/18.md"
    match = re.search(rb"(?m)^## 18\.10 .+?(?=^## |\Z)", source.read_bytes(), re.S)
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/18.10.md"
    assert match[0] == frozen.read_bytes()
    code = re.search(rb"```python\n(.*?)```", match[0], re.S)[1].decode()
    toy = []
    for budget in (50, 100):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code.replace("100, 50", f"100, {budget}"), str(source), "exec"), {})
        expected = [("SFT", 0.75, 10, 100 * budget // 10), ("蒸餾", 1.0, 50, 100 * budget // 50)]
        assert output.getvalue() == "".join(
            f"{method} 正確率 {acc} 合計秒數 {seconds} 預算 {budget} 秒內可更新 {updates} 次\n"
            for method, acc, seconds, updates in expected
        )
        toy.append({"budget": budget, "stdout": output.getvalue(), "results": expected})

    evaluations = {"teacher": attrs["teacher_test"]}
    evaluations.update({key: value["test"] for key, value in attrs["runs"].items() if "packed" not in key})
    computed = {}
    for name, evaluation in evaluations.items():
        correct, count, eos = 0, 0, 0
        for sample in evaluation["generated_samples"]:
            assert sample["expected"] == gold(sample["question"])
            ids = sample["generated_ids"]
            answer = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            hit = answer == tok.encode(sample["expected"])
            assert hit == sample["exact"]
            correct += hit
            eos += tok.eos_id in ids
            count += len(tok.encode(sample["expected"])) + 1
        assert correct == evaluation["correct"]
        assert count == evaluation["supervised_tokens"] == 69
        assert eos == evaluation["eos_count"] == 10
        nll = evaluation["nll_sum"] / count
        assert math.isclose(nll, evaluation["answer_nll"], rel_tol=0, abs_tol=1e-14)
        computed[name] = {
            "correct": correct,
            "examples": len(evaluation["generated_samples"]),
            "supervised_tokens_with_eos": count,
            "nll_sum": evaluation["nll_sum"],
            "answer_nll": nll,
            "rounded_four_places": f"{nll:.4f}",
            "eos_count": eos,
        }

    audit = attrs["hard_target_generation"]["audit"]
    records, token_counts = [], []
    for row in audit:
        assert row["gold_answer"] == row["teacher_answer"] == gold(row["question"])
        assert row["teacher_ids"] == tok.encode(row["gold_answer"]) + [tok.eos_id]
        record = {"family": row["family"], "question": row["question"], "answer": row["gold_answer"]}
        x, y = _example(record, 128)
        hard_x, hard_y = _example({**record, "_hard_ids": row["teacher_ids"], "_hard_eos": True}, 128)
        assert torch.equal(x, hard_x) and torch.equal(y, hard_y)
        token_counts.append(int((y != IGNORE).sum()))
        records.append(record)
    assert len(audit) == attrs["hard_target_generation"]["correct"] == 45
    training_families = {row["family"] for row in audit}
    test_families = {row["family"] for row in attrs["teacher_test"]["generated_samples"]}
    assert not training_families & test_families
    rng = random.Random(run["seed"])
    plan = [[rng.randrange(len(records)) for _ in range(16)] for _ in range(400)]
    effective = sum(token_counts[index] for batch in plan for index in batch)
    input_tokens = sum(len(_example(records[index], 128)[0]) for batch in plan for index in batch)
    matched = {}
    for width in (16, 32):
        group = {method: attrs["runs"][f"w{width}_{method}"] for method in ("ce", "teacher_hard", "ce_kl")}
        base = group["ce"]["training"]
        for method, result in group.items():
            training = result["training"]
            for field in ("initialization_sha256", "batch_plan_sha256", "effective_supervised_tokens", "optimizer_updates"):
                assert training[field] == base[field]
            assert training["effective_supervised_tokens"] == effective == 44985
            assert sha(json.dumps(plan).encode()) == training["batch_plan_sha256"]
        assert group["ce"]["training"]["final_sha256"] == group["teacher_hard"]["training"]["final_sha256"]
        assert group["ce"]["test"]["generated_samples"] == group["teacher_hard"]["test"]["generated_samples"]
        model = TinyLM(ModelConfig(width=width, layers=1, heads=2, max_length=128))
        parameters = sum(p.numel() for p in model.parameters())
        assert all(r["storage"]["parameter_count"] == parameters for r in group.values())
        matched[str(width)] = {
            "initialization_sha256": base["initialization_sha256"],
            "batch_plan_sha256": base["batch_plan_sha256"],
            "optimizer_updates": base["optimizer_updates"],
            "effective_supervised_tokens": effective,
            "unpadded_input_tokens_from_same_plan": input_tokens,
            "parameters": parameters,
            "hard_and_gold_final_parameters_equal": True,
            "hard_and_gold_test_generated_samples_equal": True,
        }

    test_records = [
        {"question": row["question"], "answer": row["expected"]}
        for row in attrs["teacher_test"]["generated_samples"]
    ]

    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_length=128)

        def forward(self, ids, valid=None, cache=None):
            return {"logits": torch.zeros((*ids.shape, tok.vocab_size)), "cache": None}

    metric_probe = _evaluate(UniformModel(), test_records, "cpu", tokens=24)
    assert metric_probe["supervised_tokens"] == 69
    assert abs(metric_probe["answer_nll"] - math.log(264)) < 1e-6
    story = run["results"]["tasks"]["moe_to_dense"]["runs"]
    story_values = {}
    for method in ("w32_ce", "w32_ce_kl"):
        result = story[method]["test"]
        assert result["supervised_tokens"] == 41914 and result["examples"] == 52
        nll = result["nll_sum"] / result["supervised_tokens"]
        assert abs(nll - result["answer_nll"]) < 1e-14
        story_values[method] = nll
    assert story_values["w32_ce_kl"] < story_values["w32_ce"]

    gsm_url = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
    gsm_raw = b"".join(fetch(gsm_url).splitlines(keepends=True)[:200])
    selection = attrs["out_of_domain_gsm8k"]["selection"]
    assert sha(gsm_raw) == selection["source_file_sha256"]
    original = [json.loads(line) for line in gsm_raw.splitlines()]
    eligible = []
    for index, row in enumerate(original):
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        record = {"question": row["question"], "answer": answer}
        prefix = len(_prompt(record))
        if prefix + 24 <= 128 and len(tok.encode(answer)) + 1 <= 24:
            eligible.append({"source_row": index, "prefix_tokens": prefix, "answer_tokens_with_eos": len(tok.encode(answer)) + 1})
    assert [row["source_row"] for row in eligible] == selection["selected_source_rows"] == [14, 94]
    for index, sample in zip(selection["selected_source_rows"], attrs["out_of_domain_gsm8k"]["teacher"]["generated_samples"], strict=True):
        assert sample["question"] == original[index]["question"]
        assert sample["expected"] == original[index]["answer"].split("####")[-1].strip().replace(",", "")

    external = []
    requests = [
        (
            "hinton", "https://arxiv.org/html/1503.02531v1", "arXiv:1503.02531v1; 2015-03-09",
            "models of the same size; weighted hard and soft objectives; training versus deployment requirements",
            "a model of the same size that is learned directly from the same training data",
            "Introduction; Distillation; Experiments on speech recognition",
        ),
        (
            "kim_rush", "https://arxiv.org/html/1606.07947v4", "arXiv:1606.07947v4; 2016-09-22",
            "teacher-generated sequences become a new CE dataset; conditional sequence NLL is a sum of token NLLs",
            "train the student network with cross-entropy on this new dataset.",
            "Word-Level Knowledge Distillation; Sequence-Level Knowledge Distillation",
        ),
        (
            "pytorch", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", "PyTorch 2.14 documentation",
            "class-index cross entropy uses negative log softmax; ignored targets are excluded from the mean",
            "the loss is averaged over non-ignored targets.", "CrossEntropyLoss: class-index formula; ignore_index; reduction",
        ),
        (
            "nvidia_overview", "https://docs.nvidia.com/nim/benchmarking/llm/2.0.0/overview.md", "NIM Benchmarking LLM 2.0.0",
            "input/output lengths affect memory and latency; quality must be validated separately",
            "Sequence lengths** drive memory use and latency:", "Background on How LLM Inference Works",
        ),
        (
            "nvidia_metrics", "https://docs.nvidia.com/nim/benchmarking/llm/2.0.0/metrics.md", "NIM Benchmarking LLM 2.0.0",
            "throughput is completed output tokens or requests per duration; latency definitions have distinct scopes",
            "Requests per second (RPS) is the average number of requests", "End-to-End Request Latency; Tokens Per Second; Requests Per Second",
        ),
        (
            "nvidia_parameters", "https://docs.nvidia.com/nim/benchmarking/llm/2.0.0/parameters.md", "NIM Benchmarking LLM 2.0.0",
            "input/output lengths and concurrency must be controlled when comparing latency, memory and throughput",
            "Sequence lengths affect how fast a system processes input", "Use Cases; Load Control; Other Parameters",
        ),
    ]
    for identifier, url, version, supports, excerpt, locator in requests:
        raw = fetch(url)
        text = visible(raw) if not url.endswith(".md") else raw.decode()
        assert excerpt in text
        external.append({
            "id": identifier, "url": url, "version": version,
            "accessed_on": datetime.now(timezone.utc).date().isoformat(),
            "retrieval_command": ["curl", "-L", "--fail", "--max-time", "40", "-sS", url],
            "http_result": "curl exit 0 with --fail", "response_sha256": sha(raw),
            "response_bytes": len(raw), "checked_original": True,
            "locator": locator, "short_excerpt": excerpt, "inspection_note": supports,
        })

    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-10/verify.py",
        "result": "所有獨立核對通過；沒有重新訓練或下載模型權重。",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "source_sha256": sha(match[0]), "record_sha256": sha(path.read_bytes()),
        "record_environment": {key: run[key] for key in ("seed", "device", "torch_version", "python_version", "gpu")},
        "toy_executions": toy, "attribute_evaluations": computed,
        "matched_students": matched,
        "hard_target_audit": {"records": len(audit), "identical_gold_and_generated_target_tensors": True, "training_families": len(training_families), "test_families": len(test_families), "intersection": []},
        "metric_probe": {"expected_uniform_nll": math.log(264), "observed_answer_nll": metric_probe["answer_nll"], "supervised_tokens": metric_probe["supervised_tokens"], "tolerance": 1e-6},
        "story_nll": story_values,
        "gsm8k_original_check": {"url": gsm_url, "revision": "3101c7d5072418e28b9008a6636bde82a006892c", "accessed_on": datetime.now(timezone.utc).date().isoformat(), "first_200_bytes": len(gsm_raw), "first_200_sha256": sha(gsm_raw), "eligible": eligible, "full_question_identity": True, "saved_third_party_full_text": False},
        "cost_derivation": {"same_student_latency_savings_per_request_seconds": 0, "teacher_vs_student_savings_per_request_seconds": 0.009, "preparation_cost_seconds": 40, "teacher_vs_student_break_even_requests": math.ceil(40 / 0.009), "assumption": "每次服務成本固定、品質可接受、設備及輸入條件可比；人工每次更新成本按比例延伸，沒有新增快取或平行化假設。"},
        "original_sources": external,
        "limitations": ["只重新計算保存紀錄、檢查計分器與玩具算例，未重跑原始 CUDA 訓練。", "沒有載入保存的模型權重重新量測 NLL，也沒有量測實際部署效能。", "原始實驗的 compression.py SHA 與現行工作樹不同；以當次保存的逐題紀錄核對結果，現行函式的探針只驗證現行計分規則。", "單一 seed、十題測試與兩題域外診斷不能支持普遍品質結論。"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
