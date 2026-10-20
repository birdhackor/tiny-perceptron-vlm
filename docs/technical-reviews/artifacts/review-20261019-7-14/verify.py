"""核對交叉熵算例、標籤映射、抽樣分母與保存的錯標實驗。"""

import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments import run as runner
from scripts.course_experiments.run import experiment_spec
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(observed, expected, tolerance=1e-12):
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"數值不符：實得 {observed}，預期 {expected}")


def target_count(rows, steps, batch_size=16, seed=42):
    examples = text_examples(rows, "sft")
    lengths = [int((labels != IGNORE).sum()) for _, labels in examples]
    sampler = random.Random(seed)
    total = sum(sum(sampler.choices(lengths, k=batch_size)) for _ in range(steps))
    return {"steps": steps, "batch_size": batch_size, "effective_tokens": total}


tok = ByteTokenizer()
mapping = {str(i): tok.encode(str(i)) for i in range(4)}
assert list(mapping.values()) == [[56], [57], [58], [59]], "byte 字表映射不符"

numeric = []
for wrong_score in (1.0, 5.0, 8.0):
    values = [0.0, 0.0, 3.0, wrong_score]
    z = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    denominator = sum(math.exp(v) for v in values)
    probabilities = z.softmax(-1)[0].detach().tolist()
    losses = {}
    gradients = {}
    for label in (2, 3):
        loss = F.cross_entropy(z, torch.tensor([label]))
        expected = math.log(denominator) - values[label]
        close(loss.item(), expected)
        losses[str(label)] = loss.item()
        gradient = torch.autograd.grad(loss, z, retain_graph=True)[0][0].tolist()
        for j, value in enumerate(gradient):
            close(value, probabilities[j] - int(j == label))
        gradients[str(label)] = gradient
    numeric.append({"logits": values, "probabilities": probabilities, "losses": losses, "gradients": gradients})
assert numeric[0]["losses"]["3"] > numeric[1]["losses"]["3"] > numeric[2]["losses"]["3"], "錯標代價沒有遞減"
assert numeric[0]["losses"]["2"] < numeric[1]["losses"]["2"] < numeric[2]["losses"]["2"], "正確標籤代價沒有遞增"

record_path = Path("docs/course-experiments/results/sft_ablation.json")
base_record_path = Path("docs/course-experiments/results/sft.json")
record = json.loads(record_path.read_bytes())
base_record = json.loads(base_record_path.read_bytes())
result = record["results"]
parts = split_records(generate_records("attributes-sft"), seed=42)
split_checks = {}
for name, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    saved = result["data"]["attributes"][name]
    observed = {"records": len(rows), "families": len({row["family"] for row in rows}), "sha256": hashlib.sha256(raw).hexdigest()}
    for key, value in observed.items():
        assert value == saved[key], "重建的資料切分與紀錄不符"
    for other, other_rows in parts.items():
        if name != other:
            assert not {row["family"] for row in rows} & {row["family"] for row in other_rows}, "家族跨越切分"
    split_checks[name] = observed

clean = parts["train"]
noisy = copy.deepcopy(clean)
corruptions = []
for index, row in enumerate(noisy):
    answer = row["messages"][-1]["content"]
    if answer in ("circle", "square") and len(corruptions) < max(1, len(noisy) // 10):
        row["messages"][-1]["content"] = "square" if answer == "circle" else "circle"
        corruptions.append({"row": index, "family": row["family"], "correct": answer, "wrong": row["messages"][-1]["content"]})
assert corruptions == result["corruptions"], "重建的改標紀錄不符"
assert len(corruptions) == 4, "改標筆數不符"
assert len({item["family"] for item in corruptions}) == 2, "改標家族數不符"
assert all(a["messages"][:-1] == b["messages"][:-1] for a, b in zip(clean, noisy)), "支線問題不相同"

branches = {}
for name, rows in (("clean", clean), ("noisy", noisy)):
    saved = result["runs"][name]
    counts = target_count(rows, 300)
    assert counts["effective_tokens"] == saved["training"]["effective_tokens"] == 33733, "抽樣後有效目標數不符"
    assert saved["training"]["steps"] == 300 and saved["training"]["records"] == len(rows) == 45, "支線訓練預算不符"
    assert saved["training"]["records_sha256"] == records_sha256(rows), "訓練題目指紋不符"
    evaluations = {}
    for split in ("validation", "test"):
        evaluation = saved["attributes"][split]
        samples = evaluation["samples"]
        expected_tokens = sum(int((labels != IGNORE).sum()) for _, labels in text_examples(parts[split], "sft"))
        matches, ended = 0, 0
        sample_checks = []
        for row, sample in zip(parts[split], samples, strict=True):
            assert sample["messages"] == row["messages"][:-1], "評分問題與乾淨切分不同"
            assert sample["expected"] == row["messages"][-1]["content"], "評分答案與乾淨切分不同"
            ids = sample["generated_ids"]
            eos = tok.eos_id in ids
            raw = ids[:ids.index(tok.eos_id)] if eos else ids
            exact = raw == tok.encode(sample["expected"])
            assert exact == sample["exact"] and eos == sample["eos"], "生成 token 與評分欄位不符"
            assert tok.decode(raw) == sample["generated"], "生成 token 與文字不符"
            matches += int(exact)
            ended += int(eos)
            sample_checks.append({"expected": sample["expected"], "generated": sample["generated"], "exact": exact, "eos": eos})
        assert evaluation["records"] == evaluation["examples"] == len(samples), "評分題數不符"
        assert evaluation["matches"] == matches, "完整答對題數不符"
        close(evaluation["exact_match"], matches / len(samples))
        close(evaluation["eos_rate"], ended / len(samples))
        assert evaluation["effective_tokens"] == expected_tokens, "評分有效目標分母不符"
        close(evaluation["nll"], evaluation["nll_sum"] / expected_tokens)
        evaluations[split] = {"matches": matches, "records": len(samples), "eos_ended": ended, "nll": evaluation["nll"], "nll_sum": evaluation["nll_sum"], "effective_tokens": expected_tokens, "samples": sample_checks}
    branches[name] = {"training": counts, "records_sha256": records_sha256(rows), "evaluations": evaluations}

base_training = base_record["results"]["training"]
assert base_training["steps"] == 900 and base_training["records"] == 45, "底座預算不符"
assert base_record["results"]["checkpoint"] == "model.pt", "底座路徑不符"
assert base_training["records_sha256"] == records_sha256(clean), "底座訓練資料不符"
assert base_record["results"]["after"] == result["before"]["A_attributes"], "底座前後評分紀錄不同"
assert base_record["seed"] == record["seed"] == 42, "隨機種子不符"
assert record["gpu"] == "NVIDIA L4" and record["device"] == "cuda", "保存的硬體條件不符"
assert round(record["elapsed_seconds"], 1) == 23.5, "耗時取近似值不符"

code_checks = {}
for name in ("scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"):
    observed = digest(Path(name))
    assert observed == record["code_sha256"][name], "實驗核心程式版本與紀錄不符"
    code_checks[name] = observed

cli_argv = [".venv/bin/python", "-m", "scripts.course_experiments.run", "--list-assets", "sft_ablation"]
cli = subprocess.run(cli_argv, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
assert cli.returncode == 0 and cli.stdout == "", "實驗入口的資產查詢不符"
spec = experiment_spec("sft_ablation")
assert spec["dependencies"] == ["sft"] and spec["function"] == "run_sft_ablation", "實驗入口的相依與函式不符"

dispatch_calls = []


def capture_dispatch(experiment_id, device, output, dependencies, assets, **kwargs):
    dispatch_calls.append({"experiment_id": experiment_id, "device": device, "output": str(output.relative_to(ROOT)), "dependencies": str(dependencies.relative_to(ROOT)), "assets": str(assets.relative_to(ROOT)), "step_scale": kwargs["step_scale"]})
    return {"experiment_id": experiment_id, "elapsed_seconds": 0.0, "evidence_status": "dispatch_only", "artifacts": []}


dispatch_stdout = io.StringIO()
with patch.object(sys, "argv", ["run.py", "--experiment", "sft_ablation", "--device", "cuda"]), patch.object(runner, "execute", capture_dispatch), redirect_stdout(dispatch_stdout):
    runner.main()
assert dispatch_calls == [{"experiment_id": "sft_ablation", "device": "cuda", "output": "outputs/course-experiments/course-v1/sft_ablation", "dependencies": "outputs/course-experiments/course-v1", "assets": "assets/training", "step_scale": 1.0}], "CLI 參數派送不符"

counterexample = {}
for name, correct_probabilities in (("clean", [0.6, 0.6]), ("alternative", [0.99, 0.49])):
    counterexample[name] = {"correct_probabilities": correct_probabilities, "mean_nll": -sum(math.log(p) for p in correct_probabilities) / 2, "greedy_accuracy": sum(p > 0.5 for p in correct_probabilities) / 2}

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-14/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "source_sha256": digest(Path("docs/course-review-20261019/freeze-01/sources/7.14.md")),
    "experiment_record": {"path": str(record_path), "sha256": digest(record_path), "seed": record["seed"], "gpu": record["gpu"], "elapsed_seconds": record["elapsed_seconds"], "timing_scope": record["timing_scope"], "revision": record["revision"]},
    "base_record": {"path": str(base_record_path), "sha256": digest(base_record_path), "training_steps": base_training["steps"], "after_equals_ablation_before": True},
    "byte_tokenizer_ids": mapping,
    "numeric": numeric,
    "splits": split_checks,
    "corruptions": corruptions,
    "corrupted_fraction": len(corruptions) / len(clean),
    "branches": branches,
    "core_code_sha256_matches": code_checks,
    "cli": {"argv": cli_argv, "exit_code": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr, "specification": spec},
    "training_command_dispatch": {"calls": dispatch_calls, "stdout": dispatch_stdout.getvalue(), "scope": "只核對 CLI 參數派送；execute 被攔截，未啟動訓練、寫入實驗目錄或呼叫 CUDA。"},
    "metric_counterexample": counterexample,
    "result": "全部斷言通過；未重新訓練模型，實驗成績與耗時僅核對保存紀錄及其內部一致性。"
}, ensure_ascii=False, indent=2))
