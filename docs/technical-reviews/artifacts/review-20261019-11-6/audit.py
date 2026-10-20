"""核對預算、遮罩、資料切分與既有實驗逐步紀錄；不執行訓練。"""

import contextlib
import hashlib
import html
import io
import json
import platform
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import modalities
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def receipt(url, version, selections):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", url]
    process = subprocess.run(command, capture_output=True, check=True)
    raw = process.stdout
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", raw.decode(), flags=re.S | re.I)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text)))
    excerpts = []
    for locator, start, end in selections:
        begin = text.index(start)
        finish = text.index(end, begin)
        excerpts.append({"locator": locator, "quote": text[begin:finish].strip()})
    return {
        "url": url,
        "version": version,
        "accessed_on": datetime.now().astimezone().date().isoformat(),
        "command_argv": command,
        "exit_code": process.returncode,
        "response_sha256": digest(raw),
        "response_bytes": len(raw),
        "excerpts": excerpts,
        "storage_policy": "僅保留支持主張的短摘錄；全文未寫入檔案。",
    }


result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-6/audit.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "scope": "重算既有紀錄、重建取樣目標數與小型前向計算；未重新訓練或載入原始實驗權重。",
}
paths = [
    "course/chapters/11.md",
    "docs/course-review-20261019/freeze-01/sources/11.6.md",
    "docs/course-review-20261019/freeze-04.json",
    "docs/course-experiments/results/vqa.json",
    "docs/course-experiments/results/projector.json",
    "docs/course-experiments/README.md",
    "scripts/course_experiments/modalities.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/multimodal.py",
    "tiny_perceptron/data.py",
    "docs/review-tools/section_facts.py",
]
result["input_sha256"] = {path: digest((ROOT / path).read_bytes()) for path in paths}

frozen = (ROOT / paths[1]).read_bytes()
chapter = (ROOT / paths[0]).read_bytes()
current = re.search(rb"(?ms)^## 11\.6 .*?(?=^## |\Z)", chapter).group()
assert current == frozen
fence = re.search(r"```python\n(.*?)```", frozen.decode(), re.S).group(1)
budget_runs = {}
for name, code in (
    ("original", fence),
    ("reduced_alignment", fence.replace('"描述對齊": 4000', '"描述對齊": 2000')),
    ("reallocated_budget", fence.replace('"描述對齊": 4000', '"描述對齊": 2000').replace('"問答 SFT": 6000', '"問答 SFT": 8000')),
):
    stream = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "budget-example", "exec"), namespace)
    totals = {key: sum(value.values()) for key, value in namespace["plans"].items()}
    budget_runs[name] = {"stdout": stream.getvalue(), "totals": totals}
assert list(budget_runs["original"]["totals"].values()) == [10000, 10000]
assert list(budget_runs["reduced_alignment"]["totals"].values()) == [8000, 10000]
assert list(budget_runs["reallocated_budget"]["totals"].values()) == [10000, 10000]
result["budget_examples"] = budget_runs

report = read_json("docs/course-experiments/results/vqa.json")
data = report["results"]
projector = read_json("docs/course-experiments/results/projector.json")
alignment = data["two_stage_alignment_training"]
assert alignment == projector["results"]["training"]
result["recorded_training_environment"] = {
    key: report[key] for key in ("revision", "device", "seed", "python_version", "torch_version", "gpu", "step_scale")
}
splits = modalities._vision_records(("shape?", "color?"))
for name, rows in splits.items():
    saved = data["data"]["splits"][name]
    assert rows == saved["records"]
    assert len(rows) == saved["count"]
    assert modalities._hash(rows) == saved["sha256"]
families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
assert not families["train"] & families["test"]
assert not families["train"] & families["validation"]
assert not families["validation"] & families["test"]
result["data_split"] = {
    "counts": {name: len(rows) for name, rows in splits.items()},
    "offsets": {name: sorted({row["offset"] for row in rows}) for name, rows in splits.items()},
    "colour_shape_combinations": sorted({(row["color"], row["shape"]) for row in splits["train"]}),
    "family_overlap": False,
    "records_and_hashes_match_current_generator": True,
}

ctx = SimpleNamespace(seed=report["seed"], device="cpu")
tok = ByteTokenizer()
histories = {}
for name, training, rows in (
    ("alignment", alignment, modalities._vision_records()["train"]),
    ("two_stage_question_answering", data["variants"]["all"]["training"], splits["train"]),
    ("direct_question_answering", data["variants"]["direct_vqa"]["training"], splits["train"]),
):
    expected_counts = []
    for step in range(training["steps"]):
        rng = random.Random(ctx.seed + step)
        batch = [rng.choice(rows) for _ in range(4)]
        expected_counts.append(sum(modalities._sequence(row, ctx)[2] for row in batch))
    observed_counts = [row["effective_targets"] for row in training["history"]]
    assert expected_counts == observed_counts
    assert len(observed_counts) == training["steps"]
    assert sum(observed_counts) == training["effective_targets"] == training["effective_tokens"]
    histories[name] = {
        "steps": training["steps"],
        "batch_examples": 4,
        "effective_targets": sum(observed_counts),
        "last_batch_targets": observed_counts[-1],
        "before_last_batch_targets": sum(observed_counts[:-1]),
        "per_step_counts_match_rebuilt_sampler": True,
        "trainable_parameters": training["trainable_parameters"],
        "weights_changed_recorded": training["weights_changed"],
    }
budget = histories["alignment"]["effective_targets"] + histories["two_stage_question_answering"]["effective_targets"]
direct = histories["direct_question_answering"]
assert budget == data["direct_vs_two_stage_budget"]["target_effective_tokens"] == 18238
assert direct["before_last_batch_targets"] < budget <= direct["effective_targets"]
assert direct["effective_targets"] == data["direct_vs_two_stage_budget"]["direct_effective_tokens"] == 18256
assert direct["effective_targets"] - budget == 18
result["training_histories"] = histories
result["budget_match"] = {
    "two_stage": budget,
    "direct": direct["effective_targets"],
    "excess": direct["effective_targets"] - budget,
    "stopped_at_first_batch_reaching_budget": True,
    "same_compute_claim": False,
    "alignment_record_exactly_matches_projector_experiment": True,
}

test = {}
for name in ("all", "direct_vqa"):
    evaluation = data["variants"][name]["test"]
    samples = evaluation["samples"]
    assert len(samples) == len(splits["test"]) == evaluation["examples"] == 12
    checked = []
    for record, sample in zip(splits["test"], samples, strict=True):
        assert record["family"] == sample["family"]
        assert record["question"] == sample["question"]
        assert record["answer"] == sample["target"]
        generated = sample["generated_ids"]
        raw = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
        correct = raw == tok.encode(record["answer"])
        assert correct == sample["exact_match"]
        assert tok.decode(raw) == sample["generated"]
        checked.append({key: sample[key] for key in ("row", "family", "question", "target", "generated", "exact_match")})
    correct = sum(row["exact_match"] for row in checked)
    assert correct == evaluation["correct"]
    assert correct / len(checked) == evaluation["exact_match"]
    test[name] = {"correct": correct, "examples": len(checked), "samples": checked}
result["test_scores"] = test
result["paired_differences"] = [
    {"family": first["family"], "question": first["question"], "two_stage_correct": first["exact_match"], "direct_correct": second["exact_match"]}
    for first, second in zip(test["all"]["samples"], test["direct_vqa"]["samples"], strict=True)
    if first["exact_match"] != second["exact_match"]
]
result["text_evaluation"] = {
    "two_stage_keys": [key for key in data["variants"]["all"] if key.startswith("text")],
    "direct_keys": [key for key in data["variants"]["direct_vqa"] if key.startswith("text")],
    "direct_text_retention_evidence": False,
}
assert not result["text_evaluation"]["direct_keys"]

torch.manual_seed(42)
model = MultiModalLM(TinyLM(ModelConfig(**alignment["config"])))
modalities._freeze(model, "projector")
projector_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
projector_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
assert projector_count == alignment["trainable_parameters"] == 1088
assert all(name.startswith("image_projector.") for name in projector_names)
modalities._freeze(model, "all")
all_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
assert all_count == data["variants"]["all"]["training"]["trainable_parameters"] == 145664
assert all_count == data["variants"]["direct_vqa"]["training"]["trainable_parameters"]
row = splits["train"][0]
ids, labels, count = modalities._sequence(row, ctx)
assert count == len(tok.encode(row["answer"])) + 1 == int((labels != -100).sum())
assert torch.equal(labels[labels != -100], torch.tensor(tok.encode(row["answer"]) + [tok.eos_id]))
with torch.no_grad():
    output = model(ids, labels, image=scene(row["color"], row["shape"], offset=row["offset"]))
assert int((output["labels"] != -100).sum()) == count
loss = masked_loss(output["logits"], output["labels"])
loss_again = torch.nn.functional.cross_entropy(output["logits"].reshape(-1, output["logits"].shape[-1]), output["labels"].reshape(-1), ignore_index=-100, reduction="sum") / count
assert torch.equal(loss, loss_again)
result["mask_and_parameter_probe"] = {
    "projector_trainable_names": projector_names,
    "projector_parameters": projector_count,
    "all_trainable_parameters": all_count,
    "question": row["question"],
    "answer": row["answer"],
    "answer_targets_including_eos": count,
    "prefix_masked": bool((labels[:-count] == -100).all()),
    "target_count_preserved_after_image_expansion_and_shift": True,
    "mean_cross_entropy_matches_sum_divided_by_effective_targets": True,
    "loss_value": float(loss),
    "loss_is_quality_measurement": False,
}

result["original_sources"] = [
    receipt(
        "https://arxiv.org/html/2304.08485v2",
        "arXiv:2304.08485v2, 2023-12-11",
        [
            ("Training: prediction targets", "The model is trained to predict the assistant answers", "Specifically, for a sequence"),
            ("Training: feature alignment", "In training, we keep both the visual encoder", "This stage can be understood"),
            ("Training: continuing the trained projection", "Stage 2: Fine-tuning End-to-End. We always", "We consider two specific use case scenarios"),
        ],
    ),
    receipt(
        "https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm",
        "NIST/SEMATECH e-Handbook, Randomized block designs, live HTML read on access date",
        [
            ("Nuisance factors", "Nuisance factors are those that may affect", "For example, in applying a treatment"),
            ("Controlling nuisance factors", "The basic concept is to create homogeneous blocks", "Definition of blocking factors"),
        ],
    ),
]
result["verification_complete"] = True
destination = Path(__file__).with_name("audit-output.json")
with destination.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"output": str(destination.relative_to(ROOT)), "sha256": digest(destination.read_bytes()), "training_histories": histories, "budget_match": result["budget_match"], "scores": {name: value["correct"] for name, value in test.items()}, "verification_complete": True}, ensure_ascii=False))
