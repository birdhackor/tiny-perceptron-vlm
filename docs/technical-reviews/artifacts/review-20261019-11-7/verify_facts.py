import contextlib
import hashlib
import io
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.course_experiments.common import split_records
from scripts.course_experiments.modalities import _freeze, _hash, _loss_fn, _vision_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def read_original(url, phrases, locator, note):
    command = ["curl", "-fsSL", url]
    raw = subprocess.check_output(command)
    parser = TextParser()
    parser.feed(raw.decode())
    plain = " ".join(" ".join(parser.parts).split())
    for phrase in phrases:
        assert phrase in plain, phrase
    return {
        "url": url,
        "command": command,
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": locator,
        "excerpts": phrases,
        "inspection_note": note,
    }


class CountingModel:
    def __init__(self):
        self.text_counts = []
        self.image_counts = []

    def language(self, x):
        self.text_counts.append(x.shape[1])
        return {"logits": torch.zeros((*x.shape, ByteTokenizer.vocab_size))}

    def __call__(self, ids, labels, image=None, waveform=None):
        self.image_counts.append(int((labels != -100).sum()))
        return {
            "logits": torch.zeros((1, len(ids), ByteTokenizer.vocab_size)),
            "labels": labels[None],
        }


torch.set_num_threads(1)
report_path = Path("docs/course-experiments/results/vqa.json")
report = json.loads(report_path.read_bytes())
source_path = Path("docs/course-review-20261019/freeze-01/sources/11.7.md")
source = source_path.read_text()
source_code = source.split("```python\n", 1)[1].split("```", 1)[0]
namespace = {}
example_output = io.StringIO()
with contextlib.redirect_stdout(example_output):
    exec(compile(source_code, str(source_path), "exec"), namespace)
assert namespace["total_positions"] == 1000
assert namespace["ratios"] == {"文字有效目標": 0.75, "圖文有效目標": 0.25}
exercise_output = io.StringIO()
exercise_code = source_code.replace('750, "圖文有效目標": 250', '500, "圖文有效目標": 500')
exercise_namespace = {}
with contextlib.redirect_stdout(exercise_output):
    exec(compile(exercise_code, str(source_path), "exec"), exercise_namespace)
assert exercise_namespace["ratios"] == {"文字有效目標": 0.5, "圖文有效目標": 0.5}
assert exercise_namespace["recorded"] == namespace["recorded"]

text_splits = split_records(generate_records("attributes-sft"), seed=42)
vision = _vision_records(("shape?", "color?"))
for split, rows in text_splits.items():
    assert len(rows) == report["results"]["text_data"][split]["count"]
    assert _hash(rows) == report["results"]["text_data"][split]["sha256"]
for first, second in (("train", "test"), ("train", "validation"), ("validation", "test")):
    assert not ({r["family"] for r in text_splits[first]} & {r["family"] for r in text_splits[second]})
assert len(vision["test"]) == 12
assert len(text_splits["test"]) == 10
assert any(r["messages"][0]["content"] == "color=blue;shape=circle;pitch=low;pitch?" and r["messages"][-1]["content"] == "low" for r in text_splits["test"])

variants = {}
expected = {"projector_only": (3830, 3, 5), "partial": (3830, 12, 2), "all": (3830, 9, 1), "all_replay": (4214, 9, 4)}
tok = ByteTokenizer()
for name, (targets, image_correct, text_correct) in expected.items():
    variant = report["results"]["variants"][name]
    training = variant["training"]
    assert training["steps"] == len(training["history"]) == 160
    assert sum(row["effective_targets"] for row in training["history"]) == targets == training["effective_targets"] == training["effective_tokens"]
    test = variant["test"]
    assert test["correct"] == sum(s["exact_match"] for s in test["samples"]) == image_correct
    assert test["examples"] == len(test["samples"]) == 12
    for sample, row in zip(test["samples"], vision["test"], strict=True):
        assert sample["family"] == row["family"] and sample["question"] == row["question"]
        raw = sample["generated_ids"]
        raw = raw[:raw.index(tok.eos_id)] if tok.eos_id in raw else raw
        assert (raw == tok.encode(row["answer"])) == sample["exact_match"]
    for phase, correct in (("text_before", 5), ("text_after", text_correct)):
        evaluation = variant[phase]
        assert evaluation["records"] == len(evaluation["samples"]) == 10
        assert evaluation["matches"] == sum(s["exact"] for s in evaluation["samples"]) == correct
        for sample, row in zip(evaluation["samples"], text_splits["test"], strict=True):
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == row["messages"][-1]["content"]
            raw = sample["generated_ids"]
            raw = raw[:raw.index(tok.eos_id)] if tok.eos_id in raw else raw
            assert (raw == tok.encode(sample["expected"])) == sample["exact"]
    counting = CountingModel()
    loss = _loss_fn(counting, vision["train"], SimpleNamespace(seed=42, device="cpu"), replay=text_splits["train"], replay_ratio=variant["replay_probability_per_example"])
    observed_counts = [loss(step)[1] for step in range(160)]
    assert observed_counts == [r["effective_targets"] for r in training["history"]]
    variants[name] = {
        "steps": 160,
        "batch_size": 4,
        "examples_drawn": len(counting.text_counts) + len(counting.image_counts),
        "effective_targets": targets,
        "image_test": f"{image_correct}/12",
        "text_before": "5/10",
        "text_after": f"{text_correct}/10",
        "replay_probability_per_example": variant["replay_probability_per_example"],
        "text_draws": len(counting.text_counts),
        "image_draws": len(counting.image_counts),
    }
    assert variants[name]["examples_drawn"] == 640

sampled = {"text": [], "image": []}
for step in range(160):
    rng = random.Random(42 + step)
    for _ in range(4):
        if rng.random() < 0.5:
            row = rng.choice(text_splits["train"])
            _, labels = render_chat(row["messages"])
            sampled["text"].append(int((labels != -100).sum()))
        else:
            row = rng.choice(vision["train"])
            sampled["image"].append(len(tok.encode(row["answer"])) + 1)
sample_statistics = {k: {"draws": len(v), "effective_targets": sum(v), "mean_targets": sum(v) / len(v)} for k, v in sampled.items()}
assert sum(v["effective_targets"] for v in sample_statistics.values()) == 4214
assert sample_statistics["text"]["mean_targets"] > sample_statistics["image"]["mean_targets"]
assert 4214 - 3830 == 384

replay_variant = report["results"]["variants"]["all_replay"]
transitions = []
for before, after in zip(replay_variant["text_before"]["samples"], replay_variant["text_after"]["samples"], strict=True):
    if before["exact"] != after["exact"]:
        transitions.append({"prompt": before["messages"][0]["content"], "expected": before["expected"], "before": before["generated"], "after": after["generated"], "before_correct": before["exact"], "after_correct": after["exact"]})
assert sum(t["before_correct"] for t in transitions) == 2
assert sum(t["after_correct"] for t in transitions) == 1
assert {(t["prompt"].rsplit(";", 1)[1], t["before"], t["after"]) for t in transitions} == {("color?", "blow", "red"), ("pitch?", "high", "cire"), ("joint?", "square,high", "red")}

freeze_checks = {}
for scope in ("projector", "partial", "all"):
    model = MultiModalLM(TinyLM(ModelConfig(width=16, layers=2, heads=2, max_length=128)))
    _freeze(model, scope)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if scope == "projector":
        assert all(n.startswith("image_projector.") for n in trainable)
    elif scope == "partial":
        assert all(n.startswith(("image_projector.", "language.blocks.1.")) for n in trainable)
    else:
        assert all(p.requires_grad for p in model.parameters())
    freeze_checks[scope] = sorted(trainable)

paths = ["scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"]
code_matches = {}
for path in paths:
    current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert report["code_sha256"][path] == current
    code_matches[path] = current

originals = [
    read_original("https://arxiv.org/html/1706.08840v6", ["negative backward transfer when learning about some task", "we consider access to a test set for each"], "Training Protocol and Evaluation Metrics; Backward Transfer", "核對負向 backward transfer 的定義，以及每項任務使用測試集、逐次評估的做法。固定初始函數一致不保證更新後一致，依這個評估定義逐別檢查。"),
    read_original("https://arxiv.org/abs/1902.10486v4", ["jointly trains on both examples from the current task as well as examples stored in the episodic memory"], "Abstract", "核對回放是舊樣本與目前任務樣本一起訓練；摘要中的特定 benchmark 效益不外推到本專案。"),
    read_original("https://scikit-learn.org/1.9/common_pitfalls.html", ["Never include test data when using the fit and fit_transform methods."], "Data leakage / How to avoid data leakage", "核對訓練、測試分離是估計留出題表現的條件，不要求本專案使用 scikit-learn。"),
]

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "recorded_training": {k: report[k] for k in ("seed", "python_version", "torch_version", "gpu", "revision")}},
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "experiment_result_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "example_stdout": example_output.getvalue(),
    "exercise_stdout": exercise_output.getvalue(),
    "split_counts": {k: len(v) for k, v in text_splits.items()},
    "split_hashes": {k: _hash(v) for k, v in text_splits.items()},
    "split_overlap": False,
    "variants": variants,
    "replay_target_statistics": sample_statistics,
    "extra_targets": 384,
    "replay_transitions": transitions,
    "trainable_names": freeze_checks,
    "code_matches_recorded_training": code_matches,
    "mixture_derivation": {"easy_loss": 1, "hard_loss": 3, "equal_mix": 0.5 * 1 + 0.5 * 3, "easy_heavy_mix": 0.75 * 1 + 0.25 * 3, "within_group_losses_changed": False},
    "original_sources": originals,
    "scope": "重算既有實驗紀錄，執行原始抽樣與計數函數；計數模型使用零 logits，不重訓或重做 GPU 生成，不能驗證權重來源的獨立真實性或多 seed 效益。",
    "result": "所有斷言成立",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
