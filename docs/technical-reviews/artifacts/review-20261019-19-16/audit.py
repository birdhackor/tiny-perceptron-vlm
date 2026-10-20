"""核對候選評分、有限世界標籤與既有實驗收據；不重新訓練。"""

import ast
import hashlib
import inspect
import itertools
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.lesson_products import data, evaluation, inference, model, train, training

torch.set_num_threads(1)
torch.manual_seed(0)
tokenizer = model.Tokenizer()
config = model.Config(len(tokenizer), width=64, ff=128)
scorer = model.CandidateScorer(config).eval()


def load(path):
    return json.loads((ROOT / path).read_bytes())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


with torch.inference_mode():
    outputs = {}
    for question, candidates in data.CANDIDATES.items():
        row = {"state": "卡片進度;申請編號=?", "question": question, "candidates": list(candidates)}
        tokens, slices, maximum, _ = model.candidate_batch([row], tokenizer, config)
        scores = model.grouped_scores(scorer(tokens), slices, maximum)[0]
        p = scores.softmax(-1)
        largest = 0.0
        for permutation in itertools.permutations(candidates):
            batch = model.candidate_batch([{**row, "candidates": list(permutation)}], tokenizer, config)
            alternate = model.grouped_scores(scorer(batch[0]), batch[1], batch[2])[0].softmax(-1)
            ordered = torch.tensor([alternate[permutation.index(c)] for c in candidates])
            largest = max(largest, float((ordered - p).abs().max()))
        assert largest < 1e-6, "候選重排後機率有差異"
        outputs[question] = {"permutations": math.factorial(len(candidates)), "max_probability_difference": largest}
    row = {"state": "卡片進度;申請編號=?", "question": "choice", "candidates": ["card", "address", "app", "none"]}
    demonstration_candidates = row["candidates"]
    tokens, slices, maximum, _ = model.candidate_batch([row], tokenizer, config)
    demonstration = model.grouped_scores(scorer(tokens), slices, maximum)[0].softmax(-1).tolist()
    independent = []
    for i in range(len(tokens)):
        trimmed = tokens[i][tokens[i] != 0][None]
        independent.append(float(scorer(trimmed)[0]))
    independent_difference = float((scorer(tokens) - torch.tensor(independent)).abs().max())
    assert independent_difference < 1e-6, "個別候選與批次分數有差異"
    grouped = model.grouped_scores(torch.tensor([1., 2., 3., 4., 5.]), [(0, 2), (2, 5)], 3)
    assert grouped[0, 2].isneginf(), "不存在的候選應被遮罩"

invalid_cases = [
    ("choice", ["card", "address"]), ("choice", ["none", "none"]),
    ("choice", ["unknown", "none"]), ("unknown", ["yes", "no"]),
    ("rubric", ["grade0", "grade3"]),
]
rejected = 0
for question, candidates in invalid_cases:
    try:
        model.validate_candidates(question, candidates)
    except ValueError:
        rejected += 1
assert rejected == len(invalid_cases), "候選驗證未拒絕不合法輸入"

worlds = [
    {"task": "card", "application": "123456"}, {"task": "address"},
    {"task": "app", "device": "手機"}, {"task": "sum", "x": 0, "y": 1, "z": 2},
]
labels = []
for world in worlds:
    needed = data.REQUIRED[world["task"]]
    for count in range(len(needed) + 1):
        omitted = needed[:count]
        available = len(needed) - count
        expected_info = "yes" if available == len(needed) else "no"
        expected_grade = "grade2" if available == len(needed) else "grade0" if available == 0 else "grade1"
        actual = {kind: data.judgment(world, omitted, kind, data.CANDIDATES[kind]) for kind in ("info", "rubric")}
        assert actual == {"info": expected_info, "rubric": expected_grade}, "必要欄位標籤不符"
        labels.append({"task": world["task"], "omitted": omitted, **actual})

counts = Counter()
observations = defaultdict(lambda: {"splits": set(), "labels": set()})
salt = data.choose_salt()
box_worlds = (w for w in data.box_worlds() if len(w["events"]) <= 4 and int(data.digest(w["family_id"])[:8], 16) % 5 == 0)
missing_count = 0
for world in itertools.chain(box_worlds, data.arithmetic_worlds(), data.service_worlds()):
    split = data.split_of(world["family_id"], salt)
    for row in data.c_rows(world, split):
        counts[f'{split}/{row["question"]}'] += 1
        key = data.observable_key(row, "c")
        observations[key]["splits"].add(split)
        observations[key]["labels"].add(row["gold"])
        missing_count += int(split == "test" and row.get("missing_best", False))
manifest = load("docs/lesson-products/data-v2-manifest.json")
assert dict(counts) == manifest["audit"]["c"]["counts"], "重建資料分母不符"
shared = sum(len(v["splits"]) > 1 for v in observations.values())
assert shared == manifest["audit"]["c"]["cross_split_shared_observations"], "跨切分重複數量不符"
assert all(len(v["labels"]) == 1 for v in observations.values()), "重複輸入存在衝突標籤"

receipt = load("docs/lesson-products/results/initial-c-execution.json")
run = receipt["training"]
assert (run["seed"], run["parameters"], run["last_step"], run["selected_step"]) == (9041, 84545, 800, 800)
assert run["random_initialization"] and run["weights_changed"] and run["environment"]["gpu"] == "NVIDIA L4"
policies = [r for r in run["history"] if "policy" in r]
best = max(policies, key=lambda r: (r["policy"]["choice"]["accuracy"], -r["policy"]["choice"]["cross_entropy"]))
assert best["step"] == 800, "決策資料選出的權重步數不符"

cal = load("docs/lesson-products/results/c-calibration.json")
assert cal["temperatures"] == {q: .25 for q in data.CANDIDATES}
freeze = load("docs/lesson-products/results/settings-freeze-v1.json")
package = load("docs/lesson-products/results/c-artifact-manifest.json")
assert package["settings_freeze_sha256"] == sha("docs/lesson-products/results/settings-freeze-v1.json")
assert package["files"]["calibration.json"]["sha256"] == sha("docs/lesson-products/results/c-calibration.json")
assert freeze["calibration"] == cal
for name in ("data.py", "model.py", "training.py", "inference.py", "evaluation.py"):
    assert freeze["source_sha256s"][name] == sha("tiny_perceptron/lesson_products/" + name), "凍結程式與現行程式不符"

report_path = "docs/lesson-products/results/final-report-v1.json"
report = load(report_path)
for question, expected in (("choice", 11797), ("info", 10890), ("rubric", 10890)):
    for version in ("before", "after"):
        v = report["c"][question][version]
        assert v["correct"] == v["count"] == expected
assert report["c_reorder"] == {"count": 11797, "consistent": 11797, "consistency": 1.0}
assert report["c_missing_best"] == {"count": 849, "correct": 849} and missing_count == 849
shift = report["shifts"]["c_wording"]
assert shift["choice"]["after"]["correct"] == 11493
assert shift["info"]["after"]["correct"] == shift["rubric"]["after"]["correct"] == 10773
assert abs(report["c"]["choice"]["before"]["cross_entropy"] - .000550) < .0000005
assert abs(report["c"]["choice"]["after"]["cross_entropy"] - .00000814) < .000000005
assert abs(shift["choice"]["before"]["cross_entropy"] - .183) < .0005
assert abs(shift["choice"]["after"]["cross_entropy"] - .698) < .0005

failures = load("docs/lesson-products/results/observed-failure-examples-v1.json")
assert failures["report_sha256"] == sha(report_path)
assert failures["raw_predictions_sha256"] == report["predictions"]["sha256"]
example = failures["examples"][0]
scores = torch.tensor(example["raw_scores"], dtype=torch.float64) / cal["temperatures"]["choice"]
probabilities = scores.softmax(-1)
selected = example["candidates"][int(probabilities.argmax())]
concentration = (len(probabilities) * float(probabilities.max()) - 1) / (len(probabilities) - 1)
assert selected == example["predicted"] == "boxes" and example["gold"] == "none"
assert math.isclose(concentration, example["concentration"], abs_tol=1e-14)
wrong_ce = {str(t): float(-torch.log_softmax(torch.tensor(example["raw_scores"], dtype=torch.float64) / t, -1)[example["candidates"].index("none")]) for t in (1., .25)}

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.16.md").read_text()
fences = frozen.split("```python\n")
snippet = fences[2].split("```")[0]
assert ast.dump(ast.parse(snippet).body[0], include_attributes=False) == ast.dump(ast.parse(inspect.getsource(model.CandidateScorer)).body[0], include_attributes=False), "課文節錄與原始類別不同"
assert sum(p.numel() for p in scorer.parameters()) == 84545
assert len(scorer.blocks) == 2
assert not ({id(p) for p in scorer.blocks[0].parameters()} & {id(p) for p in scorer.blocks[1].parameters()})
tie_rows = [{"candidates": ["none", "card"]}, {"candidates": ["card", "none"]}]
assert evaluation.chosen_meanings(torch.zeros((2, 2)), tie_rows) == ["card", "card"]

source_paths = [
    "tiny_perceptron/lesson_products/" + name for name in ("model.py", "data.py", "inference.py", "evaluation.py", "training.py", "train.py")
] + [
    "docs/lesson-products/results/" + name for name in ("initial-c-execution.json", "c-calibration.json", "c-artifact-manifest.json", "settings-freeze-v1.json", "final-report-v1.json", "observed-failure-examples-v1.json")
] + ["docs/lesson-products/data-v2-manifest.json", "docs/lesson-products/data-v2-token-audit.json"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-16/audit.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads())},
    "result": "全部斷言通過",
    "source_sha256": {p: sha(p) for p in source_paths},
    "architecture": {"parameters": sum(p.numel() for p in scorer.parameters()), "vocabulary": len(tokenizer), "width": config.width, "ff": config.ff, "layers": len(scorer.blocks), "snippet_matches": True},
    "permutation": outputs,
    "demonstration": dict(zip(demonstration_candidates, demonstration)),
    "independent_batch_difference": independent_difference,
    "invalid_candidates_rejected": rejected,
    "label_examples": labels,
    "regenerated_counts": dict(counts),
    "regenerated_shared_observations": shared,
    "regenerated_missing_best": missing_count,
    "archived_training": {k: run[k] for k in ("random_initialization", "seed", "last_step", "selected_step", "parameters", "environment", "batch_size", "data_manifest_sha256")},
    "archived_policy_best": best,
    "archived_calibration": cal,
    "archived_test": {"c": report["c"], "reorder": report["c_reorder"], "missing": report["c_missing_best"], "wording_shift": shift, "predictions": report["predictions"]},
    "failure_recalculated": {"state": example["state"], "gold": example["gold"], "predicted": selected, "probability": float(probabilities.max()), "concentration": concentration, "CE_by_temperature": wrong_ce},
    "limits": "訓練與完整測試成績取自既有原始執行收據；本次重新產生資料分母、執行候選評分機制，並從保存的分數重算一筆錯誤例。未載入成品權重、未讀完整逐題預測檔、未重新訓練或重跑全量測試。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
