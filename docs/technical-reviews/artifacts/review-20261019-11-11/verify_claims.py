"""獨立重算分組平均、人工算例與既有逐題紀錄；不重新訓練。"""

import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def averages(groups):
    micro = Fraction(sum(c for c, n in groups.values()), sum(n for c, n in groups.values()))
    macro = sum((Fraction(c, n) for c, n in groups.values()), Fraction()) / len(groups)
    return {"micro": str(micro), "macro": str(macro), "micro_float": float(micro), "macro_float": float(macro)}


def tally(samples, key):
    groups = defaultdict(lambda: {"correct": 0, "count": 0})
    for sample in samples:
        group = groups[key(sample)]
        group["count"] += 1
        group["correct"] += int(sample["generated"] == sample["target"])
    return dict(groups)


def main():
    source = ROOT / "docs/course-experiments/results/vision_ablation.json"
    data = json.loads(source.read_bytes())
    none = data["results"]["interventions"]["none"]
    samples = none["samples"]
    assert len(samples) == 12
    assert len({sample["row"] for sample in samples}) == 12
    for sample in samples:
        assert sample["exact_match"] == (sample["generated"] == sample["target"])
        assert sample["eos"] and sample["generation_error"] is None
    groups = tally(samples, lambda sample: sample["question"])
    assert groups == none["groups"]
    assert groups == {"shape?": {"correct": 3, "count": 6}, "color?": {"correct": 6, "count": 6}}
    shapes = [sample for sample in samples if sample["question"] == "shape?"]
    by_shape = tally(shapes, lambda sample: sample["target"])
    assert by_shape == {"circle": {"correct": 3, "count": 3}, "square": {"correct": 0, "count": 3}}
    assert {sample["generated"] for sample in shapes} == {"circle"}
    rates = averages({name: (group["correct"], group["count"]) for name, group in groups.items()})
    assert rates["micro"] == rates["macro"] == "3/4"
    assert none["exact_match"] == none["macro_accuracy"] == 0.75
    assert none["correct"] == sum(sample["exact_match"] for sample in samples) == 9

    cases = {
        "original": averages({"常見": (90, 90), "稀少": (0, 10)}),
        "same_total_alternative": averages({"常見": (81, 90), "稀少": (9, 10)}),
        "more_rare_questions": averages({"常見": (90, 90), "稀少": (0, 20)}),
    }
    assert cases["original"]["micro"] == "9/10" and cases["original"]["macro"] == "1/2"
    assert cases["same_total_alternative"]["micro"] == cases["same_total_alternative"]["macro"] == "9/10"
    assert cases["more_rare_questions"]["micro"] == "9/11"
    assert cases["more_rare_questions"]["macro"] == "1/2"
    assert abs(cases["more_rare_questions"]["micro_float"] - 0.8182) < 0.00005

    duplicate_example = {
        "unique_questions": {"答對題": 1, "答錯題": 0},
        "overlapping_slices": {"全部": ["答對題", "答錯題"], "答對子組": ["答對題"]},
        "unique_accuracy": "1/2",
        "naive_pooled_accuracy": "2/3",
    }
    unique = duplicate_example["unique_questions"]
    pooled = [unique[name] for names in duplicate_example["overlapping_slices"].values() for name in names]
    assert Fraction(sum(unique.values()), len(unique)) == Fraction(1, 2)
    assert Fraction(sum(pooled), len(pooled)) == Fraction(2, 3)

    command = [sys.executable, "-m", "scripts.course_experiments.run", "--help"]
    help_run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert help_run.returncode == 0
    assert "--experiment" in help_run.stdout and "--device" in help_run.stdout
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    inputs = [
        "docs/course-experiments/results/vision_ablation.json",
        "scripts/course_experiments/modalities.py",
        "scripts/course_experiments/run.py",
        "docs/course-experiments/plan.json",
        "docs/course-experiments/README.md",
        "docs/course-review-20261019/freeze-01/sources/11.11.md",
    ]
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-11/verify_claims.py",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "arithmetic": "Python Fraction；既有實驗只重算紀錄", "repository_head": revision},
        "input_sha256": {name: digest(ROOT / name) for name in inputs},
        "cases": cases,
        "dictionary_values_order": list({"常見": {"correct": 90}, "稀少": {"correct": 0}}.values()),
        "empirical_record": {
            "metadata": {key: data[key] for key in ("experiment_id", "revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
            "training_steps": data["results"]["training"]["steps"],
            "split_policy": data["results"]["data"]["split_policy"],
            "splits": {name: {"count": split["count"], "sha256": split["sha256"]} for name, split in data["results"]["data"]["splits"].items()},
            "samples": [{key: sample[key] for key in ("row", "family", "question", "target", "generated", "exact_match", "eos", "generation_error")} for sample in samples],
            "recomputed_groups": groups,
            "recomputed_shape_subgroups": by_shape,
            "recomputed_averages": rates,
            "shape_generated_values": sorted({sample["generated"] for sample in shapes}),
            "scope": "重算已保存的單次實驗紀錄，未重新生成模型答案；不驗證未見圖片或不同種子的品質。",
        },
        "overlap_counterexample": duplicate_example,
        "one_success_reasoning": {"bernoulli_success_probability": 0.5, "probability_of_one_success_in_one_trial": 0.5, "conclusion": "一次成功與僅一半成功率的母體相容，不能據此保證高可靠性。"},
        "experiment_cli": {"argv": command, "exit_code": help_run.returncode, "stdout": help_run.stdout, "stderr": help_run.stderr},
        "result": "全部斷言通過。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
