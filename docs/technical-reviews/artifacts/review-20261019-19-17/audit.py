"""核對規則、服務控制流程與公開逐題預測；不訓練或改寫成品。"""

import ast
import gzip
import hashlib
import inspect
import io
import itertools
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch

from tiny_perceptron.lesson_products import data, inference


ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
RECEIPTS = []


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "120", url]
    run = subprocess.run(command, capture_output=True, check=True)
    RECEIPTS.append({"command_argv": command, "exit_code": run.returncode, "url": url,
                     "bytes": len(run.stdout), "sha256": digest(run.stdout),
                     "accessed_at": datetime.now(UTC).isoformat()})
    return run.stdout


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


class FakeB:
    def __init__(self, policy):
        self.auto_policy = policy
        self.calls = []

    def synchronize(self):
        pass

    def solve(self, state, loops):
        self.calls.append({"state": state, "loops": loops})
        return {"answer": 2, "loops": loops, "seconds": 0.003}


class FakeC:
    def __init__(self, task="boxes", info="yes"):
        self.task, self.info, self.calls = task, info, []

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {"selected": self.task if question == "choice" else self.info, "seconds": 0.001}


def main():
    policy_path = "docs/lesson-products/results/b-auto-policy.json"
    policy = read_json(policy_path)
    freeze = read_json("docs/lesson-products/results/settings-freeze-v1.json")
    report = read_json("docs/lesson-products/results/final-report-v1.json")
    manifest = read_json("docs/lesson-products/data-v2-manifest.json")
    release = read_json("tiny_perceptron/lesson_products/release.json")
    base = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/"
    public_policy = fetch(base + "b/auto-policy.json")
    assert public_policy == (ROOT / policy_path).read_bytes()
    assert digest(public_policy) == freeze["products"]["b"]["auto-policy.json"]

    source = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.17.md").read_text()
    block = source.split("```python\n", 1)[1].split("```", 1)[0]
    assert ast.dump(ast.parse(block)) == ast.dump(ast.parse(inspect.getsource(inference.rule_depth)))
    assert len(inspect.getsource(inference.rule_depth).splitlines()) == 10
    rule_examples = {str(n): inference.rule_depth(policy, "boxes", n) for n in range(1, 7)}
    assert rule_examples == {"1": 3, "2": 3, "3": 3, "4": 3, "5": 5, "6": 5}
    assert inference.rule_depth(policy, "sum") == 1
    errors = {}
    for label, bad in [("version", {**policy, "version": 2}), ("depth", {**policy, "sum_depth": 2})]:
        try:
            inference.rule_depth(bad, "sum")
        except ValueError as error:
            errors[label] = str(error)
        else:
            raise AssertionError("無效設定未被拒絕")

    state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
    probes = []
    for task, info, effort, cap, status, loops in [
        ("card", "yes", "auto", 5, "unsupported", 0),
        ("boxes", "no", "high", 5, "clarification_needed", 0),
        ("boxes", "yes", "low", 5, "answered", 1),
        ("boxes", "yes", "medium", 5, "answered", 3),
        ("boxes", "yes", "high", 5, "answered", 5),
        ("boxes", "yes", "medium", 2, "budget_limited", 2),
        ("boxes", "yes", "auto", 5, "answered", 3),
        ("sum", "yes", "auto", 5, "answered", 1),
    ]:
        b, c = FakeB(policy), FakeC(task, info)
        text = "三數加法;x=1;y=2;z=3" if task == "sum" else state
        result = inference.auto_solve(b, c, text, effort=effort, max_loops=cap)
        assert result["status"] == status and result["b_loops"] == loops
        assert c.calls == ["choice", "info"]
        assert len(b.calls) == int(loops > 0)
        assert result["c_seconds"] == 0.002
        probes.append({"inputs": [task, info, effort, cap], "result": result,
                       "judge_calls": c.calls, "solver_calls": b.calls})

    missing_worlds = [{"task": "boxes", "a": a, "b": 6, "events": [("swap", 0)], "query": "B"}
                      for a in range(8)]
    missing_answers = [data.answer(world) for world in missing_worlds]
    assert missing_answers == list(range(8))
    complete_world = missing_worlds[2]
    assert data.judgment(complete_world, (), "info", ["yes", "no"]) == "yes"
    assert data.judgment(complete_world, ("a",), "info", ["yes", "no"]) == "no"
    omitted_rows = list(data.c_rows({**complete_world, "family_id": "probe"}, "train"))
    assert {row["gold"] for row in omitted_rows if row["question"] == "choice"} == {"boxes", "none"}
    assert data.CANDIDATES["choice"] == ("card", "address", "app", "boxes", "sum", "none")

    # 重新列舉世界及測試題，逐題比對公開預測的ID、標籤與深度讀出。
    salt = data.choose_salt()
    assert salt == manifest["config"]["split_salt"]
    worlds = list(data.box_worlds()) + list(data.arithmetic_worlds())
    expected = {}
    splits = Counter()
    lengths = defaultdict(Counter)
    for world in worlds:
        long = world["task"] == "boxes" and len(world["events"]) > 4
        split = "long_shift" if long else data.split_of(world["family_id"], salt)
        splits[(split, world["task"])] += 1
        if world["task"] == "boxes":
            lengths[split][len(world["events"])] += 1
        if split not in ("test", "long_shift"):
            continue
        text = data.render(world)
        identifier = data.digest((world["family_id"], text))
        expected[identifier] = {"world": world, "gold": data.answer(world), "state": text, "split": split}
    assert splits[("test", "boxes")] == 2956 and splits[("test", "sum")] == 42

    grid = freeze["auto_search"]
    grid_keys = {(row["rule"]["low_max_events"], row["rule"]["medium_max_events"], row["rule"]["sum_depth"]) for row in grid}
    expected_keys = {(a, b, depth) for a, b, depth in itertools.product(range(5), range(5), (1, 3, 5)) if a <= b}
    assert grid_keys == expected_keys and len(grid) == len(grid_keys) == 45
    feasible = [row for row in grid if row["quality"]["mean_actual_b_loops_all_complete"] <= 3]
    selected = min(feasible, key=lambda row: (-row["quality"]["accuracy_end_to_end"],
                                            row["quality"]["mean_actual_b_loops_all_complete"],
                                            json.dumps(row["rule"], sort_keys=True)))
    assert all(policy[key] == value for key, value in selected["rule"].items())
    assert policy["policy_quality"] == selected["quality"]
    for name, recorded in freeze["source_sha256s"].items():
        assert digest((ROOT / "tiny_perceptron/lesson_products" / name).read_bytes()) == recorded

    # 大型第三方下載只在記憶體核對，不保存全文、權重或預測副本。
    raw = fetch(base + "raw/final-report-v1.predictions.jsonl.gz")
    package = read_json("docs/lesson-products/results/publication-package-v1.json")
    assert digest(raw) == package["files"]["raw/final-report-v1.predictions.jsonl.gz"]["sha256"]
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    seen = set()
    counts, correct, loop_totals = Counter(), Counter(), Counter()
    long_correct, long_count = Counter(), Counter()
    score_checks = 0
    changed = 0
    changed_correct = Counter()
    for line in gzip.GzipFile(fileobj=io.BytesIO(raw)):
        raw_digest.update(line)
        raw_bytes += len(line)
        row = json.loads(line)
        if row["product"] != "b" or row["split"] not in ("test", "long_shift"):
            continue
        original = expected[row["id"]]
        assert row["gold"] == original["gold"] and row["split"] == original["split"]
        assert row["family_id"] == original["world"]["family_id"] and row["task"] == original["world"]["task"]
        assert row["id"] not in seen
        seen.add(row["id"])
        if row["split"] == "long_shift":
            length = len(original["world"]["events"])
            predicted = max(range(len(row["raw_scores"])), key=lambda i: row["raw_scores"][i])
            long_count[length] += 1
            long_correct[length] += predicted == row["gold"]
            continue
        assert row["c_choice"] == row["task"] and row["c_info"] == "yes"
        for depth, scores in row["scores_by_depth"].items():
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            assert predicted == row["answers_by_depth"][depth]
            score_checks += 1
        task = row["task"]
        counts[task] += 1
        event_count = len(original["world"]["events"]) if task == "boxes" else 0
        auto_depth = inference.rule_depth(policy, task, event_count)
        for effort, depth in [("low", 1), ("medium", 3), ("high", 5), ("auto", auto_depth)]:
            correct[effort] += row["answers_by_depth"][str(depth)] == row["gold"]
            loop_totals[effort] += depth
        if auto_depth != 3:
            assert task == "sum" and auto_depth == 1
            changed += 1
            for depth in (1, 3):
                changed_correct[str(depth)] += row["answers_by_depth"][str(depth)] == row["gold"]
    assert seen == set(expected)
    assert raw_digest.hexdigest() == report["predictions"]["sha256"]
    assert raw_bytes == report["predictions"]["bytes"]
    for effort in correct:
        assert correct[effort] == report["auto"][effort]["correct_end_to_end"]
        assert loop_totals[effort] / sum(counts.values()) == report["auto"][effort]["mean_actual_b_loops_all_complete"]
    assert changed == 42 and loop_totals["medium"] - loop_totals["auto"] == 84
    assert correct["medium"] - correct["auto"] == 1
    assert long_correct == {5: 804, 6: 441} and long_count == {5: 1600, 6: 1600}

    exercise_policy = {**policy, "medium_max_events": 2}
    exercise = {}
    for split in ("policy", "test"):
        total = sum(inference.rule_depth(exercise_policy, world["task"], len(world.get("events", ())))
                    for world in worlds if world["task"] == "sum" and data.split_of(world["family_id"], salt) == split)
        total += sum(inference.rule_depth(exercise_policy, "boxes", length) * count for length, count in lengths[split].items())
        count = splits[(split, "sum")] + splits[(split, "boxes")]
        exercise[split] = {"count": count, "total_loops": total, "mean_loops": total / count}
        assert total / count > 3

    sources = [
        ("guo", "https://proceedings.mlr.press/v70/guo17a.html", "Confidence calibration", 1250),
        ("nist", "https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/binoprop.htm", "Description:", 2500),
        ("routellm", "https://arxiv.org/html/2406.18665v3", "In this work, we focus on routing", 7000),
    ]
    inspections = []
    for identifier, url, start, length in sources:
        html = fetch(url)
        parsed = Text()
        parsed.feed(html.decode())
        text = " ".join(" ".join(parsed.parts).split())
        index = text.find(start)
        assert index >= 0
        inspected = text[index:index + length]
        print(identifier, inspected, flush=True)
        inspections.append({"id": identifier, "url": url, "full_response_sha256": digest(html),
                            "excerpt": " ".join(inspected.split()[:22]),
                            "locator": start, "inspected_characters": len(inspected)})

    return {
        "status": "passed",
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m docs.technical-reviews.artifacts.review-20261019-19-17.audit",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                        "platform": platform.platform()},
        "source_sha256s": {str(path): digest((ROOT / path).read_bytes()) for path in [
            "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/model.py",
            "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/evaluation.py",
            "docs/lesson-products/results/final-report-v1.json", "docs/lesson-products/results/settings-freeze-v1.json",
            policy_path, "docs/lesson-products/data-v2-manifest.json", "tiny_perceptron/lesson_products/release.json",
            "docs/lesson-products/results/publication-package-v1.json"]},
        "public_policy_sha256": digest(public_policy), "rule_examples": rule_examples,
        "effort": inference.EFFORT, "invalid_rule_errors": errors, "service_probes": probes,
        "missing_a_possible_final_b": missing_answers, "required": data.REQUIRED, "choice_candidates": data.CANDIDATES["choice"],
        "split_counts": {"/".join(key): value for key, value in sorted(splits.items())},
        "box_length_counts": {key: dict(value) for key, value in lengths.items()},
        "grid": {"count": len(grid), "feasible_count": len(feasible), "selected": selected},
        "test": {"task_counts": counts, "correct": correct, "total_loops": loop_totals,
                 "changed_cases": changed, "changed_case_correct": changed_correct,
                 "per_depth_argmax_checks": score_checks, "raw_bytes": raw_bytes,
                 "raw_sha256": raw_digest.hexdigest(), "verified_ids": len(seen)},
        "long_shift": {"correct": long_correct, "count": long_count}, "exercise": exercise,
        "latency": report["latency"], "external_inspections": inspections, "http_receipts": RECEIPTS,
        "limitations": ["控制流程使用替身，未將此當成模型品質證據。",
                        "從公開逐題logits重算成績，未重跑訓練或全部權重推論。",
                        "規則搜尋以凍結的開發集測量重算選擇，未重新推論開發集。",
                        "時間為原始CPU紀錄；未在本機重現延遲，也未外推到其他裝置。"],
    }


if __name__ == "__main__":
    result = main()
    destination = HERE / "audit-execution.json"
    with destination.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(destination.relative_to(ROOT)),
                      "test": result["test"], "exercise": result["exercise"]}, ensure_ascii=False))
