"""Fresh CPU verification for lesson 13.10; no course or environment mutation."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import records_sha256
from scripts.course_experiments.posttraining import CONFIG, build_records, split_records
from tiny_perceptron.posttraining import FiniteRewardModel, preference_loss

ROOT = Path.cwd()
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_10"
RUN = ARTIFACTS / f"{PREFIX}_cpu_run"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(code):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "lesson13.10", "exec"), {})
    return output.getvalue()


def check_rows(report, generated):
    audited = {}
    for split_name, rows in generated.items():
        saved = report["results"]["evaluations"][split_name]
        assert len(saved["rows"]) == len(rows)
        comparisons = []
        successes = {name: 0 for name in ("sft", "ppo", "dpo")}
        for original, observed in zip(rows, saved["rows"], strict=True):
            for key, value in original.items():
                assert observed[key] == value, (split_name, original["family"], key)
            a, b = original["operands"]
            assert 1 <= a <= b <= 10
            assert original["candidates"][0] == str(a + b)
            assert original["candidates"][2] == str(a + b + 1)
            mode = original["mode"]
            best = {"number": 0, "explain": 1, "missing": 3}[mode]
            assert original["expected_action"] == best
            if mode == "missing":
                expected_pairs = [[3, 0], [3, 1], [3, 2]]
                assert original["candidates"][3] == "請補充購買數量。"
            else:
                ranking = [0, 1, 3, 2] if mode == "number" else [1, 0, 3, 2]
                expected_pairs = [[winner, loser] for i, winner in enumerate(ranking) for loser in ranking[i + 1 :]]
            assert original["preference_pairs"] == expected_pairs
            scores = observed["reward_model_raw_scores"]
            assert len(scores) == 4 and all(math.isfinite(v) for v in scores)
            assert observed["reward_model_best_action"] == max(range(4), key=lambda k: scores[k])
            scale = report["results"]["reward"]["fixed_normalization_scale_from_train"]
            for raw, normalized in zip(scores, observed["reward_model_normalized_scores"], strict=True):
                assert abs((raw - sum(scores) / 4) / scale - normalized) < 2e-6
            for winner, loser in expected_pairs:
                comparisons.append({
                    "family": original["family"], "mode": mode, "winner": winner, "loser": loser,
                    "gap": scores[winner] - scores[loser], "win": scores[winner] > scores[loser],
                })
            for name, prediction in observed["policies"].items():
                probabilities = prediction["probabilities"]
                assert abs(sum(probabilities) - 1) < 1e-6
                assert all(0 <= p <= 1 for p in probabilities)
                chosen = max(range(4), key=lambda k: probabilities[k])
                assert chosen == prediction["chosen_action"]
                assert prediction["chosen_response"] == original["candidates"][chosen]
                assert prediction["full_request_success"] == (chosen == best)
                successes[name] += chosen == best
        wins = sum(item["win"] for item in comparisons)
        stored_metric = saved["reward_model_pairwise_accuracy"]
        assert stored_metric["numerator"] == wins
        assert stored_metric["denominator"] == len(comparisons)
        assert stored_metric["rate"] == wins / len(comparisons)
        for name, count in successes.items():
            assert saved["policies"][name]["greedy_full_request_success"]["numerator"] == count
            assert saved["policies"][name]["greedy_full_request_success"]["denominator"] == len(rows)
        audited[split_name] = {
            "families": sorted({row["family"] for row in rows}),
            "contexts": len(rows), "pair_count": len(comparisons), "rm_wins": wins,
            "policy_success_counts": successes, "all_pair_checks": comparisons,
            "records_sha256": records_sha256(rows),
        }
        assert report["results"]["splits"][split_name]["sha256"] == records_sha256(rows)
    return audited


def main():
    torch.set_num_threads(2)
    source = ROOT / "course/chapters/13.md"
    body = dict(sections(source))["13.10"]
    (ARTIFACTS / f"{PREFIX}_lesson.md").write_text(body, encoding="utf-8")
    code = re.findall(r"```python\n(.*?)\n```", body, re.S)[0]
    examples = {
        "original": execute(code),
        "original_shift_10": execute(
            code.replace("chosen = torch.tensor([preferred_score])", "chosen = torch.tensor([preferred_score + 10.0])")
            .replace("rejected = torch.tensor([0.0])", "rejected = torch.tensor([10.0])")
        ),
        "rejected_1": execute(code.replace("rejected = torch.tensor([0.0])", "rejected = torch.tensor([1.0])")),
        "rejected_1_shift_10": execute(
            code.replace("chosen = torch.tensor([preferred_score])", "chosen = torch.tensor([preferred_score + 10.0])")
            .replace("rejected = torch.tensor([0.0])", "rejected = torch.tensor([11.0])")
        ),
    }
    assert examples["original"] == (
        "差距 0.0 勝出機率 0.5 代價 0.6931\n"
        "差距 2.0 勝出機率 0.8808 代價 0.1269\n"
    )
    assert examples["rejected_1"] == (
        "差距 -1.0 勝出機率 0.2689 代價 1.3133\n"
        "差距 1.0 勝出機率 0.7311 代價 0.3133\n"
    )
    assert examples["rejected_1"] == examples["rejected_1_shift_10"]
    assert examples["original"] == examples["original_shift_10"]
    calculations = []
    for gap in (-1.0, 0.0, 1.0, 2.0):
        chosen = torch.tensor([gap], dtype=torch.float64, requires_grad=True)
        rejected = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        probability = 1 / (1 + math.exp(-gap))
        expected_loss = math.log1p(math.exp(-gap))
        loss = preference_loss(chosen, rejected)
        loss.backward()
        assert abs(loss.item() - expected_loss) < 1e-15
        assert abs(chosen.grad.item() - (probability - 1)) < 1e-15
        assert abs(rejected.grad.item() - (1 - probability)) < 1e-15
        assert torch.equal(preference_loss(chosen + 10, rejected + 10), loss)
        calculations.append({
            "gap": gap, "probability": probability, "loss": expected_loss,
            "executed_loss": loss.item(), "chosen_gradient": chosen.grad.item(),
            "rejected_gradient": rejected.grad.item(), "shift_10_exactly_equal": True,
        })
    gaps = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    batch_loss = preference_loss(gaps, torch.zeros_like(gaps))
    expected_mean = sum(row["loss"] for row in calculations) / 4
    assert abs(batch_loss.item() - expected_mean) < 1e-15
    extreme = preference_loss(torch.tensor([-1000.0, 1000.0]), torch.zeros(2))
    assert torch.isfinite(extreme) and extreme.item() == 500.0
    invalid = []
    for chosen, rejected in [(torch.zeros(1), torch.zeros(2)), (torch.empty(0), torch.empty(0))]:
        try:
            preference_loss(chosen, rejected)
        except ValueError as error:
            invalid.append(str(error))
        else:
            raise AssertionError("invalid inputs should be rejected")
    generated = split_records(build_records(), 42)
    families = [{row["family"] for row in rows} for rows in generated.values()]
    assert all(not a.intersection(b) for i, a in enumerate(families) for b in families[i + 1 :])
    assert sum(map(len, families)) == 55
    public_path = ROOT / "docs/course-experiments/results/posttraining.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    fresh = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
    source_hashes = {name: sha(ROOT / name) for name in fresh["code_sha256"]}
    assert source_hashes == fresh["code_sha256"] == public["code_sha256"]
    assert public["results"]["config"] == fresh["results"]["config"] == CONFIG
    all_rows = {"public": check_rows(public, generated), "fresh": check_rows(fresh, generated)}
    original_records = json.loads((ROOT / "outputs/posttraining-development/fixed/records.json").read_text())
    fresh_records = json.loads((RUN / "records.json").read_text())
    assert original_records == fresh_records == generated
    checkpoint_audit = []
    for checkpoint in fresh["results"]["checkpoints"]:
        path = RUN / checkpoint["file"]
        assert sha(path) == checkpoint["sha256"]
        original = ROOT / "outputs/posttraining-development/fixed" / checkpoint["file"]
        record = {"file": checkpoint["file"], "fresh_file_sha256": sha(path), "old_file_sha256": sha(original)}
        for old_checkpoint in public["results"]["checkpoints"]:
            if old_checkpoint["file"] == checkpoint["file"]:
                assert sha(original) == old_checkpoint["sha256"]
                assert checkpoint["state_sha256"] == old_checkpoint["state_sha256"]
                record["state_sha256"] = checkpoint["state_sha256"]
        checkpoint_audit.append(record)
    for name in generated:
        assert fresh["results"]["evaluations"][name]["rows"] == public["results"]["evaluations"][name]["rows"]
    model = FiniteRewardModel()
    checkpoint = torch.load(RUN / "reward.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    features = torch.tensor([row["features"] for row in generated["test"]], dtype=torch.float32)
    with torch.no_grad():
        checkpoint_scores = model(features)
    stored_scores = torch.tensor([row["reward_model_raw_scores"] for row in fresh["results"]["evaluations"]["test"]["rows"]])
    assert torch.equal(checkpoint_scores, stored_scores)
    model.requires_grad_(False)
    before = [p.clone() for p in model.parameters()]
    # Exercise an actual downstream differentiable choice while RM stays frozen.
    policy_logits = torch.zeros((18, 4), requires_grad=True)
    optimizer = torch.optim.SGD([policy_logits], lr=0.1)
    scores = model(features)
    downstream_loss = -(policy_logits.softmax(-1) * scores).sum(-1).mean()
    optimizer.zero_grad()
    downstream_loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
    assert not torch.equal(policy_logits, torch.zeros_like(policy_logits))
    figure = ROOT / "course/figures/multimodal_preference_pair.svg"
    xml = ET.parse(figure).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    result = {
        "reviewer_task": "/root/integration_technical_coordinator/fact_v2_13_10",
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_10_audit.py",
        "result": "All assertions passed; original lesson and exercise executed, all 165 rows/825 pairs audited twice.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "device": "cpu", "dtype": "float32 examples; float64 derivation", "threads": "2"},
        "lesson_sha256": hashlib.sha256(body.encode()).hexdigest(), "repository_source_sha256": source_hashes,
        "examples": examples, "calculations": calculations, "batch_mean_loss": batch_loss.item(),
        "invalid_input_checks": invalid, "extreme_stable_loss": extreme.item(),
        "formal_report_sha256": sha(public_path), "all_rows_and_pairs": all_rows,
        "checkpoints": checkpoint_audit, "frozen_rm": {"parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(features.shape), "output_shape": list(scores.shape), "dtype": str(features.dtype),
        "requires_grad_false": all(not p.requires_grad for p in model.parameters()), "gradients_none": True,
        "state_unchanged": True, "checkpoint_scores_match_all_test_rows_exactly": True},
        "training_objectives": {"reward": "mean -logsigmoid(chosen_raw_score - rejected_raw_score) over 64 pair draws per step",
        "tokens": 0, "unique_train_pairs": 660, "processed_train_pairs": 19200, "updates": 300,
        "evaluated_pairs": {"train": 660, "validation": 75, "test": 90}},
        "public_timing_seconds": {"elapsed": public["elapsed_seconds"], "experiment": public["results"]["experiment_seconds"],
        "reward": public["results"]["reward"]["seconds"], "ppo": public["results"]["ppo"]["seconds"]},
        "fresh_timing_seconds": {"elapsed": fresh["elapsed_seconds"], "experiment": fresh["results"]["experiment_seconds"],
        "reward": fresh["results"]["reward"]["seconds"], "ppo": fresh["results"]["ppo"]["seconds"]},
        "timing_scope": "One same-seed CPU run. elapsed spans run_posttraining: training/evaluation/checkpoint saves; excludes environment setup and CLI initialization.",
        "prerequisite_figure": {"path": str(figure.relative_to(ROOT)), "sha256": sha(figure),
        "viewBox": xml.attrib["viewBox"], "texts": [node.text for node in xml.findall("svg:text", ns)],
        "arrows": [node.attrib["d"] for node in xml.findall("svg:path", ns)]},
        "python_executable": sys.executable,
    }
    target = ARTIFACTS / f"{PREFIX}_audit_output.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["result"])
    print(json.dumps({"examples": examples, "rm_pairwise": {name: {key: value for key, value in row.items() if key in ("contexts", "pair_count", "rm_wins")} for name, row in all_rows["fresh"].items()}, "timing": result["fresh_timing_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
