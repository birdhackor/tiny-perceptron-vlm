"""Independently recompute lesson 13.16 examples and every recorded evaluation."""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.posttraining import build_records, split_records
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"
PUBLIC = ROOT / "docs/course-experiments/results/posttraining.json"
RAW = ROOT / "outputs/posttraining-development/fixed"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(state):
    sha = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        sha.update(name.encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def audit_evaluations(result):
    rows_out = {}
    summaries = {}
    for split, evaluation in result["evaluations"].items():
        counts = {name: Counter() for name in ["sft", "ppo", "dpo"]}
        denominators = Counter()
        checked = []
        wins, pair_count = 0, 0
        for row in evaluation["rows"]:
            a, b = row["operands"]
            mode = row["mode"]
            expected = {"number": 0, "explain": 1, "missing": 3}[mode]
            assert row["expected_action"] == expected
            if mode != "missing":
                assert row["candidates"][:3] == [str(a + b), f"{a} 加 {b} 是 {a + b}。", str(a + b + 1)]
                rank = [0, 1, 3, 2] if mode == "number" else [1, 0, 3, 2]
                pairs = [[w, loser] for i, w in enumerate(rank) for loser in rank[i + 1 :]]
            else:
                assert a > 0 and "購買數量" in row["candidates"][3]
                pairs = [[3, other] for other in range(3)]
            assert pairs == row["preference_pairs"]
            content_flags = [False, False, False, True] if mode == "missing" else [True, True, False, False]
            assert row["candidate_content_or_clarification_correct"] == content_flags
            assert row["candidate_meets_full_request"] == [action == expected for action in range(4)]
            assert row["features"] == [a / 10, b / 10, float(mode == "explain"), float(mode == "missing")]
            pair_checks = []
            for winner, loser in pairs:
                passed = row["reward_model_raw_scores"][winner] > row["reward_model_raw_scores"][loser]
                wins += passed
                pair_count += 1
                pair_checks.append({"winner": winner, "loser": loser, "ordered": passed})
            denominators[mode] += 1
            policies = {}
            for name, prediction in row["policies"].items():
                probabilities = prediction["probabilities"]
                assert abs(sum(probabilities) - 1) < 2e-6
                selected = max(range(4), key=lambda action: probabilities[action])
                assert selected == prediction["chosen_action"]
                assert prediction["chosen_response"] == row["candidates"][selected]
                success = selected == expected
                assert success == prediction["full_request_success"]
                assert prediction["content_or_clarification_correct"] == content_flags[selected]
                counts[name][mode] += success
                policies[name] = {"action": selected, "response": row["candidates"][selected],
                                  "probabilities": probabilities, "full_request_success": success,
                                  "content_or_clarification_correct": content_flags[selected]}
            checked.append({"family": row["family"], "mode": mode, "prompt": row["prompt"],
                            "candidates": row["candidates"], "expected": expected,
                            "raw_rm_scores": row["reward_model_raw_scores"], "pair_checks": pair_checks,
                            "policies": policies})
        recorded = evaluation["reward_model_pairwise_accuracy"]
        assert [wins, pair_count] == [recorded["numerator"], recorded["denominator"]]
        policy_summary = {}
        for name, mode_counts in counts.items():
            policy_summary[name] = {mode: [mode_counts[mode], denominators[mode]] for mode in denominators}
            policy_summary[name]["total"] = [sum(mode_counts.values()), sum(denominators.values())]
            for mode in denominators:
                reported = evaluation["policies"][name]["by_mode"][mode]
                assert policy_summary[name][mode] == [reported["numerator"], reported["denominator"]]
        summaries[split] = {"rm_pairs": [wins, pair_count], "policies": policy_summary}
        rows_out[split] = checked
    return summaries, rows_out


def main():
    torch.set_num_threads(2)
    public = json.loads(PUBLIC.read_text())
    result = public["results"]
    config = result["config"]
    assert result["sft"]["demonstrations"] == 44
    assert result["sft"]["processed_demonstration_draws"] == config["sft_steps"] * config["sft_batch_size"] == 1920
    assert result["reward"]["processed_pair_draws"] == config["reward_steps"] * config["reward_batch_size"] == 19200
    assert result["ppo"]["sampled_actions"] == config["ppo_rollout_batches"] * config["ppo_rollout_size"] == 7680
    assert result["ppo"]["reused_action_draws"] == result["ppo"]["sampled_actions"] * config["ppo_epochs_per_rollout"] == 23040
    assert result["ppo"]["policy_updates"] == result["ppo"]["value_updates"] == config["ppo_rollout_batches"] * config["ppo_epochs_per_rollout"] == 360
    assert result["dpo"]["policy_updates"] == config["dpo_steps"] == 360
    assert result["dpo"]["processed_pair_draws"] == config["dpo_steps"] * config["dpo_batch_size"] == 23040
    assert result["effective_tokens"] == 0 and result["schedule_completed"]
    assert result == json.loads((RAW / "experiment.json").read_text())
    assert result == json.loads((RAW / "result.json").read_text())["results"]
    code_hashes = {path: digest(ROOT / path) for path in public["code_sha256"]}
    assert code_hashes == public["code_sha256"]
    raw_records = json.loads((RAW / "records.json").read_text())
    rebuilt = split_records(build_records(), 42)
    assert raw_records == rebuilt
    family_sets = [{row["family"] for row in rows} for rows in rebuilt.values()]
    assert all(not a & b for i, a in enumerate(family_sets) for b in family_sets[i + 1 :])
    raw_hashes = {str(path.relative_to(ROOT)): digest(path) for path in RAW.iterdir() if path.is_file()}
    models = {"sft": FiniteResponsePolicy(), "ppo": FiniteResponsePolicy(), "dpo": FiniteResponsePolicy(),
              "reward": FiniteRewardModel(), "value": FiniteValueModel()}
    state_hashes = {}
    metadata = {}
    for checkpoint in result["checkpoints"]:
        name = Path(checkpoint["file"]).stem
        path = RAW / checkpoint["file"]
        assert digest(path) == checkpoint["sha256"]
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert state_digest(loaded["model"]) == checkpoint["state_sha256"]
        models[name].load_state_dict(loaded["model"])
        models[name].eval().requires_grad_(False)
        state_hashes[name] = state_digest(loaded["model"])
        metadata[name] = loaded["metadata"]
    assert metadata["ppo"]["initial_state_sha256"] == metadata["dpo"]["initial_state_sha256"] == state_hashes["sft"]
    errors = []
    with torch.no_grad():
        for split, evaluation in result["evaluations"].items():
            x = torch.tensor([row["features"] for row in evaluation["rows"]], dtype=torch.float32)
            scores = models["reward"](x)
            expected_scores = torch.tensor([row["reward_model_raw_scores"] for row in evaluation["rows"]])
            errors.append(float((scores - expected_scores).abs().max()))
            for name in ["sft", "ppo", "dpo"]:
                p = models[name](x).softmax(-1)
                expected_p = torch.tensor([row["policies"][name]["probabilities"] for row in evaluation["rows"]])
                errors.append(float((p - expected_p).abs().max()))
    assert max(errors) <= 2e-6
    summaries, rows = audit_evaluations(result)
    body = next(text for lesson, text in sections(ROOT / "course/chapters/13.md") if lesson == "13.16")
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace = {}
        exec(compile(code, "13.16 literal code", "exec"), namespace)
    assert namespace["scores"] == [1, 14] and namespace["chosen"] == 1
    exercise_scores = [int(answer == str(1 + 2)) for answer in namespace["answers"]]
    exercise_chosen = max(range(2), key=lambda index: exercise_scores[index])
    assert exercise_scores == [1, 0] and namespace["answers"][exercise_chosen] == "3"
    with tempfile.TemporaryDirectory(prefix="fact_v2_13_16_") as temporary:
        command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run",
                   "--experiment", "posttraining", "--device", "cpu", "--output", temporary]
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        replay = json.loads((Path(temporary) / "posttraining/result.json").read_text())
        rerun = replay["results"]
        replay_summaries, replay_rows = audit_evaluations(rerun)
        assert replay_summaries == summaries
        replay_states = {Path(c["file"]).stem: c["state_sha256"] for c in rerun["checkpoints"]}
        assert replay_states == state_hashes
        assert replay_rows == rows
        replay_evidence = {"command_argv": command, "exit_code": completed.returncode,
                           "stdout": completed.stdout, "stderr": completed.stderr,
                           "python": replay["python_version"], "torch": replay["torch_version"],
                           "elapsed_seconds": replay["elapsed_seconds"], "timing_scope": replay["timing_scope"],
                           "stage_seconds": {k: rerun[k]["seconds"] for k in ["sft", "reward", "ppo", "dpo"]},
                           "states": replay_states, "all_evaluation_rows_exactly_equal": True}
    audit = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_16_audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "dtype": "torch.float32", "threads": str(torch.get_num_threads()),
                        "platform": platform.platform()},
        "result": "All assertions passed; literal example, exercise, source hashes, raw checkpoint inference, all 165 rows, all 825 preference pairs, and fixed CPU replay verified",
        "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "public_report_sha256": digest(PUBLIC), "code_sha256": code_hashes,
        "raw_file_sha256": raw_hashes, "state_sha256": state_hashes,
        "literal_example": {"scores": namespace["scores"], "stdout": output.getvalue(),
                            "exercise_scores": exercise_scores, "exercise_selected": namespace["answers"][exercise_chosen]},
        "original_config": result["config"], "splits": result["splits"],
        "original_timing": {"elapsed_seconds": public["elapsed_seconds"],
                            "experiment_seconds": result["experiment_seconds"],
                            "stage_seconds": {k: result[k]["seconds"] for k in ["sft", "reward", "ppo", "dpo"]},
                            "scope": "main() timer begins before run_posttraining and ends after code hashing; run_posttraining timer excludes setup/record writing and includes evaluation/checkpoint saves; PPO timer surrounds only rollout/update loop"},
        "denominators": {"families": [44, 5, 6], "contexts": [132, 15, 18], "train_preference_pairs": 660,
                         "test_preference_pairs": 90, "seed": 42, "sft_demonstrations": 44,
                         "sft_steps": 60, "sft_draws": 1920, "reward_updates": 300,
                         "reward_draws": 19200, "ppo_rollouts": 120, "sampled_actions": 7680,
                         "ppo_policy_updates": 360, "value_updates": 360, "reused_actions": 23040,
                         "dpo_updates": 360, "dpo_draws": 23040, "effective_tokens": 0,
                         "timing_runs": 1, "warmup_runs": 0},
        "maximum_checkpoint_inference_absolute_error": max(errors),
        "tensor_shapes": {split: {"input_features": [len(rows), 4], "policy_logits": [len(rows), 4],
                                  "reward_scores": [len(rows), 4], "value": [len(rows)],
                                  "mask": "No token padding or token mask: one action per structured context"}
                          for split, rows in rebuilt.items()},
        "summary": summaries, "all_rows": rows, "fresh_cpu_replay": replay_evidence,
    }
    (ARTIFACTS / "fact_v2_13_16_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"result": audit["result"], "summary": summaries,
                      "max_inference_error": max(errors), "replay": replay_evidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
