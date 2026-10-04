"""Replay and audit the brief CPU finite-card experiment without changing its recipe."""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PREFIX = "fact_finish_g_4"


def save_json(suffix, value):
    target = BASE / f"{PREFIX}_{suffix}.json"
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    source_files = ["tiny_perceptron/posttraining.py", "scripts/course_experiments/posttraining.py"]
    source_versions = {}
    for relative in source_files:
        data = (ROOT / relative).read_bytes()
        snapshot = BASE / f"{PREFIX}_code_{relative.replace('/', '_')}.txt"
        snapshot.write_bytes(data)
        source_versions[relative] = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix=f"{PREFIX}_") as temporary:
        command = [sys.executable, "-m", "scripts.course_experiments.posttraining", "--output", temporary]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        (BASE / f"{PREFIX}_cpu_stdout.txt").write_text(completed.stdout)
        (BASE / f"{PREFIX}_cpu_stderr.txt").write_text(completed.stderr)
        for name in ["result", "records"]:
            data = (Path(temporary) / f"{name}.json").read_bytes()
            (BASE / f"{PREFIX}_cpu_{name}.json").write_bytes(data)
        result = json.loads((Path(temporary) / "result.json").read_text())
        records = json.loads((Path(temporary) / "records.json").read_text())

    results = result["results"]
    rules = {"number": 0, "explain": 1, "missing": 3}
    audited_rows = []
    for split, rows in records.items():
        for row in rows:
            a, b = row["operands"]
            expected = rules[row["mode"]]
            assert row["expected_action"] == expected
            assert len(row["candidates"]) == 4
            assert row["candidates"][0] == str(a + b)
            assert row["candidates"][2] == str(a + b + 1)
            assert row["candidate_meets_full_request"] == [i == expected for i in range(4)]
            for winner, loser in row["preference_pairs"]:
                assert 0 <= winner < 4 and 0 <= loser < 4 and winner != loser
            if row["mode"] == "missing":
                assert row["preference_pairs"] == [[3, i] for i in range(3)]
            audited_rows.append({"split": split, "family": row["family"], "mode": row["mode"]})
        for row in results["evaluations"][split]["rows"]:
            for policy in row["policies"].values():
                action = policy["chosen_action"]
                assert 0 <= action < 4
                assert policy["chosen_response"] == row["candidates"][action]
    trace = results["ppo"]["first_rollout_trace"]
    assert len(trace) == results["config"]["ppo_epochs_per_rollout"]
    assert len({row["old_log_probability_sha256"] for row in trace}) == 1
    assert len({row["reference_state_sha256"] for row in trace}) == 1
    assert results["reference_state_sha256_before"] == results["reference_state_sha256_after"]
    assert all(0 <= action < 4 for row in trace for action in row["sampled_actions"])
    assert results["effective_tokens"] == 0
    assert results["schedule_completed"] is True
    for relative, digest in source_versions.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    audit = {
        "command": command,
        "cwd": str(ROOT),
        "exit_code": completed.returncode,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": result["device"],
            "seed": str(result["seed"]),
            "cpu_threads": str(results["config"]["cpu_threads"]),
        },
        "source_sha256_before_and_after": source_versions,
        "records_audited": len(audited_rows),
        "all_record_keys_checked": audited_rows,
        "split_context_counts": {key: len(value) for key, value in records.items()},
        "verified": {
            "all_numeric_candidates_precomputed_by_python": True,
            "labels_reconstructed_from_fixed_rules": True,
            "all_reported_responses_are_lookup_of_candidate_index": True,
            "rollout_actions_are_single_integer_indices": True,
            "old_log_probabilities_fixed_across_reuse": True,
            "fixed_reference_unchanged": True,
            "autoregressive_token_generation_in_experiment": False,
        },
        "scope": "Only verifies the course CPU finite-candidate workflow; no human-feedback, open-ended language, quality superiority, or GPU speed claim.",
    }
    save_json("cpu_audit", audit)
    print(
        json.dumps(
            {"exit_code": completed.returncode, "records_audited": len(audited_rows), "verified": audit["verified"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
