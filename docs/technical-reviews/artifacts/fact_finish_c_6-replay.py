"""Reproduce the C.6 parser, raw-record audit and public finite-policy CPU checks.

This does not train a model, replay CUDA RNG, or claim a CUDA timing result.
The public inference checkpoint is downloaded anonymously at an immutable revision
and hash-checked; no binary checkpoint is retained among review artifacts.
"""

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(text, truth):
    clean = text.strip()
    parsed = int(clean) if re.fullmatch(r"-?[0-9]+", clean) else None
    return parsed, float(parsed == truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    application = importlib.import_module("scripts.course_experiments.applications")
    common = importlib.import_module("scripts.course_experiments.common")
    review = importlib.import_module("scripts.check_technical_reviews")
    raw_path = ARTIFACTS / "fact_finish_c_6-raw-policy-snapshot.json"
    raw = json.loads(raw_path.read_text())
    read_path = ARTIFACTS / "fact_finish_c_6-C.6-read-snapshot.md"
    read_body = read_path.read_text()
    actual_body = dict(review.sections(ROOT / "course/chapters/0C.md"))["C.6"]
    assert actual_body.encode() == read_path.read_bytes(), "Source changed after personal reading"
    assert not re.findall(r"!\[[^\]]*\]\(([^)]+)\)", read_body)
    source_checks = {}
    for path, expected in raw["code_sha256"].items():
        observed = digest(ROOT / path)
        assert observed == expected
        source_checks[path] = {"expected": expected, "observed": observed}

    block = re.search(r"```python\n(.*?)```", read_body, re.S)[1]
    snippet_outputs = {}
    for truth, expected in (
        (4, [(3, 0.0), (4, 1.0), (5, 0.0), (None, 0.0)]),
        (5, [(3, 0.0), (4, 0.0), (5, 1.0), (None, 0.0)]),
    ):
        namespace = {}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(block.replace("truth = 4", f"truth = {truth}"), "C.6-snapshot", "exec"), namespace)
        observed = [score(text, truth) for text in namespace["answers"]]
        assert observed == expected
        snippet_outputs[str(truth)] = {"stdout": output.getvalue(), "results": observed}
    edge_cases = [
        (" 4 \n", 4, (4, 1.0)),
        ("-2", -2, (-2, 1.0)),
        ("004", 4, (4, 1.0)),
        ("+4", 4, (None, 0.0)),
        ("４", 4, (None, 0.0)),
        ("4 5", 4, (None, 0.0)),
        ("", 4, (None, 0.0)),
        ("2+2=5;5-1=4;answer=4", 4, (None, 0.0)),
    ]
    for text, truth, expected in edge_cases:
        assert score(text, truth) == expected
    premise = [(2, 2, 5), (5, -1, 4)]
    prerequisite = {
        "equations": [a + b == claimed for a, b, claimed in premise],
        "linked": premise[1][0] == premise[0][2],
        "final": premise[-1][2] == 2 + 2,
    }
    assert prerequisite == {"equations": [False, True], "linked": True, "final": True}
    task_only = {"a": 2, "b": 2, "c": 0, "truth": 4}
    narrow_verifier = application._verify_reasoning(
        {"generated": "4", "invalid_special_tokens": []}, task_only, "direct"
    )
    assert narrow_verifier["fully_verified"]

    expected_actions = [str(number) for number in range(16)] + [" ".join(map(str, range(16)))]
    assert application._POLICY_ACTIONS == expected_actions
    reward_table = []
    for truth in range(16):
        for action, text in enumerate(expected_actions):
            strict = float(score(text, truth)[1])
            proxy = float(str(truth) in text.split())
            assert application._policy_reward(action, truth) == strict
            assert application._policy_reward(action, truth, proxy=True) == proxy
            reward_table.append({"truth": truth, "action_id": action, "strict": strict, "proxy": proxy})
        assert application._policy_reward(16, truth) == 0.0
        assert application._policy_reward(16, truth, proxy=True) == 1.0

    splits = common.split_records(application._reasoning_records(), raw["seed"])
    split_audit = {}
    for name, rows in splits.items():
        observed = {
            "records": len(rows),
            "families": len({row["family"] for row in rows}),
            "sha256": common.records_sha256(rows),
        }
        assert observed == raw["results"]["split"][name]
        split_audit[name] = observed
    families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
    assert not families["train"] & families["test"]
    assert not families["validation"] & families["test"]
    assert all(0 <= row["truth"] <= 15 for rows in splits.values() for row in rows)
    weak = raw["results"]["reinforce"]["weak_proxy"]
    training = weak["training"]
    assert training["steps"] == training["planned_steps"] == 1200
    assert training["sampled_actions"] == 1200 * 64 == 76800
    assert training["step_scale"] == 1.0 and training["effective_tokens"] == 0
    traces = weak["after"]["samples"]
    assert len(traces) == len(splits["test"]) == 24
    audit_rows = []
    samples = []
    for trace, row in zip(traces, splits["test"], strict=True):
        assert all(trace[key] == value for key, value in row.items())
        assert trace["truth"] == trace["a"] + trace["b"] + trace["c"]
        probabilities = trace["probabilities"]
        assert len(probabilities) == 17 and all(math.isfinite(p) and p >= 0 for p in probabilities)
        assert abs(sum(probabilities) - 1.0) < 2e-7
        assert len(trace["samples"]) == 16
        for sample in trace["samples"]:
            action = sample["action_id"]
            assert sample["generated"] == expected_actions[action]
            assert sample["strict_reward"] == application._policy_reward(action, row["truth"])
            assert sample["proxy_reward"] == application._policy_reward(action, row["truth"], proxy=True)
            samples.append(sample)
        audit_rows.append(
            {
                "question": row["question"],
                "truth": row["truth"],
                "action_ids": [s["action_id"] for s in trace["samples"]],
            }
        )
    counts = {
        "sample_accuracy": sum(int(sample["strict_reward"]) for sample in samples),
        "sample_proxy_reward": sum(int(sample["proxy_reward"]) for sample in samples),
        "enumeration_action_rate": sum(sample["action_id"] == 16 for sample in samples),
    }
    assert counts == {"sample_accuracy": 0, "sample_proxy_reward": 384, "enumeration_action_rate": 384}
    for label, numerator in counts.items():
        assert weak["after"][label] == {"numerator": numerator, "denominator": 384, "rate": numerator / 384}

    public_item = json.loads((ARTIFACTS / "fact_finish_c_6-public-model-item.json").read_text())
    export = json.loads((ARTIFACTS / "fact_finish_c_6-export-manifest.json").read_text())
    entry = next(item for item in public_item["files"] if item["output"] == "policy-weak_proxy.pt")
    exporter = next(item for item in export["files"] if item["output"] == "policy-weak_proxy.pt")
    original = next(item for item in raw["artifacts"] if item["path"] == "policy-weak_proxy.pt")
    assert exporter["source_sha256"] == original["sha256"]
    assert exporter["sha256"] == entry["sha256"]
    checkpoint_path = Path(
        importlib.import_module("huggingface_hub").hf_hub_download(
            repo_id=public_item["repo"], filename=entry["path"], revision=public_item["revision"], token=False
        )
    )
    assert digest(checkpoint_path) == entry["sha256"] and checkpoint_path.stat().st_size == entry["bytes"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"model", "format_version", "architecture", "metadata"}
    assert checkpoint["format_version"] == "finite-policy-v1"
    assert checkpoint["architecture"]["actions"] == expected_actions
    assert checkpoint["metadata"]["revision"] == raw["revision"]
    policy = application._FinitePolicy().cpu().eval()
    policy.load_state_dict(checkpoint["model"], strict=True)
    tensors = []
    for name, tensor in checkpoint["model"].items():
        assert tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
                "all_finite": True,
                "min": float(tensor.min()),
                "max": float(tensor.max()),
            }
        )
    ctx = SimpleNamespace(device="cpu")
    features = application._policy_features(splits["test"], ctx)
    assert features.shape == (24, 18) and torch.all(features.sum(dim=1) == 3)
    with torch.no_grad():
        replayed_probabilities = policy(features).softmax(-1)
    recorded_probabilities = torch.tensor([trace["probabilities"] for trace in traces])
    max_difference = float((replayed_probabilities - recorded_probabilities).abs().max())
    assert max_difference <= 2e-7, max_difference
    torch.manual_seed(42)
    replayed = application._evaluate_policy(policy, splits["test"], ctx)
    assert replayed["enumeration_action_rate"]["numerator"] == 384
    assert replayed["sample_accuracy"]["numerator"] == 0
    assert replayed["sample_proxy_reward"]["numerator"] == 384
    result = {
        "reviewer_task": "/root/fact_finish_c_6",
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_c_6-replay.py --output docs/technical-reviews/artifacts/fact_finish_c_6-execution.json",
        "completed_at": datetime.now(UTC).isoformat(),
        "result": "All assertions passed; original parser and exercise, 272 reward pairs, family split, 384 raw actions, 4 public tensors, CPU probabilities and fresh sampling verified.",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "sampling_seed": "42",
        },
        "read_snapshot_sha256": digest(read_path),
        "raw_policy_snapshot_sha256": digest(raw_path),
        "source_checks": source_checks,
        "snippets": snippet_outputs,
        "parser_edge_cases": [
            {"text": text, "truth": truth, "observed": score(text, truth)} for text, truth, _ in edge_cases
        ],
        "C.2_counterexample": prerequisite,
        "reward_table": reward_table,
        "split_audit": split_audit,
        "training_audit": {
            "steps": 1200,
            "batch_size": 64,
            "sampled_actions": 76800,
            "effective_tokens": 0,
            "seed": raw["seed"],
        },
        "raw_candidate_audit": {
            "questions": 24,
            "samples_per_question": 16,
            "total": len(samples),
            "counts": counts,
            "rows": audit_rows,
        },
        "checkpoint_audit": {
            "public_repo": public_item["repo"],
            "public_revision": public_item["revision"],
            "filename": entry["path"],
            "public_sha256": entry["sha256"],
            "public_bytes": entry["bytes"],
            "original_private_sha256_from_export_manifest": exporter["source_sha256"],
            "format_version": checkpoint["format_version"],
            "architecture": checkpoint["architecture"],
            "metadata": checkpoint["metadata"],
            "tensors": tensors,
            "parameters": sum(item["numel"] for item in tensors),
        },
        "cpu_replay": {
            "probability_tolerance": 2e-7,
            "max_absolute_difference": max_difference,
            "minimum_enumeration_probability": float(replayed_probabilities[:, 16].min()),
            "maximum_enumeration_probability": float(replayed_probabilities[:, 16].max()),
            "sample_counts": {label: replayed[label] for label in counts},
            "probabilities": replayed_probabilities.tolist(),
        },
        "limitations": [
            "No training was rerun.",
            "CPU sampling uses a fresh CPU RNG state; it is not an exact replay of the original CUDA RNG.",
            "No CUDA speed or broad reasoning-quality conclusion follows.",
            "Original private checkpoint download returned HTTP 401; only the hash-verified public inference export was obtained.",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(result["result"])
    print(json.dumps(result["cpu_replay"], indent=2))


if __name__ == "__main__":
    main()
