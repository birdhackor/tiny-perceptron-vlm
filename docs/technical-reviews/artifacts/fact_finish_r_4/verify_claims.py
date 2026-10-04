"""R.4 raw-record recount, pinned publication metadata and short CPU checks."""

import hashlib
import json
import platform
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.capstone_release import validate_manifest  # noqa: E402
from scripts.course_experiments.posttraining import build_records  # noqa: E402
from tiny_perceptron.capstone import (  # noqa: E402
    TOK,
    CapstoneModel,
    build_dataset,
    default_config,
    expected_final,
    load_capstone,
    parse_action,
    prompt_ids,
    save_capstone,
)
from tiny_perceptron.capstone_quantization import load_quantized_capstone, quantize_capstone  # noqa: E402
from tiny_perceptron.posttraining import FiniteResponsePolicy  # noqa: E402

OUT = Path(__file__).resolve().parent
READS = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    READS[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def check_records(path, expected_rows):
    data = read(path)
    rows = {row["id"]: row for row in expected_rows}
    records = data["records"]
    assert len(records) == len(rows) == data["count"]
    assert {record["id"] for record in records} == set(rows)
    counts = Counter()
    compact = []
    for record in records:
        row = rows[record["id"]]
        assert record["expected_action"] == row["answer"]
        assert record["expected_final"] == expected_final(row)
        assert record["family"] == row["family"] and record["task"] == row["task"]
        assert record["action_trace"]["prompt_ids"] == prompt_ids(row)
        for trace in (record["action_trace"], record["final_trace"]):
            if trace is not None:
                assert trace["raw"] == TOK.decode(trace["generated_ids"])
                assert trace["eos"] == (trace["generated_ids"][-1] == TOK.eos_id)
                assert trace["eos"] == (trace["stop_reason"] == "eos")
        parsed = parse_action(record["action_trace"])
        assert parsed == record["parsed_action"]
        action_ok = record["action_trace"]["eos"] and record["action_trace"]["raw"] == row["answer"]
        computed_answer = parsed.get("content") if parsed["status"] in ("direct", "ask") else None
        if parsed["status"] == "tool" and record["runtime"]["status"] == "ok":
            assert record["runtime"]["result"] == str(parsed["a"] + parsed["b"])
            followup = dict(row, image=None, audio=None)
            followup["user"] = f"原題：{parsed['a']}+{parsed['b']}。計算器回報：{record['runtime']['result']}。請回答。"
            assert record["final_trace"]["prompt_ids"] == prompt_ids(followup)
            final_action = parse_action(record["final_trace"])
            if final_action["status"] == "direct":
                computed_answer = final_action["content"]
        assert computed_answer == record["answer"]
        final_ok = action_ok and record["answer"] == expected_final(row)
        assert record["action_correct"] == action_ok and record["end_to_end_correct"] == final_ok
        counts["count"] += 1
        counts["action_correct"] += int(action_ok)
        counts["end_to_end_correct"] += int(final_ok)
        compact.append(
            {
                "id": record["id"],
                "task": record["task"],
                "expected": row["answer"],
                "action": record["action_trace"]["raw"],
                "eos": record["action_trace"]["eos"],
                "runtime": record["runtime"],
                "final": None if record["final_trace"] is None else record["final_trace"]["raw"],
                "answer": record["answer"],
                "correct": final_ok,
            }
        )
    assert all(data[key] == counts[key] for key in counts)
    by_task = {}
    for record in records:
        task = by_task.setdefault(record["task"], Counter())
        task["count"] += 1
        task["action_correct"] += int(record["action_correct"])
        task["end_to_end_correct"] += int(record["end_to_end_correct"])
    assert by_task == data["by_task"]
    return data, {"recomputed": dict(counts), "by_task": by_task, "all_records": compact}


def publications():
    old = read("docs/course-experiments/public-models.json")
    cap = validate_manifest(read("docs/course-experiments/capstone-public.json"))
    results = {
        "earlier_experiments": len(old["models"]),
        "capstone_versions": len(cap["models"]),
        "earlier": [],
        "capstone": [],
    }
    for model in old["models"]:
        remote = {item["path"]: item for item in read(OUT.relative_to(ROOT) / f"originals/hf-{model['id']}.json")}
        checkpoints = [
            item for item in model["files"] if Path(item["path"]).suffix in (".pt", ".pth", ".ckpt", ".safetensors")
        ]
        for item in checkpoints:
            actual = remote[item["path"]]
            assert actual["size"] == item["bytes"] and actual["lfs"]["oid"] == item["sha256"]
        assert model["repo"] == cap["repo"]
        results["earlier"].append(
            {
                "id": model["id"],
                "revision": model["revision"],
                "checkpoint_count": len(checkpoints),
                "paths_sha256_bytes": checkpoints,
            }
        )
    remotes = {}
    for path in sorted((OUT / "originals").glob("hf-capstone-*.json")):
        remotes.update({item["path"]: item for item in read(path.relative_to(ROOT))})
    for model in cap["models"]:
        checkpoint = next(item for item in model["files"] if item["output"] == model["checkpoint"])
        actual = remotes[checkpoint["path"]]
        assert actual["size"] == checkpoint["bytes"] and actual["lfs"]["oid"] == checkpoint["sha256"]
        results["capstone"].append({"id": model["id"], "checkpoint": checkpoint})
    results["earlier_checkpoints"] = sum(item["checkpoint_count"] for item in results["earlier"])
    results["repo"] = cap["repo"]
    results["capstone_revision"] = cap["revision"]
    return results


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    splits, manifest = build_dataset(42)
    families = {key: {row["family"] for row in rows} for key, rows in splits.items()}
    assert not families["train"] & families["validation"]
    assert not families["train"] & families["test"]
    assert not families["validation"] & families["test"]
    frozen = read("docs/course-experiments/capstone-evidence/deployment/data.json")
    assert frozen["manifest"] == manifest and frozen["splits"] == splits
    evaluations, raw = {}, {}
    for label, path, split in [
        ("ce", "student/test-ce.json", "test"),
        ("kd", "student/test-kd.json", "test"),
        ("kd_int4", "student/test-kd-ptq4.json", "test"),
        ("joint", "deployment/test-joint.json", "test"),
        ("validation_ce", "student/validation-ce.json", "validation"),
        ("validation_kd", "student/validation-kd.json", "validation"),
    ]:
        raw[label], evaluations[label] = check_records(
            f"docs/course-experiments/capstone-evidence/{path}", splits[split]
        )
    changes = []
    for before, after in zip(raw["kd"]["records"], raw["kd_int4"]["records"], strict=True):
        assert before["id"] == after["id"]
        if any(
            (before[key] or {}).get("generated_ids") != (after[key] or {}).get("generated_ids")
            for key in ("action_trace", "final_trace")
        ):
            changes.append(
                {
                    "id": before["id"],
                    "before_correct": before["end_to_end_correct"],
                    "after_correct": after["end_to_end_correct"],
                }
            )
    student = read("docs/course-experiments/results/capstone_student.json")
    r = student["results"]
    branches = r["branches"]
    assert branches["ce"]["effective_tokens"] == branches["kd"]["effective_tokens"] == 145163
    assert branches["ce"]["steps"] == branches["kd"]["steps"] == 350
    stages = {}
    parent_sha = None
    for stage, experiment in [
        ("pretrain", "capstone_pretrain"),
        ("sft", "capstone_sft"),
        ("joint", "capstone_joint"),
        ("dpo", "capstone_preference"),
    ]:
        report = read(f"docs/course-experiments/results/{experiment}.json")
        data = report["results"]
        assert data["parent_checkpoint_sha256"] == parent_sha
        assert data["schedule_completed"] and data["data_manifest"] == manifest
        parent_sha = data["inference_export"]["sha256"]
        stages[stage] = {
            key: data[key]
            for key in (
                "steps",
                "effective_tokens",
                "parent_checkpoint_sha256",
                "inference_export",
                "parameters",
                "history",
                "test_evaluated",
            )
        }
    assert r["teacher_checkpoint_sha256"] == parent_sha
    student_manifest = read("docs/course-experiments/capstone-evidence/student/review-manifest.json")
    for item in student_manifest["files"]:
        path = ROOT / "docs/course-experiments/capstone-evidence/student" / item["path"]
        if path.is_file():
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"] and path.stat().st_size == item["bytes"]
            )
    publications_result = publications()
    assets = read("assets/training/manifest.json")
    assert not any(
        Path(item["path"]).suffix in (".pt", ".pth", ".ckpt", ".safetensors")
        for asset in assets["assets"]
        for item in asset["files"]
    )
    licensing = [
        {"id": asset["id"], "license": asset["license"], "training_records": asset["training_records"]}
        for asset in assets["assets"]
    ]
    license_receipt = read(OUT.relative_to(ROOT) / "license-read-receipt.json")
    license_checks = []
    for entry in license_receipt:
        original = entry["original_path"]
        if not original.startswith("data/training/"):
            continue
        relative = original.removeprefix("data/training/")
        expected = next(item for asset in assets["assets"] for item in asset["files"] if item["path"] == relative)
        actual = (ROOT / entry["snapshot_path"]).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == entry["sha256"] == expected["sha256"]
        license_checks.append({"path": relative, "snapshot_sha256": entry["sha256"], "matches_asset_manifest": True})
    model = CapstoneModel(replace(default_config(dense=True), width=48))
    with tempfile.TemporaryDirectory(prefix="fact-r4-") as directory:
        fp = Path(directory) / "model.pt"
        ptq = Path(directory) / "int4.pt"
        save_capstone(fp, model, stage="joint", step=0, inference_only=True)
        restored, payload = load_capstone(fp)
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
        omitted = ["optimizer", "training_state", "torch_rng", "python_rng", "cuda_rng", "reference"]
        assert not set(omitted) & set(payload)
        compressed = quantize_capstone(fp, ptq, bits=4)
        dequantized, _ = load_quantized_capstone(ptq)
        assert all(value.dtype == torch.float32 for value in dequantized.state_dict().values())
    cpu = {
        "random_model_parameters": model.description(),
        "inference_roundtrip_all_tensors_equal": True,
        "inference_omitted": omitted,
        "compressed": compressed,
        "dequantized_all_float32": True,
        "scope": "Untrained Dense model; checks format/storage, not task competence or GPU performance",
    }
    cards = build_records()
    policy = FiniteResponsePolicy()
    features = torch.tensor([row["features"] for row in cards[:2]])
    scores = policy(features)
    assert scores.shape == (2, 4)
    assert all(len(row["candidates"]) == 4 for row in cards)
    cpu["finite_card_boundary"] = {
        "generated_template_records": len(cards),
        "policy_output_shape": list(scores.shape),
        "selected_precomputed_card": cards[0]["candidates"][int(scores[0].argmax())],
        "scope": "No PPO updates; verifies four existing answer strings are selected rather than token generation",
    }
    result = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_r_4/verify_claims.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "data_manifest": manifest,
        "evaluations": evaluations,
        "kd_int4_changed_records": changes,
        "student_recipe": {
            "initial_student_state_sha256": r["initial_student_state_sha256"],
            "teacher_checkpoint_sha256": r["teacher_checkpoint_sha256"],
            "branches": branches,
            "ptq": r["ptq"],
        },
        "training_chain": stages,
        "publications": publications_result,
        "licensing": licensing,
        "license_snapshot_checks": license_checks,
        "cpu": cpu,
        "raw_read_hashes": READS,
    }
    (OUT / "verification-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "evaluations": {key: value["recomputed"] for key, value in evaluations.items()},
                "kd_int4_changed_records": len(changes),
                "earlier_experiments": publications_result["earlier_experiments"],
                "earlier_checkpoints": publications_result["earlier_checkpoints"],
                "capstone_versions": publications_result["capstone_versions"],
                "cpu": cpu,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
