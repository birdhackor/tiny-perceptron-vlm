"""Independent T.7 CPU checks; no GPU timing or long training claims."""

import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch

from scripts.course_experiments.behavior import (
    _pair_examples,
    _preference_evaluate,
    _preference_parts,
    _state_digest,
)
from scripts.course_experiments.capstone import _balanced_sample
from scripts.course_experiments.common import evaluate_lm, split_records
from scripts.course_experiments.posttraining import build_records
from scripts.course_experiments.posttraining import split_records as split_cards
from scripts.course_experiments.text import _digest, _utf8_prefix, arithmetic_records
from tiny_perceptron.alignment import distillation_loss, dpo_loss
from tiny_perceptron.capstone import (
    TOK,
    CapstoneModel,
    build_dataset,
    evaluate_rows,
    frozen_reference,
    load_capstone,
    preference_loss,
    preference_pairs,
    prepare_batch,
)
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
WORK = Path(tempfile.mkdtemp(prefix="fact_finish_t_7-"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(model):
    return {
        name: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "bytes": t.numel() * t.element_size(),
            "sha256": hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        }
        for name, t in model.state_dict().items()
    }


def snapshot_inputs():
    paths = [
        "docs/course-experiments/results/dpo.json",
        "docs/course-experiments/results/posttraining.json",
        "docs/course-experiments/results/style.json",
        "docs/course-experiments/results/capstone_joint.json",
        "docs/course-experiments/results/capstone_preference.json",
        "docs/course-experiments/results/capstone_student.json",
        "docs/course-experiments/public-models.json",
        "docs/course-experiments/capstone-public.json",
        "docs/course-experiments/capstone-selection.json",
        "docs/course-experiments/capstone-evidence/deployment/data.json",
        "docs/course-experiments/capstone-evidence/joint/validation.json",
        "docs/course-experiments/capstone-evidence/dpo/validation.json",
        "docs/course-experiments/capstone-evidence/dpo/train-report.json",
        "docs/course-experiments/capstone-evidence/student/student-report.json",
        "docs/course-experiments/capstone-evidence/student/ce/train-report.json",
        "docs/course-experiments/capstone-evidence/student/kd/train-report.json",
        "docs/course-experiments/capstone-evidence/student/validation-ce.json",
        "docs/course-experiments/capstone-evidence/student/validation-kd.json",
        "docs/course-experiments/capstone-evidence/student/test-ce.json",
        "docs/course-experiments/capstone-evidence/student/test-kd.json",
    ]
    receipt = []
    for path in paths:
        source = ROOT / path
        data = json.loads(source.read_text())  # Parse complete original bytes, not aggregates only.
        destination = OUT / "raw" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        receipt.append({"source": path, "snapshot": str(destination.relative_to(ROOT)), "sha256": sha(source)})
        if "records" in data and isinstance(data["records"], list):
            write("inspection_" + source.parent.name + "_" + source.name, data["records"])
    write("input_receipt.json", receipt)


def fetch_capstone():
    manifest = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    result = {}
    receipt = []
    for entry in manifest["models"]:
        if entry["id"] not in {"joint", "dpo", "student-ce", "student-kd"}:
            continue
        file = next(f for f in entry["files"] if f["output"] == "model.pt")
        url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{file['path']}"
        path = WORK / (entry["id"] + ".pt")
        cached = next(
            (
                p
                for p in Path(tempfile.gettempdir()).glob("fact_finish_t_7-*/*.pt")
                if p.name == path.name and sha(p) == file["sha256"]
            ),
            None,
        )
        if cached:
            shutil.copyfile(cached, path)
        else:
            with urllib.request.urlopen(url, timeout=45) as response:
                path.write_bytes(response.read())
        assert sha(path) == file["sha256"] and path.stat().st_size == file["bytes"]
        model, payload = load_capstone(path, "cpu")
        receipt.append(
            {
                "id": entry["id"],
                "url": url,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "all_tensors_read": tensors(model),
                "stage": payload["stage"],
                "config": payload["config"],
                "metadata": payload["metadata"],
                "description": model.description(),
            }
        )
        result[entry["id"]] = (model, payload)
    write("capstone_checkpoint_receipt.json", receipt)
    return result


def check_dpo():
    report = json.loads((ROOT / "docs/course-experiments/results/dpo.json").read_text())["results"]
    arithmetic = split_records(arithmetic_records(), seed=42)
    pairs = _preference_parts(arithmetic)
    write("arithmetic_pairs.json", pairs)
    base, _ = load_checkpoint(ROOT / "checkpoints/course/style/content.pt", "cpu")
    assert _state_digest(base.state_dict()) == report["reference_sha256"]
    states = [
        {
            "path": "checkpoints/course/style/content.pt",
            "sha256": sha(ROOT / "checkpoints/course/style/content.pt"),
            "tensors": tensors(base),
        }
    ]
    results = {}
    for name, filename in [
        ("before", None),
        ("model", "model.pt"),
        ("beta1", "beta1.pt"),
        ("format", "format-model.pt"),
    ]:
        if filename:
            path = ROOT / "checkpoints/course/dpo" / filename
            model, payload = load_checkpoint(path, "cpu")
            states.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha(path),
                    "tensors": tensors(model),
                    "config": payload["config"],
                    "metadata": payload["metadata"],
                }
            )
        else:
            model = base
        selected = (
            pairs
            if name != "format"
            else {s: [{**r, "rejected": r["chosen"] + "; answer complete"} for r in rows] for s, rows in pairs.items()}
        )
        results[name] = {}
        for split in ["validation", "test"]:
            pref = _preference_evaluate(model, base, selected[split])
            generated = evaluate_lm(model, arithmetic[split], mode="sft", tokens=32)
            results[name][split] = {"preference": pref, "generation": generated}
            if name != "before":
                original = report["format_only"] if name == "format" else report["runs"][name]
                old = original["preference"][split]
                assert pref["records"] == old["records"]
                assert pref["chosen_higher_absolute_probability"] == old["chosen_higher_absolute_probability"]
                assert pref["relative_preference_improved"] == old["relative_preference_improved"]
                for actual, expected in zip(pref["samples"], old["samples"], strict=True):
                    for field in [
                        "policy_chosen_logp",
                        "policy_rejected_logp",
                        "policy_margin",
                        "reference_margin",
                        "relative_margin",
                    ]:
                        assert abs(actual[field] - expected[field]) < 1e-4, (name, split, field)
                assert [r["generated_ids"] for r in generated["samples"]] == [
                    r["generated_ids"] for r in original["arithmetic"][split]["samples"]
                ]
    counts = {}
    for name, count in [("model", 250), ("beta1", 250), ("format", 200)]:
        rows = pairs["train"]
        if name == "format":
            rows = [{**r, "rejected": r["chosen"] + "; answer complete"} for r in rows]
        examples = _pair_examples(rows, 128)
        sampler = random.Random(42)
        total = sum(
            sum(int((y != -100).sum()) for x, y in pair)
            for _ in range(count)
            for pair in sampler.choices(examples, k=8)
        )
        expected = report["format_only"]["training"] if name == "format" else report["runs"][name]["training"]
        assert total == expected["effective_answer_tokens_both_sides"]
        counts[name] = {
            "steps": count,
            "batch_pairs": 8,
            "pair_draws": count * 8,
            "effective_answer_targets_both_sides_with_eos": total,
        }
    assert [len(arithmetic[s]) for s in arithmetic] == [49, 8, 7]
    write("dpo_cpu_replay.json", results)
    write("dpo_checkpoint_receipt.json", states)
    write("dpo_denominators.json", counts)
    return {
        name: {
            split: {
                "pairs": r["preference"]["records"],
                "chosen_higher": r["preference"]["chosen_higher_absolute_probability"],
                "relative_improved": r["preference"]["relative_preference_improved"],
                "generation_matches": r["generation"]["matches"],
                "generation_records": r["generation"]["records"],
            }
            for split, r in splits.items()
        }
        for name, splits in results.items()
    }


def check_natural():
    report = json.loads((ROOT / "docs/course-experiments/results/dpo.json").read_text())["results"][
        "ultrafeedback_pilot"
    ]
    archive = ROOT / "assets/training/ultrafeedback-dpo-v1.tar.gz"
    with tarfile.open(archive) as packed:
        entry = next(m for m in packed.getmembers() if m.name.endswith("train-first-100.jsonl"))
        source_bytes = packed.extractfile(entry).read()
    raw_rows = [json.loads(line) for line in source_bytes.decode().splitlines()]
    assert len(raw_rows) == 100
    excerpts = []
    for index, row in enumerate(raw_rows):
        answers = []
        for side in ["chosen", "rejected"]:
            value = row[side]
            text = (
                next(message["content"] for message in reversed(value) if message["role"] == "assistant")
                if isinstance(value, list)
                else value
            )
            answers.append(_utf8_prefix(text, 120))
        excerpts.append(
            {
                "family": row.get("prompt_id", _digest(row["prompt"])),
                "prompt": _utf8_prefix(row["prompt"], 120),
                "chosen": answers[0],
                "rejected": answers[1],
                "source_row": index,
                "source_record_sha256": _digest(row),
            }
        )
    parts = split_records(excerpts, seed=42)
    write("natural_all_excerpts.json", parts)
    sft_path = ROOT / "checkpoints/course/dpo/ultrafeedback-sft.pt"
    policy_path = ROOT / "checkpoints/course/dpo/ultrafeedback-pilot.pt"
    reference, _ = load_checkpoint(sft_path, "cpu")
    policy, _ = load_checkpoint(policy_path, "cpu")
    write(
        "natural_checkpoint_receipt.json",
        [
            {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "all_tensors_read": tensors(m)}
            for p, m in [(sft_path, reference), (policy_path, policy)]
        ],
    )
    rows = []
    for split, metrics in report["evaluation"].items():
        actual = _preference_evaluate(policy, reference, parts[split])
        for new, old in zip(actual["samples"], metrics["samples"], strict=True):
            assert all(new[k] == old[k] for k in ["prompt", "chosen", "rejected", "source_row", "source_record_sha256"])
            for field in ["policy_chosen_logp", "policy_rejected_logp", "relative_margin"]:
                assert abs(new[field] - old[field]) < 5e-4
        for row in metrics["samples"]:
            assert all(len(row[k].encode()) <= 120 for k in ("prompt", "chosen", "rejected"))
            assert math.isclose(
                row["policy_margin"], row["policy_chosen_logp"] - row["policy_rejected_logp"], abs_tol=1e-9
            )
            assert math.isclose(row["relative_margin"], row["policy_margin"] - row["reference_margin"], abs_tol=1e-9)
            rows.append({"split": split, **row})
        assert metrics["records"] == len(metrics["samples"]) == 10
        assert sum(r["relative_margin"] > 0 for r in metrics["samples"]) == metrics["relative_preference_improved"]
    assert _utf8_prefix("甲" * 41, 120) == "甲" * 40
    write("natural_complete_records.json", rows)
    for split, split_rows in parts.items():
        serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows).encode()
        assert hashlib.sha256(serialized).hexdigest() == report["data"][split]["sha256"]
    write(
        "natural_provenance.json",
        {
            "archive": str(archive.relative_to(ROOT)),
            "archive_sha256": sha(archive),
            "member": entry.name,
            "source_bytes_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_rows_read": len(raw_rows),
            "excerpts_saved": 100,
            "cpu_replayed_preference_rows": len(rows),
            "maximum_utf8_bytes": 120,
        },
    )
    return {
        "source_records": report["source_records"],
        "split_counts": {s: report["data"][s]["records"] for s in ["train", "validation", "test"]},
        "sft_steps": report["sft_training"]["steps"],
        "dpo_steps": report["training"]["steps"],
        "free_generation_evaluation_present": False,
    }


def check_cards():
    target = WORK / "posttraining"
    command = [
        sys.executable,
        "-m",
        "scripts.course_experiments.run",
        "--experiment",
        "posttraining",
        "--device",
        "cpu",
        "--output",
        str(target),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    write(
        "posttraining_command.json",
        {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
    )
    fresh = json.loads((target / "posttraining/result.json").read_text())
    write("posttraining_cpu_result.json", fresh)
    saved = json.loads((ROOT / "docs/course-experiments/results/posttraining.json").read_text())["results"]
    parts = split_cards(build_records())
    for split, evaluation in saved["evaluations"].items():
        assert len(evaluation["rows"]) == len(parts[split])
        assert [r["features"] for r in evaluation["rows"]] == [r["features"] for r in parts[split]]
        for name, metrics in evaluation["policies"].items():
            success = 0
            for row in evaluation["rows"]:
                action = max(range(4), key=lambda a: row["policies"][name]["probabilities"][a])
                assert action == row["policies"][name]["chosen_action"]
                assert row["candidates"][action] == row["policies"][name]["chosen_response"]
                success += action == row["expected_action"]
            assert success == metrics["greedy_full_request_success"]["numerator"]
            assert (
                success
                == fresh["results"]["evaluations"][split]["policies"][name]["greedy_full_request_success"]["numerator"]
            )
    checkpoints = []
    for path in (target / "posttraining").glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints.append(
            {
                "file": path.name,
                "sha256": sha(path),
                "metadata": payload["metadata"],
                "tensors": {
                    name: {"shape": list(t.shape), "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}
                    for name, t in payload["model"].items()
                },
            }
        )
    write("posttraining_checkpoint_receipt.json", checkpoints)
    r = fresh["results"]
    assert (
        r["reference_state_sha256_before"]
        == r["reference_state_sha256_after"]
        == r["ppo"]["initial_state_sha256"]
        == r["dpo"]["initial_state_sha256"]
    )
    return {
        "test": {k: v["greedy_full_request_success"] for k, v in r["evaluations"]["test"]["policies"].items()},
        "parameters": r["parameters"],
        "config": r["config"],
        "effective_tokens": r["effective_tokens"],
        "sampled_actions": r["ppo"]["sampled_actions"],
        "reused_actions": r["ppo"]["reused_action_draws"],
        "reward_pair_draws": r["reward"]["processed_pair_draws"],
        "dpo_pair_draws": r["dpo"]["processed_pair_draws"],
    }


def check_capstone(models):
    splits, manifest = build_dataset(42)
    torch.manual_seed(42)
    from dataclasses import replace

    from tiny_perceptron.capstone import default_config

    initial = CapstoneModel(replace(default_config(dense=True), width=48))
    initial_sha = hashlib.sha256(b"".join(t.numpy().tobytes() for t in initial.state_dict().values())).hexdigest()
    student_report = json.loads((ROOT / "docs/course-experiments/results/capstone_student.json").read_text())["results"]
    assert initial_sha == student_report["initial_student_state_sha256"]
    write(
        "student_initial_state_reconstruction.json",
        {
            "state_sha256": initial_sha,
            "all_tensors_read": tensors(initial),
            "scope": "Independent CPU reconstruction of seed-42 random initialization; formal GPU batches not individually saved",
        },
    )
    data = json.loads((ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json").read_text())
    assert data["splits"] == splits and data["manifest"] == manifest
    results = {}
    for name, split, file in [
        ("joint", "validation", "joint/validation.json"),
        ("dpo", "validation", "dpo/validation.json"),
        ("student-ce", "validation", "student/validation-ce.json"),
        ("student-kd", "validation", "student/validation-kd.json"),
        ("student-ce", "test", "student/test-ce.json"),
        ("student-kd", "test", "student/test-kd.json"),
    ]:
        original = json.loads((ROOT / "docs/course-experiments/capstone-evidence" / file).read_text())
        # Every raw record: IDs, token decoding, EOS, action and final-answer rules.
        for row, raw in zip(splits[split], original["records"], strict=True):
            assert row["id"] == raw["id"] and row["answer"] == raw["expected_action"]
            for key in ["action_trace", "final_trace"]:
                trace = raw.get(key)
                if trace:
                    assert TOK.decode(trace["generated_ids"]) == trace["raw"]
                    assert trace["eos"] == (trace["generated_ids"][-1] == 2)
            action_correct = raw["action_trace"]["eos"] and raw["action_trace"]["raw"] == raw["expected_action"]
            assert raw["action_correct"] == action_correct
            assert raw["end_to_end_correct"] == (action_correct and raw["answer"] == raw["expected_final"])
        assert sum(r["end_to_end_correct"] for r in original["records"]) == original["end_to_end_correct"]
        actual = evaluate_rows(models[name][0], splits[split])
        assert [r["action_trace"]["generated_ids"] for r in actual["records"]] == [
            r["action_trace"]["generated_ids"] for r in original["records"]
        ]
        assert [r["final_trace"] for r in actual["records"]] == [r["final_trace"] for r in original["records"]]
        assert actual["end_to_end_correct"] == original["end_to_end_correct"]
        write("capstone_cpu_" + name + "_" + split + ".json", actual)
        results[name + "_" + split] = {k: v for k, v in actual.items() if k != "records"}
    sampler = random.Random(42 + 5000)
    token_total = 0
    sample_hash = hashlib.sha256()
    for _ in range(350):
        rows = _balanced_sample(splits["train"], sampler, 24)
        _, labels = prepare_batch(rows)
        token_total += int((labels != -100).sum())
        sample_hash.update("|".join(row["id"] for row in rows).encode())
    assert token_total == 145163
    results["student_recipe_reconstructed"] = {
        "steps": 350,
        "batch_size": 24,
        "effective_answer_tokens": token_total,
        "ordered_row_ids_sha256": sample_hash.hexdigest(),
        "scope": "CPU deterministic recipe reconstruction; formal GPU per-batch choices were not recorded",
    }
    write("capstone_summary.json", results)
    return results


def short_checks():
    chosen = torch.tensor([-4.0], requires_grad=True)
    rejected = torch.tensor([-3.0], requires_grad=True)
    loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach(), beta=0.1)
    loss.backward()
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert abs(chosen.grad.item() + 0.05) < 1e-7
    assert abs(rejected.grad.item() - 0.05) < 1e-7
    policy = CapstoneModel()
    reference = frozen_reference(policy)
    splits, _ = build_dataset()
    pair = preference_pairs(splits["train"])[0]
    capstone_loss, _ = preference_loss(policy, reference, [pair])
    assert abs(capstone_loss.item() - math.log(2)) < 1e-6
    assert not any(p.requires_grad for p in reference.parameters())
    student = torch.tensor([[[0.4, -0.2], [0.8, 0.3]]], requires_grad=True)
    teacher = torch.tensor([[[0.7, -0.3], [-0.1, 0.9]]])
    labels = torch.tensor([[-100, 1]])
    temperature = 2.0
    teacher_p = (teacher[0, 1] / temperature).softmax(-1)
    student_p = (student[0, 1] / temperature).softmax(-1)
    manual = (
        -0.5 * student[0, 1].log_softmax(-1)[1]
        + 0.5 * temperature**2 * (teacher_p * (teacher_p.log() - student_p.log())).sum()
    )
    kd = distillation_loss(student, teacher, labels)
    assert abs(kd.item() - manual.item()) < 1e-6
    initial = TinyLM(ModelConfig(width=8, layers=1, heads=1))
    save_checkpoint(WORK / "style.pt", initial, step=0)
    command = [sys.executable, "scripts/prepare_data.py", "--kind", "preference", "--output", str(WORK / "generated")]
    prepared = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    validation = [json.loads(x) for x in (WORK / "generated/preference/validation.jsonl").read_text().splitlines()]
    assert validation[0]["prompt"] == "0+5=?" and validation[0]["chosen"] == "5" and validation[0]["rejected"] == "6"
    write("exploration_validation.json", validation)
    command2 = [
        sys.executable,
        "scripts/train.py",
        "--task",
        "dpo",
        "--checkpoint",
        str(WORK / "style.pt"),
        "--data",
        str(WORK / "generated/preference/train.jsonl"),
        "--train",
        "--steps",
        "200",
        "--stop-after",
        "1",
        "--output",
        str(WORK / "preferred.pt"),
        "--device",
        "cpu",
    ]
    trained = subprocess.run(command2, cwd=ROOT, text=True, capture_output=True, check=True)
    new, payload = load_checkpoint(WORK / "preferred.pt", "cpu")
    saved_reference = payload["training_state"]["dpo_reference"]["model"]
    assert all(torch.equal(t, saved_reference[n]) for n, t in initial.state_dict().items())
    changed = [n for n, t in initial.state_dict().items() if not torch.equal(t, new.state_dict()[n])]
    assert changed
    infer_command = [
        sys.executable,
        "scripts/infer.py",
        str(WORK / "preferred.pt"),
        "--chat",
        "--prompt",
        "0+5=?",
        "--tokens",
        "32",
        "--temperature",
        "0",
        "--device",
        "cpu",
    ]
    inferred = subprocess.run(infer_command, cwd=ROOT, text=True, capture_output=True, check=True)
    receipt = {
        "dpo_loss": loss.item(),
        "chosen_gradient": chosen.grad.item(),
        "rejected_gradient": rejected.grad.item(),
        "capstone_loss": capstone_loss.item(),
        "first_capstone_pair": pair,
        "kd_loss": kd.item(),
        "kd_manual": manual.item(),
        "prepare": {"command": command, "stdout": prepared.stdout},
        "short_train": {
            "command": command2,
            "stdout": trained.stdout,
            "stderr": trained.stderr,
            "actual_updates": 1,
            "intended_schedule": 200,
            "changed_tensor_names": changed,
            "reference_tensor_equality": True,
        },
        "infer": {"command": infer_command, "stdout": inferred.stdout},
        "scope": "One CPU update from an untrained tiny checkpoint; validates CLI and freeze mechanism, no capability or formal GPU-training claim",
    }
    write("short_cpu_checks.json", receipt)
    return receipt


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    snapshot_inputs()
    summary = {
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": str(torch.get_num_threads()),
            "cuda_available": str(torch.cuda.is_available()),
        },
        "scope": "CPU arithmetic scoring/generation, full short finite-card experiment, deterministic denominators and public-capstone generation; no GPU timing verified",
    }
    summary["dpo"] = check_dpo()
    summary["natural"] = check_natural()
    summary["cards"] = check_cards()
    summary["capstone"] = check_capstone(fetch_capstone())
    summary["short_checks"] = short_checks()
    write("execution_summary.json", summary)
    print(
        json.dumps(
            {
                "environment": summary["environment"],
                "dpo": summary["dpo"],
                "cards_test": summary["cards"]["test"],
                "capstone_counts": {
                    k: (v.get("end_to_end_correct"), v.get("count")) for k, v in summary["capstone"].items()
                },
                "status": "all independent assertions passed",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
