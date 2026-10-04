"""Independent 19.4 CPU audit; never updates the formal capstone weights."""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from scripts.check_technical_reviews import sections
from scripts.course_experiments.capstone import _balanced_sample, train_stage
from tiny_perceptron.capstone import (
    DATA_VERSION,
    TOK,
    build_dataset,
    encode_record,
    evaluate_rows,
    frozen_reference,
    load_capstone,
    preference_loss,
    preference_pairs,
    prepare_batch,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import loss_sum, masked_loss

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_04"
ENVIRONMENT = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "device": "cpu",
    "threads": "2",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(suffix, value):
    path = Path(str(PREFIX) + suffix)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def tensor_record(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "finite": bool(torch.isfinite(tensor).all()),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def main():
    torch.set_num_threads(2)
    body = dict(sections(ROOT / "course/chapters/19.md"))["19.4"]
    Path(str(PREFIX) + "_read_section.md").write_text(body)
    section_sha = hashlib.sha256(body.encode()).hexdigest()
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    examples = []
    for task in ["style", "concept"]:
        executable = code.replace('== "style"', '== "' + task + '"')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(executable, "course/chapters/19.md#19.4", "exec"), {})
        examples.append({"task": task, "executed_code": executable, "stdout": output.getvalue()})
    splits, manifest = build_dataset()
    save("_dataset.json", {"splits": splits, "manifest": manifest})
    hand = []
    for task in ["style", "concept"]:
        row = next(row for row in splits["train"] if row["task"] == task)
        user_bytes, answer_bytes = len(row["user"].encode()), len(row["answer"].encode())
        hand.append({
            "task": task, "user": row["user"], "answer": row["answer"],
            "user_utf8_bytes": user_bytes, "answer_utf8_bytes": answer_bytes,
            "pretrain_derivation": f"user {user_bytes} + LF 1 + answer {answer_bytes} + EOS 1 = {user_bytes + answer_bytes + 2}; BOS is input only",
            "sft_derivation": f"answer {answer_bytes} + EOS 1 = {answer_bytes + 1}; all prefix targets ignored",
        })
    mixed = [next(row for row in splits["train"] if row["task"] == task) for task in ["style", "concept"]]
    batch, labels = prepare_batch(mixed)
    zeros = torch.zeros(*labels.shape, 264, requires_grad=True)
    total, count = loss_sum(zeros, labels)
    mean = masked_loss(zeros, labels)
    mean.backward()
    mixed_probe = {
        "labels": labels.tolist(), "input_ids": batch["ids"].tolist(),
        "valid": batch["valid"].tolist(), "dtype": str(labels.dtype),
        "logits_shape": list(zeros.shape), "effective_count": int(count),
        "expected_count": 10 + 29, "sum_loss": float(total.detach()),
        "mean_loss": float(mean.detach()), "expected_mean_ln264": float(torch.log(torch.tensor(264.0))),
        "ignored_gradient_max": float(zeros.grad[labels == IGNORE].abs().max()),
    }
    assert int(count) == 39 and mixed_probe["ignored_gradient_max"] == 0
    save("_examples.json", {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_04_replay.py", "environment": ENVIRONMENT, "source_sha256": section_sha, "examples": examples, "hand_calculation": hand, "mixed_probe": mixed_probe})
    stages = []
    previous_sha = None
    public = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    for index, (stage, experiment) in enumerate([
        ("pretrain", "capstone_pretrain"), ("sft", "capstone_sft"),
        ("joint", "capstone_joint"), ("dpo", "capstone_preference"),
    ]):
        print("Auditing", stage, flush=True)
        evidence = ROOT / "docs/course-experiments/capstone-evidence" / stage
        report = json.loads((evidence / "train-report.json").read_text())
        original = json.loads((ROOT / "docs/course-experiments/results" / (experiment + ".json")).read_text())
        validation = json.loads((evidence / "validation.json").read_text())
        disk_manifest = json.loads((evidence / "data-manifest.json").read_text())
        assert manifest == disk_manifest == report["data_manifest"]
        weights = list((ROOT / "outputs/integration-runs" / ("v2-" + stage) / "capstone-review" / experiment).glob("*/model.pt"))
        assert len(weights) == 1
        weights = weights[0]
        actual_sha = sha(weights)
        assert actual_sha == report["inference_export"]["sha256"]
        assert weights.stat().st_size == report["inference_export"]["bytes"]
        model, payload = load_capstone(weights)
        assert payload["metadata"]["parent_checkpoint_sha256"] == report["parent_checkpoint_sha256"] == previous_sha
        assert payload["metadata"]["data_manifest"] == manifest
        assert payload["stage"] == stage and payload["step"] == report["steps"]
        assert payload["data_version"] == DATA_VERSION and payload["tokenizer"] == TOK.state()
        assert payload["metadata"]["schedule_completed"] and payload["inference_only"]
        sampler = random.Random(42 + index * 1000)
        rows = [row for row in splits["train"] if row["image"] is None and row["audio"] is None] if index < 2 else splits["train"]
        pairs = preference_pairs(splits["train"])
        length_by_id = {row["id"]: int((encode_record(row, stage == "pretrain")[1] != IGNORE).sum()) for row in rows}
        token_count, preference_chosen_count, preference_rejected_count = 0, 0, 0
        for _ in range(report["steps"]):
            selected = _balanced_sample(rows, sampler, 24)
            token_count += sum(length_by_id[row["id"]] for row in selected)
            if stage == "dpo":
                selected_pairs = sampler.choices(pairs, k=12)
                preference_chosen_count += sum(len(pair["chosen"].encode()) + 1 for pair in selected_pairs)
                preference_rejected_count += sum(len(pair["rejected"].encode()) + 1 for pair in selected_pairs)
        assert token_count == report["effective_tokens"]
        recomputed = evaluate_rows(model, splits["validation"])
        row_differences = [old["id"] for old, new in zip(validation["records"], recomputed["records"], strict=True) if old != new]
        assert not row_differences, row_differences
        assert {key: value for key, value in recomputed.items() if key != "records"} == report["validation_summary"]
        replay_file = save("_" + stage + "_validation_replay.json", recomputed)
        expected_by_id = {row["id"]: row for row in splits["validation"]}
        for record in validation["records"]:
            row = expected_by_id[record["id"]]
            assert record["expected_action"] == row["answer"]
            trace = record["action_trace"]
            assert record["action_correct"] == (trace["eos"] and trace["raw"] == row["answer"])
            assert trace["raw"] == TOK.decode(trace["generated_ids"])
            assert record["end_to_end_correct"] == (record["action_correct"] and record["answer"] == record["expected_final"])
        public_item = next(item for item in public["models"] if item["id"] == stage)
        public_file = public_item["files"][0]
        public_path = hf_hub_download(public["repo"], public_file["path"], revision=public["revision"], token=False)
        assert sha(public_path) == public_file["sha256"]
        public_model, public_payload = load_capstone(public_path)
        assert all(torch.equal(value, public_payload["model"][name]) for name, value in payload["model"].items())
        tensors = {name: tensor_record(value) for name, value in payload["model"].items()}
        reference_probe = None
        if stage == "dpo":
            joint_weights = stages[-1]["weights_path"]
            joint_model, _ = load_capstone(ROOT / joint_weights)
            reference = frozen_reference(joint_model)
            assert all(not value.requires_grad for value in reference.parameters())
            before = {name: value.clone() for name, value in reference.state_dict().items()}
            policy_loss, margins = preference_loss(model, reference, pairs[:2])
            policy_loss.backward()
            reference_probe = {"reference_tensor_sha256": {name: tensor_record(value)["sha256"] for name, value in reference.state_dict().items()}, "all_frozen": all(not value.requires_grad for value in reference.parameters()), "no_reference_gradients": all(value.grad is None for value in reference.parameters()), "unchanged_after_backward": all(torch.equal(value, before[name]) for name, value in reference.state_dict().items()), "loss": float(policy_loss.detach()), "margins": margins, "policy_has_gradients": any(value.grad is not None for value in model.parameters())}
        stages.append({
            "stage": stage, "weights_path": weights.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": actual_sha, "bytes": weights.stat().st_size,
            "metadata": {key: value for key, value in payload.items() if key != "model"},
            "tensors": tensors, "train_report": report,
            "original_result": original, "original_validation": validation,
            "replay_validation": replay_file, "row_differences": row_differences,
            "recomputed_effective_targets": token_count,
            "preference_chosen_targets_one_model": preference_chosen_count,
            "preference_rejected_targets_one_model": preference_rejected_count,
            "dpo_scoring_targets_policy_plus_reference": 2 * (preference_chosen_count + preference_rejected_count),
            "reference_probe": reference_probe,
            "public_retrieval": {"repo": public["repo"], "revision": public["revision"], "file": public_file, "token": False, "actual_sha256": sha(public_path), "tensorwise_equal": True, "metadata": {key: value for key, value in public_payload.items() if key != "model"}},
        })
        previous_sha = actual_sha
    with tempfile.TemporaryDirectory() as directory:
        errors = []
        for kwargs in [{"stage": "sft"}, {"stage": "joint", "input_checkpoint": ROOT / stages[0]["weights_path"]}]:
            try:
                train_stage(output=directory, steps=1, seconds=1, validation=False, **kwargs)
            except ValueError as error:
                errors.append(str(error))
        assert len(errors) == 2
    selection = json.loads((ROOT / "docs/course-experiments/capstone-selection.json").read_text())
    assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"]
    assert selection["candidates"]["joint"]["validation_end_to_end_correct"] == 75
    assert selection["candidates"]["dpo"]["validation_end_to_end_correct"] == 71
    receipt = save("_audit.json", {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_04_replay.py", "environment": ENVIRONMENT, "result": "All assertions passed: examples, denominators, metadata/parent SHA, full CPU validation, original records, public pinned tensor equality, frozen reference, predecessor guards.", "source_sha256": section_sha, "stages": stages, "stage_guard_errors": errors, "selection": selection, "scope": "CPU replay validates weights and output identity; historical GPU times are checked against synchronized timer source and raw run records, not newly benchmarked."})
    fig_manifest = json.loads((ROOT / "outputs/reading-time/figures/manifest.json").read_text())
    figure = next(item for item in fig_manifest["records"] if item["source"] == "course/figures/capstone_pipeline.svg")
    assert sha(ROOT / figure["source"]) == figure["source_sha256"]
    assert sha(ROOT / figure["render"]) == figure["render_sha256"]
    render = Path(str(PREFIX) + "_pipeline.png")
    shutil.copyfile(ROOT / figure["render"], render)
    save("_figure.json", {"manifest_entry": figure, "persistent_render": render.relative_to(ROOT).as_posix(), "sha256": sha(render), "svg_xml_read": True, "inspection": "SVG source inspected: A->B->C with dashed C->D branch, fixed reference, joint recommended, frozen split and repeated validation. Visual inspection performed separately with view_image."})
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
