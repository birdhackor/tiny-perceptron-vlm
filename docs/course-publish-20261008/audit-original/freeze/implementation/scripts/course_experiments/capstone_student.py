"""Optional capstone compression branch: identical small Dense CE/KD students.

This module is intentionally independent of the frozen four-stage main recipe.
The main MoE assistant stays intact. The two students start from the exact same
random weights and see the exact same sampled training rows. Deployment timing
and modality diagnostics live in the separate capstone_deployment module.
"""

import copy
import hashlib
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from scripts.course_experiments.capstone import _balanced_sample, _sync, file_sha256, write_json
from tiny_perceptron.alignment import distillation_loss
from tiny_perceptron.capstone import (
    CapstoneModel,
    build_dataset,
    default_config,
    evaluate_rows,
    export_inference,
    load_capstone,
    prepare_batch,
    save_capstone,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss
from tiny_perceptron.training import seed_everything


def _train_student(initial, teacher, rows, output, *, mode, steps, device, seed, seconds, manifest, teacher_sha):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    sampler = random.Random(seed + 5000)
    state = {
        "sampler_rng": sampler.getstate(),
        "effective_tokens": 0,
        "history": [],
        "mode": mode,
        "batch_size": 24,
        "lr": 0.003,
        "temperature": 2.0,
        "alpha": 0.5 if mode == "kd" else 0.0,
        "elapsed_training_seconds": 0.0,
    }
    metadata = {
        "seed": seed,
        "data_manifest": manifest,
        "requested_steps": steps,
        "schedule_completed": False,
        "parent_checkpoint_sha256": None,
        "teacher_checkpoint_sha256": teacher_sha,
        "student_mode": mode,
        "initial_state_identical_across_branches": True,
        "code_sha256": {
            "student_runner": file_sha256(__file__),
            "capstone_core": file_sha256(Path(__file__).resolve().parents[2] / "tiny_perceptron/capstone.py"),
        },
    }
    progress = output / "model-training.pt"
    step = 0
    if progress.is_file():
        model, previous = load_capstone(progress, device)
        if previous["metadata"] != {
            **metadata,
            "schedule_completed": previous["metadata"]["schedule_completed"],
            "effective_tokens": previous["metadata"].get("effective_tokens", 0),
        }:
            raise ValueError("Student resume provenance/configuration changed")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        optimizer.load_state_dict(previous["optimizer"])
        state = previous["training_state"]
        sampler.setstate(state["sampler_rng"])
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        if str(device).startswith("cuda") and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        step = previous["step"]
    model.train()
    _sync(device)
    started = time.perf_counter()

    def save_progress():
        state["sampler_rng"] = sampler.getstate()
        metadata["effective_tokens"] = state["effective_tokens"]
        metadata["schedule_completed"] = step == steps
        save_capstone(progress, model, stage="joint", step=step, optimizer=optimizer, state=state, metadata=metadata)

    try:
        while step < steps and time.perf_counter() - started < seconds:
            batch_rows = _balanced_sample(rows, sampler, 24)
            batch, labels = prepare_batch(batch_rows, device)
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(**batch)
            if mode == "kd":
                with torch.no_grad():
                    teacher_logits = teacher(**batch)["logits"]
                objective = distillation_loss(
                    output_batch["logits"], teacher_logits, labels, alpha=0.5, temperature=2.0
                )
            else:
                objective = masked_loss(output_batch["logits"], labels)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite student training objective")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            state["effective_tokens"] += int((labels != IGNORE).sum())
            if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
                state["history"].append({"step": step, "objective_before_update": float(objective.detach())})
            if step % 100 == 0:
                save_progress()
    except BaseException:
        _sync(device)
        state["elapsed_training_seconds"] += time.perf_counter() - started
        save_progress()
        raise
    _sync(device)
    state["elapsed_training_seconds"] += time.perf_counter() - started
    save_progress()
    public = export_inference(progress, output / "model.pt")
    report = {
        "mode": mode,
        "steps": step,
        "requested_steps": steps,
        "schedule_completed": step == steps,
        "budget_exhausted": step < steps,
        "seconds": state["elapsed_training_seconds"],
        "parameters": model.description(),
        "effective_tokens": state["effective_tokens"],
        "history": state["history"],
        "teacher_checkpoint_sha256": teacher_sha,
        "objective": "0.5 answer-only CE + 0.5 KL(teacher||student), T=2, includes T²"
        if mode == "kd"
        else "answer-only CE",
        "initialization": "identical random Dense state, not inherited MoE weights",
        "export": public,
    }
    write_json(output / "train-report.json", report)
    return model, report


def run(ctx):
    """Optional sixth GPU job; formal steps/configuration fixed before test use."""
    from tiny_perceptron.capstone_quantization import load_quantized_capstone, quantize_capstone

    device = ctx.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Student job requested unavailable CUDA")
    torch.set_num_threads(2)
    output = Path(ctx.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(ctx.dependencies) / "capstone_preference/model.pt"
    teacher, payload = load_capstone(source, device)
    splits, manifest = build_dataset(ctx.seed)
    if (
        payload["metadata"]["data_manifest"] != manifest
        or payload["stage"] != "dpo"
        or not payload["metadata"].get("schedule_completed")
    ):
        raise ValueError("Student requires the complete, matching frozen capstone teacher")
    teacher.requires_grad_(False)
    teacher.eval()
    seed_everything(ctx.seed)
    initial = CapstoneModel(replace(default_config(dense=True), width=48)).to(device)
    initial_sha = hashlib.sha256(
        b"".join(tensor.detach().cpu().numpy().tobytes() for tensor in initial.state_dict().values())
    ).hexdigest()
    steps = max(1, round(350 * getattr(ctx, "step_scale", 1.0)))
    branches = {}
    students = {}
    for mode in ("ce", "kd"):
        model, report = _train_student(
            initial,
            teacher,
            splits["train"],
            output / mode,
            mode=mode,
            steps=steps,
            device=device,
            seed=ctx.seed,
            seconds=210,
            manifest=manifest,
            teacher_sha=file_sha256(source),
        )
        branches[mode] = report
        students[mode] = model
    completed = all(branch["schedule_completed"] for branch in branches.values())
    evaluations = {}
    if completed:
        for mode, model in students.items():
            for split in ("validation", "test"):
                evaluation = evaluate_rows(model, splits[split])
                write_json(output / f"{split}-{mode}.json", evaluation)
                evaluations[f"{split}_{mode}"] = {key: value for key, value in evaluation.items() if key != "records"}
    final = export_inference(output / "kd/model-training.pt", output / "model.pt")
    compressed = quantize_capstone(output / "model.pt", output / "model-int4.pt", bits=4)
    if completed:
        packed_model, _ = load_quantized_capstone(output / "model-int4.pt", device)
        evaluation = evaluate_rows(packed_model, splits["test"])
        write_json(output / "test-kd-ptq4.json", evaluation)
        evaluations["test_kd_ptq4"] = {key: value for key, value in evaluation.items() if key != "records"}
    report = {
        "teacher_checkpoint_sha256": file_sha256(source),
        "initial_student_state_sha256": initial_sha,
        "data_manifest": manifest,
        "steps_per_branch": steps,
        "recipe_frozen_before_test": True,
        "branches": branches,
        "evaluations": evaluations,
        "final_model": final,
        "ptq": compressed,
        "schedule_completed": completed,
        "budget_exhausted": not completed,
        "requested_steps": 2 * steps,
        "steps": sum(branch["steps"] for branch in branches.values()),
        "scope": "optional Dense compression branch; CE and KD share exact initialization and sample sequence; does not prove distillation always wins",
    }
    write_json(output / "student-report.json", report)
    return report
