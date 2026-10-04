"""Frozen-stage training/evaluation for the integrated small-world capstone.

Every stage continues the same model. Test generations are deferred until the
whole recipe is frozen; validation is the only development feedback. Time-limited
jobs produce an explicitly incomplete checkpoint rather than a false success.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from tiny_perceptron.capstone import (
    DEFAULT_STEPS,
    STAGES,
    CapstoneModel,
    build_dataset,
    default_config,
    evaluate_rows,
    export_inference,
    frozen_reference,
    load_capstone,
    preference_loss,
    preference_pairs,
    prepare_batch,
    save_capstone,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss
from tiny_perceptron.training import seed_everything

EXPERIMENTS = {
    "pretrain": "capstone_pretrain",
    "sft": "capstone_sft",
    "joint": "capstone_joint",
    "dpo": "capstone_preference",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _balanced_sample(rows, sampler, count):
    tasks = {}
    for row in rows:
        tasks.setdefault(row["task"], []).append(row)
    names = sorted(tasks)
    return [sampler.choice(tasks[sampler.choice(names)]) for _ in range(count)]


def train_stage(
    stage,
    output,
    *,
    input_checkpoint=None,
    resume=None,
    device="cpu",
    steps=None,
    seconds=540,
    seed=42,
    batch_size=24,
    dense=False,
    validation=True,
):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if batch_size < 1 or seconds <= 0:
        raise ValueError("batch_size and seconds must be positive")
    steps = DEFAULT_STEPS[stage] if steps is None else steps
    if steps < 1:
        raise ValueError("A training stage needs at least one optimizer update")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed_everything(seed)
    splits, manifest = build_dataset(seed)
    write_json(output / "data-manifest.json", manifest)
    source = resume or input_checkpoint
    if source:
        model, previous = load_capstone(source, device)
        if previous["metadata"].get("data_manifest") != manifest:
            raise ValueError("Checkpoint and frozen training data differ")
        if resume:
            if previous.get("inference_only") or previous["stage"] != stage:
                raise ValueError("Resume requires a training checkpoint for the same stage")
            if previous["metadata"]["requested_steps"] != steps:
                raise ValueError("Cannot change the fixed update schedule while resuming")
        else:
            expected = STAGES[STAGES.index(stage) - 1] if stage != "pretrain" else None
            if previous["stage"] != expected or not previous["metadata"].get("schedule_completed"):
                raise ValueError("A stage requires the completed immediate predecessor")
    else:
        if stage != "pretrain":
            raise ValueError("Later stages cannot silently start from random weights")
        model, previous = CapstoneModel(default_config(dense)).to(device), None
    text_rows = [row for row in splits["train"] if row["image"] is None and row["audio"] is None]
    rows = text_rows if stage in ("pretrain", "sft") else splits["train"]
    pairs = preference_pairs(splits["train"]) if stage == "dpo" else None
    reference = frozen_reference(model) if stage == "dpo" else None
    lr = {"pretrain": 0.003, "sft": 0.003, "joint": 0.0015, "dpo": 0.0002}[stage]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sampler = random.Random(seed + STAGES.index(stage) * 1000)
    state = {
        "sampler_rng": sampler.getstate(),
        "history": [],
        "effective_tokens": 0,
        "elapsed_training_seconds": 0.0,
        "batch_size": batch_size,
        "lr": lr,
        "auxiliary_weight": 0.01,
        "dpo_beta": 0.1,
        "dpo_supervised_anchor_weight": 0.2,
    }
    step = 0
    if resume:
        state = previous["training_state"]
        if state["batch_size"] != batch_size:
            raise ValueError("Resume batch size must match the saved experiment")
        if previous.get("optimizer") is None:
            raise ValueError("Resume checkpoint has no optimizer state")
        optimizer.load_state_dict(previous["optimizer"])
        sampler.setstate(state["sampler_rng"])
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        if str(device).startswith("cuda") and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        if reference is not None:
            if previous["reference"] is None:
                raise ValueError("DPO resume checkpoint is missing its original frozen reference")
            reference.load_state_dict(previous["reference"], strict=True)
        step = previous["step"]
    metadata = {
        "seed": seed,
        "data_manifest": manifest,
        "requested_steps": steps,
        "schedule_completed": False,
        "parent_checkpoint_sha256": previous["metadata"].get("parent_checkpoint_sha256")
        if resume
        else file_sha256(source)
        if source
        else None,
        "code_sha256": {
            "tiny_perceptron/capstone.py": file_sha256(
                Path(__file__).resolve().parents[2] / "tiny_perceptron/capstone.py"
            ),
            "scripts/course_experiments/capstone.py": file_sha256(__file__),
        },
        "task_scope": "synthetic RGB shapes/tone/text, explicit calculator, tiny trained Chinese grammar; not a general assistant",
    }
    if resume and previous["metadata"]["code_sha256"] != metadata["code_sha256"]:
        raise ValueError("Resume code hashes changed; do not silently alter the experiment")
    model.train()
    _sync(device)
    started = time.perf_counter()
    initial_step = step

    def save_progress():
        state["sampler_rng"] = sampler.getstate()
        metadata["effective_tokens"] = state["effective_tokens"]
        metadata["schedule_completed"] = step == steps
        save_capstone(
            output / "model-training.pt",
            model,
            stage=stage,
            step=step,
            optimizer=optimizer,
            state=state,
            metadata=metadata,
            reference=reference,
        )

    try:
        while step < steps and time.perf_counter() - started < seconds:
            optimizer.zero_grad(set_to_none=True)
            batch_rows = _balanced_sample(rows, sampler, batch_size)
            batch, labels = prepare_batch(batch_rows, device, pretrain=stage == "pretrain")
            if batch["ids"].shape[-1] > model.config.max_length:
                raise ValueError("Training record exceeds model context")
            result = model(**batch)
            ce = masked_loss(result["logits"], labels)
            loss = ce + 0.01 * result["auxiliary"]
            details = {}
            if stage == "dpo":
                selected = sampler.choices(pairs, k=max(1, batch_size // 2))
                preference, details = preference_loss(model, reference, selected, device)
                # Retain all capabilities while teaching the narrow style preference.
                # This explicit CE replay is part of the recipe, not pure DPO.
                loss = preference + 0.2 * ce + 0.01 * result["auxiliary"]
                details["dpo_loss"] = float(preference.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite capstone training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            state["effective_tokens"] += int((labels != IGNORE).sum())
            if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
                state["history"].append(
                    {
                        "step": step,
                        "loss_before_update": float(loss.detach()),
                        "ce_before_update": float(ce.detach()),
                        "auxiliary_before_update": float(result["auxiliary"].detach()),
                        **details,
                    }
                )
            if step % 100 == 0:
                save_progress()
    except BaseException:
        _sync(device)
        state["elapsed_training_seconds"] += time.perf_counter() - started
        save_progress()
        raise
    _sync(device)
    elapsed = time.perf_counter() - started
    state["elapsed_training_seconds"] += elapsed
    save_progress()
    receipt = export_inference(output / "model-training.pt", output / "model.pt")
    validation_result = None
    if validation and step == steps:
        validation_result = evaluate_rows(model, splits["validation"])
        write_json(output / "validation.json", validation_result)
    report = {
        "stage": stage,
        "requested_steps": steps,
        "steps": step,
        "new_steps": step - initial_step,
        "schedule_completed": step == steps,
        "budget_exhausted": step < steps,
        "seed": seed,
        "device": str(device),
        "seconds": elapsed,
        "elapsed_training_seconds": state["elapsed_training_seconds"],
        "parameters": model.description(),
        "effective_tokens": state["effective_tokens"],
        "history": state["history"],
        "data_manifest": manifest,
        "parent_checkpoint_sha256": metadata["parent_checkpoint_sha256"],
        "code_sha256": metadata["code_sha256"],
        "checkpoint": "model.pt",
        "training_checkpoint": "model-training.pt",
        "inference_export": receipt,
        "objective": "next-byte CE + valid-token router balance"
        if stage == "pretrain"
        else "assistant-only CE + valid-token router balance"
        if stage in ("sft", "joint")
        else "DPO(beta=.1,frozen immediate-predecessor) + .2 assistant-only replay CE + valid-token router balance",
        "validation_summary": None
        if validation_result is None
        else {key: value for key, value in validation_result.items() if key != "records"},
        "test_evaluated": False,
    }
    write_json(output / "train-report.json", report)
    return report


def _run_context(ctx, stage):
    predecessor = STAGES[STAGES.index(stage) - 1] if stage != "pretrain" else None
    source = None if predecessor is None else Path(ctx.dependencies) / EXPERIMENTS[predecessor] / "model.pt"
    steps = max(1, round(DEFAULT_STEPS[stage] * getattr(ctx, "step_scale", 1.0)))
    resume = Path(ctx.output) / "model-training.pt"
    return train_stage(
        stage,
        ctx.output,
        input_checkpoint=source,
        resume=resume if resume.is_file() else None,
        device=ctx.device,
        steps=steps,
        seed=ctx.seed,
    )


def run_pretrain(ctx):
    return _run_context(ctx, "pretrain")


def run_sft(ctx):
    return _run_context(ctx, "sft")


def run_joint(ctx):
    return _run_context(ctx, "joint")


def run_preference(ctx):
    return _run_context(ctx, "dpo")


def run_deployment(ctx):
    """Evaluate all fixed stages together, then the same FP32 model with PTQ.

    This is the first official test evaluation. Do not use these scores to tune
    the already-frozen updates/configuration. Failures remain in full row traces.
    """
    from tiny_perceptron.capstone_quantization import load_quantized_capstone, quantize_capstone

    output = Path(ctx.output)
    output.mkdir(parents=True, exist_ok=True)
    splits, manifest = build_dataset(ctx.seed)
    stages = {}
    seed_everything(ctx.seed)
    untrained = CapstoneModel().to(ctx.device)
    stages["untrained"] = evaluate_rows(untrained, splits["test"])
    for stage in STAGES:
        source = Path(ctx.dependencies) / EXPERIMENTS[stage] / "model.pt"
        model, payload = load_capstone(source, ctx.device)
        if payload["metadata"]["data_manifest"] != manifest or not payload["metadata"].get("schedule_completed"):
            raise ValueError("Cannot compare incomplete or differently split stages")
        stages[stage] = evaluate_rows(model, splits["test"])
        stages[stage]["checkpoint_sha256"] = file_sha256(source)
    source = Path(ctx.dependencies) / EXPERIMENTS["dpo"] / "model.pt"
    quantized = quantize_capstone(source, output / "model-int4.pt", bits=4)
    compressed, _ = load_quantized_capstone(output / "model-int4.pt", ctx.device)
    stages["ptq4"] = evaluate_rows(compressed, splits["test"])
    for stage, result in stages.items():
        write_json(output / f"test-{stage}.json", result)
    # A paired swap diagnostic holds the textual request unchanged. It checks
    # actual semantic outputs, not just feature/logit sensitivity.
    model, _ = load_capstone(source, ctx.device)
    swaps = []
    for row in splits["test"]:
        if row["task"] == "joint":
            swapped = dict(row, audio={**row["audio"], "pitch": "high" if row["audio"]["pitch"] == "low" else "low"})
            swapped["answer"] = f"DIRECT:{row['image']['color']},{swapped['audio']['pitch']}"
            swapped["id"] = row["id"] + "-audio-swap"
            swaps.append(swapped)
    sensitivity = evaluate_rows(model, swaps)
    write_json(output / "test-audio-swaps.json", sensitivity)
    export = export_inference(source, output / "model.pt")
    result = {
        "data_manifest": manifest,
        "recipe_frozen_before_test": True,
        "stages": {
            name: {key: value for key, value in result.items() if key != "records"} for name, result in stages.items()
        },
        "ptq": quantized,
        "final_model": export,
        "audio_swap": {key: value for key, value in sensitivity.items() if key != "records"},
        "limitation": "A synthetic small-world assistant. Exact per-row failures are saved; not evidence of general language or multimodal competence.",
    }
    write_json(output / "deployment-report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seconds", type=float, default=540)
    args = parser.parse_args()
    print(
        json.dumps(
            train_stage(
                args.stage,
                args.output,
                input_checkpoint=args.input_checkpoint,
                resume=args.resume,
                device=args.device,
                steps=args.steps,
                seconds=args.seconds,
            ),
            ensure_ascii=False,
        )
    )
