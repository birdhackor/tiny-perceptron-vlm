"""Frozen capstone deployment comparisons and explicit student release artifacts.

All test outputs are retained. Counterfactuals and benchmarks are diagnostics,
never a trigger to retrain, choose a favorable row, or replace a failed answer.
"""

import copy
import json
import shutil
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from scripts.course_experiments.capstone import EXPERIMENTS, file_sha256, run_deployment, write_json
from tiny_perceptron.capstone import (
    STAGES,
    CapstoneModel,
    build_dataset,
    digest,
    evaluate_rows,
    export_inference,
    generate_trace,
    load_capstone,
    modality_tensors,
    prepare_batch,
    prompt_ids,
)
from tiny_perceptron.capstone_quantization import load_quantized_capstone, quantize_capstone, tensor_bytes
from tiny_perceptron.model import masked_loss


def image_counterfactuals(rows):
    swaps = []
    colors = {"red": "green", "green": "blue", "blue": "red"}
    shapes = {"square": "circle", "circle": "square"}
    for row in rows:
        if row["task"] not in ("image_color", "image_shape", "joint"):
            continue
        swapped = copy.deepcopy(row)
        if row["task"] == "image_shape":
            swapped["image"]["shape"] = shapes[row["image"]["shape"]]
            answer = swapped["image"]["shape"]
        else:
            swapped["image"]["color"] = colors[row["image"]["color"]]
            answer = swapped["image"]["color"]
            if row["task"] == "joint":
                answer += "," + row["audio"]["pitch"]
        swapped["id"] = row["id"] + "-image-swap"
        swapped["answer"] = "DIRECT:" + answer
        swaps.append(swapped)
    return swaps


def counterfactual_pairs(original_evaluation, swapped_evaluation):
    originals = {row["id"]: row for row in original_evaluation["records"]}
    pairs = []
    for swapped in swapped_evaluation["records"]:
        original_id = swapped["id"].removesuffix("-image-swap")
        original = originals[original_id]
        pairs.append(
            {
                "original_id": original_id,
                "swapped_id": swapped["id"],
                "original_expected": original["expected_action"],
                "swapped_expected": swapped["expected_action"],
                "original_generated": original["action_trace"]["raw"],
                "swapped_generated": swapped["action_trace"]["raw"],
                "both_end_to_end_correct": original["end_to_end_correct"] and swapped["end_to_end_correct"],
                "generated_answer_changed": original["answer"] != swapped["answer"],
            }
        )
    return {
        "count": len(pairs),
        "both_end_to_end_correct": sum(row["both_end_to_end_correct"] for row in pairs),
        "pairs": pairs,
    }


@torch.no_grad()
def cache_consistency(model, rows, max_new_tokens=16, atol=1e-4, rtol=1e-4):
    """Fixed rows; compare actual decoding and logits with identical histories."""
    device = next(model.parameters()).device
    records = []
    was_training = model.training
    model.eval()
    try:
        for row in rows:
            full_trace = generate_trace(model, row, max_new_tokens, use_cache=False)
            cached_trace = generate_trace(model, row, max_new_tokens, use_cache=True)
            image, audio = modality_tensors(row)
            options = {
                "images": None if image is None else image.unsqueeze(0).to(device),
                "audio_features": None if audio is None else audio.unsqueeze(0).to(device),
            }
            ids = torch.tensor([prompt_ids(row)], device=device)
            cache = None
            comparisons = []
            for step in range(max_new_tokens):
                if ids.shape[-1] >= model.config.max_length:
                    break
                full = model(ids, **options)["logits"][0, -1]
                cached_result = model(
                    ids if cache is None else ids[:, -1:], **(options if cache is None else {}), cache=cache
                )
                cached = cached_result["logits"][0, -1]
                full_id, cached_id = int(full.argmax()), int(cached.argmax())
                comparisons.append(
                    {
                        "step": step,
                        "full_next_id": full_id,
                        "cached_next_id": cached_id,
                        "max_abs_logit_difference": float((full - cached).abs().max()),
                        "allclose": bool(torch.allclose(full, cached, atol=atol, rtol=rtol)),
                    }
                )
                cache = cached_result["cache"]
                # Same history isolates cache arithmetic from differing choices.
                ids = torch.cat((ids, ids.new_tensor([[full_id]])), -1)
                if full_id < 8:
                    break
            records.append(
                {
                    "id": row["id"],
                    "task": row["task"],
                    "full_trace": full_trace,
                    "cached_trace": cached_trace,
                    "generated_ids_equal": full_trace["generated_ids"] == cached_trace["generated_ids"],
                    "same_history_logit_comparisons": comparisons,
                }
            )
    finally:
        model.train(was_training)
    return {
        "count": len(records),
        "selection": "first validation row for each task, sorted by task name; selected without generation results",
        "atol": atol,
        "rtol": rtol,
        "max_new_tokens": max_new_tokens,
        "records": records,
        "all_generated_ids_equal": all(row["generated_ids_equal"] for row in records),
        "all_logits_close": all(
            entry["allclose"] for row in records for entry in row["same_history_logit_comparisons"]
        ),
    }


def benchmark_training_step(model, rows, warmup=3, measured=10):
    """Same fixed batch, ordinary float32 forward/backward, no optimizer updates."""
    if warmup < 0 or measured < 1:
        raise ValueError("Benchmark needs positive measured iterations")
    device = next(model.parameters()).device
    batch, labels = prepare_batch(rows, device)
    was_training = model.training
    state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.train()
    samples = []
    cuda = device.type == "cuda"
    baseline = torch.cuda.memory_allocated(device) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for iteration in range(warmup + measured):
            model.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model(**batch)
            loss = masked_loss(result["logits"], labels) + 0.01 * result["auxiliary"]
            loss.backward()
            if cuda:
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            if iteration >= warmup:
                samples.append(seconds)
        peak = torch.cuda.max_memory_allocated(device) if cuda else None
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    unchanged = all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in state_before.items())
    if not unchanged:
        raise ValueError("A mechanism benchmark unexpectedly changed model weights")
    return {
        "parameters": model.description(),
        "device": str(device),
        "dtype": "float32",
        "batch_ids": [row["id"] for row in rows],
        "batch_shape": list(batch["ids"].shape),
        "warmup_iterations": warmup,
        "measured_iterations": measured,
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "optimizer_updates": 0,
        "weights_unchanged": unchanged,
        "cuda_baseline_allocated_bytes": baseline,
        "cuda_peak_allocated_bytes": peak,
        "cuda_peak_extra_bytes": None if peak is None else peak - baseline,
    }


@torch.no_grad()
def benchmark_generation(model, row, *, warmup=3, measured=10, max_new_tokens=16):
    """One preselected row, full recomputation versus cache, with actual outputs."""
    if warmup < 0 or measured < 1 or max_new_tokens < 1:
        raise ValueError("Generation benchmark needs positive measured iterations and token limit")
    device = next(model.parameters()).device
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("This generation comparison requires the same float32 model")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    modes = {}
    for name, use_cache in (("full", False), ("cache", True)):
        iterations = []
        for iteration in range(warmup + measured):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            trace = generate_trace(model, row, max_new_tokens, use_cache=use_cache)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            iterations.append(
                {
                    "iteration": iteration,
                    "phase": "warmup" if iteration < warmup else "measured",
                    "seconds": seconds,
                    "generated_token_count_including_eos": len(trace["generated_ids"]),
                    "eos": trace["eos"],
                    "stop_reason": trace["stop_reason"],
                    "trace": trace,
                }
            )
        measured_runs = iterations[warmup:]
        samples = [record["seconds"] for record in measured_runs]
        modes[name] = {
            "use_cache": use_cache,
            "warmup_iterations": warmup,
            "measured_iterations": measured,
            "seconds": samples,
            "median_seconds": statistics.median(samples),
            "mean_seconds": statistics.mean(samples),
            "iterations": iterations,
        }
    equal = [
        left["trace"]["generated_ids"] == right["trace"]["generated_ids"]
        for left, right in zip(modes["full"]["iterations"], modes["cache"]["iterations"], strict=True)
    ]
    unchanged = all(torch.equal(tensor, model.state_dict()[name].detach().cpu()) for name, tensor in before.items())
    if not unchanged:
        raise ValueError("Generation benchmark unexpectedly changed model weights")
    return {
        "row_id": row["id"],
        "task": row["task"],
        "row": copy.deepcopy(row),
        "selection": "first style row in the frozen validation split, selected before generation results",
        "parameters": model.description(),
        "device": str(device),
        "dtype": "float32",
        "max_new_tokens": max_new_tokens,
        "modes": modes,
        "generated_ids_equal_by_iteration": equal,
        "all_generated_ids_equal": all(equal),
        "optimizer_updates": 0,
        "weights_unchanged": unchanged,
        "timing_scope": "one generate_trace call including prompt preparation, device transfers, greedy generation and decoding; CUDA synchronized before and after each call; excludes model loading, startup, HF I/O and weight verification",
        "limitation": "one fixed validation prompt and one float32 model/device; not a general generation throughput benchmark and no generation memory peak is measured",
    }


def run(ctx):
    # The frozen implementation performs all stages, untrained, PTQ4 and audio swaps.
    original = run_deployment(ctx)
    # Recommendation fixed from validation coverage before this official test.
    recommended_stage = "joint"
    output = Path(ctx.output)
    splits, manifest = build_dataset(ctx.seed)
    exports = {}
    for stage in STAGES:
        source = Path(ctx.dependencies) / EXPERIMENTS[stage] / "model.pt"
        receipt = export_inference(source, output / f"{stage}.pt")
        model, _ = load_capstone(output / f"{stage}.pt")
        receipt["tensor_bytes"] = tensor_bytes(model.state_dict())
        exports[stage] = receipt
    dpo_source = Path(ctx.dependencies) / EXPERIMENTS["dpo"] / "model.pt"
    shutil.copyfile(output / "model-int4.pt", output / "dpo-int4.pt")
    exports["dpo-int4"] = dict(original["ptq"], checkpoint=str(output / "dpo-int4.pt"))
    exports["dpo-int8"] = quantize_capstone(dpo_source, output / "dpo-int8.pt", bits=8)
    int8, _ = load_quantized_capstone(output / "dpo-int8.pt", ctx.device)
    ptq8_evaluation = evaluate_rows(int8, splits["test"])
    write_json(output / "test-ptq8.json", ptq8_evaluation)
    del int8
    final_source = Path(ctx.dependencies) / EXPERIMENTS[recommended_stage] / "model.pt"
    joint_ptq = {}
    for bits in (4, 8):
        identity = f"joint-int{bits}"
        exports[identity] = quantize_capstone(final_source, output / f"{identity}.pt", bits=bits)
        compressed, _ = load_quantized_capstone(output / f"{identity}.pt", ctx.device)
        evaluation = evaluate_rows(compressed, splits["test"])
        evaluation["diagnostic_stage"] = recommended_stage
        evaluation["source_checkpoint_sha256"] = file_sha256(final_source)
        write_json(output / f"test-joint-ptq{bits}.json", evaluation)
        joint_ptq[identity] = {key: value for key, value in evaluation.items() if key != "records"}
        del compressed
    model, _ = load_capstone(final_source, ctx.device)
    swapped = evaluate_rows(model, image_counterfactuals(splits["test"]))
    swapped["diagnostic_stage"] = recommended_stage
    swapped["source_checkpoint_sha256"] = file_sha256(final_source)
    write_json(output / "test-joint-image-swaps.json", swapped)
    baseline = json.loads((output / "test-joint.json").read_text(encoding="utf-8"))
    image_pairs = counterfactual_pairs(baseline, swapped)
    image_pairs["diagnostic_stage"] = recommended_stage
    write_json(output / "test-joint-image-pairs.json", image_pairs)
    # Keep the original DPO diagnostic, and distinguish the recommended model.
    dpo_audio_swapped = json.loads((output / "test-audio-swaps.json").read_text(encoding="utf-8"))
    dpo_audio_swapped["diagnostic_stage"] = "dpo"
    dpo_audio_swapped["source_checkpoint_sha256"] = file_sha256(dpo_source)
    write_json(output / "test-dpo-audio-swaps.json", dpo_audio_swapped)
    audio_rows = []
    for row in splits["test"]:
        if row["task"] == "joint":
            counterfactual = copy.deepcopy(row)
            counterfactual["audio"]["pitch"] = "high" if row["audio"]["pitch"] == "low" else "low"
            counterfactual["answer"] = f"DIRECT:{row['image']['color']},{counterfactual['audio']['pitch']}"
            counterfactual["id"] = row["id"] + "-audio-swap"
            audio_rows.append(counterfactual)
    audio_swapped = evaluate_rows(model, audio_rows)
    audio_swapped["diagnostic_stage"] = recommended_stage
    audio_swapped["source_checkpoint_sha256"] = file_sha256(final_source)
    write_json(output / "test-joint-audio-swaps.json", audio_swapped)
    first_by_task = {}
    for row in splits["validation"]:
        first_by_task.setdefault(row["task"], row)
    cache = cache_consistency(model, [first_by_task[task] for task in sorted(first_by_task)])
    cache["diagnostic_stage"] = recommended_stage
    cache["checkpoint_sha256"] = file_sha256(final_source)
    write_json(output / "cache-consistency.json", cache)
    if "style" not in first_by_task:
        raise ValueError("Frozen validation split lacks the predetermined style benchmark row")
    generation = benchmark_generation(model, first_by_task["style"])
    generation["checkpoint_sha256"] = file_sha256(final_source)
    generation["dataset_manifest_sha256"] = digest(manifest)
    generation["diagnostic_stage"] = recommended_stage
    write_json(output / "generation-benchmark.json", generation)
    batch_rows = splits["train"][:24]
    torch.manual_seed(ctx.seed)
    dense_config = replace(model.config, experts=0, top_k=2, width=80)
    dense = CapstoneModel(dense_config).to(ctx.device)
    benchmarks = {
        "moe": benchmark_training_step(model, batch_rows),
        "dense80": benchmark_training_step(dense, batch_rows),
        "comparison": "same first 24 training rows, warmup=3/measured=10, forward+backward without optimizer; Dense80 is random and only a mechanism baseline, not a capability-matched training recipe",
        "diagnostic_stage": recommended_stage,
        "checkpoint_sha256": file_sha256(final_source),
    }
    write_json(output / "mechanism-benchmark.json", benchmarks)
    write_json(
        output / "data.json",
        {
            "schema_version": 1,
            "license": "MIT",
            "manifest": manifest,
            "splits": splits,
            "modality_generation": "Use tiny_perceptron.capstone.modality_tensors with the stored RGB/tone specifications.",
        },
    )
    result = {
        **original,
        "recommended_stage": recommended_stage,
        "recommendation_basis": "validation capability coverage, fixed before deployment test; DPO remains a comparison branch and the precommitted student teacher",
        "final_model": exports[recommended_stage],
        "comparison_dpo_model": original["final_model"],
        "stages": {
            **{name: summary for name, summary in original["stages"].items() if name != "ptq4"},
            "dpo-int4": original["stages"]["ptq4"],
            "dpo-int8": {key: value for key, value in ptq8_evaluation.items() if key != "records"},
            **joint_ptq,
        },
        "public_stage_exports": exports,
        "ptq8": {key: value for key, value in ptq8_evaluation.items() if key != "records"},
        "ptq_source_stage": "dpo",
        "joint_ptq": joint_ptq,
        "audio_swap": {key: value for key, value in audio_swapped.items() if key != "records"},
        "dpo_audio_swap": {**original["audio_swap"], "diagnostic_stage": "dpo", "path": "test-dpo-audio-swaps.json"},
        "image_swap": {key: value for key, value in swapped.items() if key != "records"},
        "image_pairs": {key: value for key, value in image_pairs.items() if key != "pairs"},
        "cache": {key: value for key, value in cache.items() if key != "records"},
        "mechanism_benchmark": benchmarks,
        "generation_benchmark": generation,
        "data_export": {
            "path": "data.json",
            "sha256": file_sha256(output / "data.json"),
            "bytes": (output / "data.json").stat().st_size,
            "license": "MIT",
        },
        "architecture": {"type": "CapstoneModel", "config": asdict(model.config)},
        "interpretation": "PTQ reduces stored weights; loaded inference is float32. Counterfactual failures are retained. Random Dense80 timings do not establish a trained Dense/MoE capability advantage.",
    }
    write_json(output / "deployment-report.json", result)
    return result
