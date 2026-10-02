"""第14–16章的可重跑實驗：真實更新、固定資料與完整 held-out 分母。

小模型上的參數／時間比較不外推到大型模型。每個入口回傳可序列化資料，
權重與資料切分另存在 Context.output；由總 runner 決定裝置與實驗順序。
"""

import copy
import json
import math
import random
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.course_experiments.common import (
    evaluate_lm,
    extract_asset,
    load_lm,
    new_lm,
    records_sha256,
    seed,
    split_records,
    text_examples,
    write_json,
)
from tiny_perceptron.data import IGNORE, ByteTokenizer, load_jsonl, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from tiny_perceptron.modern import MoEFFN
from tiny_perceptron.training import save_checkpoint


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _description(model):
    return {**model.description(), "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}


def _copy_matching(source, target):
    """相同形狀的表共用初值；新增／變形的表保留固定 seed 的初始化。"""
    original, current = source.state_dict(), target.state_dict()
    names = []
    for name, value in current.items():
        if name in original and value.shape == original[name].shape:
            # tied 的 output 是 embedding 同一物件，不能讓第二次 copy 覆寫 embedding。
            if name == "output.weight" and target.config.tied:
                continue
            value.copy_(original[name])
            names.append(name)
    target.load_state_dict(current)
    return names


def _clone_config(source, ctx, **changes):
    seed(ctx.seed)
    target = TinyLM(replace(source.config, **changes)).to(ctx.device)
    return target, _copy_matching(source, target)


def _text_dataset(ctx):
    directory = extract_asset(ctx, "tinystories")
    files = list(directory.rglob("tinystories-train-512.jsonl"))
    if len(files) != 1:
        raise ValueError(f"預期唯一 TinyStories 資料檔，找到 {len(files)} 個")
    records = load_jsonl(files[0])
    for record in records:
        record["family"] = record.get("text_sha256", records_sha256([{"text": record["text"]}]))
    data = split_records(records, ctx.seed)
    write_json(ctx.output / "dataset.json", data)
    return data


def _sft_dataset(ctx):
    data = json.loads(ctx.dependency("sft", "dataset.json").read_text(encoding="utf-8"))
    if any(not data.get(name) for name in ("train", "validation", "test")):
        raise ValueError("sft/dataset.json 需要非空 train/validation/test")
    write_json(ctx.output / "dataset.json", data)
    return data


def _data_report(data):
    return {name: {"records": len(rows), "sha256": records_sha256(rows)} for name, rows in data.items()}


def _runtime(ctx):
    return {
        "device": str(ctx.device),
        "torch": str(torch.__version__),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if str(ctx.device).startswith("cuda") else None,
    }


def _amp(device, dtype):
    return torch.autocast(torch.device(device).type, dtype=dtype) if dtype is not None else nullcontext()


def _forward(model, ids, valid, forward=None):
    """MoE 的輔助目標只計有效輸入；stock FFN 本身不知道 PAD 遮罩。"""
    auxiliary, handles = [], []
    if model.config.experts:

        def collect(module, args, result):
            flat = args[0][valid]
            probability = module.router(flat).float().softmax(-1)
            chosen = result[2][valid.reshape(-1)]
            load = F.one_hot(chosen, len(module.experts)).float().mean((0, 1))
            auxiliary.append(len(module.experts) * (load.detach() * probability.mean(0)).sum())

        handles = [block.ffn.register_forward_hook(collect) for block in model.blocks]
    try:
        result = (forward or model)(ids, valid=valid)
    finally:
        for handle in handles:
            handle.remove()
    if auxiliary:
        result["auxiliary"] = torch.stack(auxiliary).sum()
    return result


@torch.no_grad()
def _nll(model, examples, ctx, dtype=None, forward=None):
    model.eval()
    summed, tokens = 0.0, 0
    for start in range(0, len(examples), 16):
        x, y, valid = (t.to(ctx.device) for t in pad_batch(examples[start : start + 16]))
        with _amp(ctx.device, dtype):
            result = _forward(model, x, valid, forward)
            total, count = loss_sum(result["logits"].float(), y)
        summed += float(total)
        tokens += int(count)
    return {"nll": summed / tokens, "nll_sum": summed, "effective_tokens": tokens, "examples": len(examples)}


def _train(
    model,
    records,
    ctx,
    *,
    name,
    steps,
    mode="text",
    auxiliary=0.0,
    batch_size=16,
    lr=0.003,
    forward=None,
    dtype=None,
    micro_sizes=None,
    deadline=None,
):
    """固定 seed 的 batch；累積時共用有效 token 分母，最後才 clip 與 step。"""
    examples = text_examples(records, mode, model.config.max_length)
    initial = _nll(model, examples, ctx, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sampler = random.Random(ctx.seed)
    use_scaler = dtype == torch.float16 and torch.device(ctx.device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    history, latencies, effective_tokens, updates, skipped, complete = [], [], 0, 0, 0, 0
    if str(ctx.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(ctx.device)
        memory_before = torch.cuda.memory_allocated(ctx.device)
    else:
        memory_before = None
    _sync(ctx.device)
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        batch = sampler.choices(examples, k=batch_size)
        total_tokens = sum(int((y != IGNORE).sum()) for _, y in batch)
        sizes = micro_sizes or [batch_size]
        if sum(sizes) != batch_size:
            raise ValueError("micro_sizes 總和必須等於 batch_size")
        optimizer.zero_grad(set_to_none=True)
        _sync(ctx.device)
        step_started = time.perf_counter()
        value, offset = 0.0, 0
        for size in sizes:
            x, y, valid = (t.to(ctx.device) for t in pad_batch(batch[offset : offset + size]))
            offset += size
            with _amp(ctx.device, dtype):
                result = _forward(model, x, valid, forward)
                total, _ = loss_sum(result["logits"].float(), y)
                loss = total / total_tokens + auxiliary * result["auxiliary"] * size / batch_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} 第 {step + 1} 步 loss 非有限值")
            scaler.scale(loss).backward()
            value += float(loss.detach())
        scaler.unscale_(optimizer)
        gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not gradients_finite and not use_scaler:
            raise FloatingPointError(f"{name} 第 {step + 1} 步 gradient 非有限值")
        if gradients_finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if gradients_finite:
            updates += 1
        else:
            skipped += 1
        _sync(ctx.device)
        latencies.append(time.perf_counter() - step_started)
        effective_tokens += total_tokens
        complete = step + 1
        if step == 0 or (step + 1) % max(1, steps // 8) == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": value,
                    "gradients_finite": gradients_finite,
                    "scaler_before": old_scale,
                    "scaler_after": scaler.get_scale(),
                }
            )
    _sync(ctx.device)
    seconds = time.perf_counter() - started
    memory_peak = torch.cuda.max_memory_allocated(ctx.device) if memory_before is not None else None
    metadata = {
        "seed": ctx.seed,
        "mode": mode,
        "lr": lr,
        "auxiliary": auxiliary,
        "records_sha256": records_sha256(records),
        "effective_tokens": effective_tokens,
        "amp_dtype": str(dtype),
        "successful_optimizer_updates": updates,
    }
    save_checkpoint(
        ctx.output / f"{name}.pt", model, optimizer, complete, metadata, {"grad_scaler": scaler.state_dict()}
    )
    return {
        "initial": initial,
        "final": _nll(model, examples, ctx, dtype),
        "history": history,
        "requested_steps": steps,
        "schedule": "constant",
        "learning_rate": lr,
        "batch_size": batch_size,
        "micro_batch_sizes": micro_sizes or [batch_size],
        "gradient_clip_norm": 1.0,
        "auxiliary_weight": auxiliary,
        "steps": complete,
        "optimizer_updates": updates,
        "skipped_updates": skipped,
        "budget_exhausted": complete < steps,
        "seconds": seconds,
        "effective_tokens": effective_tokens,
        "checkpoint": f"{name}.pt",
        "warm_step_median_seconds": statistics.median(latencies[3:] or latencies) if latencies else None,
        "memory_allocated_before_bytes": memory_before,
        "peak_memory_allocated_bytes": memory_peak,
        "peak_additional_allocated_bytes": memory_peak - memory_before if memory_peak is not None else None,
        "memory_note": "CUDA allocator allocated bytes；包含參數、optimizer及中間量"
        if memory_peak is not None
        else "此裝置沒有 CUDA allocator peak API，未聲稱峰值記憶體收益",
        "scaler_enabled": use_scaler,
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
    }


def _heldout(model, data, mode="text"):
    return {name: evaluate_lm(model, data[name], mode=mode) for name in ("validation", "test")}


def run_modern(ctx):
    started = time.perf_counter()
    data = _text_dataset(ctx)
    initial = new_lm(ctx, heads=4)
    variants = {
        "baseline": {},
        "rope": {"rotary": True},
        "rmsnorm": {"norm": "rms"},
        "relu2": {"activation": "relu2"},
        "swiglu": {"activation": "swiglu"},
        "tied": {"tied": True},
    }
    results = {}
    for name, changes in variants.items():
        model, shared = _clone_config(initial, ctx, **changes)
        training = _train(model, data["train"], ctx, name=name, steps=240, deadline=started + 520)
        results[name] = {
            "model": _description(model),
            "changes": changes,
            "copied_initial_tables": shared,
            "training": training,
            "heldout": _heldout(model, data),
        }
    # 下游以 baseline 為參照；這不是挑驗證集最低者。
    (ctx.output / "model.pt").write_bytes((ctx.output / "baseline.pt").read_bytes())
    return {
        "seed": ctx.seed,
        "runtime": _runtime(ctx),
        "dataset": _data_report(data),
        "variants": results,
        "seconds": time.perf_counter() - started,
        "comparison": "每項只改一個設計；同資料、batch抽樣、更新數與共有表初值。RoPE移除位置表、"
        "SwiGLU新增gate、tied共享表，所以總參數不同；固定的是更新/token預算。",
        "limitations": "單一seed、小型固定語料；不能據此宣稱架構普遍優劣或長度外推能力。",
        "all_requested_updates_completed": all(result["training"]["steps"] == 240 for result in results.values()),
    }


def _parameter_budget(model):
    total = sum(p.numel() for p in model.parameters())
    expert_parameters = router_parameters = 0
    for block in model.blocks:
        if isinstance(block.ffn, MoEFFN):
            expert_parameters += sum(p.numel() for p in block.ffn.experts.parameters())
            router_parameters += sum(p.numel() for p in block.ffn.router.parameters())
    active = total - expert_parameters
    if model.config.experts:
        active += expert_parameters * model.config.top_k // model.config.experts
    return {
        "total_parameters": total,
        "active_parameter_proxy_per_token": active,
        "expert_parameters": expert_parameters,
        "router_parameters": router_parameters,
    }


def _matched_width(config, target):
    # 只在 CPU 建構計數；避免把搜尋候選都搬到 GPU。
    candidates = []
    for width in range(16, 161, 4):
        model = TinyLM(replace(config, width=width, experts=0))
        count = sum(p.numel() for p in model.parameters())
        candidates.append((abs(count - target), width, count))
    _, width, count = min(candidates)
    return width, count - target


def _router_gradients(model, examples, ctx):
    if not model.config.experts:
        return None
    x, y, valid = (t.to(ctx.device) for t in pad_batch(examples[:8]))
    model.eval()
    result = _forward(model, x, valid)
    total, count = loss_sum(result["logits"], y)
    parameters = [block.ffn.router.weight for block in model.blocks]
    task = torch.autograd.grad(total / count, parameters, retain_graph=True, allow_unused=True)
    auxiliary = torch.autograd.grad(result["auxiliary"], parameters, allow_unused=True)
    return {
        "effective_targets": int(count),
        "task_gradient_norms": [float(g.norm()) if g is not None else 0.0 for g in task],
        "unweighted_auxiliary_gradient_norms": [float(g.norm()) if g is not None else 0.0 for g in auxiliary],
    }


@torch.no_grad()
def _routing(model, examples, ctx):
    if not model.config.experts:
        return None
    counts = [torch.zeros(model.config.experts, dtype=torch.long) for _ in model.blocks]
    probability_sum = [torch.zeros(model.config.experts, dtype=torch.float64) for _ in model.blocks]
    aux_sum, input_tokens = [0.0 for _ in model.blocks], 0
    current_valid = None
    handles = []

    def make_hook(index):
        def collect(module, args, result):
            probability = module.router(args[0][current_valid]).float().softmax(-1)
            chosen = result[2][current_valid.reshape(-1)]
            count = torch.bincount(chosen.flatten(), minlength=model.config.experts).cpu()
            counts[index].add_(count)
            probability_sum[index].add_(probability.double().sum(0).cpu())
            load = count.to(probability.device).float() / count.sum()
            aux_sum[index] += float(model.config.experts * (load * probability.mean(0)).sum()) * len(probability)

        return collect

    model.eval()
    for i, block in enumerate(model.blocks):
        handles.append(block.ffn.register_forward_hook(make_hook(i)))
    try:
        for start in range(0, len(examples), 16):
            x, _, current_valid = (t.to(ctx.device) for t in pad_batch(examples[start : start + 16]))
            input_tokens += int(current_valid.sum())
            model(x, valid=current_valid)
    finally:
        for handle in handles:
            handle.remove()
    layers = []
    for count, probability, aux in zip(counts, probability_sum, aux_sum, strict=True):
        fraction = count.double() / count.sum()
        entropy = -float((fraction * fraction.clamp_min(1e-12).log()).sum())
        layers.append(
            {
                "dispatch_counts": count.tolist(),
                "dispatch_denominator": int(count.sum()),
                "load_fraction": fraction.tolist(),
                "load_entropy_nats": entropy,
                "normalized_load_entropy": entropy / math.log(model.config.experts),
                "mean_router_probability": (probability / input_tokens).tolist(),
                "mean_batch_auxiliary": aux / input_tokens,
            }
        )
    return {"effective_input_tokens": input_tokens, "padding_excluded": True, "layers": layers}


def run_moe(ctx):
    started = time.perf_counter()
    data = _text_dataset(ctx)
    initial = new_lm(ctx, heads=4)
    moe, _ = _clone_config(initial, ctx, experts=4, top_k=2)
    budget = _parameter_budget(moe)
    top1, _ = _clone_config(initial, ctx, experts=4, top_k=1)
    targets = {
        "dense_active_top1": _parameter_budget(top1)["active_parameter_proxy_per_token"],
        "dense_active_top2": budget["active_parameter_proxy_per_token"],
        "dense_total": budget["total_parameters"],
    }
    plans = []
    for name, target in targets.items():
        width, difference = _matched_width(initial.config, target)
        plans.append(
            (
                name,
                {"width": width},
                0.0,
                {
                    "target_parameters": target,
                    "actual_minus_target": difference,
                    "initialization_note": "固定seed；width不同時共有表形狀不同，不能共享完整初始化",
                },
            )
        )
    for k in (1, 2):
        for weight in (0.0, 0.01):
            plans.append((f"top{k}_aux{weight:g}", {"experts": 4, "top_k": k}, weight, {}))
    examples = text_examples(data["validation"], max_length=initial.config.max_length)
    results = {}
    for name, changes, weight, comparison in plans:
        model, shared = _clone_config(initial, ctx, **changes)
        before = _router_gradients(model, examples, ctx)
        training = _train(model, data["train"], ctx, name=name, steps=180, auxiliary=weight, deadline=started + 515)
        results[name] = {
            "model": _description(model),
            "budget": _parameter_budget(model),
            "comparison": comparison,
            "copied_initial_tables": shared,
            "training": training,
            "heldout": _heldout(model, data),
            "router_gradients_before": before,
            "router_gradients_after": _router_gradients(model, examples, ctx),
            "validation_routing": _routing(model, examples, ctx),
        }
        if name == "top2_aux0.01":
            (ctx.output / "model.pt").write_bytes((ctx.output / f"{name}.pt").read_bytes())
    return {
        "seed": ctx.seed,
        "dataset": _data_report(data),
        "variants": results,
        "seconds": time.perf_counter() - started,
        "teacher_variant": "top2_aux0.01",
        "runtime": _runtime(ctx),
        "comparison": "分別以完整模型total與每token active參數代理匹配Dense，width取最近4倍數。"
        "相同資料/token/更新預算；active參數不是精確FLOPs，路由與索引成本另看實測時間。",
        "auxiliary_note": "輔助loss每層相加，只含有效輸入token；路由統計分母為token×top_k，PAD排除。",
        "limitations": "單一seed；不同width的Dense改變表徵容量，dropless Python dispatch不保證加速。",
        "all_requested_updates_completed": all(result["training"]["steps"] == 180 for result in results.values()),
    }


def _benchmark(function, device, warmup=3, repeats=9):
    for _ in range(warmup):
        function()
    _sync(device)
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        function()
        _sync(device)
        samples.append(time.perf_counter() - began)
    return {
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "warmup_calls": warmup,
        "measured_calls": repeats,
        "synchronized": True,
    }


@torch.no_grad()
def _fixed_decode(model, ids, tokens, cached):
    sequence, state = ids.clone(), None
    for _ in range(tokens):
        current = sequence if state is None else sequence[:, -1:]
        result = model(current, cache=state)
        sequence = torch.cat((sequence, result["logits"][:, -1].argmax(-1, keepdim=True)), 1)
        state = result["cache"] if cached else None
    return sequence


@torch.no_grad()
def _cache_probe(model, ctx):
    model.eval()
    tok = ByteTokenizer()
    prefix = [tok.bos_id] + tok.encode("Once upon a time, a girl")
    prefix = prefix[: max(1, model.config.max_length - 16)]
    ids = torch.tensor([prefix], device=ctx.device)
    tokens = min(12, model.config.max_length - ids.shape[1])
    full, cached = _fixed_decode(model, ids, tokens, False), _fixed_decode(model, ids, tokens, True)
    sequence, state, errors = ids.clone(), None, []
    for _ in range(tokens):
        reference = model(sequence)["logits"][:, -1]
        current = sequence if state is None else sequence[:, -1:]
        result = model(current, cache=state)
        errors.append(float((reference - result["logits"][:, -1]).abs().max()))
        sequence = torch.cat((sequence, reference.argmax(-1, keepdim=True)), 1)
        state = result["cache"]
    cache = model(ids)["cache"]
    return {
        "prompt_tokens": ids.shape[1],
        "generated_tokens": tokens,
        "generated_ids_full": full[0, ids.shape[1] :].tolist(),
        "generated_ids_cached": cached[0, ids.shape[1] :].tolist(),
        "generated_text": tok.decode(full[0, ids.shape[1] :].tolist()),
        "identical_greedy_ids": bool(torch.equal(full, cached)),
        "per_step_logit_max_error": errors,
        "prefill_cache_bytes": sum(t.numel() * t.element_size() for pair in cache for t in pair),
        "prefill": _benchmark(lambda: model(ids), ctx.device),
        "full_recompute_decode": _benchmark(lambda: _fixed_decode(model, ids, tokens, False), ctx.device),
        "cached_decode": _benchmark(lambda: _fixed_decode(model, ids, tokens, True), ctx.device),
        "benchmark_note": "固定輸出步數，包含prefill；計時路線不因EOS提前停止，實際文本同列記錄。",
    }


def _gradients(model, x, y, valid, forward=None, positions=None, segments=None):
    model.zero_grad(set_to_none=True)
    result = (forward or model)(x, valid=valid, positions=positions, segments=segments)
    total, count = loss_sum(result["logits"], y)
    (total / count).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return result["logits"].detach(), gradients, float((total / count).detach())


def _gradient_error(a, b):
    if a.keys() != b.keys():
        raise ValueError("比較模型的可求導參數不同")
    return max(float((a[name] - b[name]).abs().max()) for name in a)


def _sdpa_probe(model, examples, ctx):
    manual, optimized = copy.deepcopy(model), copy.deepcopy(model)
    for block in manual.blocks:
        block.attention.backend = "manual"
    for block in optimized.blocks:
        block.attention.backend = "sdpa"
    x, y, valid = (t.to(ctx.device) for t in pad_batch(examples[:4]))
    output, grads, _ = _gradients(manual, x, y, valid)
    sdpa_output, sdpa_grads, _ = _gradients(optimized, x, y, valid)
    backend = {"selected": "unknown", "operator_names": [], "flash_attention_observed": False}
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if str(ctx.device).startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile, torch.no_grad():
            optimized(x, valid=valid)
            _sync(ctx.device)
        operators = [
            event.key
            for event in profile.key_averages()
            if "scaled_dot_product" in event.key or "attention_forward" in event.key
        ]
        backend["operator_names"] = operators
        if any("flash_attention" in name for name in operators):
            backend["selected"] = "flash_cpu" if any("for_cpu" in name for name in operators) else "flash_cuda"
            backend["flash_attention_observed"] = str(ctx.device).startswith("cuda")
        elif any("efficient_attention" in name for name in operators):
            backend["selected"] = "memory_efficient"
        elif any("cudnn_attention" in name for name in operators):
            backend["selected"] = "cudnn"
        elif any("attention_math" in name for name in operators):
            backend["selected"] = "math"
    except (RuntimeError, ImportError) as error:
        backend["reason"] = str(error)
    with torch.no_grad():
        timings = {
            "manual": _benchmark(lambda: manual(x, valid=valid), ctx.device),
            "sdpa": _benchmark(lambda: optimized(x, valid=valid), ctx.device),
        }
    return {
        "shape": list(x.shape),
        "dtype": str(output.dtype),
        "output_max_error": float((output - sdpa_output).abs().max()),
        "gradient_max_error": _gradient_error(grads, sdpa_grads),
        "backend": backend,
        "timings": timings,
    }


def _checkpoint_forward(model, ids, *, valid=None, positions=None, segments=None):
    if positions is None:
        positions = torch.arange(ids.shape[1], device=ids.device)
    x = model.embedding(ids)
    if model.position is not None:
        x = x + model.position(positions)
    for block in model.blocks:
        # 綁定當前block；避免backward重算時closure指到最後一層。
        def run(h, block=block):
            return block(h, valid=valid, positions=positions, segments=segments)[0]

        x = checkpoint(run, x, use_reentrant=False)
    return {"logits": model.output(model.final_norm(x)), "auxiliary": x.new_zeros(())}


def _packing_batch(examples, ctx):
    # 原始 shifted (x,y) 各自保留；最後一個目標不被改成下一文件的BOS。
    x, y, valid = (t.to(ctx.device) for t in pad_batch(examples))
    packed_x = torch.cat([a for a, _ in examples])[None].to(ctx.device)
    packed_y = torch.cat([b for _, b in examples])[None].to(ctx.device)
    positions = torch.cat([torch.arange(len(a)) for a, _ in examples])[None].to(ctx.device)
    segments = torch.cat([torch.full((len(a),), i) for i, (a, _) in enumerate(examples)])[None].to(ctx.device)
    return (x, y, valid), (packed_x, packed_y, torch.ones_like(packed_x, dtype=torch.bool), positions, segments)


def _packing_probe(model, records, ctx):
    examples = _mechanism_examples(records, model.config.max_length)
    lengths = [min(19, model.config.max_length // 3), min(31, model.config.max_length // 3)]
    # 明確取兩段短文prefix，與完整held-out品質評估分開。
    short = [(x[:n], y[:n]) for (x, y), n in zip(examples[:2], lengths, strict=True)]
    lengths = [len(x) for x, _ in short]
    padded, packed = _packing_batch(short, ctx)
    plain, isolated = copy.deepcopy(model), copy.deepcopy(model)
    a, ga, la = _gradients(plain, *padded)
    b, gb, lb = _gradients(isolated, *packed[:3], positions=packed[3], segments=packed[4])
    flat = torch.cat([a[i, : len(item[0])] for i, item in enumerate(short)], 0)
    altered = packed[0].clone()
    altered[:, : len(short[0][0])] = (altered[:, : len(short[0][0])] + 7) % model.config.vocab_size
    with torch.no_grad():
        changed_isolated = model(altered, positions=packed[3], segments=packed[4])["logits"]
        causal = model(packed[0], positions=packed[3])["logits"]
        changed_causal = model(altered, positions=packed[3])["logits"]
    start = len(short[0][0])
    return {
        "document_prefix_lengths": lengths,
        "padded_positions": padded[0].numel(),
        "effective_input_positions": int(padded[2].sum()),
        "packed_positions": packed[0].numel(),
        "positions_reset_per_document": True,
        "cross_document_targets_added": 0,
        "logit_max_error": float((flat - b[0]).abs().max()),
        "loss_error": abs(la - lb),
        "gradient_max_error": _gradient_error(ga, gb),
        "second_document_error_after_first_document_change_with_isolation": float(
            (b[:, start:] - changed_isolated[:, start:]).abs().max()
        ),
        "second_document_error_without_isolation": float((causal[:, start:] - changed_causal[:, start:]).abs().max()),
        "note": "短prefix只用來核對packing語意；所有held-out records仍完整評估。",
    }


def _accumulation_probe(model, examples, ctx):
    whole, accumulated = copy.deepcopy(model), copy.deepcopy(model)
    chosen = [(x[:n], y[:n]) for (x, y), n in zip(examples[:3], (13, 23, 37), strict=True)]
    x, y, valid = (t.to(ctx.device) for t in pad_batch(chosen))
    _, whole_gradients, _ = _gradients(whole, x, y, valid)
    denominator = sum(int((label != IGNORE).sum()) for _, label in chosen)
    accumulated.zero_grad(set_to_none=True)
    counts = []
    for group in (chosen[:1], chosen[1:]):
        a, b, mask = (t.to(ctx.device) for t in pad_batch(group))
        total, count = loss_sum(accumulated(a, valid=mask)["logits"], b)
        (total / denominator).backward()
        counts.append(int(count))
    gradients = {name: p.grad.detach().clone() for name, p in accumulated.named_parameters() if p.grad is not None}
    return {
        "micro_batch_sizes": [1, 2],
        "effective_token_counts": counts,
        "shared_denominator": denominator,
        "gradient_max_error": _gradient_error(whole_gradients, gradients),
    }


def _mechanism_examples(records, max_length):
    """抽取真實assistant文字核對token機制，與SFT品質分母分開記錄。"""
    texts = [
        {"text": next(message["content"] for message in reversed(record["messages"]) if message["role"] == "assistant")}
        for record in records
    ]
    return text_examples(texts, max_length=max_length)


def _packing_updates(model, data, ctx, steps=40, deadline=None):
    examples = _mechanism_examples(data["train"], model.config.max_length)
    limit = max(2, model.config.max_length // 3)
    examples = [(x[:limit], y[:limit]) for x, y in examples]
    variants, models = {}, {}
    for name in ("padded", "packed"):
        current = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(current.parameters(), lr=0.003)
        sampler = random.Random(ctx.seed)
        total_tokens, history, latencies, completed = 0, [], [], 0
        current.train()
        _sync(ctx.device)
        started = time.perf_counter()
        for step in range(steps):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            selected = sampler.choices(examples, k=3)
            padded, packed = _packing_batch(selected, ctx)
            optimizer.zero_grad(set_to_none=True)
            _sync(ctx.device)
            began = time.perf_counter()
            if name == "padded":
                x, y, valid = padded
                result = current(x, valid=valid)
            else:
                x, y, valid, positions, segments = packed
                result = current(x, valid=valid, positions=positions, segments=segments)
            total, count = loss_sum(result["logits"], y)
            loss = total / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(current.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            _sync(ctx.device)
            latencies.append(time.perf_counter() - began)
            total_tokens += int(count)
            completed = step + 1
            if step == 0 or completed % 10 == 0:
                history.append({"step": completed, "loss": float(loss.detach()), "effective_tokens": int(count)})
        seconds = time.perf_counter() - started
        save_checkpoint(
            ctx.output / f"{name}.pt",
            current,
            optimizer,
            completed,
            {
                "seed": ctx.seed,
                "mode": "assistant-text-prefix",
                "max_document_tokens": limit,
                "effective_tokens": total_tokens,
                "packing": name == "packed",
            },
        )
        variants[name] = {
            "steps": completed,
            "optimizer_updates": completed,
            "seconds": seconds,
            "effective_tokens": total_tokens,
            "history": history,
            "checkpoint": f"{name}.pt",
            "warm_step_median_seconds": statistics.median(latencies[3:] or latencies) if latencies else None,
            "heldout": _heldout(current, data, "sft"),
        }
        models[name] = current
    variants["weight_max_error_after_updates"] = max(
        float((value - models["packed"].state_dict()[key]).abs().max())
        for key, value in models["padded"].state_dict().items()
    )
    variants["objective"] = "三段真實assistant文字prefix；同token loss與batch，以PAD／隔離packing各更新40次"
    return variants


def _checkpoint_probe(model, examples, ctx):
    ordinary, recompute = copy.deepcopy(model), copy.deepcopy(model)
    x, y, valid = (t.to(ctx.device) for t in pad_batch(examples[:4]))
    a, ga, _ = _gradients(ordinary, x, y, valid)
    b, gb, _ = _gradients(
        recompute, x, y, valid, forward=lambda *args, **kwargs: _checkpoint_forward(recompute, *args, **kwargs)
    )
    return {
        "logit_max_error": float((a - b).abs().max()),
        "gradient_max_error": _gradient_error(ga, gb),
        "implementation": "torch.utils.checkpoint(use_reentrant=False)，每個Transformer block重算",
    }


def _compile_worker(payload_path):
    """獨立程序允許真的Inductor編譯超時，避免總實驗卡在首次編譯。"""
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(2)
    from torch._inductor import config

    config.compile_threads = 1
    seed(payload["seed"])
    model = TinyLM(ModelConfig(**payload["config"])).to(payload["device"])
    model.load_state_dict(payload["model"])
    layer = model.blocks[0].ffn.eval()
    x = torch.randn(4, 16, model.config.width, device=payload["device"])
    compiled = torch.compile(layer, backend="inductor", fullgraph=True, dynamic=False)
    with torch.no_grad():
        reference = layer(x)
        _sync(payload["device"])
        began = time.perf_counter()
        actual = compiled(x)
        _sync(payload["device"])
        first = time.perf_counter() - began
        eager = _benchmark(lambda: layer(x), payload["device"])
        warm = _benchmark(lambda: compiled(x), payload["device"])
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    saving = eager["median_seconds"] - warm["median_seconds"]
    report = {
        "status": "completed",
        "backend": "inductor",
        "scope": "已訓練模型第一個Dense FFN",
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "fullgraph": True,
        "dynamic": False,
        "first_compiled_call_seconds": first,
        "output_max_error": float((reference - actual).abs().max()),
        "eager": eager,
        "compiled": warm,
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "graph_breaks": dict(counters["graph_break"]),
        "inductor_generated_kernel_count": int(metrics.generated_kernel_count),
        "estimated_calls_to_amortize_first_call": math.ceil(first / saving) if saving > 0 else None,
        "note": "組件級固定shape推論量測；不宣稱整個LM／MoE／動態shape已有收益。",
    }
    write_json(payload["report_path"], report)


def _compile_probe(model, ctx, timeout=150):
    if torch.device(ctx.device).type not in ("cuda", "cpu"):
        return {"status": "not_run", "backend": "inductor", "reason": "此固定shape實驗只驗證CPU/CUDA"}
    if timeout < 5:
        return {"status": "not_run", "backend": "inductor", "reason": "總實驗預算已不足"}
    payload_path, report_path = ctx.output / "compile-input.pt", ctx.output / "compile-result.json"
    report_path.unlink(missing_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "device": ctx.device,
            "seed": ctx.seed,
            "report_path": str(report_path.resolve()),
        },
        payload_path,
    )
    began = time.perf_counter()
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.course_experiments.architecture",
                "--compile-worker",
                str(payload_path.resolve()),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if process.returncode != 0:
            return {
                "status": "failed",
                "backend": "inductor",
                "returncode": process.returncode,
                "stderr_tail": process.stderr[-4000:],
                "seconds": time.perf_counter() - began,
            }
        if not report_path.is_file():
            return {"status": "failed", "backend": "inductor", "reason": "程序未寫當次報告"}
        result = json.loads(report_path.read_text(encoding="utf-8"))
        result["worker_wall_seconds"] = time.perf_counter() - began
        return result
    except subprocess.TimeoutExpired:
        return {
            "status": "timed_out",
            "backend": "inductor",
            "timeout_seconds": timeout,
            "reason": "首次真實Inductor編譯超出組件預算；未改用eager冒充編譯",
        }


def run_efficiency(ctx):
    started = time.perf_counter()
    data = _sft_dataset(ctx)
    initial = load_lm(ctx.dependency("sft"), ctx.device)
    examples = text_examples(data["train"], mode="sft", max_length=initial.config.max_length)
    heads = 4 if initial.config.width % 4 == 0 else initial.config.heads
    models, trained = {}, {}
    for name, kv in (("mha", heads), ("gqa", 1)):
        model, shared = _clone_config(initial, ctx, heads=heads, kv_heads=kv, experts=0, backend="manual")
        training = _train(model, data["train"], ctx, name=name, steps=100, mode="sft", deadline=started + 345)
        trained[name] = model
        models[name] = {
            "model": _description(model),
            "copied_initial_tables": shared,
            "training": training,
            "heldout": _heldout(model, data, "sft"),
            "cache": _cache_probe(model, ctx),
        }
    base = trained["mha"]
    sdpa = _sdpa_probe(base, examples, ctx)
    packing = _packing_probe(base, data["train"], ctx)
    accumulation = _accumulation_probe(base, _mechanism_examples(data["train"], base.config.max_length), ctx)
    recomputation = _checkpoint_probe(base, examples, ctx)
    update_variants = {}
    final_models = {}
    for name in ("ordinary", "accumulated", "activation_checkpoint", "sdpa"):
        model = copy.deepcopy(base)
        forward = None
        if name == "activation_checkpoint":

            def forward(ids, valid=None, model=model):
                return _checkpoint_forward(model, ids, valid=valid)

        if name == "sdpa":
            for block in model.blocks:
                block.attention.backend = "sdpa"
            model.config.backend = "sdpa"
        training = _train(
            model,
            data["train"],
            ctx,
            name=name,
            steps=40,
            mode="sft",
            batch_size=8,
            forward=forward,
            micro_sizes=[3, 5] if name == "accumulated" else None,
            deadline=started + 345,
        )
        update_variants[name] = {"training": training, "heldout": _heldout(model, data, "sft")}
        final_models[name] = model
    reference = final_models["ordinary"].state_dict()
    for name in ("accumulated", "activation_checkpoint", "sdpa"):
        update_variants[name]["weight_max_error_from_ordinary_after_updates"] = max(
            float((reference[key] - final_models[name].state_dict()[key]).abs().max()) for key in reference
        )
    (ctx.output / "model.pt").write_bytes((ctx.output / "ordinary.pt").read_bytes())
    packing["actual_updates"] = _packing_updates(base, data, ctx, deadline=started + 345)
    compilation = _compile_probe(base, ctx, timeout=min(150, max(0, int(started + 535 - time.perf_counter()))))
    return {
        "seed": ctx.seed,
        "source": "sft/model.pt",
        "runtime": _runtime(ctx),
        "dataset": _data_report(data),
        "models": models,
        "manual_vs_sdpa": sdpa,
        "packing": packing,
        "accumulation": accumulation,
        "activation_checkpoint": recomputation,
        "update_variants": update_variants,
        "compile": compilation,
        "seconds": time.perf_counter() - started,
        "gqa_note": "GQA變形的K/V投影重新固定seed初始化，再以同資料短續訓。"
        "不是直接把訓練好的MHA權重重排就宣稱等價；兩模型各自核對cache一致。",
        "timing_note": "暖機後重複同步量測中位數；所有時間均為當次硬體，不以SDPA名稱推定Flash後端。",
    }


def run_precision(ctx):
    started = time.perf_counter()
    data = _sft_dataset(ctx)
    initial = load_lm(ctx.dependency("sft"), ctx.device)
    results = {}
    device_type = torch.device(ctx.device).type
    for name, dtype in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        if dtype == torch.bfloat16 and device_type == "cuda" and not torch.cuda.is_bf16_supported():
            results[name] = {"status": "not_run", "reason": "此CUDA裝置未支援BF16"}
            continue
        if dtype == torch.float16 and device_type != "cuda":
            results[name] = {"status": "not_run", "reason": "本FP16訓練比較需要CUDA autocast與GradScaler"}
            continue
        if dtype == torch.bfloat16 and device_type not in ("cpu", "cuda"):
            results[name] = {"status": "not_run", "reason": "此BF16實驗只驗證CPU/CUDA"}
            continue
        model = copy.deepcopy(initial)
        try:
            training = _train(
                model, data["train"], ctx, name=name, steps=200, mode="sft", dtype=dtype, deadline=started + 520
            )
            # 部署權重仍為FP32；另在相同autocast上下文做實際推論品質測試。
            with _amp(ctx.device, dtype):
                heldout = _heldout(model, data, "sft")
            example = text_examples(data["validation"], "sft", model.config.max_length)[:8]
            x, _, valid = (t.to(ctx.device) for t in pad_batch(example))
            with torch.no_grad(), _amp(ctx.device, dtype):
                logits = model(x, valid=valid)["logits"]
                timings = _benchmark(lambda: model(x, valid=valid), ctx.device)
            results[name] = {
                "status": "completed",
                "model": _description(model),
                "training": training,
                "heldout": heldout,
                "inference": timings,
                "observed_logits_dtype": str(logits.dtype),
                "weights_dtype": str(next(model.parameters()).dtype),
                "logits_finite": bool(torch.isfinite(logits).all()),
            }
        except (RuntimeError, FloatingPointError) as error:
            results[name] = {"status": "failed", "reason": str(error), "amp_dtype": str(dtype)}
    if (ctx.output / "fp32.pt").is_file():
        (ctx.output / "model.pt").write_bytes((ctx.output / "fp32.pt").read_bytes())
    return {
        "seed": ctx.seed,
        "source": "sft/model.pt",
        "runtime": _runtime(ctx),
        "dataset": _data_report(data),
        "variants": results,
        "seconds": time.perf_counter() - started,
        "comparison": "同FP32起始權重、資料、batch及更新嘗試數；AMP依算子選dtype，"
        "權重與Adam狀態保留FP32，FP16在CUDA使用GradScaler。",
        "limitations": "FP16溢位時scaler跳過更新，成功更新數另列；單一seed短續訓不代表普遍品質保證。",
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--compile-worker":
        _compile_worker(sys.argv[2])
    else:
        raise SystemExit("請由課程實驗runner呼叫；此模組CLI只供隔離的Inductor worker")
