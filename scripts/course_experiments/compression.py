"""第 17–18 章：真正更新的量化／蒸餾與可重載的部署權重。

所有比較保留同一個 heldout 分母與 greedy 回答；packed Linear 的 forward
會先重建 FP32 權重，檔案變小不代表低位元 kernel、RAM 或速度改善。
"""

import copy
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder
from tiny_perceptron.quantization import QuantizedLinear, quantize_symmetric, replace_linear_layers
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything

from .common import extract_asset, text_examples


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _teacher(ctx, identifier, filename="model.pt"):
    path = ctx.dependency(identifier, filename)
    model, payload = load_checkpoint(path, ctx.device)
    provenance = {
        "experiment": identifier,
        "checkpoint": str(path),
        "sha256": _sha(path),
        "format_version": payload["format_version"],
        "steps": payload.get("step", 0),
        "config": payload["config"],
        "metadata": payload.get("metadata", {}),
    }
    return model, provenance


def _chat(record):
    if "messages" in record:
        return record["messages"]
    return [
        {"role": "user", "content": record["question"]},
        {"role": "assistant", "content": record["answer"]},
    ]


def _example(record, maximum):
    if "_hard_ids" in record:
        prefix = _prompt(record, maximum)
        tail = torch.tensor(record["_hard_ids"] + [ByteTokenizer().eos_id], dtype=torch.long)
        full = torch.cat((prefix, tail))
        labels = torch.cat((torch.full_like(prefix, IGNORE), tail))
        if len(full) - 1 > maximum:
            raise ValueError("教師 hard target 超過學生 max_length")
        return full[:-1], labels[1:]
    x, y = render_chat(_chat(record))
    if len(x) > maximum:
        raise ValueError("壓縮實驗不能默默截斷答案；請提供符合 max_length 的教學紀錄")
    return x, y


def _examples(records, maximum):
    if "text" in records[0]:
        return text_examples(records, mode="text", max_length=maximum)
    return [_example(record, maximum) for record in records]


def _steps(ctx, value):
    return max(1, round(value * getattr(ctx, "step_scale", 1.0)))


def _dataset(ctx, identifier):
    path = ctx.dependency(identifier, "dataset.json")
    parts = json.loads(path.read_text(encoding="utf-8"))
    families = {}
    for split in ("train", "validation", "test"):
        if not parts.get(split):
            raise ValueError(f"{identifier}/{split} 沒有紀錄")
        families[split] = {str(row["family"]) for row in parts[split]}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if families[left] & families[right]:
            raise ValueError(f"{identifier}: {left}/{right} family 洩漏")
    return parts, {
        "source": str(path),
        "sha256": _sha(path),
        "split_unit": "teacher's original families, unchanged",
        "counts": {split: len(parts[split]) for split in families},
        "families": {split: len(families[split]) for split in families},
        "family_intersections": 0,
    }


def _prompt(record, maximum):
    tok = ByteTokenizer()
    messages = _chat(record)
    ids = [tok.bos_id]
    roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
    # 最後一個 assistant 是待生成答案，其餘前文皆保留。
    for message in messages[:-1]:
        ids += [roles[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
    ids.append(tok.assistant_id)
    if len(ids) >= maximum:
        raise ValueError("prompt 沒有留下答案空間")
    return torch.tensor(ids, dtype=torch.long)


def _answer(record):
    return _chat(record)[-1]["content"]


def _storage(model, path):
    buffers = {name: value.numel() * value.element_size() for name, value in model.named_buffers()}
    parameters = {name: value.numel() * value.element_size() for name, value in model.named_parameters()}
    packed = [name for name, layer in model.named_modules() if isinstance(layer, QuantizedLinear)]
    quantized_elements = sum(
        math.prod(layer.shape) + (0 if layer.bias is None else layer.bias.numel())
        for layer in model.modules()
        if isinstance(layer, QuantizedLinear)
    )
    float_elements = sum(p.numel() for p in model.parameters())
    tensor_bytes = sum(parameters.values()) + sum(buffers.values())
    return {
        "checkpoint": str(path),
        "file_bytes": Path(path).stat().st_size,
        "parameter_count": float_elements + quantized_elements,
        "float_parameter_count": float_elements,
        "parameter_tensor_bytes": sum(parameters.values()),
        "buffer_tensor_bytes": sum(buffers.values()),
        "tensor_bytes": tensor_bytes,
        "file_overhead_bytes": Path(path).stat().st_size - tensor_bytes,
        "buffers": buffers,
        "packed_linear_modules": packed,
        "retained_float_modules": [
            name
            for name, layer in model.named_modules()
            if not isinstance(layer, QuantizedLinear) and any(True for _ in layer.parameters(recurse=False))
        ],
        "forward": "FP32 dequantized reference" if packed else "FP32",
        "optimizer_in_deployment_file": False,
        "file_note": "native FP32 and packed formats have different metadata/RNG overhead; tensor bytes isolate stored model tensors",
    }


def _save_float(ctx, model, name, steps=0, metadata=None):
    path = Path(ctx.output) / f"{name}.pt"
    save_checkpoint(path, model, step=steps, metadata=metadata)
    return path


def _packed(ctx, source, name, bits, provenance=None):
    model = replace_linear_layers(copy.deepcopy(source).eval(), bits)
    was_tied = model.config.tied
    model.config = copy.deepcopy(model.config)
    model.config.tied = False
    path = Path(ctx.output) / f"{name}.pt"
    torch.save(
        {
            "format_version": "quantized-v1",
            "config": asdict(model.config),
            "bits": bits,
            "model": model.state_dict(),
            "optimizer": None,
            "tokenizer": ByteTokenizer().state(),
            "metadata": {
                "source": provenance or {},
                "input_output_sharing_removed": was_tied,
                "quantization": "symmetric signed, per output channel, maxabs / (2**(bits-1)-1)",
                "scope": "Linear weight-only; embedding/norm/bias remain FP32; reference dequantized forward",
            },
        },
        path,
    )
    reloaded, _ = load_checkpoint(path, ctx.device)
    return reloaded.eval(), path


@torch.no_grad()
def _evaluate(model, records, device, tokens=24):
    model.eval()
    tok = ByteTokenizer()
    total, count, correct, eos_count, byte_count = 0.0, 0, 0, 0, 0
    answers = []
    start = time.perf_counter()
    examples = _examples(records, model.config.max_length)
    for offset in range(0, len(examples), 16):
        x, y, valid = (t.to(device) for t in pad_batch(examples[offset : offset + 16]))
        nll, denominator = loss_sum(model(x, valid=valid)["logits"], y)
        total += float(nll)
        count += int(denominator)
    for record in records:
        is_text = "text" in record
        if is_text:
            raw_text = tok.encode(record["text"])
            prefix = raw_text[:24]
            prompt = torch.tensor([tok.bos_id] + prefix, device=device)
            expected_ids = raw_text[len(prefix) : len(prefix) + tokens]
            expected = tok.decode(expected_ids)
            byte_count += len(record["text"].encode())
        else:
            prompt = _prompt(record, model.config.max_length).to(device)
            expected = _answer(record)
            expected_ids = tok.encode(expected)
            byte_count += len(expected.encode())
        ids = generate(model, prompt[None], max_new_tokens=tokens, temperature=0, use_cache=False)
        new = ids[0, len(prompt) :].tolist()
        raw = new[: new.index(tok.eos_id)] if tok.eos_id in new else new
        response = tok.decode(raw)
        hit = raw == expected_ids
        correct += hit
        eos_count += tok.eos_id in new
        answers.append(
            {
                "family": record.get("family"),
                "question": tok.decode(prefix) if is_text else _chat(record)[-2]["content"],
                "expected": expected,
                "generated": response,
                "generated_ids": new,
                "exact": hit,
                "ended_with_eos": tok.eos_id in new,
            }
        )
    _sync(device)
    return {
        "nll_sum": total,
        "supervised_tokens": count,
        "nll_sequence_chunks": len(examples),
        "answer_nll": total / count,
        "answer_bytes": byte_count,
        "answer_bpb": total / max(1, byte_count) / math.log(2),
        "bpb_includes_eos_nll": True,
        "correct": correct,
        "examples": len(records),
        "exact_match": correct / len(records),
        "eos_count": eos_count,
        "generation": "greedy, full recompute, no KV cache",
        "max_new_tokens": tokens,
        "exact_match_definition": "fixed reference continuation"
        if "text" in records[0]
        else "raw answer token identity",
        "seconds": time.perf_counter() - start,
        "generated_samples": answers,
    }


@torch.no_grad()
def _timing(model, record, device):
    """先暖機，再各量 prompt 與固定 8 次 decode；不以 EOS 長短掩蓋成本。"""
    if "text" in record:
        tok = ByteTokenizer()
        prompt = torch.tensor([[tok.bos_id] + tok.encode(record["text"])[:24]], device=device)
    else:
        prompt = _prompt(record, model.config.max_length)[None].to(device)
    if prompt.shape[1] + 8 > model.config.max_length:
        return {"measured": False, "reason": "prompt 沒留下 8 個 decode 位置"}
    model.eval()
    model(prompt)
    _sync(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    else:
        base = None
    prefill, decode = [], []
    for _ in range(3):
        _sync(device)
        start = time.perf_counter()
        output = model(prompt)
        _sync(device)
        prefill.append(time.perf_counter() - start)
        ids = prompt
        start = time.perf_counter()
        for _ in range(8):
            token = output["logits"][:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, token), -1)
            output = model(ids)
        _sync(device)
        decode.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if base is not None else None
    return {
        "measured": True,
        "repetitions": 3,
        "prompt_tokens": prompt.shape[1],
        "decode_tokens": 8,
        "prefill_seconds": sum(prefill) / 3,
        "decode_seconds": sum(decode) / 3,
        "decode_seconds_per_token": sum(decode) / 24,
        "cuda_allocated_before": base,
        "cuda_peak_allocated": peak,
        "cuda_additional_peak_bytes": None if peak is None else peak - base,
        "note": "CUDA allocator peak includes co-resident models; additional peak is this probe, not a low-bit RAM claim",
    }


@torch.no_grad()
def _cache_text(teacher, records, device):
    start = time.perf_counter()
    teacher.eval().requires_grad_(False)
    cache = []
    examples = _examples(records, teacher.config.max_length)
    for offset in range(0, len(examples), 16):
        x, y, valid = (t.to(device) for t in pad_batch(examples[offset : offset + 16]))
        logits = teacher(x, valid=valid)["logits"]
        cache.extend(logits[i, y[i] != IGNORE].detach().cpu() for i in range(len(x)))
    _sync(device)
    return cache, time.perf_counter() - start


def _fit_text(ctx, model, records, name, steps=200, teacher_cache=None, alpha=0.5, temperature=2.0):
    """相同 seed／init／batch plan；KL 只比較同一 gold answer 前文的有效位置。"""
    seed_everything(ctx.seed)
    model.train().requires_grad_(True)
    examples = _examples(records, model.config.max_length)
    before = _parameter_hash(model)
    rng = random.Random(ctx.seed)
    plan = [[rng.randrange(len(examples)) for _ in range(16)] for _ in range(steps)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    trace, initial_gradient, effective = [], None, 0
    _sync(ctx.device)
    start = time.perf_counter()
    for step, indices in enumerate(plan):
        x, y, valid = pad_batch([examples[i] for i in indices])
        x, y, valid = x.to(ctx.device), y.to(ctx.device), valid.to(ctx.device)
        logits = model(x, valid=valid)["logits"]
        ce = masked_loss(logits, y)
        kl = ce.new_zeros(())
        if teacher_cache is not None:
            selected = logits[y != IGNORE][None]
            targets = torch.cat([teacher_cache[i] for i in indices]).to(ctx.device)[None]
            labels = y[y != IGNORE][None]
            kl = distillation_kl(selected, targets, labels, temperature)
        loss = (1 - alpha) * ce + alpha * kl if teacher_cache is not None else ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: nonfinite loss at {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            initial_gradient = float(model.output.weight.grad.norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        effective += int((y != IGNORE).sum())
        if step in (0, steps - 1) or (step + 1) % 50 == 0:
            trace.append(
                {"step": step + 1, "loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach())}
            )
    _sync(ctx.device)
    elapsed = time.perf_counter() - start
    after = _parameter_hash(model)
    report = {
        "steps": steps,
        "optimizer_updates": steps,
        "batch_size": 16,
        "learning_rate": 0.003,
        "seconds": elapsed,
        "initialization_sha256": before,
        "final_sha256": after,
        "weights_changed": before != after,
        "first_output_weight_gradient_norm": initial_gradient,
        "batch_plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
        "training_examples": len(records),
        "training_sequence_chunks": len(examples),
        "effective_supervised_tokens": effective,
        "training_checkpoint": f"{name}-training.pt",
        "objective": "(1-alpha)*CE + alpha*KL(teacher||student)*T²" if teacher_cache is not None else "CE",
        "alpha": alpha if teacher_cache is not None else 0,
        "temperature": temperature,
        "temperature_squared_applied_once": teacher_cache is not None,
        "loss_trace": trace,
    }
    if any(isinstance(layer, _QATLinear) for layer in model.modules()):
        report["training_forward"] = (
            "restore _QATLinear wrappers before resuming; native state stores float master weights"
        )
    save_checkpoint(Path(ctx.output) / f"{name}-training.pt", model, optimizer, steps, report)
    path = _save_float(ctx, model, name, steps, report)
    return model.eval(), path, report


class _QATLinear(nn.Linear):
    """與 QuantizedLinear 同一個 per-channel 規則，僅 backward 採 STE。"""

    def __init__(self, source, bits=4):
        super().__init__(source.in_features, source.out_features, source.bias is not None)
        self.weight = source.weight
        self.bias = source.bias
        self.bits = bits

    def forward(self, x):
        integers, scale = quantize_symmetric(self.weight.detach(), self.bits, per_channel=True)
        restored = integers * scale
        simulated = self.weight + (restored - self.weight).detach()
        return F.linear(x, simulated, self.bias)


def _qat_layers(model, remove=False):
    for name, child in list(model.named_children()):
        if remove and isinstance(child, _QATLinear):
            plain = nn.Linear(child.in_features, child.out_features, child.bias is not None)
            plain.weight, plain.bias = child.weight, child.bias
            setattr(model, name, plain)
        elif not remove and isinstance(child, nn.Linear):
            setattr(model, name, _QATLinear(child))
        else:
            _qat_layers(child, remove)
    return model


def run_quantization(ctx):
    """FP32 教師繼續真實訓練後，同一權重轉換為 packed4／packed8。"""
    Path(ctx.output).mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    source, provenance = _teacher(ctx, "sft")
    parts, data = _dataset(ctx, "sft")
    model, fp_path, training = _fit_text(ctx, source, parts["train"], "fp32", _steps(ctx, 120))
    variants = {"fp32": (model, fp_path)}
    variants["packed4"] = _packed(ctx, model, "model", 4, provenance)
    variants["packed8"] = _packed(ctx, model, "packed8", 8, provenance)
    reports = {}
    for name, (candidate, path) in variants.items():
        reports[name] = {
            "storage": _storage(candidate, path),
            "validation": _evaluate(candidate, parts["validation"], ctx.device),
            "test": _evaluate(candidate, parts["test"], ctx.device),
            "timing": _timing(candidate, parts["test"][0], ctx.device),
        }
    for name in ("packed4", "packed8"):
        reports[name]["file_bytes_ratio_to_fp32"] = (
            reports[name]["storage"]["file_bytes"] / reports["fp32"]["storage"]["file_bytes"]
        )
        reports[name]["tensor_bytes_ratio_to_fp32"] = (
            reports[name]["storage"]["tensor_bytes"] / reports["fp32"]["storage"]["tensor_bytes"]
        )
        reports[name]["test_nll_delta"] = reports[name]["test"]["answer_nll"] - reports["fp32"]["test"]["answer_nll"]
        reports[name]["test_exact_match_delta"] = (
            reports[name]["test"]["exact_match"] - reports["fp32"]["test"]["exact_match"]
        )
    return {
        "experiment": "quantization",
        "sections": ["17.1", "17.5", "17.7", "17.8", "17.9", "17.10", "17.15"],
        "teacher_provenance": provenance,
        "data": data,
        "training": training,
        "runs": reports,
        "same_fp32_source": True,
        "reloaded_packed_checkpoints_before_evaluation": True,
        "checkpoint": "model.pt",
        "seconds": time.perf_counter() - began,
        "limitations": [
            "weight-only per-channel Linear quantization; no activation calibration",
            "packed storage is real; forward dequantizes to FP32 and uses ordinary Linear",
            "reported timings and allocator probes do not establish low-bit kernel speed or RAM savings",
            "synthetic heldout attribute families; no broad language quality claim",
        ],
    }


def run_qat(ctx):
    """同一 init／batch plan 的 FP 微調與 STE QAT，再各自真的打包。"""
    Path(ctx.output).mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    source, provenance = _teacher(ctx, "sft")
    if source.config.tied:
        raise ValueError("此 matched QAT 教學要求 untied 教師，避免額外改變共享語意")
    parts, data = _dataset(ctx, "sft")
    source.eval()
    start_hash = _parameter_hash(source)
    source_path = _save_float(ctx, source, "initial_fp32", provenance["steps"], provenance)
    plain, plain_path, plain_fit = _fit_text(
        ctx, copy.deepcopy(source), parts["train"], "fp_finetuned", _steps(ctx, 350)
    )
    fake = _qat_layers(copy.deepcopy(source))
    fake, _, qat_fit = _fit_text(ctx, fake, parts["train"], "qat_fake", _steps(ctx, 350))
    # fake checkpoint 中的數值仍是浮點；部署檔案必須另做真正 packing。
    qat_float = _qat_layers(copy.deepcopy(fake), remove=True)
    qat_float_path = _save_float(ctx, qat_float, "qat_float", qat_fit["steps"], qat_fit)
    initial_ptq = _packed(ctx, source, "initial_ptq4", 4, provenance)
    matched_ptq = _packed(ctx, plain, "matched_ptq4", 4, {"training": plain_fit})
    deployment = _packed(ctx, qat_float, "model", 4, {"training": qat_fit})
    x, _ = _example(parts["test"][0], source.config.max_length)
    with torch.no_grad():
        a = fake(x[None].to(ctx.device))["logits"]
        b = deployment[0](x[None].to(ctx.device))["logits"]
        matching_error = float((a - b).abs().max())
    if matching_error > 1e-4:
        raise ValueError(f"fake 與 deployed per-channel 規則不一致：{matching_error}")
    if plain_fit["initialization_sha256"] != qat_fit["initialization_sha256"]:
        raise ValueError("FP 與 QAT 初始化不同")
    if plain_fit["batch_plan_sha256"] != qat_fit["batch_plan_sha256"]:
        raise ValueError("FP 與 QAT batch plan 不同")
    variants = {
        "initial_fp32": (source, source_path),
        "initial_ptq4": initial_ptq,
        "fp_finetuned": (plain, plain_path),
        "matched_ptq4": matched_ptq,
        "qat_float": (qat_float, qat_float_path),
        "qat_packed4": deployment,
    }
    reports = {
        name: {
            "storage": _storage(model, path),
            "validation": _evaluate(model, parts["validation"], ctx.device),
            "test": _evaluate(model, parts["test"], ctx.device),
        }
        for name, (model, path) in variants.items()
    }
    return {
        "experiment": "qat",
        "sections": ["17.14", "17.15"],
        "teacher_provenance": provenance,
        "data": data,
        "initialization_sha256": start_hash,
        "training": {"fp_finetuning": plain_fit, "qat": qat_fit},
        "fake_quantization": "weight-only symmetric per output channel; signed -7..7; STE identity gradient",
        "activation_quantization": False,
        "same_initialization_and_batch_plan": True,
        "fake_deployed_max_logit_difference": matching_error,
        "runs": reports,
        "qat_vs_matched_ptq_test_nll_delta": reports["qat_packed4"]["test"]["answer_nll"]
        - reports["matched_ptq4"]["test"]["answer_nll"],
        "qat_vs_matched_ptq_test_em_delta": reports["qat_packed4"]["test"]["exact_match"]
        - reports["matched_ptq4"]["test"]["exact_match"],
        "checkpoint": "model.pt",
        "seconds": time.perf_counter() - began,
        "limitations": [
            "STE is an approximate gradient",
            "QAT improvement is measured, never assumed",
            "packed forward dequantizes to FP32",
        ],
    }


@torch.no_grad()
def _hard_targets(ctx, teacher, records, name, tokens):
    teacher.eval().requires_grad_(False)
    tok = ByteTokenizer()
    prepared, audit = [], []
    start = time.perf_counter()
    for record in records:
        if "text" in record:
            raise ValueError("hard answer 蒸餾本節使用 SFT 問答，不把自然文件改成教師真值")
        prefix = _prompt(record, teacher.config.max_length).to(ctx.device)
        ids = generate(teacher, prefix[None], max_new_tokens=tokens)[0, len(prefix) :].tolist()
        raw = ids[: ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        answer = tok.decode(raw)
        row = copy.deepcopy(record)
        row["messages"] = copy.deepcopy(_chat(record))
        row["messages"][-1]["content"] = answer
        row["_hard_ids"] = raw
        row.pop("answer", None)
        prepared.append(row)
        audit.append(
            {
                "family": record["family"],
                "question": _chat(record)[-2]["content"],
                "gold_answer": _answer(record),
                "teacher_answer": answer,
                "teacher_ids": ids,
                "teacher_correct": raw == tok.encode(_answer(record)),
                "eos": tok.eos_id in ids,
            }
        )
    _sync(ctx.device)
    elapsed = time.perf_counter() - start
    path = Path(ctx.output) / f"{name}-hard-targets.json"
    _json(path, {"records": prepared, "audit": audit})
    return prepared, {
        "seconds": elapsed,
        "records": len(records),
        "correct": sum(row["teacher_correct"] for row in audit),
        "wrong": sum(not row["teacher_correct"] for row in audit),
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "sha256": _sha(path),
        "audit": audit,
    }


def _style_scores(evaluation):
    correct, json_valid, json_total = 0, 0, 0
    for sample in evaluation["generated_samples"]:
        expected_numbers = re.findall(r"\d+", sample["expected"])
        generated_numbers = re.findall(r"\d+", sample["generated"])
        correct += bool(expected_numbers and generated_numbers and expected_numbers[0] == generated_numbers[0])
        if "style=json" in sample["question"]:
            json_total += 1
            try:
                value = json.loads(sample["generated"])
                json_valid += isinstance(value, dict) and set(value) == {"answer"}
            except (ValueError, TypeError):
                pass
    return {
        "content_correct": correct,
        "examples": evaluation["examples"],
        "content_accuracy": correct / evaluation["examples"],
        "json_parseable": json_valid,
        "json_examples": json_total,
    }


def _challenge(ctx, maximum):
    root = extract_asset(ctx, "gsm8k")
    matches = list(Path(root).rglob("gsm8k-train-first200.jsonl"))
    if len(matches) != 1:
        raise ValueError("GSM8K 教師錯誤審核需要已校驗的 200 題資料包")
    original = [json.loads(line) for line in matches[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for record in original[:6]:
        # 本 tiny 模型容納不了整道長題；明示 excerpt，絕不叫完整 GSM8K benchmark。
        original_question = record["question"]
        question = original_question.encode()[: maximum - 32].decode("utf-8", errors="ignore")
        final = record["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(
            {
                "family": hashlib.sha256(original_question.encode()).hexdigest(),
                "question": question,
                "answer": final,
                "original_question": original_question,
                "question_truncated": question != original_question,
                "source": "GSM8K unseen prompt excerpt; teacher did not train on these questions",
            }
        )
    return rows


def _distill_case(ctx, identifier, widths, steps, hard=True):
    teacher, provenance = _teacher(ctx, identifier)
    if not isinstance(teacher, TinyLM) or teacher.config.width != 64:
        raise ValueError(f"{identifier}: 蒸餾需要訓練過的 width64 TinyLM 教師")
    parts, data = _dataset(ctx, identifier)
    is_text = "text" in parts["train"][0]
    tokens = 96 if identifier == "style" else 24
    frozen_hash = _parameter_hash(teacher)
    teacher.eval().requires_grad_(False)
    teacher_path = _save_float(ctx, teacher, f"{identifier}-teacher", provenance["steps"], provenance)
    evaluation = {split: _evaluate(teacher, parts[split], ctx.device, tokens) for split in ("validation", "test")}
    # MoE 的自然故事 teacher cache 有明確上限；heldout 全部文件仍然完整計分。
    train_records = parts["train"][:32] if is_text else parts["train"]
    data["student_training_records"] = len(train_records)
    data["student_training_selection"] = (
        "first 32 original training documents" if is_text else "all original training records"
    )
    cache, cache_seconds = _cache_text(teacher, train_records, ctx.device)
    cache_path = Path(ctx.output) / f"{identifier}-teacher-logits.pt"
    torch.save(
        {
            "logits": cache,
            "teacher_sha256": provenance["sha256"],
            "alignment": "same teacher-forced gold prefix, answer labels != -100",
            "temperature": 2.0,
        },
        cache_path,
    )
    generated, hard_report = None, None
    if hard:
        generated, hard_report = _hard_targets(ctx, teacher, train_records, identifier, tokens)
    challenge = _challenge(ctx, teacher.config.max_length) if identifier == "sft" else None
    challenge_teacher = _evaluate(teacher, challenge, ctx.device, tokens) if challenge else None
    if challenge:
        _json(Path(ctx.output) / "unseen-gsm8k-excerpts.json", challenge)
    runs = {}
    final_model = None
    for width in widths:
        seed_everything(ctx.seed)
        initial = TinyLM(ModelConfig(width=width, layers=1, heads=2, max_length=teacher.config.max_length)).to(
            ctx.device
        )
        methods = ("ce", "teacher_hard", "ce_kl") if hard else ("ce", "ce_kl")
        for method in methods:
            name = f"{identifier}-w{width}-{method}"
            model, path, training = _fit_text(
                ctx,
                copy.deepcopy(initial),
                generated if method == "teacher_hard" else train_records,
                name,
                _steps(ctx, steps),
                teacher_cache=cache if method == "ce_kl" else None,
            )
            result = {
                "training": training,
                "storage": _storage(model, path),
                "validation": _evaluate(model, parts["validation"], ctx.device, tokens),
                "test": _evaluate(model, parts["test"], ctx.device, tokens),
            }
            teacher_samples = evaluation["test"]["generated_samples"]
            result["teacher_agreement"] = sum(
                left["generated_ids"] == right["generated_ids"]
                for left, right in zip(result["test"]["generated_samples"], teacher_samples, strict=True)
            ) / len(teacher_samples)
            if identifier == "style":
                result["style"] = _style_scores(result["test"])
            if challenge:
                result["unseen_gsm8k_excerpt_challenge"] = _evaluate(model, challenge, ctx.device, tokens)
            runs[f"w{width}_{method}"] = result
            if method == "ce_kl":
                packed, packed_path = _packed(
                    ctx, model, f"{name}-packed4", 4, {"teacher": provenance, "training": training}
                )
                runs[f"w{width}_{method}_packed4"] = {
                    "storage": _storage(packed, packed_path),
                    "validation": _evaluate(packed, parts["validation"], ctx.device, tokens),
                    "test": _evaluate(packed, parts["test"], ctx.device, tokens),
                }
                if identifier == "sft" and width == 32:
                    final_model = (model, training)
    if _parameter_hash(teacher) != frozen_hash:
        raise ValueError("蒸餾過程修改了教師")
    return {
        "teacher_provenance": provenance,
        "teacher_storage": _storage(teacher, teacher_path),
        "teacher_validation": evaluation["validation"],
        "teacher_test": evaluation["test"],
        "teacher_frozen_and_unchanged": True,
        "data": data,
        "student_layers": 1,
        "teacher_cache": {"seconds": cache_seconds, "file_bytes": cache_path.stat().st_size, "file": str(cache_path)},
        "hard_target_generation": hard_report,
        "unseen_gsm8k_excerpt_teacher": challenge_teacher,
        "runs": runs,
    }, final_model


def run_distillation(ctx):
    Path(ctx.output).mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    sft, final = _distill_case(ctx, "sft", (16, 32), 400)
    style, _ = _distill_case(ctx, "style", (32,), 300)
    moe, _ = _distill_case(ctx, "moe", (32,), 300, hard=False)
    model, training = final
    _save_float(ctx, model, "model", training["steps"], {"teacher": sft["teacher_provenance"], "training": training})
    return {
        "experiment": "distillation",
        "sections": [
            "18.1",
            "18.2",
            "18.3",
            "18.4",
            "18.5",
            "18.6",
            "18.7",
            "18.8",
            "18.9",
            "18.10",
            "18.11",
            "18.12",
            "18.14",
        ],
        "tasks": {"attributes": sft, "style_transfer": style, "moe_to_dense": moe},
        "teacher_signal": "hard targets are actual greedy teacher outputs; KL is teacher||student on aligned gold prefixes",
        "matched_students": "same width/layers/init/steps/batch plan; only target/objective differs; hard answers may change effective token counts",
        "checkpoint": "model.pt",
        "seconds": time.perf_counter() - began,
        "limitations": [
            "GSM8K excerpts are a separate out-of-domain error audit, not a GSM8K benchmark",
            "teacher agreement is reported separately from gold answer accuracy",
            "teacher errors and regressions remain in saved reports",
            "packed inference reconstructs FP32 weights",
        ],
    }


@torch.no_grad()
def _cache_modal(ctx, teacher, records):
    from .modalities import modal_inputs

    teacher.eval().requires_grad_(False)
    cache = []
    start = time.perf_counter()
    for record in records:
        ids, labels, image, waveform, _ = modal_inputs(record, ctx)
        output = teacher(ids, labels, image=image, waveform=waveform)
        valid = output["labels"][0] != IGNORE
        cache.append(
            {
                "logits": output["logits"][0, valid].detach().cpu(),
                "labels": output["labels"][0, valid].cpu(),
                "prediction_rows": valid.nonzero().flatten().cpu(),
            }
        )
    _sync(ctx.device)
    return cache, time.perf_counter() - start


def _fit_modal(ctx, student, records, cache, name, steps=300, use_kl=False):
    from .modalities import modal_inputs

    seed_everything(ctx.seed)
    before = _parameter_hash(student)
    student.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.003, weight_decay=0.01)
    rng = random.Random(ctx.seed)
    plan = [[rng.randrange(len(records)) for _ in range(4)] for _ in range(steps)]
    trace, aligned, effective = [], None, 0
    _sync(ctx.device)
    start = time.perf_counter()
    for step, indices in enumerate(plan):
        ce_sums, kl_sums, counts = [], [], []
        for index in indices:
            ids, labels, image, waveform, _ = modal_inputs(records[index], ctx)
            output = student(ids, labels, image=image, waveform=waveform)
            selected = output["labels"][0] != IGNORE
            student_labels = output["labels"][0, selected]
            if not torch.equal(student_labels.cpu(), cache[index]["labels"]):
                raise ValueError("不同模態前文長度必須對到完全相同的待預測答案 tokens")
            summed, count = loss_sum(output["logits"], output["labels"])
            ce_sums.append(summed)
            counts.append(int(count))
            if use_kl:
                kl = distillation_kl(
                    output["logits"][0, selected][None],
                    cache[index]["logits"].to(ctx.device)[None],
                    student_labels[None],
                    temperature=2.0,
                )
                kl_sums.append(kl * count)
            if aligned is None:
                aligned = {
                    "family": records[index]["family"],
                    "teacher_prediction_rows": cache[index]["prediction_rows"].tolist(),
                    "student_prediction_rows": selected.nonzero().flatten().tolist(),
                    "answer_ids": student_labels.tolist(),
                    "same_answer_labels": True,
                    "teacher_visual_tokens": 16,
                    "student_visual_tokens": 4,
                }
        ce = torch.stack(ce_sums).sum() / sum(counts)
        kl = torch.stack(kl_sums).sum() / sum(counts) if use_kl else ce.new_zeros(())
        loss = 0.5 * ce + 0.5 * kl if use_kl else ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: nonfinite loss at {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True))
        optimizer.step()
        effective += sum(counts)
        if step in (0, steps - 1) or (step + 1) % 50 == 0:
            trace.append({"step": step + 1, "ce": float(ce.detach()), "kl": float(kl.detach()), "gradient_norm": grad})
    _sync(ctx.device)
    elapsed = time.perf_counter() - start
    after = _parameter_hash(student)
    training = {
        "steps": steps,
        "optimizer_updates": steps,
        "batch_size": 4,
        "learning_rate": 0.003,
        "seconds": elapsed,
        "training_examples": len(records),
        "effective_answer_tokens": effective,
        "initialization_sha256": before,
        "final_sha256": after,
        "weights_changed": before != after,
        "batch_plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
        "objective": "0.5 CE + 0.5 KL(teacher||student) T²" if use_kl else "CE",
        "temperature": 2.0,
        "temperature_squared_applied_once": use_kl,
        "alignment": aligned,
        "loss_trace": trace,
    }
    training["training_checkpoint"] = f"{name}-training.pt"
    metadata = {"task": records[0]["modality"], "training": training}
    save_checkpoint(Path(ctx.output) / f"{name}-training.pt", student, optimizer, steps, metadata)
    path = _save_float(ctx, student, name, steps, metadata)
    reloaded, _ = load_checkpoint(path, ctx.device)
    return reloaded.eval(), path, training


def run_multimodal_distillation(ctx):
    """教師 16 個視覺 tokens、學生 4 個，按展開後的有效答案 labels 對齊。"""
    from .modalities import evaluate_modal

    Path(ctx.output).mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    tasks, final = {}, None
    for identifier in ("vqa", "joint"):
        teacher, provenance = _teacher(ctx, identifier)
        if not isinstance(teacher, MultiModalLM):
            raise ValueError(f"{identifier}: 需要完整訓練過的 multimodal-v1 教師")
        teacher_tokens = (teacher.vision.image_size // teacher.vision.patch_size) ** 2
        if teacher_tokens != 16:
            raise ValueError("本課程對比要求教師 16 個視覺 tokens")
        parts, data = _dataset(ctx, identifier)
        teacher.eval().requires_grad_(False)
        frozen_hash = _parameter_hash(teacher)
        teacher_path = _save_float(ctx, teacher, f"{identifier}-teacher", provenance["steps"], provenance)
        teacher_eval = {split: evaluate_modal(teacher, parts[split], ctx) for split in ("validation", "test")}
        cache, seconds = _cache_modal(ctx, teacher, parts["train"])
        cache_path = Path(ctx.output) / f"{identifier}-teacher-answer-logits.pt"
        torch.save(
            {
                "teacher_sha256": provenance["sha256"],
                "records": cache,
                "alignment": "each model's expanded labels != -100; identical gold answer prefix",
            },
            cache_path,
        )
        seed_everything(ctx.seed)
        language = TinyLM(
            ModelConfig(
                vocab_size=teacher.language.config.vocab_size,
                width=32,
                layers=1,
                heads=2,
                max_length=teacher.language.config.max_length,
            )
        )
        initial = MultiModalLM(language, vision_width=8, audio_width=8)
        initial.vision = VisionEncoder(width=8, image_size=teacher.vision.image_size, patch_size=8)
        initial.to(ctx.device)
        if sum(p.numel() for p in initial.parameters()) >= sum(p.numel() for p in teacher.parameters()):
            raise ValueError("多模態學生必須真的少於教師參數")
        runs = {}
        for method in ("ce", "ce_kl"):
            model, path, training = _fit_modal(
                ctx,
                copy.deepcopy(initial),
                parts["train"],
                cache,
                f"{identifier}-{method}",
                _steps(ctx, 350),
                method == "ce_kl",
            )
            runs[method] = {
                "training": training,
                "storage": _storage(model, path),
                "validation": evaluate_modal(model, parts["validation"], ctx),
                "test": evaluate_modal(model, parts["test"], ctx),
                "blank_image_test": evaluate_modal(model, parts["test"], ctx, "blank_image"),
            }
            if identifier == "joint":
                runs[method]["blank_audio_test"] = evaluate_modal(model, parts["test"], ctx, "blank_audio")
                if method == "ce_kl":
                    final = (model, training)
        if runs["ce"]["training"]["initialization_sha256"] != runs["ce_kl"]["training"]["initialization_sha256"]:
            raise ValueError("多模態 CE／KL 學生初始化不一致")
        if runs["ce"]["training"]["batch_plan_sha256"] != runs["ce_kl"]["training"]["batch_plan_sha256"]:
            raise ValueError("多模態 CE／KL 學生 batch plan 不一致")
        if _parameter_hash(teacher) != frozen_hash:
            raise ValueError("多模態蒸餾過程修改了教師")
        tasks[identifier] = {
            "teacher_provenance": provenance,
            "teacher_storage": _storage(teacher, teacher_path),
            "teacher_validation": teacher_eval["validation"],
            "teacher_test": teacher_eval["test"],
            "teacher_frozen_and_unchanged": True,
            "data": data,
            "visual_tokens": {"teacher": 16, "student": 4},
            "language_widths": {"teacher": teacher.language.config.width, "student": 32},
            "teacher_cache": {"seconds": seconds, "file_bytes": cache_path.stat().st_size, "file": str(cache_path)},
            "runs": runs,
            "distilled_minus_ce_test_em": runs["ce_kl"]["test"]["exact_match"] - runs["ce"]["test"]["exact_match"],
        }
    model, training = final
    _save_float(ctx, model, "model", training["steps"], {"task": "joint", "training": training})
    return {
        "experiment": "multimodal_distillation",
        "sections": ["18.7", "18.8", "18.9", "18.10", "18.13"],
        "tasks": tasks,
        "checkpoint": "model.pt",
        "seconds": time.perf_counter() - began,
        "alignment_rule": "teacher/student prediction rows are derived separately from shifted expanded labels, then identical answer ids are verified",
        "teacher_signal": "frozen teacher-forced answer logits; same image/audio/question for both models",
        "limitations": [
            "synthetic shapes and tones; no natural VQA capability claim",
            "smaller prefix does not imply successful distillation",
            "CE and KL students have the same initialization, training plan and heldout denominators",
        ],
    }
