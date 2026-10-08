"""Shared, explicit training and complete held-out evaluation for course experiments."""

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tiny_perceptron.assets import unpack_asset
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum, masked_loss
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything


@dataclass
class Context:
    device: str
    output: Path
    dependencies: Path
    assets: Path
    seed: int = 42

    def dependency(self, experiment_id, filename="model.pt"):
        path = self.dependencies / experiment_id / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
        return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed(value):
    seed_everything(value)


def new_lm(ctx, width=64, layers=2, max_length=128, **kwargs):
    seed(ctx.seed)
    return TinyLM(ModelConfig(width=width, layers=layers, max_length=max_length, **kwargs)).to(ctx.device)


def split_records(records, seed=42, family="family"):
    """Keep all descendants of a family in one split; never split individual answers."""
    groups = {}
    seen = set()
    for record in records:
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        key = str(record.get(family, hashlib.sha256(encoded.encode()).hexdigest()))
        groups.setdefault(key, []).append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three distinct families are required for train/validation/test")
    random.Random(seed).shuffle(keys)
    a = min(max(1, int(len(keys) * 0.8)), len(keys) - 2)
    b = min(max(a + 1, int(len(keys) * 0.9)), len(keys) - 1)
    return {
        name: [record for key in selected for record in groups[key]]
        for name, selected in (("train", keys[:a]), ("validation", keys[a:b]), ("test", keys[b:]))
    }


def records_sha256(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def text_examples(records, mode="text", max_length=128, tokenizer=None):
    tok = tokenizer or ByteTokenizer()
    examples = []
    for record in records:
        if mode == "sft":
            messages = record["messages"] if isinstance(record, dict) else record
            x, y = render_chat(messages, tok)
            if len(x) > max_length:
                raise ValueError(f"SFT record has {len(x)} tokens, exceeding {max_length}; crop explicitly upstream")
            examples.append((x, y))
        elif mode == "text":
            text = record["text"] if isinstance(record, dict) else record
            ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
            # Every next-token label appears once; chunk boundaries share only the preceding input token.
            for start in range(0, len(ids) - 1, max_length):
                examples.append(shifted(ids[start : start + max_length + 1]))
        else:
            raise ValueError("mode must be text or sft")
    if not examples:
        raise ValueError("No examples to train or evaluate")
    return examples


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def _nll(model, examples, batch_size=16):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    try:
        for start in range(0, len(examples), batch_size):
            x, y, valid = (t.to(device) for t in pad_batch(examples[start : start + batch_size]))
            summed, n = loss_sum(model(x, valid=valid)["logits"], y)
            total += float(summed)
            count += int(n)
    finally:
        model.train(was_training)
    return {"nll": total / count, "nll_sum": total, "effective_tokens": count, "examples": len(examples)}


def fit_lm(
    model, records, ctx, mode="text", steps=600, batch_size=16, lr=0.003, auxiliary=0.01, name="model", tokenizer=None
):
    examples = text_examples(records, mode, model.config.max_length, tokenizer)
    initial = _nll(model, examples)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sampler = random.Random(ctx.seed)
    history, effective_tokens = [], 0
    model.train()
    _sync(ctx.device)
    started = time.perf_counter()
    for step in range(steps):
        batch = sampler.choices(examples, k=batch_size)
        x, y, valid = (t.to(ctx.device) for t in pad_batch(batch))
        optimizer.zero_grad(set_to_none=True)
        result = model(x, valid=valid)
        loss = masked_loss(result["logits"], y) + auxiliary * result["auxiliary"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        effective_tokens += int((y != IGNORE).sum())
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step": step + 1, "batch_loss_before_update": float(loss.detach())})
        if (step + 1) % max(1, steps // 4) == 0:
            save_checkpoint(
                ctx.output / f"{name}-step-{step + 1}.pt",
                model,
                optimizer,
                step + 1,
                {
                    "mode": mode,
                    "schedule": "constant",
                    "lr": lr,
                    "records_sha256": records_sha256(records),
                    "effective_tokens": effective_tokens,
                },
                training_state={
                    "sampler_rng": sampler.getstate(),
                    "batch_size": batch_size,
                    "planned_steps": steps,
                    "auxiliary": auxiliary,
                },
            )
    _sync(ctx.device)
    seconds = time.perf_counter() - started
    final = _nll(model, examples)
    save_checkpoint(
        ctx.output / f"{name}.pt",
        model,
        optimizer,
        steps,
        {
            "mode": mode,
            "schedule": "constant",
            "lr": lr,
            "seed": ctx.seed,
            "records_sha256": records_sha256(records),
            "effective_tokens": effective_tokens,
        },
        training_state={
            "sampler_rng": sampler.getstate(),
            "batch_size": batch_size,
            "planned_steps": steps,
            "auxiliary": auxiliary,
        },
    )
    return {
        "initial_loss": initial["nll"],
        "final_loss": final["nll"],
        "history": history,
        "steps": steps,
        "seconds": seconds,
        "effective_tokens": effective_tokens,
        "checkpoint": f"{name}.pt",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "records": len(records),
        "records_sha256": records_sha256(records),
    }


def fit(model, loss_fn, ctx, steps, lr=0.003, name="model", metadata=None):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    model.train()
    _sync(ctx.device)
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step": step + 1, "loss_before_update": float(loss.detach())})
        if hasattr(model, "config") and (step + 1) % max(1, steps // 4) == 0:
            save_checkpoint(ctx.output / f"{name}-step-{step + 1}.pt", model, optimizer, step + 1, metadata)
    _sync(ctx.device)
    seconds = time.perf_counter() - started
    if hasattr(model, "config"):
        save_checkpoint(ctx.output / f"{name}.pt", model, optimizer, steps, metadata)
    else:
        ctx.output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": steps,
                "metadata": metadata or {},
                "torch_rng": torch.get_rng_state(),
            },
            ctx.output / f"{name}.pt",
        )
    return {"history": history, "steps": steps, "seconds": seconds, "checkpoint": f"{name}.pt"}


@torch.no_grad()
def evaluate_lm(model, records, mode="text", tokens=32, tokenizer=None):
    tok = tokenizer or ByteTokenizer()
    result = _nll(model, text_examples(records, mode, model.config.max_length, tok))
    device = next(model.parameters()).device
    samples, matches, ended = [], 0, 0
    if mode == "sft":
        for record in records:
            messages = record["messages"]
            if messages[-1]["role"] != "assistant":
                raise ValueError("SFT evaluation must end with an assistant answer")
            prompt = [tok.bos_id]
            roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
            for message in messages[:-1]:
                prompt += [roles[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
            prompt += [tok.assistant_id]
            ids = torch.tensor([prompt], device=device)
            generated = generate(model, ids, tokens, eos_id=tok.eos_id)[0, len(prompt) :].tolist()
            raw = generated[: generated.index(tok.eos_id)] if tok.eos_id in generated else generated
            predicted, expected = tok.decode(raw), messages[-1]["content"]
            # Raw token identity catches invalid special tokens hidden by the decoder.
            exact = raw == tok.encode(expected)
            matches += int(exact)
            ended += int(tok.eos_id in generated)
            samples.append(
                {
                    "messages": messages[:-1],
                    "expected": expected,
                    "generated": predicted,
                    "generated_ids": generated,
                    "exact": exact,
                    "eos": tok.eos_id in generated,
                }
            )
        result.update(
            {
                "exact_match": matches / len(records),
                "matches": matches,
                "records": len(records),
                "eos_rate": ended / len(records),
                "samples": samples,
            }
        )
    else:
        for record in records[:8]:
            text = record["text"] if isinstance(record, dict) else record
            prefix = tok.encode(text)[: min(24, max(1, model.config.max_length - tokens - 1))]
            ids = torch.tensor([[tok.bos_id] + prefix], device=device)
            generated = generate(model, ids, tokens, eos_id=tok.eos_id)[0, ids.shape[1] :].tolist()
            samples.append(
                {"prompt": tok.decode(prefix), "generated": tok.decode(generated), "generated_ids": generated}
            )
        result["samples"] = samples
        result["records"] = len(records)
    return result


def load_lm(path, device):
    model, _ = load_checkpoint(path, device)
    return model


def extract_asset(ctx, asset_id):
    manifest = json.loads((ctx.assets / "manifest.json").read_text(encoding="utf-8"))
    asset = next(a for a in manifest["assets"] if a["id"] == asset_id)
    archive = ctx.assets.parent.parent / asset["archive"]
    destination = ctx.output.parent / "unpacked-assets"
    unpack_asset(archive, asset, destination)
    return destination
