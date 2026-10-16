"""預先編碼、平衡抽樣及可精確續訓的狀態；不在訓練迴圈讀最後測試成績。"""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import CandidateScorer, Config, LoopedSolver, validate_candidates


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def encoded_data(rows, product, tokenizer, config, device="cpu"):
    """先在CPU填好一塊tensor，再整塊搬上裝置，避免逐候選啟動GPU複製。"""
    sequences, owners = [], []
    max_candidates = 1 if product == "b" else max(len(r["candidates"]) for r in rows)
    for i, row in enumerate(rows):
        if product == "b":
            sequences.append(tokenizer.encode(row["state"]))
            owners.append((i, 0))
        else:
            validate_candidates(row["question"], row["candidates"])
            for j, candidate in enumerate(row["candidates"]):
                sequences.append(tokenizer.encode(row["state"], row["question"], candidate))
                owners.append((i, j))
    length = max(map(len, sequences))
    if length > config.max_length:
        raise ValueError("資料超出事先固定的token上限")
    tokens = torch.zeros((len(rows), max_candidates, length), dtype=torch.long)
    tokens[:, :, 0] = 1  # 不存在的候選最後會遮罩，但其編碼仍有合法CLS。
    mask = torch.zeros((len(rows), max_candidates), dtype=torch.bool)
    for sequence, (i, j) in zip(sequences, owners, strict=True):
        tokens[i, j, : len(sequence)] = torch.tensor(sequence)
        mask[i, j] = True
    labels = torch.tensor([r["gold"] if product == "b" else r["candidates"].index(r["gold"]) for r in rows])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if product == "b":
            key = ("sum", 0) if row["task"] == "sum" else ("boxes", len(row["world"]["events"]))
        else:
            key = (row["question"], row["gold"])
        groups[key].append(i)
    return {
        "tokens": tokens.to(device),
        "mask": mask.to(device),
        "labels": labels.to(device),
        "groups": {key: torch.tensor(indices) for key, indices in groups.items()},
        "rows": rows,
        "product": product,
    }


def sample_batch(data, size, generator):
    groups = data["groups"]
    categories = sorted(groups)
    indices = []
    for _ in range(size):
        if data["product"] == "b":
            task = "sum" if int(torch.randint(4, (), generator=generator)) == 0 else "boxes"
            allowed = [key for key in categories if key[0] == task]
        else:
            question = ("choice", "info", "rubric")[int(torch.randint(3, (), generator=generator))]
            allowed = [key for key in categories if key[0] == question]
        if not allowed:
            allowed = categories
        key = allowed[int(torch.randint(len(allowed), (), generator=generator))]
        group = groups[key]
        indices.append(int(group[int(torch.randint(len(group), (), generator=generator))]))
    return torch.tensor(indices, dtype=torch.long, device=data["tokens"].device)


def logits_for(model, data, indices, loops=3, all_depths=False):
    tokens = data["tokens"][indices]
    if data["product"] == "b":
        return model(tokens[:, 0], loops=loops, all_depths=all_depths)
    batch, candidates, length = tokens.shape
    scores = model(tokens.reshape(batch * candidates, length)).view(batch, candidates)
    return scores.masked_fill(~data["mask"][indices], float("-inf"))


def step(model, optimizer, data, generator, batch_size, mode, depth_weights, step_number=0, depth_seed=8159):
    model.train()
    indices = sample_batch(data, batch_size, generator)
    labels = data["labels"][indices]
    optimizer.zero_grad(set_to_none=True)
    terminal = 0
    if data["product"] == "b" and mode == "multi":
        logits = logits_for(model, data, indices, loops=5, all_depths=True)
        losses = F.cross_entropy(
            logits.flatten(0, 1), labels[:, None].expand(-1, 5).reshape(-1), reduction="none"
        ).view(-1, 5)
        weights = torch.tensor(depth_weights, device=logits.device)
        loss = (losses * weights).sum(-1).mean() / weights.sum()
        terminal = 5
    else:
        # 終止深度用獨立的可重現序列，不消耗minibatch的抽樣RNG。
        depth_bucket = int(hashlib.sha256(f"{depth_seed}:{step_number}".encode()).hexdigest()[:8], 16) % 3
        terminal = (1, 3, 5)[depth_bucket] if mode == "odd" else 3
        logits = logits_for(model, data, indices, loops=terminal)
        loss = F.cross_entropy(logits, labels)
    if not bool(torch.isfinite(loss)):
        raise ValueError("非有限loss，不保存成成功訓練")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "terminal_depth": terminal if data["product"] == "b" else 0,
        "block_sequence_work": len(indices)
        * (terminal if data["product"] == "b" else data["mask"].shape[1] * model.config.c_layers),
        "block_token_work": len(indices)
        * data["tokens"].shape[-1]
        * (terminal if data["product"] == "b" else data["mask"].shape[1] * model.config.c_layers),
    }


@torch.inference_mode()
def predict(model, data, loops=3, batch_size=128):
    model.eval()
    scores = []
    for start in range(0, len(data["rows"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data["rows"])), device=data["tokens"].device)
        scores.append(logits_for(model, data, indices, loops=loops).float().cpu())
    return torch.cat(scores)


def metrics(logits, labels, temperature=1.0):
    labels = torch.as_tensor(labels, dtype=torch.long)
    logits = logits.cpu().float() / temperature
    probabilities = logits.softmax(-1)
    truth = F.one_hot(labels, logits.shape[-1]).float()
    return {
        "count": len(labels),
        "correct": int((logits.argmax(-1) == labels).sum()),
        "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        "cross_entropy": float(F.cross_entropy(logits, labels)),
        "brier": float((probabilities - truth).square().sum(-1).mean()),
    }


def checkpoint(path, model, optimizer, generator, metadata):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler_rng": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": asdict(model.config),
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(path, product, device, learning_rate=0.001):
    value = torch.load(path, map_location="cpu", weights_only=True)
    model = (LoopedSolver if product == "b" else CandidateScorer)(Config(**value["config"])).to(device)
    model.load_state_dict(value["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
    optimizer.load_state_dict(value["optimizer"])
    generator = torch.Generator().set_state(value["sampler_rng"])
    torch.set_rng_state(value["torch_rng"])
    if value["cuda_rng"]:
        if device.type != "cuda":
            raise ValueError("GPU續訓狀態需在GPU驗證；CPU推論只載入公開model state")
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    return model, optimizer, generator, value["metadata"]


def same_tree(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(same_tree(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(same_tree(a, b) for a, b in zip(first, second, strict=True))
    return first == second


def verify_resume(path, product, device, data, options):
    first, opt1, gen1, metadata = restore(path, product, device)
    start_hash = state_digest(first.state_dict())
    one = step(first, opt1, data, gen1, step_number=metadata["step"] + 1, **options)
    second, opt2, gen2, _ = restore(path, product, device)
    two = step(second, opt2, data, gen2, step_number=metadata["step"] + 1, **options)
    same = all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
    if (
        one != two
        or not same
        or not same_tree(opt1.state_dict(), opt2.state_dict())
        or not torch.equal(gen1.get_state(), gen2.get_state())
    ):
        raise ValueError("checkpoint續訓未精確重現同一批與更新")
    end_hash = state_digest(second.state_dict())
    if start_hash == end_hash:
        raise ValueError("續訓沒有改動權重")
    return {
        "exact_next_step": True,
        "loss": one["loss"],
        "before_sha256": start_hash,
        "after_sha256": end_hash,
        "forked_step_discarded": True,
        "scope": "Two independently restored checkpoint forks with identical sampler/Torch RNG, one optimizer step each; main checkpoint not advanced.",
    }
