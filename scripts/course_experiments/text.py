"""文字教學的實跑；每個入口保存資料切分、真正更新與完整留出評估。"""

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.prepare_data import generate_records
from tiny_perceptron.data import SPECIALS, ByteTokenizer, load_jsonl, pad_batch, shifted, toy_documents
from tiny_perceptron.model import generate, loss_sum, masked_loss
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.training import learning_rate, load_checkpoint, save_checkpoint

from .common import (
    evaluate_lm,
    extract_asset,
    fit,
    fit_lm,
    load_lm,
    new_lm,
    seed,
    split_records,
    text_examples,
    write_json,
)


def _steps(ctx, count):
    """正式預設完整跑；本機通路檢查可由 caller 明示 step_scale。"""
    return max(1, round(count * getattr(ctx, "step_scale", 1.0)))


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _save_splits(ctx, parts, name="data"):
    directory = ctx.output / name
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, rows in parts.items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        result[split] = {
            "records": len(rows),
            "families": len({str(row.get("family", row.get("id", _digest(row)))) for row in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path.relative_to(ctx.output)),
        }
    result["split_unit"] = "family before tokenization; no family crosses splits"
    write_json(directory / "manifest.json", result)
    return result


def _evaluations(model, parts, mode="sft", tokens=32):
    result = {split: evaluate_lm(model, parts[split], mode=mode, tokens=tokens) for split in ("validation", "test")}
    if mode == "text":
        for split, report in result.items():
            raw_bytes = sum(len(row["text"].encode("utf-8")) for row in parts[split])
            report["raw_utf8_bytes"] = raw_bytes
            report["bpb_including_eos_boundary_targets"] = report["nll_sum"] / (raw_bytes * math.log(2))
    return result


def _asset_rows(ctx, identifier, basename):
    root = Path(extract_asset(ctx, identifier))
    manifest = json.loads((ctx.assets / "manifest.json").read_text(encoding="utf-8"))
    entry = next(asset for asset in manifest["assets"] if asset["id"] == identifier)
    # 多個來源都使用 train-first-100.jsonl，必須限定到此來源的 target。
    directory = root / entry["target"]
    if not directory.is_dir() and root.name == Path(entry["target"]).name:
        directory = root
    matches = list(directory.rglob(basename))
    if root.is_file() and root.name == basename:
        matches = [root]
    if len(matches) != 1:
        raise ValueError(f"{identifier}: expected one {basename}, found {len(matches)}")
    return load_jsonl(matches[0])


def _deduplicate_text(rows):
    """只去完整內容重複；近重複仍需另外審計，不假裝已全部消除。"""
    result, seen = [], set()
    for row in rows:
        key = " ".join(row["text"].split())
        if key not in seen:
            result.append({**row, "family": hashlib.sha256(key.encode()).hexdigest()})
            seen.add(key)
    return result


def _simple_examples(documents, vocabulary, context):
    inputs, labels = [], []
    for text in documents:
        ids = [vocabulary.get(char, 1) for char in text] + [0]
        padded = [0] * context + ids
        for position, answer in enumerate(ids):
            inputs.append(padded[position : position + context])
            labels.append(answer)
    return torch.tensor(inputs), torch.tensor(labels)


@torch.no_grad()
def _simple_sample(model, vocabulary, context, prompt, device, count=24):
    reverse = {value: key for key, value in vocabulary.items()}
    history = [0] * context + [vocabulary.get(char, 1) for char in prompt]
    generated = []
    for _ in range(count):
        ids = torch.tensor([history[-context:]], device=device)
        answer = int(model(ids).argmax(-1).item())
        if answer == 0:
            break
        history.append(answer)
        generated.append(reverse.get(answer, "<UNK>"))
    return {"prompt": prompt, "generated": "".join(generated)}


def run_simple_models(ctx):
    from tiny_perceptron.data import split_documents

    parts = split_documents(toy_documents(), ctx.seed)
    rows = {split: [{"text": text, "family": text} for text in values] for split, values in parts.items()}
    manifest = _save_splits(ctx, rows)
    vocabulary = {char: index + 2 for index, char in enumerate(sorted(set("".join(parts["train"]))))}
    results = {}
    for name, context in (("bigram", 1), ("mlp1", 1), ("mlp3", 3), ("mlp5", 5)):
        seed(ctx.seed)
        model = BigramLM(len(vocabulary) + 2) if name == "bigram" else ContextMLP(len(vocabulary) + 2, context, 16)
        model.to(ctx.device)
        batches = {
            split: tuple(tensor.to(ctx.device) for tensor in _simple_examples(values, vocabulary, context))
            for split, values in parts.items()
        }
        before = {}
        with torch.no_grad():
            for split, (x, y) in batches.items():
                before[split] = float(F.cross_entropy(model(x), y))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        history = []
        began = time.perf_counter()
        for step in range(_steps(ctx, 200)):
            optimizer.zero_grad(set_to_none=True)
            x, y = batches["train"]
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError(f"{name}: non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError(f"{name}: non-finite gradient")
            optimizer.step()
            if step == 0 or (step + 1) % 20 == 0 or step + 1 == _steps(ctx, 200):
                history.append({"step": step + 1, "loss_before_update": float(loss.detach())})
        if str(ctx.device).startswith("cuda"):
            torch.cuda.synchronize()
        after = {}
        with torch.no_grad():
            for split, (x, y) in batches.items():
                after[split] = float(F.cross_entropy(model(x), y))
        torch.save(
            {
                "format_version": "simple-v1",
                "model": model.state_dict(),
                "vocabulary": vocabulary,
                "context": context,
                "width": 16,
                "kind": "bigram" if name == "bigram" else "mlp",
            },
            ctx.output / f"{name}.pt",
        )
        results[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "context": context,
            "steps": _steps(ctx, 200),
            "seconds": time.perf_counter() - began,
            "before_nll": before,
            "after_nll_same_post_update_time": after,
            "history": history,
            "samples": [
                _simple_sample(model, vocabulary, context, prompt, ctx.device) for prompt in ("顏色=", "顏色=紅；形狀=")
            ],
            "checkpoint": f"{name}.pt",
            "checkpoint_loader": "simple-v1; not the TinyLM infer.py format",
        }
    write_json(ctx.output / "vocabulary.json", vocabulary)
    return {
        "experiment": "simple_models",
        "sections": ["1.5", "1.6", "1.12", "1.14", "2.2", "2.5", "T.3"],
        "data": manifest,
        "runs": results,
        "scope": "12 controlled documents; context and parameter changes are separate costs",
    }


def _resume_probe(ctx):
    seed(ctx.seed)
    model = new_lm(ctx, width=16, layers=1, max_length=64)
    examples = text_examples(generate_records("toy-text"), mode="text", max_length=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    total = max(4, _steps(ctx, 80))
    halfway = total // 2

    def update(current_model, current_optimizer, step):
        selected = random.choices(examples, k=4)
        x, y, valid = (tensor.to(ctx.device) for tensor in pad_batch(selected))
        for group in current_optimizer.param_groups:
            group["lr"] = learning_rate(step, total, peak=0.003)
        current_optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(current_model(x, valid=valid)["logits"], y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError("resume probe has non-finite training")
        current_optimizer.step()

    for step in range(total):
        update(model, optimizer, step)
        if step + 1 == halfway:
            save_checkpoint(
                ctx.output / "resume-middle.pt",
                model,
                optimizer,
                halfway,
                {"schedule_steps": total, "peak_lr": 0.003, "seed": ctx.seed},
            )
    final = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    resumed, payload = load_checkpoint(ctx.output / "resume-middle.pt", ctx.device, restore_rng=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.003)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    for step in range(halfway, total):
        update(resumed, resumed_optimizer, step)
    difference = max(float((final[name] - value).abs().max()) for name, value in resumed.state_dict().items())
    save_checkpoint(
        ctx.output / "resume-final.pt",
        resumed,
        resumed_optimizer,
        total,
        {"schedule_steps": total, "peak_lr": 0.003, "seed": ctx.seed},
    )
    if difference > 1e-5:
        raise ValueError(f"resume did not reproduce uninterrupted training: {difference}")
    return {
        "total_steps": total,
        "saved_step": halfway,
        "max_weight_difference": difference,
        "schedule_total_preserved": True,
        "optimizer_and_rng_restored": True,
        "tolerance": 1e-5,
    }


def run_text_foundation(ctx):
    records = generate_records("toy-text")
    parts = split_records(records, seed=ctx.seed)
    manifest = _save_splits(ctx, parts)
    seed(ctx.seed)
    model = new_lm(ctx, width=64, layers=2)
    save_checkpoint(ctx.output / "start.pt", model, step=0, metadata={"seed": ctx.seed, "data": manifest})
    before = _evaluations(model, parts, mode="text")
    training = fit_lm(model, parts["train"], ctx, mode="text", steps=_steps(ctx, 600))
    after = _evaluations(model, parts, mode="text")
    probe = torch.tensor([[1, 10, 11, 12, 13]], device=ctx.device)
    changed = probe.clone()
    changed[0, -1] = 14
    with torch.no_grad():
        causal_difference = float((model(probe)["logits"][:, :-1] - model(changed)["logits"][:, :-1]).abs().max())
    if causal_difference > 1e-6:
        raise ValueError("future token affected past logits")
    factorial = {}
    pool = [
        {"text": f"object={index};color={color};shape={shape}.", "family": str(index)}
        for index in range(80)
        for color, shape in [("red" if index % 2 else "blue", "circle" if index % 3 else "square")]
    ]
    scale_parts = split_records(pool, seed=ctx.seed)
    scale_manifest = _save_splits(ctx, scale_parts, name="scaling-data")
    for width in (16, 32):
        for count in (16, 64):
            seed(ctx.seed)
            current = new_lm(ctx, width=width, layers=1)
            name = f"scaling-w{width}-n{count}"
            fit_result = fit_lm(
                current, scale_parts["train"][:count], ctx, mode="text", steps=_steps(ctx, 150), batch_size=4, name=name
            )
            factorial[name] = {
                "training": fit_result,
                "parameters": sum(p.numel() for p in current.parameters()),
                "unique_training_records": min(count, len(scale_parts["train"])),
                "evaluation": _evaluations(current, scale_parts, mode="text"),
            }
    return {
        "experiment": "text_foundation",
        "sections": ["4.5", "4.7", "5.1", "5.7", "5.8", "5.9", "5.13", "T.4"],
        "data": manifest,
        "training": training,
        "before": before,
        "after": after,
        "causal_max_difference": causal_difference,
        "resume": _resume_probe(ctx),
        "scaling": factorial,
        "scaling_data": scale_manifest,
        "comparison_budget": "same optimizer steps and batch size; record measured effective tokens and parameters",
        "scope": "controlled short text; factorial points do not establish a scaling law",
    }


def run_real_text(ctx):
    results = {}
    for identifier, basename, steps in (
        ("tinystories", "tinystories-train-512.jsonl", 800),
        ("chinese-poetry", "chinese-classical-train-365.jsonl", 800),
    ):
        raw = _asset_rows(ctx, identifier, basename)
        records = _deduplicate_text(raw)
        parts = split_records(records, seed=ctx.seed)
        manifest = _save_splits(ctx, parts, name=f"{identifier}-data")
        seed(ctx.seed)
        model = new_lm(ctx, width=64, layers=2, max_length=128)
        before = _evaluations(model, parts, mode="text")
        training = fit_lm(
            model, parts["train"], ctx, mode="text", steps=_steps(ctx, steps), batch_size=16, name=identifier
        )
        results[identifier] = {
            "source_records": len(raw),
            "deduplicated_records": len(records),
            "data": manifest,
            "training": training,
            "before": before,
            "after": _evaluations(model, parts, mode="text"),
            "license": raw[0].get("license"),
            "source_revision": raw[0].get("source_revision"),
            "split_warning": "original source split is not our test; near-duplicate clustering was not performed",
            "scope": "small fixed pilot with reset context windows, not a generally capable language model",
        }
    return {"experiment": "real_text", "sections": ["5.8", "5.10", "5.11", "6.1", "T.1", "T.4"], "runs": results}


class _BPE:
    """角色 ID 在程式端插入；普通文字編碼不把字面 <user> 當控制標記。"""

    def __init__(self, tokenizer):
        from tokenizers import Tokenizer

        self.tokenizer = tokenizer
        definition = json.loads(tokenizer.to_str())
        definition["added_tokens"] = []
        self.content = Tokenizer.from_str(json.dumps(definition))
        self.vocab_size = tokenizer.get_vocab_size()
        for name, spelling in zip(
            ("pad", "bos", "eos", "user", "assistant", "image", "audio", "system"), SPECIALS, strict=True
        ):
            setattr(self, f"{name}_id", tokenizer.token_to_id(spelling))
        self.special_ids = {tokenizer.token_to_id(spelling) for spelling in SPECIALS}

    def encode(self, text):
        return self.content.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.content.decode(
            [int(value) for value in ids if int(value) not in self.special_ids], skip_special_tokens=False
        )


def _utf8_prefix(text, maximum):
    return text.encode("utf-8")[:maximum].decode("utf-8", errors="ignore")


@torch.no_grad()
def _evaluate_tokenizer(model, rows, tokenizer):
    model.eval()
    device = next(model.parameters()).device
    total, count, raw_bytes = 0.0, 0, 0
    samples = []
    for index, row in enumerate(rows):
        ids = [tokenizer.bos_id] + tokenizer.encode(row["text"]) + [tokenizer.eos_id]
        x, y = shifted(ids)
        nll, effective = loss_sum(model(x[None].to(device))["logits"], y[None].to(device))
        total += float(nll)
        count += int(effective)
        raw_bytes += len(row["text"].encode())
        prompt = row["text"][:4]
        prefix = [tokenizer.bos_id] + tokenizer.encode(prompt)
        output = generate(model, torch.tensor([prefix], device=device), max_new_tokens=32, eos_id=tokenizer.eos_id)
        samples.append(
            {"row": index, "prompt": prompt, "generated": tokenizer.decode(output[0, len(prefix) :].tolist())}
        )
    return {
        "records": len(rows),
        "effective_tokens": count,
        "raw_utf8_bytes": raw_bytes,
        "mean_token_nll": total / count,
        "bpb_including_eos_boundary_targets": total / (raw_bytes * math.log(2)),
        "samples": samples,
        "skipped": [],
        "note": "same held-out raw text, vocabulary differs; EOS counted in total NLL",
    }


def run_tokenizer(ctx):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    raw = _asset_rows(ctx, "tinystories", "tinystories-train-512.jsonl")[:96]
    raw += _asset_rows(ctx, "chinese-poetry", "chinese-classical-train-365.jsonl")[:96]
    complete = split_records(_deduplicate_text(raw), seed=ctx.seed)
    # 文件先分側，再取同一 UTF-8 安全片段；兩模型看見相同原文曝光。
    parts = {
        split: [
            {
                **row,
                "text": _utf8_prefix(row["text"], 256),
                "complete_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            }
            for row in rows
        ]
        for split, rows in complete.items()
    }
    manifest = _save_splits(ctx, parts)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [row["text"] for row in parts["train"]],
        trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=list(SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.save(str(ctx.output / "tokenizer-bpe512.json"))
    bpe = _BPE(tokenizer)
    roundtrip = []
    for text in ("未見字🦊 new", " 小鳥🙂\nnew ", "<user>這只是引用文字"):
        ids = bpe.encode(text)
        restored = bpe.decode(ids)
        roundtrip.append(
            {
                "text": text,
                "ids": ids,
                "restored": restored,
                "same": restored == text,
                "contains_control_id": any(value in bpe.special_ids for value in ids),
            }
        )
        if restored != text or any(value in bpe.special_ids for value in ids):
            raise ValueError("BPE round-trip or literal special-token boundary failed")
    steps = _steps(ctx, 400)
    sampler = random.Random(ctx.seed)
    exposure = [sampler.choices(range(len(parts["train"])), k=8) for _ in range(steps)]
    runs = {}
    for name, tok in (("byte256", ByteTokenizer()), ("bpe512", bpe)):
        seed(ctx.seed)
        model = new_lm(ctx, width=32, layers=1, max_length=384, vocab_size=tok.vocab_size)
        examples = text_examples(parts["train"], mode="text", max_length=384, tokenizer=tok)
        if len(examples) != len(parts["train"]):
            raise ValueError("tokenizer comparison must have exactly one window per source excerpt")
        before = {split: _evaluate_tokenizer(model, parts[split], tok) for split in ("validation", "test")}

        def loss_fn(step):
            x, y, valid = (
                tensor.to(ctx.device)
                for tensor in pad_batch([examples[index] for index in exposure[step]], pad_id=tok.pad_id)
            )
            return masked_loss(model(x, valid=valid)["logits"], y)

        training = fit(
            model,
            loss_fn,
            ctx,
            steps=steps,
            lr=0.003,
            name=name,
            metadata={
                "tokenizer": name,
                "tokenizer_sha256": hashlib.sha256((ctx.output / "tokenizer-bpe512.json").read_bytes()).hexdigest()
                if name == "bpe512"
                else None,
                "same_raw_document_schedule_sha256": _digest(exposure),
            },
        )
        runs[name] = {
            "vocab_size_including_8_specials": tok.vocab_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "input_embedding_parameters": model.embedding.weight.numel(),
            "training": training,
            "before": before,
            "after": {split: _evaluate_tokenizer(model, parts[split], tok) for split in ("validation", "test")},
            "validation_text_token_lengths": [len(tok.encode(row["text"])) for row in parts["validation"]],
            "training_raw_utf8_bytes_exposed": sum(
                len(parts["train"][index]["text"].encode()) for batch in exposure for index in batch
            ),
            "checkpoint": f"{name}.pt",
            "tokenizer_file": "tokenizer-bpe512.json" if name == "bpe512" else "tokenizer-byte.json",
        }
    write_json(ctx.output / "tokenizer-byte.json", ByteTokenizer().state())
    return {
        "experiment": "tokenizer",
        "sections": ["6.3", "6.4", "6.5", "6.6", "6.7"],
        "data": manifest,
        "roundtrip": roundtrip,
        "runs": runs,
        "raw_document_schedule_sha256": _digest(exposure),
        "matched_budget": "same raw document excerpts in the same batches for the same steps; effective tokens and parameters differ",
        "scope": "256 byte symbols plus 8 boundary IDs versus a 512-budget byte-level BPE; not a universal vocabulary recommendation",
    }


def _natural_chat_pilot(ctx):
    raw = _asset_rows(ctx, "ultrachat-sft", "train-first-100.jsonl")
    records = []
    for index, row in enumerate(raw):
        messages = row["messages"]
        user = next((message["content"] for message in messages if message["role"] == "user"), None)
        assistant = next((message["content"] for message in messages if message["role"] == "assistant"), None)
        if user and assistant:
            records.append(
                {
                    "family": row.get("prompt_id", _digest(user)),
                    "source_row": index,
                    "source_record_sha256": _digest(row),
                    "scope": "first-turn UTF-8-safe excerpt",
                    "messages": [
                        {"role": "user", "content": _utf8_prefix(user, 120)},
                        {"role": "assistant", "content": _utf8_prefix(assistant, 120)},
                    ],
                }
            )
    parts = split_records(records, seed=ctx.seed)
    manifest = _save_splits(ctx, parts, name="ultrachat-excerpts")
    seed(ctx.seed)
    model = new_lm(ctx, width=32, layers=1, max_length=256)
    training = fit_lm(
        model, parts["train"], ctx, mode="sft", steps=_steps(ctx, 80), batch_size=4, name="ultrachat-pilot"
    )
    return {
        "source_records": len(raw),
        "prepared_records": len(records),
        "data": manifest,
        "training": training,
        "evaluation": _evaluations(model, parts, tokens=128),
        "license": "MIT",
        "scope": "fragment training/data-path pilot; excerpts may change meaning and do not establish full-dialogue instruction quality",
    }


def run_sft(ctx):
    records = generate_records("attributes-sft")
    parts = split_records(records, seed=ctx.seed)
    manifest = _save_splits(ctx, parts)
    write_json(ctx.output / "dataset.json", parts)
    seed(ctx.seed)
    model = new_lm(ctx, width=64, layers=2)
    save_checkpoint(ctx.output / "start.pt", model, step=0, metadata={"seed": ctx.seed, "data": manifest})
    before = _evaluations(model, parts)
    training = fit_lm(model, parts["train"], ctx, mode="sft", steps=_steps(ctx, 900), name="model")
    after = _evaluations(model, parts)
    # 真正的 pretrain → SFT 是另一支線，不能把隨機 forward 叫微調成果。
    text_records = [
        {"text": row["messages"][0]["content"] + row["messages"][1]["content"], "family": row["family"]}
        for row in parts["train"]
    ]
    seed(ctx.seed)
    pretrained = new_lm(ctx, width=64, layers=2)
    pretraining = fit_lm(pretrained, text_records, ctx, mode="text", steps=_steps(ctx, 250), name="pretrain")
    before_sft = _evaluations(pretrained, parts)
    continuation = fit_lm(pretrained, parts["train"], ctx, mode="sft", steps=_steps(ctx, 900), name="pretrain-sft")
    return {
        "experiment": "sft",
        "sections": ["7.1", "7.9", "7.11", "7.12", "7.13", "T.4"],
        "data": manifest,
        "training": training,
        "before": before,
        "after": after,
        "checkpoint": "model.pt",
        "pretrain_then_sft": {
            "pretraining": pretraining,
            "before_sft": before_sft,
            "sft": continuation,
            "after_sft": _evaluations(pretrained, parts),
        },
        "ultrachat_pilot": _natural_chat_pilot(ctx),
        "scope": "fixed attribute extraction; one held-out validation family is not broad conversational generalization",
    }


def arithmetic_records():
    records = []
    for a in range(8):
        for b in range(8):
            records.append(
                {
                    "family": f"{min(a, b)}+{max(a, b)}",
                    "a": a,
                    "b": b,
                    "messages": [
                        {"role": "user", "content": f"{a}+{b}=?"},
                        {"role": "assistant", "content": str(a + b)},
                    ],
                }
            )
    return records


def run_sft_ablation(ctx):
    attributes = split_records(generate_records("attributes-sft"), seed=ctx.seed)
    arithmetic = split_records(arithmetic_records(), seed=ctx.seed)
    base = load_lm(ctx.dependency("sft"), ctx.device)
    before = {"A_attributes": _evaluations(base, attributes), "B_arithmetic": _evaluations(base, arithmetic)}
    results = {}
    for name, records in (("b-only", arithmetic["train"]), ("replay", attributes["train"] + arithmetic["train"])):
        seed(ctx.seed)
        model = copy.deepcopy(base)
        training = fit_lm(model, records, ctx, mode="sft", steps=_steps(ctx, 500), name=name)
        results[name] = {
            "training": training,
            "A_attributes": _evaluations(model, attributes),
            "B_arithmetic": _evaluations(model, arithmetic),
            "source_records": {"A": len(attributes["train"]) if name == "replay" else 0, "B": len(arithmetic["train"])},
        }
    clean = attributes["train"]
    noisy = copy.deepcopy(clean)
    corrupted = []
    for index, row in enumerate(noisy):
        answer = row["messages"][-1]["content"]
        if answer in ("circle", "square") and len(corrupted) < max(1, len(noisy) // 10):
            row["messages"][-1]["content"] = "square" if answer == "circle" else "circle"
            corrupted.append(
                {"row": index, "family": row["family"], "correct": answer, "wrong": row["messages"][-1]["content"]}
            )
    for name, records in (("clean", clean), ("noisy", noisy)):
        seed(ctx.seed)
        model = copy.deepcopy(base)
        training = fit_lm(model, records, ctx, mode="sft", steps=_steps(ctx, 300), name=name)
        results[name] = {
            "training": training,
            "attributes": _evaluations(model, attributes),
            "corrupted_records": len(corrupted) if name == "noisy" else 0,
        }
    write_json(ctx.output / "corruptions.json", corrupted)
    return {
        "experiment": "sft_ablation",
        "sections": ["7.14", "7.15", "7.16", "5.16"],
        "data": {
            "attributes": _save_splits(ctx, attributes, name="attributes"),
            "arithmetic": _save_splits(ctx, arithmetic, name="arithmetic"),
        },
        "before": before,
        "runs": results,
        "corruptions": corrupted,
        "matched_budget": "same initialization, optimizer updates and batch size; replay supervised-token totals can differ and are reported",
        "scope": "forgetting/replay and wrong-label effects are measured outcomes, not guaranteed improvements",
    }
