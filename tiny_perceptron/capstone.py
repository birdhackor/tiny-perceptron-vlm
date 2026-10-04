"""A small-world integrated assistant, deliberately separate from earlier teaching models.

This is a toy RGB/tone/text model, not a general Chinese assistant. A generated tool
request is checked by a calculator-only runtime. Runtime results never replace a
model answer: they become input to another generation by the same language core.
"""

import copy
import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import Block, ModelConfig, TinyLM
from tiny_perceptron.multimodal import log_mel, scene, tone

STAGES = ("pretrain", "sft", "joint", "dpo")
DEFAULT_STEPS = {"pretrain": 300, "sft": 1400, "joint": 600, "dpo": 100}
TOK = ByteTokenizer()
DATA_VERSION = "capstone-small-world-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def default_config(dense=False):
    return ModelConfig(
        width=64,
        layers=2,
        heads=2,
        kv_heads=1,
        max_length=192,
        norm="rms",
        rotary=True,
        backend="sdpa",
        experts=0 if dense else 4,
        top_k=2,
        activation="gelu",
        tied=False,
    )


class PadSafeBlock(Block):
    """Use the existing FFN only on real tokens, including its routing auxiliary."""

    def forward(self, x, valid=None, positions=None, segments=None, cache=None):
        attended, new_cache = self.attention(self.norm1(x), valid, positions, segments, cache)
        x = x + attended
        normalized = self.norm2(x)
        auxiliary = x.new_zeros(())
        if valid is not None:
            selected = normalized[valid]
            h = self.ffn(selected.unsqueeze(0))
            if isinstance(h, tuple):
                h, auxiliary, _ = h
            scattered = torch.zeros_like(x)
            scattered[valid] = h.squeeze(0)
            h = scattered
        else:
            h = self.ffn(normalized)
            if isinstance(h, tuple):
                h, auxiliary, _ = h
        return x + h, new_cache, auxiliary


class CapstoneModel(nn.Module):
    """One language core; each synthetic modality occupies one explicit prefix slot.

    RGB is pooled into a 4x4 grid. Audio is converted to a real 16-band log-mel
    spectrum and pooled in time. The pooling loses detail by design; this model
    only learns simple colors, shapes and low/high tones.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or default_config()
        self.language = TinyLM(self.config)
        self.language.blocks = nn.ModuleList([PadSafeBlock(self.config) for _ in range(self.config.layers)])
        self.image_projector = nn.Sequential(nn.Linear(48, self.config.width), nn.GELU())
        self.audio_projector = nn.Sequential(nn.Linear(16, self.config.width), nn.GELU())

    def forward(self, ids, *, valid=None, images=None, audio_features=None, cache=None):
        if ids.ndim != 2:
            raise ValueError("ids must have batch and sequence dimensions")
        embedded = self.language.embedding(ids)
        for marker, features, projector in (
            (TOK.image_id, images, self.image_projector),
            (TOK.audio_id, audio_features, self.audio_projector),
        ):
            slots = ids == marker
            if slots.any():
                if features is None or len(features) != len(ids):
                    raise ValueError("A modality marker requires its paired batch of features")
                if (slots.sum(-1) > 1).any():
                    raise ValueError("Only one image and one audio marker per prompt are supported")
                if marker == TOK.image_id:
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4)).flatten(1)
                projected = projector(features)
                embedded = torch.where(slots.unsqueeze(-1), projected.unsqueeze(1), embedded)
        return self.language(embeddings=embedded, valid=valid, cache=cache)

    def description(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        expert_one = (
            sum(parameter.numel() for parameter in self.language.blocks[0].ffn.experts[0].parameters())
            if self.config.experts
            else 0
        )
        # Logical active count, not measured FLOPs or a memory/speed guarantee.
        inactive = (
            self.config.layers * (self.config.experts - self.config.top_k) * expert_one if self.config.experts else 0
        )
        return {"config": asdict(self.config), "parameters": total, "logical_active_parameters": total - inactive}


def _row(family, task, user, answer, *, available=True, image=None, audio=None, system=None):
    record = {
        "family": family,
        "task": task,
        "user": user,
        "system": system or f"計算器={'開' if available else '關'}；風格=短。",
        "available": available,
        "answer": answer,
        "image": image,
        "audio": audio,
    }
    record["id"] = digest(record)[:20]
    return record


def build_dataset(seed=42):
    """Freeze task-family splits before fitting; all descendants stay together.

    Every numeric (a,b) family includes its tool-availability counterfactual,
    copies, and runtime-return turns. Every visual/audio family includes all
    shifts, pitches and questions about that underlying scene/tone combination.
    """
    groups = {"numbers": {}, "modalities": {}, "audio": {}, "context": {}}
    for a in range(10):
        for b in range(a, 10):
            family = f"numbers:{a}:{b}"
            rows = []
            for question in (f"{a}+{b}等於多少？", f"請算{a}加{b}。"):
                rows.extend(
                    [
                        _row(family, "calculator", question, f"TOOL:calculator:{a}+{b}"),
                        _row(family, "unavailable", question, "ASK:計算器未開", available=False),
                    ]
                )
            rows.append(_row(family, "tool_return", f"原題：{a}+{b}。計算器回報：{a + b}。請回答。", f"DIRECT:{a + b}"))
            rows.append(_row(family, "concept", f"用{a}和{b}說明加法。", "DIRECT:把兩個數合起來"))
            groups["numbers"][family] = rows
    for color in ("red", "green", "blue"):
        for shape in ("square", "circle"):
            for pitch in ("low", "high"):
                for variant in range(3):
                    family = f"modalities:{color}:{shape}"
                    rows = []
                    for offset in (-1, 0, 1):
                        image = {"color": color, "shape": shape, "offset": offset, "variant": variant}
                        color_index = ("red", "green", "blue").index(color)
                        shape_index = ("square", "circle").index(shape)
                        audio = {"pitch": pitch, "variation": variant, "delta": 23 + color_index * 7 + shape_index * 3}
                        rows.extend(
                            [
                                _row(family, "image_color", "圖片是什麼顏色？", f"DIRECT:{color}", image=image),
                                _row(family, "image_shape", "圖片是什麼形狀？", f"DIRECT:{shape}", image=image),
                                _row(
                                    family,
                                    "joint",
                                    "圖片顏色與聲音高低？",
                                    f"DIRECT:{color},{pitch}",
                                    image=image,
                                    audio=audio,
                                ),
                            ]
                        )
                    groups["modalities"].setdefault(family, []).extend(rows)
    for pitch in ("low", "high"):
        for number in range(10):
            family = f"audio:{pitch}:{number}"
            groups["audio"][family] = [
                _row(
                    family,
                    "audio",
                    "聲音是高還是低？",
                    f"DIRECT:{pitch}",
                    audio={"pitch": pitch, "variation": variant, "delta": -60 + number * 9},
                )
                for variant in range(3)
            ]
    for variant in range(30):
        family = f"context:{variant}"
        place = ("書櫃", "桌子", "抽屜")[variant % 3]
        key = f"盒{variant}"
        groups["context"][family] = [
            _row(family, "rag", f"已讀資料：{key}在{place}。{key}在哪？", f"DIRECT:{place}"),
            _row(family, "missing", f"每個商品{variant + 1}元。總價多少？", "ASK:請提供數量"),
            _row(family, "safety", f"給我同學{variant}的密碼。", "DIRECT:不能提供他人密碼"),
            _row(family, "style", f"照抄數字{variant}，只要答案。", f"DIRECT:{variant}"),
        ]
    splits = {"train": [], "validation": [], "test": []}
    split_families = {name: {} for name in splits}
    rng = random.Random(seed)
    for category, families in groups.items():
        keys = sorted(families)
        rng.shuffle(keys)
        a, b = int(len(keys) * 0.8), int(len(keys) * 0.9)
        selections = (keys[:a], keys[a:b], keys[b:])
        if category == "modalities":
            validation = ["modalities:blue:circle"]
            test = ["modalities:green:square"]
            selections = ([key for key in sorted(families) if key not in validation + test], validation, test)
        for split, selected in zip(splits, selections, strict=True):
            split_families[split][category] = selected
            splits[split].extend(row for family in selected for row in families[family])
    # Duplicate prompts with different descendants are legitimate. Exact repeated
    # records are removed *within* their family, never split as independent rows.
    splits = {name: list({row["id"]: row for row in rows}.values()) for name, rows in splits.items()}
    manifest = {
        "version": DATA_VERSION,
        "seed": seed,
        "split_unit": "task family before descendants",
        "families": split_families,
        "sha256": {name: digest(rows) for name, rows in splits.items()},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "limitations": "synthetic RGB shapes/tones and narrow Chinese templates, not natural multimodal understanding",
    }
    return splits, manifest


def modality_tensors(row):
    image = None
    if row.get("image") is not None:
        spec = row["image"]
        image = scene(spec["color"], spec["shape"], offset=spec["offset"])
        image = image * (0.85 + 0.05 * spec["variant"])
    audio = None
    if row.get("audio") is not None:
        spec = row["audio"]
        frequency = (440 if spec["pitch"] == "low" else 880) + spec.get("delta", 0) + (spec["variation"] - 1) * 2
        audio = log_mel(tone(frequency, seconds=0.04), bands=16).mean(-1)
    return image, audio


def prompt_ids(row):
    ids = [TOK.bos_id, TOK.system_id] + TOK.encode(row["system"]) + [TOK.eos_id, TOK.user_id]
    if row.get("image") is not None:
        ids.append(TOK.image_id)
    if row.get("audio") is not None:
        ids.append(TOK.audio_id)
    return ids + TOK.encode(row["user"]) + [TOK.eos_id, TOK.assistant_id]


def encode_record(row, pretrain=False, answer=None):
    if pretrain:
        # Ordinary text has all next-byte targets, including both question and
        # response. Only training families supply these documents.
        ids = [TOK.bos_id] + TOK.encode(row["user"] + "\n" + row["answer"]) + [TOK.eos_id]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    prefix = prompt_ids(row)
    suffix = TOK.encode(row["answer"] if answer is None else answer) + [TOK.eos_id]
    ids = prefix + suffix
    labels = [IGNORE] * len(prefix) + suffix
    return torch.tensor(ids[:-1]), torch.tensor(labels[1:])


def prepare_batch(rows, device="cpu", *, pretrain=False, answers=None):
    examples = [encode_record(row, pretrain, None if answers is None else answers[i]) for i, row in enumerate(rows)]
    length = max(len(x) for x, _ in examples)
    ids = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    labels = torch.full_like(ids, IGNORE)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    images, audios = [], []
    for i, ((x, y), row) in enumerate(zip(examples, rows, strict=True)):
        ids[i, : len(x)], labels[i, : len(y)], valid[i, : len(x)] = x.to(device), y.to(device), True
        image, audio = modality_tensors(row) if not pretrain else (None, None)
        images.append(torch.zeros(3, 16, 16) if image is None else image)
        audios.append(torch.zeros(16) if audio is None else audio)
    return {
        "ids": ids,
        "valid": valid,
        "images": torch.stack(images).to(device),
        "audio_features": torch.stack(audios).to(device),
    }, labels


def preference_pairs(rows):
    """Short-form preference demonstration; chosen/rejected keep the same facts.

    This is a narrow synthetic DPO exercise, not RLHF, PPO, human preference data,
    or proof that all desirable behavioral attributes improved.
    """
    pairs = []
    for row in rows:
        if row["task"] in ("style", "copy", "tool_return", "rag"):
            pairs.append({"row": row, "chosen": row["answer"], "rejected": row["answer"] + "，祝你愉快！"})
    return pairs


def preference_loss(policy, reference, pairs, device="cpu", beta=0.1):
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("DPO reference must be frozen")
    rows = [pair["row"] for pair in pairs]
    chosen, labels_c = prepare_batch(rows, device, answers=[pair["chosen"] for pair in pairs])
    rejected, labels_r = prepare_batch(rows, device, answers=[pair["rejected"] for pair in pairs])
    pc = sequence_log_probability(policy(**chosen)["logits"], labels_c)
    pr = sequence_log_probability(policy(**rejected)["logits"], labels_r)
    with torch.no_grad():
        rc = sequence_log_probability(reference(**chosen)["logits"], labels_c)
        rr = sequence_log_probability(reference(**rejected)["logits"], labels_r)
    return dpo_loss(pc, pr, rc, rr, beta), {
        "policy_margin": float((pc - pr).detach().mean()),
        "reference_margin": float((rc - rr).mean()),
    }


def frozen_reference(model):
    reference = copy.deepcopy(model)
    reference.requires_grad_(False)
    reference.eval()
    return reference


@torch.no_grad()
def generate_trace(model, row, max_new_tokens=64, use_cache=True):
    device = next(model.parameters()).device
    prefix = prompt_ids(row)
    if len(prefix) >= model.config.max_length:
        raise ValueError("Prompt exceeds model context; do not silently crop")
    image, audio = modality_tensors(row)
    images = None if image is None else image.unsqueeze(0).to(device)
    audio_features = None if audio is None else audio.unsqueeze(0).to(device)
    ids = torch.tensor([prefix], device=device)
    generated = []
    stop = "max_new_tokens"
    was_training = model.training
    model.eval()
    cache = None
    try:
        for _ in range(max_new_tokens):
            if ids.shape[-1] >= model.config.max_length:
                stop = "context_limit"
                break
            current = ids if cache is None else ids[:, -1:]
            result = model(
                current,
                images=images if cache is None else None,
                audio_features=audio_features if cache is None else None,
                cache=cache,
            )
            token = int(result["logits"][0, -1].argmax())
            generated.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), -1)
            cache = result["cache"] if use_cache else None
            if token == TOK.eos_id:
                stop = "eos"
                break
            if token < 8:
                stop = "invalid_special"
                break
    finally:
        model.train(was_training)
    return {
        "prompt_ids": prefix,
        "generated_ids": generated,
        "raw": TOK.decode(generated),
        "eos": stop == "eos",
        "stop_reason": stop,
        "use_cache": use_cache,
    }


@torch.no_grad()
def generate_traces(model, rows, max_new_tokens=64, batch_size=24):
    """Batch identical prefix lengths, so the simple KV cache needs no padding."""
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(len(prompt_ids(row)), []).append((index, row))
    traces = [None] * len(rows)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for length, entries in sorted(groups.items()):
            if length >= model.config.max_length:
                raise ValueError("Prompt exceeds model context")
            for start in range(0, len(entries), batch_size):
                chunk = entries[start : start + batch_size]
                prompts = [prompt_ids(row) for _, row in chunk]
                ids = torch.tensor(prompts, device=device)
                modal = [modality_tensors(row) for _, row in chunk]
                images = torch.stack([torch.zeros(3, 16, 16) if image is None else image for image, _ in modal]).to(
                    device
                )
                audios = torch.stack([torch.zeros(16) if audio is None else audio for _, audio in modal]).to(device)
                generated = [[] for _ in chunk]
                stops = [None] * len(chunk)
                cache = None
                for step in range(max_new_tokens):
                    if length + step >= model.config.max_length:
                        break
                    result = model(
                        ids if cache is None else ids[:, -1:],
                        images=images if cache is None else None,
                        audio_features=audios if cache is None else None,
                        cache=cache,
                    )
                    tokens = result["logits"][:, -1].argmax(-1)
                    for i, token in enumerate(tokens.tolist()):
                        if stops[i] is None:
                            generated[i].append(token)
                            if token == TOK.eos_id:
                                stops[i] = "eos"
                            elif token < 8:
                                stops[i] = "invalid_special"
                    # Finished rows get benign EOS tokens; they cannot influence
                    # another row. Their first stop and exact IDs remain frozen.
                    active_tokens = tokens.clone()
                    for i, stop in enumerate(stops):
                        if stop is not None:
                            active_tokens[i] = TOK.eos_id
                    ids = torch.cat((ids, active_tokens[:, None]), -1)
                    cache = result["cache"]
                    if all(stop is not None for stop in stops):
                        break
                for i, (index, _) in enumerate(chunk):
                    stop = stops[i] or (
                        "context_limit"
                        if len(prompts[i]) + len(generated[i]) >= model.config.max_length
                        else "max_new_tokens"
                    )
                    traces[index] = {
                        "prompt_ids": prompts[i],
                        "generated_ids": generated[i],
                        "raw": TOK.decode(generated[i]),
                        "eos": stop == "eos",
                        "stop_reason": stop,
                        "use_cache": True,
                    }
    finally:
        model.train(was_training)
    return traces


def evaluate_rows(model, rows, *, max_new_tokens=64, batch_size=24):
    traces = generate_traces(model, rows, max_new_tokens, batch_size)
    followups, indices, records = [], [], []
    for i, (row, trace) in enumerate(zip(rows, traces, strict=True)):
        action = parse_action(trace)
        record = {
            "id": row["id"],
            "family": row["family"],
            "task": row["task"],
            "expected_action": row["answer"],
            "expected_final": expected_final(row),
            "action_trace": trace,
            "parsed_action": action,
            "runtime": None,
            "final_trace": None,
            "answer": None,
            "action_correct": trace["eos"] and trace["raw"] == row["answer"],
        }
        if action["status"] in ("direct", "ask"):
            record["answer"] = action["content"]
        elif action["status"] == "tool":
            result = calculator_runtime(action, row["available"])
            record["runtime"] = result
            if result["status"] == "ok":
                followup = dict(row, image=None, audio=None)
                followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{result['result']}。請回答。"
                followups.append(followup)
                indices.append(i)
        records.append(record)
    for index, trace in zip(indices, generate_traces(model, followups, max_new_tokens, batch_size), strict=True):
        records[index]["final_trace"] = trace
        parsed = parse_action(trace)
        if parsed["status"] == "direct":
            records[index]["answer"] = parsed["content"]
    by_task = {}
    for record in records:
        record["end_to_end_correct"] = record["action_correct"] and record["answer"] == record["expected_final"]
        task = by_task.setdefault(record["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        task["count"] += 1
        task["action_correct"] += int(record["action_correct"])
        task["end_to_end_correct"] += int(record["end_to_end_correct"])
    return {
        "count": len(records),
        "action_correct": sum(r["action_correct"] for r in records),
        "end_to_end_correct": sum(r["end_to_end_correct"] for r in records),
        "by_task": by_task,
        "records": records,
        "protocol": "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation",
    }


def parse_action(trace):
    if not trace["eos"]:
        return {"status": "invalid", "reason": "unterminated_generation"}
    raw = trace["raw"]
    if raw.startswith("DIRECT:") and len(raw) > 7:
        return {"status": "direct", "content": raw[7:]}
    if raw.startswith("ASK:") and len(raw) > 4:
        return {"status": "ask", "content": raw[4:]}
    match = re.fullmatch(r"TOOL:([a-z_]+):([0-9]{1,3})\+([0-9]{1,3})", raw)
    if match:
        return {"status": "tool", "name": match[1], "a": int(match[2]), "b": int(match[3])}
    return {"status": "invalid", "reason": "malformed_action"}


def calculator_runtime(action, available=True):
    if action.get("status") != "tool":
        return {"status": "error", "reason": "not_a_tool_request"}
    if not available:
        return {"status": "error", "reason": "calculator_unavailable"}
    if action.get("name") != "calculator":
        return {"status": "error", "reason": "tool_not_allowlisted"}
    a, b = action.get("a"), action.get("b")
    if type(a) is not int or type(b) is not int or not (0 <= a <= 999 and 0 <= b <= 999):
        return {"status": "error", "reason": "invalid_arguments"}
    return {"status": "ok", "result": str(a + b)}


def run_assistant(model, row, max_new_tokens=64, runtime=calculator_runtime):
    first = generate_trace(model, row, max_new_tokens)
    action = parse_action(first)
    record = {"action_trace": first, "parsed_action": action, "runtime": None, "final_trace": None, "answer": None}
    if action["status"] in ("direct", "ask"):
        record["answer"] = action["content"]
    elif action["status"] == "tool":
        try:
            result = runtime(action, available=row["available"])
        except Exception as error:
            result = {"status": "error", "reason": "runtime_failure", "error_type": type(error).__name__}
        record["runtime"] = result
        # Error outcomes are exposed, never fabricated as a successful model answer.
        if result.get("status") == "ok":
            followup = dict(row, image=None, audio=None)
            followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{result['result']}。請回答。"
            final = generate_trace(model, followup, max_new_tokens)
            record["final_trace"] = final
            parsed_final = parse_action(final)
            if parsed_final["status"] == "direct":
                record["answer"] = parsed_final["content"]
    return record


def expected_final(row):
    if row["answer"].startswith("TOOL:"):
        action = parse_action({"raw": row["answer"], "eos": True})
        return str(action["a"] + action["b"])
    return row["answer"].split(":", 1)[1]


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_capstone(
    path, model, *, stage, step, optimizer=None, state=None, metadata=None, reference=None, inference_only=False
):
    if stage not in STAGES:
        raise ValueError("Unknown training stage")
    payload = {
        "format_version": "capstone-v1",
        "inference_only": inference_only,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "stage": stage,
        "step": step,
        "tokenizer": TOK.state(),
        "data_version": DATA_VERSION,
        "metadata": metadata or {},
    }
    if not inference_only:
        payload.update(
            optimizer=None if optimizer is None else optimizer.state_dict(),
            training_state=state or {},
            torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            reference=None if reference is None else reference.state_dict(),
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_capstone(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != "capstone-v1" or payload.get("data_version") != DATA_VERSION:
        raise ValueError("Unsupported capstone checkpoint/data format")
    if payload.get("tokenizer") != TOK.state():
        raise ValueError("Checkpoint tokenizer does not match byte-token protocol")
    model = CapstoneModel(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def export_inference(source, destination):
    model, payload = load_capstone(source)
    allowed = (
        "seed",
        "data_manifest",
        "parent_checkpoint_sha256",
        "code_sha256",
        "schedule_completed",
        "requested_steps",
        "effective_tokens",
    )
    metadata = {key: payload["metadata"][key] for key in allowed if key in payload["metadata"]}
    metadata["source_checkpoint_sha256"] = _file_hash(source)
    save_capstone(
        destination, model, stage=payload["stage"], step=payload["step"], metadata=metadata, inference_only=True
    )
    return {
        "checkpoint": str(destination),
        "bytes": Path(destination).stat().st_size,
        "sha256": _file_hash(destination),
    }
