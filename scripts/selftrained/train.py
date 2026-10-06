#!/usr/bin/env python3
"""有界、可續跑的從零階段訓練。best 只按 validation loss 選擇。"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiny_perceptron.selftrained.dataset import (  # noqa: E402
    PREPROCESS_VERSION,
    RecordEncoder,
    asset_fingerprints,
    data_fingerprints,
    file_sha256,
    perception_loss,
    read_records,
    train_tokenizer,
)

CHECKPOINT_SCHEMA = "selftrained-random-v1"
STAGES = ("pretrain", "sft", "vision", "ocr", "audio", "joint")
SAMPLING_MODES = ("bucket", "task-family")
LANGUAGE_OBJECTIVE_VERSION = "selftrained-language-objective-v1"
NATIVE_LANGUAGE_OBJECTIVE_VERSION = "selftrained-language-objective-v2"
TASK_FAMILIES = ("text", "tools", "vision", "ocr", "voice")
# task-family tools cycle: 12/16 slots are numeric call/reply training.
TOOL_TASK_MULTIPLICITIES = {
    "tool_call": 6,
    "tool_reply": 6,
    "tool_unavailable": 1,
    "tool_missing": 1,
    "tool_concept": 1,
    "tool_unsupported": 1,
}
# Vision.projection 是 supervised head 上游；固定它才能保留感知 logits。
PERCEPTION_BRIDGE_PREFIXES = (
    "vision_encoder.symbol_projection.",
    "vision_encoder.coordinates.",
    "vision_encoder.slot_position.",
    "vision_encoder.norm.",
    "ocr_encoder.projection.",
    "ocr_encoder.symbol_projection.",
    "ocr_encoder.column_position.",
    "ocr_encoder.norm.",
    "audio_encoder.projection.",
    "audio_encoder.symbol_projection.",
    "audio_encoder.time_position.",
    "audio_encoder.norm.",
)
TASK_FAMILY_BY_TASK = {
    "text": "text",
    "text_pretrain": "text",
    "tool_call": "tools",
    "tool_reply": "tools",
    "tool_unavailable": "tools",
    "tool_missing": "tools",
    "tool_concept": "tools",
    "tool_unsupported": "tools",
    "vision_clothing": "vision",
    "vision_relation": "vision",
    "ocr": "ocr",
    "voice_qa": "voice",
    "voice_topic_continuation": "voice",
}


def tokenizer_sha(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer.to_dict(), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def stage_records(records, stage, split):
    def included(record):
        task = record["task"]
        if stage == "vision":
            return task.startswith("vision")
        if stage == "ocr":
            return task == "ocr"
        if stage == "audio":
            return task.startswith(("voice", "audio"))
        if stage == "pretrain":
            return task in ("text", "text_pretrain") and not (record.get("image") or record.get("audio"))
        if stage == "sft":
            return not (record.get("image") or record.get("audio"))
        return True

    return [
        record
        for record in records
        if record["split"] == split and included(record) and not record.get("evaluation_only", False)
    ]


class BalancedSampler:
    """独立 RNG 讓 Dense/MoE 使用相同順序，不受模型初始化抽樣影響。"""

    def __init__(self, records, seed, mode="bucket"):
        if mode not in SAMPLING_MODES:
            raise ValueError(f"未知 sampling mode: {mode}")
        self.mode = mode
        self.by_task = {}
        bucket_families = {}
        bucket_tasks = {}
        for record in records:
            intent = record.get("supervision", {}).get("intent", "")
            bucket = record["task"] + (":" + intent if intent else "")
            self.by_task.setdefault(bucket, []).append(record)
            if mode == "task-family":
                if record["task"] not in TASK_FAMILY_BY_TASK:
                    raise ValueError(f"task-family 不支援未知 task: {record['task']}")
                bucket_families[bucket] = TASK_FAMILY_BY_TASK[record["task"]]
                bucket_tasks[bucket] = record["task"]
        self.tasks = sorted(self.by_task)
        self.by_family = (
            {
                family: [
                    bucket
                    for bucket in self.tasks
                    if bucket_families.get(bucket) == family
                    for _ in range(TOOL_TASK_MULTIPLICITIES[bucket_tasks[bucket]] if family == "tools" else 1)
                ]
                for family in TASK_FAMILIES
            }
            if mode == "task-family"
            else {}
        )
        if mode == "task-family" and any(not tasks for tasks in self.by_family.values()):
            raise ValueError("task-family 需要完整 text/tools/vision/ocr/voice 五種資料")
        self.family_draws = {family: 0 for family in self.by_family}
        self.generator = torch.Generator().manual_seed(seed)
        self.draws = 0

    def policy(self):
        return {
            "mode": self.mode,
            "family_draw_weights": dict.fromkeys(TASK_FAMILIES, 1) if self.mode == "task-family" else {},
            "family_cycle": list(TASK_FAMILIES) if self.mode == "task-family" else [],
            "within_family_bucket_cycles": {family: list(buckets) for family, buckets in self.by_family.items()},
            "within_family_bucket_multiplicities": {
                family: {bucket: buckets.count(bucket) for bucket in sorted(set(buckets))}
                for family, buckets in self.by_family.items()
            },
            "tools_task_multiplicities": TOOL_TASK_MULTIPLICITIES.copy() if self.mode == "task-family" else {},
            "record_sampling": "uniform_with_replacement_per_bucket",
        }

    def batch(self, size):
        result = []
        for _ in range(size):
            # round robin 任務、任務內有放回抽樣；節省多模態資料複製。
            if self.mode == "task-family":
                family = TASK_FAMILIES[self.draws % len(TASK_FAMILIES)]
                family_tasks = self.by_family[family]
                task = family_tasks[self.family_draws[family] % len(family_tasks)]
                self.family_draws[family] += 1
            else:
                task = self.tasks[self.draws % len(self.tasks)]
            candidates = self.by_task[task]
            index = int(torch.randint(len(candidates), (), generator=self.generator))
            result.append(candidates[index])
            self.draws += 1
        return result

    def state_dict(self):
        state = {
            "draws": self.draws,
            "generator": self.generator.get_state(),
            "mode": self.mode,
            "family_draws": self.family_draws.copy(),
        }
        if self.mode == "task-family":
            state["sampler_policy"] = self.policy()
        return state

    def load_state_dict(self, state):
        if state.get("mode", "bucket") != self.mode:
            raise ValueError("sampler resume sampling mode 不同")
        if self.mode == "task-family":
            if state.get("sampler_policy") != self.policy():
                raise ValueError("sampler resume task-family policy 不同或缺失")
            expected = {
                family: state["draws"] // len(TASK_FAMILIES) + int(index < state["draws"] % len(TASK_FAMILIES))
                for index, family in enumerate(TASK_FAMILIES)
            }
            if state.get("family_draws") != expected:
                raise ValueError("sampler resume family counters 與實際 draws 不符")
            self.family_draws = state["family_draws"].copy()
        self.draws = state["draws"]
        self.generator.set_state(state["generator"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def export_inference(output_dir, *, required=False):
    """公開檔只含選定模型與契約，不帶 optimizer/RNG 或訓練標籤。"""
    selected = output_dir / "best.pt"
    if not selected.exists():
        return False
    try:
        from safetensors.torch import save_file
    except ImportError:
        if required:
            raise RuntimeError("--export-inference 需要 safetensors 純權重格式套件") from None
        return False
    from tiny_perceptron.selftrained.dataset import file_sha256
    from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    # tied embeddings 的兩個名稱保留；clone 避免 safe writer 的共享 storage 歧義。
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in checkpoint["model"].items()}
    save_file(
        tensors,
        str(output_dir / "model.safetensors"),
        metadata={"origin": "all-neural-weights-random", "schema": CHECKPOINT_SCHEMA},
    )
    (output_dir / "model-config.json").write_text(json.dumps(checkpoint["config"], indent=2) + "\n", encoding="utf-8")
    CharacterTokenizer.from_dict(checkpoint["tokenizer"]).save(output_dir / "tokenizer.json")
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "selected_checkpoint_sha256": file_sha256(selected),
        "files": {
            name: file_sha256(output_dir / name)
            for name in ("model.safetensors", "model-config.json", "tokenizer.json")
        },
        "stage": checkpoint["stage"],
        "selected_step": checkpoint["step"],
        "origin": checkpoint["origin"],
        "preprocess_version": checkpoint["preprocess_version"],
        "data_sha256": checkpoint["data_sha256"],
        "asset_sha256": checkpoint["asset_sha256"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "freeze_perception_backbones": checkpoint.get("training_options", {}).get("freeze_perception_backbones", False),
        "sampling_mode": checkpoint.get("training_options", {}).get("sampling_mode", "bucket"),
        **checkpoint_language_objective(checkpoint),
        "selection": "validation_loss",
    }
    (output_dir / "inference-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return True


def set_trainable(model, stage, freeze_perception_backbones=False):
    if freeze_perception_backbones and stage != "joint":
        raise ValueError("--freeze-perception-backbones 只可用於 joint")
    prefix = {"vision": "vision_encoder.", "ocr": "ocr_encoder.", "audio": "audio_encoder."}.get(stage)
    for name, parameter in model.named_parameters():
        if freeze_perception_backbones:
            parameter.requires_grad = name.startswith(("lm.", *PERCEPTION_BRIDGE_PREFIXES))
        else:
            parameter.requires_grad = name.startswith(prefix) if prefix else stage == "joint" or name.startswith("lm.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"stage={stage} 沒有可訓練參數，檢查模型 prefix")
    return trainable


def validate_loss_weights(stage, tool_loss_weight=1.0, numeric_run_loss_weight=1.0, native_voice_loss_weight=1.0):
    for name, value in (
        ("tool_loss_weight", tool_loss_weight),
        ("numeric_run_loss_weight", numeric_run_loss_weight),
        ("native_voice_loss_weight", native_voice_loss_weight),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 1:
            raise ValueError(f"{name} 必須是 finite number >= 1，不能是 boolean")
    weighted = tool_loss_weight != 1 or numeric_run_loss_weight != 1 or native_voice_loss_weight != 1
    if weighted and stage != "joint":
        raise ValueError("nondefault language loss weights 只可用於 joint")
    return weighted


def language_objective_metadata(options):
    tool = options.get("tool_loss_weight", 1.0)
    numeric = options.get("numeric_run_loss_weight", 1.0)
    native = options.get("native_voice_loss_weight", 1.0)
    validate_loss_weights(options.get("stage", "joint"), tool, numeric, native)
    policy = {
        "version": LANGUAGE_OBJECTIVE_VERSION,
        "training_only": True,
        "legacy_default_branch": "reuse_original_language_loss_scalar_and_graph",
        "tool_rows": ["tool_call", "tool_reply:supervision.replay_kind=actual_executor"],
        "excluded_rows": ["evaluation_only", "supervision.protocol_fixture"],
        "row_scope": "all_existing_supervised_assistant_targets_in_eligible_row",
        "numeric_targets": "ASCII_digits_union_immediate_supervised_nondigit_boundary_or_EOS_union_minus_if_next_supervised_digit",
        "alignment": "RecordEncoder_already_shifted_labels; ignored_gap_breaks_adjacency",
        "combination": "tool_row_multiplier_times_numeric_multiplier; ignored_labels_zero",
        "normalization": "sum(weight_times_token_CE)/sum(weight)",
        "weighted_reduction_dtype": "float32",
        "validation_and_selection": "original_unweighted_objective; validation_loss",
    }
    metadata = {"tool_loss_weight": tool, "numeric_run_loss_weight": numeric, "language_objective_policy": policy}
    if native != 1:
        policy.update(
            version=NATIVE_LANGUAGE_OBJECTIVE_VERSION,
            native_one_branch="exact_existing_v1_scalar_graph_metadata_and_logs",
            native_voice_rows={
                "split": "train",
                "tasks": ["voice_qa", "voice_topic_continuation"],
                "augmentation": "key_absent",
                "excluded": ["evaluation_only", "supervision.protocol_fixture"],
            },
            native_row_scope="all_existing_supervised_assistant_targets_including_public_history_confirmations",
            combination="tool_row_multiplier_times_numeric_multiplier_times_native_voice_row_multiplier; ignored_labels_zero",
        )
        metadata["native_voice_loss_weight"] = native
    return metadata


def checkpoint_language_objective(source):
    options = source.get("training_options", {})
    metadata = language_objective_metadata({**options, "stage": source.get("stage", "joint")})
    if any(
        key in source and source[key] != metadata.get(key, 1.0)
        for key in ("tool_loss_weight", "numeric_run_loss_weight", "native_voice_loss_weight")
    ):
        raise ValueError("checkpoint language objective coefficients 與 training_options 不符")
    policies = [
        value
        for value in (options.get("language_objective_policy"), source.get("language_objective_policy"))
        if value is not None
    ]
    if any(value != metadata["language_objective_policy"] for value in policies):
        raise ValueError("checkpoint language objective version/policy 不同")
    if not policies and (
        metadata["tool_loss_weight"] != 1
        or metadata["numeric_run_loss_weight"] != 1
        or metadata.get("native_voice_loss_weight", 1) != 1
    ):
        raise ValueError("nondefault checkpoint 缺少 language objective version/policy")
    # A legacy missing policy/coefficients represents the original 1/1 objective only.
    return metadata


def validate_resume_language_objective(source, options):
    if checkpoint_language_objective(source) != language_objective_metadata(options):
        raise ValueError("resume 改變 language objective coefficients/version/policy；請 init-checkpoint 開始新 joint")


def genuine_tool_loss_row(record):
    supervision = record.get("supervision", {})
    if record.get("evaluation_only", False) or supervision.get("protocol_fixture", False):
        return False
    return record["task"] == "tool_call" or (
        record["task"] == "tool_reply" and supervision.get("replay_kind") == "actual_executor"
    )


def native_voice_loss_row(record):
    return (
        record.get("split") == "train"
        and record.get("task") in ("voice_qa", "voice_topic_continuation")
        and "augmentation" not in record
        and not record.get("evaluation_only", False)
        and not record.get("supervision", {}).get("protocol_fixture", False)
    )


def numeric_run_loss_mask(labels, tokenizer):
    valid = labels != -100
    digit_ids = [tokenizer.character_ids[c] for c in "0123456789" if c in tokenizer.character_ids]
    digits = torch.isin(labels, torch.tensor(digit_ids, dtype=labels.dtype, device=labels.device)) & valid
    boundary = torch.zeros_like(valid)
    boundary[:, 1:] = valid[:, 1:] & digits[:, :-1] & ~digits[:, 1:]
    minus = torch.zeros_like(valid)
    minus_id = tokenizer.character_ids.get("-")
    if minus_id is not None:
        minus[:, :-1] = valid[:, :-1] & (labels[:, :-1] == minus_id) & digits[:, 1:]
    return (digits | boundary | minus) & valid


def language_loss_weights(
    labels, records, tokenizer, tool_loss_weight, numeric_run_loss_weight, native_voice_loss_weight=1.0
):
    if labels.ndim != 2 or labels.shape[0] != len(records):
        raise ValueError("records 與 already-shifted labels 的 batch dimension 不同")
    row_weights = torch.tensor(
        [tool_loss_weight if genuine_tool_loss_row(record) else 1.0 for record in records],
        dtype=torch.float32,
        device=labels.device,
    )[:, None]
    numeric = torch.where(numeric_run_loss_mask(labels, tokenizer), numeric_run_loss_weight, 1.0).float()
    weights = row_weights * numeric * (labels != -100).float()
    if native_voice_loss_weight != 1:
        native_rows = torch.tensor(
            [native_voice_loss_weight if native_voice_loss_row(record) else 1.0 for record in records],
            dtype=torch.float32,
            device=labels.device,
        )[:, None]
        weights = weights * native_rows
    return weights


def weighted_language_loss(
    output, labels, records, tokenizer, tool_loss_weight=1.0, numeric_run_loss_weight=1.0, native_voice_loss_weight=1.0
):
    if tool_loss_weight == 1 and numeric_run_loss_weight == 1 and native_voice_loss_weight == 1:
        return output.get("language_loss", output.get("loss"))
    logits = output["logits"]
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits 必須與 existing already-shifted labels 對齊")
    weights = language_loss_weights(
        labels, records, tokenizer, tool_loss_weight, numeric_run_loss_weight, native_voice_loss_weight
    )
    mass = weights.sum(dtype=torch.float32)
    if not bool(torch.isfinite(mass) & (mass > 0)):
        raise ValueError("沒有 finite positive supervised token mass")
    ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(labels)
    return (ce * weights).sum(dtype=torch.float32) / mass


def validate_weighted_source(source, args):
    if source is None or source.get("origin", {}).get("kind") != "all-neural-weights-random":
        raise ValueError("nondefault joint 必須承接 own random-initialized checkpoint")
    checkpoint_language_objective(source)
    if not args.freeze_perception_backbones or args.sampling_mode != "task-family":
        raise ValueError("nondefault joint 必須保留 frozen perception backbones 與 task-family sampling")
    old = source.get("training_options", {})
    if (
        old.get("freeze_perception_backbones", False) != args.freeze_perception_backbones
        or old.get("sampling_mode", "bucket") != args.sampling_mode
    ):
        raise ValueError("nondefault joint 必須保留 source backbone/sampling options")
    if source["config"]["max_length"] != args.context:
        raise ValueError("nondefault joint 必須保留 source context/config")
    if args.init_checkpoint and (
        source.get("stage") != "joint" or source.get("checkpoint_kind") != "selected_validation_best"
    ):
        raise ValueError("nondefault 新 joint 必須 init own joint selected_validation_best；不能從 latest 改 objective")


def validate_native_source(source_path, source, args):
    """Bind the loaded selected checkpoint to its actual completed Volume source metadata."""
    from scripts.selftrained.modal_runner import native_joint_source_gate

    evidence = native_joint_source_gate(source_path, args.architecture, target_job=vars(args))
    training, inference, job = evidence["training"], evidence["inference"], evidence["job"]
    old = source["training_options"]
    source_metadata = checkpoint_language_objective(source)
    if (
        source_metadata
        != language_objective_metadata({"stage": "joint", "tool_loss_weight": 4, "numeric_run_loss_weight": 4})
        or source["step"] != inference["selected_step"]
        or source["preprocess_version"] != inference.get("preprocess_version")
        or source["stage_history"] != training["stage_history"]
        or any(
            source[key] != training[key]
            for key in ("schema", "config", "data_sha256", "asset_sha256", "tokenizer_sha256", "origin")
        )
        or any(training.get(key) != value for key, value in source_metadata.items())
    ):
        raise ValueError("native loaded selected checkpoint differs from completed4/4 metadata")
    for name, default in (
        ("batch_size", 16),
        ("seed", 20261006),
        ("learning_rate", 0.001),
        ("weight_decay", 0.01),
        ("perception_weight", 1.0),
        ("router_weight", 0.01),
        ("freeze_perception_backbones", False),
        ("sampling_mode", "bucket"),
        ("context", 512),
        ("steps", 300),
        ("eval_every", 100),
        ("save_every", 100),
    ):
        expected = job.get(name, default)
        if name in ("learning_rate", "weight_decay", "perception_weight", "router_weight"):
            expected = float(expected)
        if old.get(name, default) != expected or (name in training and training[name] != expected):
            raise ValueError(f"native source checkpoint/receipt/execution options differ: {name}")
        if name not in ("steps", "eval_every", "save_every") and getattr(args, name) != old.get(name, default):
            raise ValueError(f"native fresh stage must preserve source {name}")
    return evidence["binding"]


def objective(
    model,
    encoder,
    records,
    stage,
    perception_weight,
    router_weight,
    tool_loss_weight=1.0,
    numeric_run_loss_weight=1.0,
    native_voice_loss_weight=1.0,
):
    weighted = validate_loss_weights(stage, tool_loss_weight, numeric_run_loss_weight, native_voice_loss_weight)
    batch = encoder.batch(records, pretrain=stage == "pretrain")
    if stage in ("vision", "ocr", "audio"):
        batch["labels"] = None
    output = model(**batch)
    head_loss, head_metrics = perception_loss(output.get("perception", []), records)
    unweighted_language_loss = output.get("language_loss", output.get("loss"))
    language_loss = unweighted_language_loss
    if weighted:
        language_loss = weighted_language_loss(
            output,
            batch["labels"],
            records,
            encoder.tokenizer,
            tool_loss_weight,
            numeric_run_loss_weight,
            native_voice_loss_weight,
        )
    if stage in ("vision", "ocr", "audio"):
        if head_loss is None:
            raise ValueError(f"{stage} 沒有實際感知監督")
        loss = head_loss
    else:
        if language_loss is None:
            raise ValueError("沒有 assistant target loss")
        loss = language_loss
        if head_loss is not None:
            loss = loss + perception_weight * head_loss
        aux = output.get("aux_loss")
        if aux is not None:
            loss = loss + router_weight * aux
    detail = {
        "language_loss": float(language_loss.detach()) if language_loss is not None else None,
        "perception_loss": float(head_loss.detach()) if head_loss is not None else None,
        "tokens": int(batch["attention_mask"].sum()),
        "target_tokens": int((batch["labels"] != -100).sum()) if batch["labels"] is not None else 0,
        "head_metrics": head_metrics,
    }
    if weighted:
        detail.pop("language_loss")
        detail["weighted_language_loss"] = float(language_loss.detach())
        detail["unweighted_language_loss"] = float(unweighted_language_loss.detach())
    if native_voice_loss_weight != 1:
        labels = batch["labels"]
        weights = language_loss_weights(
            labels, records, encoder.tokenizer, tool_loss_weight, numeric_run_loss_weight, native_voice_loss_weight
        )
        ce = torch.nn.functional.cross_entropy(
            output["logits"].float().reshape(-1, output["logits"].shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        detail["tool_numeric_language_loss"] = float(
            weighted_language_loss(
                output, labels, records, encoder.tokenizer, tool_loss_weight, numeric_run_loss_weight
            ).detach()
        )
        detail["total_weighted_language_mass"] = float(weights.sum())
        for name, native_group in (("native_voice", True), ("augmented_voice", False)):
            eligible = [
                native_voice_loss_row(record)
                if native_group
                else (
                    record.get("split") == "train"
                    and record.get("task") in ("voice_qa", "voice_topic_continuation")
                    and "augmentation" in record
                    and not record.get("evaluation_only", False)
                    and not record.get("supervision", {}).get("protocol_fixture", False)
                )
                for record in records
            ]
            mask = torch.tensor(eligible, device=labels.device)[:, None] & labels.ne(-100)
            count = int(mask.sum())
            detail[f"{name}_target_tokens"] = count
            detail[f"{name}_unweighted_language_loss"] = float(ce[mask].mean().detach()) if count else None
            detail[f"{name}_weighted_mass"] = float(weights[mask].sum())
    return loss, detail


@torch.no_grad()
def validation_loss(model, encoder, records, stage, batch_size, perception_weight, router_weight):
    model.eval()
    total, count = 0.0, 0
    per_task = {}
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        loss, _ = objective(model, encoder, batch_records, stage, perception_weight, router_weight)
        value = float(loss)
        total += value * len(batch_records)
        count += len(batch_records)
        for record in batch_records:
            per_task.setdefault(record["task"], {"count": 0, "batch_loss_sum": 0.0})
            per_task[record["task"]]["count"] += 1
            per_task[record["task"]]["batch_loss_sum"] += value
    model.train()
    if count == 0:
        raise ValueError("必須有 validation 列，不能以 train/test 選 checkpoint")
    return total / count, per_task


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--records", action="append", required=True, help="完整 JSONL；可重複，每階段用相同資料檔")
    result.add_argument("--asset-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--architecture", choices=("moe", "dense"), default="moe")
    result.add_argument("--stage", choices=STAGES, required=True)
    result.add_argument("--steps", type=int, required=True, help="此 stage 的總步數，resume 延續至此步")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--context", type=int, default=512)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--seed", type=int, default=20261006)
    result.add_argument("--config", help="SelftrainedConfig 的 JSON；vocab_size 自動設定")
    result.add_argument("--resume", help="本 stage 的完整 checkpoint（恢復 optimizer/RNG/抽樣）")
    result.add_argument("--init-checkpoint", help="自己前一 stage 的 checkpoint（新 optimizer）")
    result.add_argument("--eval-every", type=int, default=100)
    result.add_argument("--save-every", type=int, default=100)
    result.add_argument("--max-tokens", type=int, help="本 stage 輸入 token 預算；遇超額 batch 就停止")
    result.add_argument("--perception-weight", type=float, default=1.0)
    result.add_argument("--router-weight", type=float, default=0.01)
    result.add_argument(
        "--tool-loss-weight",
        type=float,
        default=1.0,
        help="joint training only；genuine tool rows 的 existing assistant targets multiplier，finite >= 1",
    )
    result.add_argument(
        "--numeric-run-loss-weight",
        type=float,
        default=1.0,
        help="joint training only；ASCII digits、next stopping target、leading minus multiplier，finite >= 1",
    )
    result.add_argument(
        "--native-voice-loss-weight",
        type=float,
        default=1.0,
        help="optional new joint only；native train voice rows without augmentation key；default preserves V1",
    )
    result.add_argument(
        "--freeze-perception-backbones",
        action="store_true",
        help="joint only：固定三種感知 backbone/head（含 Vision.projection），只訓練 LM 與 bridges",
    )
    result.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default="bucket",
        help="bucket 保留原 task:intent 平衡；task-family 只可 joint，五種 family 各占 20%",
    )
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument(
        "--export-inference", action="store_true", help="要求 safetensors 選定權重輸出；缺套件時明確失敗"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    native_weight = args.native_voice_loss_weight
    weighted = validate_loss_weights(args.stage, args.tool_loss_weight, args.numeric_run_loss_weight, native_weight)
    if native_weight == 1:
        # Preserve all old serialized args/options keys, explicit native1 included.
        del args.native_voice_loss_weight
    elif (
        (args.tool_loss_weight, args.numeric_run_loss_weight, native_weight) != (4, 1, 4)
        or args.steps > 4000
        or args.batch_size != 16
        or args.context != 512
        or args.learning_rate != 0.0002
        or not args.freeze_perception_backbones
        or args.sampling_mode != "task-family"
        or args.max_tokens is not None
    ):
        raise ValueError(
            "native voice candidate requires fixedtool4/numeric1/native4, <=4000 steps, batch16/context512/LR0.0002 and frozen task-family"
        )
    objective_metadata = language_objective_metadata(vars(args))
    args.language_objective_policy = objective_metadata["language_objective_policy"]
    if args.resume and args.init_checkpoint:
        raise ValueError("resume 與 init-checkpoint 不能同時使用")
    if weighted and not (args.resume or args.init_checkpoint):
        raise ValueError("nondefault joint 必須 init own selected joint checkpoint 或 exact same-objective resume")
    if args.stage != "joint" and (args.freeze_perception_backbones or args.sampling_mode != "bucket"):
        raise ValueError("freeze-perception-backbones / task-family sampling 只可用於 joint")
    if min(args.steps, args.batch_size, args.context, args.eval_every, args.save_every, args.threads) < 1:
        raise ValueError("步數、batch、context、interval、threads 必須為正")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
    from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

    all_records = read_records(args.records)
    hashes = data_fingerprints(args.records)
    asset_hashes = asset_fingerprints(all_records, args.asset_dir)
    records = stage_records(all_records, args.stage, "train")
    validation = stage_records(all_records, args.stage, "validation")
    if not records or not validation:
        raise ValueError(f"{args.stage} 需要 train 與 validation，不能拿 test 填補")
    source_path = args.resume or args.init_checkpoint
    source = torch.load(source_path, map_location="cpu", weights_only=False) if source_path else None
    if source is not None and source.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("只接受本從零管線 checkpoint")
    if source is not None and source.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError("checkpoint preprocess_version 與目前模態前處理不同")
    tokenizer = CharacterTokenizer.from_dict(source["tokenizer"]) if source else train_tokenizer(all_records)
    if source is not None and tokenizer_sha(tokenizer) != source["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer 指紋不匹配")
    if source and source["data_sha256"] != hashes:
        raise ValueError("資料指紋不同；請建立新實驗而非沿用 frozen stage")
    if source and source["asset_sha256"] != asset_hashes:
        raise ValueError("模態資產指紋不同")
    if weighted:
        validate_weighted_source(source, args)
    native_source_binding = None
    if native_weight != 1 and args.init_checkpoint:
        native_source_binding = validate_native_source(source_path, source, args)
    if args.resume:
        validate_resume_language_objective(source, vars(args))
        if native_weight != 1:
            for name in ("eval_every", "save_every"):
                if source["training_options"].get(name, 100) != getattr(args, name):
                    raise ValueError(f"native exact resume 改變 {name}")
    config_values = (
        dict(source["config"]) if source else (json.loads(Path(args.config).read_text()) if args.config else {})
    )
    if source and config_values["architecture"] != args.architecture:
        raise ValueError("Dense/MoE 不能混用權重")
    config_values.update(vocab_size=tokenizer.vocab_size, architecture=args.architecture, max_length=args.context)
    config = SelftrainedConfig(**config_values)
    model = LimitedAssistant(config).to(args.device)
    if source:
        model.load_state_dict(source["model"])
    parameters = set_trainable(model, args.stage, args.freeze_perception_backbones)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    encoder = RecordEncoder(tokenizer, args.asset_dir, args.context, args.device)
    # 先審查所有 stage train/val 長度；不讀 test 模態或 target。
    for record in records + validation:
        row = encoder.encode(record, pretrain=args.stage == "pretrain")
        if tokenizer.unk_id in row["input_ids"]:
            # validation 可有未知字，train 不得因 stage 增詞而靜默退化。
            if record["split"] == "train":
                raise ValueError(f"train 有 tokenizer 未見字: {record['id']}")
    sampler = BalancedSampler(records, args.seed, mode=args.sampling_mode)
    step, token_count, target_token_count, best_val = 0, 0, 0, math.inf
    history = list(source.get("stage_history", [])) if source else []
    origin = source.get("origin") if source else {"kind": "all-neural-weights-random", "seed": args.seed}
    if args.resume:
        if source["stage"] != args.stage or source["config"] != dataclasses.asdict(config):
            raise ValueError("resume 必須使用相同 stage/config")
        old = source["training_options"]
        for name in ("batch_size", "seed", "learning_rate", "weight_decay", "perception_weight", "router_weight"):
            if old[name] != getattr(args, name):
                raise ValueError(f"resume 改變 {name}；請使用 init-checkpoint 明確開始新階段")
        for name, default in (("freeze_perception_backbones", False), ("sampling_mode", "bucket")):
            if old.get(name, default) != getattr(args, name):
                raise ValueError(f"resume 改變 {name}；請使用 init-checkpoint 明確開始新階段")
        optimizer.load_state_dict(source["optimizer"])
        sampler.load_state_dict(source["sampler"])
        restore_rng(source["rng"])
        step, token_count, target_token_count = source["step"], source["tokens"], source["target_tokens"]
        best_val = source["best_validation_loss"]
    elif source:
        history.append(
            {
                "stage": source["stage"],
                "step": source["step"],
                "tokens": source["tokens"],
                "checkpoint": str(source_path),
                "checkpoint_sha256": file_sha256(source_path),
                "selection": "validation_loss"
                if source.get("checkpoint_kind") == "selected_validation_best"
                else "explicit_init_checkpoint",
            }
        )
        if weighted:
            initialization = {
                "source_stage": source["stage"],
                "loaded_step": source["step"],
                "source_checkpoint_sha256": file_sha256(source_path),
                "source_checkpoint_kind": source["checkpoint_kind"],
                "source_language_objective": checkpoint_language_objective(source),
                "new_language_objective": objective_metadata,
                "reset_state": {
                    "optimizer": "new",
                    "rng": "fresh_stage_seed_not_source_rng",
                    "sampler_draws": 0,
                    "family_draws": 0,
                    "stage_step": 0,
                    "tokens": 0,
                    "target_tokens": 0,
                },
            }
            if native_source_binding is not None:
                initialization["source_integrity"] = native_source_binding
                initialization["source_completed_steps"] = native_source_binding["source_completed_steps"]
            history[-1] = {**history[-1], "new_joint_initialization": initialization}
            origin = {**origin, "new_joint_initialization": initialization}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    stop_requested = False
    selected_checkpoint = source.get("selected_checkpoint") if args.resume else None
    if args.resume and selected_checkpoint is None and source["step"] == source.get("best_step"):
        selected_checkpoint = source
    if selected_checkpoint is not None:
        if args.resume:
            validate_resume_language_objective(selected_checkpoint, vars(args))
        atomic_checkpoint(output_dir / "best.pt", selected_checkpoint)

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"event": "stop_requested", "signal": signum, "saving_at": "next_step_boundary"}), flush=True)

    previous_handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGTERM, signal.SIGINT)}

    def checkpoint(include_selected=True):
        result = {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_kind": "latest",
            "model": model.state_dict(),
            "config": dataclasses.asdict(config),
            "tokenizer": tokenizer.to_dict(),
            "tokenizer_sha256": tokenizer_sha(tokenizer),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(),
            "sampler": sampler.state_dict(),
            "step": step,
            "stage": args.stage,
            "tokens": token_count,
            "target_tokens": target_token_count,
            "best_validation_loss": best_val,
            "data_sha256": hashes,
            "asset_sha256": asset_hashes,
            "preprocess_version": PREPROCESS_VERSION,
            "training_options": vars(args),
            **objective_metadata,
            "stage_history": history,
            "origin": origin,
        }
        if include_selected:
            result["selected_checkpoint"] = selected_checkpoint
        return result

    def select_and_save():
        nonlocal best_val, selected_checkpoint
        value, per_task = validation_loss(
            model, encoder, validation, args.stage, args.batch_size, args.perception_weight, args.router_weight
        )
        improved = value < best_val
        if improved:
            best_val = value
        row = {
            "event": "validation",
            "step": step,
            "validation_loss": value,
            "selected": improved,
            "per_task": per_task,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if improved:
            selected = checkpoint(include_selected=False)
            selected["checkpoint_kind"] = "selected_validation_best"
            atomic_checkpoint(output_dir / "best.pt", selected)
            # CPU snapshot 不隨下一次 optimizer.step 改變；resume 到新目錄仍可還原 best。
            selected_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    model.train()
    while step < args.steps and not stop_requested:
        before_sample = sampler.state_dict()
        batch_records = sampler.batch(args.batch_size)
        prospective = sum(len(encoder.encode(r, pretrain=args.stage == "pretrain")["input_ids"]) for r in batch_records)
        if args.max_tokens and token_count + prospective > args.max_tokens:
            sampler.load_state_dict(before_sample)
            break
        optimizer.zero_grad(set_to_none=True)
        loss, detail = objective(
            model,
            encoder,
            batch_records,
            args.stage,
            args.perception_weight,
            args.router_weight,
            args.tool_loss_weight,
            args.numeric_run_loss_weight,
            native_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step={step} loss 非有限")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("gradient 非有限")
        optimizer.step()
        step += 1
        token_count += detail.pop("tokens")
        target_token_count += detail.pop("target_tokens")
        row = {
            "event": "train",
            "stage": args.stage,
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "tokens": token_count,
            "target_tokens": target_token_count,
            "sample_ids": [r["id"] for r in batch_records],
            **detail,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step % args.eval_every == 0 and not stop_requested:
            select_and_save()
        if step % args.save_every == 0:
            atomic_checkpoint(output_dir / "latest.pt", checkpoint())
        if step == 1 or step % min(args.save_every, 25) == 0:
            print(
                json.dumps({key: row[key] for key in ("stage", "step", "loss", "gradient_norm", "tokens")}), flush=True
            )
    if not stop_requested:
        select_and_save()
    atomic_checkpoint(output_dir / "latest.pt", checkpoint())
    inference_exported = export_inference(output_dir, required=args.export_inference)
    receipt = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": args.stage,
        "architecture": args.architecture,
        "steps": step,
        "tokens": token_count,
        "target_tokens": target_token_count,
        "train_records": len(records),
        "validation_records": len(validation),
        "test_used_for_selection": False,
        "seed": args.seed,
        "data_sha256": hashes,
        "tokenizer_sha256": tokenizer_sha(tokenizer),
        "asset_sha256": asset_hashes,
        "config": dataclasses.asdict(config),
        "selection": "validation_loss (teacher-forced; not generation success)",
        "best_validation_loss": best_val,
        "seconds": time.monotonic() - started,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "freeze_perception_backbones": args.freeze_perception_backbones,
        "sampling_mode": args.sampling_mode,
        **objective_metadata,
        "sampler_policy": sampler.policy(),
        "comparison": {
            "per_expert_ffn_hidden": config.ffn_hidden,
            "active_ffn_count": config.top_k if config.architecture == "moe" else 1,
            "sampling_mode": args.sampling_mode,
            "freeze_perception_backbones": args.freeze_perception_backbones,
            "sample_draws": sampler.draws,
            "input_tokens_observed": token_count,
            "pairing_requires": "same data/tokenizer/seed/sampler/backbone options; compare actual draws and input tokens in both receipts",
        },
        "origin": checkpoint()["origin"],
        "stage_history": history,
        "interrupted": stop_requested,
        "completed_requested_steps": step >= args.steps,
        "selected_checkpoint_available": (output_dir / "best.pt").exists(),
        "inference_exported": inference_exported,
    }
    (output_dir / "train-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    for number, handler in previous_handlers.items():
        signal.signal(number, handler)
    return receipt


if __name__ == "__main__":
    main()
