"""A pretrained, adapter-trained practical branch, separate from the tiny MoE.

Images retain a sequence of visual patches. Speech follows an explicit ASR ->
text -> the same chat model route; this module does not claim native audio chat.
Optional pretrained-model dependencies are imported only by their runners.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
ASR_ID = "openai/whisper-small"
ASR_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
ASR_VARIANTS = {
    "small": (ASR_ID, ASR_REVISION),
    "turbo": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
}
LORA_TARGETS = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def asset_path(name, data_root):
    root = Path(data_root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Asset must be an existing file inside data_root: {name}")
    return path


def row_assets(row):
    names = [row[field] for field in ("image", "audio") if row.get(field)]
    for turn in row.get("history", []):
        if isinstance(turn.get("content"), list):
            names.extend(part["image"] for part in turn["content"] if part.get("type") == "image")
    return names


def load_manifest(path, data_root=None):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Natural assistant manifest must use schema_version=1")
    rows = list(manifest.get("rows", []))
    if not rows and "splits" in manifest:
        rows = [dict(row, split=split) for split, group in manifest["splits"].items() for row in group]
    audio_rows = manifest.get("audio_rows", [])
    for row in audio_rows:
        if row.get("task", "asr") not in {"asr", "speech_chat", "speech_transcription"}:
            raise ValueError("Audio task must be legacy asr, speech_chat or speech_transcription")
    identifiers, families = set(), {}
    root = Path(data_root or path.parent).resolve()
    for row in rows + audio_rows:
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Every row must declare train, validation, or test")
        if not row.get("id") or row["id"] in identifiers:
            raise ValueError("Every row needs a globally unique id")
        identifiers.add(row["id"])
        if not isinstance(row.get("user"), str):
            raise ValueError("Every row needs a string user prompt / reference transcript")
        if not row.get("audio") and not isinstance(row.get("answer"), str):
            raise ValueError("Training / visual rows need a string answer")
        family = row.get("family", row["id"])
        if family in families and families[family] != row["split"]:
            raise ValueError(f"Related family crosses splits: {family}")
        families[family] = row["split"]
        for name in row_assets(row):
            asset_path(name, root)
    manifest["rows"] = rows
    manifest["audio_rows"] = audio_rows
    manifest["manifest_sha256"] = sha256(path)
    manifest["asset_sha256"] = {
        name: sha256(asset_path(name, root))
        for name in sorted({name for row in rows + audio_rows for name in row_assets(row)})
    }
    if "files" in manifest:
        declared = manifest["files"]
        if isinstance(declared, list):
            declared = {item.get("path", item.get("name")): item for item in declared}
        for name, actual_hash in manifest["asset_sha256"].items():
            expected = declared.get(name)
            if not expected or expected.get("sha256") != actual_hash:
                raise ValueError(f"Asset does not match frozen manifest SHA: {name}")
            if "bytes" in expected and int(expected["bytes"]) != asset_path(name, root).stat().st_size:
                raise ValueError(f"Asset does not match frozen manifest byte count: {name}")
    return manifest, root


def messages_for(row, data_root, *, assistant=False):
    """Keep prior turns and their images; the final reference answer is optional."""
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": [{"type": "text", "text": row["system"]}]})
    for turn in row.get("history", []):
        if turn["role"] not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported history role")
        content = turn["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        messages.append({"role": turn["role"], "content": content})
    content = []
    if row.get("image"):
        content.append({"type": "image", "image": str(asset_path(row["image"], data_root))})
    content.append({"type": "text", "text": row["user"]})
    messages.append({"role": "user", "content": content})
    if assistant:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return messages


def encode_messages(processor, messages, data_root, *, generation_prompt):
    images = []
    for turn in messages:
        for part in turn["content"]:
            if part["type"] == "image":
                path = asset_path(part["image"], data_root)
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            elif part["type"] != "text":
                raise ValueError("This branch supports text and still images; speech is transcribed separately")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    return dict(processor(**kwargs))


def encode_training_row(processor, row, data_root, max_tokens=2048):
    prompt = encode_messages(processor, messages_for(row, data_root), data_root, generation_prompt=True)
    full = encode_messages(processor, messages_for(row, data_root, assistant=True), data_root, generation_prompt=False)
    prefix_length = prompt["input_ids"].shape[-1]
    if not torch.equal(prompt["input_ids"], full["input_ids"][:, :prefix_length]):
        raise ValueError("Chat template prompt is not an exact prefix; assistant masking would be wrong")
    if full["input_ids"].shape[-1] > max_tokens:
        raise ValueError(f"Row {row['id']} exceeds {max_tokens} tokens; refusing silent multimodal truncation")
    labels = full["input_ids"].clone()
    labels[:, :prefix_length] = -100
    if "attention_mask" in full:
        labels[full["attention_mask"] == 0] = -100
    if not (labels != -100).any():
        raise ValueError("Reference answer must provide at least one supervised token")
    full["labels"] = labels
    return full


def normalized(text, strip_whitespace=False):
    text = unicodedata.normalize("NFKC", text).strip()
    return "".join(text.split()) if strip_whitespace else text


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def asr_cer_metrics(reference, prediction):
    """Keep raw Unicode codepoints and disclose the separate normalization."""
    target, output = normalized(reference, True), normalized(prediction, True)
    raw_errors, errors = edit_distance(reference, prediction), edit_distance(target, output)
    return {
        "raw_errors": raw_errors,
        "raw_reference_characters": len(reference),
        "raw_cer": raw_errors / len(reference) if reference else None,
        "raw_metric": "Levenshtein distance over original Unicode codepoints; preserve whitespace and punctuation",
        "errors": errors,
        "reference_characters": len(target),
        "normalized_cer": errors / len(target) if target else None,
        "normalized_reference": target,
        "normalized_prediction": output,
        "normalization": "Unicode NFKC, remove all Unicode whitespace via str.split(); preserve case and punctuation after NFKC; no Traditional/Simplified conversion",
    }


def contains_fact(prediction, fact):
    fact = normalized(fact).casefold()
    if re.fullmatch(r"[a-z0-9 ]+", fact):
        return re.search(r"(?<!\w)" + re.escape(fact) + r"(?!\w)", prediction) is not None
    return fact in prediction


def score_output(row, prediction):
    reference = row.get("references") or {"kind": "exact"}
    kind = reference.get("kind", "exact")
    if kind == "manual":
        return {"kind": kind, "passed": None, "reason": "Needs independent semantic review of the raw answer"}
    if kind in {"ocr", "ocr_order"}:
        strip = reference.get("strip_whitespace", kind == "ocr")
        target = normalized(reference.get("text", row["answer"]), strip)
        output = normalized(prediction, strip)
        errors = edit_distance(target, output)
        return {
            "kind": kind,
            "passed": target == output,
            "errors": errors,
            "reference_characters": len(target),
            "cer": errors / len(target) if target else None,
            "empty_reference_false_positive": bool(output) if not target else None,
        }
    if kind == "facts":
        output = normalized(prediction).casefold()
        groups = reference.get("fact_groups", [])
        if not groups or any(not group for group in groups):
            raise ValueError("Fact rubrics need nonempty groups; otherwise passing would be vacuous")
        matched = [any(contains_fact(output, term) for term in group) for group in groups]
        forbidden = [term for term in reference.get("forbidden", []) if contains_fact(output, term)]
        return {
            "kind": kind,
            "passed": all(matched) and not forbidden,
            "matched_fact_groups": matched,
            "forbidden_matches": forbidden,
            "scope": "Finite phrase rubric, not a complete semantic judge",
        }
    if kind != "exact":
        raise ValueError(f"Unsupported reference kind: {kind}")
    accepted = reference.get("accepted", [row["answer"]])
    return {"kind": kind, "passed": normalized(prediction) in [normalized(text) for text in accepted]}


def environment_versions():
    result = {}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "huggingface-hub", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_core(options, *, adapter=None, train=False):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[options.dtype]
    if options.device == "cpu" and dtype != torch.float32:
        raise ValueError("Use float32 for the documented CPU route")
    common = {
        "revision": options.model_revision,
        "cache_dir": options.cache_dir,
        "local_files_only": options.local_files_only,
    }
    processor = AutoProcessor.from_pretrained(
        options.model, min_pixels=options.min_pixels, max_pixels=options.max_pixels, **common
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        options.model, dtype=dtype, attn_implementation="sdpa", **common
    ).to(options.device)
    model.requires_grad_(False)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=options.lora_rank,
                lora_alpha=2 * options.lora_rank,
                lora_dropout=0.0,
                target_modules=LORA_TARGETS,
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    if train:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if any(parameter.requires_grad for name, parameter in model.named_parameters() if ".visual." in name):
            raise RuntimeError("Practical branch must freeze the pretrained vision encoder")
        model.train()
    else:
        model.eval()
    return model, processor


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, processor, row, data_root, options):
    inputs = encode_messages(processor, messages_for(row, data_root), data_root, generation_prompt=True)
    input_tokens = inputs["input_ids"].shape[-1]
    if input_tokens + options.max_new_tokens > options.max_tokens:
        raise ValueError(f"Prompt {row['id']} exceeds the declared inference token budget")
    inputs = {key: value.to(options.device) for key, value in inputs.items()}
    synchronize(options.device)
    started = time.monotonic()
    output = model.generate(
        **inputs,
        max_new_tokens=options.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    synchronize(options.device)
    seconds = time.monotonic() - started
    if hasattr(output, "sequences"):
        output = output.sequences
    suffix = output[:, input_tokens:]
    token_ids = suffix[0].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
    reached_limit = len(token_ids) >= options.max_new_tokens
    stop_reason = "eos" if ended_with_eos else ("max_new_tokens" if reached_limit else "unknown")
    answer = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {
        "id": row["id"],
        "split": row.get("split"),
        "task": row.get("task"),
        "family": row.get("family"),
        "user": row["user"],
        "image": row.get("image"),
        "reference_answer": row.get("answer"),
        "prediction": answer,
        "input_tokens": input_tokens,
        "generated_tokens": suffix.numel(),
        "generated_token_ids": token_ids,
        "generated_token_ids_scope": "Raw generated suffix including EOS; prompt excluded; no token rewriting",
        "eos_token_ids": eos_ids,
        "ended_with_eos": ended_with_eos,
        "stop_reason": stop_reason,
        "truncated": reached_limit and not ended_with_eos,
        "completion_unknown": not reached_limit and not ended_with_eos,
        "completion_scope": "EOS reports decoder stopping, not semantic completeness, truth, or full image/audio understanding",
        "generation_seconds": seconds,
        "tokens_per_second": suffix.numel() / seconds,
        "throughput_scope": "Full generate call including prompt prefill; generated tokens / measured wall seconds",
        "reached_max_new_tokens": reached_limit,
        "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
        "score": score_output(row, answer) if row.get("answer") is not None else None,
    }


def parameter_counts(model):
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def tensor_values(parameter):
    values = parameter.detach().cpu().contiguous()
    return {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "sha256_values": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def adapter_tensor_hashes(model):
    return {name: tensor_values(parameter) for name, parameter in model.named_parameters() if "lora_" in name}


def frozen_parameter_samples(model):
    """A disclosed small value sample, not an exhaustive hash of the 2B base."""
    frozen = sorted((name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad)
    selections = {0, len(frozen) // 4, len(frozen) // 2, 3 * len(frozen) // 4, len(frozen) - 1}
    for index, (name, _) in enumerate(frozen):
        if any(
            fragment in name
            for fragment in (
                "embed_tokens.weight",
                ".visual.patch_embed.proj.weight",
                "layers.0.self_attn.q_proj.base_layer.weight",
            )
        ):
            selections.add(index)
    result = {}
    for index in sorted(selections):
        if index < 0 or index >= len(frozen):
            continue
        name, parameter = frozen[index]
        flat = parameter.detach().reshape(-1)
        positions = sorted({0, len(flat) // 2, len(flat) - 1})
        values = flat[positions].cpu()
        result[name] = {"flat_indices": positions, "values": values.float().tolist(), **tensor_values(values)}
    return result


def provenance(options, manifest=None):
    return {
        "schema_version": 1,
        "observed_at": datetime.now(UTC).isoformat(),
        "architecture": "Dense pretrained Qwen3-VL; explicit frozen Whisper ASR -> same text/vision core",
        "model": options.model,
        "model_revision": options.model_revision,
        "asr_model": options.asr_model,
        "asr_revision": options.asr_revision,
        "versions": environment_versions(),
        "device": options.device,
        "dtype": options.dtype,
        "gpu_name": torch.cuda.get_device_name(options.device)
        if str(options.device).startswith("cuda") and torch.cuda.is_available()
        else None,
        "attention_implementation": "sdpa",
        "max_pixels": options.max_pixels,
        "min_pixels": options.min_pixels,
        "max_tokens": options.max_tokens,
        "seed": options.seed,
        "dataset_version": manifest.get("dataset_version") if manifest else None,
        "manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
        "asset_sha256": manifest.get("asset_sha256", {}) if manifest else {},
        "sources": manifest.get("sources", []) if manifest else [],
        "pretrained_boundary": "Base vision/language and ASR knowledge was learned by their original authors; this course trains a language-attention LoRA adapter only",
    }


def save_checkpoint(model, optimizer, output, record):
    output = Path(output)
    temporary = output / "adapter.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary / "training_state.pt",
    )
    write_json(temporary / "training.json", record)
    destination, previous = output / "adapter", output / "adapter.previous"
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    temporary.rename(destination)
    write_json(output / "training.json", record)
    if previous.exists():
        shutil.rmtree(previous)


def checkpoint_schedule(value, requested_steps):
    """At most two explicit, increasing update boundaries; never infer an epoch."""
    if isinstance(value, str):
        if not value:
            return ()
        fields = value.split(",")
        if any(not re.fullmatch(r"[0-9]+", field.strip()) for field in fields):
            raise ValueError("Checkpoint steps must be comma-separated positive integers")
        steps = tuple(int(field) for field in fields)
    else:
        steps = tuple(value)
    if (
        len(steps) > 2
        or any(type(step) is not int or not 1 <= step <= requested_steps for step in steps)
        or tuple(sorted(set(steps))) != steps
    ):
        raise ValueError("Checkpoint steps must contain at most two increasing unique completed updates")
    return steps


def archive_checkpoint(model, optimizer, output, record):
    """Save the current real tensors/state, independently of the rotating latest copy."""
    step = record["completed_steps"]
    destination = Path(output) / "checkpoints" / f"step-{step:06d}"
    if destination.exists():
        raise ValueError("An archived checkpoint is immutable; refusing to overwrite it")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_checkpoint(model, optimizer, temporary, record)
    payload = temporary / "adapter"
    files = [
        {"path": file.name, "bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(payload.iterdir())
        if file.is_file()
    ]
    write_json(
        payload / "checkpoint.json",
        {"completed_steps": step, "manifest_sha256": record["manifest_sha256"], "files": files},
    )
    payload.rename(destination)
    shutil.rmtree(temporary)
    return {
        "completed_steps": step,
        "path": destination.relative_to(output).as_posix(),
        "files": files,
        "checkpoint_metadata_sha256": sha256(destination / "checkpoint.json"),
    }


def training_row_at(rows, step, seed):
    indices = list(range(len(rows)))
    random.Random(seed + step // len(rows)).shuffle(indices)
    return rows[indices[step % len(rows)]]


def run_train(options):
    manifest, data_root = load_manifest(options.manifest, options.data_root)
    rows = [row for row in manifest["rows"] if row["split"] == "train"]
    if not rows:
        raise ValueError("No training rows")
    checkpoints = checkpoint_schedule(getattr(options, "checkpoint_steps", ()), options.steps)
    torch.manual_seed(options.seed)
    started = time.monotonic()
    model, processor = load_core(options, adapter=options.adapter, train=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any("lora_" not in name for name in names):
        raise RuntimeError("Optimizer must update only the explicitly selected LoRA tensors")
    optimizer = torch.optim.AdamW(parameters, lr=options.learning_rate)
    record = dict(
        provenance(options, manifest),
        **parameter_counts(model),
        lora_rank=options.lora_rank,
        lora_targets=LORA_TARGETS,
        requested_steps=options.steps,
        completed_steps=0,
        gradient_accumulation=options.gradient_accumulation,
        learning_rate=options.learning_rate,
        history=[],
        trained_rows=0,
        status="training",
        optimizer_parameter_names=names,
        trainable_parameter_names=names,
        optimizer_only_lora=True,
        checkpoint_steps=list(checkpoints),
        archived_checkpoints=[],
    )
    if options.adapter:
        prior = json.loads((Path(options.adapter) / "training.json").read_text())
        for field in (
            "manifest_sha256",
            "asset_sha256",
            "model_revision",
            "lora_rank",
            "gradient_accumulation",
            "seed",
            "learning_rate",
        ):
            if prior[field] != record[field]:
                raise ValueError(f"Resume contract changed: {field}")
        if options.steps < prior["completed_steps"]:
            raise ValueError("Requested total steps cannot precede the resumed checkpoint")
        record["history"] = prior["history"]
        record["completed_steps"] = prior["completed_steps"]
        record["trained_rows"] = prior["trained_rows"]
        # Prior archives remain in the earlier immutable run; never relabel them
        # as newly saved versions in this output directory.
        record["prior_archived_checkpoints"] = prior.get("archived_checkpoints", [])
        if any(step <= prior["completed_steps"] for step in checkpoints):
            raise ValueError("Resumed archival steps must follow the resumed completed update")
        record["resume_from"] = {
            "adapter_training_sha256": sha256(Path(options.adapter) / "training.json"),
            "completed_steps": prior["completed_steps"],
            "requested_steps": prior["requested_steps"],
        }
        state = torch.load(Path(options.adapter) / "training_state.pt", map_location=options.device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    record["initial_adapter_tensors"] = adapter_tensor_hashes(model)
    record["frozen_parameter_samples_initial"] = frozen_parameter_samples(model)
    record["frozen_check_scope"] = (
        "A few disclosed indices in selected frozen tensors; not all 2B base values compared bitwise"
    )
    if str(options.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(options.device)
    checkpoint_time = time.monotonic()
    for step in range(record["completed_steps"], options.steps):
        if time.monotonic() - started >= options.max_seconds:
            record["status"] = "time_limit_checkpoint"
            break
        optimizer.zero_grad(set_to_none=True)
        token_sum, token_count, observed_ids = 0.0, 0, []
        encoded = []
        for micro in range(options.gradient_accumulation):
            index = step * options.gradient_accumulation + micro
            row = training_row_at(rows, index, options.seed)
            batch = encode_training_row(processor, row, data_root, options.max_tokens)
            encoded.append((row, batch))
        denominator = sum(int((batch["labels"][:, 1:] != -100).sum()) for _, batch in encoded)
        for row, batch in encoded:
            batch = {key: value.to(options.device) for key, value in batch.items()}
            supervised = int((batch["labels"][:, 1:] != -100).sum())
            result = model(**batch)
            if not torch.isfinite(result.loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            (result.loss * supervised / denominator).backward()
            token_sum += float(result.loss.detach()) * supervised
            token_count += supervised
            observed_ids.append(row["id"])
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite adapter gradients")
        optimizer.step()
        record["completed_steps"] = step + 1
        record["trained_rows"] += len(encoded)
        record["history"].append(
            {
                "step": step + 1,
                "answer_token_loss_before_update": token_sum / token_count,
                "supervised_tokens": token_count,
                "row_ids": observed_ids,
                "gradient_norm_before_clip": float(grad_norm),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if step + 1 in checkpoints:
            record["archived_checkpoints"].append(archive_checkpoint(model, optimizer, options.output, record))
        if (step + 1) % options.checkpoint_every == 0 or time.monotonic() - checkpoint_time >= 60:
            save_checkpoint(model, optimizer, options.output, record)
            checkpoint_time = time.monotonic()
    if record["completed_steps"] == options.steps:
        record["status"] = "completed"
    synchronize(options.device)
    record["elapsed_seconds"] = time.monotonic() - started
    record["final_adapter_tensors"] = adapter_tensor_hashes(model)
    record["changed_adapter_tensor_count"] = sum(
        initial != record["final_adapter_tensors"][name] for name, initial in record["initial_adapter_tensors"].items()
    )
    record["frozen_parameter_samples_final"] = frozen_parameter_samples(model)
    record["frozen_parameter_samples_unchanged"] = (
        record["frozen_parameter_samples_initial"] == record["frozen_parameter_samples_final"]
    )
    if not record["frozen_parameter_samples_unchanged"]:
        raise RuntimeError("Frozen base parameter sample changed unexpectedly")
    record["peak_cuda_memory_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(options.device) if str(options.device).startswith("cuda") else None
    )
    save_checkpoint(model, optimizer, options.output, record)
    write_json(Path(options.output) / "provenance.json", provenance(options, manifest))
    write_json(Path(options.output) / "result.json", record)
    return record


def summarize(records):
    tasks = {}
    for record in records:
        task = tasks.setdefault(
            record["task"], {"count": 0, "scored": 0, "passed": 0, "ocr_errors": 0, "ocr_reference_characters": 0}
        )
        task["count"] += 1
        score = record.get("score")
        if score and score["passed"] is not None:
            task["scored"] += 1
            task["passed"] += int(score["passed"])
        if score and "errors" in score:
            task["ocr_errors"] += score["errors"]
            task["ocr_reference_characters"] += score["reference_characters"]
    for task in tasks.values():
        task["pass_rate"] = task["passed"] / task["scored"] if task["scored"] else None
        count = task["ocr_reference_characters"]
        task["micro_cer"] = task["ocr_errors"] / count if count else None
    return tasks


def load_asr(options):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    common = {
        "revision": options.asr_revision,
        "cache_dir": options.cache_dir,
        "local_files_only": options.local_files_only,
    }
    processor = WhisperProcessor.from_pretrained(options.asr_model, **common)
    model = WhisperForConditionalGeneration.from_pretrained(options.asr_model, dtype=torch.float32, **common)
    return model.eval(), processor


@torch.inference_mode()
def transcribe(model, processor, path):
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Audio must contain finite samples")
    if sample_rate != 16000:
        from scipy.signal import resample_poly

        divisor = math.gcd(sample_rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    if len(waveform) > 30 * 16000:
        raise ValueError("Teaching speech route accepts at most 30 seconds; add chunking explicitly for longer audio")
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    started = time.monotonic()
    limit = 128
    generated = model.generate(
        inputs.input_features,
        attention_mask=inputs.attention_mask,
        language="chinese",
        task="transcribe",
        max_new_tokens=limit,
        do_sample=False,
        return_timestamps=False,
        return_dict_in_generate=True,
    )
    # Whisper's ordinary tensor return strips decoder prompt / EOS. The raw
    # ModelOutput sequences retain them, so stopping evidence stays auditable.
    ids = generated.sequences
    token_ids = ids[0].tolist()
    start_id = model.generation_config.decoder_start_token_id
    prefix = [start_id] + [
        token for _, token in processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    ]
    prefix_matches = token_ids[: len(prefix)] == prefix
    suffix = token_ids[len(prefix) :] if prefix_matches else None
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
    reached_limit = len(suffix) >= limit if suffix is not None else None
    if ended_with_eos:
        stop_reason = "eos"
    elif reached_limit:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "unknown_without_eos"
    text = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {
        "transcript": text,
        "seconds": time.monotonic() - started,
        "original_sample_rate": sample_rate,
        "resampled_rate": 16000,
        "audio_seconds": len(waveform) / 16000,
        "audio_sha256": sha256(path),
        "raw_token_ids": token_ids,
        "raw_token_count": len(token_ids),
        "expected_decoder_prompt_ids": prefix,
        "decoder_prompt_matches": prefix_matches,
        "generated_token_ids": suffix,
        "generated_token_count": len(suffix) if suffix is not None else None,
        "max_new_tokens": limit,
        "eos_token_ids": eos_ids,
        "ended_with_eos": ended_with_eos,
        "reached_token_limit": reached_limit,
        "stop_reason": stop_reason,
        "truncated": bool(reached_limit and not ended_with_eos),
        "completion_unknown": not ended_with_eos and not reached_limit,
        "completion_scope": "EOS records decoder stopping; it does not prove that the complete utterance was correctly recognized",
    }


def evaluate_rows(model, processor, rows, data_root, options, variant, deadline=None):
    records = []
    for row in rows:
        if deadline is not None and time.monotonic() >= deadline:
            break
        result = generate(model, processor, row, data_root, options)
        result["variant"] = variant
        records.append(result)
        write_json(Path(options.output) / f"generations-{variant}.json", records)
    return records


def transcription_summary(records):
    result = {
        "count": len(records),
        "raw_errors": sum(row["raw_errors"] for row in records),
        "raw_reference_characters": sum(row["raw_reference_characters"] for row in records),
        "errors": sum(row["errors"] for row in records),
        "reference_characters": sum(row["reference_characters"] for row in records),
        "truncated_count": sum(bool(row.get("truncated")) for row in records),
        "completion_unknown_count": sum(bool(row.get("completion_unknown")) for row in records),
    }
    result["raw_micro_cer"] = (
        result["raw_errors"] / result["raw_reference_characters"] if result["raw_reference_characters"] else None
    )
    result["normalized_micro_cer"] = (
        result["errors"] / result["reference_characters"] if result["reference_characters"] else None
    )
    return result


@contextlib.contextmanager
def evaluation_adapter(model, name):
    previous = model.active_adapter
    model.set_adapter(name)
    try:
        yield
    finally:
        model.set_adapter(previous)


def run_evaluate(options, *, baseline=False):
    selected_only = getattr(options, "selected_only", False)
    comparisons = getattr(options, "comparison_adapters", [])
    if selected_only and comparisons:
        raise ValueError("Selected-only evaluation cannot include comparison adapters")
    started = time.monotonic()
    deadline = started + options.max_seconds
    manifest, data_root = load_manifest(options.manifest, options.data_root)
    rows = [row for row in manifest["rows"] if row["split"] == options.split]
    audio_rows = [row for row in manifest["audio_rows"] if row["split"] == options.split]
    audio_chat_rows = [row for row in audio_rows if row.get("task", "asr") != "speech_transcription"]
    if not rows and not audio_rows:
        raise ValueError("No evaluation rows in requested split")
    model, processor = load_core(options, adapter=None if baseline else options.adapter)
    if comparisons and (baseline or not options.adapter or Path(comparisons[0][1]) != Path(options.adapter)):
        raise ValueError("Comparison adapters need an adapter evaluation with the first exact loaded path")
    variants = []
    if not (selected_only and options.adapter and not baseline):
        variants.append(
            ("base", model.disable_adapter())
            if options.adapter and not baseline
            else ("base", contextlib.nullcontext())
        )
    if options.adapter and not baseline:
        if comparisons:
            variants.append((comparisons[0][0], evaluation_adapter(model, "default")))
            for label, path in comparisons[1:]:
                model.load_adapter(path, adapter_name=label, is_trainable=False)
                variants.append((label, evaluation_adapter(model, label)))
            model.eval()
        else:
            variants.append((getattr(options, "adapter_label", "adapter"), contextlib.nullcontext()))
    result = dict(
        provenance(options, manifest),
        **parameter_counts(model),
        split=options.split,
        selected_only=selected_only,
        requested_visual_text_rows=len(rows),
        requested_audio_rows=len(audio_rows),
        requested_audio_chat_rows=len(audio_chat_rows),
        status="in_progress",
        variants={},
    )
    write_json(Path(options.output) / "result.json", result)
    raw = []
    transcripts = []
    asr_model = asr_processor = None
    if audio_rows:
        asr_model, asr_processor = load_asr(options)
        for row in audio_rows:
            if time.monotonic() >= deadline:
                break
            observation = transcribe(asr_model, asr_processor, asset_path(row["audio"], data_root))
            observation.update(
                id=row["id"],
                task=row.get("task", "asr"),
                source=row.get("source"),
                reference_transcript=row["user"],
                **asr_cer_metrics(row["user"], observation["transcript"]),
                speaker=row.get("speaker"),
                input_mode="speech; actual ASR hypothesis, never the reference transcript",
            )
            transcripts.append(observation)
        result["asr"] = {
            "parameters": sum(p.numel() for p in asr_model.parameters()),
            "device": "cpu",
            "count": len(transcripts),
            "errors": sum(row["errors"] for row in transcripts),
            "reference_characters": sum(row["reference_characters"] for row in transcripts),
            "raw_errors": sum(row["raw_errors"] for row in transcripts),
            "raw_reference_characters": sum(row["raw_reference_characters"] for row in transcripts),
            "truncated_count": sum(bool(row.get("truncated")) for row in transcripts),
            "completion_unknown_count": sum(bool(row.get("completion_unknown")) for row in transcripts),
            "normalized_metric": "Unicode NFKC + remove all Unicode whitespace; codepoint CER; no case/punctuation/script correction",
            "raw_metric": "Original Unicode codepoint CER, including whitespace and punctuation",
        }
        count = result["asr"]["reference_characters"]
        result["asr"]["micro_cer"] = result["asr"]["errors"] / count if count else None
        result["asr"]["normalized_micro_cer"] = result["asr"]["micro_cer"]
        result["asr"]["micro_cer_scope"] = "normalized_micro_cer; raw_micro_cer is reported separately"
        raw_count = result["asr"]["raw_reference_characters"]
        result["asr"]["raw_micro_cer"] = result["asr"]["raw_errors"] / raw_count if raw_count else None
        result["asr"]["completed"] = len(transcripts) == len(audio_rows)
        result["asr"]["by_task"] = {
            task: transcription_summary([record for record in transcripts if record["task"] == task])
            for task in sorted({record["task"] for record in transcripts})
        }
        groups = {}
        for record in transcripts:
            source = record.get("source")
            if isinstance(source, dict):
                source = (
                    source.get("id")
                    or source.get("dataset")
                    or source.get("repo")
                    or json.dumps(source, sort_keys=True)
                )
            groups.setdefault(str(source) if source is not None else "unspecified", []).append(record)
        result["asr"]["by_source"] = {source: transcription_summary(group) for source, group in groups.items()}
        write_json(Path(options.output) / "transcripts.json", transcripts)
    for variant, context in variants:
        with context:
            records = evaluate_rows(model, processor, rows, data_root, options, variant, deadline)
            for row, observation in zip(audio_rows[: len(transcripts)], transcripts, strict=True):
                if row.get("task", "asr") == "speech_transcription":
                    continue
                if time.monotonic() >= deadline:
                    break
                actual = dict(row, user=observation["transcript"], task="speech_chat")
                spoken = generate(model, processor, actual, data_root, options)
                typed = generate(model, processor, dict(row, task="typed_chat"), data_root, options)
                spoken.update(
                    variant=variant,
                    transcript=observation["transcript"],
                    reference_user=row["user"],
                    input_mode="speech",
                    asr_truncated=observation.get("truncated"),
                    asr_completion_unknown=observation.get("completion_unknown"),
                    asr_stop_reason=observation.get("stop_reason"),
                    asr_generated_token_count=observation.get("generated_token_count"),
                )
                typed.update(variant=variant, input_mode="typed reference transcript")
                records.extend([spoken, typed])
            raw.extend(records)
            result["variants"][variant] = {
                "tasks": summarize(records),
                "generation_count": len(records),
                "completed": len(records) == len(rows) + 2 * len(audio_chat_rows),
            }
            write_json(Path(options.output) / f"generations-{variant}.json", records)
            write_json(Path(options.output) / "result.json", result)
    write_json(Path(options.output) / "generations.json", raw)
    write_json(Path(options.output) / "provenance.json", provenance(options, manifest))
    result["status"] = (
        "completed"
        if all(value["completed"] for value in result["variants"].values())
        and result.get("asr", {}).get("completed", True)
        else "time_limit_partial"
    )
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(Path(options.output) / "result.json", result)
    return result


def prepare(options):
    from huggingface_hub import snapshot_download

    paths = {}
    for label, model, revision in (
        ("core", options.model, options.model_revision),
        ("asr", options.asr_model, options.asr_revision),
    ):
        path = snapshot_download(
            model,
            revision=revision,
            cache_dir=options.cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "LICENSE*", "README.md"],
            local_files_only=options.local_files_only,
        )
        paths[label] = {
            "model": model,
            "revision": revision,
            "snapshot": path,
            "files": [
                {"name": str(file.relative_to(path)), "bytes": file.stat().st_size, "sha256": sha256(file)}
                for file in sorted(Path(path).rglob("*"))
                if file.is_file()
            ],
        }
    result = dict(provenance(options), snapshots=paths, status="completed")
    write_json(Path(options.output) / "result.json", result)
    return result
