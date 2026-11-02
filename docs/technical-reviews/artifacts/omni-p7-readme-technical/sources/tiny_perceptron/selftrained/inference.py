"""Verified safetensors inference through the same finite assistant and public encoder.

No training checkpoint deserialization, pretrained model, target label, transcript,
or task-to-answer dispatch is used here. A manifest binds its three payload files;
its own hash can be pinned by a caller or its source by an immutable HF revision.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path, PurePosixPath

import torch

from tiny_perceptron.selftrained.dataset import PREPROCESS_VERSION, RecordEncoder, file_sha256
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, generation_report
from tiny_perceptron.selftrained.tools import PUBLIC_MODAL_KEYS, run_tool_loop

INFERENCE_SCHEMA = "selftrained-inference-v1"
EXPORT_SCHEMA = "selftrained-random-v1"
PAYLOAD_FILES = ("model.safetensors", "model-config.json", "tokenizer.json")
MANIFEST_FILE = "inference-manifest.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
TASKS = (
    "text",
    "tool_call",
    "tool_reply",
    "tool_concept",
    "tool_missing",
    "tool_unavailable",
    "tool_unsupported",
    "vision_clothing",
    "vision_relation",
    "ocr",
    "voice_qa",
    "voice_topic_continuation",
)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_json_object)


def _sha256(value):
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("Expected a lowercase 64-hex SHA-256")
    return value


def _pixel_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(type(value) is int for value in box)
        and 0 <= box[0] < box[2]
        and 0 <= box[1] < box[3]
    )


def verify_export(model_dir, *, manifest_sha256=None):
    """Verify every payload before parsing model configuration or allocating weights."""
    root = Path(model_dir)
    manifest_path = root / MANIFEST_FILE
    actual_manifest_hash = file_sha256(manifest_path)
    if manifest_sha256 is not None and actual_manifest_hash != _sha256(manifest_sha256):
        raise ValueError("Inference manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("Unknown inference export schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_FILES):
        raise ValueError("Manifest must bind exactly the three safe inference payloads")
    if manifest.get("origin", {}).get("kind") != "all-neural-weights-random":
        raise ValueError("Export must declare the from-scratch neural origin")
    if manifest.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError("Export preprocessing version differs from this encoder")
    if manifest.get("selection") != "validation_loss":
        raise ValueError("Only a validation-selected export is supported")
    _sha256(manifest.get("selected_checkpoint_sha256"))
    for name in PAYLOAD_FILES:
        expected = _sha256(files[name])
        if file_sha256(root / name) != expected:
            raise ValueError(f"Inference payload SHA-256 mismatch: {name}")
    return manifest, actual_manifest_hash


def fetch_public_export(repo, revision, model_dir, *, prefix="", manifest_sha256=None):
    """Fetch four public files at one immutable commit, with authentication disabled.

    The release repository/revision are supplied by the user; no unpublished
    release is guessed. No remote code or pickle checkpoint is downloaded.
    """
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("HF revision must be an immutable lowercase 40-hex commit")
    if not isinstance(repo, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None:
        raise ValueError("HF repo must be an owner/name model repository")
    path_prefix = PurePosixPath(prefix)
    if path_prefix.is_absolute() or ".." in path_prefix.parts or "\\" in prefix:
        raise ValueError("HF prefix must be a relative repository directory")
    from huggingface_hub import hf_hub_download

    downloaded = {}
    for name in (MANIFEST_FILE, *PAYLOAD_FILES):
        filename = str(path_prefix / name)
        downloaded[name] = Path(
            hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type="model", token=False)
        )
    # Cached paths keep their repository directory, and form one complete export.
    source_dir = downloaded[MANIFEST_FILE].parent
    verify_export(source_dir, manifest_sha256=manifest_sha256)
    destination = Path(model_dir)
    for name, source in downloaded.items():
        target = destination / name
        if target.exists() and file_sha256(target) != file_sha256(source):
            raise FileExistsError(f"Refusing to overwrite a different local inference file: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in downloaded.items():
        target = destination / name
        if not target.exists():
            shutil.copyfile(source, target)
    verify_export(destination, manifest_sha256=manifest_sha256)
    return {"repo": repo, "revision": revision, "prefix": prefix, "authentication": "disabled"}


def _public_history(messages, *, require_user=True):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a nonempty JSON list")
    history = copy.deepcopy(messages)
    for message in history:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a JSON object")
        if set(message) - {"role", "content", *PUBLIC_MODAL_KEYS}:
            raise ValueError("Messages accept role/content and public image/audio/ROI/layout metadata only")
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unknown message role")
        if not isinstance(message.get("content"), str):
            raise ValueError("Message content must be text")
        if PUBLIC_MODAL_KEYS & message.keys() and message["role"] != "user":
            raise ValueError("Public modality metadata belongs to its original user message")
        for key in ("image", "audio"):
            if key in message and (not isinstance(message[key], str) or not message[key]):
                raise ValueError("Modality asset paths must be nonempty relative strings")
        roi = message.get("roi")
        if roi is not None and not _pixel_box(roi):
            raise ValueError("ROI must be nonnegative pixel [x1,y1,x2,y2] with positive area")
        if ("roi" in message or "image_layout" in message) and "image" not in message:
            raise ValueError("ROI/layout metadata must accompany its image in the same message")
        if "image_layout" in message:
            layout = message["image_layout"]
            if not isinstance(layout, dict) or set(layout) - {"axis", "slots"}:
                raise ValueError("image_layout accepts public axis/slots geometry only")
            slots = layout.get("slots")
            if not isinstance(slots, list) or len(slots) != 2 or not all(_pixel_box(box) for box in slots):
                raise ValueError("image_layout requires two nonnegative pixel slot boxes")
            if layout.get("axis") not in (None, "horizontal", "vertical"):
                raise ValueError("image_layout axis must be horizontal or vertical")
    if require_user and history[-1]["role"] != "user":
        raise ValueError("Input history must end with the current user, without an assistant target")
    return history


def attach_public_metadata(
    messages, *, image=None, audio=None, roi=None, image_layout=None, modality_message_index=None
):
    """Attach new CLI metadata once; preserve an earlier audio message on later rewrites."""
    history = _public_history(messages)
    metadata = {
        key: value
        for key, value in {"image": image, "audio": audio, "roi": roi, "image_layout": image_layout}.items()
        if value is not None
    }
    if not metadata:
        if modality_message_index is not None:
            raise ValueError("modality_message_index requires new public modality metadata")
        return history
    user_indexes = [index for index, message in enumerate(history) if message["role"] == "user"]
    if modality_message_index is None:
        existing = [
            index
            for index in user_indexes
            if any(history[index].get(kind) == metadata[kind] for kind in ("image", "audio") if kind in metadata)
        ]
        if len(existing) == 1:
            modality_message_index = existing[0]
        elif len(user_indexes) == 1:
            modality_message_index = user_indexes[0]
        else:
            raise ValueError(
                "For multi-turn assets, specify modality_message_index or keep metadata on its original user"
            )
    if type(modality_message_index) is not int or modality_message_index not in user_indexes:
        raise ValueError("modality_message_index must identify a user message in this history")
    for key, value in metadata.items():
        existing = history[modality_message_index].get(key)
        if existing is not None and existing != value:
            raise ValueError("New public metadata conflicts with the retained user message")
    history[modality_message_index].update(metadata)
    return _public_history(history)


class InferenceAssistant:
    """One verified checkpoint and one encoder shared across modalities and tool hops."""

    def __init__(self, model_dir, asset_dir, *, device="cpu", manifest_sha256=None):
        from safetensors import safe_open
        from safetensors.torch import load_file

        if device not in ("cpu", "cuda"):
            raise ValueError("Inference device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        root = Path(model_dir)
        manifest, manifest_hash = verify_export(root, manifest_sha256=manifest_sha256)
        config = SelftrainedConfig(**_read_json(root / "model-config.json"))
        tokenizer = CharacterTokenizer.load(root / "tokenizer.json")
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("Tokenizer vocabulary differs from model configuration")
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as archive:
            metadata = archive.metadata() or {}
        if metadata.get("origin") != "all-neural-weights-random" or metadata.get("schema") != EXPORT_SCHEMA:
            raise ValueError("Safetensors metadata differs from the from-scratch export contract")
        state = load_file(root / "model.safetensors", device="cpu")
        model = LimitedAssistant(config)
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError("Safetensors keys differ from the complete multimodal model")
        for name, tensor in state.items():
            if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                raise ValueError(f"Safetensors shape/dtype mismatch: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Safetensors contains nonfinite weights: {name}")
        if config.tied and not torch.equal(state["lm.embedding.weight"], state["lm.output.weight"]):
            raise ValueError("Tied embedding/output tensors must agree")
        verify_export(root, manifest_sha256=manifest_hash)
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.encoder = RecordEncoder(tokenizer, asset_dir, config.max_length, device)
        self.receipt = {
            "manifest_sha256": manifest_hash,
            "manifest_hash_pinned": manifest_sha256 is not None,
            "files": manifest["files"],
            "origin": manifest["origin"],
            "stage": manifest["stage"],
            "selected_step": manifest["selected_step"],
            "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"],
            "preprocess_version": manifest["preprocess_version"],
            "config": _read_json(root / "model-config.json"),
            "parameters": model.description()["parameters"],
            "generation_policy": model.description()["generation_policy"],
        }

    @torch.no_grad()
    def reply(self, messages, *, task="text", max_new_tokens=128, tools=False, **public_metadata):
        if task not in TASKS:
            raise ValueError("Unknown finite assistant task")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        history = attach_public_metadata(messages, **public_metadata)
        if task == "ocr" and history[-1].get("image") and "roi" not in history[-1]:
            raise ValueError("OCR images require a public ROI")
        calls = []

        def generate(actual_history):
            actual_history = _public_history(actual_history, require_user=False)
            # Unlike a dataset record this contains no final assistant target or supervision.
            record = {"id": "user-inference", "task": task, "messages": actual_history}
            encoded = self.encoder.encode(record, generation=True, messages=actual_history)
            if len(encoded["input_ids"]) + max_new_tokens > self.encoder.context:
                raise ValueError("Complete history plus generation exceeds model context; history is not truncated")
            ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.encoder.device)
            generated = self.model.generate(
                ids,
                modalities=[self.encoder.to_device(encoded["modalities"])],
                max_new_tokens=max_new_tokens,
                eos_id=self.encoder.tokenizer.eos_id,
            )
            tokens = generated[0].detach().cpu().tolist()
            output = self.encoder.tokenizer.decode(tokens, skip_special_tokens=True)
            report = generation_report(self.encoder.tokenizer, tokens)
            calls.append(
                {
                    "call_index": len(calls) + 1,
                    "model_sha256": self.receipt["files"]["model.safetensors"],
                    "prompt_messages": actual_history,
                    "prompt_ids": encoded["input_ids"],
                    "prompt_unknown_tokens": encoded["input_ids"].count(self.encoder.tokenizer.unk_id),
                    "generated_ids": tokens,
                    "raw_output": output,
                    "eos": report["eos"],
                    "invalid_control_tokens": report["invalid_special_tokens"],
                    "generation_status": "invalid_special_tokens"
                    if report["invalid_special_tokens"]
                    else "eos"
                    if report["eos"]
                    else "token_limit",
                    "modality_kinds": [payload["kind"] for payload in encoded["modalities"]],
                }
            )
            return output

        trace = run_tool_loop(generate, history, tools_enabled=tools)
        continued = history.copy()
        if trace["tool_message"] is not None:
            continued.extend([{"role": "assistant", "content": trace["initial_output"]}, trace["tool_message"]])
        continued.append({"role": "assistant", "content": trace["final_output"]})
        return {
            "schema": INFERENCE_SCHEMA,
            "model": self.receipt,
            "task": task,
            "answer": trace["final_output"],
            "messages": continued,
            "generations": calls,
            "tool_trace": trace,
        }


class ChatSession:
    """Keep actual replies and original modality messages across interactive turns."""

    def __init__(self, assistant, messages=None):
        self.assistant = assistant
        self.messages = copy.deepcopy(messages or [])

    def ask(self, content, *, task="text", **kwargs):
        pending = self.messages + [{"role": "user", "content": content}]
        result = self.assistant.reply(pending, task=task, **kwargs)
        self.messages = result["messages"]
        return result
