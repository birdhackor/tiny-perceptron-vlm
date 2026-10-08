#!/usr/bin/env python3
"""Generate from verified from-scratch safetensors with real public conversation history.

Use --history-output to save the actual messages for the next CLI turn. A later
typed rewrite retains audio on its original user message. Public HF fetching
requires an explicitly supplied immutable commit; no release is assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiny_perceptron.selftrained.inference import (  # noqa: E402
    TASKS,
    InferenceAssistant,
    fetch_public_export,
)


def pixel_roi(value):
    try:
        coordinates = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 integer pixels") from None
    if len(coordinates) != 4 or not (0 <= coordinates[0] < coordinates[2] and 0 <= coordinates[1] < coordinates[3]):
        raise argparse.ArgumentTypeError("ROI must be nonnegative x1,y1,x2,y2 with positive area")
    return coordinates


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", required=True, help="Directory containing the three safe payloads and manifest")
    result.add_argument("--messages", required=True, help="JSON list of public messages ending with the current user")
    result.add_argument("--asset-dir", required=True, help="Root for relative image/audio asset paths")
    result.add_argument(
        "--task", choices=TASKS, default="text", help="Preprocessing/task metadata, never an answer lookup"
    )
    result.add_argument("--image", help="Public relative image path inside asset-dir")
    result.add_argument("--audio", help="Public relative audio path inside asset-dir")
    result.add_argument("--roi", type=pixel_roi, help="Public x1,y1,x2,y2 pixel ROI")
    result.add_argument("--image-layout", help="JSON file containing public slot geometry")
    result.add_argument(
        "--modality-message-index", type=int, help="Original user-message index for a new multi-turn asset"
    )
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument(
        "--tools", action="store_true", help="Enable the bounded calculator and same-core second generation"
    )
    result.add_argument("--manifest-sha256", help="Optional external SHA-256 pin for the manifest itself")
    result.add_argument("--repo", help="Optional public HF owner/name model repository")
    result.add_argument("--revision", help="Required with repo: immutable lowercase 40-hex commit")
    result.add_argument("--prefix", default="", help="Safe-export directory inside the public HF repository")
    result.add_argument("--output", help="Also save the complete JSON result to this file")
    result.add_argument("--history-output", help="Save actual continued messages as a JSON list for another turn")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.threads < 1:
        argument_parser.error("threads must be positive")
    if bool(args.repo) != bool(args.revision):
        argument_parser.error("repo and its immutable revision must be provided together")
    if args.prefix and not args.repo:
        argument_parser.error("prefix is used only with a public repo and revision")
    torch.set_num_threads(args.threads)
    source = None
    if args.repo:
        source = fetch_public_export(
            args.repo, args.revision, args.model_dir, prefix=args.prefix, manifest_sha256=args.manifest_sha256
        )
    messages = json.loads(Path(args.messages).read_text(encoding="utf-8"))
    layout = json.loads(Path(args.image_layout).read_text(encoding="utf-8")) if args.image_layout else None
    assistant = InferenceAssistant(
        args.model_dir, args.asset_dir, device=args.device, manifest_sha256=args.manifest_sha256
    )
    result = assistant.reply(
        messages,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        tools=args.tools,
        image=args.image,
        audio=args.audio,
        roi=args.roi,
        image_layout=layout,
        modality_message_index=args.modality_message_index,
    )
    if source is not None:
        result["public_source"] = source
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    for path, text in (
        (args.output, content),
        (args.history_output, json.dumps(result["messages"], ensure_ascii=False, indent=2) + "\n"),
    ):
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    print(content, end="")
    return result


if __name__ == "__main__":
    main()
