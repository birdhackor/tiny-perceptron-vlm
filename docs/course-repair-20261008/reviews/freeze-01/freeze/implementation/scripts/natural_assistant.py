"""Run the separate pretrained natural-image / Chinese-OCR / speech-chat branch."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_perceptron import natural_assistant as assistant  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("prepare", "baseline", "train", "validation", "evaluate", "chat", "serve"))
    result.add_argument("--manifest", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--model", default=assistant.MODEL_ID)
    result.add_argument("--model-revision", default=assistant.MODEL_REVISION)
    result.add_argument("--asr-model", default=assistant.ASR_ID)
    result.add_argument("--asr-revision", default=assistant.ASR_REVISION)
    result.add_argument("--asr-variant", choices=tuple(assistant.ASR_VARIANTS))
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    result.add_argument("--min-pixels", type=int, default=65536)
    result.add_argument("--max-pixels", type=int, default=524288)
    result.add_argument("--max-tokens", type=int, default=2048)
    result.add_argument("--max-new-tokens", type=int, default=384)
    result.add_argument("--max-seconds", type=int, default=3300)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--adapter", type=Path)
    result.add_argument("--adapter-label", default="adapter")
    result.add_argument("--comparison-adapter", action="append", default=[], help="Validation only: LABEL=PATH")
    result.add_argument(
        "--selected-only", action="store_true", help="Evaluate only the explicitly selected adapter, or base if absent"
    )
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--learning-rate", type=float, choices=(1e-4, 3e-5), default=3e-5)
    result.add_argument("--lora-rank", type=int, default=8)
    result.add_argument("--gradient-accumulation", type=int, default=2)
    result.add_argument("--checkpoint-every", type=int, default=25)
    result.add_argument("--checkpoint-steps", default="", help="At most two explicit completed updates, e.g. 500,1000")
    result.add_argument("--split", choices=("validation", "test"), default="validation")
    result.add_argument("--user")
    result.add_argument("--image")
    result.add_argument("--audio")
    result.add_argument("--history", type=Path, help="JSON array of earlier messages, including earlier images")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8766)
    return result


def main():
    options = parser().parse_args()
    if options.asr_variant:
        options.asr_model, options.asr_revision = assistant.ASR_VARIANTS[options.asr_variant]
    if (options.asr_model, options.asr_revision) not in assistant.ASR_VARIANTS.values():
        raise ValueError("ASR must use the fixed small or turbo model/revision pair")
    options.checkpoint_steps = assistant.checkpoint_schedule(options.checkpoint_steps, options.steps)
    if options.checkpoint_steps and options.stage != "train":
        raise ValueError("Checkpoint archival is a training-only option")
    if options.stage == "train" and (options.steps > 3000 or options.max_seconds > 3300):
        raise ValueError("Training is bounded to at most 3000 updates and a 3300-second soft budget")
    if not re.fullmatch(r"adapter(?:-step-[0-9]{6})?", options.adapter_label):
        raise ValueError("Adapter label must identify latest or an explicit archived step")
    options.comparison_adapters = []
    for specification in options.comparison_adapter:
        label, separator, path = specification.partition("=")
        if not separator or not path or not re.fullmatch(r"adapter-step-[0-9]{6}", label):
            raise ValueError("Comparison adapters need adapter-step-NNNNNN=PATH")
        options.comparison_adapters.append((label, Path(path)))
    labels = [label for label, _ in options.comparison_adapters]
    if len(labels) > 2 or len(set(labels)) != len(labels):
        raise ValueError("At most two uniquely labelled comparison adapters are allowed")
    if labels and options.stage != "validation":
        raise ValueError("Multiple checkpoint candidates are validation-only")
    if options.selected_only and options.comparison_adapters:
        raise ValueError("Selected-only evaluation cannot include comparison adapters")
    if options.selected_only and options.stage != "evaluate":
        raise ValueError("--selected-only is a final evaluate option, not validation or training")
    options.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "min_pixels",
        "max_pixels",
        "max_tokens",
        "max_new_tokens",
        "max_seconds",
        "steps",
        "lora_rank",
        "gradient_accumulation",
        "checkpoint_every",
    ):
        if getattr(options, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if options.min_pixels > options.max_pixels:
        raise ValueError("min_pixels cannot exceed max_pixels")
    if options.stage == "prepare":
        result = assistant.prepare(options)
    elif options.stage == "train":
        if not options.manifest:
            raise ValueError("Training needs --manifest")
        result = assistant.run_train(options)
    elif options.stage in {"baseline", "validation", "evaluate"}:
        if not options.manifest:
            raise ValueError("Evaluation needs --manifest")
        if options.stage in {"baseline", "validation"} and options.split != "validation":
            raise ValueError("Baseline / adapter selection uses validation, never the final test split")
        result = assistant.run_evaluate(options, baseline=options.stage == "baseline")
    elif options.stage == "serve":
        from tiny_perceptron.natural_ui import serve

        serve(options, host=options.host, port=options.port)
        return
    else:
        root = options.data_root or Path.cwd()
        if options.audio:
            asr_model, asr_processor = assistant.load_asr(options)
            transcript = assistant.transcribe(asr_model, asr_processor, assistant.asset_path(options.audio, root))
            user = transcript["transcript"]
        else:
            transcript, user = None, options.user
        if user is None:
            raise ValueError("Chat needs --user or --audio")
        row = {
            "id": "interactive",
            "user": user,
            "image": options.image,
            "history": json.loads(options.history.read_text()) if options.history else [],
        }
        model, processor = assistant.load_core(options, adapter=options.adapter)
        result = assistant.generate(model, processor, row, root, options)
        result["asr"] = transcript
        result["history"] = assistant.messages_for(row, root) + [
            {"role": "assistant", "content": [{"type": "text", "text": result["prediction"]}]}
        ]
        assistant.write_json(options.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
