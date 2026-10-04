"""Train, inspect and locally serve the final synthetic small-world assistant."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--stage", choices=("pretrain", "sft", "joint", "dpo"), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--input-checkpoint")
    train.add_argument("--resume")
    train.add_argument("--device", default="cpu")
    train.add_argument("--steps", type=int)
    train.add_argument("--seconds", type=float, default=540)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=24)
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--calculator-disabled", action="store_true")
    infer.add_argument("--image-color", choices=("red", "green", "blue"))
    infer.add_argument("--image-shape", choices=("square", "circle"), default="square")
    infer.add_argument("--audio-pitch", choices=("low", "high"))
    infer.add_argument("--max-new-tokens", type=int, default=64)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--seed", type=int, default=42)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    from tiny_perceptron.capstone import build_dataset, evaluate_rows, export_inference, load_capstone, run_assistant

    if args.command == "train":
        from scripts.course_experiments.capstone import train_stage

        report = train_stage(
            args.stage,
            args.output,
            input_checkpoint=args.input_checkpoint,
            resume=args.resume,
            device=args.device,
            steps=args.steps,
            seconds=args.seconds,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        result = {key: value for key, value in report.items() if key not in ("data_manifest", "history")}
    elif args.command == "export":
        result = export_inference(args.checkpoint, args.output)
    elif args.command == "serve":
        from tiny_perceptron.capstone_ui import serve

        serve(args.checkpoint, host=args.host, port=args.port)
        return
    else:
        import torch

        torch.set_num_threads(2)
        try:
            model, _ = load_capstone(args.checkpoint, args.device)
        except ValueError as error:
            from tiny_perceptron.capstone_quantization import load_quantized_capstone

            try:
                model, _ = load_quantized_capstone(args.checkpoint, args.device)
            except ValueError:
                raise error
        if args.command == "infer":
            available = not args.calculator_disabled
            row = {
                "user": args.prompt,
                "system": f"計算器={'開' if available else '關'}；風格=短。",
                "available": available,
                "image": None,
                "audio": None,
            }
            if args.image_color:
                row["image"] = {"color": args.image_color, "shape": args.image_shape, "offset": 0, "variant": 1}
            if args.audio_pitch:
                row["audio"] = {"pitch": args.audio_pitch, "variation": 1}
            result = run_assistant(model, row, max_new_tokens=args.max_new_tokens)
        else:
            from scripts.course_experiments.capstone import write_json

            splits, manifest = build_dataset(args.seed)
            result = evaluate_rows(model, splits[args.split])
            result["data_manifest"] = manifest
            result["split"] = args.split
            write_json(args.output, result)
            result = {key: value for key, value in result.items() if key not in ("records", "data_manifest")}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
