"""C-v3 紀錄集合模型的 CPU 開發訓練；不建立、讀取或評分 final。"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from .data import KINDS, action_gold, evaluation_examples, info_gold, make_example, make_families, rubric_gold
from .model import Tokenizer, collate
from .record_model import RecordConfig as Config
from .record_model import RecordDecisionModel as DecisionModel
from .record_model import encode_record_example as encode_example


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_receipts():
    return {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).parent / name for name in ("data.py", "model.py", "record_model.py", "record_experiment.py")
        )
    }


@torch.inference_mode()
def evaluate(model, examples, encoded, batch_size=32):
    model.eval()
    groups = defaultdict(lambda: [0, 0])
    total_loss = 0.0
    for start in range(0, len(encoded), batch_size):
        batch = collate(encoded[start : start + batch_size], next(model.parameters()).device)
        logits = model(batch)
        total_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
        predictions = logits.argmax(-1).tolist()
        for example, row, prediction in zip(
            examples[start : start + batch_size], encoded[start : start + batch_size], predictions
        ):
            correct = int(prediction == row["label"])
            for key in (
                example.kind,
                f"{example.kind}/source/{example.source}",
                f"{example.kind}/panel/{example.panel}",
                f"{example.kind}/gold/{example.gold if example.kind != 'choice' else ('none' if example.gold == 'none' else 'answer')}",
            ):
                groups[key][0] += correct
                groups[key][1] += 1
    metrics = {key: {"correct": c, "n": n, "accuracy": c / n} for key, (c, n) in sorted(groups.items())}
    metrics["nll"] = total_loss / len(encoded)
    # 最差核心題型優先；所有題型皆達標才比較其餘表現。
    metrics["selection_score"] = min(value["accuracy"] for key, value in metrics.items() if "/gold/" in key)
    model.train()
    return metrics


def train(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    current_arguments = {key: value for key, value in vars(args).items() if key not in {"output", "threads"}}
    resumed = (root / "latest.pt").exists()
    if resumed:
        checkpoint = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint["sources"] != source_receipts():
            raise ValueError("來源已變更：請另開開發實驗，不能默默沿用舊 checkpoint")
        plan = json.loads((root / "plan.json").read_text())
        if current_arguments != plan["arguments"]:
            raise ValueError("resume 須維持全部資料、模型與訓練參數")
        families = json.loads((root / "train-families.json").read_text())
        curriculum = json.loads((root / "curriculum-families.json").read_text())
        dev = make_families("dev", args.dev_families, 826021)
        tokenizer = Tokenizer(json.loads((root / "tokenizer.json").read_text()))
    else:
        if (root / "plan.json").exists():
            raise ValueError("實驗已存在但缺 resume checkpoint，請使用新目錄")
        families = make_families("train", args.families, 826019)
        curriculum = make_families("train", 1000, 826039, min_devices=1, max_devices=3)
        dev = make_families("dev", args.dev_families, 826021)
        vocab_examples = evaluation_examples(families[:1000], 826023)
        tokenizer = Tokenizer.fit(
            text
            for example in vocab_examples
            for text in (example.state, example.question, *(description for _, description in example.candidates))
        )
        tokenizer.save(root / "tokenizer.json")
        write_json(root / "train-families.json", families)
        write_json(root / "curriculum-families.json", curriculum)
        plan = {
            "created_at": datetime.now(UTC).isoformat(),
            "schema": "c-product-record-development-v1",
            "architecture": "record-local-state-set-v1",
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "families": args.families,
            "dev_families": args.dev_families,
            "learning_rate": args.learning_rate,
            "arguments": current_arguments,
            "source_receipts": source_receipts(),
            "model": {
                "width": args.width,
                "state_layers": args.state_layers,
                "branch_layers": 1,
                "readers": args.readers,
            },
            "training": "fresh random initialization; equal kind and target-class sampling; 65% focused same-device candidates; first600 updates1-2devices, next600 three devices, then90%4-6devices+10%curriculum; auxiliary task/permission/ready CE weight0.5; no routes/pretrained/teacher/gold feature input",
            "selection": "maximum minimum dev class accuracy, then mean kind accuracy, then NLL",
            "test_scope": "No final generated or evaluated; old exposed final not used for selection",
            "resource": "existing local CPU; no paid Modal compute",
        }
        write_json(root / "plan.json", plan)
    examples = evaluation_examples(dev, 826025)
    write_json(root / "dev-examples.json", [example.to_dict() for example in examples])
    encoded = [encode_example(example, tokenizer) for example in examples]
    config = Config(tokenizer.size, width=args.width, state_layers=args.state_layers, readers=args.readers)
    model = DecisionModel(config)
    stages = [
        [family for family in curriculum if len(family["facts"]) <= 2],
        [family for family in curriculum if len(family["facts"]) == 3],
        families,
        curriculum,
    ]
    selectors = []
    for stage in stages:
        groups = {kind: defaultdict(list) for kind in KINDS}
        for family in stage:
            fact = family["facts"][0]
            for kind, label in (
                ("choice", action_gold(fact)),
                ("info", info_gold(fact)),
                ("rubric", rubric_gold(fact)),
            ):
                groups[kind][label].append(family)
        selectors.append(groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    start_step, best, best_step = 0, (-1.0, -1.0, -math.inf), 0
    if resumed:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start_step, best, best_step = checkpoint["step"], tuple(checkpoint["best"]), checkpoint["best_step"]
    log = (root / "training.jsonl").open("a")
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "training_started",
                "parameters": sum(p.numel() for p in model.parameters()),
                "vocabulary": tokenizer.size,
                "model": model.metadata(),
                "start_step": start_step,
                "dev_observations": len(examples),
                "threads": args.threads,
            }
        ),
        flush=True,
    )
    loss_sum, interval_start = 0.0, start_step
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        rows, targets = [], []
        for i in range(args.batch_size):
            kind = KINDS[(step * args.batch_size + i) % 3]
            stage = 0 if step <= 600 else 1 if step <= 1200 else 3 if rng.random() < 0.1 else 2
            groups = selectors[stage][kind]
            family = rng.choice(groups[rng.choice(tuple(sorted(groups)))])
            example = make_example(family, rng, kind=kind, focused=rng.random() < 0.65)
            rows.append(encode_example(example, tokenizer))
            task, permission, ready = family["facts"][0]
            targets.append((task, permission + 1, ready + 1))
        batch = collate(rows)
        warmup = min(150, args.steps // 10)
        multiplier = (
            step / warmup
            if step <= warmup
            else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(batch, return_auxiliary=True)
        auxiliary_targets = torch.tensor(targets)
        task_loss = F.cross_entropy(logits, batch["labels"])
        auxiliary_loss = F.cross_entropy(auxiliary.reshape(-1, 3), auxiliary_targets.reshape(-1))
        loss = task_loss + 0.5 * auxiliary_loss
        if not torch.isfinite(loss):
            raise RuntimeError("loss 非有限值，停止並保留原始紀錄")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        if step % 50 == 0:
            event = {
                "event": "train",
                "step": step,
                "loss": loss_sum / (step - interval_start),
                "elapsed_seconds": time.monotonic() - started,
                "lr": optimizer.param_groups[0]["lr"],
                "last_task_loss": float(task_loss.detach()),
                "last_auxiliary_loss": float(auxiliary_loss.detach()),
            }
            print(json.dumps(event), flush=True)
            log.write(json.dumps(event) + "\n")
            log.flush()
            loss_sum, interval_start = 0.0, step
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(model, examples, encoded, args.batch_size)
            selection = (metrics["selection_score"], sum(metrics[k]["accuracy"] for k in KINDS) / 3, -metrics["nll"])
            if selection > best:
                best, best_step = selection, step
                save_file(
                    {key: value.detach().contiguous().clone() for key, value in model.state_dict().items()},
                    root / "best.safetensors",
                )
                write_json(
                    root / "best.json",
                    {
                        "step": step,
                        "config": model.metadata(),
                        "metrics": metrics,
                        "weights_sha256": hashlib.sha256((root / "best.safetensors").read_bytes()).hexdigest(),
                    },
                )
            event = {
                "event": "dev",
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "best_step": best_step,
                "metrics": metrics,
            }
            log.write(json.dumps(event) + "\n")
            log.flush()
            print(
                json.dumps({**event, "metrics": {key: metrics[key] for key in (*KINDS, "nll", "selection_score")}}),
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "random_state": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "step": step,
                    "best": best,
                    "best_step": best_step,
                    "sources": source_receipts(),
                },
                root / "latest.pt",
            )
    log.close()
    model.load_state_dict(load_file(root / "best.safetensors"))
    print(
        json.dumps(
            {"event": "training_complete", "best_step": best_step, "elapsed_seconds": time.monotonic() - started}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--families", type=int, default=8000)
    parser.add_argument("--dev-families", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validate-every", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--state-layers", type=int, default=2)
    parser.add_argument("--readers", type=int, default=2)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
