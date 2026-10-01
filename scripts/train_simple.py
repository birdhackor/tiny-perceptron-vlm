"""Bigram／MLP 字元練習；預設只驗梯度，--train 才更新與存檔。"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.data import load_jsonl, split_documents, toy_documents
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.training import choose_device, seed_everything


def examples(documents, vocabulary, context):
    rows, labels = [], []
    for document in documents:
        ids = [vocabulary.get(c, 1) for c in document] + [0]
        padded = [0] * context + ids
        for i, target in enumerate(ids):
            rows.append(padded[i : i + context])
            labels.append(target)
    return torch.tensor(rows), torch.tensor(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["bigram", "mlp"], default="bigram")
    p.add_argument("--train", action="store_true")
    p.add_argument("--data", type=Path)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("checkpoints/simple.pt"))
    args = p.parse_args()
    if min(args.steps, args.width, args.context) < 1:
        raise ValueError("steps/width/context 必須為正")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    documents = [r["text"] for r in load_jsonl(args.data)] if args.data else toy_documents()
    split = split_documents(documents, args.seed)
    if not split["train"] or not split["validation"]:
        raise ValueError("至少提供足夠的不同文件，才能分出訓練／驗證資料")
    # 0 是邊界，1 是未知字；詞表只看 train，避免先偷看 validation。
    vocabulary = {c: i + 2 for i, c in enumerate(sorted(set("".join(split["train"]))))}
    device = choose_device(args.device)
    context = 1 if args.model == "bigram" else args.context
    model = (
        BigramLM(len(vocabulary) + 2)
        if args.model == "bigram"
        else ContextMLP(len(vocabulary) + 2, context, args.width)
    ).to(device)
    x, y = (t.to(device) for t in examples(split["train"], vocabulary, context))
    vx, vy = (t.to(device) for t in examples(split["validation"], vocabulary, context))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(args.steps if args.train else 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        if args.train:
            optimizer.step()
    with torch.no_grad():
        validation = F.cross_entropy(model(vx), vy).item()
    report = {
        "mode": "train" if args.train else "dry-run-no-weight-update",
        "model": args.model,
        "train_loss": loss.detach().item(),
        "validation_loss": validation,
        "split_unit": "whole deduplicated document",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    if args.train:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "vocabulary": vocabulary,
                "context": context,
                "width": args.width,
                "kind": args.model,
                "report": report,
            },
            args.output,
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
