"""合成規則資料的視覺／音訊encoder分類入口；預設不更新權重。"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.multimodal import AudioEncoder, VisionEncoder, scene, tone
from tiny_perceptron.training import choose_device, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=["vision", "audio"], default="vision")
    p.add_argument("--train", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.steps < 1:
        raise ValueError("steps必須為正")
    seed_everything(args.seed)
    device = choose_device(args.device)
    encoder = (VisionEncoder() if args.modality == "vision" else AudioEncoder()).to(device)
    classifier = nn.Linear(16, 2).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    if args.modality == "vision":
        items = [
            (scene(c, s, offset=o), int(s == "circle"), c == "blue" and o == 1)
            for c in ("red", "green", "blue")
            for s in ("square", "circle")
            for o in (-1, 0, 1)
        ]
    else:
        items = [(tone(f), int(f > 500), f in (180, 1000)) for f in (180, 220, 260, 780, 880, 1000)]
    training = [(x, y) for x, y, test in items if not test]
    holdout = [(x, y) for x, y, test in items if test]
    inputs = torch.stack([x for x, y in training]).to(device)
    labels = torch.tensor([y for x, y in training], device=device)
    history = []
    for _ in range(args.steps if args.train else 1):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(encoder(inputs).mean(1))
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if args.train:
            optimizer.step()
        history.append(loss.detach().item())
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        x = torch.stack([x for x, y in holdout]).to(device)
        y = torch.tensor([y for x, y in holdout], device=device)
        accuracy = (classifier(encoder(x).mean(1)).argmax(-1) == y).float().mean().item()
    report = {
        "mode": "train" if args.train else "dry-run-no-weight-update",
        "modality": args.modality,
        "train_examples": len(training),
        "holdout_examples": len(holdout),
        "loss": history,
        "holdout_accuracy": accuracy,
        "note": "dry-run accuracy來自隨機權重，不是已學能力；此holdout僅限合成規則。",
    }
    if args.train:
        output = args.output or Path(f"checkpoints/{args.modality}-encoder.pt")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": encoder.state_dict(), "classifier": classifier.state_dict(), "report": report}, output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
