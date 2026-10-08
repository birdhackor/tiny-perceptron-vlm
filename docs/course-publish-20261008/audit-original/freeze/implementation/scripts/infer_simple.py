"""載入接字表／固定窗口 MLP；這是合成字元練習，不是一般聊天模型。"""

import argparse
import json

import torch

from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.training import choose_device


def load_simple_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") not in (None, "simple-v1"):
        raise ValueError("需要 simple-v1 或 train_simple.py 的舊字元 checkpoint")
    vocabulary = saved.get("vocabulary", saved.get("vocab"))
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("字元 checkpoint 缺少 vocabulary")
    if (
        any(not isinstance(char, str) or len(char) != 1 for char in vocabulary)
        or any(type(value) is not int for value in vocabulary.values())
        or set(vocabulary.values()) != set(range(2, len(vocabulary) + 2))
    ):
        raise ValueError("字表 ID 必須連續且唯一；0 保留邊界，1 保留 UNK")
    context, width = saved["context"], saved["width"]
    if type(context) is not int or type(width) is not int or min(context, width) < 1:
        raise ValueError("context/width 必須是正整數")
    if saved["kind"] == "bigram":
        if context != 1:
            raise ValueError("接字表的 context 必須是 1")
        model = BigramLM(len(vocabulary) + 2)
    elif saved["kind"] == "mlp":
        model = ContextMLP(len(vocabulary) + 2, context, width)
    else:
        raise ValueError("不支援的字元模型 kind；需要 bigram/mlp")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved, vocabulary


@torch.no_grad()
def generate_simple(model, vocabulary, context, prompt, tokens=24):
    if tokens < 1:
        raise ValueError("tokens 必須為正")
    reverse = {value: char for char, value in vocabulary.items()}
    history = [0] * context + [vocabulary.get(char, 1) for char in prompt]
    generated = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(tokens):
            inputs = torch.tensor([history[-context:]], device=next(model.parameters()).device)
            answer = int(model(inputs).argmax(-1).item())
            generated.append(answer)
            if answer == 0:
                break
            history.append(answer)
    finally:
        model.train(was_training)
    return {
        "answer": "".join(reverse.get(value, "<UNK>") for value in generated if value != 0),
        "generated_ids": generated,
        "eos": bool(generated and generated[-1] == 0),
        "unknown_generated": generated.count(1),
        "unknown_prompt_characters": sum(char not in vocabulary for char in prompt),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="顏色=")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model, saved, vocabulary = load_simple_checkpoint(args.checkpoint, choose_device(args.device))
    result = generate_simple(model, vocabulary, saved["context"], args.prompt, args.tokens)
    print(
        json.dumps(
            {
                "kind": saved["kind"],
                "context": saved["context"],
                "prompt": args.prompt,
                **result,
                "note": "僅供合成字元接續練習；邊界/EOS=0，未見字/UNK=1。",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
