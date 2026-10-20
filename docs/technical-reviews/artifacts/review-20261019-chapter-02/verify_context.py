"""核對 BigramLM 與 ContextMLP 的輸入、計算路徑及輸入可見性的限制。"""

import hashlib
import json
import platform
from pathlib import Path

import torch

from tiny_perceptron.simple import BigramLM, ContextMLP


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    prefixes = ["顏色=", "形狀="]
    chars = sorted(set("".join(prefixes)))
    ids = torch.tensor([[chars.index(char) for char in text] for text in prefixes])
    vocab_size = len(chars)
    bigram = BigramLM(vocab_size)
    model = ContextMLP(vocab_size, context=3, width=2)
    with torch.no_grad():
        bigram.table.weight.copy_(torch.arange(vocab_size**2).reshape(vocab_size, vocab_size))
        model.embedding.weight.copy_(torch.arange(vocab_size * 2).reshape(vocab_size, 2) / 10)
        model.hidden.weight.copy_(torch.arange(12).reshape(2, 6) / 10)
        model.hidden.bias.copy_(torch.tensor([0.1, -0.2]))
        model.output.weight.copy_(torch.arange(vocab_size * 2).reshape(vocab_size, 2) / 10)
        model.output.bias.copy_(torch.arange(vocab_size) / 10)

    bigram_logits = bigram(ids)
    embeddings = model.embedding(ids)
    flat = embeddings.flatten(1)
    hidden = torch.tanh(flat @ model.hidden.weight.T + model.hidden.bias)
    manual_logits = hidden @ model.output.weight.T + model.output.bias
    captured_hidden_inputs = []
    handle = model.hidden.register_forward_pre_hook(
        lambda module, args: captured_hidden_inputs.append(args[0].detach().clone())
    )
    logits = model(ids)
    handle.remove()
    logits.sum().backward()
    gradients = {name: bool(parameter.grad.abs().sum() > 0) for name, parameter in model.named_parameters()}

    with torch.no_grad():
        model.hidden.weight.zero_()
    ignored_logits = model(ids)
    checks = {
        "兩個前文最後一字都是等號": all(text[-1] == "=" for text in prefixes),
        "接字表只選最後一字的列": torch.equal(bigram_logits, bigram.table.weight[ids[:, -1]]),
        "接字表對兩個前文產生相同分數": torch.equal(bigram_logits[0], bigram_logits[1]),
        "每字依索引取出對應參數列": torch.equal(embeddings, model.embedding.weight[ids]),
        "展平順序等於按位置拼接": torch.equal(flat, torch.cat([embeddings[:, i] for i in range(3)], dim=1)),
        "前文所有位置確實送入 hidden": torch.equal(captured_hidden_inputs[0], flat),
        "不同前文的拼接特徵不同": not torch.equal(flat[0], flat[1]),
        "候選分數符合逐層公式": torch.allclose(logits, manual_logits, rtol=1e-6, atol=1e-6),
        "所有參數均可求導且本次梯度非零": all(gradients.values()),
        "可見不同前文仍容許相同候選分數": torch.equal(ignored_logits[0], ignored_logits[1]),
    }
    assert all(checks.values()), "輸入或逐層公式核對失敗"
    receipt = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-02/verify_context.py",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "dtype": str(embeddings.dtype),
            "platform": platform.platform(),
        },
        "repository_code_sha256": hashlib.sha256(Path("tiny_perceptron/simple.py").read_bytes()).hexdigest(),
        "configuration": {"prefixes": prefixes, "chars": chars, "ids": ids.tolist(), "context": 3, "width": 2, "seed": 7},
        "bigram": {"selected_ids": ids[:, -1].tolist(), "logits": bigram_logits.detach().tolist()},
        "mlp": {
            "embedding_shape": list(embeddings.shape),
            "flattened": flat.detach().tolist(),
            "hidden": hidden.detach().tolist(),
            "logits": logits.detach().tolist(),
            "parameter_gradients_nonzero": gradients,
        },
        "counterexample": {
            "change": "hidden.weight 設成零；輸入、embedding、bias 與 output 其餘參數不變。",
            "formula": "hidden=tanh(0*x+b_hidden)，所以 hidden 與最終候選分數都可與前文無關。",
            "logits": ignored_logits.detach().tolist(),
        },
        "checks": checks,
        "result": "全部核對通過；未訓練模型，未量測品質或速度。",
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
