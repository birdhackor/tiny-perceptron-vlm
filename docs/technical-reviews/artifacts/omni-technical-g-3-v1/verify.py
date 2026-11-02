"""G.3 的有界 CPU 核對；逐段執行明確連結教材裡的實際程式。"""

import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import SPECIALS
from tiny_perceptron.simple import ContextMLP

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
PAGES = {p["page_id"]: p for p in json.loads(FREEZE.read_text())["pages"]}
torch.set_num_threads(1)
torch.manual_seed(0)
records = []


def blocks(page):
    entry = PAGES[page]
    path = ROOT / entry["snapshot"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["source_sha256"]
    return re.findall(r"```python\n(.*?)```", raw.decode(), re.S)


def run_block(page, contains, namespace):
    selected = [block for block in blocks(page) if contains in block]
    assert len(selected) == 1, (page, contains, len(selected))
    code = selected[0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, f"{PAGES[page]['snapshot']}:{contains}", "exec"), namespace)
    records.append({
        "page": page,
        "contains": contains,
        "block_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": out.getvalue(),
    })


def main():
    # 6.6 的原樣 class，SPECIALS 來自同頁約定的實裝常數。
    ns = {"SPECIALS": SPECIALS}
    run_block("6.6", "class ByteTokenizer:", ns)
    tokenizer = ns["ByteTokenizer"]()
    dog = tokenizer.encode("狗")
    assert dog == [239, 147, 159] and tokenizer.decode(dog) == "狗"
    assert tokenizer.pad_id == 0
    records.append({"case": "byte_character_and_pad", "input": "狗", "ids": dog,
                    "roundtrip": tokenizer.decode(dog), "pad_id": tokenizer.pad_id})

    # 6.8 的原樣三個視窗算例；helper 實作另按 inspect 保存定位。
    ns = {}
    run_block("6.8", "context = 3", ns)
    assert ns["x"].tolist() == [6, 7, 8] and ns["y"].tolist() == [7, 8, 9]

    # 7.10 的內嵌 pad_batch，不以外部 helper 取代這份程式。
    ns = {"torch": torch, "IGNORE": -100}
    run_block("7.10", "def pad_batch(", ns)
    examples = [(torch.tensor([1, 2, 3]), torch.tensor([-100, 3, 2])),
                (torch.tensor([4]), torch.tensor([2]))]
    x, y, valid = ns["pad_batch"](examples)
    assert x.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert y.tolist() == [[-100, 3, 2], [2, -100, -100]]
    assert valid.tolist() == [[True, True, True], [True, False, False]]
    scores = torch.zeros(2, 3, 5, dtype=torch.float64, requires_grad=True)
    loss = F.cross_entropy(scores.reshape(-1, 5), y.reshape(-1), ignore_index=-100)
    loss.backward()
    assert abs(loss.item() - math.log(5)) < 1e-12
    assert (scores.grad[y == -100] == 0).all()
    records.append({"case": "pad_and_target_masks", "x": x.tolist(), "y": y.tolist(),
                    "valid": valid.tolist(), "effective_targets": int((y != -100).sum()),
                    "uniform_loss": loss.item(), "ignored_logits_gradient_zero": True,
                    "note": "第一筆第0格 valid=True，但目標忽略；讀取許可與監督不同。"})

    ns = {}
    run_block("7.5", "logits.retain_grad()", ns)
    direct = ns["logits"].grad[0, 2].norm().item()
    indirect = ns["model"].embedding.weight.grad[ns["x"][2]].norm().item()
    assert direct == 0.0 and indirect > 0.0
    records.append({"case": "prompt_visible_without_own_target", "direct_logit_gradient": direct,
                    "embedding_gradient_via_answer": indirect, "optimizer_steps": 0})

    ns = {}
    run_block("7.19", "T = 16", ns)
    assert ns["P"] == 12 and ns["T"] == 16

    # 2.5/T.3 的實際類別與命令所選 ContextMLP；只檢查構造，不訓練。
    model = ContextMLP(17, context=3, width=16)
    contexts = torch.tensor([[1, 2, 3], [3, 2, 1]])
    embedding = model.embedding(contexts)
    flat = embedding.flatten(1)
    logits = model(contexts)
    assert embedding.shape == (2, 3, 16) and flat.shape == (2, 48) and logits.shape == (2, 17)
    assert model.hidden.in_features == 48 and model.hidden.out_features == 16
    assert model.output.in_features == 16 and model.output.out_features == 17
    records.append({"case": "context_mlp", "context": 3, "width": 16,
                    "embedding_shape": list(embedding.shape), "flatten_shape": list(flat.shape),
                    "output_shape": list(logits.shape), "parameter_count": sum(p.numel() for p in model.parameters()),
                    "trained": False})

    ns = {}
    run_block("10.6", "vision = torch.randn", ns)
    assert ns["language_features"].shape == (1, 16, 12)
    ns = {"torch": torch, "nn": nn, "F": F}
    run_block("4.4", "class DenseFFN", ns)
    ffn = ns["DenseFFN"](4)
    assert ffn(torch.ones(1, 2, 4)).shape == (1, 2, 4)
    records.append({"case": "positionwise_ffn", "up_weight": list(ffn.up.weight.shape),
                    "down_weight": list(ffn.down.weight.shape), "input_and_output": [1, 2, 4]})

    ns = {}
    run_block("14.2", "ln = F.layer_norm", ns)
    run_block("14.2", "class RMSNorm", ns)
    assert torch.allclose(ns["RMSNorm"](3)(ns["x"]), ns["rms"], atol=1e-6)
    unit_inputs = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1e-15, 0.0]], dtype=torch.float64)
    normalized = F.normalize(unit_inputs, p=2, dim=-1, eps=1e-12)
    lengths = normalized.norm(dim=-1)
    assert torch.allclose(lengths, torch.tensor([1.0, 0.0, 0.001], dtype=torch.float64), atol=1e-15)
    records.append({"case": "l2_epsilon_edge", "input": unit_inputs.tolist(),
                    "normalized": normalized.tolist(), "lengths": lengths.tolist(), "eps": 1e-12})

    ns = {}
    run_block("5.11", 'docs = ["貓', ns)
    assert ns["normalized"] == ["貓 看狗", "貓 看狗", "狗 看貓"]

    ns = {"torch": torch, "math": math}
    run_block("12.6", "def mel_filter_bank", ns)
    bank = ns["mel_filter_bank"](bands=16)
    power = torch.ones(201, 3)
    counts = (bank > 0).sum(1)
    assert bank.shape == (16, 201) and (bank @ power).shape == (16, 3)
    assert bool((bank >= 0).all()) and counts[0] == 6 and counts[-1] == 55
    records.append({"case": "embedded_mel_bank", "shape": list(bank.shape),
                    "mel_power_shape": list((bank @ power).shape), "positive_bins": counts.tolist()})

    print(json.dumps({"started_and_finished_at_utc": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-g-3-v1/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torch_git": str(torch.version.git_version), "device": "cpu", "threads": "1"},
        "repository_dependencies": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in ["tiny_perceptron/data.py", "tiny_perceptron/simple.py", "tiny_perceptron/model.py",
                         "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "scripts/train_simple.py"]
        },
        "result": "all bounded assertions passed; no optimizer step, training, GPU, or model quality measurement",
        "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
