"""獨立核對對話標籤、byte 計數、loss 與注意力的分工。"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, data):
    path = ARTIFACTS / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def fetch_sources():
    commit = torch.version.git_version
    specs = [
        {
            "id": "official-loss",
            "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/loss.py",
            "version": f"PyTorch {torch.__version__}; git {commit}",
            "locator": "CrossEntropyLoss: class-index loss equation and ignore_index argument",
            "excerpt": "Specifies a target value that is ignored\n            and does not contribute to the input gradient.",
            "markers": ["class CrossEntropyLoss", "ignore_index", "label_smoothing"],
        },
        {
            "id": "official-attention",
            "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html",
            "version": "PyTorch 2.14 versioned documentation",
            "locator": "scaled_dot_product_attention: equivalent implementation and attn_mask parameter",
            "excerpt": "A boolean mask where a value of True indicates that the element should take part in attention.",
            "markers": ["scaled_dot_product_attention(query", "attn_mask", "attn_weight"],
        },
        {
            "id": "official-python",
            "url": "https://docs.python.org/3.13/library/functions.html",
            "version": "Python 3.13.16 documentation",
            "locator": "enumerate and zip builtins",
            "excerpt": "Return an enumerate object.",
            "markers": ["enumerate", "zip", "strict=True"],
        },
    ]
    receipts = []
    for spec in specs:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", spec["url"]]
        completed = subprocess.run(command, capture_output=True, check=True)
        original = completed.stdout
        text = original.decode("utf-8")
        if spec["url"].endswith(".html"):
            parser = PlainText()
            parser.feed(text)
            text = " ".join(parser.parts)
        normalized = " ".join(text.split())
        excerpt = " ".join(spec["excerpt"].split())
        assert excerpt in normalized
        assert all(marker in normalized for marker in spec["markers"])
        receipts.append({
            **spec,
            "command_argv": command,
            "exit_code": completed.returncode,
            "stderr": completed.stderr.decode(),
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "downloaded_bytes": len(original),
            "downloaded_sha256": digest(original),
            "checked_original": True,
            "full_text_retained": False,
        })
    save("original-sources.json", {"sources": receipts})


def verify(section_run):
    assert section_run.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    bundle = {}
    for name in ("extraction.json", "environment.json", "execution.json"):
        bundle[name] = json.loads((section_run / name).read_text())
    for name in ("stdout.txt", "stderr.txt", "fence-1.py"):
        bundle[name] = (section_run / name).read_text()
    bundle["invocation"] = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/07.md#7.3' --execute"
    assert bundle["execution.json"]["exit_code"] == 0
    assert bundle["extraction.json"]["source_sha256"] == "c19d14077bb6208e4b989554972cdec71baa24c777ca1ea82b15d9b92d9f29a8"
    assert bundle["extraction.json"]["figure_sha256"] == {}
    save("section-execution.json", bundle)

    torch.set_num_threads(1)
    torch.manual_seed(42)
    tok = ByteTokenizer()
    cases = []
    expected_inputs = {
        "答": [1, 3, 237, 157, 151, 2, 4, 239, 181, 156],
        "答案": [1, 3, 237, 157, 151, 2, 4, 239, 181, 156, 238, 169, 144],
        "OK": [1, 3, 237, 157, 151, 2, 4, 87, 83],
    }
    expected_targets = {
        "答": [-100, -100, -100, -100, -100, -100, 239, 181, 156, 2],
        "答案": [-100, -100, -100, -100, -100, -100, 239, 181, 156, 238, 169, 144, 2],
        "OK": [-100, -100, -100, -100, -100, -100, 87, 83, 2],
    }
    for answer, byte_hex, expected_count in (("答", "e7ad94", 4), ("答案", "e7ad94e6a188", 7), ("OK", "4f4b", 3)):
        messages = [{"role": "user", "content": "問"}, {"role": "assistant", "content": answer}]
        x, y = render_chat(messages)
        explicit_x, explicit_y = render_chat(messages, tok)
        raw = answer.encode("utf-8")
        assert raw.hex() == byte_hex
        assert "問".encode("utf-8").hex() == "e5958f"
        assert x.tolist() == expected_inputs[answer]
        assert y.tolist() == expected_targets[answer]
        assert torch.equal(x, explicit_x) and torch.equal(y, explicit_y)
        assert x.dtype == y.dtype == torch.int64
        valid = y != -100
        count = valid.sum().item()
        assert valid.dtype == torch.bool
        assert type(count) is int and count == expected_count
        rows = list(enumerate(zip(x.tolist(), y.tolist(), strict=True)))
        assert len(rows) == len(x)
        assert x[2:5].tolist() == tok.encode("問")
        assert x.min().item() >= 0
        statuses = ["計代價" if label != -100 else "不計代價" for label in y.tolist()]
        assert statuses == ["不計代價"] * 6 + ["計代價"] * expected_count
        cases.append({
            "answer": answer, "utf8_hex": raw.hex(), "byte_count": len(raw),
            "x": x.tolist(), "y": y.tolist(), "length": len(x),
            "x_dtype": str(x.dtype), "y_dtype": str(y.dtype),
            "mask_dtype": str(valid.dtype), "count_type": type(count).__name__,
            "effective_targets": count, "statuses": statuses,
            "default_matches_explicit_tokenizer": True,
            "enumerated_rows": rows,
        })

    messages = [{"role": "user", "content": "問"}, {"role": "assistant", "content": "答"}]
    x, y = render_chat(messages)
    model = TinyLM(ModelConfig(width=8))
    embedding_inputs = []
    handle = model.embedding.register_forward_pre_hook(lambda module, args: embedding_inputs.append(args[0].tolist()))
    logits = model(x[None])["logits"]
    handle.remove()
    logits.retain_grad()
    loss = masked_loss(logits, y[None])
    mask = y != -100
    selected = logits[0, mask]
    manual = -selected.log_softmax(-1)[torch.arange(mask.sum()), y[mask]].mean()
    assert torch.allclose(loss, manual, atol=1e-6, rtol=0)
    per_position = F.cross_entropy(logits[0], y, ignore_index=-100, reduction="none")
    assert torch.equal(per_position[~mask], torch.zeros_like(per_position[~mask]))
    loss.backward()
    assert logits.grad[0, ~mask].abs().max().item() == 0
    question_grads = model.embedding.weight.grad[x[2:5]].norm(dim=1)
    assert (question_grads > 0).all().item()
    positions = torch.arange(len(x))
    allowed = attention_mask(positions, positions)[0, 0]
    assert allowed[6, 2:5].all().item()
    assert embedding_inputs == [x[None].tolist()]
    q = torch.zeros(1, 1, len(x), 2)
    values = torch.arange(len(x), dtype=torch.float32)[None, None, :, None]
    weights_output = F.scaled_dot_product_attention(q, q, values, attn_mask=allowed, dropout_p=0.0)
    assert weights_output[0, 0, 6, 0].item() == 3.0
    changed_values = values.clone()
    changed_values[:, :, 2:5] += 1
    changed_output = F.scaled_dot_product_attention(q, q, changed_values, attn_mask=allowed, dropout_p=0.0)
    assert changed_output[0, 0, 6, 0].item() > weights_output[0, 0, 6, 0].item()
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-3/verify.py --section-run " + str(section_run.relative_to(ROOT)),
        "environment": {
            "python": sys.version, "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version), "device": "cpu",
            "platform": platform.platform(), "seed": "42", "threads": "1",
            "model_dtype": str(next(model.parameters()).dtype),
        },
        "source_files": {path: digest((ROOT / path).read_bytes()) for path in (
            "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
        )},
        "question_utf8_hex": "問".encode("utf-8").hex(),
        "question_byte_count": len("問".encode("utf-8")),
        "cases": cases,
        "loss_and_attention": {
            "embedding_inputs": embedding_inputs, "loss": loss.item(), "manual_loss": manual.item(),
            "per_position_loss": per_position.detach().tolist(),
            "ignored_position_max_logit_gradient": logits.grad[0, ~mask].abs().max().item(),
            "question_embedding_gradient_norms": question_grads.tolist(),
            "first_supervised_position": int(mask.nonzero()[0].item()),
            "first_supervised_position_can_read_question": allowed[6, 2:5].tolist(),
            "attention_output_at_first_supervised_position": weights_output[0, 0, 6, 0].item(),
            "attention_output_after_question_values_change": changed_output[0, 0, 6, 0].item(),
        },
        "result": "全部斷言通過；已核對完整 ID、目標、dtype、計數、loss 分母、embedding 輸入與注意力可見性。",
    }
    save("verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-run", type=Path, required=True)
    args = parser.parse_args()
    fetch_sources()
    verify(args.section_run.resolve())
