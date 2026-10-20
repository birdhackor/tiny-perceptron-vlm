"""核對回答對齊、練習變體與同位置的 loss；不更新模型參數。"""

import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def example(question, answer):
    x, y = render_chat(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    )
    positions = (y != IGNORE).nonzero()
    first = positions[0].item()
    return x, y, {
        "question": question,
        "answer": answer,
        "x": x.tolist(),
        "y": y.tolist(),
        "dtype": str(x.dtype),
        "nonzero": positions.tolist(),
        "nonzero_shape": list(positions.shape),
        "first_type": type(first).__name__,
        "first": first,
        "first_input": x[first].item(),
        "first_target": y[first].item(),
        "effective_count": (y != IGNORE).sum().item(),
        "effective_pairs": [
            {"position": i.item(), "input": x[i].item(), "target": y[i].item()}
            for i in positions[:, 0]
        ],
    }


def main():
    torch.set_num_threads(1)
    tok = ByteTokenizer()
    x, y, base = example("Q", "A")
    assert x.tolist() == [1, 3, 89, 2, 4, 73]
    assert y.tolist() == [-100, -100, -100, -100, 73, 2]
    assert base["first"] == 4
    assert x[base["first"]].item() == tok.assistant_id
    assert y[base["first"]].item() == tok.encode("A")[0]
    assert base["nonzero"] == [[4], [5]]

    double_x, double_y = x[:-1], y[1:]
    double_first = (double_y != IGNORE).nonzero()[0].item()
    double = {
        "x": double_x.tolist(),
        "y": double_y.tolist(),
        "first": double_first,
        "first_input": double_x[double_first].item(),
        "first_target": double_y[double_first].item(),
        "effective_count": (double_y != IGNORE).sum().item(),
    }
    assert double_x.tolist() == [1, 3, 89, 2, 4]
    assert double_y.tolist() == [-100, -100, -100, 73, 2]
    assert double["effective_count"] == base["effective_count"] == 2
    assert double["first_input"] == tok.eos_id
    assert double_x[4].item() == tok.assistant_id and double_y[4].item() == tok.eos_id

    _, _, longer = example("QQ", "A")
    bx, by, changed = example("QQ", "B")
    assert (longer["first"], longer["first_input"], longer["first_target"]) == (5, 4, 73)
    assert (changed["first"], changed["first_input"], changed["first_target"]) == (5, 4, 74)
    try:
        assert by[changed["first"]].item() == tok.encode("A")[0]
    except AssertionError:
        changed["assert_answer_A"] = "AssertionError"
    else:
        raise AssertionError("改成 B 後仍與 A 相等")
    assert bx[changed["first"]].item() == tok.assistant_id
    assert by[changed["first"]].item() == tok.encode("B")[0]
    changed["assert_answer_B"] = "passed"

    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=8, heads=1, layers=1)).eval()
    model_api = {"forward_signature": str(inspect.signature(model.forward))}
    try:
        model(x[None], labels=y[None])
    except TypeError as error:
        model_api["labels_call_error_type"] = type(error).__name__
        model_api["labels_call_error"] = str(error)
    else:
        raise AssertionError("TinyLM 接受了預期不支援的 labels 參數")
    with torch.no_grad():
        logits = model(x[None])["logits"]
        model_api["return_keys"] = sorted(model(x[None]))
        per_position = []
        for i in range(x.numel()):
            prefix_logits = model(x[: i + 1][None])["logits"][:, -1]
            per_position.append((logits[:, i] - prefix_logits).abs().max().item())
        assert logits.shape == (1, 6, 264)
        assert max(per_position) < 1e-5
        model_loss = masked_loss(logits, y[None])
        logp = logits.log_softmax(-1)
        direct = -(logp[0, 4, 73] + logp[0, 5, 2]) / 2
        assert torch.allclose(model_loss, direct, atol=1e-6, rtol=0)

    scores = torch.zeros((1, 6, tok.vocab_size), requires_grad=True)
    loss = masked_loss(scores, y[None])
    loss.backward()
    gradient = scores.grad
    assert torch.equal(gradient[0, :4], torch.zeros_like(gradient[0, :4]))
    assert gradient[0, 4].abs().sum() > 0 and gradient[0, 5].abs().sum() > 0
    assert gradient[0, 4, 73] < 0 and gradient[0, 5, 2] < 0
    allowed = attention_mask(torch.arange(6), torch.arange(6))[0, 0]
    assert allowed[4].tolist() == [True, True, True, True, True, False]

    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-4/alignment-probe.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform(), "seed": "42", "threads": "1"},
        "result": "全部斷言通過；沒有訓練或參數更新。",
        "source_files": {p: digest(p) for p in ("tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py")},
        "byte_mapping": {s: {"utf8_bytes": list(s.encode("utf-8")), "ids": tok.encode(s)} for s in ("Q", "A", "B")},
        "original_ids": [tok.bos_id, tok.user_id, *tok.encode("Q"), tok.eos_id, tok.assistant_id, *tok.encode("A"), tok.eos_id],
        "base": base, "double_shift": double, "longer_question": longer, "changed_answer": changed,
        "model_api": model_api,
        "model": {"logits_shape": list(logits.shape), "prefix_max_abs_differences": per_position, "prefix_tolerance": 1e-5, "masked_loss": model_loss.item(), "direct_same_position_loss": direct.item(), "loss_tolerance": 1e-6, "gradient_l1_by_position": gradient.abs().sum(-1).tolist(), "assistant_position_A_gradient": gradient[0, 4, 73].item(), "answer_position_EOS_gradient": gradient[0, 5, 2].item(), "allowed_keys_at_assistant": allowed[4].tolist()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
