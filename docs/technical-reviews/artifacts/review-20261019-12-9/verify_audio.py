"""核對聲音接頭的形狀、監督位置、既有實驗紀錄與官方契約。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.modalities import _audio_records, _freeze
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, expand_modalities, tone


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def example(frequency, answer, expected_shape):
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = MultiModalLM(TinyLM(ModelConfig(width=8)))
    waveform = tone(frequency)
    encoded = model.audio(waveform.unsqueeze(0))
    projected = model.audio_projector(encoded)[0]
    prefix = [tok.bos_id, tok.user_id, tok.audio_id, tok.eos_id, tok.assistant_id]
    tail = tok.encode(answer) + [tok.eos_id]
    ids = torch.tensor(prefix + tail)
    labels = torch.tensor([-100] * len(prefix) + tail)
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    output = model(ids, labels, waveform=waveform)
    x, y = expand_modalities(
        ids, labels, model.language.embedding, {tok.audio_id: projected}, {tok.image_id, tok.audio_id}
    )
    assert tuple(output["logits"].shape) == expected_shape, "分數形狀不符"
    assert torch.equal(y, output["labels"]), "標籤錯位不符"
    assert torch.equal(x[0, 2:13], projected), "聲音佔位位置未換成聲音特徵"
    assert y[y != -100].tolist() == tail, "有效目標不符"
    assert torch.equal(model.language(embeddings=x)["logits"], output["logits"]), "文字核心輸入不符"
    loss = masked_loss(output["logits"], output["labels"])
    independent_loss = torch.nn.functional.cross_entropy(output["logits"][y != -100], y[y != -100])
    torch.testing.assert_close(loss, independent_loss, rtol=1e-6, atol=1e-6, msg="忽略位置或分母不符")
    loss.backward()
    projector_gradient = model.audio_projector.weight.grad.norm().item()
    encoder_gradient = model.audio.projection.weight.grad.norm().item()
    marker_gradient = model.language.embedding.weight.grad[tok.audio_id].norm().item()
    unchanged = all(torch.equal(parameters[name], p.detach()) for name, p in model.named_parameters())
    assert projector_gradient > 0 and encoder_gradient > 0, "聲音通路沒有梯度"
    assert marker_gradient == 0, "聲音佔位 token 的 embedding 被使用"
    assert unchanged, "只做 backward 不應更新參數"
    errors = {}
    for name, kwargs in [("未提供媒體", {}), ("只有圖片", {"image": torch.zeros(3, 16, 16)})]:
        try:
            model(ids, labels, **kwargs)
        except ValueError as error:
            errors[name] = str(error)
        else:
            raise AssertionError("缺少聲音時沒有報錯")
    return {
        "frequency": frequency,
        "answer": answer,
        "waveform_shape": list(waveform.shape),
        "batched_waveform_shape": list(waveform.unsqueeze(0).shape),
        "encoded_shape": list(encoded.shape),
        "projected_shape": list(projected.shape),
        "prefix_ids": prefix,
        "answer_ids": tail,
        "audio_id": tok.audio_id,
        "vocab_size": tok.vocab_size,
        "original_length": len(ids),
        "expanded_length": len(ids) - 1 + len(projected),
        "logits_shape": list(output["logits"].shape),
        "shifted_labels": y.tolist(),
        "effective_target_count": int((y != -100).sum()),
        "loss": loss.item(),
        "independent_loss": independent_loss.item(),
        "loss_absolute_difference": abs(loss.item() - independent_loss.item()),
        "projector_gradient_norm": projector_gradient,
        "encoder_gradient_norm": encoder_gradient,
        "marker_embedding_gradient_norm": marker_gradient,
        "parameters_unchanged": unchanged,
        "missing_waveform_errors": errors,
    }


def experiment_audit():
    paths = {name: f"docs/course-experiments/results/{name}.json" for name in ("audio", "encoders", "sft")}
    reports = {name: json.loads((ROOT / path).read_bytes()) for name, path in paths.items()}
    report = reports["audio"]
    results = report["results"]
    splits = {key: value["records"] for key, value in results["data"]["splits"].items()}
    assert splits == _audio_records(), "實驗資料與生成器不符"
    for split, rows in splits.items():
        computed = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert computed == results["data"]["splits"][split]["sha256"], "資料切分雜湊不符"
    training_frequencies = {row["frequency"] for row in splits["train"]}
    encoder_training = reports["encoders"]["results"]["audio"]["data"]["splits"]["train"]["records"]
    assert not training_frequencies & {row["frequency"] for row in splits["test"]}, "頻率洩漏到問答訓練集"
    assert not {row["frequency"] for row in encoder_training} & {
        row["frequency"] for row in splits["test"]
    }, "頻率洩漏到聲音分類訓練集"
    joined = []
    tok = ByteTokenizer()
    for row, sample in zip(splits["test"], results["test"]["samples"], strict=True):
        assert row["family"] == sample["family"] and row["answer"] == sample["target"], "題目對應不符"
        assert row["answer"] == ("high" if row["frequency"] > 300 else "low"), "分類邊界不符"
        generated = sample["generated_ids"]
        raw = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
        exact = raw == tok.encode(row["answer"])
        assert exact == sample["exact_match"], "原始 token 計分與紀錄不符"
        assert tok.decode(raw) == sample["generated"], "生成 token 與文字不符"
        joined.append({**row, "generated": sample["generated"], "exact_match": exact})
    errors = [row for row in joined if not row["exact_match"]]
    correct = sum(row["exact_match"] for row in joined)
    assert correct == results["test"]["correct"] == 11 and len(joined) == 14, "測試分數不符"
    assert [(r["frequency"], r["amplitude"], r["seconds"]) for r in errors] == [
        (290, 0.25, 0.1), (300, 0.25, 0.1), (300, 0.5, 0.12)
    ], "錯誤分布不符"
    training = results["training"]
    assert training["steps"] == len(training["history"]) == 300, "訓練步數不符"
    assert [row["step"] for row in training["history"]] == list(range(1, 301)), "訓練步數不連續"
    assert sum(row["effective_targets"] for row in training["history"]) == training["effective_targets"], "有效目標分母不符"
    relevant = [
        "tiny_perceptron/multimodal.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py",
        "tiny_perceptron/training.py", "scripts/course_experiments/modalities.py",
        "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    ]
    matches = {path: digest(path) == report["code_sha256"][path] for path in relevant}
    assert all(matches.values()), "實驗來源版本不符"
    trainable = MultiModalLM(TinyLM(ModelConfig(width=64, layers=2)))
    _freeze(trainable, "all", "audio")
    trainable_groups = {
        name: all(p.requires_grad for p in getattr(trainable, name).parameters())
        for name in ("audio", "audio_projector", "language")
    }
    assert all(trainable_groups.values()), "共同訓練範圍不符"
    return {
        "report_hashes": {name: {"path": path, "sha256": digest(path)} for name, path in paths.items()},
        "reported_environment": {key: report[key] for key in ("python_version", "torch_version", "device", "gpu", "seed", "revision")},
        "steps": training["steps"],
        "effective_targets": training["effective_targets"],
        "test_frequencies": sorted({row["frequency"] for row in joined}),
        "train_frequencies": sorted(training_frequencies),
        "encoder_train_frequencies": sorted({row["frequency"] for row in encoder_training}),
        "correct": correct,
        "count": len(joined),
        "joined_test_samples": joined,
        "errors": errors,
        "source_hash_matches": matches,
        "trainable_groups": trainable_groups,
        "encoder_training_steps": reports["encoders"]["results"]["audio"]["training"]["steps"],
        "sft_training_steps": reports["sft"]["results"]["training"]["steps"],
        "scope": "重算既有逐題紀錄與核對來源契約；未重跑 CUDA 訓練或載入私有權重。",
    }


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def originals():
    specs = [
        ("linear", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", "PyTorch 2.14", "Linear / Shape", ["where all but the last dimension are the same shape as the input"]),
        ("backward", "https://docs.pytorch.org/docs/2.14/generated/torch.autograd.backward.html", "PyTorch 2.14", "torch.autograd.backward", ["The graph is differentiated using the chain rule.", "This function accumulates gradients in the leaves"]),
        ("optimizer", "https://docs.pytorch.org/docs/2.14/optim.html", "PyTorch 2.14", "Taking an optimization step", ["All optimizers implement a step() method, that updates the parameters."]),
        ("teacher_forcing", "https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html", "PyTorch Tutorials 2.14.0+cu130", "Training the Model / Training", ["Teacher forcing: Feed the target as the next input", "Without teacher forcing: use its own predictions as the next input"]),
    ]
    collected = []
    for identifier, url, version, locator, excerpts in specs:
        command = ["curl", "-fsSL", "--max-time", "30", url]
        fetched = subprocess.run(command, capture_output=True, check=True)
        parser = VisibleText()
        parser.feed(fetched.stdout.decode())
        text = re.sub(r"\s+", " ", " ".join(parser.parts))
        for excerpt in excerpts:
            assert excerpt in text, "官方原文摘錄不存在"
        collected.append({
            "id": identifier, "url": url, "version": version, "locator": locator,
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "command_argv": command,
            "original_response_sha256": hashlib.sha256(fetched.stdout).hexdigest(),
            "excerpts": excerpts,
            "response_bytes": len(fetched.stdout),
            "inspection": "以 curl 讀取官方原文，定位契約並僅保存支持主張的小段摘錄。",
        })
    return collected


torch.set_num_threads(1)
torch.set_default_device("cpu")
report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-9/verify_audio.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "examples": [example(440, "high", (1, 19, 264)), example(200, "low", (1, 18, 264))],
    "experiment_audit": experiment_audit(),
    "original_sources": originals(),
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
