"""學生評估不可把非法控制字或未結束答案偽裝成答對。"""

import hashlib
import json
import math
import sys

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scripts import evaluate
from tiny_perceptron.data import SPECIALS, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import save_checkpoint


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def emit(monkeypatch, ids):
    def generate(model, prefix, max_new_tokens, **kwargs):
        assert kwargs["eos_id"] == 2
        return torch.cat((prefix, torch.tensor([ids], device=prefix.device)), dim=1)

    monkeypatch.setattr(evaluate, "generate", generate)


def force_token(model, token):
    with torch.no_grad():
        model.final_norm.weight.zero_()
        model.final_norm.bias.zero_()
        model.final_norm.bias[0] = 1
        model.output.weight.zero_()
        model.output.weight[token, 0] = 1


def run_cli(monkeypatch, capsys, checkpoint, data, output, *options):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            str(checkpoint),
            "--data",
            str(data),
            "--output",
            str(output),
            "--mode",
            "sft",
            "--device",
            "cpu",
            *map(str, options),
        ],
    )
    evaluate.main()
    summary = json.loads(capsys.readouterr().out)
    full = json.loads(output.read_text())
    assert "samples" not in summary and len(full["samples"]) == full["generation_evaluated_records"]
    return full


def test_illegal_control_is_visible_and_cannot_be_counted_as_exact(monkeypatch):
    tok = ByteTokenizer()
    ids = [tok.user_id, *tok.encode("A"), tok.eos_id]
    emit(monkeypatch, ids)
    model = TinyLM(ModelConfig(width=8, max_length=64))
    report = evaluate.evaluate(model, [{"question": "q", "answer": "A"}], "sft")
    sample = report["samples"][0]
    assert sample["generated"] == "<user>A" and sample["generated_ids"] == ids
    assert sample["invalid_special_tokens"] == [{"position": 0, "id": 3, "token": "<user>"}]
    assert sample["exact_match"] is False and sample["completed_exact_match"] is False
    assert report["exact_match"] == report["completed_exact_match"] == 0.0
    assert report["eos_rate"] == 1.0 and model.training


@pytest.mark.parametrize("text", [" A", "A ", "\nA\n"])
def test_exact_does_not_strip_whitespace(monkeypatch, text):
    emit(monkeypatch, [*ByteTokenizer().encode(text), 2])
    report = evaluate.evaluate(TinyLM(ModelConfig(width=8)), [{"question": "q", "answer": "A"}], "sft")
    assert report["samples"][0]["generated"] == text
    assert report["exact_match"] == report["completed_exact_match"] == 0.0


def test_real_generation_token_cap_is_content_exact_without_completed_exact():
    model = TinyLM(ModelConfig(width=8, max_length=64))
    force_token(model, ByteTokenizer().encode("A")[0])
    report = evaluate.evaluate(model, [{"question": "q", "answer": "A"}], "sft", max_new_tokens=1)
    sample = report["samples"][0]
    assert sample["generated_ids"] == ByteTokenizer().encode("A")
    assert report["exact_match"] == 1.0 and report["completed_exact_match"] == 0.0 and report["eos_rate"] == 0.0
    assert sample["generation_status"] == "token_or_context_limit" and model.training


def test_completed_answer_needs_raw_exact_and_normal_eos(monkeypatch):
    emit(monkeypatch, [*ByteTokenizer().encode("A"), 2])
    model = TinyLM(ModelConfig(width=8))
    model.eval()
    report = evaluate.evaluate(model, [{"question": "q", "answer": "A"}], "sft")
    assert report["exact_match"] == report["completed_exact_match"] == report["eos_rate"] == 1.0
    assert not model.training


def test_text_windows_score_each_raw_byte_and_one_eos_never_bos(monkeypatch):
    recorded = []
    original = evaluate.loss_sum

    def inspect(logits, labels):
        recorded.extend(labels.flatten().tolist())
        return original(logits, labels)

    monkeypatch.setattr(evaluate, "loss_sum", inspect)
    text = "你好A"
    model = TinyLM(ModelConfig(width=8, max_length=4))
    report = evaluate.evaluate(model, [{"text": text}])
    assert recorded == ByteTokenizer().encode(text) + [2] and 1 not in recorded
    assert report["effective_tokens"] == len(text.encode("utf-8")) + 1
    assert report["bpb_including_eos_boundary_targets"] == pytest.approx(
        report["mean_token_nll"] * report["effective_tokens"] / (len(text.encode("utf-8")) * math.log(2))
    )
    assert "bpb_including_bos_eos_boundary_targets" not in report
    assert report["loss_evaluated_records"] == 1 and report["generation_evaluated_records"] == 0
    assert report["skipped"] == [
        {"row": 0, "phase": "generation", "reason": "prompt_context_too_long", "source": {"text": text}, "prompt": text}
    ]
    assert report["eos_rate"] is None and report["metric_denominators"]["eos_rate"] == 0 and model.training


def test_denominators_and_skip_reasons_are_explicit(monkeypatch):
    emit(monkeypatch, [*ByteTokenizer().encode("A"), 2])
    records = [
        {"question": "q", "answer": "A"},
        {"question": "q" * 100, "answer": "A"},
        {"question": "q", "answer": "B"},
    ]
    report = evaluate.evaluate(TinyLM(ModelConfig(width=8, max_length=32)), records, "sft")
    assert report["records_read"] == report["records_selected"] == 3
    assert report["loss_evaluated_records"] == report["generation_evaluated_records"] == 2
    assert report["loss_skipped_records"] == report["generation_skipped_records"] == 1
    assert report["metric_denominators"]["exact_match"] == report["metric_denominators"]["eos_rate"] == 2
    assert report["exact_match"] == report["completed_exact_match"] == 0.5
    assert report["skipped"][0]["source"] == records[1]


def test_no_valid_targets_is_an_error_and_restores_training_mode():
    model = TinyLM(ModelConfig(width=8, max_length=8))
    with pytest.raises(ValueError, match="沒有有效"):
        evaluate.evaluate(model, [{"question": "q" * 40, "answer": "A"}], "sft")
    assert model.training
    model.eval()
    with pytest.raises(ValueError, match="沒有有效"):
        evaluate.evaluate(model, [])
    assert not model.training


def test_cli_default_limit_and_all_preserve_read_and_selected_denominators(tmp_path, monkeypatch, capsys):
    model = TinyLM(ModelConfig(width=8, max_length=32))
    force_token(model, 2)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model)
    data = tmp_path / "validation.jsonl"
    data.write_text("\n".join(json.dumps({"question": "q", "answer": "A"}) for _ in range(25)) + "\n")
    short = run_cli(monkeypatch, capsys, checkpoint, data, tmp_path / "short.json", "--tokens", 1)
    full = run_cli(monkeypatch, capsys, checkpoint, data, tmp_path / "all.json", "--limit", "all", "--tokens", 1)
    assert short["records_read"] == full["records_read"] == 25
    assert short["records_selected"] == short["generation_evaluated_records"] == 20 and short["unselected_records"] == 5
    assert full["records_selected"] == full["generation_evaluated_records"] == 25 and full["unselected_records"] == 0
    assert full["limit"] == "all" and len(full["samples"]) == 25


def bpe_file(path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["a q <user> 中文🙂"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=list(SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tok.save(str(path))
    return tok


def test_cli_bpe_requires_matching_config_and_bound_sha(tmp_path, monkeypatch, capsys):
    tokenizer_file = tmp_path / "tokenizer.json"
    tok = bpe_file(tokenizer_file)
    digest = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    model = TinyLM(ModelConfig(width=8, vocab_size=tok.get_vocab_size(), max_length=64))
    force_token(model, 2)
    checkpoint = tmp_path / "bpe.pt"
    save_checkpoint(checkpoint, model, metadata={"tokenizer": "bpe", "tokenizer_sha256": digest})
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({"question": "literal <user> 🙂", "answer": "a"}) + "\n")
    with pytest.raises(ValueError, match="--tokenizer"):
        run_cli(monkeypatch, capsys, checkpoint, data, tmp_path / "missing.json")
    report = run_cli(
        monkeypatch, capsys, checkpoint, data, tmp_path / "bpe.json", "--tokenizer", tokenizer_file, "--tokens", 1
    )
    assert report["tokenizer"]["sha256"] == digest and report["tokenizer"]["vocab_size"] == tok.get_vocab_size()
    assert report["effective_tokens"] == len(tok.encode("a").ids) + 1
    assert report["samples"][0]["generated_ids"] == [2]
    changed = tmp_path / "changed.json"
    changed.write_text(tokenizer_file.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        run_cli(monkeypatch, capsys, checkpoint, data, tmp_path / "bad-sha.json", "--tokenizer", changed)
    byte_model = TinyLM(ModelConfig(width=8))
    save_checkpoint(tmp_path / "byte.pt", byte_model)
    with pytest.raises(ValueError, match="詞表大小"):
        run_cli(
            monkeypatch, capsys, tmp_path / "byte.pt", data, tmp_path / "bad-config.json", "--tokenizer", tokenizer_file
        )


def test_cli_byte_json_binding_and_native_text_only(tmp_path, monkeypatch, capsys):
    tokenizer_file = tmp_path / "byte.json"
    tokenizer_file.write_text(json.dumps(ByteTokenizer().state()))
    digest = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    model = TinyLM(ModelConfig(width=8))
    force_token(model, 2)
    checkpoint = tmp_path / "byte.pt"
    save_checkpoint(checkpoint, model, metadata={"tokenizer_sha256": digest})
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({"question": "q", "answer": "A"}) + "\n")
    report = run_cli(
        monkeypatch, capsys, checkpoint, data, tmp_path / "good.json", "--tokenizer", tokenizer_file, "--tokens", 1
    )
    assert report["tokenizer"]["sha256"] == digest
    with pytest.raises(ValueError, match="SHA-256"):
        run_cli(monkeypatch, capsys, checkpoint, data, tmp_path / "missing.json")
    save_checkpoint(tmp_path / "modal.pt", MultiModalLM(model))
    with pytest.raises(ValueError, match="TinyLM 文字"):
        run_cli(monkeypatch, capsys, tmp_path / "modal.pt", data, tmp_path / "modal.json")
