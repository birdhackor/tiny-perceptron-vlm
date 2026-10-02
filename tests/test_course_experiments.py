"""Regression checks for complete evaluations, leakage and generated modal markers."""

import pytest
import torch

from scripts.course_experiments.common import evaluate_lm, split_records, text_examples
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, generate_modal


def test_family_variants_never_cross_splits():
    records = [{"family": str(i), "text": f"{i}:{style}"} for i in range(12) for style in range(3)]
    parts = split_records(records + records[:3])
    groups = [{r["family"] for r in rows} for rows in parts.values()]
    assert sum(map(len, parts.values())) == len(records)
    assert all(not (left & right) for i, left in enumerate(groups) for right in groups[i + 1 :])


def test_chunking_scores_every_document_target_once():
    tok = ByteTokenizer()
    text = "中文 boundary 😀 with multiple chunks"
    examples = text_examples([{"text": text}], max_length=7)
    labels = torch.cat([y for _, y in examples]).tolist()
    assert labels == tok.encode(text) + [tok.eos_id]
    assert all(len(x) <= 7 for x, _ in examples)


def test_full_heldout_evaluation_does_not_stop_at_twenty():
    model = TinyLM(ModelConfig(width=8, max_length=64))
    rows = [
        {"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "yes"}]} for i in range(23)
    ]
    report = evaluate_lm(model, rows, mode="sft", tokens=1)
    assert report["records"] == len(report["samples"]) == 23
    assert report["effective_tokens"] == 23 * 4  # y, e, s and EOS


def test_text_generation_prompt_keeps_unicode_characters_complete():
    model = TinyLM(ModelConfig(width=8, max_length=64))
    text = "留別王侍御維\n孟浩然與未見字🦊"
    sample = evaluate_lm(model, [{"text": text}], tokens=1)["samples"][0]
    assert "\ufffd" not in sample["prompt"]
    assert text.startswith(sample["prompt"])
    assert len(sample["prompt"].encode("utf-8")) <= 24


@pytest.mark.parametrize("marker", [5, 6])
def test_generated_modal_marker_is_returned_as_failure_without_reexpansion(marker, monkeypatch):
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=64)))
    model.train()
    prefix = torch.tensor([1, 3, 78, 2, 4])
    calls = []

    def pretend_model(ids, **kwargs):
        calls.append(ids.clone())
        if marker in ids.tolist():
            raise ValueError("A generated marker must not be reinterpreted as an input placeholder")
        logits = torch.zeros(1, len(ids) - 1, 264)
        logits[..., marker] = 10
        return {"logits": logits}

    monkeypatch.setattr(model, "forward", pretend_model)
    generated = generate_modal(model, prefix, max_new_tokens=4)
    assert generated.tolist() == prefix.tolist() + [marker]
    assert len(calls) == 1
    assert model.training
