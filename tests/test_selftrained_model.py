"""Small CPU checks of genuine architecture invariants, not learned capability claims."""

import copy

import pytest
import torch
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE
from tiny_perceptron.selftrained.model import (
    AudioEncoder,
    LimitedAssistant,
    PaddingSafeMoE,
    SelftrainedConfig,
    VisionEncoder,
    decode_ocr,
)
from tiny_perceptron.selftrained.tokenizer import OCR_CHARACTERS, CharacterTokenizer, generation_report


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(5)
    torch.set_num_threads(1)


def small_model(**kwargs):
    options = dict(vocab_size=32, width=32, layers=2, heads=4, kv_heads=2, ffn_hidden=64, max_length=128)
    options.update(kwargs)
    return LimitedAssistant(SelftrainedConfig(**options))


def image_payload():
    return {
        "kind": "image",
        "values": torch.rand(2, 1, 32, 32),
        "coordinates": torch.tensor([[0.25, 0.5], [0.75, 0.5]]),
    }


def test_tokenizer_train_alphabet_roles_and_roundtrip(tmp_path):
    tokenizer = CharacterTokenizer.build(["答案。<assistant>", "開口"])
    assert tokenizer.image_id == 7 and tokenizer.ocr_id == 8 and tokenizer.audio_id == 9
    assert tokenizer.assistant_id not in tokenizer.encode("<assistant>")
    assert tokenizer.decode(tokenizer.encode("答案。")) == "答案。"
    assert tokenizer.encode("陌") == [tokenizer.unk_id]
    assert tokenizer.decode([tokenizer.unk_id]) == "�"
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = CharacterTokenizer.load(path)
    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.sha256 == tokenizer.sha256
    assert not generation_report(tokenizer, [tokenizer.tool_id, tokenizer.eos_id])["valid_answer_tokens"]
    with pytest.raises(ValueError, match="fixed role IDs"):
        CharacterTokenizer.from_dict({**tokenizer.to_dict(), "specials": []})


def test_router_ignores_padding_in_values_load_auxiliary_and_gradient():
    router = PaddingSafeMoE(8, experts=3, top_k=1, hidden=16)
    active = torch.randn(1, 4, 8)
    expected, expected_aux, expected_route = router(active)
    padded = torch.cat((torch.randn(1, 2, 8) * 100, active), 1).requires_grad_()
    valid = torch.tensor([[False, False, True, True, True, True]])
    actual, actual_aux, actual_route = router(padded, valid)
    assert torch.equal(actual[:, :2], torch.zeros_like(actual[:, :2]))
    assert torch.allclose(actual[:, 2:], expected)
    assert torch.allclose(actual_aux, expected_aux)
    assert torch.equal(actual_route["counts"], expected_route["counts"])
    assert actual_route["counts"].sum() == 4
    assert (actual_route["chosen"][:, :2] == -1).all()
    actual.square().sum().backward()
    assert router.router.weight.grad.abs().sum() > 0
    assert padded.grad[:, :2].abs().sum() == 0
    empty, empty_aux, empty_route = router(padded.detach(), torch.zeros_like(valid))
    assert torch.isfinite(empty).all() and empty_aux == 0
    assert empty_route["counts"].sum() == 0 and empty_route["valid_tokens"] == 0


@pytest.mark.parametrize("architecture", ["moe", "dense"])
def test_lm_causality_left_padding_and_all_padding_are_safe(architecture):
    model = small_model(architecture=architecture)
    ids = torch.tensor([[1, 3, 11, 4, 12]])
    changed = ids.clone()
    changed[0, -1] = 13
    baseline = model(ids)
    assert torch.allclose(baseline["logits"][:, :-1], model(changed)["logits"][:, :-1], atol=2e-6)
    padded = torch.cat((torch.zeros(1, 3, dtype=torch.long), ids), 1)
    valid = padded != 0
    result = model(padded, attention_mask=valid)
    assert torch.allclose(baseline["logits"], result["logits"][:, 3:], atol=3e-6)
    assert torch.allclose(baseline["aux_loss"], result["aux_loss"], atol=1e-6)
    empty = model(torch.zeros_like(ids), attention_mask=torch.zeros_like(ids, dtype=torch.bool))
    assert torch.isfinite(empty["logits"]).all()
    assert empty["logits"].abs().sum() == 0


def test_sdpa_and_manual_masked_logits_and_gradients_match():
    manual = small_model(backend="manual")
    sdpa = copy.deepcopy(manual)
    for block in sdpa.lm.blocks:
        block.attention.backend = "sdpa"
    ids = torch.tensor([[0, 1, 3, 11, 4], [1, 3, 12, 4, 13]])
    a = manual(ids, attention_mask=ids != 0)
    b = sdpa(ids, attention_mask=ids != 0)
    assert torch.allclose(a["logits"], b["logits"], atol=3e-6)
    (a["logits"].square().mean() + a["aux_loss"]).backward()
    (b["logits"].square().mean() + b["aux_loss"]).backward()
    for p, q in zip(manual.lm.parameters(), sdpa.lm.parameters(), strict=True):
        if p.grad is not None or q.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=2e-5, rtol=3e-4)


@pytest.mark.parametrize("architecture", ["moe", "dense"])
def test_multimodal_prefix_cache_logits_and_generated_continuations_match(architecture):
    model = small_model(architecture=architecture)
    payloads = [[image_payload()]]
    prefix = torch.tensor([[1, 3, 7, 7, 4]])
    full_ids = torch.cat((prefix, torch.tensor([[11, 12]])), 1)
    expected = model(full_ids, modalities=payloads)["logits"]
    first = model(prefix, modalities=payloads)
    pieces, cache = [first["logits"]], first["cache"]
    for index in (5, 6):
        result = model(full_ids[:, index : index + 1], cache=cache)
        pieces.append(result["logits"])
        cache = result["cache"]
    assert torch.allclose(torch.cat(pieces, 1), expected, atol=4e-6)
    assert cache[0][0].shape == (1, 2, 7, 8)
    assert torch.equal(
        model.generate(prefix, modalities=payloads, max_new_tokens=5, use_cache=True),
        model.generate(prefix, modalities=payloads, max_new_tokens=5, use_cache=False),
    )
    assert model.training
    with pytest.raises(ValueError, match="unpadded"):
        model(full_ids[:, :1], attention_mask=torch.ones(1, 1, dtype=torch.bool), cache=cache)


def test_answer_loss_reaches_every_random_perception_branch_and_router():
    model = small_model()
    payloads = [
        [image_payload()],
        [{"kind": "ocr", "values": torch.rand(1, 32, 128)}],
        [{"kind": "audio", "values": torch.randn(73, 40)}],
    ]
    rows, label_rows = [], []
    for kind, count in ((7, 2), (8, 32), (9, 16)):
        ids = [1, 3] + [kind] * count + [4, 11]
        labels = [IGNORE] * (len(ids) - 2) + [11, 2]
        rows.append(ids)
        label_rows.append(labels)
    length = max(map(len, rows))
    ids = torch.tensor([row + [0] * (length - len(row)) for row in rows])
    labels = torch.tensor([row + [IGNORE] * (length - len(row)) for row in label_rows])
    result = model(ids, attention_mask=ids != 0, labels=labels, modalities=payloads)
    assert [entry["logits"].shape for entry in result["perception"]] == [(2, 3), (32, 13), (3,)]
    assert [entry["batch_index"] for entry in result["perception"]] == [0, 1, 2]
    # Use only answer loss: auxiliary labels cannot explain this gradient path.
    result["language_loss"].backward()
    for encoder in (model.vision_encoder, model.ocr_encoder, model.audio_encoder):
        first = next(encoder.parameters())
        assert first.grad is not None and first.grad.abs().sum() > 0
        assert encoder.head.weight.grad is not None and encoder.head.weight.grad.abs().sum() > 0
        assert encoder.projection.weight.grad is not None and encoder.projection.weight.grad.abs().sum() > 0
    assert model.lm.embedding.weight.grad.abs().sum() > 0
    assert any(block.ffn.router.weight.grad.abs().sum() > 0 for block in model.lm.blocks)
    assert all(route["counts"].sum() == (ids != 0).sum() for route in result["routing"])


def test_vision_preserves_slot_order_and_public_two_dimensional_geometry():
    encoder = VisionEncoder(32)
    images = torch.rand(2, 1, 32, 32)
    horizontal = torch.tensor([[0.25, 0.5], [0.75, 0.5]])
    vertical = torch.tensor([[0.5, 0.25], [0.5, 0.75]])
    tokens, logits = encoder(images, horizontal)
    switched, switched_logits = encoder(images.flip(0), horizontal)
    vertical_tokens, vertical_logits = encoder(images, vertical)
    assert torch.allclose(switched_logits, logits.flip(0))
    assert torch.equal(vertical_logits, logits)
    assert not torch.allclose(tokens, switched)
    assert not torch.allclose(tokens, vertical_tokens)


def test_audio_retains_temporal_order_and_ignores_only_declared_padding():
    encoder = AudioEncoder(32)
    frames = torch.randn(69, 40)
    tokens, logits = encoder(frames)
    reversed_tokens, _ = encoder(frames.flip(0))
    assert tokens.shape == (16, 32)
    assert not torch.allclose(tokens, reversed_tokens)
    padded = torch.cat((frames, torch.randn(25, 40) * 100), 0)
    valid = torch.arange(94) < 69
    padded_tokens, padded_logits = encoder(padded, valid)
    assert torch.equal(tokens, padded_tokens) and torch.equal(logits, padded_logits)
    end_changed = frames.clone()
    end_changed[-8:] += 5
    assert not torch.allclose(tokens, encoder(end_changed)[0])
    broken = valid.clone()
    broken[3] = False
    with pytest.raises(ValueError, match="contiguous"):
        encoder(padded, broken)


def test_ocr_ctc_trains_composed_glyphs_and_collapses_repeated_glyphs():
    model = small_model()
    _, logits = model.ocr_encoder(torch.rand(1, 32, 128))
    target = torch.tensor([1, 1, 12, 4])
    loss = F.ctc_loss(logits.log_softmax(-1)[:, None], target, torch.tensor([32]), torch.tensor([4]), blank=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.ocr_encoder.head.weight.grad.abs().sum() > 0
    assert model.ocr_encoder.cnn[0].weight.grad.abs().sum() > 0
    predictions = torch.full((8, 13), -20.0)
    predictions[torch.arange(8), torch.tensor([1, 1, 0, 1, 12, 12, 0, 4])] = 20
    assert decode_ocr(predictions) == OCR_CHARACTERS[0] * 2 + OCR_CHARACTERS[11] + OCR_CHARACTERS[3]


def test_inputs_reject_missing_noncontiguous_and_answer_bearing_payloads():
    model = small_model()
    payload = image_payload()
    with pytest.raises(ValueError, match="exactly match"):
        model(torch.tensor([[1, 3, 7, 7, 4]]))
    with pytest.raises(ValueError, match="contiguous"):
        model(torch.tensor([[1, 3, 7, 11, 7, 4]]), modalities=[[payload]])
    with pytest.raises(ValueError, match="public geometry only"):
        model(torch.tensor([[1, 3, 7, 7, 4]]), modalities=[[{**payload, "transcript": "gold"}]])
    with pytest.raises(ValueError, match="boolean"):
        model(torch.tensor([[1, 3, 11, 4]]), attention_mask=torch.ones(1, 4))
    ids = torch.tensor([[1, 3, 11, 4, 0]])
    with pytest.raises(ValueError, match="Padding cannot"):
        model(ids, attention_mask=ids != 0, labels=torch.tensor([[IGNORE, IGNORE, IGNORE, 12, 2]]))


def test_multiple_modalities_in_one_history_keep_distinct_prediction_indices():
    model = small_model()
    payloads = [image_payload(), {"kind": "audio", "values": torch.randn(29, 40)}, image_payload()]
    ids = torch.tensor([[1, 3, 7, 7, 4, 11, 2, 3] + [9] * 16 + [4, 12, 2, 3, 7, 7, 4]])
    result = model(ids, modalities=[payloads])
    assert [(p["modality_index"], p["kind"]) for p in result["perception"]] == [
        (0, "image"),
        (1, "audio"),
        (2, "image"),
    ]


def test_parameter_accounting_includes_all_encoders_and_states_comparison():
    model, dense = small_model(), small_model(architecture="dense")
    report, dense_report = model.description(), dense.description()
    assert report["parameters"] == report["language_parameters"] + sum(report["perception_parameters"].values())
    assert report["parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert (
        report["active_language_parameters_per_token"] - report["router_parameters"]
        == dense_report["language_parameters"]
    )
    assert report["active_language_parameters_per_token"] < report["language_parameters"]
    assert report["perception_parameters"] == dense_report["perception_parameters"]
    assert "not total parameters" in report["dense_comparison"]
