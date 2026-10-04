"""Integrity checks for the capstone, not tests that promise toy-model accuracy."""

import hashlib
import json

import pytest
import torch

from scripts.course_experiments.capstone import train_stage
from tiny_perceptron.capstone import (
    TOK,
    CapstoneModel,
    build_dataset,
    calculator_runtime,
    encode_record,
    export_inference,
    frozen_reference,
    generate_trace,
    generate_traces,
    load_capstone,
    modality_tensors,
    parse_action,
    preference_loss,
    preference_pairs,
    prepare_batch,
    prompt_ids,
    run_assistant,
    save_capstone,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import ModelConfig, masked_loss


def small_model():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    return CapstoneModel(
        ModelConfig(
            width=16, layers=1, heads=2, kv_heads=1, experts=4, norm="rms", rotary=True, backend="sdpa", max_length=192
        )
    )


def actual_input_sha(row):
    digest = hashlib.sha256(json.dumps(prompt_ids(row)).encode())
    for feature in modality_tensors(row):
        if feature is not None:
            digest.update(feature.numpy().tobytes())
    return digest.hexdigest()


def test_family_split_and_actual_prompt_modal_content_do_not_leak():
    splits, manifest = build_dataset()
    again, another = build_dataset()
    assert splits == again and manifest == another
    families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
    inputs = {name: {actual_input_sha(row) for row in rows} for name, rows in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not families[left] & families[right]
        assert not inputs[left] & inputs[right]
    assert all(any(row["task"] == "joint" for row in rows) for rows in splits.values())


def test_sft_labels_ignore_entire_prompt_but_include_every_answer_byte_and_eos():
    splits, _ = build_dataset()
    row = splits["train"][0]
    _, labels = encode_record(row)
    expected = TOK.encode(row["answer"]) + [TOK.eos_id]
    assert labels[labels != IGNORE].tolist() == expected
    assert labels[: len(prompt_ids(row)) - 1].eq(IGNORE).all()
    _, pretrain = encode_record(row, pretrain=True)
    assert (pretrain != IGNORE).all()


def test_real_modalities_affect_shared_core_and_both_projectors_receive_gradient():
    model = small_model()
    rows, _ = build_dataset()
    row = next(row for row in rows["train"] if row["task"] == "joint")
    batch, labels = prepare_batch([row])
    original = model(**batch)["logits"]
    changed = dict(batch, images=batch["images"].roll(1, 1), audio_features=batch["audio_features"].flip(-1))
    alternate = model(**changed)["logits"]
    assert not torch.allclose(original, alternate)
    masked_loss(original, labels).backward()
    for projector in (model.image_projector, model.audio_projector):
        assert all(parameter.grad is not None for parameter in projector.parameters())
        assert sum(float(parameter.grad.abs().sum()) for parameter in projector.parameters()) > 0
    with pytest.raises(ValueError, match="modality"):
        model(batch["ids"])


def test_pad_tokens_do_not_change_router_auxiliary_or_real_logits():
    model = small_model()
    ids = torch.tensor([[1, 3, 70, 71, 4]])
    original = model(ids)
    padded = torch.cat((ids, torch.zeros(1, 7, dtype=torch.long)), -1)
    valid = torch.arange(12).unsqueeze(0) < 5
    alternate = model(padded, valid=valid)
    torch.testing.assert_close(original["logits"], alternate["logits"][:, :5], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(original["auxiliary"], alternate["auxiliary"], atol=1e-7, rtol=1e-6)


def test_modal_prefill_cache_matches_full_recompute_and_encodes_only_once():
    model = small_model()
    splits, _ = build_dataset()
    row = next(row for row in splits["test"] if row["task"] == "joint")
    ids = torch.tensor([prompt_ids(row)])
    image, audio = modality_tensors(row)
    kwargs = {"images": image.unsqueeze(0), "audio_features": audio.unsqueeze(0)}
    calls = []
    handle = model.image_projector.register_forward_hook(lambda *_: calls.append(1))
    pref = model(ids, **kwargs)
    token = torch.tensor([[75]])
    decoded = model(token, cache=pref["cache"])
    assert len(calls) == 1
    full = model(torch.cat((ids, token), -1), **kwargs)
    handle.remove()
    torch.testing.assert_close(decoded["logits"][:, -1], full["logits"][:, -1], atol=2e-6, rtol=2e-5)
    cached = generate_trace(model, row, max_new_tokens=3, use_cache=True)
    recomputed = generate_trace(model, row, max_new_tokens=3, use_cache=False)
    assert cached["generated_ids"] == recomputed["generated_ids"]
    assert cached["eos"] == recomputed["eos"]
    assert generate_traces(model, [row, row], max_new_tokens=3)[0]["generated_ids"] == cached["generated_ids"]


@pytest.mark.parametrize(
    "raw", ["TOOL:calculator:1+2;rm", "TOOL:calculator:-1+2", "TOOL:calculator:1", "DIRECT:", "hello"]
)
def test_action_parser_rejects_malformed_generated_requests(raw):
    assert parse_action({"raw": raw, "eos": True})["status"] == "invalid"


def test_runtime_checks_availability_allowlist_types_bounds_and_eos():
    action = parse_action({"raw": "TOOL:calculator:1+2", "eos": True})
    assert calculator_runtime(action)["result"] == "3"
    assert calculator_runtime(action, False)["reason"] == "calculator_unavailable"
    assert calculator_runtime(dict(action, name="shell"))["reason"] == "tool_not_allowlisted"
    assert calculator_runtime(dict(action, a=True))["reason"] == "invalid_arguments"
    assert calculator_runtime(dict(action, a=1000))["reason"] == "invalid_arguments"
    assert parse_action({"raw": "DIRECT:3", "eos": False})["status"] == "invalid"


def test_runtime_return_does_not_overwrite_a_wrong_model_final_answer(monkeypatch):
    row = {"user": "1+2等於多少？", "system": "計算器=開；風格=短。", "available": True, "image": None, "audio": None}
    traces = iter([{"raw": "TOOL:calculator:1+2", "eos": True}, {"raw": "DIRECT:9", "eos": True}])
    seen = []

    def generated(model, request, count):
        seen.append(request["user"])
        return next(traces)

    monkeypatch.setattr("tiny_perceptron.capstone.generate_trace", generated)
    result = run_assistant(None, row)
    assert result["runtime"]["result"] == "3" and result["answer"] == "9"
    assert "計算器回報：3" in seen[1]
    monkeypatch.setattr(
        "tiny_perceptron.capstone.generate_trace", lambda *_: {"raw": "TOOL:calculator:1+2", "eos": True}
    )
    failed = run_assistant(None, row, runtime=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network")))
    assert failed["runtime"]["reason"] == "runtime_failure" and failed["answer"] is None


def test_dpo_reference_frozen_and_same_facts_not_style_fabrication():
    model = small_model()
    reference = frozen_reference(model)
    splits, _ = build_dataset()
    pairs = preference_pairs(splits["train"])[:2]
    assert all(pair["rejected"].startswith(pair["chosen"]) for pair in pairs)
    loss, _ = preference_loss(model, reference, pairs)
    assert float(loss.detach()) == pytest.approx(0.693147, abs=1e-5)
    loss.backward()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in reference.parameters())
    assert any(parameter.grad is not None for parameter in model.parameters())
    reference.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        preference_loss(model, reference, pairs)


def test_checkpoint_strips_training_state_and_preserves_model_provenance(tmp_path):
    model = small_model()
    source, public = tmp_path / "training.pt", tmp_path / "model.pt"
    optimizer = torch.optim.AdamW(model.parameters())
    save_capstone(
        source,
        model,
        stage="joint",
        step=3,
        optimizer=optimizer,
        metadata={"seed": 42, "secret": "never-publish", "raw_training_record": "private"},
    )
    receipt = export_inference(source, public)
    loaded, payload = load_capstone(public)
    assert receipt["bytes"] > 0
    assert "optimizer" not in payload and "torch_rng" not in payload and "reference" not in payload
    assert "secret" not in payload["metadata"] and "raw_training_record" not in payload["metadata"]
    assert payload["metadata"]["source_checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


def test_cpu_three_update_stage_smoke_and_dependency_guards(tmp_path):
    # Three optimizer updates in total; this validates plumbing, not capability.
    pre = train_stage("pretrain", tmp_path / "pre", steps=1, batch_size=2, validation=False)
    sft = train_stage(
        "sft", tmp_path / "sft", steps=1, batch_size=2, validation=False, input_checkpoint=tmp_path / "pre/model.pt"
    )
    joint = train_stage(
        "joint", tmp_path / "joint", steps=1, batch_size=2, validation=False, input_checkpoint=tmp_path / "sft/model.pt"
    )
    assert all(report["schedule_completed"] and not report["test_evaluated"] for report in (pre, sft, joint))
    assert sft["parent_checkpoint_sha256"] == hashlib.sha256((tmp_path / "pre/model.pt").read_bytes()).hexdigest()
    _, payload = load_capstone(tmp_path / "joint/model-training.pt")
    assert payload["optimizer"]["state"] and payload["training_state"]["sampler_rng"]
    assert payload["torch_rng"].numel() > 0 and payload["step"] == 1
    with pytest.raises(ValueError, match="random"):
        train_stage("sft", tmp_path / "bad", steps=1, validation=False)
    with pytest.raises(ValueError, match="predecessor"):
        train_stage("joint", tmp_path / "wrong", input_checkpoint=tmp_path / "pre/model.pt", steps=1, validation=False)
    with pytest.raises(ValueError, match="training checkpoint"):
        train_stage("joint", tmp_path / "resume", resume=tmp_path / "joint/model.pt", steps=1, validation=False)
