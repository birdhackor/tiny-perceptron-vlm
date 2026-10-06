"""V2 sampling/frozen bridges: CPU wiring evidence, not capability acceptance."""

import copy
import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from scripts.selftrained import train
from tiny_perceptron.selftrained.dataset import RecordEncoder, train_tokenizer
from tiny_perceptron.selftrained.inference import verify_export
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


@pytest.fixture
def family_records():
    # Every row represents a distinct source, with deliberately unequal buckets.
    records = []
    for task, intents in (
        ("text", ("address", "app_error", "card_issues")),
        ("tool_call", ("",)),
        ("tool_reply", ("",)),
        ("vision_clothing", ("",)),
        ("vision_relation", ("",)),
        ("ocr", ("",)),
        ("voice_qa", ("address", "app_error", "card_issues")),
        ("voice_topic_continuation", ("address", "app_error", "card_issues")),
    ):
        for intent in intents:
            for source in range(3 if task == "ocr" else 2):
                identity = f"{task}:{intent}:source{source}"
                records.append({"id": identity, "group_id": identity, "task": task, "supervision": {"intent": intent}})
    return records


def test_family_sampling_covers_independent_sources_and_intents(family_records):
    sampler = train.BalancedSampler(family_records, 19, mode="task-family")
    sampled = sampler.batch(600)
    families = Counter(train.TASK_FAMILY_BY_TASK[record["task"]] for record in sampled)
    assert families == dict.fromkeys(train.TASK_FAMILIES, 120)
    assert {record["group_id"] for record in sampled} == {record["group_id"] for record in family_records}
    buckets = Counter((record["task"], record["supervision"]["intent"]) for record in sampled)
    assert buckets[("text", "card_issues")] == 40
    assert buckets[("voice_qa", "card_issues")] == 20
    assert buckets[("vision_relation", "")] == 60
    assert buckets[("ocr", "")] == 120


@pytest.mark.parametrize("mode", ["bucket", "task-family"])
def test_sampler_mid_family_resume_is_independent_of_model_rng(family_records, mode):
    whole = train.BalancedSampler(family_records, 31, mode=mode)
    expected = [record["id"] for record in whole.batch(117)]
    partial = train.BalancedSampler(family_records, 31, mode=mode)
    prefix = [record["id"] for record in partial.batch(17)]
    state = partial.state_dict()
    torch.rand(1000)
    restored = train.BalancedSampler(family_records, 999, mode=mode)
    restored.load_state_dict(state)
    assert prefix + [record["id"] for record in restored.batch(100)] == expected
    assert torch.equal(restored.generator.get_state(), whole.generator.get_state())
    assert restored.family_draws == whole.family_draws


def test_sampler_rejects_changed_mode_counters_or_unmapped_families(family_records):
    partial = train.BalancedSampler(family_records, 7, mode="task-family")
    partial.batch(17)
    with pytest.raises(ValueError, match="sampling mode"):
        train.BalancedSampler(family_records, 7).load_state_dict(partial.state_dict())
    broken = partial.state_dict()
    broken["family_draws"]["ocr"] += 1
    with pytest.raises(ValueError, match="counters"):
        partial.load_state_dict(broken)
    with pytest.raises(ValueError, match="未知 task"):
        train.BalancedSampler(family_records + [{"task": "unknown"}], 7, mode="task-family")
    with pytest.raises(ValueError, match="五種資料"):
        train.BalancedSampler([r for r in family_records if r["task"] != "ocr"], 7, mode="task-family")


def test_legacy_bucket_state_and_default_sequence_remain_compatible(family_records):
    original = train.BalancedSampler(family_records, 17)
    original.batch(11)
    state = original.state_dict()
    legacy = {name: state[name] for name in ("draws", "generator")}
    restored = train.BalancedSampler(family_records, 333, mode="bucket")
    restored.load_state_dict(legacy)
    assert [r["id"] for r in restored.batch(50)] == [r["id"] for r in original.batch(50)]
    # The default still cycles all existing task:intent buckets, rather than families.
    assert Counter(r["task"] for r in train.BalancedSampler(family_records, 17).batch(13))["text"] == 3


@pytest.fixture
def neural_corpus(tmp_path):
    Image.fromarray(np.arange(64 * 128, dtype=np.uint8).reshape(64, 128)).save(tmp_path / "pixels.png")
    sf.write(tmp_path / "wave.wav", np.sin(np.arange(1600) * 0.05).astype("float32"), 8000)
    records = []
    for split in ("train", "validation"):
        for task in ("text", "tool_call", "vision_clothing", "ocr", "voice_qa"):
            for source in range(2):
                identity = f"{split}-{task}-source{source}"
                record = {
                    "id": identity,
                    "group_id": identity,
                    "split": split,
                    "task": task,
                    "messages": [{"role": "user", "content": "請回答。"}, {"role": "assistant", "content": "好。"}],
                    "supervision": {},
                }
                if task == "vision_clothing":
                    record.update(image="pixels.png", image_layout={"slots": [[0, 0, 64, 64], [64, 0, 128, 64]]})
                    record["supervision"]["vision_labels"] = [source, 2]
                elif task == "ocr":
                    record.update(image="pixels.png", roi=[source * 10, 0, 100, 32])
                    record["supervision"]["ocr_text"] = "大小"
                    record["messages"][-1]["content"] = "大小"
                elif task == "voice_qa":
                    record.update(audio="wave.wav", modality_message_index=0)
                    record["supervision"].update(intent="address" if source == 0 else "app_error", intent_id=source)
                records.append(record)
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "experts": 2, "top_k": 2})
    )
    return tmp_path, path, config, records


@pytest.mark.parametrize("architecture", ["moe", "dense"])
def test_real_optimizer_step_preserves_heads_and_updates_every_bridge(neural_corpus, architecture):
    root, _, _, records = neural_corpus
    torch.set_num_threads(2)
    torch.manual_seed(19)
    tokenizer = train_tokenizer(records)
    encoder = RecordEncoder(tokenizer, root, 128)
    model = LimitedAssistant(
        SelftrainedConfig(
            vocab_size=tokenizer.vocab_size,
            architecture=architecture,
            width=16,
            layers=1,
            heads=2,
            kv_heads=1,
            ffn_hidden=32,
            experts=2,
            max_length=128,
        )
    )
    # Verify the transition after a perception-only phase, which froze the LM.
    train.set_trainable(model, "audio")
    parameters = train.set_trainable(model, "joint", freeze_perception_backbones=True)
    examples = [record for record in records if record["split"] == "train"]
    batch = encoder.batch(examples)
    with torch.no_grad():
        before_heads = [entry["logits"].clone() for entry in model(**batch)["perception"]]
    before_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss, _ = train.objective(model, encoder, examples, "joint", 1.0, 0.01)
    loss.backward()
    for prefix in ("lm.", *train.PERCEPTION_BRIDGE_PREFIXES):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)
        ), prefix
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(parameter.grad is None for _, parameter in frozen)
    assert not model.vision_encoder.projection.weight.requires_grad
    assert not model.ocr_encoder.column_projection.weight.requires_grad
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    optimizer.step()
    with torch.no_grad():
        after_heads = [entry["logits"] for entry in model(**batch)["perception"]]
    assert all(torch.equal(before, after) for before, after in zip(before_heads, after_heads))
    assert all(torch.equal(before_parameters[name], parameter) for name, parameter in frozen)
    for prefix in ("lm.", *train.PERCEPTION_BRIDGE_PREFIXES):
        assert any(
            not torch.equal(before_parameters[n], p) for n, p in model.named_parameters() if n.startswith(prefix)
        ), prefix
    train.set_trainable(model, "joint")
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("option", [["--freeze-perception-backbones"], ["--sampling-mode", "task-family"]])
def test_new_options_are_rejected_outside_joint_before_loading(option):
    with pytest.raises(ValueError, match="只可用於 joint"):
        train.main(
            [
                "--stage",
                "sft",
                "--steps",
                "1",
                "--records",
                "absent",
                "--asset-dir",
                "absent",
                "--output-dir",
                "absent",
                *option,
            ]
        )


def test_frozen_family_trainer_resume_metadata_and_option_guards(neural_corpus, capsys):
    root, path, config, _ = neural_corpus
    common = [
        "--records",
        str(path),
        "--asset-dir",
        str(root),
        "--stage",
        "joint",
        "--batch-size",
        "7",
        "--context",
        "128",
        "--config",
        str(config),
        "--eval-every",
        "10",
        "--save-every",
        "1",
        "--freeze-perception-backbones",
        "--sampling-mode",
        "task-family",
        "--export-inference",
    ]
    train.main(common + ["--output-dir", str(root / "whole"), "--steps", "3"])
    train.main(common + ["--output-dir", str(root / "part"), "--steps", "1"])
    train.main(
        common + ["--output-dir", str(root / "resumed"), "--steps", "3", "--resume", str(root / "part/latest.pt")]
    )
    whole = torch.load(root / "whole/latest.pt", weights_only=False)
    resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
    assert whole["tokens"] == resumed["tokens"]
    assert whole["sampler"]["draws"] == resumed["sampler"]["draws"] == 21
    assert whole["sampler"]["family_draws"] == resumed["sampler"]["family_draws"]
    assert torch.equal(whole["sampler"]["generator"], resumed["sampler"]["generator"])
    assert all(torch.equal(weight, resumed["model"][name]) for name, weight in whole["model"].items())
    for directory in (root / "whole", root / "resumed"):
        receipt = json.loads((directory / "train-receipt.json").read_text())
        manifest = json.loads((directory / "inference-manifest.json").read_text())
        for document in (receipt, manifest, resumed["training_options"]):
            assert document["freeze_perception_backbones"] is True
            assert document["sampling_mode"] == "task-family"
        assert receipt["comparison"]["sample_draws"] == 21
        assert receipt["comparison"]["input_tokens_observed"] == receipt["tokens"]
        assert receipt["comparison"]["active_ffn_count"] == 2
        verified, _ = verify_export(directory)
        assert verified == manifest
    for field, replacement in (
        ("--freeze-perception-backbones", []),
        ("--sampling-mode", ["--sampling-mode", "bucket"]),
    ):
        changed = copy.copy(common)
        index = changed.index(field)
        del changed[index : index + (1 if field == "--freeze-perception-backbones" else 2)]
        changed[index:index] = replacement
        with pytest.raises(ValueError, match="resume 改變"):
            train.main(
                changed
                + ["--output-dir", str(root / "rejected"), "--steps", "3", "--resume", str(root / "part/latest.pt")]
            )
        assert not (root / "rejected/latest.pt").exists()
    capsys.readouterr()
