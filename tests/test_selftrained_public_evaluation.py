"""Public GPU evaluation integrity using local synthetic exports; no cloud or heldout generation."""

import dataclasses
import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.selftrained import evaluate
from scripts.selftrained import hf_transport as transport
from scripts.selftrained import modal_runner as runner
from tiny_perceptron.selftrained.dataset import PREPROCESS_VERSION
from tiny_perceptron.selftrained.inference import InferenceAssistant
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import SPECIALS as ACTUAL_SPECIALS
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def file_item(path):
    return {"path": path.name, "sha256": transport.digest(path), "bytes": path.stat().st_size}


def repin(bundle):
    root = bundle.root
    inference = json.loads((root / "inference-manifest.json").read_text())
    inference["files"] = {
        name: transport.digest(root / name) for name in transport.SAFE_EXPORT_FILES if name != "inference-manifest.json"
    }
    write_json(root / "inference-manifest.json", inference)
    bundle.public["files"] = [file_item(root / name) for name in sorted(transport.SAFE_EXPORT_FILES)]


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(32)
    root = tmp_path / "safe-export"
    root.mkdir()
    records = tmp_path / "records.jsonl"
    records.write_text('{"id":"engineering-fixture","image":"pixels.png"}\n')
    (tmp_path / "pixels.png").write_bytes(b"engineering fixture asset; no generation")
    tokenizer = CharacterTokenizer.build(["你好。"], required_chars="".join(chr(i) for i in range(32, 127)))
    config = SelftrainedConfig(vocab_size=tokenizer.vocab_size, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32)
    state = {
        name: tensor.detach().contiguous().clone() for name, tensor in LimitedAssistant(config).state_dict().items()
    }
    save_file(
        state,
        root / "model.safetensors",
        metadata={"origin": "all-neural-weights-random", "schema": "selftrained-random-v1"},
    )
    write_json(root / "model-config.json", dataclasses.asdict(config))
    tokenizer.save(root / "tokenizer.json")
    private = tmp_path / "best.pt"
    private.write_bytes(b"trusted selected private lineage; never deserialized")
    checkpoint = {"stage": "joint", "run_id": "actual-source", "path": "best.pt", "sha256": transport.digest(private)}
    assets = {"pixels.png": transport.digest(tmp_path / "pixels.png")}
    data = {records.name: transport.digest(records)}
    canonical_tokenizer_sha = hashlib.sha256(
        json.dumps(tokenizer.to_dict(), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    inference = {
        "schema": "selftrained-random-v1",
        "selected_checkpoint_sha256": checkpoint["sha256"],
        "files": {},
        "stage": "joint",
        "selected_step": 6,
        "origin": {"kind": "all-neural-weights-random", "seed": 32},
        "preprocess_version": PREPROCESS_VERSION,
        "selection": "validation_loss",
        "data_sha256": data,
        "asset_sha256": assets,
        "tokenizer_sha256": canonical_tokenizer_sha,
    }
    write_json(root / "inference-manifest.json", inference)
    manifest = {
        "records": [file_item(records)],
        "assets": [file_item(tmp_path / "pixels.png")],
        "model_config": {"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "max_length": 512},
    }
    public = {
        "repo_id": transport.PUBLIC_MODEL_REPO,
        "revision": "a" * 40,
        "prefix": "selftrained/v2/moe-joint",
        "files": [],
    }
    training = {
        **inference,
        "config": dataclasses.asdict(config),
        "architecture": "moe",
        "steps": 6,
        "completed_requested_steps": True,
        "selected_checkpoint_available": True,
        "inference_exported": True,
        "test_used_for_selection": False,
    }
    job = {
        "schema_version": 1,
        "stage": "validation",
        "architecture": "moe",
        "checkpoint": checkpoint,
        "public_export": public,
        "max_new_tokens": 128,
    }
    result = SimpleNamespace(
        root=root,
        records=records,
        assets=tmp_path,
        public=public,
        manifest=manifest,
        checkpoint=checkpoint,
        private=private,
        training=training,
        job=job,
        manifest_sha="b" * 64,
    )
    repin(result)
    return result


@pytest.mark.parametrize(
    "violation",
    [
        "repo",
        "revision",
        "prefix",
        "wrong-architecture",
        "traversal",
        "extra",
        "duplicate",
        "pickle",
        "large",
        "negative",
        "bool-bytes",
        "sha",
        "unexpected-field",
    ],
)
def test_public_descriptor_rejects_unauthorized_or_unbounded_downloads(bundle, violation):
    public = deepcopy(bundle.public)
    if violation == "repo":
        public["repo_id"] = "other/mature-model"
    elif violation == "revision":
        public["revision"] = "main"
    elif violation == "prefix":
        public["prefix"] = "selftrained/v1/moe-joint"
    elif violation == "wrong-architecture":
        public["prefix"] = "selftrained/v2/dense-joint"
    elif violation == "traversal":
        public["prefix"] = "selftrained/v2/../moe-joint"
    elif violation == "extra":
        public["files"].append({"path": "best.pt", "sha256": "c" * 64, "bytes": 1})
    elif violation == "duplicate":
        public["files"][0] = deepcopy(public["files"][1])
    elif violation == "pickle":
        public["files"][0]["path"] = "best.pt"
    elif violation == "large":
        public["files"][0]["bytes"] = transport.MAX_RELEASE_BYTES
    elif violation == "negative":
        public["files"][0]["bytes"] = -1
    elif violation == "bool-bytes":
        public["files"][0]["bytes"] = True
    elif violation == "sha":
        public["files"][0]["sha256"] = "main"
    elif violation == "unexpected-field":
        public["token"] = "forbidden"
    with pytest.raises(ValueError):
        transport.validate_public_evaluation(public, "moe")


def fake_hf(bundle, monkeypatch, *, metadata_change=None, source_change=None):
    import huggingface_hub

    calls = []
    files = {item["path"]: item for item in bundle.public["files"]}

    def metadata(url, **kwargs):
        assert kwargs == {"token": False}
        assert url.startswith(
            f"https://huggingface.co/{transport.PUBLIC_MODEL_REPO}/resolve/{bundle.public['revision']}/{bundle.public['prefix']}/"
        )
        name = url.rsplit("/", 1)[-1]
        value = SimpleNamespace(size=files[name]["bytes"], commit_hash=bundle.public["revision"])
        if metadata_change:
            metadata_change(value)
        calls.append(("metadata", name))
        return value

    def download(**kwargs):
        assert kwargs == {
            "repo_id": transport.PUBLIC_MODEL_REPO,
            "filename": f"{bundle.public['prefix']}/{Path(kwargs['filename']).name}",
            "revision": bundle.public["revision"],
            "repo_type": "model",
            "token": False,
            "endpoint": "https://huggingface.co",
        }
        name = Path(kwargs["filename"]).name
        assert name in transport.SAFE_EXPORT_FILES
        calls.append(("download", name))
        return source_change(name) if source_change else bundle.root / name

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def test_download_fetches_exact_immutable_four_files_without_auth_and_rechecks_bytes(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "unused-engineering-sentinel")
    monkeypatch.setenv("HF_ENDPOINT", "https://not-authorized.invalid")
    calls = fake_hf(bundle, monkeypatch)
    output = tmp_path / "download"
    receipt = transport.download_public_evaluation(bundle.public, output, bundle.checkpoint, bundle.manifest, "moe")
    assert set(path.name for path in output.iterdir()) == transport.SAFE_EXPORT_FILES
    assert [name for kind, name in calls if kind == "download"] == sorted(transport.SAFE_EXPORT_FILES)
    assert all(kind == "metadata" for kind, _ in calls[:4])
    assert receipt["authentication"] == "disabled" and receipt["public_export"] == bundle.public
    assert receipt["total_bytes"] == sum(item["bytes"] for item in bundle.public["files"])
    assert receipt["selected_checkpoint_sha256"] == bundle.checkpoint["sha256"]
    assert receipt["checkpoint_sha256"] == transport.digest(output / "model.safetensors")
    (output / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Size/hash"):
        transport.download_public_evaluation(bundle.public, output, bundle.checkpoint, bundle.manifest, "moe")


@pytest.mark.parametrize("key", ["size", "commit_hash"])
def test_hf_metadata_mismatch_rejected_before_any_payload_download(bundle, tmp_path, monkeypatch, key):
    calls = fake_hf(
        bundle, monkeypatch, metadata_change=lambda value: setattr(value, key, 0 if key == "size" else "f" * 40)
    )
    with pytest.raises(ValueError, match="metadata"):
        transport.download_public_evaluation(
            bundle.public, tmp_path / "download", bundle.checkpoint, bundle.manifest, "moe"
        )
    assert not any(kind == "download" for kind, _ in calls)


@pytest.mark.parametrize("change", ["bytes", "sha"])
def test_actual_public_file_mismatch_rejected(bundle, tmp_path, monkeypatch, change):
    bad = tmp_path / "corrupt"
    original = (bundle.root / "model.safetensors").read_bytes()
    bad.write_bytes(original + b"extra" if change == "bytes" else b"X" + original[1:])
    fake_hf(bundle, monkeypatch, source_change=lambda name: bad if name == "model.safetensors" else bundle.root / name)
    with pytest.raises(ValueError, match="Actual public HF file"):
        transport.download_public_evaluation(
            bundle.public, tmp_path / "download", bundle.checkpoint, bundle.manifest, "moe"
        )


@pytest.mark.parametrize(
    "field", ["selected_checkpoint_sha256", "origin", "stage", "data_sha256", "asset_sha256", "tokenizer_sha256"]
)
def test_safe_metadata_identity_rejects_rehashed_wrong_source(bundle, field):
    path = bundle.root / "inference-manifest.json"
    inference = json.loads(path.read_text())
    inference[field] = (
        {"kind": "pretrained"}
        if field == "origin"
        else {}
        if field.endswith("sha256") and field in ("data_sha256", "asset_sha256")
        else "wrong"
    )
    write_json(path, inference)
    repin(bundle)
    with pytest.raises(ValueError):
        transport.public_evaluation_identity(
            bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", bundle.training
        )


@pytest.mark.parametrize("field,value", [("architecture", "dense"), ("width", 32)])
def test_safe_config_rejects_rehashed_wrong_architecture_or_model(bundle, field, value):
    path = bundle.root / "model-config.json"
    config = json.loads(path.read_text())
    config[field] = value
    write_json(path, config)
    repin(bundle)
    with pytest.raises(ValueError, match="frozen data/config"):
        transport.public_evaluation_identity(
            bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", bundle.training
        )


def test_control_contract_constants_match_current_frozen_preprocess_and_tokenizer():
    assert transport.PUBLIC_EVALUATION_PREPROCESS == PREPROCESS_VERSION
    assert transport.PUBLIC_TOKENIZER_SPECIALS == list(ACTUAL_SPECIALS)


@pytest.mark.parametrize(
    "violation",
    [
        "extra-key",
        "role-id",
        "tokenizer-type",
        "unsorted",
        "duplicate",
        "multi-character",
        "added-character",
        "same-size-change",
        "duplicate-json-key",
    ],
)
def test_rehashed_public_tokenizer_rejected_before_reservation(bundle, violation):
    path = bundle.root / "tokenizer.json"
    tokenizer = json.loads(path.read_text())
    if violation == "extra-key":
        tokenizer["extra"] = "not part of canonical schema"
    elif violation == "role-id":
        tokenizer["specials"][0] = "<changed>"
    elif violation == "tokenizer-type":
        tokenizer["type"] = "different"
    elif violation == "unsorted":
        tokenizer["characters"] = list(reversed(tokenizer["characters"]))
    elif violation == "duplicate":
        tokenizer["characters"].append(tokenizer["characters"][-1])
    elif violation == "multi-character":
        tokenizer["characters"][-1] = "兩字"
    elif violation == "added-character":
        tokenizer["characters"] = sorted(tokenizer["characters"] + ["龘"])
    elif violation == "same-size-change":
        tokenizer["characters"] = sorted(tokenizer["characters"][:-1] + ["龘"])
    if violation == "duplicate-json-key":
        path.write_text(json.dumps(tokenizer)[:-1] + ',"type":"selftrained_char_v1"}')
    else:
        write_json(path, tokenizer)
    repin(bundle)
    with pytest.raises(ValueError):
        transport.public_evaluation_identity(
            bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", bundle.training
        )


def test_changed_actual_tokenizer_canonical_hash_cannot_override_training_identity(bundle):
    tokenizer_path = bundle.root / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    tokenizer["characters"] = sorted(tokenizer["characters"][:-1] + ["龘"])
    write_json(tokenizer_path, tokenizer)
    inference_path = bundle.root / "inference-manifest.json"
    inference = json.loads(inference_path.read_text())
    inference["tokenizer_sha256"] = hashlib.sha256(
        json.dumps(tokenizer, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    write_json(inference_path, inference)
    repin(bundle)
    with pytest.raises(ValueError, match="trusted training receipt"):
        transport.public_evaluation_identity(
            bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", bundle.training
        )


@pytest.mark.parametrize("mode", ["old-inference", "old-training", "missing-genuine-training-version"])
def test_preprocess_is_current_v2_and_present_training_version_must_agree(bundle, mode):
    training = deepcopy(bundle.training)
    if mode == "old-inference":
        path = bundle.root / "inference-manifest.json"
        inference = json.loads(path.read_text())
        inference["preprocess_version"] = "legacy-v1"
        write_json(path, inference)
        repin(bundle)
    elif mode == "old-training":
        training["preprocess_version"] = "legacy-v1"
    else:
        training.pop("preprocess_version")
    if mode == "missing-genuine-training-version":
        # Actual train.py receipts lack the field. It is not fabricated or
        # claimed to have been compared; inference remains strictly current V2.
        assert transport.public_evaluation_identity(
            bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", training
        )
    else:
        with pytest.raises(ValueError):
            transport.public_evaluation_identity(
                bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", training
            )


def private_source(bundle, tmp_path, monkeypatch):
    experiments = tmp_path / "private-volume"
    source = experiments / "selftrained-v2/joint/actual-source"
    source.mkdir(parents=True)
    for name in transport.SAFE_EXPORT_FILES:
        shutil.copyfile(bundle.root / name, source / name)
    shutil.copyfile(bundle.private, source / "best.pt")
    write_json(source / "train-receipt.json", bundle.training)
    execution = {
        "status": "completed",
        "revision": "d" * 40,
        "manifest_sha256": bundle.manifest_sha,
        "stage": "joint",
        "run_id": "actual-source",
        "job": {"architecture": "moe", "steps": 6},
    }
    write_json(source / "execution.json", execution)
    receipt = {**execution, "files": [file_item(path) for path in sorted(source.iterdir())]}
    write_json(source / "receipt.json", receipt)
    monkeypatch.setattr(runner, "EXPERIMENTS", experiments)
    return source


def test_pre_reservation_gate_binds_completed_private_receipt_to_public_pins(bundle, tmp_path, monkeypatch):
    source = private_source(bundle, tmp_path, monkeypatch)
    # The public weights are matched to the completed receipt rather than read
    # over Volume here; actual public bytes are checked again at GPU download.
    (source / "model.safetensors").unlink()
    identity = runner.public_evaluation_source_gate("selftrained-v2", bundle.job, bundle.manifest_sha, bundle.manifest)
    assert identity == runner.evaluation_weight_identity(bundle.job)


@pytest.mark.parametrize(
    "violation",
    [
        "failed",
        "manifest",
        "architecture",
        "public-pin",
        "receipt-pin",
        "training",
        "random-origin",
        "missing-training",
        "old-batch",
    ],
)
def test_pre_reservation_source_gate_rejects_wrong_actual_provenance(bundle, tmp_path, monkeypatch, violation):
    source = private_source(bundle, tmp_path, monkeypatch)
    execution = json.loads((source / "execution.json").read_text())
    receipt = json.loads((source / "receipt.json").read_text())
    if violation == "failed":
        execution["status"] = "failed"
    elif violation == "manifest":
        execution["manifest_sha256"] = "e" * 64
    elif violation == "architecture":
        execution["job"]["architecture"] = "dense"
    elif violation == "public-pin":
        bundle.public["files"][0]["sha256"] = "e" * 64
    elif violation == "receipt-pin":
        receipt["files"][0]["sha256"] = "e" * 64
    elif violation == "training":
        (source / "train-receipt.json").write_text("tampered")
    elif violation == "random-origin":
        training = deepcopy(bundle.training)
        training["origin"] = {"kind": "pretrained"}
        write_json(source / "train-receipt.json", training)
        receipt["files"] = [file_item(path) for path in sorted(source.iterdir()) if path.name != "receipt.json"]
    elif violation == "missing-training":
        receipt["files"] = [item for item in receipt["files"] if item["path"] != "train-receipt.json"]
    write_json(source / "execution.json", execution)
    write_json(source / "receipt.json", receipt)
    with pytest.raises(ValueError):
        runner.public_evaluation_source_gate(
            "selftrained-v1" if violation == "old-batch" else "selftrained-v2",
            bundle.job,
            bundle.manifest_sha,
            bundle.manifest,
        )


def protocol(bundle):
    return {
        "version": "selftrained-generation-v2",
        "generation_budget_policy": "full-history-ceiling-min-remaining-context-v1",
        "test_once": True,
        "architecture": "moe",
        "max_new_tokens": 128,
        "data_sha256": {bundle.records.name: transport.digest(bundle.records)},
        "controls": "all",
        "code_sha256": {"scripts/selftrained/evaluate.py": "c" * 64},
        **runner.evaluation_weight_identity(bundle.job),
    }


@pytest.mark.parametrize(
    "field",
    [
        "weight_source",
        "checkpoint_sha256",
        "safe_weights_sha256",
        "inference_manifest_sha256",
        "selected_checkpoint_sha256",
    ],
)
def test_safe_final_protocol_rejects_private_mode_or_changed_frozen_identity(bundle, tmp_path, field):
    value = protocol(bundle)
    path = tmp_path / "frozen.json"
    write_json(path, value)
    assert runner.test_protocol_gate(path, bundle.job, bundle.manifest, value["code_sha256"]) == value
    value[field] = "private_checkpoint" if field == "weight_source" else "e" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner.test_protocol_gate(path, bundle.job, bundle.manifest, value["code_sha256"])


def test_private_final_protocol_preserved_but_safe_protocol_cannot_cross_modes(bundle, tmp_path):
    job = deepcopy(bundle.job)
    job.pop("public_export")
    value = {**protocol(bundle), **runner.evaluation_weight_identity(job)}
    for key in ("weight_source", "safe_weights_sha256", "inference_manifest_sha256", "selected_checkpoint_sha256"):
        value.pop(key)
    path = tmp_path / "frozen.json"
    write_json(path, value)
    assert runner.test_protocol_gate(path, job, bundle.manifest, value["code_sha256"]) == value
    with pytest.raises(ValueError):
        runner.test_protocol_gate(path, bundle.job, bundle.manifest, value["code_sha256"])
    write_json(path, protocol(bundle))
    with pytest.raises(ValueError):
        runner.test_protocol_gate(path, job, bundle.manifest, value["code_sha256"])


def test_resume_weight_identity_and_public_descriptor_are_both_bound(bundle, tmp_path):
    output = tmp_path / "outputs.jsonl"
    output.write_bytes(b"")
    receipt = {
        "schema": "selftrained-evaluation-journal-v1",
        "split": "validation",
        "checkpoint_sha256": runner.evaluation_weight_identity(bundle.job)["checkpoint_sha256"],
        "protocol_sha256": None,
        "completed_count": 0,
        "expected_count": 2,
        "conditions_sha256": "c" * 64,
        "output_byte_count": 0,
        "outputs_sha256": hashlib.sha256(b"").hexdigest(),
        "completed_ids_sha256": hashlib.sha256(b"").hexdigest(),
    }
    path = output.with_name("evaluation-receipt.json")
    write_json(path, receipt)
    job = deepcopy(bundle.job)
    job["resume_evaluation"] = {
        "stage": "validation",
        "run_id": "partial",
        "path": "outputs.jsonl",
        "sha256": transport.digest(output),
        "receipt_sha256": transport.digest(path),
    }
    assert runner.evaluation_resume_gate(output, job) == receipt
    runner.evaluation_public_lineage_gate({"job": job}, job)
    different = deepcopy(job)
    different["public_export"]["revision"] = "f" * 40
    with pytest.raises(ValueError):
        runner.evaluation_public_lineage_gate({"job": job}, different)
    private = deepcopy(job)
    private.pop("public_export")
    with pytest.raises(ValueError):
        runner.evaluation_resume_gate(output, private)
    with pytest.raises(ValueError):
        runner.evaluation_public_lineage_gate({"job": private}, job)


@pytest.mark.parametrize("stage", ["validation", "freeze", "test"])
def test_runner_uses_actual_mutually_exclusive_safe_cli(bundle, tmp_path, monkeypatch, stage):
    job = deepcopy(bundle.job)
    job["stage"] = stage
    job["protocol"] = {"stage": "freeze", "run_id": "actual-freeze", "path": "frozen.json", "sha256": "d" * 64}
    monkeypatch.setattr(runner, "artifact_path", lambda batch, item: tmp_path / item["path"])
    runner.validate_job(job)
    command = runner.trainer_command(
        job, bundle.manifest, bundle.assets, tmp_path / "eval", "selftrained-v2", bundle.root
    )
    parsed = evaluate.parser().parse_args(command[2:])
    assert parsed.model_dir == str(bundle.root) and parsed.checkpoint is None
    assert parsed.split == ("test" if stage == "test" else "validation")
    assert bool(parsed.freeze_protocol) == (stage == "freeze")
    with pytest.raises(ValueError):
        runner.trainer_command(job, bundle.manifest, bundle.assets, tmp_path / "eval", "selftrained-v2")
    with pytest.raises(SystemExit):
        evaluate.parser().parse_args(command[2:] + ["--checkpoint", "private.pt"])


def test_public_export_is_for_evaluation_only_and_joint_best_lineage(bundle):
    for stage in ("pretrain", "release", "prepare"):
        with pytest.raises(ValueError, match="only for evaluation"):
            runner.validate_job({**bundle.job, "stage": stage})
    for field, value in (("stage", "ocr"), ("path", "latest.pt")):
        job = deepcopy(bundle.job)
        job["checkpoint"][field] = value
        with pytest.raises(ValueError, match="selected private joint"):
            runner.validate_job(job)


def test_verified_safe_loader_never_deserializes_pickle_and_actual_protocol_uses_safe_hashes(bundle, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Public safe evaluation must never use torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    identity = transport.public_evaluation_identity(
        bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "moe", bundle.training
    )
    assistant = InferenceAssistant(bundle.root, bundle.assets, manifest_sha256=identity["inference_manifest_sha256"])
    inference = json.loads((bundle.root / "inference-manifest.json").read_text())
    args = SimpleNamespace(
        model_dir=str(bundle.root),
        checkpoint=None,
        records=[str(bundle.records)],
        asset_dir=str(bundle.assets),
        max_new_tokens=128,
        controls="all",
    )
    actual = evaluate.protocol_contents(
        args, {**inference, "config": assistant.receipt["config"]}, [{"image": "pixels.png", "messages": []}]
    )
    assert all(actual[key] == value for key, value in identity.items())
    assert not calls and not assistant.model.training


def test_authorized_dense_public_export_uses_actual_dense_safe_loader(bundle, monkeypatch):
    config = SelftrainedConfig(
        **{**json.loads((bundle.root / "model-config.json").read_text()), "architecture": "dense"}
    )
    state = {
        name: tensor.detach().contiguous().clone() for name, tensor in LimitedAssistant(config).state_dict().items()
    }
    save_file(
        state,
        bundle.root / "model.safetensors",
        metadata={"origin": "all-neural-weights-random", "schema": "selftrained-random-v1"},
    )
    write_json(bundle.root / "model-config.json", dataclasses.asdict(config))
    bundle.public["prefix"] = "selftrained/v2/dense-joint"
    bundle.job["architecture"] = "dense"
    bundle.training["architecture"] = "dense"
    bundle.training["config"] = dataclasses.asdict(config)
    repin(bundle)
    runner.validate_job(bundle.job)
    identity = transport.public_evaluation_identity(
        bundle.root, bundle.public, bundle.checkpoint, bundle.manifest, "dense", bundle.training
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense public evaluation must not deserialize a private checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    assistant = InferenceAssistant(bundle.root, bundle.assets, manifest_sha256=identity["inference_manifest_sha256"])
    assert assistant.receipt["config"]["architecture"] == "dense"


def test_rejected_public_source_cannot_reserve_or_commit_ledger_and_gpu_has_no_secret(bundle, tmp_path, monkeypatch):
    registered = {}
    volume_calls = []

    class Image:
        def __getattr__(self, name):
            assert name in ("pip_install", "env", "add_local_dir")
            return lambda *args, **kwargs: self

    class App:
        def function(self, **options):
            def register(function):
                registered[function.__name__] = (function, options)
                return function

            return register

    volume = SimpleNamespace(reload=lambda: volume_calls.append("reload"), commit=lambda: volume_calls.append("commit"))
    modal = SimpleNamespace(
        App=lambda name: App(),
        Image=SimpleNamespace(debian_slim=lambda **kwargs: Image()),
        Volume=SimpleNamespace(from_name=lambda *args, **kwargs: volume),
        Secret=SimpleNamespace(from_name=lambda *args, **kwargs: "release-only-secret-reference"),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setenv("SELFTRAINED_MODAL_PHASE", "control")
    ledger = tmp_path / "budget.json"
    write_json(
        ledger,
        {
            "budget_usd": "55.00",
            "reserved_total_usd": "36.71",
            "reservations": [{"reserved_usd": "36.71", "status": "failed"}],
        },
    )
    before = ledger.read_bytes()
    monkeypatch.setattr(runner, "LEDGER", ledger)
    monkeypatch.setattr(runner, "prepared_records_gate", lambda *args: None)

    def rejected(*args):
        raise ValueError("unreviewed public source")

    monkeypatch.setattr(runner, "public_evaluation_source_gate", rejected)
    runner.register_modal()
    assert "secrets" not in registered["gpu_remote"][1]
    assert registered["release_remote"][1]["secrets"] == ["release-only-secret-reference"]
    reserve, _ = registered["reserve_remote"]
    with pytest.raises(ValueError, match="unreviewed public source"):
        reserve(
            "attempt",
            "selftrained-v2",
            "a" * 40,
            bundle.manifest,
            bundle.manifest_sha,
            bundle.job,
            "c" * 64,
            {},
            {},
            transport.digest(ledger),
        )
    assert ledger.read_bytes() == before and volume_calls == ["reload"]
