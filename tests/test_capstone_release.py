"""Capstone artifacts retain inference only; published students load verified bytes."""

import copy
import json
import shutil
from dataclasses import asdict

import pytest
import torch

from scripts.capstone_release import (
    PUBLIC_REPO,
    build_public_manifest,
    build_public_manifest_from_receipt,
    clean_capstone_payload,
    export_capstone,
    file_sha256,
    merge_public_manifests,
    prepare_approval,
    public_stage_id,
    validate_capstone_payload,
    validate_manifest,
)
from scripts.course_release import validate_approval, validate_export
from scripts.fetch_capstone import fetch_capstone
from tiny_perceptron.capstone import CapstoneModel, export_inference, load_capstone, save_capstone
from tiny_perceptron.capstone_quantization import (
    load_quantized_capstone,
    quantizable_weights,
    quantize_capstone,
    restore_quantized_payload,
    tensor_bytes,
)
from tiny_perceptron.model import ModelConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(17)
    model = CapstoneModel(ModelConfig(width=8, heads=2, kv_heads=1, experts=4, top_k=2, rotary=True, norm="rms"))
    path = tmp_path / "source.pt"
    save_capstone(
        path,
        model,
        stage="dpo",
        step=7,
        optimizer=torch.optim.AdamW(model.parameters()),
        metadata={"secret": "DO NOT EXPORT", "dataset": ["PRIVATE"], "seed": 17},
        reference=model,
        state={"samples": ["PRIVATE"]},
    )
    return path, model


def approved(source, directory, path="source.pt", stage="dpo"):
    draft = prepare_approval(
        directory,
        {stage: path},
        training_revision="a" * 40,
        private_repo="owner/private",
        private_revision="b" * 40,
        private_prefix="course/capstone-v1/capstone/run-1",
    )
    assert draft["approved"] is False and draft["reviewed"] is False
    assert draft["files"][0]["redistribution_approved"] is False
    with pytest.raises(ValueError, match="root"):
        validate_approval(draft, "capstone", "capstone-v1", "owner/private")
    # This test approval is deliberately limited to generated test tensors.
    draft.update(approved=True, reviewed=True)
    draft["files"][0]["redistribution_approved"] = True
    return draft


def test_export_removes_training_references_and_preserves_all_logits(source, tmp_path):
    path, model = source
    approval = approved(source, tmp_path)
    export_capstone(tmp_path, tmp_path / "public", approval, {"training_revision": "a" * 40, "secret": "PRIVATE"})
    target = tmp_path / "public/dpo/model.pt"
    loaded, clean = load_capstone(target)
    assert clean["architecture"] == {"type": "CapstoneModel"}
    assert clean["inference_only"] is True and clean["config"] == asdict(model.config)
    assert set(clean) == {
        "format_version",
        "inference_only",
        "config",
        "model",
        "stage",
        "step",
        "tokenizer",
        "data_version",
        "architecture",
        "metadata",
    }
    assert clean["metadata"] == {"seed": 17, "training_revision": "a" * 40}
    assert validate_export(target) == "capstone-v1"
    ids = torch.tensor([[1, 3, 5, 6, 8]])
    options = {"images": torch.rand(1, 3, 16, 16), "audio_features": torch.randn(1, 16)}
    model.eval(), loaded.eval()
    assert torch.equal(model(ids, **options)["logits"], loaded(ids, **options)["logits"])
    assert "MIT License" in (tmp_path / "public/LICENSE").read_text()
    assert (tmp_path / "public/THIRD_PARTY_NOTICES.md").is_file()
    assert file_sha256(path) == approval["files"][0]["sha256"]


@pytest.mark.parametrize("architecture", [{}, {"type": "TinyLM"}, {"type": "CapstoneModel", "secret": "PRIVATE"}])
def test_capstone_export_requires_exact_explicit_architecture(source, architecture):
    saved = torch.load(source[0], weights_only=True)
    with pytest.raises(ValueError, match="architecture"):
        clean_capstone_payload(saved, {}, {"architecture": architecture})


@pytest.mark.parametrize("where", ["config", "tokenizer", "model"])
def test_nested_training_payload_cannot_be_disguised_as_inference(source, where):
    saved = torch.load(source[0], weights_only=True)
    saved[where]["secret"] = "PRIVATE"
    with pytest.raises(ValueError):
        clean_capstone_payload(saved, {}, {"architecture": {"type": "CapstoneModel"}})


def test_metadata_cannot_be_renamed_into_checkpoint(source, tmp_path):
    (tmp_path / "notes.json").write_text("{}")
    approval = approved(source, tmp_path)
    approval["files"] = [
        {
            "path": "notes.json",
            "output": "model.pt",
            "kind": "metadata",
            "sha256": file_sha256(tmp_path / "notes.json"),
            "license": "MIT",
            "redistribution_approved": True,
        }
    ]
    with pytest.raises(ValueError, match="checkpoint"):
        validate_approval(approval, "capstone", "capstone-v1", "owner/private")


@pytest.mark.parametrize("bits", [4, 8])
def test_ptq_packs_real_weights_retains_router_and_reloads(source, tmp_path, bits):
    path, original = source
    output = tmp_path / f"int{bits}.pt"
    receipt = quantize_capstone(path, output, bits)
    loaded, payload = load_quantized_capstone(output)
    assert receipt["tensor_bytes"] < receipt["float_tensor_bytes"]
    assert receipt["tensor_bytes"] == tensor_bytes(payload["model"]) + tensor_bytes(payload["quantized"])
    assert receipt["file_bytes"] == output.stat().st_size
    assert receipt["sha256"] == file_sha256(output)
    assert not {"optimizer", "reference", "torch_rng", "python_rng", "training_state"} & set(payload)
    assert payload["metadata"] == {"source_checkpoint_sha256": file_sha256(path)}
    names = quantizable_weights(original)
    assert set(payload["quantized"]) == set(names) and all("router" not in name for name in names)
    for name, tensor in payload["model"].items():
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, original.state_dict()[name])
    for name, packed in payload["quantized"].items():
        expected = original.state_dict()[name]
        if bits == 4:
            assert packed["values"].dtype == torch.uint8
            assert packed["values"].numel() == (expected.numel() + 1) // 2
        else:
            assert packed["values"].dtype == torch.int8 and packed["values"].shape == expected.shape
        reconstructed = loaded.state_dict()[name]
        assert torch.all((expected - reconstructed).abs() <= packed["scale"] * 0.5001)
    logits = loaded(torch.tensor([[1, 3, 8]]))["logits"]
    assert logits.shape == (1, 3, 264) and torch.isfinite(logits).all()
    approval = approved(source, tmp_path, output.name, f"dpo-int{bits}")
    export_capstone(tmp_path, tmp_path / f"public-int{bits}", approval, {})
    exported = tmp_path / f"public-int{bits}/dpo-int{bits}/model.pt"
    clean_model, clean = load_quantized_capstone(exported)
    assert validate_capstone_payload(clean) == "capstone-ptq-v1"
    loaded.eval(), clean_model.eval()
    assert torch.equal(loaded(torch.tensor([[1, 3, 8]]))["logits"], clean_model(torch.tensor([[1, 3, 8]]))["logits"])


@pytest.mark.parametrize("change", ["router", "scale", "packed", "extra", "float"])
def test_ptq_loader_rejects_corrupted_or_smuggled_tensors(source, tmp_path, change):
    output = tmp_path / "int4.pt"
    quantize_capstone(source[0], output, 4)
    payload = torch.load(output, weights_only=True)
    name = next(iter(payload["quantized"]))
    if change == "router":
        payload["quantization"]["linear_weights"].append("language.blocks.0.ffn.router.weight")
    elif change == "scale":
        payload["quantized"][name]["scale"].fill_(float("nan"))
    elif change == "packed":
        payload["quantized"][name]["values"] = payload["quantized"][name]["values"][:-1]
    elif change == "extra":
        payload["quantized"][name]["secret"] = "PRIVATE"
    else:
        payload["model"]["language.embedding.weight"] = payload["model"]["language.embedding.weight"].half()
    with pytest.raises(ValueError):
        restore_quantized_payload(payload)


def public_fixture(source, tmp_path, monkeypatch):
    directory = tmp_path / "public"
    export_capstone(tmp_path, directory, approved(source, tmp_path), {})
    calls = []

    def download(repo, filename, **options):
        calls.append((repo, filename, options))
        return str(directory / filename.removeprefix("capstone/run-1/"))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    manifest = build_public_manifest(directory, revision="c" * 40, prefix="capstone/run-1")
    assert all(options == {"revision": "c" * 40, "token": False} for _, _, options in calls)
    assert all(repo == PUBLIC_REPO for repo, _, _ in calls)
    return directory, manifest, calls


def test_public_verification_and_download_load_actual_checkpoint(source, tmp_path, monkeypatch):
    directory, manifest, calls = public_fixture(source, tmp_path, monkeypatch)
    assert len(calls) == 4 and manifest["models"][0]["id"] == "dpo"
    target = fetch_capstone(manifest, "dpo", tmp_path / "downloads")
    loaded, clean = load_capstone(target / "model.pt")
    assert loaded.config == source[1].config and clean["inference_only"] is True
    assert (target / "model.pt").read_bytes() == (directory / "dpo/model.pt").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        fetch_capstone(manifest, "dpo", tmp_path / "downloads")


def test_failed_download_never_installs_partial_stage(source, tmp_path, monkeypatch):
    directory, manifest, _ = public_fixture(source, tmp_path, monkeypatch)
    (directory / "LICENSE").write_text("corrupted after publication")
    with pytest.raises(ValueError, match="SHA/size"):
        fetch_capstone(manifest, "dpo", tmp_path / "downloads")
    assert not (tmp_path / "downloads/dpo").exists()
    assert not list((tmp_path / "downloads").iterdir())


def test_public_manifest_cannot_be_created_when_hf_bytes_do_not_match(source, tmp_path, monkeypatch):
    directory, _, _ = public_fixture(source, tmp_path, monkeypatch)
    wrong = tmp_path / "wrong"
    wrong.write_bytes(b"wrong")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(wrong))
    with pytest.raises(ValueError, match="public HF bytes"):
        build_public_manifest(directory, revision="c" * 40, prefix="capstone/run-1")


@pytest.mark.parametrize("change", ["revision", "repo", "path", "output", "duplicate", "hash"])
def test_manifest_rejects_moving_revisions_and_unsafe_paths(source, tmp_path, monkeypatch, change):
    _, original, _ = public_fixture(source, tmp_path, monkeypatch)
    manifest = copy.deepcopy(original)
    if change == "revision":
        manifest["revision"] = "main"
    elif change == "repo":
        manifest["repo"] = "another/repo"
    elif change in ("path", "output"):
        manifest["models"][0]["files"][0][change] = "../escape.pt"
    elif change == "duplicate":
        manifest["models"].append(manifest["models"][0])
    else:
        manifest["models"][0]["files"][0]["sha256"] = "unknown"
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_missing_manifest_explains_unpublished_status(tmp_path, capsys, monkeypatch):
    from scripts.fetch_capstone import main

    monkeypatch.setattr("sys.argv", ["fetch_capstone.py", "--manifest", str(tmp_path / "unpublished.json"), "--list"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert "have not been published" in capsys.readouterr().err


def test_receipt_verification_uses_final_public_card_and_rechecks_export_hashes(source, tmp_path, monkeypatch):
    directory, _, _ = public_fixture(source, tmp_path, monkeypatch)
    (directory / "README.md").write_text("# Actual final pinned card\n\nGenerated by the release workflow.\n")
    files = [
        {
            "path": "capstone/run-1/" + path.relative_to(directory).as_posix(),
            "output": path.relative_to(directory).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in directory.rglob("*")
        if path.is_file()
    ]
    receipt = {"id": "capstone_deployment", "repo": PUBLIC_REPO, "revision": "c" * 40, "files": files}
    manifest = build_public_manifest_from_receipt(receipt)
    assert manifest["models"][0]["id"] == "dpo"
    card = next(item for item in manifest["models"][0]["files"] if item["output"] == "README.md")
    assert card["sha256"] == file_sha256(directory / "README.md")
    bad = copy.deepcopy(receipt)
    next(item for item in bad["files"] if item["output"] == "dpo/model.pt")["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="export manifest"):
        build_public_manifest_from_receipt(bad)


def test_export_clones_tensor_storage_to_exclude_unrelated_backing_data(source):
    saved = torch.load(source[0], weights_only=True)
    name = "language.embedding.weight"
    tensor = saved["model"][name]
    storage = torch.full((tensor.numel() + 5000,), 123456.0)
    storage[: tensor.numel()] = tensor.flatten()
    saved["model"][name] = storage[: tensor.numel()].view_as(tensor)
    clean = clean_capstone_payload(saved, {}, {"architecture": {"type": "CapstoneModel"}})
    assert clean["model"][name].untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert torch.equal(clean["model"][name], tensor)


def test_prepare_supports_actual_deployment_experiment_without_claiming_approval(source, tmp_path):
    draft = prepare_approval(
        tmp_path,
        {"dpo": "source.pt"},
        training_revision="a" * 40,
        private_repo="owner/private",
        private_revision="b" * 40,
        private_prefix="course/course-v1/capstone_deployment/run-1",
        batch_id="course-v1",
        experiment_id="capstone_deployment",
    )
    assert draft["experiment_id"] == "capstone_deployment" and draft["batch_id"] == "course-v1"
    assert draft["approved"] is False


def test_image_counterfactual_changes_actual_features_and_expected_answer():
    from scripts.course_experiments.capstone_deployment import image_counterfactuals
    from tiny_perceptron.capstone import build_dataset, modality_tensors

    rows = build_dataset()[0]["test"]
    swaps = image_counterfactuals(rows)
    original = {row["id"]: row for row in rows}
    assert swaps
    for swapped in swaps:
        row = original[swapped["id"].removesuffix("-image-swap")]
        assert swapped["user"] == row["user"] and swapped["system"] == row["system"]
        assert swapped["audio"] == row["audio"] and swapped["answer"] != row["answer"]
        image, _ = modality_tensors(row)
        new_image, _ = modality_tensors(swapped)
        assert not torch.equal(image, new_image)


def test_cache_check_covers_text_and_paired_modalities(source):
    from scripts.course_experiments.capstone_deployment import cache_consistency
    from tiny_perceptron.capstone import build_dataset

    first = {}
    for row in build_dataset()[0]["validation"]:
        first.setdefault(row["task"], row)
    rows = [first[task] for task in ("calculator", "joint", "image_color", "audio")]
    result = cache_consistency(source[1], rows, max_new_tokens=4)
    assert result["count"] == 4 and result["all_generated_ids_equal"] is True
    assert result["all_logits_close"] is True
    assert all(record["same_history_logit_comparisons"] for record in result["records"])


def test_forward_backward_benchmark_never_updates_weights(source):
    from scripts.course_experiments.capstone_deployment import benchmark_training_step
    from tiny_perceptron.capstone import build_dataset

    rows = build_dataset()[0]["train"][:2]
    result = benchmark_training_step(source[1], rows, warmup=1, measured=2)
    assert result["optimizer_updates"] == 0 and result["weights_unchanged"] is True
    assert len(result["seconds"]) == 2 and all(value > 0 for value in result["seconds"])
    assert result["batch_ids"] == [row["id"] for row in rows]


def test_prepared_data_export_matches_every_stage_and_is_downloaded_with_weights(source, tmp_path, monkeypatch):
    from tiny_perceptron.capstone import build_dataset, digest

    splits, manifest = build_dataset()
    saved = torch.load(source[0], weights_only=True)
    saved["metadata"]["data_manifest"] = manifest
    torch.save(saved, source[0])
    data = {"schema_version": 1, "license": "MIT", "manifest": manifest, "splits": splits}
    (tmp_path / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    approval = approved(source, tmp_path)
    assert approval["files"][1]["kind"] == "dataset" and approval["files"][1]["redistribution_approved"] is False
    approval["files"][1]["redistribution_approved"] = True
    directory = tmp_path / "public"
    export_capstone(tmp_path, directory, approval, {})
    clean = torch.load(directory / "dpo/model.pt", weights_only=True)
    assert "data_manifest" not in clean["metadata"]
    assert clean["metadata"]["dataset_manifest_sha256"] == digest(manifest)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, **options: str(directory / filename.removeprefix("capstone/run-1/")),
    )
    public = build_public_manifest(directory, revision="c" * 40, prefix="capstone/run-1")
    target = fetch_capstone(public, "dpo", tmp_path / "downloads")
    downloaded = json.loads((target / "data.json").read_text(encoding="utf-8"))
    assert downloaded == data
    quantize_capstone(source[0], tmp_path / "dpo-int4.pt", 4)
    packed = torch.load(tmp_path / "dpo-int4.pt", weights_only=True)
    assert packed["metadata"]["dataset_manifest_sha256"] == digest(manifest)


def test_prepared_data_rejects_wrong_rows_or_checkpoint_binding(source, tmp_path):
    from tiny_perceptron.capstone import build_dataset

    splits, manifest = build_dataset()
    data = {"schema_version": 1, "license": "MIT", "manifest": manifest, "splits": splits}
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="bind"):
        prepare_approval(
            tmp_path,
            {"dpo": "source.pt"},
            training_revision="a" * 40,
            private_repo="owner/private",
            private_revision="b" * 40,
            private_prefix="course/capstone-v1/capstone/run-1",
        )
    data["splits"]["train"][0]["answer"] = "changed"
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="rows"):
        prepare_approval(
            tmp_path,
            {"dpo": "source.pt"},
            training_revision="a" * 40,
            private_repo="owner/private",
            private_revision="b" * 40,
            private_prefix="course/capstone-v1/capstone/run-1",
        )


def student_release_fixture(source, tmp_path, monkeypatch):
    from tiny_perceptron.capstone import STAGES, build_dataset

    splits, manifest = build_dataset()
    saved = torch.load(source[0], weights_only=True)
    saved["metadata"]["data_manifest"] = manifest
    main_sources = tmp_path / "main-sources"
    main_sources.mkdir()
    for stage in STAGES:
        training = tmp_path / f"{stage}-training.pt"
        torch.save(dict(saved, stage=stage), training)
        parent = tmp_path / "parents" / stage / "model.pt"
        export_inference(training, parent)
        # Retain the private training-origin hash; approved raw inference identity
        # is a separate field and must drive the teacher comparison.
        shutil.copyfile(parent, main_sources / f"{stage}.pt")
    teacher_sha = file_sha256(tmp_path / "parents/dpo/model.pt")
    quantize_capstone(tmp_path / "parents/dpo/model.pt", main_sources / "dpo-int4.pt", 4)
    quantize_capstone(tmp_path / "parents/dpo/model.pt", main_sources / "dpo-int8.pt", 8)
    (main_sources / "data.json").write_text(
        json.dumps({"schema_version": 1, "license": "MIT", "manifest": manifest, "splits": splits}), encoding="utf-8"
    )
    student_sources = tmp_path / "student-sources"
    student_sources.mkdir()
    dense = CapstoneModel(ModelConfig(width=48, layers=2, heads=2, kv_heads=1, rotary=True, norm="rms"))
    for mode in ("ce", "kd"):
        training = tmp_path / f"student-{mode}-training.pt"
        save_capstone(
            training,
            dense,
            stage="joint",
            step=1,
            metadata={"student_mode": mode, "teacher_checkpoint_sha256": teacher_sha, "data_manifest": manifest},
        )
        export_inference(training, student_sources / f"{mode}/model.pt")
    quantize_capstone(student_sources / "kd/model.pt", student_sources / "model-int4.pt", 4)
    specifications = {
        "main": (
            main_sources,
            {stage: f"{stage}.pt" for stage in (*STAGES, "dpo-int4", "dpo-int8")},
            "capstone_deployment",
            "c" * 40,
        ),
        "student": (
            student_sources,
            {"student-ce": "ce/model.pt", "student-kd": "kd/model.pt", "student-kd-int4": "model-int4.pt"},
            "capstone_student",
            "d" * 40,
        ),
    }
    remote = {}
    calls = []

    def download(repo, filename, **options):
        calls.append((repo, filename, options))
        return str(remote[filename])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    manifests, approvals = {}, {}
    for label, (directory, stages, experiment, revision) in specifications.items():
        approval = prepare_approval(
            directory,
            stages,
            training_revision="a" * 40,
            private_repo="owner/private",
            private_revision="b" * 40,
            private_prefix=f"course/course-integration-v2/{experiment}/run-1",
            batch_id="course-integration-v2",
            experiment_id=experiment,
            public_source_sha256_by_stage={"dpo": teacher_sha} if label == "main" else None,
        )
        assert not approval["approved"] and not approval["reviewed"]
        approval.update(approved=True, reviewed=True)
        for file in approval["files"]:
            file["redistribution_approved"] = True
        exported = tmp_path / f"public-{label}"
        export_capstone(directory, exported, approval, {})
        prefix = f"course/course-integration-v2/{experiment}"
        for path in exported.rglob("*"):
            if path.is_file():
                remote[f"{prefix}/{path.relative_to(exported).as_posix()}"] = path
        manifests[label] = build_public_manifest(exported, revision=revision, prefix=prefix)
        approvals[label] = approval
    return manifests, approvals, remote, calls


def test_main_and_student_aliases_export_merge_and_download_without_collapsing(source, tmp_path, monkeypatch):
    from tiny_perceptron.capstone import DATA_VERSION

    manifests, approvals, remote, calls = student_release_fixture(source, tmp_path, monkeypatch)
    assert {model["id"] for model in manifests["student"]["models"]} == {"student-ce", "student-kd", "student-kd-int4"}
    assert approvals["student"]["model_card"]["training_data"][0]["source"] == f"{DATA_VERSION} synthetic data"
    calls.clear()
    combined = merge_public_manifests(manifests["main"], manifests["student"], revision="d" * 40)
    assert len(combined["models"]) == 9 and combined["revision"] == "d" * 40
    assert all(options == {"revision": "d" * 40, "token": False} for _, _, options in calls)
    assert set(name for _, name, _ in calls) == set(remote) - {
        name for name in remote if name.endswith("/export-manifest.json")
    }
    for identity in ("student-ce", "student-kd", "student-kd-int4"):
        folder = fetch_capstone(combined, identity, tmp_path / "downloads")
        payload = torch.load(folder / "model.pt", weights_only=True)
        assert payload["stage"] == "joint" and payload["config"]["experts"] == 0
        assert public_stage_id(payload) == identity
        assert payload["metadata"]["student_branch"] == ("student-ce" if identity == "student-ce" else "student-kd")
        assert payload["metadata"]["teacher_checkpoint_sha256"]
        assert payload["metadata"]["dataset_manifest_sha256"]
        assert (folder / "data.json").is_file()
    with pytest.raises(ValueError, match="collapse"):
        merge_public_manifests(manifests["main"], manifests["main"], revision="d" * 40)
    # A later HF commit cannot silently overwrite a main artifact.
    corrupted = tmp_path / "corrupted-model.pt"
    corrupted.write_bytes(b"changed at latest commit")
    original = next(model for model in manifests["main"]["models"] if model["id"] == "dpo")
    checkpoint_path = next(file["path"] for file in original["files"] if file["output"] == "model.pt")
    remote[checkpoint_path] = corrupted
    with pytest.raises(ValueError, match="changed bytes"):
        merge_public_manifests(manifests["main"], manifests["student"], revision="d" * 40)


@pytest.mark.parametrize("change", ["branch", "stage", "moe", "teacher", "dataset", "ce-int4"])
def test_student_alias_rejects_conflicting_or_incomplete_provenance(source, change):
    from tiny_perceptron.capstone import build_dataset, digest

    saved = torch.load(source[0], weights_only=True)
    saved["stage"] = "joint"
    saved["config"]["experts"] = 0
    saved["metadata"] = {
        "student_branch": "student-kd",
        "teacher_checkpoint_sha256": "a" * 64,
        "dataset_manifest_sha256": digest(build_dataset()[1]),
    }
    if change == "branch":
        saved["metadata"]["student_branch"] = "student-unknown"
    elif change == "stage":
        saved["stage"] = "dpo"
    elif change == "moe":
        saved["config"]["experts"] = 4
    elif change == "teacher":
        saved["metadata"]["teacher_checkpoint_sha256"] = "unverified"
    elif change == "dataset":
        saved["metadata"].pop("dataset_manifest_sha256")
    else:
        saved["metadata"]["student_branch"] = "student-ce"
        saved.update(format_version="capstone-ptq-v1", quantization={"bits": 4})
    with pytest.raises(ValueError):
        public_stage_id(saved)
