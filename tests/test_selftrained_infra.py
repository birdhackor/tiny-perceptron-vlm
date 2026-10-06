"""Exercise budget, exact CLI and transport behavior without cloud mutations."""

import importlib.util
import io
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tarfile
import tomllib
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / "scripts/selftrained" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"selftrained_test_{name}", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("modal_runner")
transport = module("hf_transport")
RATES = {
    "cpu_hour_cost": "0.04730",
    "mem_gib_hour_cost": "0.00800",
    "gpu_hour_cost_l4": "0.80000",
    "egress_gib_cost": "0.00",
}


def test_remote_basename_helper_import_requires_both_image_search_paths(tmp_path, monkeypatch):
    # Modal cloudpickle stores these global helpers as ordinary module refs;
    # stdlib pickle exercises the same import behavior without cloud access.
    source = ROOT / "scripts/selftrained/modal_runner.py"
    spec = importlib.util.spec_from_file_location("modal_runner", source)
    imported = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "modal_runner", imported)
    spec.loader.exec_module(imported)
    payload = tmp_path / "helper.pickle"
    payload.write_bytes(pickle.dumps(imported.ledger_total, protocol=4))
    image_root = tmp_path / "repo"
    scripts = image_root / "scripts/selftrained"
    scripts.mkdir(parents=True)
    shutil.copyfile(source, scripts / "modal_runner.py")
    code = "import pickle,sys; f=pickle.load(open(sys.argv[1],'rb')); print(f({'reserved_total_usd':'36.20','reservations':[{'reserved_usd':'36.20'}]}))"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(image_root)}
    failed = subprocess.run(
        [sys.executable, "-c", code, str(payload)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert failed.returncode != 0 and "No module named 'modal_runner'" in failed.stderr
    env["PYTHONPATH"] = str(image_root) + os.pathsep + str(scripts)
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(payload)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "36.20"


def test_cuda_recipe_uses_locked_selftrained_extra_without_pip_in_uv_environment(monkeypatch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["selftrained"] == ["safetensors==0.8.0"]
    recipes = []

    class Image:
        def __init__(self):
            self.locked = False
            recipes.append(self)

        def pip_install(self, *packages):
            if self.locked:
                raise RuntimeError("uv_sync environments have no pip module")
            return self

        def uv_sync(self, directory, **options):
            self.locked = True
            self.options = options
            assert Path(directory) == ROOT
            assert options["extras"] == ["cu126", "selftrained"]
            assert options["extra_options"] == "--no-dev"
            return self

        def __getattr__(self, name):
            assert name in ("env", "workdir", "add_local_dir")
            return lambda *args, **kwargs: self

    modal = SimpleNamespace(
        App=lambda name: SimpleNamespace(function=lambda **kwargs: lambda function: function),
        Image=SimpleNamespace(debian_slim=lambda **kwargs: Image()),
        Volume=SimpleNamespace(from_name=lambda *args, **kwargs: None),
        Secret=SimpleNamespace(from_name=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setenv("SELFTRAINED_MODAL_PHASE", "execute")
    runner.register_modal()
    assert len(recipes) == 2 and recipes[1].locked


def historical_ledger(total="36.20"):
    return {
        "budget_usd": "40.00",
        "reserved_total_usd": total,
        "reservations": [{"run_id": "history", "reserved_usd": total, "status": "failed"}],
    }


def reserve(ledger, run_id="attempt", job=None):
    return runner.reserve_entry(
        ledger,
        run_id,
        "new-batch",
        "a" * 40,
        "b" * 64,
        job or {"schema_version": 1, "stage": "pretrain"},
        "c" * 64,
        {"rates": RATES},
        {"train.py": "d" * 64},
    )


def test_shared_history_survives_failures_batches_and_retry():
    ledger = historical_ledger()
    original = deepcopy(ledger["reservations"][0])
    first = reserve(ledger)
    assert first["reserved_usd"] == "0.29"
    runner.finish_entry(ledger, "attempt", "failed")
    second = reserve(ledger, "new-attempt")
    assert Decimal(ledger["reserved_total_usd"]) == Decimal("36.78")
    assert ledger["reservations"][0] == original
    assert second["batch_id"] == "new-batch"
    with pytest.raises(ValueError, match="fresh run"):
        reserve(ledger)


def test_refuses_empty_stale_exhausted_and_bad_ledgers_without_mutation():
    for ledger in ({}, historical_ledger("9.83"), historical_ledger("39.99"), historical_ledger("NaN")):
        original = deepcopy(ledger)
        with pytest.raises((RuntimeError, ValueError)):
            reserve(ledger)
        assert ledger == original
    ledger = historical_ledger()
    ledger["reserved_total_usd"] = "10.00"
    reserve(ledger)
    assert ledger["reserved_total_usd"] == "36.49"


def test_completed_job_cannot_be_repeated_or_downgraded_by_failure_cleanup():
    ledger = historical_ledger()
    reserve(ledger)
    runner.finish_entry(ledger, "attempt", "completed")
    runner.finish_entry(ledger, "attempt", "failed-or-cancelled")
    assert ledger["reservations"][-1]["status"] == "completed"
    with pytest.raises(ValueError, match="already completed"):
        reserve(ledger, "repeat")
    with pytest.raises(ValueError, match="already completed"):
        reserve(
            ledger,
            "longer-timeout",
            job={"schema_version": 1, "stage": "pretrain", "wall_seconds": 600, "max_seconds": 420},
        )


def test_cost_guard_uses_live_units_and_reserves_auxiliary_cpu_memory_egress_storage():
    guard = runner.reservation_guard("pretrain", RATES)
    assert Decimal(guard["bounded_compute_usd"]) == Decimal("902") * Decimal("0.95860") / Decimal("3600")
    assert guard["bounded_egress_gib"] == "0.25"
    assert Decimal(guard["auxiliary_compute_usd"]) > 0
    assert Decimal(guard["build_and_storage_allowance_usd"]) == Decimal("0.04")
    assert runner.reservation_guard("prepare", RATES)["reserved_usd"] == "0.06"
    assert runner.reservation_guard("release", RATES)["reserved_usd"] == "0.06"
    for rates in (
        RATES | {"gpu_hour_cost_l4": "10"},
        RATES | {"mem_gib_hour_cost": "NaN"},
        {"gpu_L4_per_second": "0.000222"},
    ):
        with pytest.raises((RuntimeError, ValueError)):
            runner.reservation_guard("pretrain", rates)


def test_selected_wall_timeout_and_reservation_share_exact_profile():
    live = RATES | {"egress_gib_cost": "0.04000"}
    short = runner.reservation_guard("joint", live, 600)
    long = runner.reservation_guard("freeze", live, 900)
    assert short["resource_spec"]["seconds"] == 600 and short["reserved_usd"] == "0.22"
    assert long["resource_spec"]["seconds"] == 900 and long["reserved_usd"] == "0.30"
    ledger = historical_ledger()
    entry = reserve(ledger, job={"schema_version": 1, "stage": "pretrain", "wall_seconds": 600, "max_seconds": 420})
    assert entry["compute_guard"]["resource_spec"]["seconds"] == 600
    with pytest.raises(ValueError, match="180 seconds"):
        runner.validate_job({"schema_version": 1, "stage": "pretrain", "wall_seconds": 600, "max_seconds": 421})
    with pytest.raises(ValueError, match="600 or 900"):
        runner.validate_job({"schema_version": 1, "stage": "pretrain", "wall_seconds": 1200})


def test_live_ledger_probe_is_read_only_client_api_and_preserves_raw_history(monkeypatch):
    raw = json.dumps(historical_ledger("37.12")).encode()
    operations = []

    class Volume:
        @staticmethod
        def from_name(name, create_if_missing):
            assert name == "tiny-perceptron-course" and create_if_missing is False
            operations.append("volume lookup")
            return SimpleNamespace(read_file=lambda path: iter([raw[:20], raw[20:]]))

    modal = SimpleNamespace(
        Volume=Volume,
        Workspace=SimpleNamespace(from_context=lambda: SimpleNamespace(billing=SimpleNamespace(rates=lambda: RATES))),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    evidence = runner.read_live_ledger()
    assert evidence["remaining_reserved_budget_usd"] == "2.88"
    assert evidence["ledger"]["reservations"][0]["status"] == "failed"
    assert evidence["remote_container_started"] is False
    assert evidence["ledger_written"] is False
    assert operations == ["volume lookup"]


def descriptor(stage="joint", path="best.pt"):
    return {"stage": stage, "run_id": "prior-attempt", "path": path, "sha256": "e" * 64}


def test_exact_train_and_evaluate_clis_are_parsed_by_actual_scripts(tmp_path, monkeypatch):
    train, evaluate = module("train"), module("evaluate")
    monkeypatch.setattr(runner, "artifact_path", lambda batch, item: tmp_path / item["path"])
    manifest = {"records": [{"path": "records/text.jsonl"}, {"path": "records/audio.jsonl"}], "model_config": {}}
    job = {
        "schema_version": 1,
        "stage": "joint",
        "architecture": "dense",
        "steps": 25,
        "init_checkpoint": descriptor("audio"),
    }
    command = runner.trainer_command(job, manifest, tmp_path, tmp_path / "train", "batch")
    parsed = train.parser().parse_args(command[2:])
    assert parsed.architecture == "dense" and parsed.stage == "joint" and parsed.steps == 25
    assert parsed.device == "cuda" and parsed.threads == 2
    assert len(parsed.records) == 2
    assert parsed.init_checkpoint == str(tmp_path / "best.pt")
    freeze = {"schema_version": 1, "stage": "freeze", "checkpoint": descriptor()}
    parsed = evaluate.parser().parse_args(
        runner.trainer_command(freeze, manifest, tmp_path, tmp_path / "val", "batch")[2:]
    )
    assert parsed.split == "validation" and parsed.freeze_protocol.endswith("frozen.json")
    test = {
        "schema_version": 1,
        "stage": "test",
        "checkpoint": descriptor(),
        "protocol": descriptor("freeze", "frozen.json"),
    }
    parsed = evaluate.parser().parse_args(
        runner.trainer_command(test, manifest, tmp_path, tmp_path / "test", "batch")[2:]
    )
    assert parsed.split == "test" and parsed.protocol.endswith("frozen.json") and parsed.limit is None


@pytest.mark.parametrize(
    "job",
    [
        {"stage": "pretrain", "max_seconds": 721},
        {"stage": "pretrain", "learning_rate": "NaN"},
        {"stage": "sft"},
        {"stage": "test", "checkpoint": descriptor(), "protocol": descriptor("freeze", "frozen.json"), "limit": 1},
        {"stage": "freeze", "checkpoint": descriptor(), "limit": 1},
        {"stage": "pretrain", "resume": descriptor(), "init_checkpoint": descriptor()},
    ],
)
def test_unbounded_or_unfrozen_jobs_fail_before_cloud_call(job):
    with pytest.raises(ValueError):
        runner.validate_job({"schema_version": 1, **job})


def test_protocol_gate_rejects_changed_checkpoint_generation_code_and_started_test(tmp_path):
    path = tmp_path / "frozen.json"
    manifest = {"records": [{"path": "records/text.jsonl", "sha256": "b" * 64}]}
    job = {"checkpoint": descriptor(), "architecture": "moe", "max_new_tokens": 64}
    code = {"scripts/selftrained/evaluate.py": "c" * 64}
    protocol = {
        "version": "selftrained-generation-v1",
        "test_once": True,
        "checkpoint_sha256": job["checkpoint"]["sha256"],
        "architecture": "moe",
        "max_new_tokens": 64,
        "data_sha256": {"text.jsonl": "b" * 64},
        "controls": "all",
        "code_sha256": code,
    }
    path.write_text(json.dumps(protocol))
    assert runner.test_protocol_gate(path, job, manifest, code) == protocol
    for changed in (job | {"max_new_tokens": 63}, job | {"checkpoint": descriptor() | {"sha256": "f" * 64}}):
        with pytest.raises(ValueError, match="frozen"):
            runner.test_protocol_gate(path, changed, manifest, code)
    with pytest.raises(ValueError, match="frozen"):
        runner.test_protocol_gate(path, job, manifest, {"scripts/selftrained/evaluate.py": "d" * 64})
    path.with_suffix(".json.test-started.json").write_text("{}")
    with pytest.raises(ValueError, match="already started"):
        runner.test_protocol_gate(path, job, manifest, code)


def test_prior_artifact_binds_dataset_manifest_and_dense_moe_architecture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXPERIMENTS", tmp_path)
    directory = tmp_path / "batch/joint/prior-attempt"
    directory.mkdir(parents=True)
    weight = directory / "best.pt"
    weight.write_bytes(b"trusted internal checkpoint")
    metadata = {"manifest_sha256": "b" * 64, "job": {"architecture": "dense"}, "revision": "a" * 40}
    (directory / "execution.json").write_text(json.dumps(metadata))
    item = descriptor() | {"sha256": runner.sha256(weight)}
    path, execution = runner.artifact_gate("batch", item, "b" * 64, "dense")
    assert path == weight and execution == metadata
    with pytest.raises(ValueError, match="different frozen"):
        runner.artifact_gate("batch", item, "c" * 64, "dense")
    with pytest.raises(ValueError, match="cannot share"):
        runner.artifact_gate("batch", item, "b" * 64, "moe")


def test_reserve_checks_prepared_identity_and_records_while_gpu_checks_all_assets(tmp_path, monkeypatch):
    from scripts.selftrained import hf_transport

    monkeypatch.setattr(runner, "EXPERIMENTS", tmp_path)
    manifest_sha = "b" * 64
    directory = runner.data_root(manifest_sha)
    directory.mkdir(parents=True)
    records = []
    for number in range(12):
        path = directory / f"records-{number}.jsonl"
        path.write_bytes(b'{"split":"train"}\n')
        records.append({"path": path.name, "bytes": path.stat().st_size, "sha256": runner.sha256(path)})
    asset = directory / "image.png"
    asset.write_bytes(b"verified prepared asset")
    assets = [{"path": asset.name, "bytes": asset.stat().st_size, "sha256": runner.sha256(asset)}]
    # These absent assets must never be touched by the small control container.
    assets += [{"path": f"other-{i}.png", "bytes": 1, "sha256": "c" * 64} for i in range(6257)]
    manifest = {"package": {"sha256": "d" * 64}, "records": records, "assets": assets}
    receipt = {
        "status": "completed",
        "manifest_sha256": manifest_sha,
        "package": manifest["package"],
        "records": records,
        "asset_count": len(assets),
    }
    receipt_path = directory / "prepare-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    calls = []
    original_verify = hf_transport.verify_file

    def verify(root, item):
        calls.append(item["path"])
        return original_verify(root, item)

    monkeypatch.setattr(hf_transport, "verify_file", verify)
    assert runner.prepared_records_gate(manifest, manifest_sha) == directory
    assert calls == [item["path"] for item in records]
    for changed in (
        receipt | {"manifest_sha256": "e" * 64},
        receipt | {"package": {"sha256": "e" * 64}},
        receipt | {"records": records[:-1]},
        receipt | {"asset_count": 1},
        receipt | {"status": "failed"},
    ):
        receipt_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="identity"):
            runner.prepared_records_gate(manifest, manifest_sha)
    receipt_path.write_text(json.dumps(receipt))
    asset.write_bytes(b"mutated prepared asset")
    calls.clear()
    assert runner.prepared_records_gate(manifest, manifest_sha) == directory
    assert calls == [item["path"] for item in records]
    with pytest.raises(ValueError, match="Size/hash mismatch: image.png"):
        runner.readiness(manifest, manifest_sha)
    (directory / records[0]["path"]).write_bytes(b"mutated record")
    with pytest.raises(ValueError, match="Size/hash mismatch"):
        runner.prepared_records_gate(manifest, manifest_sha)


def test_partial_evaluation_resume_pins_raw_prefix_receipt_and_freeze_protocol(tmp_path):
    output = tmp_path / "outputs.jsonl"
    prefix = b'{"record":{"id":"frozen-case-1"},"trace":{"final_output":"answer"}}\n'
    output.write_bytes(prefix + b"fragment without complete newline")
    receipt = {
        "schema": "selftrained-evaluation-journal-v1",
        "split": "test",
        "checkpoint_sha256": descriptor()["sha256"],
        "protocol_sha256": "b" * 64,
        "conditions_sha256": "c" * 64,
        "completed_count": 1,
        "expected_count": 2,
        "output_byte_count": len(prefix),
        "outputs_sha256": runner.hashlib.sha256(prefix).hexdigest(),
        "completed_ids_sha256": runner.hashlib.sha256(b"frozen-case-1\n").hexdigest(),
        "status": "interrupted",
    }
    receipt_path = output.with_name("evaluation-receipt.json")
    receipt_path.write_text(json.dumps(receipt))
    job = {
        "stage": "test",
        "checkpoint": descriptor(),
        "resume_evaluation": descriptor("test", "outputs.jsonl") | {"receipt_sha256": runner.sha256(receipt_path)},
    }
    assert runner.evaluation_resume_gate(output, job, "b" * 64) == receipt
    with pytest.raises(ValueError, match="same frozen"):
        runner.evaluation_resume_gate(output, job, "d" * 64)
    changed = receipt | {"completed_count": 2}
    receipt_path.write_text(json.dumps(changed))
    job["resume_evaluation"]["receipt_sha256"] = runner.sha256(receipt_path)
    with pytest.raises(ValueError, match="unfinished"):
        runner.evaluation_resume_gate(output, job, "b" * 64)


def archive(tmp_path, members):
    file = tmp_path / "package.tar.gz"
    with tarfile.open(file, "w:gz") as out:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.type = kind
            if kind == tarfile.SYMTYPE:
                info.linkname = "/tmp/outside"
            out.addfile(info, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return file, {
        "bytes": file.stat().st_size,
        "sha256": transport.digest(file),
        "unpacked_bytes": sum(len(content) for _, content, _ in members),
    }


def test_data_archive_checks_bytes_hash_and_member_paths(tmp_path):
    file, package = archive(tmp_path, [("records/train.jsonl", b'{"split":"train"}\n', tarfile.REGTYPE)])
    output = transport.unpack_verified_archive(file, tmp_path / "verified", package)
    assert (output / "records/train.jsonl").read_bytes() == b'{"split":"train"}\n'
    with pytest.raises(ValueError, match="pinned bytes"):
        transport.unpack_verified_archive(file, tmp_path / "bad", package | {"sha256": "f" * 64})
    for members in (
        [("records/../../outside", b"x", tarfile.REGTYPE)],
        [("link", b"", tarfile.SYMTYPE)],
        [("dup", b"x", tarfile.REGTYPE), ("dup", b"x", tarfile.REGTYPE)],
    ):
        file, package = archive(tmp_path, members)
        with pytest.raises(ValueError):
            transport.unpack_verified_archive(file, tmp_path / "bad", package)


def test_public_release_rejects_resume_checkpoint_and_unreviewed_bytes(tmp_path):
    source = tmp_path / "latest.pt"
    source.write_bytes(b"private optimizer state")
    item = {
        "path": source.name,
        "bytes": source.stat().st_size,
        "sha256": transport.digest(source),
        "kind": "inference",
        "license": "mit",
        "redistribution_approved": True,
    }
    release = {
        "repo_id": transport.PUBLIC_MODEL_REPO,
        "private": False,
        "revision": "a" * 40,
        "manifest_sha256": "b" * 64,
        "prefix": "selftrained/v1",
        "files": [item],
    }
    with pytest.raises(ValueError, match="Resume checkpoints"):
        transport.approved_public_files(tmp_path, release, "a" * 40, "b" * 64)
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"inference tensors")
    item.update(path=source.name, bytes=source.stat().st_size, sha256=transport.digest(source))
    assert transport.approved_public_files(tmp_path, release, "a" * 40, "b" * 64) == [item]
    item["redistribution_approved"] = False
    with pytest.raises(ValueError, match="redistribution"):
        transport.approved_public_files(tmp_path, release, "a" * 40, "b" * 64)


def batch_exports(tmp_path):
    manifest_sha = "b" * 64
    manifest = {
        "records": [{"path": "records.jsonl", "sha256": "c" * 64}],
        "assets": [{"path": "images/image.png", "sha256": "d" * 64}],
    }
    release = {
        "repo_id": transport.PUBLIC_MODEL_REPO,
        "private": False,
        "prefix": "selftrained/v1",
        "manifest_sha256": manifest_sha,
        "exports": [],
    }
    sources = {}
    for name, (architecture, stage) in transport.EXPORT_STAGES.items():
        source = tmp_path / "batch" / stage / name
        source.mkdir(parents=True)
        (source / "best.pt").write_bytes(b"private selected checkpoint " + name.encode())
        config = {"architecture": architecture, "max_length": 512, "vocab_size": 2}
        (source / "model-config.json").write_text(json.dumps(config))
        (source / "tokenizer.json").write_text('{"characters":["a"]}')
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
        execution = {
            "status": "completed",
            "revision": "a" * 40,
            "manifest_sha256": manifest_sha,
            "stage": stage,
            "run_id": name,
            "job": {"architecture": architecture, "steps": 600},
        }
        (source / "execution.json").write_text(json.dumps(execution))
        receipt = {
            "architecture": architecture,
            "stage": stage,
            "data_sha256": {"records.jsonl": "c" * 64},
            "asset_sha256": {"images/image.png": "d" * 64},
            "completed_requested_steps": True,
            "selected_checkpoint_available": True,
            "inference_exported": True,
            "test_used_for_selection": False,
            "origin": {"kind": "all-neural-weights-random"},
            "steps": 600,
            "config": config,
            "selection": "validation_loss (teacher-forced; not generation success)",
        }
        (source / "train-receipt.json").write_text(json.dumps(receipt))
        inference = {
            "selected_checkpoint_sha256": transport.digest(source / "best.pt"),
            "files": {
                filename: transport.digest(source / filename)
                for filename in ("model.safetensors", "model-config.json", "tokenizer.json")
            },
            "stage": stage,
            "selection": "validation_loss",
            "origin": receipt["origin"],
        }
        (source / "inference-manifest.json").write_text(json.dumps(inference))
        export = {
            "name": name,
            "architecture": architecture,
            "revision": execution["revision"],
            "source": {
                "stage": stage,
                "run_id": name,
                "path": "best.pt",
                "sha256": transport.digest(source / "best.pt"),
            },
            "execution_sha256": transport.digest(source / "execution.json"),
            "train_receipt_sha256": transport.digest(source / "train-receipt.json"),
            "files": [
                {
                    "path": filename,
                    "bytes": (source / filename).stat().st_size,
                    "sha256": transport.digest(source / filename),
                    "kind": "inference" if filename.endswith(".safetensors") else "metadata",
                    "license": "MIT",
                    "redistribution_approved": True,
                }
                for filename in sorted(transport.SAFE_EXPORT_FILES)
            ],
        }
        release["exports"].append(export)
        sources[name] = {"directory": source, "execution": execution}
    return sources, release, manifest_sha, manifest


def test_batch_release_binds_all_four_actual_sources_before_cloud_access(tmp_path, monkeypatch):
    sources, release, manifest_sha, manifest = batch_exports(tmp_path)
    approved = transport.approved_batch_exports(sources, release, manifest_sha, manifest)
    assert len(approved) == 4 and sum(len(item["files"]) for item in approved) == 16
    monkeypatch.setattr(runner, "EXPERIMENTS", tmp_path)
    assert runner.release_sources_gate("batch", release, manifest_sha, manifest).keys() == sources.keys()
    runner.validate_job({"schema_version": 1, "stage": "release", "release": release})
    changed = deepcopy(release)
    changed["exports"][0]["revision"] = "e" * 40
    with pytest.raises(ValueError, match="immutable execution"):
        transport.approved_batch_exports(sources, changed, manifest_sha, manifest)
    changed = deepcopy(release)
    changed["exports"][0]["files"][0]["path"] = "latest.pt"
    with pytest.raises(ValueError, match="four reviewed safe"):
        transport.approved_batch_exports(sources, changed, manifest_sha, manifest)
    with pytest.raises(ValueError, match="frozen manifest"):
        transport.approved_batch_exports(sources, release, "e" * 64, manifest)
    total = sum(file["bytes"] for item in approved for file in item["files"])
    monkeypatch.setattr(transport, "MAX_RELEASE_BYTES", total - 1)
    with pytest.raises(ValueError, match="complete four-export batch"):
        transport.approved_batch_exports(sources, release, manifest_sha, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_requested_steps", False),
        ("test_used_for_selection", True),
        ("data_sha256", {"records.jsonl": "e" * 64}),
    ],
)
def test_batch_release_rejects_completed_provenance_mutations_even_with_new_review_hash(tmp_path, field, value):
    sources, release, manifest_sha, manifest = batch_exports(tmp_path)
    export = release["exports"][0]
    path = sources[export["name"]]["directory"] / "train-receipt.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    export["train_receipt_sha256"] = transport.digest(path)
    with pytest.raises(ValueError, match="completed from-random"):
        transport.approved_batch_exports(sources, release, manifest_sha, manifest)


def test_batch_publish_is_one_parent_bound_commit_and_can_verify_existing_bytes(tmp_path, monkeypatch):
    import huggingface_hub

    sources, release, manifest_sha, manifest = batch_exports(tmp_path)
    commits, downloads = [], []
    existing = []
    public_files = {}
    for export in release["exports"]:
        for file in export["files"]:
            public_path = f"{release['prefix']}/{export['name']}/{file['path']}"
            local = tmp_path / "public" / export["name"] / file["path"]
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[export["name"]]["directory"] / file["path"], local)
            public_files[public_path] = local

    class API:
        def __init__(self, token):
            assert token == "offline-fixture"

        def repo_info(self, repo_id, repo_type):
            assert repo_id == transport.PUBLIC_MODEL_REPO and repo_type == "model"
            return SimpleNamespace(private=False, sha="e" * 40)

        def list_repo_tree(self, *args, **kwargs):
            assert kwargs["revision"] == "e" * 40 and kwargs["recursive"]
            return iter(existing)

        def create_commit(self, **kwargs):
            assert kwargs["parent_commit"] == "e" * 40
            commits.append(kwargs)
            return SimpleNamespace(oid="f" * 40, commit_url="https://huggingface.co/offline/commit/" + "f" * 40)

    def download(repo_id, path, **kwargs):
        assert repo_id == transport.PUBLIC_MODEL_REPO and kwargs["revision"] == "e" * 40
        downloads.append(path)
        return str(public_files[path])

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    receipt = transport.publish_batch_inference(
        sources, release, manifest_sha, manifest, "offline-fixture", tmp_path / "stage"
    )
    assert receipt["new_commit_created"] and receipt["commit_sha"] == "f" * 40
    assert len(commits) == 1 and len(commits[0]["operations"]) == 16
    assert all(not operation.path_in_repo.endswith(".pt") for operation in commits[0]["operations"])
    for export in release["exports"]:
        for file in export["files"]:
            existing.append(
                SimpleNamespace(
                    path=f"{release['prefix']}/{export['name']}/{file['path']}",
                    size=file["bytes"],
                    lfs={"sha256": file["sha256"]} if file["path"].endswith(".safetensors") else None,
                )
            )
    recovered = transport.publish_batch_inference(
        sources, release, manifest_sha, manifest, "offline-fixture", tmp_path / "recover"
    )
    assert recovered["recovered_existing_revision"] and not recovered["new_commit_created"]
    assert recovered["commit_sha"] == "e" * 40 and len(commits) == 1 and len(downloads) == 12
    metadata = next(path for path in public_files if path.endswith("model-config.json"))
    public_files[metadata].write_bytes(b"changed published bytes")
    with pytest.raises(ValueError, match="reviewed safe bytes"):
        transport.publish_batch_inference(
            sources, release, manifest_sha, manifest, "offline-fixture", tmp_path / "tampered"
        )
    assert len(commits) == 1
