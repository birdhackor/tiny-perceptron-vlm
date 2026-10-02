"""不登入雲端的契約測試：成本上限、單項 GPU、公開清單與 checkpoint 清理。"""

import ast
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/modal_course.py"
TREE = ast.parse(SOURCE.read_text())


def helpers():
    """只執行 stdlib helper，不 import Modal、torch 或觸發遠端建置。"""
    names = {
        "ROOT",
        "BUDGET_USD",
        "GPU_RESERVATION_USD",
        "CPU_RESERVATION_USD",
        "REFERENCE_RATES",
        "asset_ignore",
        "json_value",
        "write_json",
        "sha256",
        "safe_name",
        "compute_reservation_guard",
        "cpu_reservation_guard",
        "reserve_budget",
        "private_only_paths",
        "approved_files",
        "inference_payload",
    }
    nodes = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    namespace = {
        "__file__": str(SOURCE),
        "Path": Path,
        "Decimal": Decimal,
        "datetime": datetime,
        "UTC": UTC,
        "json": json,
        "os": os,
        "re": re,
        "hashlib": hashlib,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def snapshot():
    # 首次 preflight 的真實 API 欄位與每小時單價。
    return {"rates": {"gpu_hour_cost_l4": "0.80000", "cpu_hour_cost": "0.04730", "mem_gib_hour_cost": "0.00800"}}


def test_failed_attempts_and_other_batches_keep_the_same_budget():
    helper = helpers()
    ledger = {}
    for number in range(45):
        helper["reserve_budget"](ledger, f"run-{number}", f"batch-{number}", "text", "run", snapshot())
        ledger["reservations"][-1]["status"] = "failed"
    assert Decimal(ledger["reserved_total_usd"]) == Decimal("9.90")
    with pytest.raises(RuntimeError, match="超過"):
        helper["reserve_budget"](ledger, "next", "new-batch", "text", "run", snapshot())
    assert len(ledger["reservations"]) == 45
    with pytest.raises(ValueError, match="已保留"):
        helper["reserve_budget"](ledger, "run-0", "same", "text", "run", snapshot())


def test_live_rates_must_fit_reservation_and_known_units():
    helper = helpers()
    guard = helper["compute_reservation_guard"](snapshot()["rates"])
    expected = Decimal("0.95860") * Decimal("602") / Decimal("3600") + Decimal("0.04")
    assert abs(Decimal(guard["conservative_compute_usd"]) - expected) < Decimal("1e-24")
    assert guard["seconds_per_hour"] == 3600
    assert Decimal(guard["conservative_compute_usd"]) < Decimal("0.22")
    costly = snapshot()["rates"] | {"gpu_hour_cost_l4": "2.00"}
    with pytest.raises(RuntimeError, match="未啟動"):
        helper["compute_reservation_guard"](costly)
    with pytest.raises(ValueError, match="單位"):
        helper["compute_reservation_guard"]({"unknown": "0.001"})
    with pytest.raises(ValueError, match="單位"):
        helper["compute_reservation_guard"]({"gpu_L4": "0.000222", "cpu": "0.0000131", "memory": "0.00000222"})


@pytest.mark.parametrize("mode", ["preflight", "run-cpu", "release"])
def test_cpu_execution_and_release_share_the_budget_without_gpu(mode):
    helper = helpers()
    ledger = {}
    reservation = helper["reserve_budget"](ledger, "cpu-run", "course-v1", "simple_models", mode, snapshot())
    assert reservation["this_job_usd"] == "0.04"
    assert ledger["reservations"][0]["compute_guard"]["gpu_used"] is False
    assert Decimal(ledger["reservations"][0]["compute_guard"]["compute_upper_bound_usd"]) < Decimal("0.04")
    with pytest.raises(RuntimeError, match="未啟動"):
        helper["cpu_reservation_guard"](snapshot()["rates"] | {"cpu_hour_cost": "100"}, mode)


def test_public_exports_require_matching_revision_license_and_hash(tmp_path):
    helper = helpers()
    file = tmp_path / "model.pt"
    file.write_bytes(b"model")
    result = {"revision": "commit", "results": {}}
    item = {
        "path": "model.pt",
        "sha256": helper["sha256"](file),
        "license": "mit",
        "redistribution_approved": True,
        "kind": "checkpoint",
    }
    approval = {"approved": True, "revision": "commit", "files": [item]}
    assert helper["approved_files"](tmp_path, approval, result) == [item]
    for changed in (approval | {"approved": False}, approval | {"revision": "other"}, approval | {"files": []}):
        with pytest.raises(ValueError):
            helper["approved_files"](tmp_path, changed, result)
    with pytest.raises(ValueError, match="授權"):
        helper["approved_files"](tmp_path, approval | {"files": [item | {"redistribution_approved": False}]}, result)
    restricted = result | {"results": {"safety": {"private_only_artifacts": ["model.pt"]}}}
    with pytest.raises(ValueError, match="private_only"):
        helper["approved_files"](tmp_path, approval, restricted)
    data_dir = tmp_path / "pku-excerpts"
    data_dir.mkdir()
    private_data = data_dir / "train.jsonl"
    private_data.write_text(' {"text": "private"}\n')
    data_item = {
        **item,
        "path": "pku-excerpts/train.jsonl",
        "kind": "dataset",
        "sha256": helper["sha256"](private_data),
    }
    restricted_data = result | {"results": {"safety": {"private_only_data": ["pku-excerpts"]}}}
    with pytest.raises(ValueError, match="private_only_data"):
        helper["approved_files"](tmp_path, approval | {"files": [data_item]}, restricted_data)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="已變更"):
        helper["approved_files"](tmp_path, approval, result)


def test_inference_export_removes_optimizer_rng_teacher_and_unknown_content():
    clean = helpers()["inference_payload"](
        {
            "format_version": 1,
            "config": {"width": 8},
            "model": {"weight": "tensor"},
            "optimizer": {"state": "secret"},
            "torch_rng": "rng",
            "python_rng": "rng",
            "training_state": {"teacher": "private"},
            "metadata": {"dataset": "private"},
            "unknown_dataset": "private",
        },
        {"revision": "commit"},
    )
    assert set(clean) == {"format_version", "config", "model", "metadata"}
    assert clean["metadata"] == {"revision": "commit"}


def test_only_one_bounded_gpu_function_and_manual_serial_workflow():
    functions = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}
    gpu_functions = []
    for name, node in functions.items():
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                options = {keyword.arg: keyword.value for keyword in decorator.keywords}
                if "gpu" in options:
                    gpu_functions.append(name)
                    assert ast.literal_eval(options["gpu"]) == "L4"
                    assert ast.literal_eval(options["cpu"]) == (2, 2)
                    assert ast.literal_eval(options["memory"]) == (8192, 8192)
                    assert ast.literal_eval(options["timeout"]) == 600
                    assert ast.literal_eval(options["max_containers"]) == 1
                    assert ast.literal_eval(options["retries"]) == 0
                    assert "secrets" not in options
    assert gpu_functions == ["train"]
    for name in ("train_cpu", "stage_approval", "release"):
        options = {keyword.arg: keyword.value for keyword in functions[name].decorator_list[0].keywords}
        assert "gpu" not in options and ast.literal_eval(options["timeout"]) == 600
    workflow = (ROOT / ".github/workflows/course-experiments.yml").read_text()
    assert "workflow_dispatch:" in workflow and "\n  push:" not in workflow and "\n  pull_request:" not in workflow
    assert "cancel-in-progress: false" in workflow and "modal==1.6.0" in workflow
    assert "vars.MODAL_TOKEN_ID || secrets.MODAL_TOKEN_ID" in workflow
    assert "--list-assets" in workflow and 'git lfs pull --include="$include"' in workflow
    assert "/app/docs/course-experiments/plan.json" in SOURCE.read_text()
    assert "options: [run, run-cpu, preflight, release]" in workflow
    assert "git', 'show'" in workflow and "--approval-file" in workflow
    assert "public-manifest.json" in SOURCE.read_text()


def test_archive_mount_filter_reads_relative_paths_and_rejects_lfs_pointers(tmp_path, monkeypatch):
    helper = helpers()
    helper["asset_ignore"].__globals__["ROOT"] = tmp_path
    assets = tmp_path / "assets/training"
    assets.mkdir(parents=True)
    archive = assets / "toy-v1.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setenv("COURSE_ASSET_PATHS", "assets/training/toy-v1.tar.gz")
    assert helper["asset_ignore"](Path("toy-v1.tar.gz")) is False
    assert helper["asset_ignore"](Path("other-v1.tar.gz")) is True
    assert helper["asset_ignore"](Path("manifest.json")) is False
    archive.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        helper["asset_ignore"](Path("toy-v1.tar.gz"))
