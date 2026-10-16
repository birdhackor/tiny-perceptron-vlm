"""獨立發布器離線audit；合法synthetic包，真程式控制流，假雲端/HF。"""

import copy
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import publication, publish


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = datetime.now(UTC).isoformat()
    checked_names = ["scripts/lesson_products/publish.py", "scripts/lesson_products/publication.py", ".github/workflows/lesson-products-publish.yml", "tests/test_lesson_products_publication.py", "scripts/lesson_products/modal_runner.py", "scripts/lesson_products/checkpoints.py", "scripts/selftrained/modal_runner.py", "scripts/selftrained/finance.py", publish.POLICY_PATH, "docs/lesson-products/data-v2-manifest.json"]
    before = {name: sha(ROOT / name) for name in checked_names}
    results = []
    with tempfile.TemporaryDirectory(dir=ROOT / "work", prefix="publication-audit-") as folder:
        work = Path(folder)
        count = 0

        def fixture():
            nonlocal count
            count += 1
            case = work / str(count)
            directory = case / "release"
            directory.mkdir(parents=True)
            (directory / "data-manifest.json").write_bytes((ROOT / "docs/lesson-products/data-v2-manifest.json").read_bytes())
            data_sha = sha(directory / "data-manifest.json")
            frozen_products = {}
            for product in ("b", "c"):
                path = directory / product
                path.mkdir()
                names = ["model.pt", "config.json", "tokenizer.json", "training-receipt.json", "auto-policy.json" if product == "b" else "calibration.json"]
                for name in names:
                    (path / name).write_bytes(("synthetic " + product + " " + name + "; not actual model").encode())
                frozen_products[product] = {name: sha(path / name) for name in names}
            freeze = {"version": 1, "recorded_at": "2026-10-16T00:00:00+00:00", "data_manifest_sha256": data_sha, "products": frozen_products, "source_sha256s": {}, "protocol": {}, "test_outcomes_observed": False, "calibration": {}, "auto_search": [], "mechanisms": {}}
            (directory / "settings-freeze.json").write_text(json.dumps(freeze))
            report = {"settings_freeze_sha256": sha(directory / "settings-freeze.json"), "data_manifest_sha256": data_sha,
                      "gates": {"b_primary": True, "c_choice": True, "c_reorder": True, "b_secondary_target": False}, "scope": "Synthetic finite fixture; secondary failure is explicit, no actual training/test claim."}
            (directory / "report.json").write_text(json.dumps(report))
            for product in ("b", "c"):
                manifest = {"version": 1, "product": product, "settings_freeze_sha256": sha(directory / "settings-freeze.json"), "data_manifest_sha256": data_sha,
                            "files": {name: {"sha256": sha(directory / product / name), "bytes": (directory / product / name).stat().st_size} for name in frozen_products[product]}}
                (directory / product / "artifact-manifest.json").write_text(json.dumps(manifest))
            spec = {"version": 1, "stage": "release", "job_kind": "lesson_products_release", "wall_seconds": 600,
                    "gross_quota_policy": {"path": publish.POLICY_PATH, "sha256": publish.POLICY_SHA}, "asset_id": 123,
                    "transport_sha256": "0" * 64, "data_manifest_sha256": data_sha, "prefix": "lesson-products/synthetic-release", "files": {}}
            archive = case / "transport.tar.gz"

            def refresh():
                spec["files"] = {p.relative_to(directory).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(directory.rglob("*")) if p.is_file()}
                with tarfile.open(archive, "w:gz") as tar:
                    for name in spec["files"]:
                        tar.add(directory / name, arcname=name, recursive=False)
                spec["transport_sha256"] = sha(archive)

            refresh()
            return case, directory, archive, spec, refresh

        case, directory, archive, spec, refresh = fixture()
        positive = publication.verify_release(directory, spec)
        extracted = publication.unpack_verified(archive, spec["transport_sha256"], case / "extracted")
        assert publication.verify_release(extracted, spec) == positive
        results.append({"id": "complete_package_secondary_explicit_fail_positive", "actual": "accepted", "passed": True, "file_count": positive["files"], "gates": positive["gates"]})
        for fault in ("necessary_gate", "extra_file", "data_identity"):
            case, directory, archive, spec, refresh = fixture()
            publication.verify_release(directory, spec)
            if fault == "necessary_gate":
                report = json.loads((directory / "report.json").read_bytes())
                report["gates"]["c_choice"] = False
                (directory / "report.json").write_text(json.dumps(report))
                refresh()
            elif fault == "extra_file":
                (directory / "unlisted.txt").write_text("not in spec")
            else:
                spec["data_manifest_sha256"] = "0" * 64
            caught = None
            try:
                publication.verify_release(directory, spec)
            except ValueError as error:
                caught = str(error)
            assert caught
            results.append({"id": "package_rejects_" + fault, "base_passed": True, "actual": "rejected", "passed": True, "error": caught})

        case, directory, archive, spec, refresh = fixture()
        publication.verify_release(directory, spec)
        freeze = json.loads((directory / "settings-freeze.json").read_bytes())
        freeze["products"]["b"].pop("model.pt")
        (directory / "b/model.pt").write_bytes(b"different untested synthetic weights")
        (directory / "settings-freeze.json").write_text(json.dumps(freeze))
        new_freeze = sha(directory / "settings-freeze.json")
        report = json.loads((directory / "report.json").read_bytes())
        report["settings_freeze_sha256"] = new_freeze
        (directory / "report.json").write_text(json.dumps(report))
        for product in ("b", "c"):
            path = directory / product / "artifact-manifest.json"
            value = json.loads(path.read_bytes())
            value["settings_freeze_sha256"] = new_freeze
            if product == "b":
                value["files"]["model.pt"] = {"sha256": sha(directory / "b/model.pt"), "bytes": (directory / "b/model.pt").stat().st_size}
            path.write_text(json.dumps(value))
        refresh()
        caught = None
        try:
            publication.verify_release(directory, spec)
        except ValueError as error:
            caught = str(error)
        results.append({"id": "necessary_model_must_be_present_in_settings_freeze", "base_passed": True, "expected": "rejected", "actual": "rejected" if caught else "accepted", "passed": caught is not None, "error": caught, "support_graph_rebound": ["freeze SHA", "report freeze link", "both artifact manifest freeze links", "b model manifest hash/bytes", "complete spec file hashes", "transport SHA"]})

        source = ROOT / "docs/lesson-products/reviews/artifacts/engineering-runner-probes-original.py"
        module_spec = importlib.util.spec_from_file_location("publication_fake_classes", source)
        classes = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(classes)

        def remote(kind):
            case, directory, archive, spec, refresh = fixture()
            publication.verify_release(directory, spec)
            app = classes.FakeApp()
            events, uploads, downloads = [], [], []
            volume = SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: events.append("commit"))
            sdk = SimpleNamespace(App=lambda _: app, Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
                                  Secret=SimpleNamespace(from_name=lambda *a, **k: "synthetic-secret"), Image=SimpleNamespace(debian_slim=lambda **k: classes.FakeImage()),
                                  Workspace=SimpleNamespace(from_context=lambda: "synthetic-workspace"), current_function_call_id=lambda: "fc-synthetic")
            with patch.dict(sys.modules, {"modal": sdk}):
                _, function, finish = publish.register(archive)
            for name, cpu, memory, timeout in (("publish_remote", (1, 1), (2048, 2048), 600), ("finish_remote", (.25, .25), (1024, 1024), 180)):
                options = app.functions[name]["options"]
                assert options["cpu"] == cpu and options["memory"] == memory and options["timeout"] == timeout
                assert options["retries"] == 0 and options["max_containers"] == 1 and "gpu" not in options
            budget, quota = case / "budget.json", case / "quota.json"
            initial = {"reserved_total_usd": "10.00", "reservations": [{"run_id": "history", "reserved_usd": "10.00", "status": "completed"}]}
            publish.write_json(budget, initial)
            publish.write_json(quota, {"prior_bounds": ["history", "carry"]})
            policy_path = case / "repo" / publish.POLICY_PATH
            policy_path.parent.mkdir(parents=True)
            policy_path.write_bytes((ROOT / publish.POLICY_PATH).read_bytes())
            storage = {}
            prefix = spec["prefix"] + "/"
            if kind in ("existing", "existing-mismatch"):
                storage = {prefix + name: (directory / name).read_bytes() for name in spec["files"]}
                if kind == "existing-mismatch":
                    storage[prefix + "b/model.pt"] = b"different prior model"
            elif kind == "partial-prefix":
                storage[prefix + "b/model.pt"] = b"must not overwrite this prior artifact"
            storage_before = storage.copy()
            token_marker = "SYNTHETIC_TOKEN_MARKER_NOT_A_REAL_CREDENTIAL"

            class API:
                def __init__(self, token):
                    assert token == token_marker

                def repo_info(self, repo, repo_type):
                    if kind == "token-error":
                        raise RuntimeError("synthetic Authorization: Bearer " + token_marker)
                    return SimpleNamespace(private=kind == "private", sha="a" * 40)

                def file_exists(self, repo, name):
                    return name in storage

                def upload_folder(self, **kwargs):
                    assert "commit" in events
                    uploads.append({"prefix": kwargs["path_in_repo"], "parent_commit": kwargs.get("parent_commit")})
                    for name in spec["files"]:
                        storage[prefix + name] = (Path(kwargs["folder_path"]) / name).read_bytes()
                    return SimpleNamespace(oid="b" * 40)

            def download(repo, name, revision, token, force_download):
                assert token is False and force_download is True and revision in ("a" * 40, "b" * 40)
                downloads.append({"path": name, "revision": revision, "anonymous": True, "force_download": True})
                target = case / "anonymous" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(storage[name])
                return target

            def reserve(book, run_id, batch, revision, transport, descriptor, jobsha, live, hashes, policy, policysha, previous):
                assert descriptor["stage"] == "release" and descriptor["wall_seconds"] == 600 and policysha == publish.POLICY_SHA
                entry = {"run_id": run_id, "reserved_usd": ".10", "status": "reserved", "experiment_id": "selftrained_release", "gross_quota_guard": {"synthetic": True}}
                book["reservations"].append(entry)
                book["reserved_total_usd"] = "10.10"
                return entry, {**previous, "new_bound": ".10"}

            original_unpack = publication.unpack_verified
            def unpack(ignored, expected, destination):
                return original_unpack(archive, expected, destination)

            def mapped_path(value):
                value = str(value)
                return case / value.lstrip("/") if value.startswith(("/repo", "/course")) else Path(value)

            with patch.dict(sys.modules, {"modal": sdk, "huggingface_hub": SimpleNamespace(HfApi=API, hf_hub_download=download)}), patch.dict(os.environ, {"HF_TOKEN": token_marker}), patch.object(publish, "LEDGER", budget), patch.object(publish, "QUOTA", quota), patch.object(publish, "Path", side_effect=mapped_path), patch.object(publish, "check_source_files", return_value=None), patch.object(publish, "live_snapshot", return_value={}), patch.object(publish, "official_snapshot", return_value={"synthetic": True}), patch.object(publish, "reserve_entry", side_effect=reserve), patch.object(publish, "unpack_verified", side_effect=unpack):
                receipt = function("audit", "c" * 40, spec, {}, sha(budget), sha(quota), "synthetic/public")
            book = json.loads(budget.read_bytes())
            assert book["reservations"][0] == initial["reservations"][0]
            assert book["reservations"][-1]["receipt_sha256"] == publish.canonical_sha(receipt)
            assert book["reserved_total_usd"] == "10.10"
            retained = json.loads(quota.read_bytes())
            assert retained["prior_bounds"] == ["history", "carry"] and retained["new_bound"] == ".10"
            result = {"id": "remote_" + kind, "receipt_status": receipt["status"], "uploads": uploads, "anonymous_download_count": len(downloads), "downloads": downloads,
                      "storage_changed": storage != storage_before, "old_bound_retained": True, "receipt_hash_matches_ledger": True,
                      "primary_and_control_resource_options_checked": True, "error": receipt.get("error"), "scope": "True publish_remote, external APIs/numerical reserve/check_source fixture replacements; finance separately tested."}
            if kind == "partial-prefix":
                result.update(expected="no upload/no overwrite", passed=not uploads and storage == storage_before,
                              prior_model_overwritten=storage[prefix + "b/model.pt"] != storage_before[prefix + "b/model.pt"])
            elif kind == "token-error":
                result.update(expected="credential marker redacted", passed=token_marker not in json.dumps(receipt), credential_marker_in_receipt=token_marker in json.dumps(receipt))
            else:
                assert receipt["status"] == ("completed" if kind in ("new", "existing") else "failed")
                assert len(uploads) == (1 if kind == "new" else 0)
                result["passed"] = True
            return result

        for kind in ("new", "existing", "existing-mismatch", "private", "partial-prefix", "token-error"):
            results.append(remote(kind))

    assert before == {name: sha(ROOT / name) for name in checked_names}
    observed = [x["id"] for x in results if not x["passed"]]
    assert observed == ["necessary_model_must_be_present_in_settings_freeze", "remote_partial-prefix", "remote_token-error"]
    report = {"reviewer_task": "/root/tech_20261016_review_tools", "initial_context": "fresh", "current_context": "informed", "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
              "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform()},
              "checked_files": [{"path": name, "sha256": digest} for name, digest in before.items()], "results": results,
              "necessary_counterexamples": observed, "verdict": "revise", "limits": ["No actual model/Torch/GPU, no authenticated external calls or writes; all packages and token strings explicit synthetic fixtures.", "Spec/transport are fixture-authorized hashes. No genuine production spec exists; source/draft-release provenance not exercised here.", "All successful full/existing-prefix paths anonymously verify every file at fixed40hex commit; failed counterexamples are preserved as failures of contract, not claimed passes."]}
    path = ART / "engineering-publication-probes-results-original.json"
    with path.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps({"results": len(results), "necessary_counterexamples": observed, "receipt_sha256": sha(path)}))


if __name__ == "__main__":
    main()
