"""保留舊收據，於新版重跑guard/recovery並新增資料鏈與binary傳回反例。"""

import copy
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import checkpoints as checkpoint
from scripts.lesson_products import modal_runner as runner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_and_run():
    previous = {}
    for name, digest in (("engineering-runner-original.json", "368139cecf659abc40602b93f2ec66b3230499f2aaba23b72c382e5f843057e0"), ("engineering-runner-callback-01.json", "b0665d9ce2b64db230d96ed9186547762b26075c1697a0da1229a2d311a8cbad")):
        path = ROOT / "docs/lesson-products/reviews" / name
        assert sha(path) == digest
        previous[name] = json.loads(path.read_bytes())
    reference = next(x for x in previous["engineering-runner-original.json"]["artifacts"] if x["id"] == "probe_code")
    source = ROOT / reference["path"]
    assert sha(source) == reference["sha256"]
    text = source.read_text()
    generated = text.split("        def recovery_after_remote_failure():", 1)[0] + "    assert before ==" + text.split("    assert before ==", 1)[1]
    marker = '                    return {"status": "completed", "gpu_name": "synthetic L4; not actual hardware"}'
    replacement = '''                    raw = b"synthetic client artifact; no model"
                    return {"receipt": {"status": "completed", "gpu_name": "synthetic L4; not actual hardware", "client_artifacts": {"model.pt": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}}, "artifacts": {"model.pt": raw}}'''
    assert generated.count(marker) == 1
    generated = generated.replace(marker, replacement)
    generated = generated.replace("engineering-runner-probes-results-original.json", "engineering-runner-regression-probes-results-callback-02.json")
    generated = generated.replace("離線一致性/付費守門/失敗收尾；其中恢復不完整是實證缺口，all expectations met不是runner可付費pass。無真正Modal/HF I/O、GPU或optimizer step。", "18 original guard regressions, updated synthetic client success return to real envelope format; no actual services/hardware/training.")
    namespace = {"__file__": str(__file__), "__name__": "guard_regressions_callback02"}
    exec(compile(generated, str(__file__) + "::original-guard-prefix", "exec"), namespace)
    assert namespace["main"]() == 0

    reference = next(x for x in previous["engineering-runner-callback-01.json"]["artifacts"] if x["id"] == "recovery_code")
    source = ROOT / reference["path"]
    assert sha(source) == reference["sha256"]
    generated = source.read_text().replace("engineering-runner-recovery-probes-results-callback-01.json", "engineering-runner-recovery-probes-results-callback-02.json")
    line = '            receipt = execute("original", "a" * 40, job, hashes, "synthetic/private")'
    assert generated.count(line) == 1
    generated = generated.replace(line, '''            envelope = execute("original", "a" * 40, job, hashes, "synthetic/private")
            receipt = envelope["receipt"]''')
    line = '        assert receipt["status"] == "failed" and book["reservations"][-1]["status"] == "failed"'
    assert generated.count(line) == 1
    generated = generated.replace(line, line + '''
        json.dumps(receipt)
        assert book["reservations"][-1]["receipt_sha256"] == runner.canonical_sha(receipt)
        transferred = checkpoint.save_transferred(envelope, work / "client-failure-artifacts")
        assert set(transferred) == set(envelope["artifacts"])
        assert all((work / "client-failure-artifacts" / name).read_bytes() == raw for name, raw in envelope["artifacts"].items())
        results.append({"id": "failed_remote_envelope_to_local", "expected": "pass", "actual": "pass", "passed": True, "file_count": len(transferred), "receipt_json_only": True, "ledger_receipt_hash_matches": True})''')
    old = '''    assert observed == ["HF_preflight_binds_original_data_identity", "Volume_preflight_binds_original_data_identity"]'''
    assert generated.count(old) == 1
    generated = generated.replace(old, '    assert observed == []')
    generated = generated.replace('"verdict": "revise"', '"verdict": "pass"')
    namespace = {"__file__": str(__file__), "__name__": "recovery_regressions_callback02"}
    exec(compile(generated, str(__file__) + "::immutable-callback01-recovery", "exec"), namespace)
    namespace["main"]()


def new_cases():
    started = datetime.now(UTC).isoformat()
    results = []
    workroot = ROOT / "work"
    workroot.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workroot, prefix="runner-envelope-callback02-") as folder:
        work = Path(folder)
        counter = 0

        def record(name, operation):
            try:
                detail = operation()
            except Exception as error:
                results.append({"id": name, "passed": False, "error_type": type(error).__name__, "error": str(error)})
            else:
                results.append({"id": name, "passed": True, "details": detail})

        def source_case(kind, fault):
            nonlocal counter
            counter += 1
            case = work / str(counter)
            source = case / "lesson-products/original"
            source.mkdir(parents=True)
            for name in checkpoint.FILES:
                (source / name).write_bytes(("explicit synthetic checkpoint:" + name).encode())
            job = json.loads((ROOT / "docs/lesson-products/jobs/train-b-v1.json").read_bytes())
            hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in checkpoint.TRAIN_SOURCES}
            hashes[checkpoint.DATA_MANIFEST] = "d" * 64
            checkpoint.bundle(source, "original", 1000, hashes, job)
            resume = {"run_id": "original", "source": kind, "hf_revision": "a" * 40 if kind == "hf" else None, "bundle_sha256": sha(source / "checkpoint-bundle.json")}
            book = {"reservations": [{"run_id": "original", "experiment_id": "selftrained_lesson_products", "status": "failed", "manifest_sha256": hashes[checkpoint.DATA_MANIFEST], "source_sha256": hashes.copy(), "runtime_options": copy.deepcopy(job)}]}
            observed = []

            def download(repo, name, revision, token):
                observed.append(revision)
                assert revision == "a" * 40
                return source / name.rsplit("/", 1)[1]

            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}), patch.dict(os.environ, {"HF_TOKEN": "synthetic token"}):
                checkpoint.prepare_resume(resume, case / "positive", case, "synthetic/private", hashes, job, book)
                entry = book["reservations"][0]
                if fault == "entry_manifest":
                    entry["manifest_sha256"] = "0" * 64
                elif fault == "entry_data_source":
                    entry["source_sha256"][checkpoint.DATA_MANIFEST] = "0" * 64
                elif fault == "entry_training_source":
                    entry["source_sha256"][checkpoint.TRAIN_SOURCES[0]] = "0" * 64
                elif fault == "entry_recipe":
                    entry["runtime_options"]["seed"] += 1
                elif fault == "missing_entry_source":
                    entry.pop("source_sha256")
                elif fault == "bundle_data":
                    path = source / "checkpoint-bundle.json"
                    data = json.loads(path.read_bytes())
                    data["data_manifest_sha256"] = "0" * 64
                    path.write_text(json.dumps(data))
                    resume["bundle_sha256"] = sha(path)
                else:
                    raise AssertionError(fault)
                try:
                    checkpoint.prepare_resume(resume, case / "negative", case, "synthetic/private", hashes, job, book)
                except ValueError as error:
                    assert "續訓來源" in str(error)
                    return {"base_passed": True, "actual": "rejected_on_CPU", "error": str(error), "downloads_pinned": all(x == "a" * 40 for x in observed)}
                raise AssertionError("mismatched original source accepted")

        for kind in ("volume", "hf"):
            for fault in ("entry_manifest", "entry_data_source", "entry_training_source", "entry_recipe", "missing_entry_source", "bundle_data"):
                record(kind + "_" + fault, lambda kind=kind, fault=fault: source_case(kind, fault))

        raw = b"synthetic artifact, not real weights"
        expected = {"model.pt": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
        original = {"receipt": {"status": "completed", "client_artifacts": expected}, "artifacts": {"model.pt": raw}}

        def envelope_fault(fault):
            nonlocal counter
            counter += 1
            checkpoint.save_transferred(original, work / f"envelope-control-{counter}")
            value = copy.deepcopy(original)
            if fault == "corrupt_bytes":
                value["artifacts"]["model.pt"] = b"wrong"
            elif fault == "wrong_size":
                value["receipt"]["client_artifacts"]["model.pt"]["bytes"] += 1
            elif fault == "missing_payload":
                value["artifacts"].clear()
            elif fault == "extra_payload":
                value["artifacts"]["best.pt"] = raw
            elif fault == "path_traversal":
                value = {"receipt": {"client_artifacts": {"../escape": next(iter(expected.values()))}}, "artifacts": {"../escape": raw}}
            elif fault == "leaf_symlink":
                destination = work / f"envelope-negative-{counter}"
                destination.mkdir()
                sentinel = work / f"external-{counter}"
                sentinel.write_bytes(b"unchanged")
                (destination / "model.pt").symlink_to(sentinel)
            destination = work / f"envelope-negative-{counter}"
            try:
                checkpoint.save_transferred(value, destination)
            except ValueError as error:
                if fault == "leaf_symlink":
                    assert sentinel.read_bytes() == b"unchanged"
                return {"base_passed": True, "actual": "rejected", "error": str(error)}
            raise AssertionError("bad envelope accepted")

        for fault in ("corrupt_bytes", "wrong_size", "missing_payload", "extra_payload", "path_traversal", "leaf_symlink"):
            record("envelope_" + fault, lambda fault=fault: envelope_fault(fault))

        def bounded_transfer():
            source = work / "size-boundary"
            source.mkdir()
            (source / "model.pt").write_bytes(b"x" * (20 * 1024 * 1024))
            (source / "b.jsonl").write_text("synthetic raw data must not be included")
            payload, hashes = checkpoint.transfer_payload(source)
            assert set(payload) == {"model.pt"}
            checkpoint.save_transferred({"receipt": {"client_artifacts": hashes}, "artifacts": payload}, work / "size-exact-positive")
            (source / "config.json").write_bytes(b"x")
            caught = []
            try:
                checkpoint.transfer_payload(source)
            except ValueError as error:
                caught.append(str(error))
            payload["config.json"] = b"x"
            hashes["config.json"] = {"bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
            try:
                checkpoint.save_transferred({"receipt": {"client_artifacts": hashes}, "artifacts": payload}, work / "size-over-negative")
            except ValueError as error:
                caught.append(str(error))
            assert len(caught) == 2 and not (work / "size-over-negative").exists()
            return {"exact_20MiB_accepted": True, "20MiB_plus_one_byte_rejected_at_source_and_client": True, "raw_data_excluded": True, "errors": caught}

        record("transfer_size_and_raw_data_boundary", bounded_transfer)
    report = {"reviewer_task": "/root/tech_20261016_review_tools", "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "results": results, "all_passed": all(r["passed"] for r in results), "scope": "Pure local temp input fixtures; no training, paid/cloud/HF operations. Counterexamples each first pass their own unchanged supporting fixture."}
    path = ART / "engineering-runner-source-envelope-results-callback-02.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"additional_probes": len(results), "all_passed": report["all_passed"], "failed": [x for x in results if not x["passed"]], "result_sha256": sha(path)}))
    assert report["all_passed"]


if __name__ == "__main__":
    derive_and_run()
    new_cases()
