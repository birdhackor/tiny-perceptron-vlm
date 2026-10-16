"""付費前獨立離線反例；所有 Modal/HF 與訓練均為明列的替身。"""

import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import modal_runner as runner

ART = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeImage:
    def __init__(self):
        self.operations = []

    def __getattr__(self, name):
        def operation(*args, **kwargs):
            self.operations.append((name, list(map(str, args)), kwargs))
            return self
        return operation


class FakeApp:
    def __init__(self):
        self.functions = {}

    def function(self, **kwargs):
        def decorate(function):
            self.functions[function.__name__] = {"function": function, "options": kwargs}
            return function
        return decorate

    def run(self):
        return contextlib.nullcontext()


def main():
    started = datetime.now(UTC).isoformat()
    results = []
    job = json.loads((ROOT / "docs/lesson-products/jobs/train-b-v1.json").read_bytes())
    checked_names = runner.SOURCE_FILES + ["tests/test_lesson_products_runner.py", "tiny_perceptron/lesson_products/__init__.py", "tiny_perceptron/__init__.py"]
    checked_names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "docs/lesson-products/jobs").glob("*.json"))]
    before = {name: sha(ROOT / name) for name in checked_names}

    def record(name, operation):
        try:
            detail = operation()
        except Exception as error:
            results.append({"id": name, "passed": False, "error_type": type(error).__name__, "error": str(error)})
        else:
            results.append({"id": name, "passed": True, "details": detail})

    workspace = ROOT / "work"
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="engineering-runner-audit-", dir=workspace) as folder:
        work = Path(folder)
        events = []
        app = FakeApp()
        volume = SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: events.append("commit"))
        images = []

        def image(**kwargs):
            value = FakeImage()
            images.append(value)
            return value

        class FunctionTimeoutError(Exception):
            pass

        class RemoteError(Exception):
            pass

        class OutputExpiredError(Exception):
            pass

        sdk = SimpleNamespace(App=lambda _: app, Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
                              Secret=SimpleNamespace(from_name=lambda *a, **k: "synthetic-secret"),
                              Image=SimpleNamespace(debian_slim=image), current_function_call_id=lambda: "fc-synthetic",
                              exception=SimpleNamespace(FunctionTimeoutError=FunctionTimeoutError, RemoteError=RemoteError, OutputExpiredError=OutputExpiredError))
        with patch.dict(sys.modules, {"modal": sdk}):
            _, reserve, execute, finish = runner.register("run", work / "input")

        def image_contract():
            for name in ("reserve_remote", "finish_remote"):
                options = app.functions[name]["options"]
                assert options["cpu"] == (.25, .25) and options["memory"] == (1024, 1024)
                assert options["timeout"] == 180 and options["retries"] == 0
                assert options["max_containers"] == 1 and options["scaledown_window"] == 2
                assert "gpu" not in options
            options = app.functions["execute_remote"]["options"]
            assert options["gpu"] == "L4" and options["cpu"] == (2, 2) and options["memory"] == (8192, 8192)
            assert options["timeout"] == 1800 and options["retries"] == 0 and options["max_containers"] == 1
            operations = [operation for value in images for operation in value.operations]
            files = {args[1] for name, args, kwargs in operations if name == "add_local_file"}
            directories = {args[1] for name, args, kwargs in operations if name == "add_local_dir"}
            assert {"/repo/uv.lock", "/repo/pyproject.toml", "/repo/.github/workflows/gpu-smoke.yml", "/repo/" + runner.POLICY_PATH} <= files
            assert {"/repo/tiny_perceptron", "/repo/scripts", "/repo/docs/lesson-products", "/input-data"} <= directories
            sync = [(args, kwargs) for name, args, kwargs in operations if name == "uv_sync"]
            assert len(sync) == 1 and sync[0][1] == {"extras": ["cu126", "selftrained"], "uv_version": "0.12.22", "extra_options": "--no-dev"}
            return {"decorators_checked": 3, "control_bound": "2*(180+2)s at cpu.25/RAM1GiB", "gpu_bound": "job wall+2s at one L4/cpu2/RAM8GiB", "source_mounts_cover_all_listed_inputs_and_jobs": True, "actual_image_build": False}

        record("image_sources_and_bounded_function_options", image_contract)
        ledger_path = work / "budget.json"

        def ledger(status="running"):
            return {"reservations": [{"run_id": "old", "reserved_usd": "10.00", "status": "completed"},
                                     {"run_id": "audit", "reserved_usd": ".50", "status": status, "experiment_id": "selftrained_lesson_products",
                                      "function_call_id": "fc-synthetic"}], "reserved_total_usd": "10.50"}

        def finish_case(kind):
            state = ledger("reserved" if kind == "no-dispatch" else "running")
            if kind == "no-dispatch":
                state["reservations"][-1].pop("function_call_id")
            runner.write_json(ledger_path, state)
            calls = []

            class Call:
                def cancel(self, terminate_containers):
                    calls.append(["cancel", terminate_containers])

                def get(self, timeout):
                    calls.append(["get", timeout])
                    if kind == "poll-timeout":
                        raise TimeoutError("still running")
                    if kind == "network":
                        raise ConnectionError("unknown status")
                    if kind == "remote-terminal":
                        raise RemoteError("terminal cancellation output")
                    if kind == "completion-race":
                        updated = json.loads(ledger_path.read_bytes())
                        updated["reservations"][-1]["status"] = "completed"
                        runner.write_json(ledger_path, updated)
                    return {"status": "completed"}

            sdk.FunctionCall = SimpleNamespace(from_id=lambda _: Call())
            caught = None
            with patch.dict(sys.modules, {"modal": sdk}), patch.object(runner, "LEDGER", ledger_path):
                try:
                    finish("audit", "fc-wrong" if kind == "wrong-id" else None)
                except Exception as error:
                    caught = {"type": type(error).__name__, "message": str(error)}
            after = json.loads(ledger_path.read_bytes())
            actual = after["reservations"][-1]["status"]
            if kind in ("poll-timeout", "network", "wrong-id", "no-dispatch"):
                assert caught and actual == state["reservations"][-1]["status"]
            else:
                assert caught is None and actual == ("completed" if kind == "completion-race" else "failed-or-cancelled")
            assert after["reserved_total_usd"] == "10.50"
            return {"status_after": actual, "error": caught, "call_events": calls, "all_reservations_retained": True}

        for kind in ("poll-timeout", "network", "wrong-id", "no-dispatch", "remote-terminal", "completion-race"):
            record("finish_" + kind, lambda kind=kind: finish_case(kind))

        def hf_case(kind):
            observed = []
            storage = {}

            class API:
                def __init__(self, token):
                    pass

                def repo_info(self, repo, repo_type):
                    return SimpleNamespace(private=(repo == "private") if kind != "wrong-visibility" else False)

                def upload_file(self, **kwargs):
                    revision = "commit-" + kwargs["repo_id"]
                    storage[(kwargs["repo_id"], revision)] = kwargs["path_or_fileobj"]
                    observed.append({"action": "upload", "repo": kwargs["repo_id"], "revision": revision})
                    return SimpleNamespace(oid=revision)

            def download(repo, name, revision, token, force_download):
                observed.append({"action": "download", "repo": repo, "revision": revision, "anonymous": token is False, "force_download": force_download})
                path = work / ("download-" + repo)
                path.write_bytes(storage[(repo, revision)] + (b"changed" if kind == "wrong-bytes" else b""))
                return path

            caught = None
            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=API, hf_hub_download=download)}), patch.dict(os.environ, {"HF_TOKEN": "synthetic-token"}):
                try:
                    receipts = runner.hf_roundtrip("private", "public", "audit", "a" * 40, "b" * 64)
                except ValueError as error:
                    caught = str(error)
            if kind == "valid":
                assert caught is None and len(receipts) == 2
                assert observed[-1]["anonymous"] is True and observed[1]["anonymous"] is False
                assert all(x["force_download"] for x in observed if x["action"] == "download")
            else:
                assert caught
                if kind == "wrong-visibility":
                    assert observed == []
            return {"events": observed, "error": caught, "real_HF_IO": False}

        for kind in ("valid", "wrong-visibility", "wrong-bytes"):
            record("hf_roundtrip_" + kind, lambda kind=kind: hf_case(kind))

        def reserve_case(kind):
            remote = work / ("reserve-" + kind)
            remote.mkdir()
            budget, quota = remote / "budget.json", remote / "gross-quota.json"
            initial = ledger("completed")
            initial["reservations"][-1]["run_id"] = "earlier"
            runner.write_json(budget, initial)
            prior_quota = {"synthetic_prior_bounds": ["earlier .50", "carry .54"]}
            runner.write_json(quota, prior_quota)
            policy_path = remote / "repo" / runner.POLICY_PATH
            policy_path.parent.mkdir(parents=True)
            policy_path.write_bytes((ROOT / runner.POLICY_PATH).read_bytes())
            events.clear()
            seen = []
            sdk.Workspace = SimpleNamespace(from_context=lambda: "synthetic workspace")

            def bounded_reserve(book, run_id, batch, revision, manifest_sha, descriptor, job_sha, live, hashes, policy, policy_sha, prior):
                assert prior == prior_quota and policy_sha == runner.POLICY_SHA
                assert len(book["reservations"]) == len(initial["reservations"])
                entry = {"run_id": run_id, "reserved_usd": ".50", "status": "reserved", "experiment_id": "selftrained_pretrain", "gross_quota_guard": {"synthetic": True}}
                book["reservations"].append(entry)
                book["reserved_total_usd"] = "11.00"
                following = copy.deepcopy(prior)
                following["new_bound"] = ".50"
                return entry, following

            def hf(*args):
                events.append("HF preflight")
                book = json.loads(budget.read_bytes())
                assert book["reservations"][-1]["status"] == "reserved"
                assert "commit" in events and json.loads(quota.read_bytes())["new_bound"] == ".50"
                if kind == "hf-failure":
                    raise RuntimeError("synthetic private/public roundtrip failed")
                return [{"synthetic": "private"}, {"synthetic": "public"}]

            def mapped_path(value):
                value = str(value)
                return remote / value.lstrip("/") if value.startswith(("/repo", "/course")) else Path(value)

            caught = None
            with patch.dict(sys.modules, {"modal": sdk}), patch.object(runner, "LEDGER", budget), patch.object(runner, "QUOTA", quota), patch.object(runner, "Path", side_effect=mapped_path), patch.object(runner, "check_source_files", return_value=None), patch.object(runner, "live_snapshot", return_value={}), patch.object(runner, "official_snapshot", return_value={"synthetic": True}), patch.object(runner, "reserve_entry", side_effect=bounded_reserve), patch.object(runner, "hf_roundtrip", side_effect=hf):
                try:
                    receipt = reserve("audit", "a" * 40, job, {runner.DATA_MANIFEST: "b" * 64}, sha(budget), "0" * 64 if kind == "stale-sha" else sha(quota), "private", "public")
                except Exception as error:
                    caught = {"type": type(error).__name__, "message": str(error)}
            after = json.loads(budget.read_bytes())
            if kind == "stale-sha":
                assert caught and after == initial and json.loads(quota.read_bytes()) == prior_quota
                assert "HF preflight" not in events
            else:
                assert len(after["reservations"]) == len(initial["reservations"]) + 1
                assert after["reservations"][:-1] == initial["reservations"]
                assert json.loads(quota.read_bytes())["synthetic_prior_bounds"] == prior_quota["synthetic_prior_bounds"]
                assert after["reservations"][-1]["status"] == ("failed" if kind == "hf-failure" else "reserved")
                if kind == "valid":
                    assert caught is None and len(receipt["hf_preflight"]) == 2
                else:
                    assert caught
            return {"events": events.copy(), "error": caught, "prior_ledger_entries_unchanged": True, "quota_prior_bounds_retained": True,
                    "stubbed": ["live APIs", "source mapping", "reserve_entry numerical guard (separately ran47 finance tests)", "HF bytes"]}

        for kind in ("valid", "hf-failure", "stale-sha"):
            record("reserve_" + kind, lambda kind=kind: reserve_case(kind))

        def source_hash_control():
            source = work / "source-hash-fixture.py"
            source.write_bytes(b"# synthetic original source\n")
            hashes = {source.name: sha(source)}
            runner.check_source_files(work, hashes)
            source.write_bytes(b"# changed source\n")
            try:
                runner.check_source_files(work, hashes)
            except ValueError as error:
                return {"base_passed": True, "changed_source_rejected": True, "error": str(error)}
            raise AssertionError("source change accepted")

        record("remote_source_hash_positive_and_tamper", source_hash_control)

        def client_case(kind):
            case = work / kind
            (case / "input").mkdir(parents=True)
            (case / "input/manifest.json").write_bytes(b"synthetic dataset manifest")
            policy_file = case / runner.POLICY_PATH
            policy_file.parent.mkdir(parents=True)
            policy_file.write_bytes((ROOT / runner.POLICY_PATH).read_bytes())
            hashes = {runner.DATA_MANIFEST: sha(case / "input/manifest.json")}
            sequence = []
            count = 0

            def live(*args):
                nonlocal count
                count += 1
                sequence.append("ledger_read")
                if count == 2 and kind == "finally-read-failure":
                    raise ConnectionError("synthetic final ledger read failure")
                status = "completed" if kind == "success" and count == 2 else "failed" if kind == "reserve-failure" and count == 2 else "running"
                return {"ledger": {"reservations": [{"run_id": "audit", "status": status}]}, "ledger_sha256": "a" * 64,
                        "gross_quota_state_sha256": "b" * 64, "gross_quota_state": {}, "live": {"gross_quota_snapshot": {"synthetic": True}}}

            def reserve_call(*args):
                sequence.append("reserve_and_HF_preflight")
                if kind == "reserve-failure":
                    raise RuntimeError("synthetic HF preflight failed")
                return {"reserved_usd": ".50", "hf_preflight": ["synthetic-private", "synthetic-public"]}

            class Call:
                object_id = "fc-synthetic"

                def get(self):
                    sequence.append("get")
                    if kind != "success":
                        raise TimeoutError("synthetic client lost result")
                    return {"status": "completed", "gpu_name": "synthetic L4; not actual hardware"}

            class Execute:
                def with_options(self, **kwargs):
                    assert kwargs == {"timeout": job["wall_seconds"]}
                    return self

                def spawn(self, *args):
                    sequence.append("spawn")
                    return Call()

            def finish_call(*args):
                sequence.append("finish")
                return {"status": "failed-or-cancelled"}

            args = ["runner", "--action", "run", "--job", "docs/lesson-products/jobs/train-b-v1.json", "--revision", "a" * 40, "--run-id", "audit", "--data-directory", str(case / "input"), "--checkpoint-repo", "private", "--release-repo", "public"]
            with patch.object(runner, "ROOT", case), patch.object(runner, "committed_inputs", return_value=(job, hashes)), patch.object(runner, "read_live_ledger", side_effect=live), patch.object(runner, "reserve_entry", return_value=None), patch.object(runner, "register", return_value=(FakeApp(), SimpleNamespace(remote=reserve_call), Execute(), SimpleNamespace(remote=finish_call))), patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                exit_code = 0
                try:
                    runner.main()
                except SystemExit as error:
                    exit_code = error.code
            result = json.loads((case / "outputs/lesson-products/runner-result.json").read_bytes())
            assert result["all_bounds_retained"] is True
            if kind == "success":
                assert exit_code == 0 and sequence.index("reserve_and_HF_preflight") < sequence.index("spawn") and "finish" not in sequence
            else:
                assert exit_code == 1 and result["error_type"] in ("TimeoutError", "RuntimeError")
            if kind == "reserve-failure":
                assert "spawn" not in sequence
            else:
                assert result["function_call_id"] == "fc-synthetic"
            if kind == "finally-read-failure":
                assert result["finish_error"]["type"] == "ConnectionError" and result["error_type"] == "TimeoutError"
            if kind == "result-timeout":
                assert sequence.count("finish") == 1
            assert sequence.count("reserve_and_HF_preflight") + sequence.count("finish") <= 2
            return {"events": sequence, "exit_code": exit_code, "status": result["status"], "primary_error": result.get("error_type"), "finish_error": result.get("finish_error"), "call_id_preserved": "function_call_id" in result, "all_bounds_retained": True,
                    "stubbed": ["external APIs", "committed_inputs", "local dry reserve_entry"]}

        for kind in ("success", "reserve-failure", "result-timeout", "finally-read-failure"):
            record("client_" + kind, lambda kind=kind: client_case(kind))

        def recovery_after_remote_failure():
            import torch
            import tiny_perceptron.lesson_products.train as training

            remote = work / "remote"
            directory = remote / "course/lesson-products/audit"
            directory.mkdir(parents=True)
            (directory / "reservation.json").write_text('{"synthetic_preflight":true}')
            input_manifest = remote / "input-data/manifest.json"
            input_manifest.parent.mkdir(parents=True)
            input_manifest.write_text("synthetic dataset")
            hashes = {runner.DATA_MANIFEST: sha(input_manifest)}
            state = ledger("reserved")
            state["reservations"][-1].update(job_sha256=runner.canonical_sha(job), source_sha256=hashes, gross_quota_guard={"synthetic": True})
            runner.write_json(ledger_path, state)
            uploaded = []

            class API:
                def __init__(self, token):
                    pass

                def upload_file(self, **kwargs):
                    uploaded.append({"path": kwargs["path_in_repo"], "bytes": Path(kwargs["path_or_fileobj"]).stat().st_size})
                    return SimpleNamespace(oid="synthetic-commit")

                def upload_folder(self, **kwargs):
                    raise AssertionError("失敗前不應已完成final folder upload")

            def fake_train(data, output, product, mode, **kwargs):
                assert "resume" not in kwargs and "resume_best" not in kwargs
                path = Path(output)
                (path / "best.pt").write_text('{"step":200,"synthetic_checkpoint":true}')
                (path / "last.pt").write_text('{"step":1000,"best_step":200,"synthetic_checkpoint":true}')
                (path / "config.json").write_text("{}")
                (path / "tokenizer.json").write_text("{}")
                kwargs["on_checkpoint"](path / "last.pt", 1000)
                raise RuntimeError("synthetic model failure after periodic checkpoint")

            original_path = Path
            def mapped_path(value):
                value = str(value)
                return remote / value.lstrip("/") if value.startswith(("/course", "/input-data")) else original_path(value)

            with patch.dict(sys.modules, {"modal": sdk, "huggingface_hub": SimpleNamespace(HfApi=API)}), patch.dict(os.environ, {"HF_TOKEN": "synthetic-token"}), patch.object(runner, "LEDGER", ledger_path), patch.object(runner, "Path", side_effect=mapped_path), patch.object(runner, "check_source_files", return_value=None), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_name", return_value="synthetic L4; no actual device"), patch.object(training, "train", side_effect=fake_train):
                receipt = execute("audit", "a" * 40, job, hashes, "private")
            after = json.loads(ledger_path.read_bytes())
            assert receipt["status"] == "failed" and after["reservations"][-1]["status"] == "failed"
            assert {x["path"] for x in uploaded} == {"lesson-products/audit/last.pt"}
            assert (directory / "best.pt").exists() and (directory / "execution.json").exists()
            assert after["reserved_total_usd"] == "10.50"
            return {"synthetic_training_only": True, "receipt_status": receipt["status"], "volume_files": sorted(p.name for p in directory.iterdir()), "HF_files": uploaded, "last_metadata": {"step": 1000, "best_step": 200}, "runner_passed_resume": False, "HF_has_selected_best": False,
                    "interpretation": "Volume有best+last；中途HF僅last，無best200，不能宣稱兩條完整續訓入口均已可用。"}

        record("remote_failure_preserves_volume_but_HF_resume_is_incomplete", recovery_after_remote_failure)
        resumed = copy.deepcopy(job)
        resumed["resume"] = {"run_id": "previous", "path": "last.pt", "sha256": "c" * 64}
        try:
            runner.validate_job(resumed)
        except ValueError as error:
            results.append({"id": "runner_resume_descriptor_not_supported", "passed": True, "details": {"actual": "rejected", "message": str(error), "scope": "Observed missing capability, not a passing recovery claim"}})
        else:
            raise AssertionError("unexpected resume support")

    assert before == {name: sha(ROOT / name) for name in checked_names}
    output = {"reviewer_task": "/root/tech_20261016_review_tools", "initial_context": "fresh", "current_context": "informed",
              "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
              "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform(), "torch_distribution": importlib.metadata.version("torch")},
              "checked_files": [{"path": name, "sha256": digest} for name, digest in before.items()],
              "probes": results, "all_probe_expectations_met": all(x["passed"] for x in results),
              "scope": "離線一致性/付費守門/失敗收尾；其中恢復不完整是實證缺口，all expectations met不是runner可付費pass。無真正Modal/HF I/O、GPU或optimizer step。"}
    path = ART / "engineering-runner-probes-results-original.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"probes": len(results), "all_probe_expectations_met": output["all_probe_expectations_met"], "failed": [x for x in results if not x["passed"]], "result_sha256": sha(path)}, ensure_ascii=False))
    return 0 if output["all_probe_expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
