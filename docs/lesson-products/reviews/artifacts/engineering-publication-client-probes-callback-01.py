"""Independent current-source/schema, mocked GitHub preflight, and client-error probes.

No subprocess executes: Git/gh replies are in-memory fixture bytes. Modal/HF never start.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import publication, publish


def load_fixture_factory():
    path = ART / "engineering-publication-probes-original.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "5d178cd120e8fbbb256bf0e6d6b8bd2f69ac38ca142aedd2f32ef7065839486d"
    spec = importlib.util.spec_from_file_location("immutable_publication_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text()
    node = next(x for x in ast.walk(ast.parse(source)) if isinstance(x, ast.FunctionDef) and x.name == "fixture")
    body = textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]))
    body = body.replace('spec["transport_sha256"] = sha(archive)', 'spec["transport_sha256"] = sha(archive)\n        spec.update(release_id=456, release_commit="c" * 40, asset_name="fixture.tar.gz", asset_bytes=archive.stat().st_size)')
    factory = "def fixture_factory(work):\n    count = 0\n" + textwrap.indent(body, "    ") + "\n    return fixture\n"
    exec(compile(factory, "verified_original_fixture_extraction", "exec"), module.__dict__)
    return module.fixture_factory


def main():
    start = datetime.now(UTC).isoformat()
    names = ["scripts/lesson_products/publish.py", "scripts/lesson_products/publication.py", ".github/workflows/lesson-products-publish.yml", "tests/test_lesson_products_publication.py"]
    before = {p: publication.sha(ROOT / p) for p in names}
    results = []
    with tempfile.TemporaryDirectory(dir=ROOT / "work", prefix="publication-client-callback-") as temporary:
        fixture = load_fixture_factory()(Path(temporary))
        # Real inputs() against a synthetic committed tree: git outputs are byte-exact,
        # not permissive mocks, and the source mutation retains the original Git blob.
        case, payload, archive, valid, refresh = fixture()
        publication.verify_release(payload, valid)
        fake_root = case / "checkout"
        spec_path = "docs/lesson-products/jobs/synthetic-release.json"
        source_names = names[:2] + ["scripts/lesson_products/modal_runner.py", "scripts/lesson_products/checkpoints.py", "scripts/selftrained/modal_runner.py", "scripts/selftrained/finance.py", publish.POLICY_PATH, "docs/lesson-products/data-v2-manifest.json", names[2]]
        for name in source_names:
            dest = fake_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        spec_file = fake_root / spec_path
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(json.dumps(valid))
        revision = "d" * 40
        blobs = {name: (fake_root / name).read_bytes() for name in source_names + [spec_path]}
        git_calls = []
        def git(command, cwd):
            assert cwd == fake_root
            git_calls.append(command)
            if command == ["git", "rev-parse", "HEAD"]:
                return (revision + "\n").encode()
            assert command[:2] == ["git", "show"]
            assert command[2].startswith(revision + ":")
            return blobs[command[2].split(":", 1)[1]]
        with patch.object(publish, "ROOT", fake_root), patch.object(publish.subprocess, "check_output", side_effect=git):
            accepted, hashes = publish.inputs(revision, spec_path)
            assert accepted == valid and set(hashes) == set(blobs)
            results.append({"id": "inputs_exact_committed_tree_positive", "passed": True, "source_count": len(hashes), "git_show_calls": len(git_calls) - 1})
            for kind in ("wrong_checkout", "source_drift", "release_id_bool", "moving_release_commit", "bad_asset_name", "asset_bytes_bool", "asset_too_large", "policy_changed"):
                candidate = copy.deepcopy(valid)
                if kind == "source_drift":
                    (fake_root / names[0]).write_bytes(blobs[names[0]] + b"\n# synthetic uncommitted change\n")
                elif kind == "release_id_bool": candidate["release_id"] = True
                elif kind == "moving_release_commit": candidate["release_commit"] = "main"
                elif kind == "bad_asset_name": candidate["asset_name"] = "../payload.tar.gz"
                elif kind == "asset_bytes_bool": candidate["asset_bytes"] = True
                elif kind == "asset_too_large": candidate["asset_bytes"] = 64 * 1024 * 1024 + 1
                elif kind == "policy_changed": candidate["gross_quota_policy"]["sha256"] = "0" * 64
                spec_file.write_text(json.dumps(candidate))
                blobs[spec_path] = spec_file.read_bytes()
                caught = None
                try:
                    publish.inputs("e" * 40 if kind == "wrong_checkout" else revision, spec_path)
                except ValueError as error:
                    caught = str(error)
                assert caught, kind
                results.append({"id": "inputs_rejects_" + kind, "base_passed": True, "passed": True, "error": caught})
                (fake_root / names[0]).write_bytes(blobs[names[0]])
            spec_file.write_text(json.dumps(valid))
            blobs[spec_path] = spec_file.read_bytes()
            assert publish.inputs(revision, spec_path)[0] == valid

        def client(kind):
            case, payload, archive, spec, refresh = fixture()
            assert publication.verify_release(payload, spec)["files"] == 15
            client_root = case / "client"
            policy_target = client_root / publish.POLICY_PATH
            policy_target.parent.mkdir(parents=True)
            policy_target.write_bytes((ROOT / publish.POLICY_PATH).read_bytes())
            bundle = case / "download" / "asset.tar.gz"
            cached = kind != "fresh_positive"
            if cached:
                bundle.parent.mkdir()
                bundle.write_bytes(archive.read_bytes())
            parent = {"draft": True, "target_commitish": spec["release_commit"], "assets": [{"id": spec["asset_id"], "name": spec["asset_name"], "size": spec["asset_bytes"]}]}
            if kind == "not_draft": parent["draft"] = False
            elif kind == "wrong_parent_commit": parent["target_commitish"] = "f" * 40
            elif kind == "missing_asset": parent["assets"] = []
            elif kind == "wrong_asset_name": parent["assets"][0]["name"] = "other.tar.gz"
            elif kind == "wrong_asset_size": parent["assets"][0]["size"] += 1
            elif kind == "corrupt_cached_transport": bundle.write_bytes(bundle.read_bytes() + b"bad")
            elif kind == "rebound_size_disagrees":
                spec["asset_bytes"] += 1
                parent["assets"][0]["size"] = spec["asset_bytes"]
            env = {"GITHUB_REPOSITORY": "birdhackor/tiny-perceptron-vlm", "HF_TOKEN": "SYNTHETIC_HF_CALLBACK_MARKER", "GH_TOKEN": "SYNTHETIC_GH_CALLBACK_MARKER", "AUDIT_API_KEY": "SYNTHETIC_API_CALLBACK_MARKER"}
            markers = [env[key] for key in ("HF_TOKEN", "GH_TOKEN", "AUDIT_API_KEY")]
            events, live_reads = [], []
            initial = {"ledger": {"reserved_total_usd": "10.00", "reservations": [{"run_id": "history", "status": "completed", "reserved_usd": "10.00"}]}, "ledger_sha256": "1" * 64, "gross_quota_state": {"prior_bounds": ["history", "carry"]}, "gross_quota_state_sha256": "2" * 64, "live": {"gross_quota_snapshot": {"synthetic": True}}}
            initial_before = copy.deepcopy(initial)
            def metadata(command):
                events.append("gh-parent-metadata")
                assert command == ["gh", "api", "repos/birdhackor/tiny-perceptron-vlm/releases/456"]
                return json.dumps(parent).encode()
            def download(command, stdout, check):
                assert check is True
                assert command == ["gh", "api", "repos/birdhackor/tiny-perceptron-vlm/releases/assets/123", "-H", "Accept: application/octet-stream"]
                events.append("gh-asset-download")
                stdout.write(archive.read_bytes())
            def read_live(*args):
                live_reads.append(1)
                events.append("read-live")
                if len(live_reads) == 1:
                    return copy.deepcopy(initial)
                if kind == "client_and_finally_error":
                    raise RuntimeError("finally " + " ".join(markers) + " https://fixture.invalid/sensitive?q=marker")
                following = copy.deepcopy(initial)
                following["ledger"]["reservations"].append({"run_id": "callback", "status": "running" if kind == "client_and_finish_error" else "completed"})
                return following
            def reserve(book, *args):
                events.append("local-reserve-check")
                book["reservations"].append({"run_id": "local-only"})
            class Call:
                object_id = "fc-synthetic-client"
                def get(self):
                    events.append("get")
                    if kind in ("client_and_finally_error", "client_and_finish_error"):
                        raise RuntimeError("client " + " ".join(markers) + " https://fixture.invalid/sensitive?q=marker")
                    return {"status": "completed", "gpu_used": False}
            def spawn(*args):
                events.append("spawn")
                assert events.index("local-reserve-check") < events.index("spawn")
                return Call()
            def finish(*args):
                events.append("finish")
                raise RuntimeError("finish " + " ".join(markers))
            def register(path):
                events.append("register")
                assert path == bundle
                return SimpleNamespace(run=contextlib.nullcontext), SimpleNamespace(spawn=spawn), SimpleNamespace(remote=finish)
            argv = ["publish", "--spec", "docs/lesson-products/jobs/synthetic-release.json", "--revision", "d" * 40, "--run-id", "callback", "--bundle", str(bundle), "--release-repo", "synthetic/public"]
            caught = None
            console = io.StringIO()
            with patch.dict(os.environ, env), patch.object(sys, "argv", argv), patch.object(publish, "ROOT", client_root), patch.object(publish, "inputs", return_value=(spec, {})), patch.object(publish.subprocess, "check_output", side_effect=metadata), patch.object(publish.subprocess, "run", side_effect=download), patch.object(publish, "read_live_ledger", side_effect=read_live), patch.object(publish, "reserve_entry", side_effect=reserve), patch.object(publish, "register", side_effect=register), contextlib.redirect_stdout(console):
                try:
                    publish.main()
                except (ValueError, SystemExit) as error:
                    caught = {"type": type(error).__name__, "message": str(error)}
            good = kind in ("fresh_positive", "cached_positive")
            client_error = kind in ("client_and_finally_error", "client_and_finish_error")
            if good or client_error:
                result_file = client_root / "outputs/lesson-products-release/result.json"
                raw = result_file.read_text()
                result = json.loads(raw)
                assert result["status"] == ("completed" if good else "failed")
                assert result["all_bounds_retained"] is True
                assert result["function_call_id"] == "fc-synthetic-client"
                assert (caught is None) if good else (caught["type"] == "SystemExit")
                if client_error:
                    assert all(marker not in raw + console.getvalue() for marker in markers)
                    assert "[REDACTED]" in result["error"] and "[REDACTED]" in result["finish_error"]["message"]
                    assert result["error_type"] == "RuntimeError"
                assert initial == initial_before
            else:
                assert caught and caught["type"] == "ValueError", kind
                assert not live_reads and "register" not in events and "spawn" not in events
                assert "gh-asset-download" not in events  # all negative cases deliberately cached
            assert events[0] == "gh-parent-metadata"
            assert events.count("gh-asset-download") == (0 if cached else 1)
            return {"id": "client_" + kind, "valid_package_base_passed": True, "passed": True, "cached_bundle": cached, "events": events, "caught": caught, "no_real_external_call": True, "dummy_markers_absent_from_saved_output_and_stdout": True if client_error else None}
        for kind in ("fresh_positive", "cached_positive", "not_draft", "wrong_parent_commit", "missing_asset", "wrong_asset_name", "wrong_asset_size", "corrupt_cached_transport", "rebound_size_disagrees", "client_and_finally_error", "client_and_finish_error"):
            results.append(client(kind))
    assert before == {p: publication.sha(ROOT / p) for p in names}
    output = {"reviewer_task": "/root/tech_20261016_review_tools", "context": "informed callback", "started_at": start, "completed_at": datetime.now(UTC).isoformat(), "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform()}, "checked_files": [{"path": p, "sha256": digest} for p, digest in before.items()], "results": results, "all_expected": True, "limits": ["All Git/GitHub/Modal/live-ledger replies are explicit in-memory fixtures; no subprocess/network/cloud runs.", "Synthetic model bytes and formal report fixture do not constitute real model performance or actual draft release provenance.", "True inputs/schema and main/preflight/client branches execute; the source provider and cloud clients are deliberately separate isolated replacements."]}
    receipt = ART / "engineering-publication-client-results-callback-01.json"
    with receipt.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"cases": len(results), "all_expected": True, "receipt_sha256": publication.sha(receipt)}))


if __name__ == "__main__":
    main()
