"""Synthetic CPU fixtures only: no Hub/Modal/auth/model calls or payload reads."""

import ast
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

CANDIDATE = Path(__file__).resolve().parent
ACTUAL = Path("/workspace/selftrained-v2")
sys.path.insert(0, str(CANDIDATE))


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"scripts.selftrained.{name}", CANDIDATE / "scripts/selftrained" / f"{name}.py"
    )
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


transport = load("hf_transport")
runner = load("modal_runner")
PAYLOAD = b"---\nlicense: mit\n---\n# Synthetic reviewed repository card\n"
MANIFEST_SHA = "a" * 64
PARENT, NEW = "b" * 40, "c" * 40


def release_for(payload=PAYLOAD):
    return {
        "mode": "repository-card",
        "repo_id": transport.PUBLIC_MODEL_REPO,
        "private": False,
        "parent_commit": PARENT,
        "manifest_sha256": MANIFEST_SHA,
        "confirm_write": "README.md",
        "file": {
            "path": transport.REPOSITORY_CARD_SOURCE,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "kind": "model_card",
            "license": "MIT",
            "redistribution_approved": True,
        },
    }


def job_for(release=None):
    return {
        "schema_version": 1,
        "stage": "release",
        "release": release or release_for(),
        "gross_quota_policy": {"path": "docs/selftrained/v2-gross-quota-20261006.json", "sha256": "d" * 64},
    }


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden in this synthetic proof")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    state = SimpleNamespace(
        parent=PARENT,
        private=False,
        existing=False,
        calls=[],
        downloads=[],
        metadata_sha=NEW,
        metadata_size=len(PAYLOAD),
        content=PAYLOAD,
        raise_commit=False,
        mark_committed=True,
        result_oid=NEW,
    )
    local = tmp_path / "README.md"

    class API:
        def __init__(self, token, endpoint):
            assert endpoint == "https://huggingface.co"
            self.token = token

        def repo_info(self, repo_id, **kw):
            assert self.token is False and repo_id == transport.PUBLIC_MODEL_REPO
            assert kw == {"repo_type": "model", "revision": "main"}
            return SimpleNamespace(private=state.private, sha=state.parent)

        def get_paths_info(self, repo_id, paths, **kw):
            assert self.token is False and paths == ["README.md"] and repo_id == transport.PUBLIC_MODEL_REPO
            assert kw == {"repo_type": "model", "revision": PARENT, "token": False}
            if not state.existing:
                return []
            oid = hashlib.sha1(f"blob {len(PAYLOAD)}\0".encode() + PAYLOAD).hexdigest()
            return [SimpleNamespace(path="README.md", size=len(PAYLOAD), blob_id=oid)]

        def create_commit(self, **kw):
            assert self.token == "synthetic-placeholder"
            state.calls.append(kw)
            assert kw["parent_commit"] == PARENT and kw["revision"] == "main" and kw["create_pr"] is False
            assert kw["repo_type"] == "model" and kw["repo_id"] == transport.PUBLIC_MODEL_REPO
            assert len(kw["operations"]) == 1
            op = kw["operations"][0]
            assert op.path_in_repo == "README.md" and op.path_or_fileobj == PAYLOAD
            if state.raise_commit:
                raise RuntimeError("Synthetic ambiguous commit response")
            if state.mark_committed:
                op._is_committed = True
            return SimpleNamespace(
                oid=state.result_oid, commit_url="https://huggingface.co/synthetic/commit/" + state.result_oid
            )

    def metadata(url, token):
        assert token is False and "/README.md" in url
        return SimpleNamespace(commit_hash=state.metadata_sha, size=state.metadata_size)

    def download(repo_id, path, **kw):
        assert repo_id == transport.PUBLIC_MODEL_REPO and path == "README.md"
        assert kw["token"] is False and kw["endpoint"] == "https://huggingface.co" and kw["repo_type"] == "model"
        assert kw["revision"] in (PARENT, NEW)
        state.downloads.append(kw)
        local.write_bytes(state.content)
        return str(local)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return state


def test_normal_singleton_cas_and_anonymous_roundtrip(hub):
    result = transport.publish_repository_card(PAYLOAD, release_for(), MANIFEST_SHA, "synthetic-placeholder")
    assert result["new_commit_created"] is True and result["no_op"] is False
    assert result["recovered_existing_revision"] is False and len(hub.calls) == 1 and len(hub.downloads) == 1
    assert result["files"][0]["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


def test_exact_parent_noop_does_not_create(hub):
    hub.existing, hub.metadata_sha = True, PARENT
    result = transport.publish_repository_card(PAYLOAD, release_for(), MANIFEST_SHA, "synthetic-placeholder")
    assert result["no_op"] and not result["new_commit_created"] and not result["recovered_existing_revision"]
    assert not hub.calls


@pytest.mark.parametrize("private,parent", [(True, PARENT), (False, NEW), (None, PARENT)])
def test_private_or_stale_parent_rejects_before_create(hub, private, parent):
    hub.private, hub.parent = private, parent
    with pytest.raises(ValueError, match="HEAD"):
        transport.publish_repository_card(PAYLOAD, release_for(), MANIFEST_SHA, "synthetic-placeholder")
    assert not hub.calls and not hub.downloads


@pytest.mark.parametrize(
    "field,value",
    [
        ("confirm_write", True),
        ("confirm_write", "sub/README.md"),
        ("repo_id", "other/repo"),
        ("private", True),
        ("parent_commit", "b" * 7),
        ("mode", "batch"),
    ],
)
def test_bad_top_descriptor_rejects(hub, field, value):
    release = release_for()
    release[field] = value
    with pytest.raises(ValueError):
        transport.publish_repository_card(PAYLOAD, release, MANIFEST_SHA, "synthetic-placeholder")
    assert not hub.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "../../secret"),
        ("path", "scripts/selftrained/train.py"),
        ("license", "Apache-2.0"),
        ("bytes", True),
        ("bytes", 65537),
        ("redistribution_approved", False),
        ("kind", "inference"),
    ],
)
def test_bad_file_descriptor_rejects(field, value):
    release = release_for()
    release["file"][field] = value
    with pytest.raises(ValueError):
        runner.validate_job(job_for(release))


def test_changed_actual_bytes_or_manifest_rejects_before_create(hub):
    for payload, manifest in [(PAYLOAD + b"x", MANIFEST_SHA), (PAYLOAD, "e" * 64)]:
        with pytest.raises(ValueError):
            transport.publish_repository_card(payload, release_for(), manifest, "synthetic-placeholder")
    assert not hub.calls


@pytest.mark.parametrize(
    "payload",
    [
        b"# No YAML\n",
        b"---\nlicense: apache-2.0\n---\nX\n",
        b"---\nlicense: mit\nlicense: apache-2.0\n---\nX\n",
        b"---\nlicense: mit\n---\n\x00",
    ],
)
def test_actual_frontmatter_or_text_rejects(payload):
    with pytest.raises(ValueError):
        transport.approved_repository_card(payload, release_for(payload), MANIFEST_SHA)


@pytest.mark.parametrize("change", ["commit_error", "sdk_skip", "bad_oid", "post_sha", "post_size", "post_bytes"])
def test_ambiguous_or_failed_roundtrip_never_returns_success(hub, change):
    if change == "commit_error":
        hub.raise_commit = True
    if change == "sdk_skip":
        hub.mark_committed = False
    if change == "bad_oid":
        hub.result_oid = "short"
    if change == "post_sha":
        hub.metadata_sha = PARENT
    if change == "post_size":
        hub.metadata_size += 1
    if change == "post_bytes":
        hub.content = b"x" * len(PAYLOAD)
    with pytest.raises((ValueError, RuntimeError)):
        transport.publish_repository_card(PAYLOAD, release_for(), MANIFEST_SHA, "synthetic-placeholder")
    assert len(hub.calls) == 1


def test_remote_source_binding_rejects_before_public_api(hub):
    with pytest.raises(ValueError, match="source SHA"):
        runner.repository_card_source_gate(PAYLOAD, release_for(), "f" * 40, MANIFEST_SHA, {})
    assert not hub.calls and not hub.downloads


def test_missing_policy_or_extra_model_options_rejects():
    for change in ("gross_quota_policy", "checkpoint"):
        job = job_for()
        if change == "gross_quota_policy":
            del job[change]
        else:
            job[change] = {"stage": "joint", "run_id": "synthetic", "path": "best.pt", "sha256": "0" * 64}
        with pytest.raises(ValueError):
            runner.validate_job(job)


def test_genuine_git_blob_and_changed_local_source(tmp_path, monkeypatch):
    root = tmp_path / "git"
    root.mkdir()
    file = root / transport.REPOSITORY_CARD_SOURCE
    file.parent.mkdir(parents=True)
    file.write_bytes(PAYLOAD)
    subprocess.run(["git", "init", "-q", root], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Synthetic",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Synthetic",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic card fixture"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr(runner, "ROOT", root)
    assert runner.committed_repository_card(job_for(), revision, MANIFEST_SHA) == PAYLOAD
    file.write_bytes(PAYLOAD.replace(b"reviewed", b"modified"))
    with pytest.raises(ValueError, match="Git blob"):
        runner.committed_repository_card(job_for(), revision, MANIFEST_SHA)


def test_legacy_batch_and_untouched_function_asts():
    job = json.loads((ACTUAL / "docs/selftrained/v2-jobs/release-final-four-exports.json").read_text())
    assert runner.validate_job(job) is job and runner.committed_repository_card(job, "not-used", MANIFEST_SHA) is None
    for module, changed in [("hf_transport", set()), ("modal_runner", {"validate_job", "register_modal"})]:
        old = ast.parse((ACTUAL / "scripts/selftrained" / f"{module}.py").read_text())
        new = ast.parse((CANDIDATE / "scripts/selftrained" / f"{module}.py").read_text())
        current = {n.name: ast.dump(n, include_attributes=False) for n in new.body if isinstance(n, ast.FunctionDef)}
        for node in old.body:
            if isinstance(node, ast.FunctionDef) and node.name not in changed:
                assert current[node.name] == ast.dump(node, include_attributes=False), node.name
    assert (CANDIDATE / "scripts/selftrained/finance.py").read_bytes() == (
        ACTUAL / "scripts/selftrained/finance.py"
    ).read_bytes()
    assert runner.resource_spec("release") == {"cpu": 1, "memory_gib": 2, "seconds": 600, "gpu": False}


def test_nested_server_gate_order_and_release_resources_unchanged():
    new = ast.parse((CANDIDATE / "scripts/selftrained/modal_runner.py").read_text())
    old = ast.parse((ACTUAL / "scripts/selftrained/modal_runner.py").read_text())

    def nested(tree):
        register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register_modal")
        return {n.name: n for n in register.body if isinstance(n, ast.FunctionDef)}

    current, baseline = nested(new), nested(old)
    assert ast.dump(current["release_remote"], include_attributes=False) == ast.dump(
        baseline["release_remote"], include_attributes=False
    )
    reserve = ast.unparse(current["reserve_remote"])
    assert (
        reserve.index("repository_card_source_gate(")
        < reserve.index("result = reserve_entry(")
        < reserve.index("write_json(GROSS_QUOTA")
    )
    execute = ast.unparse(current["execute"])
    assert (
        execute.index("repository_card_source_gate(")
        < execute.index("entry['status'] = 'running'")
        < execute.index("publish_repository_card(")
    )
    for name in ("finish_remote", "gpu_remote", "prepare_remote"):
        assert ast.dump(current[name], include_attributes=False) == ast.dump(baseline[name], include_attributes=False)


@pytest.mark.parametrize(
    "stage,reason",
    [("reserve", "bytes"), ("reserve", "parent"), ("reserve", "no_op"), ("execute", "bytes"), ("execute", "parent")],
)
def test_actual_nested_gate_rejects_before_any_state_write(tmp_path, hub, stage, reason):
    tree = ast.parse((CANDIDATE / "scripts/selftrained/modal_runner.py").read_text())
    register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register_modal")
    name = "reserve_remote" if stage == "reserve" else "execute"
    node = next(n for n in register.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    writes = []

    def forbidden_write(*args, **kwargs):
        writes.append("write")
        raise AssertionError("Rejected source must not write state")

    source_hashes = {transport.REPOSITORY_CARD_SOURCE: release_for()["file"]["sha256"]}
    entry = {
        "run_id": "synthetic",
        "stage": "release",
        "revision": "f" * 40,
        "manifest_sha256": MANIFEST_SHA,
        "job_sha256": "e" * 64,
        "batch_id": "synthetic",
        "status": "reserved",
        "source_sha256": source_hashes,
    }
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"reservations": [entry]}))
    before = ledger.read_bytes()
    volume = SimpleNamespace(reload=lambda: None, commit=forbidden_write)
    scope = {**runner.__dict__, "volume": volume, "LEDGER": ledger, "write_json": forbidden_write}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual-candidate-nested-gate>", "exec"), scope)
    payload = PAYLOAD + b"x" if reason == "bytes" else PAYLOAD
    if reason == "parent":
        hub.parent = NEW
    if reason == "no_op":
        hub.existing, hub.metadata_sha = True, PARENT
    with pytest.raises(ValueError):
        if stage == "reserve":
            scope[name](
                "synthetic",
                "synthetic",
                "f" * 40,
                {},
                MANIFEST_SHA,
                job_for(),
                "e" * 64,
                {},
                source_hashes,
                hashlib.sha256(before).hexdigest(),
                repository_card_bytes=payload,
            )
        else:
            scope[name]("release", "synthetic", "synthetic", "f" * 40, {}, MANIFEST_SHA, job_for(), "e" * 64, payload)
    assert ledger.read_bytes() == before and not writes and not hub.calls


def test_genuine_committed_cli_check_without_hub_dependency(tmp_path):
    root = tmp_path / "cli"
    root.mkdir()
    paths = [
        "scripts/selftrained/modal_runner.py",
        "scripts/selftrained/hf_transport.py",
        "scripts/selftrained/finance.py",
        "scripts/selftrained/train.py",
        "scripts/selftrained/evaluate.py",
        "docs/selftrained/v2-manifest.json",
        "docs/selftrained/v2-gross-quota-20261006.json",
    ]
    for path in paths:
        source = CANDIDATE / path if path in paths[:3] else ACTUAL / path
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    manifest_sha = hashlib.sha256((root / paths[5]).read_bytes()).hexdigest()
    policy_sha = hashlib.sha256((root / paths[6]).read_bytes()).hexdigest()
    release = release_for()
    release["manifest_sha256"] = manifest_sha
    job = job_for(release)
    job["gross_quota_policy"]["sha256"] = policy_sha
    (root / transport.REPOSITORY_CARD_SOURCE).parent.mkdir(parents=True)
    (root / transport.REPOSITORY_CARD_SOURCE).write_bytes(PAYLOAD)
    (root / "docs/selftrained/card-job.json").write_text(json.dumps(job))
    subprocess.run(["git", "init", "-q", root], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Synthetic",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Synthetic",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    env.pop("PYTHONPATH", None)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic committed CLI fixture"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # -S hides site packages; check must use only stdlib and local committed source.
    command = [
        sys.executable,
        "-S",
        str(root / paths[0]),
        "check",
        "--revision",
        revision,
        "--manifest",
        paths[5],
        "--job",
        "docs/selftrained/card-job.json",
    ]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["source_ready"] is True
