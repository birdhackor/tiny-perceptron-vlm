"""核准的根 README 只在固定權重與匿名下載都通過後發布；完全離線。"""

import ast
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_model_index import build_index
from scripts.course_release import file_sha256, validate_approval

ROOT = Path(__file__).resolve().parents[1]


def _index(content="# 教學模型索引\n\n只有已發布模型；權重逐檔授權。\n"):
    return {"content": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def _approval():
    return {
        "schema_version": 1,
        "approved": True,
        "reviewed": True,
        "experiment_id": "test",
        "batch_id": "course-v1",
        "revision": "1" * 40,
        "private_source": {"repo": "owner/private", "revision": "2" * 40, "prefix": "course/course-v1/test/run-1"},
        "files": [
            {
                "path": "model.pt",
                "kind": "checkpoint",
                "sha256": "3" * 64,
                "license": "MIT",
                "redistribution_approved": True,
            }
        ],
        "model_card": {"summary": "教學模型", "scope": "受控任務", "limitations": ["只適用此小世界"]},
    }


@pytest.mark.parametrize(
    "index",
    [
        None,
        [],
        {},
        _index(" "),
        _index("缺少索引標題的純文字全文"),
        _index("# short"),
        _index("# 模型\n" + "a" * 262144),
        _index("# 模型索引\n\n含有不允許的 NUL\x00 控制字元。"),
        _index() | {"sha256": "0" * 64},
        _index() | {"sha256": "main"},
        _index() | {"sha256": 42},
        _index() | {"content": ["Markdown"]},
        _index() | {"reviewed": True},
        {"content": "# Valid title\n\nUnpaired surrogate: \ud800", "sha256": "0" * 64},
    ],
)
def test_repository_index_requires_exact_reviewed_markdown_bytes(index):
    with pytest.raises(ValueError, match="repository_index"):
        validate_approval(_approval() | {"repository_index": index}, "test", "course-v1", "owner/private")


def test_optional_index_preserves_source_and_does_not_enter_approved_artifacts():
    approved = _approval()
    original = copy.deepcopy(approved)
    expected = approved["private_source"]
    assert validate_approval(approved, "test", "course-v1", "owner/private") == expected
    approved["repository_index"] = _index()
    assert validate_approval(approved, "test", "course-v1", "owner/private") == expected
    assert approved["files"] == original["files"]


def _local_release(tmp_path, monkeypatch, with_index=True, corrupt_weight=False, corrupt_index=False):
    """從 AST 取實際 release 本體，模擬 HF immutable commits，不載入 Modal。"""
    import huggingface_hub

    import scripts.course_release as exports

    approval = _approval()
    if with_index:
        approval["repository_index"] = _index()
    source = tmp_path / "course-v1/test/release-source"
    source.mkdir(parents=True)
    approval_text = json.dumps(approval, ensure_ascii=False)
    digest = hashlib.sha256(approval_text.encode()).hexdigest()
    directory = source / digest
    directory.mkdir()
    (directory / "approved-public-exports.json").write_text(approval_text, encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"status": "completed", "revision": "1" * 40}))
    (directory / "approval-attestation.json").write_text(json.dumps({"approval_git_revision": "4" * 40}))

    def export(_directory, target, _approval, _provenance):
        (target / "model.pt").write_bytes(b"inference-only weight bytes")
        (target / "LICENSE").write_text("MIT")
        (target / "export-manifest.json").write_text('{"files":[]}')

    monkeypatch.setattr(exports, "build_export", export)

    class FakeAPI:
        def __init__(self):
            self.commits = {}
            self.events = []
            self.previous = {}

        def upload_folder(self, **kwargs):
            snapshot = self.previous.copy()
            folder = Path(kwargs["folder_path"])
            for path in folder.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(folder).as_posix()
                if kwargs.get("allow_patterns") and relative not in kwargs["allow_patterns"]:
                    continue
                snapshot[f"{kwargs['path_in_repo']}/{relative}"] = path.read_bytes()
            return self._commit(snapshot, "folder")

        def upload_file(self, **kwargs):
            assert kwargs["path_in_repo"] == "README.md"
            assert kwargs["path_or_fileobj"] == approval["repository_index"]["content"].encode("utf-8")
            snapshot = self.previous.copy()
            snapshot["README.md"] = kwargs["path_or_fileobj"]
            return self._commit(snapshot, "root-index")

        def _commit(self, snapshot, event):
            revision = f"{len(self.commits) + 1:040x}"
            self.commits[revision] = snapshot
            self.previous = snapshot
            self.events.append((event, revision))
            return SimpleNamespace(oid=revision)

    api = FakeAPI()

    def download(**kwargs):
        assert kwargs["token"] is False
        path = kwargs["filename"]
        data = api.commits[kwargs["revision"]][path]
        api.events.append(("anonymous-download", kwargs["revision"], path))
        if corrupt_weight and path.endswith("model.pt") or corrupt_index and path == "README.md":
            data += b"altered"
        target = Path(kwargs["local_dir"]) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **_kwargs: api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setenv("HF_TOKEN", "offline-placeholder")

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    tree = ast.parse((ROOT / "scripts/modal_course.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "release")
    function.decorator_list = []
    namespace = {
        "VOLUME_ROOT": tmp_path,
        "volume": SimpleNamespace(reload=lambda: None, commit=lambda: None),
        "json": json,
        "Path": Path,
        "os": os,
        "tempfile": tempfile,
        "sha256": file_sha256,
        "write_json": write_json,
        "approved_files": lambda *_args: approval["files"],
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "release-offline-test", "exec"), namespace)
    return lambda: json.loads(
        namespace["release"]("owner/private", "owner/public", "test", "course-v1", "run-1", digest)
    ), api


@pytest.mark.parametrize("with_index", [False, True])
def test_root_index_runs_after_weight_roundtrip_without_changing_experiment_revision(tmp_path, monkeypatch, with_index):
    run, api = _local_release(tmp_path, monkeypatch, with_index=with_index)
    response = run()
    experiment_revision = f"{2:040x}"
    assert response["revision"] == response["public_manifest"]["revision"] == experiment_revision
    assert not any(item["path"] == "README.md" for item in response["public_manifest"]["files"])
    if with_index:
        position = next(i for i, event in enumerate(api.events) if event[0] == "root-index")
        preceding_downloads = [event for event in api.events[:position] if event[0] == "anonymous-download"]
        assert len(preceding_downloads) == len(response["public_manifest"]["files"])
        assert response["root_index_revision"] == f"{3:040x}"
        assert response["root_index_sha256"] == _index()["sha256"]
        assert response["root_index_anonymous_download_verified"] is True
        assert api.commits[experiment_revision] == {
            key: value for key, value in api.commits[response["root_index_revision"]].items() if key != "README.md"
        }
    else:
        assert all(event[0] != "root-index" for event in api.events)
        assert not any(key.startswith("root_index") for key in response)


def test_failed_weight_download_never_publishes_root_index(tmp_path, monkeypatch):
    run, api = _local_release(tmp_path, monkeypatch, corrupt_weight=True)
    with pytest.raises(RuntimeError, match="公開檔案 SHA"):
        run()
    assert all(event[0] != "root-index" for event in api.events)


def test_root_index_anonymous_download_is_hash_checked(tmp_path, monkeypatch):
    run, _api = _local_release(tmp_path, monkeypatch, corrupt_index=True)
    with pytest.raises(RuntimeError, match="根 README SHA"):
        run()


def _public_manifest():
    files = [
        {"path": f"course/batch/example/{name}", "output": name, "sha256": "5" * 64, "bytes": 16}
        for name in ("README.md", "LICENSE", "export-manifest.json", "model.pt")
    ]
    return {
        "schema_version": 1,
        "models": [
            {
                "id": "example",
                "repo": "owner/public",
                "revision": "6" * 40,
                "files": files,
                "inference": {
                    "script": "scripts/infer.py",
                    "checkpoint": "model.pt",
                    "prompt": "2+3=?",
                    "chat": True,
                    "tokens": 8,
                },
            }
        ],
    }


def test_index_contains_only_published_pinned_models_and_exact_student_commands():
    manifest = _public_manifest()
    content = build_index(manifest)
    assert "public/resolve/" + "6" * 40 + "/course/batch/example/model.pt?download=true" in content
    assert "scripts/fetch_course_models.py --model example" in content
    assert "scripts/infer.py checkpoints/course/example/model.pt --prompt '2+3=?' --chat --tokens 8" in content
    assert "程式碼採 **MIT**" in content and "權重與資料按檔案適用個別授權" in content
    assert "不代表所有課程實驗都已完成" in content
    assert "example" in content and "future-model" not in content
    validate_approval(_approval() | {"repository_index": _index(content)}, "test", "course-v1", "owner/private")
    manifest["models"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="固定"):
        build_index(manifest)
    with pytest.raises(ValueError, match="沒有已發布"):
        build_index({"schema_version": 1, "models": []})
