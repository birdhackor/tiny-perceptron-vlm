"""發布 receipt 必須綁定不可變 Git 核准資料，並只寫公共下載指標。"""

import copy
import hashlib
import json
import subprocess

import pytest

import scripts.ingest_course_release as release_ingest


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    """模擬 Git show 的固定 blob；工作目錄故意放不同的未核准草稿。"""
    repository = tmp_path / "repo"
    repository.mkdir()
    manifest = tmp_path / "published/public-models.json"
    proofs = tmp_path / "published/public-releases"
    plan = repository / "docs/course-experiments/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "course-v1",
                "sequence": [{"id": name} for name in ("simple_models", "text_foundation", "real_text")],
            }
        )
    )
    blobs = {}
    invocations = []

    def git_show(argv, **kwargs):
        assert argv[:2] == ["git", "show"]
        assert kwargs["cwd"] == repository
        assert kwargs["check"] is True and kwargs["capture_output"] is True
        invocations.append(argv[2])
        if argv[2] not in blobs:
            raise subprocess.CalledProcessError(1, argv, stderr=b"missing blob")
        return subprocess.CompletedProcess(argv, 0, stdout=blobs[argv[2]], stderr=b"")

    monkeypatch.setattr(release_ingest.subprocess, "run", git_show)

    def create(name="text_foundation", revision="4" * 40, approval_change=None):
        approval = {
            "schema_version": 1,
            "approved": True,
            "reviewed": True,
            "experiment_id": name,
            "batch_id": "course-v1",
            "revision": "1" * 40,
            "private_source": {
                "repo": "owner/private",
                "revision": "2" * 40,
                "prefix": f"course/course-v1/{name}/run-1",
            },
            "files": [
                {
                    "path": "model.pt",
                    "kind": "checkpoint",
                    "sha256": "a" * 64,
                    "license": "MIT",
                    "redistribution_approved": True,
                },
                {
                    "path": "tokenizer.json",
                    "kind": "metadata",
                    "sha256": "b" * 64,
                    "license": "MIT",
                    "redistribution_approved": True,
                },
            ],
            "inference": {
                "script": "scripts/infer.py",
                "checkpoint": "model.pt",
                "prompt": "1+2=?",
                "tokens": 8,
                "chat": True,
            },
            "model_card": {
                "summary": "Toy",
                "scope": "Controlled",
                "limitations": ["Tiny"],
                "training_data": [{"text": "PRIVATE HUMAN RECORD"}],
            },
        }
        if approval_change:
            approval.update(approval_change)
        raw = (json.dumps(approval, ensure_ascii=False, indent=2) + "\n").encode()
        git_revision = "3" * 40
        blob_name = f"{git_revision}:docs/course-experiments/releases/{name}.json"
        blobs[blob_name] = raw
        draft = repository / f"docs/course-experiments/releases/{name}.json"
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.write_text('{"reviewed":false,"content":"UNREVIEWED WORKING DRAFT"}')
        prefix = f"course/course-v1/{name}"
        files = [
            {
                "path": f"{prefix}/{output}",
                "output": output,
                "sha256": "b" * 64 if output == "tokenizer.json" else "c" * 64,
                "bytes": 16,
            }
            for output in ("LICENSE", "README.md", "export-manifest.json", "model.pt", "tokenizer.json")
        ]
        result = {
            "experiment_id": name,
            "mode": "release",
            "preflight": {"billing": "PRIVATE ACCOUNT DATA"},
            "billing": {"account": "PRIVATE ACCOUNT DATA"},
            "raw_text": "PRIVATE HUMAN RECORD",
            "approval": {
                "approval_git_revision": git_revision,
                "approval_sha256": hashlib.sha256(raw).hexdigest(),
                "private_source": approval["private_source"],
                "files_verified": len(approval["files"]),
            },
            "release": {
                "experiment_id": name,
                "repo": "owner/public",
                "revision": revision,
                "weights_revision": "5" * 40,
                "anonymous_download_verified": True,
                "gpu_used": False,
                "raw_data": [{"text": "PRIVATE HUMAN RECORD"}],
                "public_manifest": {
                    "id": name,
                    "repo": "owner/public",
                    "revision": revision,
                    "files": files,
                    "inference": copy.deepcopy(approval["inference"]),
                },
            },
        }
        return result, approval, blob_name

    def ingest(result, **kwargs):
        path = tmp_path / "actions-result.json"
        path.write_text(json.dumps(result))
        return release_ingest.ingest(
            path,
            repo_root=repository,
            manifest_path=manifest,
            proof_dir=proofs,
            plan_path=plan,
            checkpoint_repo="owner/private",
            release_repo="owner/public",
            **kwargs,
        )

    return create, ingest, manifest, proofs, blobs, invocations


def test_immutable_git_approval_and_minimal_durable_proof(receipts):
    create, ingest, manifest, proofs, _blobs, invocations = receipts
    result, _approval, blob_name = create()
    # 額外結果／file 欄位不會被搬進公開證據。
    result["release"]["public_manifest"]["files"][0]["data"] = "PRIVATE HUMAN RECORD"
    summary = ingest(result)
    assert invocations == [blob_name] and summary["status"] == "ingested"
    model = json.loads(manifest.read_text())["models"][0]
    proof = json.loads((proofs / "text_foundation.json").read_text())
    assert model == proof["release"]["public_manifest"]
    assert proof["approval"]["approval_sha256"] == result["approval"]["approval_sha256"]
    assert proof["approval"]["approval_git_revision"] == "3" * 40
    assert proof["release"]["anonymous_download_verified"] is True and proof["release"]["gpu_used"] is False
    assert "Remote release worker" in proof["release"]["anonymous_verification_scope"]
    combined = manifest.read_text() + (proofs / "text_foundation.json").read_text()
    assert not any(
        text in combined
        for text in ("PRIVATE", "UNREVIEWED", "billing", "preflight", "raw_data", "model_card", "training_data")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(mode="preflight"),
        lambda r: r.update(status="failed"),
        lambda r: r.update(billing_error={"message": "failure"}),
        lambda r: r["release"].update(status="failed"),
        lambda r: r["release"].update(anonymous_download_verified=False),
        lambda r: r["release"].update(anonymous_download_verified=1),
        lambda r: r["release"].update(gpu_used=True),
        lambda r: r["release"].update(gpu_used=0),
        lambda r: r["release"].update(experiment_id="real_text"),
        lambda r: r["release"]["public_manifest"].update(id="real_text"),
        lambda r: r["release"].update(repo="other/public"),
        lambda r: r["release"]["public_manifest"].update(repo="other/public"),
        lambda r: r["release"].update(revision="main"),
        lambda r: r["release"]["public_manifest"].update(revision="6" * 40),
        lambda r: r["release"].update(weights_revision="main"),
        lambda r: r["approval"].update(approval_git_revision="main"),
        lambda r: r["approval"].update(approval_sha256="0" * 64),
        lambda r: r["approval"].update(files_verified=1),
        lambda r: r["approval"]["private_source"].update(revision="9" * 40),
        lambda r: r["release"]["public_manifest"]["files"][0].update(path="README.md"),
        lambda r: r["release"]["public_manifest"]["files"][0].update(output="../README.md"),
        lambda r: r["release"]["public_manifest"]["files"][0].update(sha256="bad"),
        lambda r: r["release"]["public_manifest"]["files"][0].update(bytes=-1),
        lambda r: r["release"]["public_manifest"]["files"][0].update(bytes=True),
        lambda r: r["release"]["public_manifest"]["files"].append(
            copy.deepcopy(r["release"]["public_manifest"]["files"][0])
        ),
        lambda r: r["release"]["public_manifest"]["files"][-1].update(sha256="0" * 64),
        lambda r: r["release"]["public_manifest"]["inference"].update(prompt="altered"),
    ],
)
def test_failed_mismatched_and_unsafe_results_do_not_write_outputs(receipts, mutation):
    create, ingest, manifest, proofs, _blobs, _invocations = receipts
    result, _approval, _name = create()
    mutation(result)
    with pytest.raises(ValueError):
        ingest(result)
    assert not manifest.exists() and not proofs.exists()


@pytest.mark.parametrize("missing", ["LICENSE", "README.md", "export-manifest.json", "model.pt", "tokenizer.json"])
def test_all_mandatory_and_selected_artifacts_must_be_present(receipts, missing):
    create, ingest, manifest, proofs, _blobs, _invocations = receipts
    result, _approval, _name = create()
    result["release"]["public_manifest"]["files"] = [
        item for item in result["release"]["public_manifest"]["files"] if item["output"] != missing
    ]
    with pytest.raises(ValueError, match="mandatory|approved artifact"):
        ingest(result)
    assert not manifest.exists() and not proofs.exists()


@pytest.mark.parametrize(
    "change", [{"approved": False}, {"reviewed": False}, {"pku_derived": True}, {"batch_id": "different"}]
)
def test_hashed_git_blob_still_requires_root_review_and_correct_private_scope(receipts, change):
    create, ingest, _manifest, _proofs, _blobs, _invocations = receipts
    result, _approval, _name = create(approval_change=change)
    with pytest.raises(ValueError):
        ingest(result)


def test_unreviewed_extra_artifact_and_missing_git_blob_are_rejected(receipts):
    create, ingest, _manifest, _proofs, blobs, _invocations = receipts
    result, _approval, blob_name = create()
    extra = {
        "path": "course/course-v1/text_foundation/secret.json",
        "output": "secret.json",
        "sha256": "f" * 64,
        "bytes": 16,
    }
    result["release"]["public_manifest"]["files"].append(extra)
    with pytest.raises(ValueError, match="未經 root"):
        ingest(result)
    result["release"]["public_manifest"]["files"].pop()
    del blobs[blob_name]
    with pytest.raises(ValueError, match="不使用工作目錄"):
        ingest(result)


def test_additions_follow_plan_and_exact_repeat_is_idempotent(receipts):
    create, ingest, manifest, proofs, _blobs, _invocations = receipts
    later, _approval, _name = create("real_text")
    earlier, _approval, _name = create("text_foundation")
    ingest(later)
    ingest(earlier)
    assert [item["id"] for item in json.loads(manifest.read_text())["models"]] == ["text_foundation", "real_text"]
    proof_path = proofs / "text_foundation.json"
    mtimes = manifest.stat().st_mtime_ns, proof_path.stat().st_mtime_ns
    response = ingest(earlier)
    assert response["status"] == "unchanged"
    assert mtimes == (manifest.stat().st_mtime_ns, proof_path.stat().st_mtime_ns)


def test_new_sha_requires_explicit_replace_and_updates_both_outputs(receipts):
    create, ingest, manifest, proofs, _blobs, _invocations = receipts
    original, _approval, _name = create()
    ingest(original)
    before = manifest.read_bytes(), (proofs / "text_foundation.json").read_bytes()
    replacement, _approval, _name = create(revision="6" * 40)
    with pytest.raises(ValueError, match="--replace"):
        ingest(replacement)
    assert before == (manifest.read_bytes(), (proofs / "text_foundation.json").read_bytes())
    ingest(replacement, replace=True)
    assert json.loads(manifest.read_text())["models"][0]["revision"] == "6" * 40
    assert json.loads((proofs / "text_foundation.json").read_text())["release"]["revision"] == "6" * 40


def test_index_proof_keeps_only_fixed_sha_and_not_reviewed_markdown(receipts):
    create, ingest, _manifest, proofs, _blobs, _invocations = receipts
    content = "# Public index\n\nReviewed model table.\n"
    index = {"content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    result, _approval, _name = create(approval_change={"repository_index": index})
    result["release"].update(
        root_index_revision="7" * 40, root_index_sha256=index["sha256"], root_index_anonymous_download_verified=True
    )
    ingest(result)
    proof = json.loads((proofs / "text_foundation.json").read_text())
    assert proof["release"]["root_index_sha256"] == index["sha256"]
    assert content not in (proofs / "text_foundation.json").read_text()
