"""T.11 owner callback: fresh anonymous transport, real student CLI and one CPU example.

No old cache is used, and no training/final/research/publication entry point is called.
Only small receipts are written in the repository; downloads and weights stay in /tmp.
"""

import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent
RECEIPT = OUT / "public-delta-v2.json"


def utc():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scrub_query(text):
    return re.sub(r"https?://[^\s<>\"']+", lambda item: urllib.parse.urlunsplit((*urllib.parse.urlsplit(item[0])[:3], "", "")), text)


def save(report):
    RECEIPT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_cli(report, argv, env):
    row = {"argv": argv, "cwd": str(ROOT), "started_at": utc()}
    started = time.monotonic()
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    row.update(returncode=process.returncode, stdout=scrub_query(process.stdout), stderr=scrub_query(process.stderr), finished_at=utc(), elapsed_seconds=time.monotonic() - started)
    report["cli"].append(row)
    save(report)
    assert process.returncode == 0, f"CLI failed with exit{process.returncode}; see receipt"
    return json.loads(process.stdout)


def main():
    import torch

    assert not RECEIPT.exists(), "immutable delta receipt already exists"
    descriptor_path = ROOT / "docs/decision-product/v3/public-release.json"
    descriptor = json.loads(descriptor_path.read_text())
    current1 = ROOT / "docs/course-revision-20261101-c-v3/technical-v21/numbered_t11/current-v1.json"
    original = current1.with_name("original-v1.json")
    previous = json.loads(current1.read_text())
    assert sha(original) == "8c1b0d977f56bd8a6ba6712b631a4256dbca35533a35aec6ff608f21f70575fb"
    assert sha(current1) == "3f928669f76d58e0d4004c733ea917963993729e84a2885a94a3a6e31b3a7ba7"
    assert descriptor["sha256"] == "552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88"
    assert descriptor["bytes"] == 6168350
    temp = pathlib.Path(tempfile.mkdtemp(prefix="t11-public-delta-v2-", dir="/tmp"))
    assert list(temp.iterdir()) == []
    bundle = temp / "student-installed-bundle"
    assert not bundle.exists()
    env = os.environ.copy()
    removed = []
    for name in ("GIT_LFS_AUTHORIZATION", "GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "AUTHORIZATION", "HTTP_AUTHORIZATION"):
        if name in env:
            removed.append(name)
            env.pop(name)
    env.update(UV_PROJECT_ENVIRONMENT="/workspace/tiny-perceptron-vlm/.venv", PYTHONPATH=str(ROOT), UV_CACHE_DIR="/tmp/uv-c3", UV_OFFLINE="1")
    before = {row["path"]: sha(ROOT / row["path"]) for row in previous["sources"] if row["kind"] == "repository_code"}
    for row in previous["sources"]:
        if row["kind"] == "repository_code":
            assert before[row["path"]] == row["sha256"]
    for row in previous["artifacts"]:
        assert sha(ROOT / row["path"]) == row["sha256"]
    report = {"schema_version": 1, "reviewer_task": "/root/c_v3_technical_coord/numbered_t11", "review_kind": "owner_public_transport_delta_callback_not_new_original", "started_at": utc(), "root": str(ROOT), "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "uv": subprocess.check_output(["uv", "--version"], text=True).strip()}, "scope": "Fresh anonymous HTTP GET plus real student download CLI into a genuinely new empty output; actual judge CLI loads only those downloaded four files. One built-in example, not final/quality/training/research. Old404 and earlier reports remain valid historical events.", "temporary_directory": str(temp), "temporary_directory_started_empty": True, "bundle_path": str(bundle), "bundle_started_absent": True, "archive_and_weights_not_in_git": True, "credential_variable_names_removed_from_student_subprocess": removed, "descriptor": {"path": descriptor_path.relative_to(ROOT).as_posix(), "sha256": sha(descriptor_path), "url": descriptor["url"], "expected_archive_bytes": descriptor["bytes"], "expected_archive_sha256": descriptor["sha256"]}, "original_report": {"path": original.relative_to(ROOT).as_posix(), "sha256": sha(original)}, "previous_current": {"path": current1.relative_to(ROOT).as_posix(), "sha256": sha(current1)}, "prior_repository_source_count": len(before), "prior_artifacts_verified_unchanged": len(previous["artifacts"]), "cli": []}
    save(report)
    request = urllib.request.Request(descriptor["url"], headers={"User-Agent": "tiny-perceptron-course"})
    assert not any(name.lower() == "authorization" for name, _ in request.header_items())
    opener = urllib.request.build_opener()
    assert not any("AuthHandler" in type(handler).__name__ for handler in opener.handlers)
    archive = temp / "independent-anonymous-get.zip"
    digest = hashlib.sha256()
    count = 0
    http = {"started_at": utc(), "method": "GET", "url": descriptor["url"], "request_header_names": [name for name, _ in request.header_items()], "authorization_header_sent_by_client": False, "anonymous_default_opener": True, "cache_file_used": False, "opener_handlers": [type(handler).__name__ for handler in opener.handlers]}
    report["http"] = http
    save(report)
    with opener.open(request, timeout=90) as response, archive.open("xb") as target:
        final = urllib.parse.urlsplit(response.url)
        http.update(status=response.status, final_host=final.netloc, final_path=final.path, signed_redirect_query_recorded=False, content_length=response.headers.get("Content-Length"), content_type=response.headers.get("Content-Type"))
        while block := response.read(1024 * 1024):
            count += len(block)
            assert count <= descriptor["bytes"], "response exceeds frozen archive size"
            target.write(block)
            digest.update(block)
    http.update(finished_at=utc(), received_bytes=count, sha256=digest.hexdigest(), temporary_archive_path=str(archive), archive_reused_for_student_cli=False)
    save(report)
    assert http["status"] == 200 and count == descriptor["bytes"] and digest.hexdigest() == descriptor["sha256"]
    # Literal student module; only --output/--bundle isolate this proof from old files.
    prefix = ["uv", "run", "--extra", "cpu", "--no-sync", "python", "-m", "scripts.course_product"]
    assert not bundle.exists()
    report["student_output_absent_immediately_before_download"] = True
    download = run_cli(report, [*prefix, "download", "--output", str(bundle)], env)
    assert download["verified"] is True and pathlib.Path(download["bundle"]) == bundle
    assert {path.name for path in bundle.iterdir()} == set(descriptor["bundle"])
    installed = {}
    for name, expected in descriptor["bundle"].items():
        path = bundle / name
        assert path.is_file() and not path.is_symlink()
        installed[name] = {"bytes": path.stat().st_size, "sha256": sha(path), "matches_fixed_descriptor": sha(path) == expected}
        assert installed[name]["matches_fixed_descriptor"]
    report["fresh_installed_files"] = installed
    report["student_cli_used_independent_http_archive"] = False
    save(report)
    judged = run_cli(report, [*prefix, "judge", "--bundle", str(bundle)], env)
    assert judged["product"] == "C-v3" and judged["dispatch"] == "typed-devices" and judged["target"] == "主錄音室"
    observed = {kind: judged["results"][kind]["selected"] for kind in ("choice", "info", "rubric")}
    assert observed == {"choice": "record", "info": "yes", "rubric": "2"}
    assert judged["results"]["choice"]["status"] == "accepted"
    assert {name: sha(bundle / name) for name in descriptor["bundle"]} == {name: row["sha256"] for name, row in installed.items()}
    after = {name: sha(ROOT / name) for name in before}
    assert before == after
    for row in previous["artifacts"]:
        assert sha(ROOT / row["path"]) == row["sha256"]
    assert sha(original) == report["original_report"]["sha256"] and sha(current1) == report["previous_current"]["sha256"]
    report.update(finished_at=utc(), success=True, judge_selected=observed, judge_bundle_verified_unchanged=True, prior_repository_sources_unchanged=True, prior_artifacts_unchanged=True, original_and_current_v1_unchanged=True, fresh_training_runs=0, fresh_final_runs=0, fresh_research_runs=0, modal_runs=0, publication_actions=0, real_cli_model_examples=1)
    save(report)
    print(json.dumps({"receipt": RECEIPT.relative_to(ROOT).as_posix(), "sha256": sha(RECEIPT), "http_status": http["status"], "archive_bytes": count, "archive_sha256": http["sha256"], "installed_files": len(installed), "cli_exits": [row["returncode"] for row in report["cli"]], "judge_selected": observed, "finished_at": report["finished_at"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
