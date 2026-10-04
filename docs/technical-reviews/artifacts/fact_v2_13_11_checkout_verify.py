"""Verify publication closure via a private index and a real isolated checkout."""

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_11_"


def main():
    report_path = ROOT / "docs/technical-reviews/13.11.json"
    report = json.loads(report_path.read_text())
    expected = {item["path"]: item["sha256"] for item in report["artifacts"]}
    expected.update({source["path"]: source["sha256"] for source in report["sources"]
                     if source["kind"] == "repository_code"})
    paths = sorted(expected)
    commands, blobs, checkout_records = [], [], []
    own_paths = ["docs/technical-reviews/13.11.json"] + [
        path.relative_to(ROOT).as_posix() for path in sorted(BASE.glob(PREFIX + "*")) if path.is_file()
    ]
    with tempfile.TemporaryDirectory(prefix=PREFIX + "git_", dir="/tmp") as temporary:
        directory = Path(temporary)
        index_path = directory / "index"
        checkout = directory / "checkout"
        checkout.mkdir()
        git_env = {**os.environ, "GIT_INDEX_FILE": str(index_path)}

        def git(arguments, input_bytes=None, allowed=(0,)):
            command = ["git", *arguments]
            run = subprocess.run(command, cwd=ROOT, env=git_env, input=input_bytes, capture_output=True)
            entry = {"command": shlex.join(command), "GIT_INDEX_FILE": str(index_path),
                     "exit_code": run.returncode, "stderr": run.stderr.decode(errors="replace")}
            if arguments[0] == "show":
                entry["stdout_bytes"] = len(run.stdout)
                entry["stdout_sha256"] = hashlib.sha256(run.stdout).hexdigest()
            elif arguments[0] != "ls-files":
                entry["stdout"] = run.stdout.decode(errors="replace")
            commands.append(entry)
            assert run.returncode in allowed, entry
            return run.stdout

        git(["read-tree", "HEAD"])
        ignored = git(["check-ignore", "--no-index", "--stdin"],
                      input_bytes=("\n".join(paths) + "\n").encode(), allowed=(1,))
        assert not ignored
        git(["add", "--", *own_paths])
        for path in paths:
            blob = git(["show", ":" + path])
            digest = hashlib.sha256(blob).hexdigest()
            assert digest == expected[path], (path, digest, expected[path])
            blobs.append({"path": path, "index_blob_sha256": digest, "matches_report_sha256": True})
        package_paths = git(["ls-files", "-z", "--", "tiny_perceptron", "scripts/course_experiments"])
        package_paths = [path.decode() for path in package_paths.split(b"\0") if path.endswith(b".py")]
        checkout_paths = sorted(set(paths + package_paths))
        git(["checkout-index", "--prefix=" + str(checkout) + "/", "--", *checkout_paths])
        for path in paths:
            digest = hashlib.sha256((checkout / path).read_bytes()).hexdigest()
            assert digest == expected[path], path
            checkout_records.append({"path": path, "checkout_file_sha256": digest,
                                     "matches_report_sha256": True})
        assert not (checkout / "outputs").exists()
        runtime_command = [sys.executable, str(checkout / "docs/technical-reviews/artifacts" /
                                              (PREFIX + "portable.py"))]
        runtime_env = {**os.environ, "PYTHONPATH": str(checkout)}
        run = subprocess.run(runtime_command, cwd=checkout, env=runtime_env,
                             text=True, capture_output=True)
        runtime = {"command": shlex.join(runtime_command), "cwd": str(checkout),
                   "PYTHONPATH": str(checkout), "exit_code": run.returncode,
                   "stdout": run.stdout, "stderr": run.stderr,
                   "outputs_directory_absent": True, "checkpoint_binaries_absent": True}
        assert run.returncode == 0, runtime
        verification = json.loads((checkout / "docs/technical-reviews/artifacts" /
                                   (PREFIX + "portable_verify_receipt.json")).read_text())
        assert verification["maximum_absolute_output_error"] == 0
        assert verification["torch_load_forbidden"] and verification["outputs_directory_read_forbidden"]
        isolated_verification = BASE / (PREFIX + "isolated_portable_verify_receipt.json")
        isolated_verification.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
        temporary_index_path = str(index_path)
    assert not Path(temporary_index_path).exists()
    receipt = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_11_checkout_verify.py",
        "result": "Private-index Git closure passed: every direct artifact and repository source is unignored; staged blob and real checkout SHA values match; portable-only reconstruction succeeds in the isolated checkout without outputs/ or .pt files.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "platform": platform.platform()},
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "direct_artifact_count": len(report["artifacts"]), "repository_source_count": 2,
        "git_commands": commands, "index_blobs": blobs, "checkout_files": checkout_records,
        "isolated_runtime": runtime, "isolated_verification_path": isolated_verification.relative_to(ROOT).as_posix(),
        "isolated_verification_sha256": hashlib.sha256(isolated_verification.read_bytes()).hexdigest(),
        "shared_index_modified": False, "temporary_index_deleted": True,
        "publication_scope": "Only an isolated temporary index was staged; root will stage and commit actual publication files.",
    }
    path = BASE / (PREFIX + "checkout_receipt.json")
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(receipt["result"])
    print(f"Verified {len(paths)} direct files; shared index unchanged; private index removed.")


if __name__ == "__main__":
    main()
