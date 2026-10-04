"""Prove direct review evidence checks out without ignored outputs or weights."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import torch

from scripts.check_technical_reviews import check

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_17_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((ROOT / "docs/technical-reviews/13.17.json").read_text())
    evidence = report["artifacts"] + [item for item in report["sources"] if item["kind"] == "repository_code"]
    expected = {item["path"]: item["sha256"] for item in evidence}
    paths = sorted(expected)
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "-v", "--", *paths], cwd=ROOT, capture_output=True, text=True, check=False)
    assert ignored.returncode == 1 and not ignored.stdout, ignored.stdout
    index_before = sha(ROOT / ".git/index")
    with tempfile.TemporaryDirectory(prefix=PREFIX + "checkout_", dir="/tmp") as temporary:
        temp = Path(temporary)
        checkout = temp / "checkout"
        checkout.mkdir()
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(temp / "temporary-index")

        def git(*arguments):
            result = subprocess.run(["git", *arguments], cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
            assert result.returncode == 0, result.stderr

        git("read-tree", "HEAD")
        support = []
        for pattern in ("course/chapters/*.md", "tiny_perceptron/*.py", "scripts/course_experiments/*.py"):
            support.extend(str(path.relative_to(ROOT)) for path in ROOT.glob(pattern))
        support.extend([
            "course/first-steps.md", "course/README.md", "course/training.md", "course/glossary.md",
            "scripts/check_technical_reviews.py", "docs/technical-reviews/13.17.json",
        ])
        support.extend(report["figure_sha256"])
        # scripts/ and scripts/course_experiments/ are Python namespace packages.
        # They have no __init__.py; only existing files are added to the temporary index.
        archives = [str(path.relative_to(ROOT)) for path in OUT.glob(PREFIX + "*") if path.is_file()]
        checkout_paths = sorted(set(paths + support + archives))
        assert all((ROOT / path).is_file() for path in checkout_paths)
        git("add", "--", *checkout_paths)
        git("checkout-index", f"--prefix={checkout}/", "--", *checkout_paths)
        copied = []
        for path in paths:
            assert sha(checkout / path) == expected[path], path
            copied.append({"path": path, "sha256": expected[path], "checkout_available": True, "ignored": False})
        assert not (checkout / "outputs").exists()
        assert not list(checkout.rglob("*.pt"))
        audit_environment = os.environ.copy()
        audit_environment["PYTHONPATH"] = str(checkout)
        command = [str(ROOT / ".venv/bin/python"), "docs/technical-reviews/artifacts/fact_v2_13_17_audit.py"]
        audit = subprocess.run(command, cwd=checkout, env=audit_environment, capture_output=True, text=True, check=False)
        assert audit.returncode == 0, (audit.stdout, audit.stderr)
        assert (checkout / "docs/technical-reviews/artifacts/fact_v2_13_17_audit_result.json").read_bytes() == (OUT / (PREFIX + "audit_result.json")).read_bytes()
        result = check(root=checkout, lessons=["13.17"])
        assert not result["failures"], result
        index_after = sha(ROOT / ".git/index")
        assert index_after == index_before
        receipt = {
            "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_17_checkout_verify.py",
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
            "direct_dependencies": copied,
            "direct_dependency_count": len(copied),
            "ignore_check": {"command": ["git", "check-ignore", "--no-index", "-v", "--", *paths], "exit_code": ignored.returncode, "stdout": ignored.stdout},
            "checkout_audit": {"command": command, "exit_code": audit.returncode, "stdout": audit.stdout, "stderr": audit.stderr, "result_bytes_identical": True, "ignored_outputs_present": False, "pt_files_present": False},
            "checkout_checker": result,
            "shared_git_index_before": index_before,
            "shared_git_index_after": index_after,
            "shared_git_index_unchanged": True,
            "publication_scope": "Temporary index proves current new unignored evidence can be Git-added and checked out; not a repository commit or push. New review files still need inclusion in the coordinator final commit.",
            "initial_probe_correction": "Initial temporary-index support list included nonexistent scripts/__init__.py; corrected to existing namespace-package files. Real index and repository code remained unchanged.",
            "result": "All 22 direct artifacts and 5 repository sources checked out with exact reviewed SHAs; clean audit and lesson checker passed without ignored outputs or .pt files; real index unchanged.",
        }
    (OUT / (PREFIX + "checkout_closure.json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(receipt["result"])
    print("checkout checker:", result)


if __name__ == "__main__":
    main()
