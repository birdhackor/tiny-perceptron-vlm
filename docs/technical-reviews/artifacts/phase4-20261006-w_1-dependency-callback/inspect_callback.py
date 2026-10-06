"""Same original W.1 reviewer: metadata/hash/TOML callback, no task rerun."""

import copy
import difflib
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-20261005-w_1-factual-independent"
TASK = "/root/phase4_factual_coordinator/factual_w_1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def save(name, value):
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


prior = json.loads((BASE / "prior/W.1.json").read_bytes())
proof = json.loads((BASE / "prior/artifacts" / OLD.name / "cpu-proof.json").read_bytes())
assert prior["reviewer_task"] == TASK
assert sha(BASE / "prior/W.1.json") == "fdf74458956e980e7af9155dc2500c110e3f42f18e4baf3f12acba0c7b3cd769"
assert sha(BASE / "prior/artifacts" / OLD.name / "cpu-proof.json") == "d1595e3db70de2fffbac6228a7d1eb1b8ddd7dadc863d78453643033c49ecbed"
raw = (ROOT / "course/first-steps.md").read_bytes()
raw.decode("utf-8")
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
index = next(i for i, h in enumerate(headings) if h[0].startswith(b"## W.1 "))
section = raw[headings[index].start():headings[index + 1].start()]
intro = raw[:headings[0].start()]
assert digest(section) == prior["source_sha256"]
assert digest(intro) == prior["intro_sha256"]
assert prior["figure_sha256"] == {}
assert not re.search(rb"!\[[^\]]*\]\(|<img\b", section)
assert (BASE / "current/section.md").read_bytes() == section
assert (BASE / "current/intro.md").read_bytes() == intro

previous_bytes = (OLD / "frozen/pyproject.toml").read_bytes()
current_bytes = (ROOT / "pyproject.toml").read_bytes()
previous = tomllib.loads(previous_bytes.decode())
current = tomllib.loads(current_bytes.decode())
print("current TOML top keys/types:", {k: type(v).__name__ for k, v in current.items()})
pointers = {
    "/project/requires-python": current["project"]["requires-python"],
    "/project/dependencies": current["project"]["dependencies"],
    "/project/optional-dependencies": current["project"]["optional-dependencies"],
    "/dependency-groups/notebook": current["dependency-groups"]["notebook"],
    "/build-system": current["build-system"],
    "/tool/uv": current["tool"]["uv"],
    "/tool/ruff/extend-exclude": current["tool"]["ruff"]["extend-exclude"],
}
for pointer, value in pointers.items():
    print(pointer, json.dumps(value, ensure_ascii=False))
old_remainder, new_remainder = copy.deepcopy(previous), copy.deepcopy(current)
old_excludes = old_remainder["tool"]["ruff"].pop("extend-exclude")
new_excludes = new_remainder["tool"]["ruff"].pop("extend-exclude")
assert old_remainder == new_remainder
assert len(new_excludes) == len(old_excludes) + 1
assert [value for value in new_excludes if value not in old_excludes] == ["docs/course-revision-20261005"]
assert [value for value in new_excludes if value != "docs/course-revision-20261005"] == old_excludes
diff = "".join(difflib.unified_diff(previous_bytes.decode().splitlines(keepends=True),
                                  current_bytes.decode().splitlines(keepends=True),
                                  fromfile="own-original-frozen/pyproject.toml", tofile="current/pyproject.toml"))
assert diff == (BASE / "pyproject-raw.diff").read_text()

artifacts = []
for artifact in prior["artifacts"]:
    actual = sha(ROOT / artifact["path"])
    assert actual == artifact["sha256"], artifact["id"]
    artifacts.append({"artifact_id": artifact["id"], "path": artifact["path"],
                      "recorded_sha256": artifact["sha256"], "actual_sha256": actual,
                      "unchanged": True, "original_kind": artifact["kind"]})
sources = []
for source in prior["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = sha(ROOT / source["path"])
    changed = actual != source["sha256"]
    assert changed == (source["id"] == "repo_project"), source["id"]
    sources.append({"source_id": source["id"], "path": source["path"],
                    "previous_sha256": source["sha256"], "current_sha256": actual,
                    "changed": changed})
assert sha(ROOT / prior["frozen_input"]["path"]) == prior["frozen_input"]["sha256"]

versions = {name: importlib.metadata.version(name) for name in proof["environment"]["packages"]}
assert versions == proof["environment"]["packages"]
assert sys.version == proof["environment"]["python"]
assert sys.executable == proof["environment"]["python_executable"]
commands = []
for name, argv in [("uv-version", ["uv", "--version"]), ("git-version", ["git", "--version"])]:
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=10, check=False)
    assert completed.returncode == 0
    assert completed.stdout == (OLD / (name + ".stdout.txt")).read_bytes()
    (BASE / (name + ".stdout.txt")).write_bytes(completed.stdout)
    (BASE / (name + ".stderr.txt")).write_bytes(completed.stderr)
    commands.append({"argv": argv, "cwd": str(ROOT), "exit_code": completed.returncode,
                     "stdout": name + ".stdout.txt", "stdout_sha256": digest(completed.stdout),
                     "equals_original_version_stdout": True})

environment = {"python": sys.version, "python_executable": sys.executable,
               "package_metadata": versions, "device_scope": "No fresh torch import or CPU/GPU computation; same original CPU evidence retained.",
               "CLI_version_commands": commands}
save("current-environment.json", environment)
inspection = {
    "schema_version": 1, "reviewer_task": TASK, "inspected_on": "2026-10-06",
    "kind": "same-original-reviewer-narrow-dependency-version-callback",
    "current_source_sha256": digest(section), "current_intro_sha256": digest(intro),
    "current_source_snapshot": str((BASE / "current/section.md").relative_to(ROOT)),
    "current_intro_snapshot": str((BASE / "current/intro.md").relative_to(ROOT)),
    "current_whole_snapshot": {"path": str((BASE / "current/course/first-steps.md").relative_to(ROOT)),
                               "sha256": digest(raw), "scope": "Full raw file saved as this callback frozen input; personally read only actual introduction plus complete W.1."},
    "retained_original_frozen_input": prior["frozen_input"],
    "old_project_sha256": digest(previous_bytes), "current_project_sha256": digest(current_bytes),
    "current_project_snapshot": str((BASE / "current/pyproject.toml").relative_to(ROOT)),
    "raw_diff_path": str((BASE / "pyproject-raw.diff").relative_to(ROOT)),
    "raw_diff_sha256": sha(BASE / "pyproject-raw.diff"),
    "personally_inspected_TOML_pointers": pointers,
    "parsed_change": {"path": "/tool/ruff/extend-exclude", "inserted": "docs/course-revision-20261005",
                      "all_other_parsed_fields_exactly_equal": True},
    "judgment": "Only Ruff evidence-directory exclusion plus its ordinary preservation-method comment changed. CPU extra/index, notebook group, all installation dependencies/build/uv flags and product execution contracts unchanged. Original 2026-10-05 CPU execution evidence still supports W.1.",
    "contamination": {"event": False, "method_comment_classification": "Comment describes preserving version-bound review artifacts; contains no experiment outcome, old reviewer verdict or author correction answer."},
    "repository_source_hash_checks": sources, "original_artifact_hash_checks": artifacts,
    "original_CPU_proof": {"path": str((OLD / "cpu-proof.json").relative_to(ROOT)), "sha256": sha(OLD / "cpu-proof.json"),
                           "retained_support": ["2026-10-05 git/uv version and offline dry-run", "original activation/direct check_env CPU execution", "isolated kernel registration and Lab HTTP behavior", "original and bird-input complete notebook runs in separate fresh kernels", "same-kernel old derived state and indentation probes"],
                           "not_reexecuted_in_callback": True},
    "environment": environment,
    "limits": {"full_installation_or_notebook_CPU_rerun": False, "GPU_training_or_new_model_data_download": False,
               "source_or_figure_changes": False, "new_reviewer_or_reader_full_session": False,
               "online_source_refetch": False, "prior_official_sources": "Retained original versioned HTTPS snapshots and inspection; all recorded snapshot artifact SHA exact-match verified."},
}
save("current-inspection.json", inspection)
print("All other parsed TOML fields exactly equal: True")
print("Repository sources: only repo_project raw file changed; every other cited repository source exact SHA match")
print("All", len(artifacts), "original registered artifacts exact SHA match")
print("Original CPU environment package metadata and CLI version stdout exactly match; no task execution rerun")
print("Current inspection SHA-256:", sha(BASE / "current-inspection.json"))
