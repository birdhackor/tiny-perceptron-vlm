"""核對新增還原來源的舊身分，並在獨立 process 跑完整與移除新增項目的真實重放。"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(root):
    spec = importlib.util.spec_from_file_location("reviewed_replay", root / "docs/review-tools/baseline_replay.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def safe_file(root, name):
    pure = PurePosixPath(name)
    assert not pure.is_absolute() and all(p not in {"", ".", ".."} for p in name.split("/"))
    current = root
    for part in pure.parts:
        current = current / part
        assert not current.is_symlink(), name
    assert current.resolve().is_relative_to(root) and current.is_file(), name
    return current.read_bytes()


def tracked_state(root, env):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, env=env).split(b"\0")
    result = {}
    for raw_name in sorted(n for n in names if n):
        name = os.fsdecode(raw_name)
        path = root / name
        info = path.lstat()
        linked = stat.S_ISLNK(info.st_mode)
        raw = os.fsencode(os.readlink(path)) if linked else path.read_bytes()
        result[name] = {"sha256": sha(raw), "mode": stat.S_IMODE(info.st_mode), "symlink": linked}
    return result


def state_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def worker(args):
    entries = json.loads(safe_file(args.root, "docs/course-review-20261019/baseline-supplements.json"))["files"]
    if args.case == "without_added":
        entries = [entry for entry in entries if entry["live_path"] != "scripts/check_technical_reviews.py"]
    result = module(args.root).replay(args.root, entries)
    print(json.dumps({"status": result["status"], "failures": result["failures"],
                      "record_count": len(result.get("records", [])), "supplement_count": len(entries)}, ensure_ascii=False))
    return bool(result["failures"])


def controller(args):
    root = args.root.resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_SYSTEM=os.devnull, GIT_LFS_SKIP_SMUDGE="1")
    references = []

    def read_reference(reference):
        raw = safe_file(root, reference["path"])
        assert sha(raw) == reference["sha256"]
        historic = subprocess.check_output(["git", "show", "cc6afc5d:" + reference["path"]], cwd=root, env=env)
        assert historic == raw
        references.append({"path": reference["path"], "sha256": sha(raw), "matches_preserved_history": True})
        return json.loads(raw)

    policy_name = "docs/course-publish-20261008/review-manifest.json"
    policy_raw = safe_file(root, policy_name)
    policy = read_reference({"path": policy_name, "sha256": sha(policy_raw)})
    index = read_reference(policy["archive_index"])
    catalog = read_reference(policy["catalog"])
    inputs = read_reference(policy["current_technical_inputs"])
    reference = catalog["manifest_references"]["freeze01"]
    aliases = [entry for entry in index["artifacts"] if entry.get("original_path") == reference["path"]]
    assert len(aliases) == 1 and aliases[0]["sha256"] == reference["sha256"]
    freeze = read_reference({"path": aliases[0]["path"], "sha256": reference["sha256"]})
    entries = json.loads(safe_file(root, "docs/course-review-20261019/baseline-supplements.json"))["files"]
    added = [entry for entry in entries if entry["live_path"] == "scripts/check_technical_reviews.py"]
    assert len(added) == 1
    entry = added[0]
    expected_path = (PurePosixPath(aliases[0]["path"]).parent / "freeze/implementation" / entry["live_path"]).as_posix()
    source_raw = safe_file(root, entry["path"])
    historical_source = subprocess.check_output(["git", "show", "cc6afc5d:" + entry["path"]], cwd=root, env=env)
    tracking = subprocess.check_output(["git", "ls-files", "--error-unmatch", "--", entry["path"]], cwd=root, env=env, text=True).strip()
    expected_sha = freeze["implementation_sha256"][entry["live_path"]]
    assert entry["path"] == expected_path and sha(source_raw) == entry["sha256"] == expected_sha
    assert source_raw == historical_source and tracking == entry["path"]
    other_identity = inputs["files_sha256"].get(entry["live_path"])
    assert other_identity is None or other_identity == expected_sha
    identity = {"entry": entry, "expected_snapshot_path": expected_path, "source_sha256": sha(source_raw),
                "old_implementation_sha256": expected_sha, "old_technical_input_sha256": other_identity,
                "current_live_sha256": sha(safe_file(root, entry["live_path"])),
                "source_is_tracked": True, "source_and_parents_are_not_symlinks": True,
                "source_is_regular_file_inside_repo": True, "source_matches_preserved_history": True,
                "reference_chain": references, "alias_original_path": reference["path"],
                "checked_restore_count": len(module(root).checked_restores(root, entries))}
    initial = tracked_state(root, env)
    initial_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, env=env, text=True)
    assert not initial_status
    cases = []
    for case in ["all", "without_added"]:
        before = tracked_state(root, env)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--root", str(root), "--case", case]
        done = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        after = tracked_state(root, env)
        observed = json.loads(done.stdout)
        expected = {"status": "passed_metadata_checks", "failures": []} if case == "all" else {
            "status": "failed", "failures": ["current implementation drift: scripts/check_technical_reviews.py"]}
        matches = done.returncode == (0 if case == "all" else 1) and all(observed[key] == value for key, value in expected.items())
        cases.append({"case": case, "command": shlex.join(command), "cwd": str(root), "exit_code": done.returncode,
                      "stdout": done.stdout, "stderr": done.stderr, "expected": expected, "matches_expected": matches,
                      "tracked_before_sha256": state_sha(before), "tracked_after_sha256": state_sha(after),
                      "tracked_unchanged": before == after})
    final = tracked_state(root, env)
    record = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, env=env, text=True).strip(),
              "identity": identity, "cases": cases, "tracked_count": len(initial),
              "initial_state_sha256": state_sha(initial), "final_state_sha256": state_sha(final),
              "all_tracked_unchanged": initial == final, "initial_status": initial_status,
              "final_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, env=env, text=True),
              "environment": {"python": sys.version, "platform": platform.platform(),
                              **{key: env[key] for key in ["PYTHONDONTWRITEBYTECODE", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_LFS_SKIP_SMUDGE"]}}}
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return not (all(row["matches_expected"] and row["tracked_unchanged"] for row in cases) and initial == final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "controller"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    raise SystemExit(worker(args) if args.mode == "worker" else controller(args))
