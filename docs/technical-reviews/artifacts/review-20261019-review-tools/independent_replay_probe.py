"""獨立重放對照：每種情況用新 process，保存完整輸出並核對全部追蹤檔。"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import stat
import subprocess
import sys
import traceback
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("independent_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(root, env):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, env=env)
    result = {}
    for name in sorted(os.fsdecode(n) for n in names.split(b"\0") if n):
        path = root / name
        info = path.lstat()
        link = stat.S_ISLNK(info.st_mode)
        raw = os.fsencode(os.readlink(path)) if link else path.read_bytes()
        result[name] = {"sha256": digest(raw), "mode": stat.S_IMODE(info.st_mode), "symlink": link}
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def worker(args):
    root = args.root.resolve()
    entries = json.loads((root / "docs/course-review-20261019/baseline-supplements.json").read_bytes())["files"]
    supplements = [] if args.case in {"none", "previous_none"} else entries[:1] if args.case == "fetch" else entries
    if args.case == "wrong_identity":
        supplements = [dict(entries[0], path=entries[1]["path"], sha256=entries[1]["sha256"]), entries[1]]
    path = root / "docs/review-tools/baseline_replay.py"
    if args.case == "previous_none":
        path = root / "outputs/independent-probe/previous_baseline_replay.py"
    try:
        module = load_module(path)
        result = module.replay(root) if args.case == "previous_none" else module.replay(root, supplements)
        print(json.dumps({"kind": "return", "result": result}, ensure_ascii=False))
        return bool(result["failures"])
    except Exception as error:
        print(json.dumps({"kind": "exception", "type": type(error).__name__, "message": str(error)}, ensure_ascii=False))
        traceback.print_exc()
        return 1


def controller(args):
    root, output = args.root.resolve(), args.output.resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_SYSTEM=os.devnull, GIT_LFS_SKIP_SMUDGE="1")
    python = Path(sys.executable).absolute()
    probe = Path(__file__).resolve()
    initial = state(root, env)
    clean_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, env=env, text=True)
    if clean_status:
        raise RuntimeError("clone 的追蹤內容不乾淨：" + clean_status)
    with (output / "tracked-sha256.json.gz").open("xb") as stream:
        stream.write(gzip.compress(canonical(initial), mtime=0))
    previous = root / "outputs/independent-probe/previous_baseline_replay.py"
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_bytes(subprocess.check_output(["git", "show", "cc6afc5d~1:docs/review-tools/baseline_replay.py"], cwd=root, env=env))
    entries = json.loads((root / "docs/course-review-20261019/baseline-supplements.json").read_bytes())["files"]
    source = root / entries[0]["path"]
    original_source = source.read_bytes()
    expectations = {
        "none": "current implementation drift: scripts/fetch_natural_data.py",
        "fetch": "current technical input drift: docs/selftrained/v2-manifest.json",
        "both": "passed_metadata_checks",
        "wrong_identity": "補充還原清單 files[0] 的 sha256 與舊雜湊表 implementation_sha256 記的值不同：'scripts/fetch_natural_data.py'",
        "damaged_source": "補充還原清單 files[0] 的來源位元組與 sha256 不符：" + repr(entries[0]["path"]),
        "previous_none": "current implementation drift: scripts/fetch_natural_data.py",
    }
    rows = []
    for case, expected in expectations.items():
        if case == "damaged_source":
            source.write_bytes(original_source + b"\n# isolated source damage\n")
        before = state(root, env)
        command = [str(python), "-B", str(probe), "worker", "--root", str(root), "--case", case]
        done = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        after = state(root, env)
        try:
            observed = json.loads(done.stdout)
        except ValueError:
            observed = None
        if observed and observed["kind"] == "return":
            result = observed["result"]
            actual = result["status"] if not result["failures"] else result["failures"][0]
            record_count = len(result.get("records", []))
        else:
            actual = observed.get("message") if observed else "無法解析輸出"
            record_count = None
        row = {
            "case": case, "command": shlex.join(command), "cwd": str(root),
            "exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            **{k: env[k] for k in ["PYTHONDONTWRITEBYTECODE", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_LFS_SKIP_SMUDGE"]}},
            "expected": expected, "actual": actual,
            "matches_expected": actual == expected and done.returncode == (0 if case == "both" else 1),
            "record_count": record_count, "tracked_file_count": len(before),
            "tracked_state_before_sha256": digest(canonical(before)),
            "tracked_state_after_sha256": digest(canonical(after)),
            "tracked_changes_from_replay": [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)],
            "setup_changes_from_clean_clone": [p for p in initial.keys() | before.keys() if initial.get(p) != before.get(p)],
        }
        write_json(output / ("replay-" + case + ".json"), row)
        rows.append({k: v for k, v in row.items() if k not in {"stdout", "stderr", "environment"}})
        if case == "damaged_source":
            source.write_bytes(original_source)
        print(json.dumps({"case": case, "exit_code": done.returncode, "actual": actual,
                          "matches_expected": row["matches_expected"], "tracked_changes": row["tracked_changes_from_replay"]}, ensure_ascii=False), flush=True)
    plain = json.loads((output / "replay-none.json").read_bytes())
    prior = json.loads((output / "replay-previous_none.json").read_bytes())
    final = state(root, env)
    summary = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, env=env, text=True).strip(),
        "initial_status": clean_status, "cases": rows,
        "previous_and_current_no_supplement_results_equal": json.loads(plain["stdout"]) == json.loads(prior["stdout"]),
        "initial_state_sha256": digest(canonical(initial)), "final_state_sha256": digest(canonical(final)),
        "tracked_file_count": len(initial), "all_tracked_files_restored": initial == final,
        "source_damage_setup": {"path": entries[0]["path"], "clean_sha256": digest(original_source),
                                "damaged_sha256": digest(original_source + b"\n# isolated source damage\n")},
        "hash_list_format": "gzip JSON，symlink 的 SHA 為 link target 字串；普通檔 SHA 為完整內容，另存 mode。",
        "final_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, env=env, text=True),
    }
    write_json(output / "replay-summary.json", summary)
    return not (all(row["matches_expected"] and not row["tracked_changes_from_replay"] for row in rows)
                and initial == final and summary["previous_and_current_no_supplement_results_equal"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "controller"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    raise SystemExit(worker(args) if args.mode == "worker" else controller(args))
