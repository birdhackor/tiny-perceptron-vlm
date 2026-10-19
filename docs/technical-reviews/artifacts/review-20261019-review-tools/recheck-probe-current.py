"""沿用原審閱者的雜湊與重放探針，獨立 process 核對現行版與錯法注入。"""

import argparse
import gzip
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from independent_replay_probe import canonical, digest, load_module, state, worker as replay_worker


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def environment(area):
    temp = area / "pytest-temp"
    temp.mkdir(exist_ok=True)
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GIT_CONFIG_GLOBAL=os.devnull,
                GIT_CONFIG_SYSTEM=os.devnull, GIT_LFS_SKIP_SMUDGE="1", TMPDIR=str(temp))


def run(command, root, env):
    done = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    return {"command": shlex.join(command), "cwd": str(root), "exit_code": done.returncode,
            "stdout": done.stdout, "stderr": done.stderr,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            **{k: env[k] for k in ["PYTHONDONTWRITEBYTECODE", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_LFS_SKIP_SMUDGE", "TMPDIR"]}}}


def record_git(root, env, *arguments):
    return run(["git", *arguments], root, env)


def boundary_worker(args):
    module = load_module(args.root / "docs/review-tools/baseline_replay.py")
    original_os, original_shutil, original_tempfile = module.os, module.shutil, module.tempfile
    calls = {"link": [], "copy2": [], "temporary_directory": [], "external_opens": []}
    armed = False

    def audit(event, values):
        if armed and event == "open" and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0]))
            if path.is_absolute() and path.resolve().is_relative_to(args.outside):
                calls["external_opens"].append(str(path))

    sys.addaudithook(audit)

    class Proxy:
        def __init__(self, module, names):
            self.module = module
            for name in names:
                setattr(self, name, self.record(name))

        def record(self, name):
            def call(*values, **options):
                calls[name].append([str(value) for value in values])
                if name == "link" and args.branch == "copy":
                    raise OSError("獨立探針模擬不能建立 hard link")
                return getattr(self.module, name)(*values, **options)
            return call

        def __getattr__(self, name):
            return getattr(self.module, name)

    module.os = Proxy(original_os, ["link"])
    module.shutil = Proxy(original_shutil, ["copy2"])

    def temporary_directory(*values, **options):
        calls["temporary_directory"].append({"args": list(values), "options": {k: str(v) for k, v in options.items()}})
        return original_tempfile.TemporaryDirectory(*values, **options)

    module.tempfile = SimpleNamespace(TemporaryDirectory=temporary_directory)
    supplements = json.loads((args.root / "docs/course-review-20261019/baseline-supplements.json").read_bytes())["files"]
    armed = True
    try:
        result = module.replay(args.root, supplements)
        observed = {"kind": "return", "result": result}
    except Exception as error:
        observed = {"kind": "exception", "type": type(error).__name__, "message": str(error)}
    finally:
        armed = False
    print(json.dumps({"observed": observed, "calls": calls}, ensure_ascii=False))
    return int(observed["kind"] != "exception" or observed.get("type") != "ValueError" or any(calls.values()))


def controller(args):
    root, output = args.root.resolve(), args.output.resolve()
    area = root.parent
    env = environment(area)
    initial = state(root, env)
    assert not record_git(root, env, "status", "--porcelain")["stdout"]
    with (output / "tracked-sha256-current.json.gz").open("xb") as stream:
        stream.write(gzip.compress(canonical(initial), mtime=0))
    metadata = {"head": record_git(root, env, "rev-parse", "HEAD")["stdout"].strip(),
                "status": record_git(root, env, "status", "--porcelain"),
                "diff": record_git(root, env, "diff", "--raw"),
                "tracked_count": len(initial), "initial_state_sha256": digest(canonical(initial)),
                "initial_symlinks": {k: v for k, v in initial.items() if v["symlink"]}}
    entries = json.loads((root / "docs/course-review-20261019/baseline-supplements.json").read_bytes())["files"]
    source = root / entries[0]["path"]
    original_source = source.read_bytes()
    expectations = {
        "none": "current implementation drift: scripts/fetch_natural_data.py",
        "fetch": "current technical input drift: docs/selftrained/v2-manifest.json",
        "both": "passed_metadata_checks",
        "wrong_identity": "補充還原清單 files[0] 的 sha256 與舊雜湊表 implementation_sha256 記的值不同：'scripts/fetch_natural_data.py'",
        "damaged_source": "補充還原清單 files[0] 的來源位元組與 sha256 不符：" + repr(entries[0]["path"]),
    }
    rows = []
    for case, expected in expectations.items():
        if case == "damaged_source":
            source_mode = source.stat().st_mode & 0o777
            source.unlink()
            with source.open("xb") as stream:
                stream.write(original_source + b"\n# isolated source damage\n")
            source.chmod(source_mode)
        before = state(root, env)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "replay", "--root", str(root), "--case", case]
        row = run(command, root, env)
        after = state(root, env)
        observed = json.loads(row["stdout"])
        if observed["kind"] == "return":
            result = observed["result"]
            actual = result["failures"][0] if result["failures"] else result["status"]
        else:
            actual = observed["message"]
        row.update(case=case, expected=expected, actual=actual,
                   matches_expected=actual == expected and row["exit_code"] == (0 if case == "both" else 1),
                   before_sha256=digest(canonical(before)), after_sha256=digest(canonical(after)),
                   tracked_unchanged=before == after,
                   setup_changed_paths=[p for p in initial if initial[p] != before[p]])
        rows.append(row)
        if case == "damaged_source":
            source.unlink()
            with source.open("xb") as stream:
                stream.write(original_source)
            source.chmod(source_mode)
        print(json.dumps({"case": case, "matches_expected": row["matches_expected"], "tracked_unchanged": before == after}, ensure_ascii=False), flush=True)
    write_json(output / "replay-results-current.json", {"metadata": metadata, "cases": rows,
               "source_damage": {"path": entries[0]["path"], "original_sha256": digest(original_source),
                                 "damaged_sha256": digest(original_source + b"\n# isolated source damage\n")},
               "final_state_sha256": digest(canonical(state(root, env))), "all_tracked_unchanged": state(root, env) == initial})

    outside = area / "external-boundary-data"
    outside.mkdir()
    boundaries = []
    for name in ["docs/course-publish-20261008/current-source-freeze.json", "docs/course-revision-20261016/baseline-overlay.json", "tests"]:
        target = root / name
        moved = outside / target.name
        origin = {p: v for p, v in initial.items() if p == name or p.startswith(name + "/")}
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=name == "tests")
        setup = {"external_origin": "從本次乾淨 clone rename，沒有生成或改寫內容", "redirected": name,
                 "symlink_target": str(moved), "origin_tracked_count": len(origin), "origin_state_sha256": digest(canonical(origin)),
                 "status": record_git(root, env, "status", "--porcelain"), "diff": record_git(root, env, "diff", "--raw")}
        try:
            for branch in ["hardlink", "copy"]:
                row = run([sys.executable, "-B", str(Path(__file__).resolve()), "boundary", "--root", str(root), "--outside", str(outside), "--branch", branch], root, env)
                observed = json.loads(row["stdout"])
                if name == "tests":
                    first = next(p for p in sorted(initial) if p.startswith("tests/"))
                    expected = f"追蹤檔 {first!r} 的上層 經過 symlink：'tests'"
                else:
                    expected = f"還原清單 {name} 經過 symlink：{name!r}"
                row.update(setup=setup, branch=branch, expected=expected,
                           matches_expected=row["exit_code"] == 0 and observed["observed"].get("message") == expected and not any(observed["calls"].values()))
                boundaries.append(row)
        finally:
            target.unlink()
            moved.rename(target)
    write_json(output / "boundary-results-current.json", {"metadata": metadata, "cases": boundaries,
               "all_tracked_restored": state(root, env) == initial,
               "final_state_sha256": digest(canonical(state(root, env)))})

    mutant = root / "outputs/independent-mutation"
    for name in ["docs/review-tools/baseline_replay.py", "docs/review-tools/incremental_revision.py", "tests/test_incremental_revision_review.py"]:
        destination = mutant / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((root / name).read_bytes())
    test = "tests/test_incremental_revision_review.py::test_tracked_symlink_alone_in_a_redirected_folder_is_rejected_by_its_parent_check"
    test_command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", test]
    control = run(test_command, mutant, env)
    path = mutant / "docs/review-tools/baseline_replay.py"
    original = path.read_text()
    needle = "    for name in names:\n        parent = PurePosixPath(name).parent.as_posix()\n"
    assert original.count(needle) == 1
    mutated = original.replace(needle, "    for name in names:\n        if (root / name).is_symlink():\n            continue\n        parent = PurePosixPath(name).parent.as_posix()\n")
    path.unlink()
    with path.open("x") as stream:
        stream.write(mutated)
    result = run(test_command, mutant, env)
    patch = run(["git", "diff", "--no-index", "--", str(root / "docs/review-tools/baseline_replay.py"), str(path)], root, env)
    write_json(output / "mutation-results-current.json", {"control": control, "mutant": result,
               "mutation_diff": patch, "original_source_sha256": digest(original.encode()),
               "mutant_source_sha256": digest(mutated.encode()),
               "caught": control["exit_code"] == 0 and "2 passed" in control["stdout"] and result["exit_code"] == 1 and "2 failed" in result["stdout"] and "DID NOT RAISE" in result["stdout"],
               "fixture_origin": "clone 內另存三個現行檔，原樣 control 後只在副本加入 symlink 葉節點跳過父路徑的錯法。", "main_checkout_unchanged": True})
    final = state(root, env)
    write_json(output / "probe-summary-current.json", {"metadata": metadata,
               "replay_cases_ok": all(row["matches_expected"] and row["tracked_unchanged"] for row in rows),
               "boundary_cases_ok": all(row["matches_expected"] for row in boundaries),
               "mutation_caught": control["exit_code"] == 0 and "2 passed" in control["stdout"] and result["exit_code"] == 1 and "2 failed" in result["stdout"],
               "tracked_file_count": len(initial), "initial_state_sha256": digest(canonical(initial)),
               "final_state_sha256": digest(canonical(final)), "all_tracked_unchanged": final == initial,
               "final_status": record_git(root, env, "status", "--porcelain")})
    return not (all(row["matches_expected"] and row["tracked_unchanged"] for row in rows) and all(row["matches_expected"] for row in boundaries) and final == initial and result["exit_code"] == 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["controller", "replay", "boundary"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--outside", type=Path)
    parser.add_argument("--branch")
    args = parser.parse_args()
    raise SystemExit(controller(args) if args.mode == "controller" else replay_worker(args) if args.mode == "replay" else boundary_worker(args))
