"""彙整逐例比較、網站完整擴充設定的反例，以及審閱前後完整性收據。"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_result(path):
    data = json.loads(path.read_bytes())
    assert data["exit_code"] == 0, path
    return json.loads(data["stdout"])


def main(args):
    repo = args.repo.resolve()
    art = repo / "docs/technical-reviews/artifacts/review-20261019-review-tools"
    provenance = json.loads((art / "provenance-current-04.json").read_bytes())
    previous = read_result(art / "author-patch-results-current-03.json")
    current = read_result(art / "author-patch-results-current-04.json")
    old = {row["case"]: row for row in previous["cases"]}
    permitted = {"plain", "link_url", "link_label", "reference_link", "image_source", "html_image", "indented_code", "html_pre", "absent_current"}
    comparisons = []
    for row in current["cases"]:
        name = row["case"]
        accepted = not row["gate"]["failures"]
        comparisons.append({"case": name, "previous_status": old[name]["gate"]["status"], "current_status": row["gate"]["status"],
                            "status_changed": old[name]["gate"]["status"] != row["gate"]["status"],
                            "current_policy_should_accept": name in permitted, "current_policy_matches": accepted == (name in permitted),
                            "preflight_failing": row["preflight_failing"], "issue_inventory_failing": row["issue_inventory_failing"],
                            "absent_field_results_equal": row.get("absent_field_results_equal")})
    assert len(comparisons) == 23 and all(row["current_policy_matches"] for row in comparisons)
    admission = read_result(art / "fence-admission-results-current-04.json")
    unsafe_names = {"nbsp_backtick", "nbsp_tilde", "emspace_backtick", "invalid_backtick_info", "invalid_superfences_options"}
    unsafe = [row for row in admission["cases"] if row["case"] in unsafe_names]
    assert len(unsafe) == 5 and all(not r["gate"]["failures"] and not r["preflight_failing"] and not r["issue_inventory_failing"] for r in unsafe)
    sys.path[:0] = ["/Users/yclin/.cache/uv/archive-v0/_s1u9QcvnBYGXI-2", "/Users/yclin/.cache/uv/archive-v0/4qFlT7bxJOvyO4qC"]
    import markdown
    config = tomllib.loads((repo / "zensical.toml").read_text())["project"]["markdown_extensions"]
    extensions = {name: value for name, value in config.items() if name != "pymdownx"}
    extensions.update({"pymdownx." + name: value for name, value in config["pymdownx"].items()})
    spec = importlib.util.spec_from_file_location("fence_support", art / "fence-semantics-probe-current-04.py")
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    full_site = []
    for row in unsafe:
        html = markdown.markdown(row["input"]["current_body"], extensions=list(extensions), extension_configs=extensions)
        blocks = support.Blocks()
        blocks.feed(html)
        in_code = any("new-token" in text for text in blocks.blocks)
        assert in_code
        full_site.append({"case": row["case"], "new_in_pre": in_code, "html": html})
    files = [{"path": item["path"], "sha256": sha(repo / item["path"]), "matches_start": sha(repo / item["path"]) == item["sha256"],
              "matches_previous_report": sha(repo / item["path"]) == item["previous_sha256"]} for item in provenance["checked_files"]]
    assert all(item["matches_start"] for item in files)
    old_seals = [{"path": item["path"], "sha256": sha(repo / item["path"]), "matches": sha(repo / item["path"]) == item["sha256"]}
                 for item in provenance["old_reports"]]
    old_seals += [{"path": name, "sha256": sha(repo / name), "matches": sha(repo / name) == item["expected"]}
                  for name, item in provenance["old_artifacts"].items()]
    assert all(item["matches"] for item in old_seals)
    commands = []
    for cmd in [["git", "status", "--short", "-uno"], ["git", "diff", "b62ec6e7", "HEAD", "--", "docs/review-tools/incremental_revision.py", "tests/test_incremental_revision_review.py", "docs/technical-review-guide.md"]]:
        result = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
        commands.append({"command": cmd, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        assert result.returncode == 0
    snapshots = []
    for name in ("docs/review-tools/incremental_revision.py", "tests/test_incremental_revision_review.py", "docs/technical-review-guide.md"):
        raw = subprocess.check_output(["git", "show", "b62ec6e7:" + name], cwd=repo)
        snapshots.append({"path": name, "matches_reviewed_commit": raw == (repo / name).read_bytes()})
    assert all(row["matches_reviewed_commit"] for row in snapshots)
    print(json.dumps({"created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "policy": {"guide_path": "docs/technical-review-guide.md", "guide_sha256": sha(repo / "docs/technical-review-guide.md"),
                   "interpretation": "原有圖、連結、縮排程式與 HTML pre 的內容責任已明列交由作者；不因這些例子通過而重提必要問題。程式圍欄仍承諾攔住。"},
        "previous_probe": {"cases": comparisons, "changed_cases": [r["case"] for r in comparisons if r["status_changed"]],
                           "old_violation_flags": "舊探針的 expected_pass／violation 依舊規約寫死；本次依 current_policy_should_accept 重新解讀，不把七個現在允許的案例算成漏洞。"},
        "previous_issues": [{"id": "AUTHOR-PATCH-MARKUP", "resolved": True, "reason": "規約及註解現在明列交由作者負責；五個原有案例結果不變。"},
                            {"id": "AUTHOR-PATCH-CODEBLOCK", "resolved": True, "reason": "原有縮排／引用假收尾現在都拒絕；縮排 code／HTML pre 現已明列為不涵蓋。這次兩種不同失敗原因另立新 issue，不擴張舊 issue 的身分。"}],
        "fence_semantics": {"cases": 38, "unsafe_cases": read_result(art / "fence-semantics-results-current-04.json")["unsafe_cases"],
                            "safe_differences": ["tab 展開後等寬的收尾，網站已關閉，守衛仍保守拒絕。", "CommonMark 在 quote 容器結束時關閉區塊，掃描器仍保留舊 opener。"],
                            "normative_url": "https://spec.commonmark.org/0.31.2/#fenced-code-blocks",
                            "official_parser_documentation": "https://markdown-it-py.readthedocs.io/en/latest/using.html#the-parser",
                            "web_receipt": {"tool": "web.run", "actions": ["open CommonMark 0.31.2", "find closing code fence", "open markdown-it-py Using markdown_it"],
                                            "observed": "CommonMark 只准收尾後接 space/tab，反引號 info string 不能含反引號；markdown-it-py 官方將 commonmark preset 定義為嚴格 CommonMark。"}},
        "full_site_extensions": {"config": extensions, "cases": full_site},
        "admission": {"accepted_unsafe_cases": sorted(unsafe_names), "scope": admission["fixture"]},
        "final_integrity": {"checked_files": files, "old_seals": old_seals, "commands": commands, "commit_snapshots": snapshots}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    main(parser.parse_args())
