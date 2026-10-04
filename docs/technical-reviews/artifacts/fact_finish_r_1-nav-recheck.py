"""Recheck R.1's shared warmup navigation after the 14-route expansion."""

import hashlib
import json
import platform
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_r_1-nav-recheck"
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup  # noqa: E402

from scripts.check_technical_reviews import sections  # noqa: E402
from scripts.export_course import DOCUMENTS, reading_markdown  # noqa: E402


def digest(data):
    return hashlib.sha256(data).hexdigest()


def flatten(items):
    for item in items:
        for label, value in item.items():
            if isinstance(value, list):
                yield from flatten(value)
            else:
                yield label, value


def main():
    report_path = BASE / f"{PREFIX}-baseline-report.json"
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    prior_nav_source = next(source for source in report["sources"] if source["id"] == "s_nav")
    before_bytes = (BASE / f"{PREFIX}-before.toml").read_bytes()
    current_bytes = (BASE / f"{PREFIX}-zensical.toml").read_bytes()
    assert digest(before_bytes) == prior_nav_source["sha256"]
    assert current_bytes == (ROOT / "zensical.toml").read_bytes()
    before = tomllib.loads(before_bytes.decode("utf-8"))["project"]
    current = tomllib.loads(current_bytes.decode("utf-8"))["project"]
    old_entries, new_entries = list(flatten(before["nav"])), list(flatten(current["nav"]))
    additions = [entry for entry in new_entries if entry not in old_entries]
    removals = [entry for entry in old_entries if entry not in new_entries]
    expected_additions = {
        "11.14.md",
        "11.15.md",
        "11.16.md",
        "12.13.md",
        "12.14.md",
        "chapter-20.md",
        *{f"20.{i}.md" for i in range(1, 9)},
    }
    assert len(additions) == 14 and {target for _, target in additions} == expected_additions
    assert not removals
    assert [entry for entry in new_entries if entry in old_entries] == old_entries
    assert {key: value for key, value in before.items() if key != "nav"} == {
        key: value for key, value in current.items() if key != "nav"
    }
    assert before["nav"][1] == current["nav"][1]
    warmup_entries = [entry for entry in new_entries if entry[1] == "first-steps.md"]
    assert warmup_entries == [("基礎暖身與操作", "first-steps.md")]

    index_bytes = (ROOT / "course/lesson-index.json").read_bytes()
    (BASE / f"{PREFIX}-lesson-index.json").write_bytes(index_bytes)
    index = json.loads(index_bytes)
    expected_paths = {
        "index.md",
        *{f"{name}.md" for name in DOCUMENTS},
        *{f"{item['id']}.md" for item in index},
        *{f"chapter-{Path(item['source']).stem}.md" for item in index},
    }
    nav_paths = [target for _, target in new_entries]
    assert len(set(nav_paths)) == len(nav_paths) == len(expected_paths) == 296
    assert set(nav_paths) == expected_paths

    guide_path = ROOT / "course/README.md"
    guide = guide_path.read_text(encoding="utf-8")
    intro = guide[: re.search(r"^## ", guide, re.M).start()]
    r1 = dict(sections(guide_path))["R.1"]
    assert next(iter(sections(guide_path)))[0] == "R.1"
    assert r1 == (BASE / f"{PREFIX}-source-R.1.md").read_text(encoding="utf-8")
    assert intro == (BASE / f"{PREFIX}-intro.md").read_text(encoding="utf-8")
    assert digest(r1.encode()) == report["source_sha256"]
    assert digest(intro.encode()) == report["intro_sha256"]
    for source in report["sources"]:
        if source["kind"] == "repository_code" and source["id"] != "s_nav":
            assert digest((ROOT / source["path"]).read_bytes()) == source["sha256"]
    dependent_claims = [
        claim["id"]
        for claim in report["claims"]
        if any(evidence["source_id"] == "s_nav" for evidence in claim["evidence"])
    ]
    assert dependent_claims == ["c4"]
    source_path = ROOT / "course/chapters/01.md"
    transformed = reading_markdown(
        r1,
        guide_path,
        {(ROOT / "course/first-steps.md").resolve(): "first-steps.md"},
        {source_path.resolve(): {"1.1": "1.1.md"}},
        "main",
    )
    assert all(
        target in transformed
        for target in ["(1.1.md)", "(first-steps.md)", "(first-steps.md#W.1)", "(first-steps.md#W.2)"]
    )
    public_nav_receipts = []
    for label in ["site-1.1", "site-guide", "site-warmup"]:
        snapshot = BASE / f"fact_finish_r_1-{label}.html"
        soup = BeautifulSoup(snapshot.read_text(encoding="utf-8"), "html.parser")
        links = [
            a["href"]
            for a in soup.select("nav a[href]")
            if urljoin(current["site_url"] + "1.1.html", a["href"]) == current["site_url"] + "first-steps.html"
        ]
        assert links
        public_nav_receipts.append(
            {
                "snapshot": snapshot.relative_to(ROOT).as_posix(),
                "sha256": digest(snapshot.read_bytes()),
                "warmup_nav_hrefs": links,
            }
        )

    command = ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_r_1-nav-recheck.py"
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "CPU parsing and link verification; no model execution",
    }
    result = "All assertions passed: 14 added routes, 0 removals, 296 unique nav destinations match the current export inventory; shared warmup entry and R.1 links are unchanged."
    audit = {
        "reviewer_task": "/root/fact_finish_r_1",
        "rechecked_at": datetime.now(UTC).isoformat(),
        "command": command,
        "environment": environment,
        "prior_report_sha256": digest(original_report),
        "old_nav_sha256": digest(before_bytes),
        "current_nav_sha256": digest(current_bytes),
        "source_sha256": digest(r1.encode()),
        "intro_sha256": digest(intro.encode()),
        "added_nav_entries": additions,
        "removed_nav_entries": removals,
        "old_nav_count": len(old_entries),
        "new_nav_count": len(new_entries),
        "export_inventory_paths": sorted(expected_paths),
        "unique_nav_and_export_inventory_equal": True,
        "starting_reading_group_unchanged": True,
        "other_project_settings_unchanged": True,
        "warmup_entry": warmup_entries,
        "dependent_claim_ids": dependent_claims,
        "existing_public_nav_snapshots_checked": public_nav_receipts,
        "inspection": "本人完整重讀R.1/導言、完整現有nav和實際git diff；基線bytes正好等於本人前次s_nav SHA。只讀index路徑metadata，不讀第20章或finaltest品質輸出。新增14項沒有改既有入口、順序、開始閱讀群組或其他project設定。",
        "scope": "Current working configuration and export inventory are checked. Earlier public-page snapshots establish their previously fetched navigation only; this recheck does not claim the new routes are deployed, or assess their lesson content or model quality. No Colab cloud kernel or GPU run.",
        "result": result,
    }
    audit_path = BASE / f"{PREFIX}-execution.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
