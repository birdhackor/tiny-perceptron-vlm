"""核對凍結輸入與導覽，並保存原始來源的小段摘錄及讀取收據。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(raw):
    parser = PlainText()
    parser.feed(raw)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (ARTIFACTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    snapshot_path = "docs/course-review-20261019/freeze-01/sources/chapter-0C.md"
    snapshot = (ROOT / snapshot_path).read_bytes()
    freeze_bytes = (ROOT / freeze_path).read_bytes()
    page = next(page for page in json.loads(freeze_bytes)["pages"] if page["page_id"] == "chapter-0C")
    live_path = ROOT / "course/chapters/0C.md"
    live_intro = live_path.read_bytes().split(b"\n## ", 1)[0] + b"\n"
    links = []
    for link in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", snapshot.decode()):
        path, _, anchor = link.partition("#")
        target = (live_path.parent / path).resolve()
        target_text = target.read_text()
        anchor_found = not anchor or re.search(r"^## " + re.escape(anchor) + r"(?:\s|$)", target_text, re.M) is not None
        links.append({"link": link, "path": target.relative_to(ROOT).as_posix(), "exists": target.is_file(), "anchor_found": anchor_found})
    source_text = snapshot.decode()
    scope = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-chapter-0C/verify_inputs.py",
        "environment": {"python": sys.version, "platform": platform.platform(), "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
        "snapshot": {"path": snapshot_path, "sha256": digest(snapshot), "matches_requested": digest(snapshot) == "8b28ccea49e692cfb79467a376ae30a8a57ab4d1cb55d29520af9c42b44f64b2"},
        "freeze": {"path": freeze_path, "sha256": digest(freeze_bytes), "matches_requested": digest(freeze_bytes) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"},
        "snapshot_matches_manifest": page["snapshot"] == snapshot_path and page["source_sha256"] == digest(snapshot),
        "manifest_figures": page["figures_sha256"],
        "image_references": re.findall(r"!\[[^\]]*\]\([^)]+\)", source_text),
        "code_fences": source_text.count("```"),
        "live_intro_matches_snapshot": live_intro == snapshot,
        "navigation_links": links,
        "has_prerequisite_headings": {heading: (heading in (ROOT / "course/chapters/07.md").read_text()) for heading in ["## 7.23 ", "## 7.24 ", "## 7.25 "]},
    }
    assert scope["snapshot"]["matches_requested"] and scope["freeze"]["matches_requested"]
    assert scope["snapshot_matches_manifest"] and scope["manifest_figures"] == {}
    assert not scope["image_references"] and not scope["code_fences"]
    assert scope["live_intro_matches_snapshot"]
    assert all(item["exists"] and item["anchor_found"] for item in links)
    save("scope-check.json", scope)
    specifications = [
        {
            "source_id": "s1", "url": "https://arxiv.org/html/2201.11903v6", "version": "arXiv:2201.11903v6, 2023-01-10",
            "sections": ["S1", "S2"], "excerpt_section": "S2",
            "excerpt": "a coherent series of intermediate reasoning steps that lead to the final answer for a problem",
            "inspection_note": "Introduction 將 prompting 與為每個任務 finetune checkpoint 分開；Chain-of-Thought Prompting 說明用提示示範生成中間步驟，再產生最後答案。這支持可固定參數改變輸出形式，不保證任意模型、任意問題都答對，也不把文字步驟視為內部計算的完整紀錄。",
        },
        {
            "source_id": "s2", "url": "https://arxiv.org/html/2110.14168v2", "version": "arXiv:2110.14168v2",
            "sections": ["S4.SS2", "S5.SS1"], "excerpt_section": "S5.SS1",
            "excerpt": "At test time, we can choose to generate arbitrarily many solutions to be judged by the verifier before selecting the highest ranked completion.",
            "inspection_note": "Verification 明確分開事前訓練 generator/verifier 與測試時抽樣、評分、選擇；Test Time Compute 固定已訓練模型，改候選數與評選流程。這支持多候選後選一份和作答時計算的分類，不保證多抽必然改善，也不保證驗證器永遠選中正解。",
        },
        {
            "source_id": "s3", "url": "https://arxiv.org/html/2501.12948v1", "version": "arXiv:2501.12948v1, 2025-01-22",
            "sections": ["S2.SS2.SSS1.Px1", "S2.SS2.SSS2"], "excerpt_section": "S2.SS2.SSS2",
            "excerpt": "The reward is the source of the training signal, which decides the optimization direction of RL.",
            "inspection_note": "Reward Modeling 對有確定結果的數學題使用規則核對 correctness，作為 accuracy reward；Group Relative Policy Optimization 的目標函式以 reward-derived advantage 最佳化 policy 參數。支持正確性分數可以拿來訓練、訓練會更新參數；單獨生成或計算 reward 不等於已執行參數更新，也不能保證每一次數值更新都非零。",
        },
    ]
    receipts = []
    for spec in specifications:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", spec["url"]]
        completed = subprocess.run(command, capture_output=True, check=True)
        raw = completed.stdout
        html = raw.decode()
        section_checks = []
        excerpt_text = None
        for section_id in spec["sections"]:
            match = re.search(r'<section[^>]*id="' + re.escape(section_id) + '"', html)
            assert match is not None
            end = html.find("<section", match.end())
            segment = html[match.start():end if end >= 0 else None]
            text = plain_text(segment)
            heading = re.search(r"<h[1-6][^>]*>(.*?)</h[1-6]>", segment, re.S)
            section_checks.append({"id": section_id, "heading": plain_text(heading.group(1)), "text_sha256": digest(text.encode()), "text_characters": len(text)})
            if section_id == spec["excerpt_section"]:
                excerpt_text = text
        assert spec["excerpt"] in excerpt_text
        assert len(spec["excerpt"].split()) <= 25
        receipt = dict(spec)
        receipt.update({"command": command, "exit_code": completed.returncode, "stderr": completed.stderr.decode(), "accessed_at_utc": datetime.now(UTC).isoformat(), "accessed_on": "2026-10-20", "response_bytes": len(raw), "response_sha256": digest(raw), "checked_original": True, "sections_checked": section_checks})
        receipts.append(receipt)
    save("original-sources.json", {"storage_note": "外部全文僅在記憶體讀取，未保存於 repo；此處只保留取證收據、定位、短摘錄及核對說明。", "sources": receipts})
    print(json.dumps({"scope_checks": "pass", "original_sources_checked": len(receipts), "files": ["scope-check.json", "original-sources.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
