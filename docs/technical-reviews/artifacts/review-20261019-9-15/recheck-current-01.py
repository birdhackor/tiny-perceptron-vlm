"""核對新版原文、原始來源、執行結果與可沿用的證據。"""

import difflib
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sections(raw):
    headers = list(re.finditer(rb"(?m)^## ([^\r\n]+)", raw))
    return {
        h[1].decode().split(" ", 1)[1]: raw[h.start():headers[i + 1].start() if i + 1 < len(headers) else len(raw)]
        for i, h in enumerate(headers)
    }


def main():
    old_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-15/technical-9-15-original-02.json"
    old_raw = old_path.read_bytes()
    old = json.loads(old_raw)
    assert digest(old_raw) == "c5c61623ea8ef1d2c5ae84db0aa402352fd3f8a4098e56dccb787cbcd486f9d2"
    previous_path = ROOT / "docs/course-review-20261019/freeze-01/sources/9.15.md"
    current_path = ROOT / "docs/course-review-20261019/freeze-05/sources/9.15.md"
    previous, current = previous_path.read_bytes(), current_path.read_bytes()
    assert digest(current) == "acacc54a4f1c3df9ea2398713b4abc7dc6b3283a5dc8783936ccced18ee5a54f"
    assert previous.replace(b"2026-10-16", b"2026-10-20", 1) == current
    chapter_path = ROOT / "course/chapters/09.md"
    chapter = chapter_path.read_bytes()
    assert digest(chapter) == "62274ee1570b419bf5872ebbf4d84b63115442ac4960a826464ea28fececbd8b"
    assert sections(chapter)["confidence 是「有幾成機會答對」嗎？"] == current
    old_chapter = subprocess.run(
        ["git", "show", "5ef56bc756f3:course/chapters/09.md"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    assert digest(old_chapter) == "9d2039c6aba3bfe4e66c3a5c442180628321780de5cda00a0cc7fd96d85d03ee"
    all_sections = sections(chapter)
    earlier_sections = sections(old_chapter)
    dependencies = {}
    for title in ("最大 softmax 機率很高就可靠嗎？", "信心和正確率對得上嗎？", "缺資訊，應該輸出低信心嗎？", "有機率輸出，還要怎樣訓練與校準？"):
        before, after = earlier_sections[title], all_sections[title]
        if title == "最大 softmax 機率很高就可靠嗎？":
            assert before.replace("round`四捨五入".encode(), "round`取到指定的小數位數".encode()) == after
        else:
            assert before == after
        dependencies[title] = {"previous_sha256": digest(before), "current_sha256": digest(after), "identical": before == after}
    hashes = {}
    for item in old["sources"] + old["artifacts"]:
        if "path" not in item:
            continue
        actual = digest((ROOT / item["path"]).read_bytes())
        unchanged = actual == item["sha256"]
        if item["path"] != "course/chapters/09.md":
            assert unchanged
        hashes[item["path"]] = {"previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": unchanged}
    argv = ["curl", "--fail", "--silent", "--show-error", "--location", "--include", "--max-time", "45", "https://docs.typesafe.ai/confidence.md"]
    raw = subprocess.run(argv, capture_output=True, check=True).stdout
    headers = {}
    while raw.startswith(b"HTTP/"):
        head, raw = raw.split(b"\r\n\r\n", 1)
        lines = head.decode().splitlines()
        status = lines[0]
        headers = {k.lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:]) if k}
    doc = raw.decode()
    choice = doc.split("### Choice\n", 1)[1].split("### Score\n", 1)[0]
    noul = doc.split("### Noul\n", 1)[1].split("### Choice\n", 1)[0]
    score = doc.split("### Score\n", 1)[1]
    equation = re.search(r"\$\$(.*?)\$\$", choice, re.S)[1].strip()
    assert r"\frac{p_{\max} - \frac{1}{n}}{1 - \frac{1}{n}}" in equation
    assert "TypeSafe returns no separate `confidence` for it." in noul
    assert "levels are ordered" in score
    official = {
        "command_argv": argv, "http_status": status, "http_date": headers.get("date"),
        "last_modified": headers.get("last-modified"), "body_sha256": digest(raw),
        "body_bytes": len(raw), "accessed_utc": datetime.now(UTC).isoformat(),
        "full_body_saved": False, "locator": "How confidence is calculated → Choice / Noul / Score",
        "excerpts": {"choice_equation": equation, "noul": "TypeSafe returns no separate `confidence` for it.", "score_order": "levels are ordered"},
    }
    assert official["accessed_utc"].startswith("2026-10-20")
    run = Path(sys.argv[1]).resolve()
    assert run.parent == ROOT / "outputs/reviewer-tools/runs"
    records = {}
    for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"):
        body = (run / name).read_bytes()
        records[name] = {"sha256": digest(body), "content": body.decode()}
    extraction = json.loads(records["extraction.json"]["content"])
    execution = json.loads(records["execution.json"]["content"])
    environment = json.loads(records["environment.json"]["content"])
    assert extraction["source_sha256"] == digest(current)
    assert extraction["figure_sha256"] == {}
    assert len(extraction["python_fences"]) == 1
    assert execution["exit_code"] == 0
    assert records["stdout.txt"]["content"] == "最大機率 0.6 集中度 0.4\n"
    assert records["stderr.txt"]["content"] == ""
    value = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + str(run.relative_to(ROOT)),
        "section_command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.15" --execute',
        "environment": {"python": environment["python"], "torch": environment["torch"], "device": environment["device_requested"]},
        "original_report_sha256": digest(old_raw), "source_sha256": digest(current),
        "freeze_sha256": digest((ROOT / "docs/course-review-20261019/freeze-05.json").read_bytes()),
        "source_diff": "".join(difflib.unified_diff(previous.decode().splitlines(True), current.decode().splitlines(True), fromfile=str(previous_path.relative_to(ROOT)), tofile=str(current_path.relative_to(ROOT)))),
        "chapter_diff_command": "git diff 5ef56bc756f3 3af7f13ef959 -- course/chapters/09.md",
        "chapter_diff": subprocess.run(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/09.md"], cwd=ROOT, capture_output=True, check=True, text=True).stdout,
        "dependency_sections": dependencies, "reused_evidence_hashes": hashes,
        "official_source": official, "section_files": records,
        "result": "所有斷言通過；新版只調整官方文件日期，與實際讀取日期相符；本課信心及資訊充分的前置依據仍成立，原始程式輸出不變。",
        "scope": "只讀取本人先前明確引用的證據與本次工具回傳的執行目錄；沿用數值及固定logits證據，不重複訓練或驗證正式權重品質。",
    }
    with (ARTIFACTS / "recheck-current-01.json").open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(value["result"])


if __name__ == "__main__":
    main()
