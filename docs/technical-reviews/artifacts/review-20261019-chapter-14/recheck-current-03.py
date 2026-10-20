"""核對引用頁面的連結變動，重跑短程式並保留唯讀證據的完整性。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-14/technical-chapter-14-current-01.json"
SNAPSHOT = ROOT / "docs/course-review-20261019/freeze-05/sources/chapter-14.md"
FREEZE = ROOT / "docs/course-review-20261019/freeze-07.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_sections(text):
    headings = list(re.finditer(r"^## ([A-Z0-9]+\.\d+) (.+)$", text, re.M))
    return {
        heading[1]: {"title": heading[2], "text": text[heading.start():headings[index + 1].start() if index + 1 < len(headings) else len(text)]}
        for index, heading in enumerate(headings)
    }


expected = {
    PREVIOUS: "42fcd2aac9709917f04147c7cd66854f2f684d8444490a91776d80a1c05972ef",
    SNAPSHOT: "aeaf98907ca67c721bbadf9b4403f5405accf77100e63c489f9697c8b9d3ae03",
    FREEZE: "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753",
}
for path, expected_digest in expected.items():
    if digest(path) != expected_digest:
        raise ValueError("指定輸入與保存的指紋不符：" + str(path))
previous = json.loads(PREVIOUS.read_bytes())
artifact_integrity = []
for artifact in previous["artifacts"]:
    actual = digest(ROOT / artifact["path"])
    artifact_integrity.append({"path": artifact["path"], "sha256": actual, "unchanged": actual == artifact["sha256"]})
if not all(item["unchanged"] for item in artifact_integrity):
    raise ValueError("沿用的證據檔已改變")
repository_sources = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(ROOT / source["path"])
        repository_sources.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
        if source["path"] != "course/chapters/14.md" and actual != source["sha256"]:
            raise ValueError("其他已引用的依據檔有變動：" + source["path"])

chapter_path = ROOT / "course/chapters/14.md"
chapter_raw = chapter_path.read_bytes()
if digest(chapter_path) != "81bdbd56dda7443cb2f97952a6eaa318f76da447a7caa3a966255e21de83f41d":
    raise ValueError("章檔與指定的新版本不同")
first_section = re.search(rb"^## ", chapter_raw, re.M)
if chapter_raw[:first_section.start()] != SNAPSHOT.read_bytes():
    raise ValueError("章首與凍結原文不一致")
chapter_sections = split_sections(chapter_raw.decode())

command = ["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/chapters/14.md"]
difference = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
revision_files = []
for revision in ("ecf6dec71280", "c6b9339d2595"):
    raw = subprocess.check_output(["git", "show", revision + ":course/chapters/14.md"], cwd=ROOT)
    revision_files.append({"revision": revision, "sha256": hashlib.sha256(raw).hexdigest()})

probe_path = ARTIFACTS / "review_probe.py"
probe_code = probe_path.read_text()
probe_code = probe_code.replace("freeze-01/sources/chapter-14.md", "freeze-05/sources/chapter-14.md")
probe_code = probe_code.replace("freeze-04.json", "freeze-07.json")
probe_code = probe_code.replace("chapter-14/review_probe.py", "chapter-14/recheck-current-03.py")
probe_code = probe_code.replace('"library_scope_counterexample"', '"qk_local_example_boundary"')
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(probe_code, str(probe_path), "exec"), {"__file__": str(probe_path), "__name__": "__main__"})
probe = json.loads(captured.getvalue())

previous_result = json.loads((ARTIFACTS / "recheck-output-current-02.json").read_bytes())
fixed_examples = []
for item in previous_result["fixed_number_example_inventory"]:
    current_section = chapter_sections[item["section_id"]]
    fixed_examples.append({"section_id": item["section_id"], "title": current_section["title"], "example_still_present": item["fixed_example"] in current_section["text"]})
if not all(item["example_still_present"] for item in fixed_examples):
    raise ValueError("固定數字的示例不再完整")

paragraph = next(p for p in chapter_raw.decode().split("\n\n") if "輸入 token 軸" in p)
links = [(label, link) for label, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", paragraph) if link.startswith("01.md#")]
if len(links) != 1:
    raise ValueError("改句沒有唯一的前置連結")
label, target = links[0]
filename, fragment = target.split("#", 1)
target_path = chapter_path.parent / filename
target_section = split_sections(target_path.read_text())[fragment]
code_blocks = re.findall(r"```python\n(.*?)\n```", target_section["text"], re.S)
if len(code_blocks) != 1:
    raise ValueError("前置生成示範沒有唯一的程式圍欄")
namespace = {"__name__": "__main__"}
execution_output = io.StringIO()
with contextlib.redirect_stdout(execution_output):
    exec(compile(code_blocks[0], str(target_path), "exec"), namespace)
generated = namespace["output"]

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-14/recheck-current-03.py",
    "environment": {"python": platform.python_version(), "torch": probe["environment"]["torch"], "device": "cpu", "platform": platform.platform()},
    "previous_report": {"path": PREVIOUS.relative_to(ROOT).as_posix(), "sha256": digest(PREVIOUS)},
    "source_snapshot": {"path": SNAPSHOT.relative_to(ROOT).as_posix(), "sha256": digest(SNAPSHOT)},
    "source_freeze": {"path": FREEZE.relative_to(ROOT).as_posix(), "sha256": digest(FREEZE)},
    "previous_artifact_integrity": artifact_integrity,
    "repository_source_integrity": repository_sources,
    "current_intro_sha256": hashlib.sha256(chapter_raw[:first_section.start()]).hexdigest(),
    "diff_receipt": {"command": command, "exit_code": difference.returncode, "stdout": difference.stdout, "revision_files": revision_files},
    "fixed_number_example_inventory": fixed_examples,
    "changed_dependency_link": {
        "current_paragraph": paragraph,
        "target": target,
        "label": label,
        "target_path": target_path.relative_to(ROOT).as_posix(),
        "target_file_sha256": digest(target_path),
        "target_section_title": target_section["title"],
        "target_section_sha256": hashlib.sha256(target_section["text"].encode()).hexdigest(),
        "target_intro": target_section["text"].split("\n\n")[1],
        "extracted_code_sha256": hashlib.sha256(code_blocks[0].encode()).hexdigest(),
        "execution_stdout": execution_output.getvalue(),
        "generated_sequence": generated,
        "initial_length": 1,
        "sampling_steps": len(generated) - 1,
        "final_length": len(generated),
        "assertion_passed": len(generated) == 13,
        "scope": "只核對新前置連結確實教選字、接回前文與重複生成；不重審該節其他實驗結果。",
    },
    "current_probe": probe,
}, ensure_ascii=False, indent=2))
