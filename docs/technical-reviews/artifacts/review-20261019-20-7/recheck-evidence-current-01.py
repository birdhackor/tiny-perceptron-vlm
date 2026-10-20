"""保存新版原文、既有證據、程式與圖的複查收據，不改寫舊檔。"""

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_20-7/technical-20-7-original-01.json"
FROZEN = ROOT / "docs/course-review-20261019/freeze-05/sources/20.7.md"
FREEZE = ROOT / "docs/course-review-20261019/freeze-06.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def main():
    run = (ROOT / sys.argv[1]).resolve()
    assert run == ROOT / "outputs/reviewer-tools/runs/20261020T101629845306Z"
    assert digest(ORIGINAL) == "a076a669cb357a8eef54ebb1fcef35fe61ad9a452b4ae9638354b1f691095374"
    assert digest(FROZEN) == "47b61c26efaa98d29b12e5cca21b371db7e70dda89d9f4521880b06572118b2c"
    assert digest(FREEZE) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
    report = read_json(ORIGINAL)
    files = []
    for group in ["sources", "artifacts"]:
        for item in report[group]:
            if not item.get("path"):
                continue
            actual = digest(ROOT / item["path"])
            assert actual == item["sha256"], item["path"]
            files.append({"group": group, "id": item["id"], "path": item["path"],
                          "expected_sha256": item["sha256"], "current_sha256": actual, "unchanged": True})
    old = (ROOT / "docs/course-review-20261019/freeze-01/sources/20.7.md").read_bytes()
    new = FROZEN.read_bytes()
    chapter = (ROOT / "course/chapters/20.md").read_bytes()
    start = chapter.index(b"## 20.7 ")
    end = chapter.index(b"## ", start + 1)
    assert chapter[start:end] == new
    page = next(p for p in read_json(FREEZE)["pages"] if p["page_id"] == "20.7")
    assert page["source_sha256"] == digest(FROZEN)
    before, after = old.decode().splitlines(), new.decode().splitlines()
    changed = []
    for tag, a, b, c, d in SequenceMatcher(a=before, b=after).get_opcodes():
        if tag != "equal":
            changed.append({"change": tag, "before": before[a:b], "after": after[c:d]})
    assert len(changed) == 1
    command = ["git", "diff", "--no-index", "docs/course-review-20261019/freeze-01/sources/20.7.md",
               "docs/course-review-20261019/freeze-05/sources/20.7.md"]
    difference = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert difference.returncode == 1
    execution = read_json(run / "execution.json")
    extraction = read_json(run / "extraction.json")
    environment = read_json(run / "environment.json")
    assert execution["exit_code"] == 0
    assert extraction["source_sha256"] == digest(FROZEN)
    assert (run / "section.md").read_bytes() == new
    expected_stdout = "輸入X [1, 3, 89, 2, 4, 73]\n目標Y [-100, -100, -100, -100, 73, 2]\n有效目標數 2\n第一個有效目標的位置 4\n有效目標ID [73, 2]\n"
    assert (run / "stdout.txt").read_text() == expected_stdout
    assert (run / "stderr.txt").read_bytes() == b""
    qwen = read_json(DIRECTORY / "qwen-text-output.json")
    assert qwen["wrapper_sha256"] == digest(ROOT / "tiny_perceptron/natural_assistant.py")
    assert qwen["supervised_ids"] == [32, 151645, 198]
    assert qwen["supervised_after_shift"] == [32, 151645, 198]
    assert qwen["newline_text"] == "\n"
    assert qwen["full_text"][len(qwen["prompt_text"]):] == "A<|im_end|>\n"
    figure_receipt = ROOT / "docs/course-review-20261019/figure-renders-04.json"
    figure = next(r for r in read_json(figure_receipt)["records"] if r["page_id"] == "20.7")
    assert page["figures_sha256"] == report["figure_sha256"]
    assert digest(ROOT / figure["figure"]) == report["figure_sha256"][figure["figure"]]
    for key in ["frozen_svg", "png"]:
        assert digest(ROOT / figure[key]["path"]) == figure[key]["sha256"]
    facts = {
        "command": "PYTHONDONTWRITEBYTECODE=1 " + shlex.join([".venv/bin/python", str(Path(__file__).resolve().relative_to(ROOT)), sys.argv[1]]),
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu",
                        "torch": environment["torch"]},
        "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(ORIGINAL)},
        "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
        "source": {"path": str(FROZEN.relative_to(ROOT)), "sha256": digest(FROZEN), "current_section_equal": True},
        "prior_files": files,
        "diff": {"command_argv": command, "exit_code": difference.returncode, "stdout": difference.stdout,
                 "stderr": difference.stderr, "changed_lines": changed},
        "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.7" --execute',
                              "execution": execution, "extraction": extraction, "environment": environment,
                              "stdout": (run / "stdout.txt").read_text(), "stderr": (run / "stderr.txt").read_text(),
                              "code": (run / "fence-1.py").read_text(),
                              "copied_from": str(run.relative_to(ROOT)),
                              "original_files_sha256": {n: digest(run / n) for n in ["execution.json", "extraction.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"]}},
        "changed_claim_verification": {"supervised_ids": qwen["supervised_ids"], "after_shift": qwen["supervised_after_shift"],
                                       "suffix": qwen["supervised_text"], "newline_id": qwen["newline_id"],
                                       "basis": "沿用原審閱的固定版本模板／tokenizer 純文字執行結果；本次重新驗證檔案 SHA-256、資料包裝函式 SHA-256 及輸出欄位。不是再次執行 Qwen 探針。",
                                       "judgment": "新版對答案、結束標記及後接換行均受監督的敘述與證據相符。"},
        "figure": {"receipt": {"path": str(figure_receipt.relative_to(ROOT)), "sha256": digest(figure_receipt)},
                   "record": figure, "viewed_with": "view_image",
                   "inspection": "本次重新觀看指定 PNG 並核對 SVG；六個輸入格及四灰兩藍目標格、assistant→A 與 A→EOS 方向和兩個 EOS 的用途均與 toy 程式一致。圖有明示不是完整 Qwen 序列，與新版新增的 Qwen 換行說明無衝突。PNG 不列為 artifact。"},
        "result": "全部雜湊、原文差異、沿用證據及新版教材執行結果斷言通過。",
    }
    out = DIRECTORY / "recheck-evidence-current-01.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(facts, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(out.relative_to(ROOT)), "sha256": digest(out), "result": facts["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
