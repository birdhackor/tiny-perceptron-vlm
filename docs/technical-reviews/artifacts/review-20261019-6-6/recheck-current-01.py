"""核對既存證據完整性與生成前綴，所有收據只用新檔名保存。"""

import hashlib
import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tokenizers
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def section(raw, identifier):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    matches = [i for i, heading in enumerate(headings)
               if re.match(rb"^## " + re.escape(identifier.encode()) + rb" ", heading[0])]
    assert len(matches) == 1, "必須恰有一個指定標題"
    i = matches[0]
    return raw[headings[i].start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_6-6/technical-6-6-original-01.json"
original_sha = sha(original_path.read_bytes())
assert original_sha == "b519955501773e7686d7e974b65ece9f8d982b9391bbc7186309a86880cf08d5"
original = json.loads(original_path.read_bytes())
assert original["issues"] == []
source_audit = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    current = sha((ROOT / source["path"]).read_bytes())
    unchanged = current == source["sha256"]
    if source["id"] != "s_chat_code":
        assert unchanged, source["path"]
    source_audit.append({"source_id": source["id"], "path": source["path"],
                         "previous_sha256": source["sha256"], "current_sha256": current,
                         "unchanged": unchanged})
artifact_audit = []
for artifact in original["artifacts"]:
    current = sha((ROOT / artifact["path"]).read_bytes())
    assert current == artifact["sha256"], artifact["path"]
    artifact_audit.append({"artifact_id": artifact["id"], "path": artifact["path"],
                           "sha256": current, "unchanged": True})
old_self_check = OUT / "self-check.json"
old_self_check_sha = sha(old_self_check.read_bytes())

freeze = (ROOT / "docs/course-review-20261019/freeze-01/sources/6.6.md").read_bytes()
current = section((ROOT / "course/chapters/06.md").read_bytes(), "6.6")
assert freeze == current
assert sha(current) == original["source_sha256"]
freeze_manifest = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze_manifest.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"

old_revision = "5ef56bc756f3"
new_revision = "3af7f13ef959"
history = []
for revision in (old_revision, new_revision):
    argv = ["git", "show", revision + ":course/chapters/07.md"]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    history.append(result.stdout)
old_chapter, committed_chapter = history
current_chapter = (ROOT / "course/chapters/07.md").read_bytes()
assert sha(old_chapter) == "4f8f31898954117a1e8203d98269da246ec4b7794d8997dd5264055945761ff9"
assert sha(current_chapter) == "4b02242d4c281a8a0554a3a87bdc898c040bb7afa3250dc8c055184407952174"
assert committed_chapter == current_chapter
old_prompt = section(old_chapter, "7.2")
current_prompt = section(current_chapter, "7.2")
assert old_prompt == current_prompt
argv = ["git", "diff", old_revision, new_revision, "--", "course/chapters/07.md"]
difference = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
assert len(re.findall(r"(?m)^@@ ", difference.stdout)) == 3

code_blocks = re.findall(rb"(?ms)^```python\r?\n(.*?)^```", current_prompt)
assert len(code_blocks) == 1
namespace = {"__name__": "__main__"}
stdout = io.StringIO()
with redirect_stdout(stdout):
    exec(compile(code_blocks[0], "course/chapters/07.md:模型怎麼知道輪到誰？", "exec"), namespace)
expected = [1, 3, 57, 51, 57, 69, 71, 2, 4]
assert namespace["prompt"] == expected
assert namespace["x"][:len(expected)].tolist() == expected

tok = ByteTokenizer()
literal = tok.encode("<assistant>")
user_ids = [tok.bos_id, tok.user_id] + literal + [tok.eos_id, tok.assistant_id]
assistant_ids = [tok.bos_id, tok.assistant_id] + literal + [tok.eos_id, tok.assistant_id]
x, _ = render_chat([{"role": "user", "content": "<assistant>"},
                    {"role": "assistant", "content": "答"}])
assert x[:len(user_ids)].tolist() == user_ids
assert user_ids.count(tok.assistant_id) == 1
assert assistant_ids.count(tok.assistant_id) == 2
assert tok.decode(literal) == "<assistant>" and min(literal) >= 8

run = (ROOT / sys.argv[1]).resolve()
assert run.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
files = {}
for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt"):
    raw = (run / name).read_text()
    files[name] = json.loads(raw) if name.endswith(".json") else raw
assert files["execution.json"]["exit_code"] == 0
assert files["extraction.json"]["source_sha256"] == sha(freeze)
assert len(files["extraction.json"]["python_fences"]) == 3
assert files["extraction.json"]["figure_sha256"] == {}
assert files["stderr.txt"] == ""
assert (run / "section.md").read_bytes() == freeze
previous_run = json.loads((OUT / "section-run.json").read_bytes())
assert files["stdout.txt"] == previous_run["files"]["stdout.txt"]
bundle = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/06.md#6.6" --execute',
          "result": "三個 Python fences 執行成功；stdout 與原始收據相同，stderr 為空。", "files": files}
save("section-run-current-01.json", bundle)
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + Path(__file__).relative_to(ROOT).as_posix() + " " + run.relative_to(ROOT).as_posix(),
    "result": "全部斷言通過；引用的生成前綴未變且當次執行結果正確。",
    "environment": {"python": sys.version, "torch": torch.__version__, "tokenizers": tokenizers.__version__, "device": "cpu"},
    "original_report": {"path": original_path.relative_to(ROOT).as_posix(), "sha256": original_sha},
    "previous_issues": [], "resolved_issue_ids": [],
    "target_source_unchanged": True, "source_sha256": sha(current),
    "source_audit": source_audit, "reused_artifact_audit": artifact_audit,
    "chapter_diff": {"command": " ".join(argv), "exit_code": difference.returncode, "stdout": difference.stdout,
                     "old_chapter_sha256": sha(old_chapter), "current_chapter_sha256": sha(current_chapter),
                     "committed_and_worktree_equal": True,
                     "relevant_section_unchanged": True, "relevant_section_sha256": sha(current_prompt),
                     "scope": "三處差異在引用的 prompt 小節之外；僅判斷對本頁證據的影響，不裁決那些小節的其他主張。"},
    "prompt_execution": {"input_question": namespace["question"], "prompt": namespace["prompt"],
                         "training_prefix": namespace["x"][:len(expected)].tolist(), "stdout": stdout.getvalue(),
                         "literal_user_ids": user_ids, "literal_assistant_ids": assistant_ids},
    "section_execution": {"path": (OUT / "section-run-current-01.json").relative_to(ROOT).as_posix(),
                          "sha256": sha((OUT / "section-run-current-01.json").read_bytes())},
}
assert sha(original_path.read_bytes()) == original_sha
assert sha(old_self_check.read_bytes()) == old_self_check_sha
for artifact in artifact_audit:
    assert sha((ROOT / artifact["path"]).read_bytes()) == artifact["sha256"]
save("recheck-output-current-01.json", receipt)
print(json.dumps({"result": receipt["result"], "source_sha256": sha(current),
                  "relevant_section_sha256": sha(current_prompt), "prompt": expected}, ensure_ascii=False))
