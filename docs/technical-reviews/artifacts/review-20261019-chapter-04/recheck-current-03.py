"""保存現行依賴差異、原證據完整性及模型與算例的複查結果。"""

import contextlib
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
REPORTS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-04"
PREVIOUS = REPORTS / "technical-chapter-04-current-02.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, content):
    with path.open("x") as stream:
        stream.write(content)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def section(text, identifier):
    match = re.search(r"^## " + re.escape(identifier) + r" .*$", text, re.MULTILINE)
    assert match is not None, "找不到待核對的小節"
    following = re.search(r"^## ", text[match.end():], re.MULTILINE)
    end = match.end() + following.start() if following else len(text)
    return text[match.start():end]


def run_blocks(text, path):
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        for code in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
            exec(compile(code, path, "exec"), namespace)
    return output.getvalue(), namespace


assert digest(PREVIOUS) == "421471d88dfaea7322f94989d0c0b0397278b69c9a2cfa677882bdd1acae04d0"
previous = json.loads(PREVIOUS.read_text())
protected = {}
queue = [PREVIOUS]
while queue:
    path = queue.pop()
    if path in protected:
        continue
    protected[path] = digest(path)
    report = json.loads(path.read_text())
    for artifact in report["artifacts"]:
        item = ROOT / artifact["path"]
        assert digest(item) == artifact["sha256"], "舊證據的指紋已變動"
        protected[item] = digest(item)
    for field in ("original_report", "supersedes", "previous_report"):
        if field in report:
            item = ROOT / report[field]["path"]
            assert item.parent == REPORTS, "報告鏈指向範圍外的報告"
            assert digest(item) == report[field]["sha256"], "舊報告的指紋已變動"
            queue.append(item)

expected = {
    "course/chapters/03.md": "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5",
    "course/chapters/14.md": "81bdbd56dda7443cb2f97952a6eaa318f76da447a7caa3a966255e21de83f41d",
}
diffs = []
unchanged = []
texts = {}
for path, expected_hash in expected.items():
    assert digest(ROOT / path) == expected_hash, "現行檔案與派工指紋不符"
    before = git("show", "ecf6dec71280:" + path)
    current = (ROOT / path).read_text()
    prior_source = next(source for source in previous["sources"] if source.get("path") == path)
    assert hashlib.sha256(before.encode()).hexdigest() == prior_source["sha256"], "指定差異起點不是前次核對版本"
    assert git("show", "c6b9339d2595:" + path) == current, "指定差異終點不等於現行檔案"
    command = ["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", path]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    diffs.append({"command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    identifiers = ["3.2", "3.5", "3.6", "3.7"] if path.endswith("03.md") else ["14.2", "14.11"]
    for identifier in identifiers:
        old_section = section(before, identifier)
        new_section = section(current, identifier)
        assert old_section == new_section, "原先依據的小節已改動"
        unchanged.append({"path": path, "selector": identifier, "unchanged": True, "sha256": hashlib.sha256(new_section.encode()).hexdigest()})
    if path.endswith("14.md"):
        old_intro, new_intro = before.split("## ", 1)[0], current.split("## ", 1)[0]
        assert old_intro == new_intro, "先前核對的條件式導覽已改動"
        unchanged.append({"path": path, "selector": "before-first-section", "unchanged": True, "sha256": hashlib.sha256(new_intro.encode()).hexdigest()})
    texts[path] = {"before": before, "current": current}

model_code = (BASE / "verify_model.py").read_text()
model_code = model_code.replace("freeze-04.json", "freeze-07.json")
model_code = model_code.replace("a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753")
model_code = model_code.replace("repository-inspection.json", "repository-inspection-current-03.json")
model_code = model_code.replace('(ARTIFACTS / "repository-inspection-current-03.json").write_text(', 'write_new(ARTIFACTS / "repository-inspection-current-03.json", ')
model_code = model_code.replace('/verify_model.py"', '/recheck-current-03.py"')
model_stdout = io.StringIO()
with contextlib.redirect_stdout(model_stdout):
    exec(compile(model_code, str(BASE / "verify_model.py"), "exec"), {"__file__": str(BASE / "verify_model.py"), "__name__": "__main__", "write_new": write_new})
write_new(BASE / "model-verification-current-03.txt", model_stdout.getvalue())
model_run = json.loads(model_stdout.getvalue())

path03 = "course/chapters/03.md"
old_match = section(texts[path03]["before"], "3.3")
new_match = section(texts[path03]["current"], "3.3")
old_code = re.findall(r"```python\n(.*?)```", old_match, re.DOTALL)
new_code = re.findall(r"```python\n(.*?)```", new_match, re.DOTALL)
assert old_code == new_code, "修改匹配說明時連帶改了程式"
match_stdout, matches = run_blocks(new_match, path03)
shared_scores = matches["has"] @ matches["has"].T
separate_scores = matches["need"] @ matches["has"].T
assert torch.equal(shared_scores, torch.eye(2)), "共用表示的指定矩陣不符"
assert torch.equal(separate_scores, torch.tensor([[0.0, 1.0], [1.0, 0.0]])), "分開表示的指定矩陣不符"
assert torch.equal(shared_scores.argmax(-1), torch.tensor([0, 1])), "本例最高分未落在自己"

path14 = "course/chapters/14.md"
old_loop = section(texts[path14]["before"], "14.12")
new_loop = section(texts[path14]["current"], "14.12")
old_link = "[7.20](07.md#7.20)"
new_link = "[1.14](01.md#1.14)"
assert old_loop.replace(old_link, new_link) == new_loop, "循環狀態小節不只改了指定連結"
loop_stdout, loop = run_blocks(new_loop, path14)
assert loop["state"] == 3.75 and loop["e"] == 4.0
path01 = "course/chapters/01.md"
text01 = (ROOT / path01).read_text()
generation = section(text01, "1.14")
assert "## 1.14 下一字怎樣接成一段新文字？" in generation
assert "output.append(chars[current])" in generation
assert "current = torch.multinomial(p[current], 1, generator=g).item()" in generation
assert "先算下一字分配、選一字、接回句尾，再把新字當下一次前文" in generation
generation_stdout, generated = run_blocks(generation, path01)
assert len(generated["output"]) == 13
assert generated["output"][0] == "貓"

assert all(digest(path) == value for path, value in protected.items()), "複查過程更動了舊證據"
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --offline python docs/technical-reviews/artifacts/review-20261019-chapter-04/recheck-current-03.py",
    "environment": model_run["environment"],
    "previous_issues": previous["issues"],
    "protected_file_integrity": [{"path": path.relative_to(ROOT).as_posix(), "sha256": value, "unchanged": True} for path, value in protected.items()],
    "dependency_diffs": diffs,
    "unchanged_dependency_sections": unchanged,
    "matching_example": {"stdout": match_stdout, "program_unchanged": True, "shared_scores": shared_scores.tolist(), "separate_scores": separate_scores.tolist(), "argmax_shared": shared_scores.argmax(-1).tolist(), "derivation": "令 A=[1,0]、B=[0,1]，A·A=B·B=1，A·B=B·A=0；因此本例各列最高分落在自己。一般共用表示的匹配矩陣是 XXᵀ；其轉置仍為 XXᵀ，故 A 對 B 與 B 對 A 的匹配分數相同。改用互換的 query 時，need @ has.T=[[0,1],[1,0]]。這些是匹配分數，不是 softmax 權重。"},
    "generation_link": {"only_link_changed": True, "old_link": old_link, "new_link": new_link, "target_path": path01, "target_file_sha256": digest(ROOT / path01), "target_heading": generation.splitlines()[0], "target_section_sha256": hashlib.sha256(generation.encode()).hexdigest(), "target_stdout": generation_stdout, "target_initial_tokens": 1, "target_generated_tokens": len(generated["output"]) - 1, "target_total_tokens": len(generated["output"]), "loop_example_stdout": loop_stdout, "loop_final_state": loop["state"], "scope": "新連結正確定位逐字生成的說明與短例。只核對改動的前置連結及相關機制，不另宣稱已全面審閱該段原有的其他實驗與整合模型成果。"},
    "model_reexecution": {"path": (BASE / "model-verification-current-03.txt").relative_to(ROOT).as_posix(), "sha256": digest(BASE / "model-verification-current-03.txt"), "adjustments": ["指定現行凍結 manifest 與指紋", "指定新的執行／repository-inspection 檔名與排他寫入", "相同模型核對邏輯未變"]},
    "result": "原報告鏈與既有 artifacts 均完整；原頁模型短核對重跑通過；兩處依賴文字改寫均已核對。",
}, ensure_ascii=False, indent=2))
