# 核對來源變更與作者補丁的文字倒退邊界；不讀取實際補丁清單或他人的審閱內容。
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_readme/technical-readme-current-02.json"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command):
    p = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=90)
    return {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main():
    import torch
    previous = json.loads(PREVIOUS.read_bytes())
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"}, "previous_report": {"path": PREVIOUS.relative_to(ROOT).as_posix(), "sha256": sha(PREVIOUS)}, "sources": [], "artifacts": [], "diffs": [], "patch_cases": [], "commands": []}
    for group in ("sources", "artifacts"):
        for entry in previous[group]:
            if "path" in entry:
                observed = sha(ROOT/entry["path"])
                result[group].append({"id": entry["id"], "path": entry["path"], "expected": entry["sha256"], "observed": observed, "unchanged": observed == entry["sha256"]})
    changed = [r["path"] for r in result["sources"] if not r["unchanged"]]
    assert set(changed) == {"docs/review-tools/incremental_revision.py", "docs/technical-review-guide.md"}
    assert all(r["unchanged"] for r in result["artifacts"])
    for name in changed:
        result["diffs"].append(run(["git", "diff", "6a683dc4d249", "41651edb", "--", name]))
    result["hashes"] = {name: sha(ROOT/name) for name in ("README.md", "docs/course-review-20261019/freeze-05/sources/readme.md", "docs/course-review-20261019/freeze-06.json", "course/figures/rewrite-01-character-ids.svg")}
    assert result["hashes"]["README.md"] == previous["source_sha256"] == result["hashes"]["docs/course-review-20261019/freeze-05/sources/readme.md"]
    tool = load("readme_patch_current", ROOT/"docs/review-tools/incremental_revision.py")
    base_patch = {"id": "synthetic-text-change", "path": "sample.md", "old": "較早文字", "new": "目前文字", "requested_by": "author", "requested_on": "2026-10-20", "reason": "純合成驗證資料，不代表實際作者授權"}
    def case(name, text, rejected=False, patch_data=None, absent_ok=False):
        selected = base_patch if patch_data is None else patch_data
        row = {"name": name, "input": text, "patch": selected, "expected_rejected": rejected, "absent_ok": absent_ok}
        try:
            row["versions"] = tool.undo_patches(text, [(0, selected)], "合成文字", absent_ok=absent_ok)
            row["rejected"] = False
        except ValueError as error:
            row["error"] = str(error)
            row["rejected"] = True
        row["passed"] = row["rejected"] == rejected
        result["patch_cases"].append(row)
        assert row["passed"], name
    case("普通正文唯一文字可倒退", "前文。目前文字。後文。")
    case("單純圍欄外正文可倒退", "```python\nx=1\n```\n目前文字")
    case("單純圍欄內文字拒絕", "```python\n目前文字\n```", True)
    case("四反引號圍欄拒絕", "````python\nx=1\n````\n目前文字", True)
    case("波浪號圍欄拒絕", "~~~python\nx=1\n~~~\n目前文字", True)
    case("縮排的反引號圍欄拒絕", " ```python\nx=1\n ```\n目前文字", True)
    case("引用區的反引號圍欄拒絕", "> ```python\n> x=1\n> ```\n目前文字", True)
    case("圍欄屬性語法拒絕", "```python {extra}\nx=1\n```\n目前文字", True)
    case("缺收尾拒絕", "目前文字\n```python\nx=1", True)
    case("內層帶語言名圍欄拒絕", "目前文字\n```python\n```text\n```", True)
    case("含三反引號字面值拒絕", '目前文字\n```python\nx="```"\n```', True)
    case("重複文字拒絕", "目前文字。目前文字。", True)
    case("缺少指定文字拒絕", "不同文字", True)
    restored = {**base_patch, "old": "`python"}
    case("倒退後產生不單純圍欄拒絕", "``目前文字\nx=1\n", True, restored)
    for name, text in (("行內程式碼由作者負責", "`目前文字`"), ("數學式由作者負責", "$目前文字$"), ("連結文字由作者負責", "[目前文字](https://example.invalid/)"), ("HTML程式區塊由作者負責", "<pre>目前文字</pre>"), ("四空白程式區塊由作者負責", "    目前文字")):
        case(name, text)
    case("CRLF的單純圍欄接受", "```python\r\nx=1\r\n```\r\n目前文字")
    case("其他頁未出現補丁可略過", "~~~~text\n無此文字", absent_ok=True)
    result["metadata_cases"] = []
    with tempfile.TemporaryDirectory(prefix="scratch-current-", dir=HERE) as tmp:
        tmp = Path(tmp)
        def metadata_case(name, item, rejected=False):
            path = tmp / ("metadata-" + str(len(result["metadata_cases"])) + ".json")
            payload = {"schema_version": 1, "kind": tool.PATCHES_KIND, "statement": tool.PATCHES_STATEMENT, "patches": [item]}
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False))
            reference = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
            row = {"name":name,"fixture":payload,"expected_rejected":rejected}
            try:
                row["observed"] = tool.author_patches(ROOT, {"author_patches":reference}, {"sample":{"source":"sample.md"}})
                row["rejected"] = False
            except ValueError as error:
                row["error"] = str(error)
                row["rejected"] = True
            row["passed"] = row["rejected"] == rejected
            result["metadata_cases"].append(row)
            assert row["passed"], name
        metadata_case("正確格式接受", base_patch)
        metadata_case("超過文字長度上限拒絕", {**base_patch,"new":"字"*(tool.PATCH_LIMIT+1)}, True)
        metadata_case("跨行拒絕", {**base_patch,"new":"目前\n文字"}, True)
        metadata_case("完整連結語法作為替換片段拒絕", {**base_patch,"new":"[文字](url)"}, True)
        metadata_case("非作者聲明拒絕", {**base_patch,"requested_by":"agent"}, True)
        uv = ["uv","run","--no-sync","python"]
        for args in (["scripts/build_course.py","--check"], ["scripts/fetch_training_assets.py","--list"], ["scripts/fetch_course_models.py","--list"]):
            result["commands"].append(run(uv+args))
        result["commands"].append(run(uv+["scripts/prepare_data.py","--kind","toy-text","--output",str(tmp/"data")]))
        data = tmp/"data/toy-text/train.jsonl"
        target = tmp/"unexpected.pt"
        command = uv+["scripts/train.py","--task","text","--data",str(data),"--device","cpu","--output",str(target)]
        result["commands"].append(run(command))
        result["inputs"] = {p.name:p.read_text() for p in sorted(data.parent.iterdir())}
        from scripts import train
        actual = torch.optim.AdamW
        optimizers = []
        class CheckedAdamW(actual):
            def __init__(self, params, *args, **kwargs):
                params = list(params)
                self.checked = params
                self.before = [p.detach().clone() for p in params]
                optimizers.append(self)
                super().__init__(params,*args,**kwargs)
            def step(self,*args,**kwargs):
                raise AssertionError("dry-run不應更新參數")
        output = io.StringIO()
        argv = command[4:]
        with patch.object(torch.optim,"AdamW",CheckedAdamW),patch.object(sys,"argv",argv),contextlib.redirect_stdout(output):
            train.main()
        result["dry_run"] = {"stdout":output.getvalue(),"argv":argv,"parameters_unchanged":all(torch.equal(p,b) for o in optimizers for p,b in zip(o.checked,o.before)),"checkpoint_exists":target.exists()}
        assert result["dry_run"]["parameters_unchanged"] and not target.exists()
    validator = load("readme_validator_current", ROOT/"scripts/check_technical_reviews.py")
    current = copy.deepcopy(previous)
    for source in current["sources"]:
        if source.get("path") in changed:
            source["sha256"] = sha(ROOT/source["path"])
    result["validator_errors_after_source_refresh"] = validator._validate(ROOT, ROOT/"README.md", "readme", (ROOT/"README.md").read_text(), current, {current["reviewer_task"]:["readme"]}, {"/root/rv1019_reader_entry"})
    assert not result["validator_errors_after_source_refresh"]
    assert all(r["returncode"]==0 for r in result["commands"])
    target = HERE/"check-current-03.json"
    with target.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target),"changed_sources":changed,"unchanged_artifacts":len(result["artifacts"]),"patch_cases":len(result["patch_cases"]),"metadata_cases":len(result["metadata_cases"]),"dry_run":result["dry_run"]},ensure_ascii=False))

if __name__=="__main__":
    main()
