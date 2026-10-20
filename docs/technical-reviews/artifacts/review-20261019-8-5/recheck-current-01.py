import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_8-5/technical-8-5-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    match = re.search(rb"(?ms)^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)", raw)
    assert match is not None, "找不到指定小節"
    return match.group()


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True)
    return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def execute_fence(body, filename):
    code = re.search(r"```python\n(.*?)```", body.decode(), re.S).group(1)
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, filename, "exec"), namespace)
    return {"code_sha256": sha(code.encode()), "stdout": output.getvalue()}, namespace


original_raw = ORIGINAL.read_bytes()
assert sha(original_raw) == "596c605b2d3078211d5e3e598d2eb98bb3543b11ef170d47f927f80e7e93a192", "原報告已改變"
original = json.loads(original_raw)
assert original["issues"] == [], "原報告問題清單與預期不同"
repo_versions = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = sha((ROOT / source["path"]).read_bytes())
        repo_versions.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
        if source["id"] == "s-prerequisite":
            assert actual == "4b02242d4c281a8a0554a3a87bdc898c040bb7afa3250dc8c055184407952174", "前置章檔版本不同"
        else:
            assert actual == source["sha256"], "其他程式來源已改變"
artifact_versions = []
for artifact in original["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], "沿用證據的內容已改變"
    artifact_versions.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

versions = {}
for revision in ("5ef56bc756f3", "3af7f13ef959"):
    result = command(["git", "show", revision + ":course/chapters/07.md"])
    assert result["exit_code"] == 0, "無法讀取指定前置版本"
    raw = result["stdout"].encode()
    versions[revision] = {"file_sha256": sha(raw), "prerequisite_section_sha256": sha(section(raw, "7.13"))}
old_prerequisite = versions["5ef56bc756f3"]["prerequisite_section_sha256"]
new_prerequisite = section((ROOT / "course/chapters/07.md").read_bytes(), "7.13")
assert versions["5ef56bc756f3"]["file_sha256"] == "4f8f31898954117a1e8203d98269da246ec4b7794d8997dd5264055945761ff9"
assert versions["3af7f13ef959"]["file_sha256"] == sha((ROOT / "course/chapters/07.md").read_bytes())
assert old_prerequisite == sha(new_prerequisite) == versions["3af7f13ef959"]["prerequisite_section_sha256"], "引用小節有變動"
diff = command(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/07.md"])
assert diff["exit_code"] == 0
prerequisite_run, prerequisite_namespace = execute_fence(new_prerequisite, "course/chapters/07.md")
assert prerequisite_namespace["parsed"] == {"answer": 3}, "前置範例未解析成預期字典"

frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/8.5.md"
frozen = frozen_path.read_bytes()
current = section((ROOT / "course/chapters/08.md").read_bytes(), "8.5")
assert current == frozen, "現行課文與凍結原文不同"
assert sha(frozen) == original["source_sha256"]
section_run, _ = execute_fence(frozen, str(frozen_path.relative_to(ROOT)))

run_path = (ROOT / sys.argv[1]).resolve()
assert run_path.is_relative_to(ROOT / "outputs/reviewer-tools/runs"), "執行證據路徑不同"
run_files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_path / name).read_bytes()
    run_files[name] = {"sha256": sha(raw), "content": raw.decode()}
extraction = json.loads(run_files["extraction.json"]["content"])
execution = json.loads(run_files["execution.json"]["content"])
assert execution["exit_code"] == 0
assert extraction["source_sha256"] == sha(frozen)
assert extraction["figure_sha256"] == {}
assert len(extraction["python_fences"]) == 1
assert run_files["stdout.txt"]["content"] == section_run["stdout"]
old_check = json.loads((ARTIFACT / "verification.json").read_bytes())
assert section_run["stdout"] == old_check["original_fence_stdout"]

cases = []
for previous in old_check["cases"]:
    error, value = None, None
    try:
        value = json.loads(previous["input"])
        valid = isinstance(value, dict) and set(value) == {"answer"}
        correct = valid and value["answer"] == 4
    except json.JSONDecodeError as exception:
        valid = correct = False
        error = type(exception).__name__
    observed = {"input": previous["input"], "parsed_type": type(value).__name__, "parsed_repr": repr(value), "format": valid, "content": correct, "error": error}
    assert observed == previous, "額外案例與原判斷不同"
    cases.append(observed)

record_path = ROOT / old_check["experiment"]["path"]
assert sha(record_path.read_bytes()) == old_check["experiment"]["sha256"], "保存實驗資料有變動"
record = json.loads(record_path.read_bytes())
samples = [item for item in record["results"]["after"]["test"]["samples"] if item.get("style") == "json"]
checked = []
for sample in samples:
    a, b = map(int, re.fullmatch(r"style=json; (\d+)\+(\d+)=\?", sample["messages"][0]["content"]).groups())
    parsed = json.loads(sample["generated"])
    integer = isinstance(parsed, dict) and set(parsed) == {"answer"} and type(parsed["answer"]) is int
    correct = integer and parsed["answer"] == a + b
    assert json.loads(sample["expected"]) == {"answer": a + b}
    assert [byte + 8 for byte in sample["generated"].encode()] + [2] == sample["generated_ids"]
    checked.append({"question": sample["messages"][0]["content"], "generated": sample["generated"], "sum": a + b, "exact_integer_and_keys": integer, "correct": correct, "ids_match": True})
counts = {"records": len(checked), "format_and_integer": sum(item["exact_integer_and_keys"] for item in checked), "content_correct": sum(item["correct"] for item in checked)}
assert counts == old_check["experiment"]["recomputed_counts"]

claim_notes = {
    "c1": "重新計算整數加法，2+2=4；值為 5 不符。",
    "c2": "前置小節的原始 bytes 在兩個指定版本及現行檔完全相同；json.loads(reply) 實跑仍得到 dict。",
    "c3": "前綴字串仍拋 JSONDecodeError，未先刪除前綴。",
    "c4": "指定工具重跑及凍結 fence 獨立執行，三行 stdout 均與原紀錄相同。",
    "c5": "第二筆錯誤後仍執行第三筆，except 行為未變。",
    "c6": "重新執行 object、array、number 根值案例，isinstance 判定均與原紀錄相同。",
    "c7": "重新執行空字典、缺鍵與多 note 欄位案例，集合相等判定不變。",
    "c8": "NaN、Infinity 案例依舊可解析；沿用未變的官方短摘錄與 RFC 證據。",
    "c9": "重新核對格式短路、4.0 與字串 4，內容欄僅做值相等而未加精確 int 限制。",
    "c10": "note 練習重跑仍為格式 False、內容 False。",
    "c11": "原始 fence 是固定 literal，無模型呼叫、strip 或抽取前處理；bytes 未變。",
    "c12": "可解析而錯值的反例仍成立；原文未改任務與評估定義，沿用原邏輯推導及官方語法來源。",
    "c13": "重新解析保存七題；欄位、精確 int 與生成 ID 全部吻合，沒有擴大為重現訓練。",
    "c14": "重新從七個題目求和，七個 answer 全部不等於正確和。",
}
assert set(claim_notes) == {claim["id"] for claim in original["claims"]}
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + sys.argv[1],
    "result": "全部實質主張仍成立；前置引用小節逐位元組未變；三筆原文輸出、13 個案例及七題保存結果重算均相符。",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "parser": "Python stdlib json"},
    "original_report_sha256": sha(original_raw),
    "original_issues": original["issues"],
    "source_sha256": sha(frozen),
    "source_identical_to_freeze": current == frozen,
    "source_figures": extraction["figure_sha256"],
    "repository_source_versions": repo_versions,
    "retained_artifact_versions": artifact_versions,
    "prerequisite_versions": versions,
    "prerequisite_section_current_sha256": sha(new_prerequisite),
    "prerequisite_code_execution": prerequisite_run,
    "prerequisite_diff": diff,
    "changed_sentences_in_reviewed_page": [],
    "changed_dependency_scope": "差異位於其他小節的 TinyLM/labels、PAD 位置查表與 PPO 段落；這些句子不是本頁證據。引用小節原始 bytes 未變，不新增對範圍外句子的技術裁決。",
    "section_facts_command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.5" --execute',
    "section_execution_copied_files": run_files,
    "independent_section_run": section_run,
    "cases": cases,
    "experiment_record_sha256": sha(record_path.read_bytes()),
    "experiment_checked_samples": checked,
    "experiment_counts": counts,
    "numeric_sum": 2 + 2,
    "claim_rechecks": [{"claim_id": claim["id"], "kind": claim["kind"], "status": "verified", "details": claim_notes[claim["id"]]} for claim in original["claims"]],
    "reused_external_evidence": "沿用原報告已實讀的官方來源與 a-docs；未重新下載第三方全文。",
    "limits": "未重新訓練或從權重推論；不把保存實驗紀錄當成歷史訓練環境重現。",
}
with (ARTIFACT / "recheck-current-01.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({"result": report["result"], "claim_count": len(claim_notes), "prerequisite_section_sha256": sha(new_prerequisite), "experiment_counts": counts}, ensure_ascii=False))
