"""核對修訂原文、既有證據與小節程式；只建立新的複查證據。"""

import difflib
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
DEST = Path(__file__).resolve().parent
ORIGINAL = Path("docs/course-review-20261019/reviews/rv1019_technical_20-3/technical-20-3-original-01.json")
PREVIOUS_SOURCE = Path("docs/course-review-20261019/freeze-01/sources/20.3.md")
CURRENT_SOURCE = Path("docs/course-review-20261019/freeze-05/sources/20.3.md")
FREEZE = Path("docs/course-review-20261019/freeze-06.json")
RUN = Path("outputs/reviewer-tools/runs/20261020T101624496865Z")


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def checked(path, expected):
    actual = digest(path)
    assert actual == expected, f"證據內容已改變：{path}"
    return {"path": str(path), "expected_sha256": expected, "sha256": actual, "unchanged": True}


original_hash = "d74df764f422d9541b72b1b9e53c3322fa0d29097413725fa7987c1238f66d96"
original_receipt = checked(ORIGINAL, original_hash)
report = read_json(ORIGINAL)
old = (ROOT / PREVIOUS_SOURCE).read_bytes()
current = (ROOT / CURRENT_SOURCE).read_bytes()
assert hashlib.sha256(old).hexdigest() == report["source_sha256"]
assert hashlib.sha256(current).hexdigest() == "87dc549532ea1117c3c17e039bd7a82497ddd4f74ed7e37bdecd18ae6ea8ff91"
assert digest(FREEZE) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
old_phrase = "`round(..., 2)` 四捨五入到小數兩位。"
new_phrase = "`round(..., 2)` 取到小數兩位。"
assert old.count(old_phrase.encode()) == 1
assert current == old.replace(old_phrase.encode(), new_phrase.encode(), 1)
freeze_page = next(item for item in read_json(FREEZE)["pages"] if item["page_id"] == "20.3")
assert freeze_page["source_sha256"] == digest(CURRENT_SOURCE)
assert freeze_page["figures_sha256"] == {}
assert (ROOT / freeze_page["snapshot"]).read_bytes() == current
assert (ROOT / RUN / "section.md").read_bytes() == current

reuse = [checked(item["path"], item["sha256"]) for item in report["artifacts"]]
reuse.append(checked("docs/technical-reviews/artifacts/review-20261019-20-3/self-check.json", "2114b53336562976f2c18613bf2a9cbf4a783b1a5b329c23f3aea9c4e185f658"))
verification = read_json("docs/technical-reviews/artifacts/review-20261019-20-3/verification.json")
repository_sources = [checked(item["path"], item["sha256"]) for item in report["sources"] if item["kind"] == "repository_code"]
records = [checked(item["path"], item["sha256"]) for item in verification["records"].values()]
repository_inputs = [checked(item["path"], item["sha256"]) for item in verification["repository_excerpts"].values()]
primary = read_json("docs/technical-reviews/artifacts/review-20261019-20-3/primary-sources.json")
round_source = primary["sources"]["python_functions"]
round_excerpt = round_source["excerpts"]["round"]
assert "rounded to *ndigits* precision after the decimal" in round_excerpt
assert "rounding is done toward the even choice" in round_excerpt

section = {"origin": str(RUN), "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/20.md#20.3' --execute", "files": {}}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (ROOT / RUN / name).read_bytes()
    section["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode()}
execution = json.loads(section["files"]["execution.json"]["text"])
extraction = json.loads(section["files"]["extraction.json"]["text"])
assert execution["exit_code"] == 0 and execution["source_sha256"] == digest(CURRENT_SOURCE)
assert extraction["figure_sha256"] == {} and len(extraction["python_fences"]) == 1
previous_section = read_json("docs/technical-reviews/artifacts/review-20261019-20-3/section-execution.json")
for name in ("fence-1.py", "stdout.txt", "stderr.txt"):
    assert section["files"][name]["sha256"] == previous_section["files"][name]["sha256"]

round_cases = [(7.9256744384765625, 7.93), (3.9628372192382812, 3.96), (0.125, 0.12), (0.375, 0.38), (1.0, 1.0), (-0.125, -0.12)]
round_results = []
for value, expected in round_cases:
    actual = round(value, 2)
    assert actual == expected
    round_results.append({"input": value, "ndigits": 2, "expected": expected, "observed": actual, "input_exact_ratio": list(value.as_integer_ratio())})
qwen = primary["sources"]["qwen_metadata"]["selected_fields"]["safetensors"]["total"]
whisper = primary["sources"]["whisper_metadata"]["selected_fields"]["safetensors"]["total"]
prepare = read_json(verification["records"]["prepare"]["path"])
training = read_json(verification["records"]["training"]["path"])
ui = read_json(verification["records"]["ui"]["path"])
snapshot_bytes = sum(item["bytes"] for snapshot in prepare["snapshots"].values() for item in snapshot["files"])
lora = 28 * 8 * ((2048 + 2048) + (2048 + 1024))
arithmetic = {
    "qwen_parameters": qwen,
    "qwen_float32_bytes": qwen * 4,
    "qwen_float32_gib": qwen * 4 / 1024**3,
    "qwen_float16_bytes": qwen * 2,
    "qwen_float16_gib": qwen * 2 / 1024**3,
    "whisper_parameters": whisper,
    "whisper_float32_gib": whisper * 4 / 1024**3,
    "whisper_float16_gib": whisper * 2 / 1024**3,
    "combined_parameters": qwen + whisper,
    "snapshot_bytes": snapshot_bytes,
    "snapshot_bytes_per_parameter": snapshot_bytes / (qwen + whisper),
    "cpu_rss_gib": ui["server_cpu_resources"]["maximum_resident_set_kib"] / 1024**2,
    "lora_parameters": lora,
    "gradient_m_v_float32_bytes": lora * 4 * 3,
    "gradient_m_v_float32_gib": lora * 4 * 3 / 1024**3,
    "peak_cuda_allocated_gib": training["resources"]["peak_cuda_memory_allocated_bytes"] / 1024**3,
}
for name, actual in arithmetic.items():
    assert actual == verification["arithmetic"][name], f"算式與既有結果不同：{name}"
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-3/recheck-current-01.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "section_torch": json.loads(section["files"]["environment.json"]["text"])["torch"]},
    "original_report": original_receipt,
    "source": {"path": str(CURRENT_SOURCE), "sha256": digest(CURRENT_SOURCE), "only_changed_phrase": {"before": old_phrase, "after": new_phrase}, "exact_replacement_verified": True},
    "source_freeze": {"path": str(FREEZE), "sha256": digest(FREEZE)},
    "diff": "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), current.decode().splitlines(keepends=True), fromfile=str(PREVIOUS_SOURCE), tofile=str(CURRENT_SOURCE))),
    "unchanged_artifacts": reuse,
    "unchanged_repository_sources": repository_sources,
    "unchanged_measurement_records": records,
    "unchanged_repository_inputs": repository_inputs,
    "changed_statement_evidence": {"current_statement": new_phrase, "official_url": round_source["url"], "official_version": "CPython v3.13.0/Doc/library/functions.rst", "receipt_artifact_path": "docs/technical-reviews/artifacts/review-20261019-20-3/primary-sources.json", "receipt_artifact_sha256": digest(Path("docs/technical-reviews/artifacts/review-20261019-20-3/primary-sources.json")), "locator": "round(number, ndigits=None)", "inspection_note": "重新讀取上次自行擷取的官方 round 原文：ndigits 指小數點後精度；新版刪除了不正確的五入規則，與這個契約相符。數值取到兩位不是保證字串固定顯示兩位，也沒有改成截斷規則。", "round_cases": round_results, "verdict": "verified"},
    "arithmetic_recalculated": arithmetic,
    "section_execution": section,
    "claim_inventory": [{"id": item["id"], "kind": item["kind"], "changed": item["status"] == "contradicted", "recheck_result": "verified"} for item in report["claims"]],
    "claim_kinds": dict(Counter(item["kind"] for item in report["claims"])),
    "independence": "僅讀原文、自己的原報告與證據、其中明確指名的量測檔及程式；outputs只讀本回合section_facts產生並明確回報的檔案。未讀其他角色報告。",
    "result": "修訂敘述與官方契約相符；其餘原文、證據與相關實作內容未變，所有重算與新版小節執行通過。",
    "limits": "沒有重新量測大型模型CPU服務或L4訓練，也沒有讀歷史GPU原始輸出或私有權重；資源主張仍限於保存紀錄與成功驗證收據。",
}
with (DEST / "recheck-current-01.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"result": "verified", "changed_phrases": 1, "unchanged_artifacts": len(reuse), "unchanged_measurement_records": len(records), "numeric_rechecks": len(arithmetic), "section_exit_code": execution["exit_code"]}, ensure_ascii=False))
