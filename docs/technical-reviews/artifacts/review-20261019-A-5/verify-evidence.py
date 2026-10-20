"""以保存的原文、輸入 ID 與逐題輸出重算診斷例子，不重新訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_code(code):
    stream = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "人工診斷例子", "exec"), namespace)
    return stream.getvalue(), namespace


frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/A.5.md"
frozen = frozen_path.read_text()
chapter = (ROOT / "course/chapters/0A.md").read_text()
live = re.search(r"^## A\.5 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
require(live == frozen, "目前小節必須等於凍結原文")
code = re.search(r"```python\n(.*?)```", frozen, re.S)[1]
stdout, namespace = run_code(code)
require(namespace["hits"] == 2 and namespace["correct"] == 1, "人工例子的計數不符")
needle = '{"question": "乙店地址", "hit": True, "correct": False}'
require(code.count(needle) == 1, "練習應只替換乙的正確性欄位")
modified_stdout, modified = run_code(code.replace(needle, needle.replace('"correct": False', '"correct": True')))
require(modified["hits"] == 2 and modified["correct"] == 2, "修改乙後的計數不符")
require("乙店地址" not in modified_stdout and "丙店地址" in modified_stdout, "答錯清單不符")

record_path = ROOT / "docs/course-experiments/results/rag.json"
record = json.loads(record_path.read_text())
result = record["results"]
rows = [r for r in result["samples"] if r["mode"] in {"correct_context", "retrieved_context"}]
require(len(rows) == 24, "兩種條件應包含相同題目的完整生成")
derived = {}
audited_rows = []
for mode in ["correct_context", "retrieved_context"]:
    selected = [r for r in rows if r["mode"] == mode]
    families = sorted({r["family"] for r in selected})
    require(len(selected) == len(families) == 12, "每種條件應各有十二個不重複店名")
    totals = {"questions": len(selected), "present": 0, "fact": 0, "rule": 0, "absent_unknown": 0}
    for row in selected:
        key = row["family"]
        fact = row["fact"]
        fact_answer = f"{fact['address']}[{fact['source']}]"
        matching = [d for d in row["documents"] if d["text"] == f"{key} address={fact['address']}" and d["id"] == fact["source"]]
        present = bool(matching)
        require(len(matching) <= 1, "每題最多只有一份所需公告")
        expected = fact_answer if present else "UNKNOWN"
        generation = row["generation"]
        samples = generation["samples"]
        require(len(samples) == generation["candidate_count"] == 1, "每題只保留一份生成")
        sample = samples[0]
        messages = generation["messages"]
        require(len(messages) == 1 and messages[0]["role"] == "user", "檢查實際輸入的訊息格式")
        document_text = "\n".join(f"[{d['id']}] {d['text']}" for d in row["documents"]) or "EMPTY"
        expected_message = f"Docs:\n{document_text}\nFind:{key}\nReply:address[source] or UNKNOWN"
        require(messages[0]["content"] == expected_message, "公告與問題必須完整出現在模型輸入")
        expected_ids = [1, 3] + [b + 8 for b in expected_message.encode("utf-8")] + [2, 4]
        require(generation["input_ids"] == expected_ids, "輸入 ID 應與 ByteTokenizer 的完整編碼相同")
        require(generation["input_tokens"] == len(expected_ids), "前文長度不符")
        require(len(expected_ids) + generation["max_new_tokens"] <= result["model"]["config"]["max_length"], "輸入與答案預留不得超過視窗")
        ids = sample["generated_ids"]
        require(ids[-1] == 2 and sample["eos"], "保存的生成應以 EOS 結束")
        invalid = [v for v in ids[:-1] if v < 8]
        require(invalid == sample["invalid_special_tokens"] == [], "生成不可混入控制 token")
        generated = bytes(v - 8 for v in ids[:-1]).decode("utf-8")
        require(generated == sample["generated"], "生成 ID 與文字必須一致")
        rule_correct = generated == expected
        fact_correct = generated == fact_answer
        unknown = generated == "UNKNOWN"
        retrieval_hit = f"doc-{key}" in row["retrieved_ids"]
        require(retrieval_hit == row["retrieval_hit"], "retrieval_hit 應表示持久文件編號命中")
        require(row["expected"] == expected, "預期答案應依公告是否在輸入中決定")
        require(row["exact_match"] == rule_correct, "依規則答對的旗標不符")
        require(row["fact_answer_correct"] == fact_correct, "地址與來源都正確的旗標不符")
        require(row["unknown"] == unknown, "UNKNOWN 旗標不符")
        if mode == "retrieved_context":
            require(present == retrieval_hit, "實際檢索條件下文件命中應等於輸入含所需公告")
        totals["present"] += present
        totals["fact"] += fact_correct
        totals["rule"] += rule_correct
        totals["absent_unknown"] += not present and unknown
        audited_rows.append({"family": key, "mode": mode, "documents": row["documents"], "present": present, "retrieval_hit": retrieval_hit, "expected": expected, "generated": generated, "fact_correct": fact_correct, "rule_correct": rule_correct, "input_tokens": len(expected_ids)})
    for field, count in [("exact_match", totals["rule"]), ("fact_answer_correct", totals["fact"])]:
        metric = result["metrics"][mode][field]
        require(metric["numerator"] == count and metric["denominator"] == 12, "逐題重算與表列指標不符")
        require(abs(metric["rate"] - count / 12) <= 1e-15, "比例與分子分母不符")
    derived[mode] = {"families": families, **totals}

require(derived["correct_context"]["families"] == derived["retrieved_context"]["families"], "兩條件必須使用同一批店名")
require([derived["correct_context"][k] for k in ["present", "fact", "rule"]] == [12, 5, 5], "直接公告表格不符")
require([derived["retrieved_context"][k] for k in ["present", "fact", "rule", "absent_unknown"]] == [9, 3, 6, 3], "實際檢索表格不符")
require(derived["correct_context"]["questions"] - derived["correct_context"]["rule"] == 7, "正確公告條件的錯誤題數不符")
k017 = next(r for r in audited_rows if r["family"] == "K017" and r["mode"] == "correct_context")
require(k017["expected"] == k017["generated"] == "C2[D1]", "短代號算例不符")

code_hashes = {}
for name in ["scripts/course_experiments/applications.py", "tiny_perceptron/data.py", "tiny_perceptron/retrieval.py"]:
    digest = sha(ROOT / name)
    code_hashes[name] = {"current_sha256": digest, "recorded_sha256": record["code_sha256"].get(name), "matches": digest == record["code_sha256"].get(name)}
pinned = subprocess.run(["git", "show", record["revision"] + ":scripts/course_experiments/applications.py"], cwd=ROOT, capture_output=True, text=True)
linked = {key: hashlib.sha256(re.search(r"^## " + re.escape(key) + r" .*?(?=^## |\Z)", chapter, re.M | re.S)[0].encode()).hexdigest() for key in ["A.2", "A.7", "A.10"]}

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-5/verify-evidence.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; 純 Python 重算，未執行模型訓練或推論"},
    "result": "所有原文、人工例子、二十四份生成的輸入與輸出、指標及分母核對通過",
    "source_sha256": sha(frozen_path),
    "record_path": str(record_path.relative_to(ROOT)),
    "record_sha256": sha(record_path),
    "toy_original_stdout": stdout,
    "toy_modified_stdout": modified_stdout,
    "derived_metrics": derived,
    "audited_rows": audited_rows,
    "recorded_training_environment": {k: record[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]},
    "recorded_training": result["training"],
    "recorded_split": result["split"],
    "model": result["model"],
    "linked_sections_sha256": linked,
    "code_hashes": code_hashes,
    "historical_source_probe": {"command": "git show " + record["revision"] + ":scripts/course_experiments/applications.py", "returncode": pinned.returncode, "stderr": pinned.stderr.strip()},
    "limitations": ["重算保存的紀錄不能重新證明 GPU 訓練與生成確實發生；本次沒有重跑訓練。", "目前 applications.py 的全檔 SHA 與實驗紀錄不同，紀錄內 revision 的該路徑也無法以 git show 取得；目前程式僅支持現行計分規則，歷史觀察由保存的逐題輸入與輸出支持。", "測試只有單一 seed 的十二份架空短公告，不能推廣為一般文件問答的能力分數。"]
}, ensure_ascii=False, indent=2))
