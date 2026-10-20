"""核對文件檢索示範及已保存的逐題生成紀錄，不重新訓練模型。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.retrieval import lexical_terms, retrieve
from scripts.course_experiments.applications import _rag_splits, _split_summary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(documents):
    hits = retrieve("書店地址", documents)
    if not hits:
        result = "沒有取回相關來源，請補公告"
    elif "尚未公布" in hits[0]["text"]:
        result = "來源未給地址，目前無法確定"
    else:
        result = "需進一步核對來源與答案"
    return {"ids": [doc["id"] for doc in hits], "decision": result}


cases = [
    ([{"id": "d1", "text": "貓在窗邊"}], [], "沒有取回相關來源，請補公告"),
    ([{"id": "d2", "text": "書店地址尚未公布"}], ["d2"], "來源未給地址，目前無法確定"),
    ([{"id": "d2", "text": "書店地址是青街8號"}], ["d2"], "需進一步核對來源與答案"),
    ([{"id": "d2", "text": "書店地址仍未公開"}], ["d2"], "需進一步核對來源與答案"),
]
example_results = []
for docs, ids, expected in cases:
    actual = decision(docs)
    assert actual == {"ids": ids, "decision": expected}
    terms = Counter(lexical_terms("書店地址"))
    scores = [sum(min(Counter(lexical_terms(doc["text"]))[t], n) for t, n in terms.items()) for doc in docs]
    example_results.append({"documents": docs, "scores": scores, **actual})

control_flow = {
    "not_empty_list": not [],
    "not_nonempty_list": not [{"text": "尚未公布"}],
    "first_document_text": [{"text": "尚未公布"}][0]["text"],
    "exact_substring": "尚未公布" in "書店地址尚未公布",
    "paraphrase_substring": "尚未公布" in "書店地址仍未公開",
}
assert control_flow == {
    "not_empty_list": True,
    "not_nonempty_list": False,
    "first_document_text": "尚未公布",
    "exact_substring": True,
    "paraphrase_substring": False,
}
visited = []
if True:
    visited.append("first")
elif visited.append("elif_condition"):
    visited.append("second")
assert visited == ["first"]
control_flow["elif_after_true_first_condition"] = visited

dated = [
    {"id": "a", "text": "書店地址青街8號", "date": "2026-01-01"},
    {"id": "z", "text": "書店地址紅街9號", "date": "2026-10-20"},
]
ranked = [doc["id"] for doc in retrieve("書店地址", dated)]
dated[0]["date"], dated[1]["date"] = dated[1]["date"], dated[0]["date"]
assert ranked == [doc["id"] for doc in retrieve("書店地址", dated)] == ["a", "z"]

path = ROOT / "docs/course-experiments/results/rag.json"
record = json.loads(path.read_bytes())
result = record["results"]
rows = result["samples"]
compact = []
conflicting_document_sets = 0
for row in rows:
    generation = row["generation"]
    assert generation["candidate_count"] == 1
    sample = generation["samples"][0]
    actual_unknown = sample["generated"] == "UNKNOWN" and not sample["invalid_special_tokens"]
    assert actual_unknown == row["unknown"]
    assert generation["temperature"] == 0.0
    assert generation["max_new_tokens"] == 16
    values_by_key = {}
    for document in row["documents"]:
        parsed = re.fullmatch(r"([KZ][0-9]{3}) address=([A-D][0-9])", document["text"])
        assert parsed is not None, "文件需符合保存實驗的地址欄位格式"
        values_by_key.setdefault(parsed[1], set()).add(parsed[2])
    conflicting_document_sets += any(len(values) > 1 for values in values_by_key.values())
    compact.append({
        "family": row["family"],
        "mode": row["mode"],
        "fact": row["fact"],
        "documents": row["documents"],
        "prompt": generation["messages"],
        "generated": sample["generated"],
        "invalid_special_tokens": sample["invalid_special_tokens"],
        "unknown": actual_unknown,
        "expected": row["expected"],
    })

recomputed = {}
for mode in sorted({row["mode"] for row in rows}):
    selected = [row for row in rows if row["mode"] == mode]
    numerator = sum(row["unknown"] for row in selected)
    count = len(selected)
    metric = {"numerator": numerator, "denominator": count, "rate": numerator / count}
    assert metric == result["metrics"][mode]["unknown"]
    recomputed[mode] = metric
    for row in selected:
        key = row["fact"]["key"]
        docs = row["documents"]
        if mode == "without_context":
            assert docs == []
        elif mode == "distractor_only":
            assert len(docs) == 1 and not docs[0]["text"].startswith(key + " ")
        elif mode == "correct_context":
            assert len(docs) == 1
            assert docs[0]["text"] == f"{key} address={row['fact']['address']}"

assert recomputed["without_context"]["numerator"] == 12
assert recomputed["distractor_only"]["numerator"] == 7
assert recomputed["correct_context"]["numerator"] == 7
assert all(recomputed[mode]["denominator"] == 12 for mode in recomputed)
assert conflicting_document_sets == 0

splits = _rag_splits(record["seed"])
split_summary = _split_summary(splits)
assert split_summary == result["split"]
training_unknown = sum(row["messages"][-1]["content"] == "UNKNOWN" for row in splits["train"])
assert training_unknown == 192

application = ROOT / "scripts/course_experiments/applications.py"
application_text = application.read_text()
functions = {}
for node in ast.parse(application_text).body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_rag_question", "_rag_records", "_rag_splits", "_grounding", "_rag_counterfactual_contexts", "run_rag"
    }:
        source = ast.get_source_segment(application_text, node)
        functions[node.name] = hashlib.sha256(source.encode()).hexdigest()
old = subprocess.run(
    ["git", "show", record["revision"] + ":scripts/course_experiments/applications.py"],
    cwd=ROOT, capture_output=True, text=True,
)

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-6/verify.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "torch": __import__("torch").__version__},
    "inputs": {str(p.relative_to(ROOT)): digest(p) for p in [path, application, ROOT / "tiny_perceptron/retrieval.py"]},
    "retrieval_examples": example_results,
    "control_flow": control_flow,
    "date_metadata_swap_ranking": ranked,
    "experiment_metadata": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status", "step_scale")},
    "training": result["training"],
    "model": result["model"],
    "split": split_summary,
    "training_unknown_targets": {"numerator": training_unknown, "denominator": len(splits["train"]), "split_digest_matches_record": True},
    "evaluation": {"sample_rows": len(rows), "unique_families": len({row["family"] for row in rows}), "candidate_count": 1, "temperature": 0.0, "max_new_tokens": 16, "conflicting_document_sets": conflicting_document_sets, "all_documents_are_single_ascii_address_fields": True},
    "unknown_recomputed": recomputed,
    "samples": compact,
    "current_rag_function_sha256": functions,
    "code_provenance_limit": {
        "recorded_application_sha256": record["code_sha256"]["scripts/course_experiments/applications.py"],
        "current_application_sha256": digest(application),
        "historical_git_show_exit_code": old.returncode,
        "historical_git_show_stderr": old.stderr,
        "scope": "本次重算保存的生成紀錄並核對現行產生器的資料雜湊；未重新訓練，也未以現行整檔冒充歷史執行版本。",
    },
    "result": "全部斷言成立；三個指定條件的分子、分母與文件條件均符合保存紀錄。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
