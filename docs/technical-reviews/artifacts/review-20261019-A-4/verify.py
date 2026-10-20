"""離線重算儲存的引用評分，並驗證地址字串示例。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "docs/course-experiments/results/rag.json"
APPLICATIONS = ROOT / "scripts/course_experiments/applications.py"
SECTION_RECORD = Path(__file__).with_name("section-execution.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    section = json.loads(SECTION_RECORD.read_text(encoding="utf-8"))
    code = section["copied_files"]["fence-1.py"]["content"]
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "original_address_example", "exec"), {})
    expected_stdout = (
        "青街8號 引用存在 True 地址在原文 True\n"
        "紅街9號 引用存在 True 地址在原文 False\n"
    )
    check(stdout.getvalue() == expected_stdout, "原始地址示例輸出不符")

    toy_source = {"id": "d2", "text": "書店地址是青街8號"}
    toy_answers = [
        {"address": "青街8號", "citation": "d2"},
        {"address": "紅街9號", "citation": "d2"},
        {"address": "青街8號", "citation": "d9"},
    ]
    toy_results = []
    for answer in toy_answers:
        valid = answer["citation"] == toy_source["id"]
        supports = valid and answer["address"] in toy_source["text"]
        toy_results.append(answer | {"valid": valid, "supports": supports})
    check(
        [(x["valid"], x["supports"]) for x in toy_results]
        == [(True, True), (True, False), (False, False)],
        "地址與引用的布林結果不符",
    )
    negative = {"text": "書店地址不是青街8號", "asserted_address": "青街8號"}
    negative["substring_present"] = negative["asserted_address"] in negative["text"]
    paraphrase = {"text": "書店位於青街八號", "asserted_address": "青街8號"}
    paraphrase["substring_present"] = paraphrase["asserted_address"] in paraphrase["text"]
    check(negative["substring_present"], "否定句的字串比對結果不符")
    check(not paraphrase["substring_present"], "同義地址的字串比對結果不符")

    tree = ast.parse(APPLICATIONS.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_grounding", "_rag_question"}
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(APPLICATIONS), "exec"), namespace)
    grounding = namespace["_grounding"]
    rag_question = namespace["_rag_question"]
    rows = record["results"]["samples"]
    checked_rows = []
    for row in rows:
        sample = row["generation"]["samples"][0]
        generated = sample["generated"]
        match = re.fullmatch(r"([A-E][0-9])\[(D[0-9])\]", generated)
        address, citation = match.groups() if match else (None, None)
        clean = not sample["invalid_special_tokens"]
        documents = row["documents"]
        valid = clean and citation in {d["id"] for d in documents}
        supported = clean and bool(match) and any(
            d["id"] == citation and f"address={address}" in d["text"] for d in documents
        )
        fact = row["context_fact"]
        fact_answer = f"{fact['address']}[{fact['source']}]"
        correct = clean and generated == fact_answer
        address_correct = clean and address == fact["address"]
        check(valid == row["citation_valid"], "重算的引用存在評分不符")
        check(supported == row["supported_by_cited_source"], "重算的地址支持評分不符")
        check(correct == row["fact_answer_correct"], "重算的事實答案評分不符")
        check(address == row["parsed_address"], "重算的地址解析不符")
        check(citation == row["parsed_citation"], "重算的來源解析不符")
        observed = grounding(sample, documents, row["expected"])
        check(all(row[k] == v for k, v in observed.items()), "現行評分函式與已存評分不符")
        prompt = rag_question(fact["key"], documents)
        check(prompt == row["generation"]["messages"][0]["content"], "公告內容與實際提示不符")
        ids = sample["generated_ids"]
        raw_ids = ids[:-1] if ids and ids[-1] == 2 else ids
        decoded = bytes(i - 8 for i in raw_ids if i >= 8).decode("utf-8", errors="replace")
        check(decoded == generated, "儲存的 token 與回答文字不符")
        checked_rows.append({
            "family": row["family"], "mode": row["mode"], "generated": generated,
            "citation_valid": valid, "supported_by_cited_source": supported,
            "fact_answer_correct": correct, "address_correct": address_correct,
        })

    metrics = {}
    for mode in ("correct_with_distractor", "distractor_only"):
        selected_rows = [r for r in checked_rows if r["mode"] == mode]
        original_rows = [r for r in rows if r["mode"] == mode]
        check(len(selected_rows) == 12, "待核對條件的樣本數不符")
        check(len({r["family"] for r in selected_rows}) == 12, "待核對條件的獨立店名數不符")
        for row in original_rows:
            correct_docs = [d for d in row["documents"] if d["text"].split()[0] == row["fact"]["key"]]
            expected_correct_count = int(mode == "correct_with_distractor")
            check(len(correct_docs) == expected_correct_count, "條件內正確公告的數量不符")
            check(len(row["documents"]) == expected_correct_count + 1, "條件內公告總量不符")
        metrics[mode] = {}
        for field in ("citation_valid", "supported_by_cited_source", "fact_answer_correct"):
            numerator = sum(r[field] for r in selected_rows)
            denominator = len(selected_rows)
            metric = {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator}
            check(metric == record["results"]["metrics"][mode][field], "彙總與逐筆重算結果不符")
            metrics[mode][field] = metric
        metrics[mode]["address_correct"] = {
            "numerator": sum(r["address_correct"] for r in selected_rows), "denominator": len(selected_rows)
        }
    check(
        [metrics["correct_with_distractor"][f]["numerator"] for f in
         ("citation_valid", "supported_by_cited_source", "fact_answer_correct")]
        == [12, 2, 2], "正確公告加干擾公告的數字不符"
    )
    check(metrics["distractor_only"]["supported_by_cited_source"]["numerator"] == 5,
          "只有干擾公告的支持數不符")
    check(metrics["distractor_only"]["address_correct"]["numerator"] == 0,
          "只有干擾公告時答對所問地址的數量不符")
    example = next(r for r in rows if r["family"] == "K003" and r["mode"] == "correct_with_distractor")
    check(example["generation"]["samples"][0]["generated"] == "C9[D9]", "指定回答與紀錄不符")
    check(example["documents"] == [
        {"id": "D0", "text": "K013 address=C9"}, {"id": "D9", "text": "K003 address=D2"}
    ], "指定公告與紀錄不符")

    historical_command = ["git", "show", f"{record['revision']}:scripts/course_experiments/applications.py"]
    historical = subprocess.run(historical_command, capture_output=True, text=True)
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-4/verify.py",
        "result": "全部地址示例、逐筆評分、儲存 token、提示及條件彙總核對成功。",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in (RECORD, APPLICATIONS, SECTION_RECORD)},
        "original_example_stdout": stdout.getvalue(),
        "toy_results": toy_results,
        "negative_example": negative,
        "paraphrase_example": paraphrase,
        "verified_sample_rows": len(checked_rows),
        "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "metrics": metrics,
        "checked_rows": checked_rows,
        "specific_example": {k: example[k] for k in ("family", "mode", "fact", "documents")}
        | {"prompt": example["generation"]["messages"][0]["content"],
           "generated": example["generation"]["samples"][0]["generated"]},
        "experiment_environment": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu")}
        | {"training_steps": record["results"]["training"]["steps"], "model": record["results"]["model"],
           "split": record["results"]["split"], "test_facts_sha256": record["results"]["test_facts_sha256"]},
        "code_provenance": {
            "recorded_applications_sha256": record["code_sha256"]["scripts/course_experiments/applications.py"],
            "current_applications_sha256": digest(APPLICATIONS),
            "historical_lookup": {"command_argv": historical_command, "exit_code": historical.returncode,
                                  "stdout": historical.stdout, "stderr": historical.stderr},
            "scope": "當前 _grounding 與已存評分逐筆一致；沒有聲稱現行整檔即原始訓練版本，也沒有重訓或重跑 CUDA 推論。",
        },
        "limits": record["results"]["limits"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
