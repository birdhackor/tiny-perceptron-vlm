"""核對日期資料、保存的逐題生成與評估呼叫；不訓練模型。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import behavior, common, text
from tiny_perceptron.data import ByteTokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (DEST / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks = []

    def handle_data(self, value):
        self.chunks.append(value)


def fetch_excerpt(url, locator, quotation, version):
    command = ["curl", "-fsSL", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    parser = PlainText()
    parser.feed(raw.decode())
    flattened = " ".join(" ".join(parser.chunks).split())
    assert quotation in flattened, "原文摘錄必須存在於實際取得的文件"
    return {
        "url": url,
        "command_argv": command,
        "accessed_on": "2026-10-20",
        "version": version,
        "retrieved_bytes": len(raw),
        "retrieved_sha256": sha(raw),
        "locator": locator,
        "excerpt": quotation,
    }


def main():
    helper_run = ROOT / sys.argv[1]
    names = ["stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json", "fence-1.py"]
    copied = {
        name: {"sha256": sha((helper_run / name).read_bytes()), "text": (helper_run / name).read_text()}
        for name in names
    }
    execution = json.loads(copied["execution.json"]["text"])
    assert execution["exit_code"] == 0, "小節程式應正常結束"
    expected_stdout = (
        "{'task': '確認日期', 'date': None} → 你希望哪一天？\n"
        "{'task': '確認日期', 'date': '2026-10-03'} → 已確認日期：2026-10-03\n"
    )
    assert copied["stdout.txt"]["text"] == expected_stdout, "小節的兩個日期分支應與目標相同"
    save("section-execution.json", {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.6" --execute',
        "original_run": str(helper_run.relative_to(ROOT)),
        "copied_files": copied,
    })

    report_path = ROOT / "docs/course-experiments/results/style.json"
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    results = report["results"]
    behavior_path = ROOT / "scripts/course_experiments/behavior.py"
    behavior_source = behavior_path.read_text()
    tree = ast.parse(behavior_source)
    run_style = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_style")
    date_loop = next(node for node in run_style.body if isinstance(node, ast.For) and ast.unparse(node.target) == "day")
    dates = []
    namespace = {"dates": dates, "_conversation": behavior._conversation}
    exec(compile(ast.Module(body=[date_loop], type_ignores=[]), str(behavior_path), "exec"), namespace)
    parts = common.split_records(dates, seed=report["seed"])
    counts = {
        split: {"dates": len({row["family"] for row in rows}), "records": len(rows),
                "families": sorted({row["family"] for row in rows})}
        for split, rows in parts.items()
    }
    assert [(counts[s]["dates"], counts[s]["records"]) for s in parts] == [(19, 38), (2, 4), (3, 6)], "日期分組數應與正文一致"
    families = [{row["family"] for row in rows} for rows in parts.values()]
    assert all(not (a & b) for i, a in enumerate(families) for b in families[i + 1:]), "日期家族不能跨資料切分"
    for rows in parts.values():
        for row in rows:
            prompt = row["messages"][0]["content"]
            if "date=?" in prompt:
                assert "2026-10" not in prompt and re.search(r";id=\d+;", prompt), "缺日期輸入保留題號而不寫入完整日期"

    arithmetic = common.split_records(text.arithmetic_records(), seed=report["seed"])
    conditional = {
        split: [behavior._style_record(row, style, True) for row in rows for style in ("concise", "vivid", "json")] + parts[split]
        for split, rows in arithmetic.items()
    }
    dataset_fingerprints = {}
    for split, rows in conditional.items():
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        observed = sha(encoded)
        expected = results["data"][split]["sha256"]
        assert observed == expected, "重建的完整混合資料必須符合原實驗指紋"
        dataset_fingerprints[split] = {"expected": expected, "observed": observed, "records": len(rows)}
    training_hash = common.records_sha256(conditional["train"])
    assert training_hash == results["training"]["records_sha256"], "更新參數所用資料必須與訓練集指紋相符"

    tok = ByteTokenizer()
    samples = {}
    scores = {}
    for stage in ("before", "after"):
        selected = [row for row in results[stage]["test"]["samples"] if row.get("kind") == "clarification"]
        assert len(selected) == len(parts["test"]), "逐題紀錄應覆蓋所有日期測試題"
        for sample, row in zip(selected, parts["test"], strict=True):
            assert sample["messages"] == row["messages"][:-1], "保存的輸入必須等於重建的測試輸入"
            assert sample["expected"] == row["messages"][-1]["content"], "保存的參考答案必須與標註規則相同"
            ids = sample["generated_ids"]
            raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            assert tok.decode(raw_ids) == sample["generated"], "可見文字應能從原始 token 重建"
            assert sample["exact"] == (raw_ids == tok.encode(sample["expected"])), "正確率要以原始 token 比對"
        samples[stage] = selected
        scores[stage] = {
            label: {"records": len(chosen), "correct": sum(sample["exact"] for sample in chosen)}
            for label, chosen in (
                ("missing", [s for s in selected if "date=?" in s["messages"][0]["content"]]),
                ("provided", [s for s in selected if "date=?" not in s["messages"][0]["content"]]),
            )
        }
    assert scores["after"] == {"missing": {"records": 3, "correct": 3}, "provided": {"records": 3, "correct": 0}}, "分項成績應與保存紀錄一致"
    assert samples["after"][0]["generated"] == "已確認2026-10-10。", "第一筆日期生成應與正文例子一致"

    timeline = []
    for node in run_style.body:
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            if target in {"before", "training", "after"}:
                timeline.append(ast.get_source_segment(behavior_source, node))
    style_evals = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_style_evaluations")
    current_hashes = {
        path: {"current_sha256": sha((ROOT / path).read_bytes()), "recorded_sha256": report["code_sha256"][path]}
        for path in ("scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "tiny_perceptron/data.py")
    }
    historic = subprocess.run(
        ["git", "show", report["revision"] + ":scripts/course_experiments/behavior.py"],
        capture_output=True, text=True, check=False,
    )
    truth_table = []
    for date, clock in (("2026-10-03", "10:00"), (None, "10:00"), ("2026-10-03", None), (None, None)):
        missing = [name for name, value in (("日期", date), ("時間", clock)) if value is None]
        truth_table.append({"date": date, "time": clock, "missing": missing, "action": "追問" if missing else "確認"})
    empty_value = ""
    save("audit-results.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-6/audit_facts.py " + sys.argv[1],
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "section_stdout_matches": True,
        "none_identity": {"None_is_None": None is None, "empty_string_is_None": empty_value is None},
        "source_result": {"path": str(report_path.relative_to(ROOT)), "sha256": sha(report_raw)},
        "original_experiment_environment": {key: report[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
        "original_training": results["training"],
        "date_counts": counts,
        "date_families_disjoint": True,
        "missing_input_example": next(r for r in parts["test"] if "date=?" in r["messages"][0]["content"]),
        "dataset_fingerprints": dataset_fingerprints,
        "training_records_sha256": training_hash,
        "samples": samples,
        "scores": scores,
        "after_total": {"correct": 3, "records": 6, "fraction": 0.5},
        "date_time_truth_table": truth_table,
        "current_code_and_recorded_hashes": current_hashes,
        "run_style_timeline": timeline,
        "style_evaluations_source": ast.get_source_segment(behavior_source, style_evals),
        "historical_code_lookup": {"command": "git show " + report["revision"] + ":scripts/course_experiments/behavior.py", "exit_code": historic.returncode, "stderr": historic.stderr},
        "scope": "重建資料並重算保存的 token 與計分；沒有重訓，也沒有重新以 checkpoint 生成。當前 behavior.py 與實驗記錄的 hash 不同，但重建的三份完整資料與原實驗指紋全部相等。原 revision 找不到當時的 behavior.py，因此評估時序同時用保存的 before/after 日期逐題紀錄核對。",
    })
    excerpts = [
        fetch_excerpt("https://aclanthology.org/2021.naacl-main.340/", "Abstract, opening definition", "questions that identify useful missing information in a given context", "NAACL-HLT 2021, DOI 10.18653/v1/2021.naacl-main.340; publisher abstract"),
        fetch_excerpt("https://scikit-learn.org/1.9/modules/cross_validation.html", "Cross-validation iterators for grouped data / Group k-fold", "ensures that the same group is not represented in both testing and training sets.", "scikit-learn 1.9 documentation"),
        fetch_excerpt("https://docs.python.org/3.13/library/constants.html#None", "Built-in Constants / None", "An object frequently used to represent the absence of a value", "Python 3.13 documentation"),
        fetch_excerpt("https://docs.python.org/3.13/reference/expressions.html#is-not", "Identity comparisons", "The operators is and is not test for an object’s identity", "Python 3.13 language reference"),
    ]
    save("source-excerpts.json", {"sources": excerpts, "note": "curl 取得的全文僅在記憶體中處理；只保存與主張有關的小段摘錄及全文指紋。"})
    print(json.dumps({"result": "日期資料指紋、原始 token、分項成績與小節輸出核對完成", "scores": scores, "test_evaluated_before_and_after": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
