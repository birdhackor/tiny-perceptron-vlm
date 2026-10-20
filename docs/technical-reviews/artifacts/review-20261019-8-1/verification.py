"""核對手寫評分範例、資料模板、已保存的生成結果與權威來源。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ARTIFACT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pieces = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.pieces.append(data)


def read_sources():
    sources = [
        (
            "rubric",
            "https://www.cmu.edu/teaching/designteach/teach/rubrics.html",
            "What are Rubrics?；Advantages of Using Rubrics",
            "A rubric is a scoring tool that explicitly represents the performance expectations for an assignment or piece of work.",
            ["A rubric divides the assigned work", "By recording the component scores"],
        ),
        (
            "generalization",
            "https://developers.google.com/machine-learning/crash-course/overfitting/overfitting",
            "Overfitting 開頭；Fitting, overfitting, and underfitting",
            "a model that generalizes well makes good predictions on new data.",
            ["memorizes", "A truly excellent model successfully categorizes new examples"],
        ),
        (
            "fine_tuning",
            "https://developers.google.com/machine-learning/crash-course/llm/tuning",
            "Fine-tuning",
            "Fine-tuning trains on examples specific to the task your application will perform.",
            ["updating the weight and bias", "additional training helps the model make better predictions on a specific task"],
        ),
    ]
    receipts = []
    for name, url, locator, excerpt, checks in sources:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
        fetched = subprocess.run(command, capture_output=True, check=True)
        parser = TextParser()
        parser.feed(fetched.stdout.decode("utf-8"))
        text = " ".join(" ".join(parser.pieces).split())
        if excerpt not in text or not all(check in text for check in checks):
            raise ValueError("來源文字與待核對定位不符：" + name)
        updated = re.search(r"Last updated ([0-9-]+) UTC", text)
        receipts.append({
            "name": name,
            "url": url,
            "command_argv": command,
            "exit_code": fetched.returncode,
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "response_sha256": hashlib.sha256(fetched.stdout).hexdigest(),
            "version": "Last updated " + updated[1] + " UTC" if updated else "未標示修訂號的現行 HTML",
            "locator": locator,
            "excerpt": excerpt,
            "checked_original": True,
            "inspection_note": {
                "rubric": "已讀評分規則定義、各組件與表現層級、分項計分的用途；本節採二元通過層級，並未聲稱是標準心理量表。",
                "generalization": "已讀記憶訓練集而新資料失敗的定義，以及新例子表現的討論；支持不能由重複單一句子推定新題泛化。",
                "fine_tuning": "已讀使用特定任務範例再訓練的段落；只支持示範教學流程，不支持示範是所有學習方法的必要條件，也不保證泛化。",
            }[name],
        })
    save("primary-source-receipts.json", receipts)


def run_section():
    existing = ARTIFACT / "section-execution.json"
    if existing.exists():
        receipt = json.loads(existing.read_text())
        summary = json.loads(receipt["invocation_stdout"])
        frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/8.1.md"
        if receipt["exit_code"] != 0 or summary["source_sha256"] != sha(frozen):
            raise ValueError("已有的本次執行收據不符凍結來源")
        return summary
    command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/08.md#8.1", "--execute"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    directory = Path(summary["artifact_directory"])
    copied = {}
    for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (directory / name).read_bytes()
        copied[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "content": raw.decode("utf-8")}
    save("section-execution.json", {
        "command_argv": command,
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.1" --execute',
        "exit_code": result.returncode,
        "invocation_stdout": result.stdout,
        "invocation_stderr": result.stderr,
        "copied_from": str(directory.relative_to(ROOT)),
        "copied_files": copied,
    })
    return summary


def main():
    import torch
    from scripts.course_experiments.behavior import _style_metrics, _style_record
    from tiny_perceptron.data import ByteTokenizer

    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/8.1.md"
    body = frozen.read_text()
    code = re.search(r"```python\n(.*?)```", body, flags=re.S)[1]
    tree = ast.parse(code)
    original_output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(original_output):
        exec(compile(tree, str(frozen), "exec"), namespace)
    expected = "4 True False\n4，就像兩雙筷子共有四根。 True True\n5，數字如星光流動。 False False\n"
    if original_output.getvalue() != expected:
        raise ValueError("原始範例輸出不符")
    original_rows = copy.deepcopy(namespace["examples"])
    namespace["examples"][1]["回答"] = "4，像一片深邃的海"
    stale_label = namespace["examples"][1]["貼切比喻"]
    namespace["examples"][1]["貼切比喻"] = False
    changed_output = io.StringIO()
    with contextlib.redirect_stdout(changed_output):
        exec(compile(ast.Module(body=[tree.body[1]], type_ignores=[]), "manual-label-loop", "exec"), namespace)
    if changed_output.getvalue().splitlines()[1] != "4，像一片深邃的海 True False":
        raise ValueError("修改後範例輸出不符")

    row = {"a": 2, "b": 2, "family": "2+2"}
    templates = {style: _style_record(row, style, True) for style in ("concise", "vivid", "json")}
    record_path = ROOT / "docs/course-experiments/results/style.json"
    saved = json.loads(record_path.read_text())
    test = saved["results"]["after"]["test"]
    arithmetic_samples = [copy.deepcopy(sample) for sample in test["samples"] if sample.get("style") in templates]
    arithmetic_rows = []
    for sample in arithmetic_samples:
        question = sample["messages"][0]["content"]
        match = re.fullmatch(r"style=(concise|vivid|json); (\d+)\+(\d+)=\?", question)
        if match is None:
            raise ValueError("生成紀錄的題目無法解析")
        style, a, b = match.groups()
        arithmetic_rows.append(_style_record({"a": int(a), "b": int(b), "family": sample["family"]}, style, True))
    rescored = _style_metrics({"samples": arithmetic_samples}, arithmetic_rows)
    if rescored["rubric"] != {style: test["rubric"][style] for style in templates}:
        raise ValueError("逐題重算與已保存的分項計數不同")
    tok = ByteTokenizer()
    wrong_text = "5，像把兩組積木合在一起再數。"
    synthetic = {"generated": wrong_text, "generated_ids": tok.encode(wrong_text) + [tok.eos_id], "exact": False}
    counterexample = _style_metrics({"samples": [synthetic]}, [templates["vivid"]])
    if not synthetic["style_correct"] or synthetic["content_correct"]:
        raise ValueError("固定文句與數值正確的分項沒有分開")

    section = run_section()
    read_sources()
    save("verification-output.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-1/verification.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [frozen, record_path, ROOT / "scripts/course_experiments/behavior.py", ROOT / "tiny_perceptron/tokenization.py", Path(__file__)]},
        "arithmetic": {"two_plus_two": 2 + 2, "two_pairs": len([["筷子", "筷子"], ["筷子", "筷子"]][0]) + len([["筷子", "筷子"], ["筷子", "筷子"]][1]), "five_equals_two_plus_two": 5 == 2 + 2},
        "original_rows": original_rows,
        "original_stdout": original_output.getvalue(),
        "code_ast_node_types": [type(node).__name__ for node in tree.body],
        "label_after_only_changing_reply": stale_label,
        "exercise_stdout_after_manual_label_change": changed_output.getvalue(),
        "templates": templates,
        "saved_experiment_environment": {key: saved[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status"]},
        "recomputed_test_rubric": rescored["rubric"],
        "recomputed_samples": [{key: sample[key] for key in ["messages", "generated", "style", "content_correct", "style_correct", "eos"]} for sample in arithmetic_samples],
        "synthetic_wrong_arithmetic": counterexample,
        "section_summary": section,
        "scope": "實跑手寫範例、模板產生器、評分函式，並重算既有生成紀錄；未重新訓練或重新生成模型回答。",
    })
    print(json.dumps({"status": "pass", "original_stdout": original_output.getvalue(), "exercise_stdout": changed_output.getvalue(), "recomputed_rubric": rescored["rubric"], "section_summary": section}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
