"""獨立重算凍結小節的算例、資料切分與保存的逐題生成；不訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import Context, new_lm, records_sha256
from scripts.course_experiments.tool_choice import build_records, parse_action, split_records, summarize
from tiny_perceptron.data import ByteTokenizer, render_chat


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rate(n, d):
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def independent_metrics(rows, actions):
    correct = sum(int(row["expected_action"] == action) for row, action in zip(rows, actions, strict=True))
    needed = [i for i, row in enumerate(rows) if row["expected_action"] == "TOOL"]
    other = [i for i, row in enumerate(rows) if row["expected_action"] != "TOOL"]
    off = [i for i, row in enumerate(rows) if row["calculator_available"] is False]
    return {
        "accuracy": rate(correct, len(rows)),
        "valid_action": rate(sum(int(a in ("DIRECT", "TOOL", "ASK")) for a in actions), len(rows)),
        "needed_tool_selected": rate(sum(int(actions[i] == "TOOL") for i in needed), len(needed)),
        "needed_tool_missed": rate(sum(int(actions[i] != "TOOL") for i in needed), len(needed)),
        "unnecessary_tool_selected": rate(sum(int(actions[i] == "TOOL") for i in other), len(other)),
        "unavailable_tool_selected": rate(sum(int(actions[i] == "TOOL") for i in off), len(off)),
        "confusion": {
            expected: dict(Counter(a or "INVALID" for row, a in zip(rows, actions, strict=True)
                                   if row["expected_action"] == expected))
            for expected in ("DIRECT", "TOOL", "ASK")
        },
        "by_intent": {
            intent: rate(sum(int(row["expected_action"] == a) for row, a in zip(rows, actions, strict=True)
                             if row["intent"] == intent), sum(int(row["intent"] == intent) for row in rows))
            for intent in sorted({row["intent"] for row in rows})
        },
        "by_availability": {
            str(on).lower(): rate(sum(int(row["expected_action"] == a) for row, a in zip(rows, actions, strict=True)
                                     if row["calculator_available"] == on),
                                 sum(int(row["calculator_available"] == on) for row in rows))
            for on in (True, False)
        },
    }


def main():
    torch.set_num_threads(1)
    source = (ROOT / "docs/course-review-20261019/freeze-01/sources/B.7.md").read_text()
    code = re.search(r"```python\n(.*?)```", source, re.S)[1]
    toy_outputs = {}
    for label, body in (("original", code), ("exercise", code.replace(
        'predicted = ["TOOL", "DIRECT", "TOOL", "DIRECT", "ASK", "DIRECT"]',
        'predicted = ["TOOL", "TOOL", "TOOL", "DIRECT", "ASK", "DIRECT"]'))):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(body, "frozen-example", "exec"), {})
        toy_outputs[label] = capture.getvalue()
    require(toy_outputs["original"] == "動作選對 3 / 6\n需要工具卻沒選 1 / 2\n不需要工具卻選了 1 / 4\n", "原始算例輸出不符")
    require(toy_outputs["exercise"] == "動作選對 4 / 6\n需要工具卻沒選 0 / 2\n不需要工具卻選了 1 / 4\n", "練習算例輸出不符")
    mismatch = None
    try:
        sum(a == b for a, b in zip([1, 2], [1], strict=True))
    except ValueError as error:
        mismatch = {"type": type(error).__name__, "message": str(error)}
    require(mismatch is not None and sum([True, False, True]) == 2, "Python 語意核對失敗")

    packaged = json.loads((ROOT / "docs/course-experiments/results/tool_choice.json").read_bytes())
    saved = packaged["results"]
    seed_value = saved["reproducibility"]["seed"]
    splits = split_records(build_records(), seed_value)
    families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
    require(all(not families[a] & families[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))), "家族跨組")
    split_info = {}
    for name, rows in splits.items():
        require(len(rows) == saved["data"][name]["records"], "資料筆數不符")
        require(sorted(families[name]) == saved["data"][name]["families"], "家族清單不符")
        require(records_sha256(rows) == saved["data"][name]["sha256"], "資料雜湊不符")
        split_info[name] = {"records": len(rows), "families": sorted(families[name]), "sha256": records_sha256(rows)}
    all_rows = build_records()
    require(len({row["family"] for row in all_rows}) == 55, "數字對家族總數不符")
    require({x for row in all_rows for x in row["operands"]} == set(range(10)), "數字範圍不符")
    for row in all_rows + build_records(paraphrase=True):
        a, b = row["operands"]
        require(0 <= a <= b <= 9 and row["family"] == f"pair:{a}:{b}", "家族定義不符")
        want = ("TOOL" if row["calculator_available"] else "ASK") if row["intent"] == "numerical_addition" else ("ASK" if row["intent"] == "missing_quantity" else "DIRECT")
        require(row["expected_action"] == want, "標註規則不符")
        require(row["messages"] == [
            {"role": "system", "content": "計算器可用。" if row["calculator_available"] else "計算器停用。"},
            {"role": "user", "content": row["messages"][1]["content"]},
            {"role": "assistant", "content": want}], "資料格式不符")

    paraphrases = [row for row in build_records(paraphrase=True) if row["family"] in families["test"]]
    require(records_sha256(paraphrases) == saved["data"]["paraphrase_diagnostic"]["sha256"], "新問法資料雜湊不符")
    tok = ByteTokenizer()
    recomputed = {}
    examples = []
    for name, expected_rows in (("validation", splits["validation"]), ("test", splits["test"]), ("paraphrase_diagnostic", paraphrases)):
        block = saved[name]
        rows = [sample["record"] for sample in block["samples"]]
        require(rows == expected_rows, "保存題目與重建資料不同")
        actions = []
        for sample in block["samples"]:
            row, generation = sample["record"], sample["generation"]
            require(generation["messages"] == row["messages"][:-1], "模型輸入混入答案")
            require(len(generation["samples"]) == 1 and generation["temperature"] == 0.0, "解碼設定不符")
            prompt_ids = [tok.bos_id]
            for message in generation["messages"]:
                prompt_ids += [{"system": tok.system_id, "user": tok.user_id}[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
            prompt_ids += [tok.assistant_id]
            require(prompt_ids == generation["input_ids"], "保存的輸入 token 不符")
            raw = generation["samples"][0]
            ids = raw["generated_ids"]
            ended = bool(ids and ids[-1] == tok.eos_id)
            tokens = ids[:-1] if ended else ids
            require(raw["eos"] == ended and raw["invalid_special_tokens"] == [i for i in tokens if i < 8], "EOS 或特殊 token 紀錄不符")
            require(tok.decode(tokens) == raw["generated"], "生成 token 與文字不符")
            action = raw["generated"] if ended and not raw["invalid_special_tokens"] and raw["generated"] in ("DIRECT", "TOOL", "ASK") else None
            require(action == sample["chosen_action"] == parse_action(raw), "選卡解析不符")
            actions.append(action)
            if row["family"] == "pair:1:2":
                examples.append({"split": name, "question": row["messages"][1]["content"], "system": row["messages"][0]["content"], "expected": row["expected_action"], "generated": raw["generated"], "ask_reason_metadata": row["ask_reason"]})
        metrics = independent_metrics(rows, actions)
        require(metrics == block["metrics"] == summarize(rows, actions), "分項重算與保存指標不同")
        baselines = {label: independent_metrics(rows, [action] * len(rows)) for label, action in (("always_direct", "DIRECT"), ("always_tool", "TOOL"))}
        require(baselines == block["baselines"], "固定卡對照重算不符")
        recomputed[name] = {"metrics": metrics, "baselines": {k: v["accuracy"] for k, v in baselines.items()}, "action_counts": dict(Counter(actions)), "raw_samples_checked": len(rows)}

    train_questions = {row["messages"][1]["content"] for row in splits["train"]}
    require(not train_questions & {row["messages"][1]["content"] for row in paraphrases}, "新問法出現在訓練中")
    require(not train_questions & {row["messages"][1]["content"] for row in splits["test"]}, "測試問題出現在訓練中")
    require({x for row in splits["test"] for x in row["operands"]} <= {x for row in splits["train"] for x in row["operands"]}, "未見組合與單獨數字的區分不符")
    model = new_lm(Context("cpu", Path(__file__).parent, Path(__file__).parent, ROOT / "assets", seed_value), width=64, layers=2, heads=2, max_length=160)
    require(model.description() == saved["model"], "隨機建立模型的設定不符")
    require(saved["training"]["records_sha256"] == records_sha256(splits["train"]) and saved["training"]["steps"] == 900, "訓練資料或步數不符")
    require(saved["reproducibility"]["script_sha256"] == sha("scripts/course_experiments/tool_choice.py"), "實驗程式版本不同")
    effective = sum(int((render_chat(row["messages"], tok)[1] != -100).sum()) for row in splits["train"])

    tools = json.loads((ROOT / "docs/course-experiments/results/tools.json").read_bytes())["results"]
    recomputed_tools = []
    for episode in tools["samples"]:
        executions = [event for event in episode["trace"] if event.get("executed")]
        answer_ok = episode["status"] == "done" and type(episode["answer"]) in (int, float) and episode["answer"] == episode["expected"]
        tool_ok = not executions if episode["operation"] == "copy" else bool(executions and executions[-1]["finite_result"] and any(event["request_matches_question"] for event in executions) and episode["answer"] == executions[-1]["tool_result"])
        recomputed_tools.append(answer_ok and tool_ok)
    json_accuracy = rate(sum(recomputed_tools), len(recomputed_tools))
    require(json_accuracy == tools["accuracy"] == rate(19, 20), "JSON 模型整題計分不符")
    require(tools["model"] != saved["model"], "兩個實驗的模型設定未區分")
    script_ast = ast.parse((ROOT / "scripts/course_experiments/tool_choice.py").read_text())
    run = next(node for node in script_ast.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    run_calls = [node.func.id for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    require(run_calls.count("fit_lm") == 1 and run_calls.count("evaluate") == 3, "單次配方的呼叫結構不符")
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-7/verify_evidence.py",
        "result": "所有獨立重算與一致性斷言通過；未訓練、未重新生成模型答案。",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads())},
        "inputs_sha256": {p: sha(p) for p in ["docs/course-review-20261019/freeze-01/sources/B.7.md", "docs/course-experiments/results/tool_choice.json", "docs/course-experiments/results/tools.json", "scripts/course_experiments/tool_choice.py", "scripts/course_experiments/common.py", "scripts/course_experiments/applications.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py"]},
        "toy_outputs": toy_outputs,
        "python_semantics": {"strict_length_mismatch": mismatch, "boolean_sum": sum([True, False, True])},
        "split_reconstruction": split_info,
        "total_families": 55,
        "rows_per_original_family": 16,
        "rows_per_paraphrase_family": 8,
        "test_and_paraphrase_records": [len(splits["test"]), len(paraphrases)],
        "metrics_recomputed_from_raw_tokens": recomputed,
        "examples": examples,
        "training_record": saved["training"],
        "historical_environment": saved["reproducibility"],
        "frozen_evaluation_record": saved["frozen_evaluation"],
        "new_random_model": model.description(),
        "training_unique_supervised_tokens": effective,
        "fixed_schedule_call_structure": {"fit_lm": run_calls.count("fit_lm"), "evaluate": run_calls.count("evaluate")},
        "json_model_comparison": {"accuracy_recomputed": json_accuracy, "model": tools["model"], "tool_choice_model": saved["model"], "tool_choice_has_tool_trace": any("trace" in sample for sample in saved["test"]["samples"])},
        "limitations": ["保存的生成 token 與指標已逐題核對；沒有原始權重的推論重播。", "固定排程與不依分數更新，只能由這個程式的呼叫結構及當次紀錄核對，不能獨立稽核未保存的人工探索歷史。", "只讀取實驗原始資料與明確指定的凍結原文，沒有讀取其他審閱者的判斷。"]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
