"""離線核對計算器、教材節錄與原始 CPU 執行紀錄。"""

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.selftrained import tools


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_bytes())


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def section_run():
    command = [
        ".venv/bin/python", "docs/review-tools/section_facts.py",
        "course/chapters/19.md#19.7", "--execute",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    run = Path(result["artifact_directory"])
    # 只讀取本次子行程剛產生的執行檔案。
    with (HERE / "section-stdout.txt").open("xb") as handle:
        handle.write((run / "stdout.txt").read_bytes())
    save("section-run.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 " + " ".join(command),
        "returncode": completed.returncode,
        "launcher_stdout": completed.stdout,
        "launcher_stderr": completed.stderr,
        "execution": json.loads((run / "execution.json").read_bytes()),
        "environment": json.loads((run / "environment.json").read_bytes()),
        "extraction": json.loads((run / "extraction.json").read_bytes()),
        "worker_stderr": (run / "stderr.txt").read_text(),
        "copied_stdout_sha256": sha(HERE / "section-stdout.txt"),
    })
    assert result["source_sha256"] == "2be1c26ab663f7988639f5e9652894948543f1c62c7f7071d0b9af7217d29786"
    assert result["execution_exit_code"] == 0
    return result


def calculator_audit():
    raw = '{"tool":"calculator","operation":"multiply","a":26,"b":16}'
    checks = []
    for operation in tools.OPERATIONS:
        for a in range(100):
            for b in range(100):
                text = tools.serialize_tool_call(operation, a, b)
                call = tools.parse_tool_call(text)
                value = tools.execute_tool_call(call)["result"]
                expected = {"add": a + b, "subtract": a - b, "multiply": a * b}[operation]
                assert value == expected and type(value) is int
                assert call.to_dict() == {"tool": "calculator", "operation": operation, "a": a, "b": b}
    invalid = {
        "非字串": None,
        "字元過長": " " * 257,
        "不完整 JSON": "{",
        "最外層陣列": "[]",
        "最外層整數": "26",
        "重複欄位": raw.replace('"a":26', '"a":25,"a":26'),
        "缺欄位": raw.replace(',"b":16', ""),
        "多欄位": raw.replace('"b":16', '"b":16,"c":0'),
        "工具名稱": raw.replace("calculator", "shell"),
        "不支援運算": raw.replace("multiply", "divide"),
        "超過上限": raw.replace('"a":26', '"a":100'),
        "負輸入": raw.replace('"a":26', '"a":-1'),
        "布林": raw.replace('"a":26', '"a":true'),
        "浮點數": raw.replace('"a":26', '"a":26.0'),
        "數字字串": raw.replace('"a":26', '"a":"26"'),
        "非數": raw.replace('"a":26', '"a":NaN'),
    }
    for name, text in invalid.items():
        try:
            tools.parse_tool_call(text)
        except tools.ToolCallError as error:
            checks.append({"input_kind": name, "input": text, "error": str(error)})
        else:
            raise AssertionError(name)
    padded = raw + " " * (256 - len(raw))
    assert len(padded) == 256 and tools.parse_tool_call(padded).a == 26
    assert tools.MAX_CALL_CHARACTERS == 256
    assert tools.CALL_KEYS == {"tool", "operation", "a", "b"}
    assert tools.CalculatorCall("add", 0, 0).to_dict() == {
        "tool": "calculator", "operation": "add", "a": 0, "b": 0,
    }
    results = {
        "valid": tools.execute_tool_call(raw),
        "copied_wrong_operand": tools.execute_tool_call(raw.replace("26", "25")),
        "accepted_character_boundary": len(padded),
        "exhaustive_operand_pairs_and_operations": 30000,
        "rejected": checks,
    }
    assert results["valid"] == {"tool": "calculator", "ok": True, "result": 416}
    assert results["copied_wrong_operand"]["result"] == 400
    return results


def code_audit():
    source = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.7.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.S)
    assert len(blocks) == 3
    nodes = {node.name: node for node in ast.parse((ROOT / "tiny_perceptron/selftrained/tools.py").read_text()).body
             if isinstance(node, ast.FunctionDef)}
    matching = {}
    for node in ast.parse(blocks[1]).body:
        if isinstance(node, ast.FunctionDef):
            matching[node.name] = ast.dump(node, include_attributes=False) == ast.dump(nodes[node.name], include_attributes=False)
    assert all(matching.values()) and len(matching) == 2
    ns = {}
    eager = None
    try:
        exec(compile("def f() -> MissingType:\n    return 1\n", "annotation-probe", "exec", dont_inherit=True), ns)
    except NameError as error:
        eager = str(error)
    assert eager is not None
    annotated = {}
    exec("def f(x: int):\n    return x\n", annotated)
    assert annotated["f"]("文字") == "文字"
    tree = ast.parse((ROOT / "tiny_perceptron/selftrained/tools.py").read_text())
    forbidden_calls = [node.func.id for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id in {"eval", "exec", "compile"}]
    assert forbidden_calls == []
    return {
        "pasted_functions_ast_equal": matching,
        "missing_return_annotation_in_current_python": eager,
        "annotations_do_not_enforce_input_type": True,
        "tools_module_expression_evaluation_calls": forbidden_calls,
        "pinned_python": (ROOT / ".python-version").read_text().strip(),
    }


def protocol_audit():
    messages = read_json("docs/selftrained/examples/v2/tool_call.messages.json")
    raw = '{"tool":"calculator","operation":"multiply","a":26,"b":16}'
    histories = []
    outputs = iter([raw, "結果是416。"])

    def generate(history):
        histories.append(history)
        return next(outputs)

    trace = tools.run_tool_loop(generate, messages)
    assert trace["executed"] and trace["tool_result"]["result"] == 416
    assert histories[1] == messages + [
        {"role": "assistant", "content": raw},
        {"role": "tool", "content": '{"tool":"calculator","ok":true,"result":416}'},
    ]
    assert trace["final_output"] == "結果是416。"
    disabled_outputs = iter([raw, "工具目前未開啟。"])
    disabled = tools.run_tool_loop(lambda _: next(disabled_outputs), messages, tools_enabled=False)
    assert not disabled["executed"] and disabled["tool_result"]["error"] == "unavailable"
    return {
        "generator_kind": "人工固定輸出 callback，未呼叫神經模型",
        "callback_count": len(histories), "history_preserved": histories[1] == messages + histories[1][-2:],
        "appended_messages": histories[1][-2:], "trace": trace, "disabled": disabled,
    }


def empirical_audit():
    raw_root = "docs/selftrained/results/public-cpu-raw/"
    stdout_path = ROOT / (raw_root + "tool_call-1-stdout.txt")
    stdout = json.loads(stdout_path.read_bytes())
    wrapper = read_json(raw_root + "tool_call-1-result.json")
    argv = read_json(raw_root + "tool_call-1-argv.json")
    index = read_json("docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-index.json")
    expected_hashes = {item["path"]: item["sha256"] for item in index["files"]}
    for filename in ("tool_call-1-stdout.txt", "tool_call-1-result.json", "tool_call-1-argv.json"):
        assert sha(ROOT / (raw_root + filename)) == expected_hashes[filename]
    assert sha(stdout_path) == wrapper["stdout_sha256"]
    assert stdout["generations"] == wrapper["actual_generations"]
    assert stdout["tool_trace"] == wrapper["actual_tool_trace"]
    assert stdout["answer"] == "結果是416。"
    gens = stdout["generations"]
    assert len(gens) == 2 and all(g["eos"] and g["generation_status"] == "eos" for g in gens)
    assert gens[0]["model_sha256"] == gens[1]["model_sha256"] == stdout["model"]["files"]["model.safetensors"]
    first = gens[0]["raw_output"]
    assert json.loads(first) == {"tool": "calculator", "operation": "multiply", "a": 26, "b": 16}
    assert tools.execute_tool_call(first) == stdout["tool_trace"]["tool_result"]
    assert gens[0]["prompt_messages"] == argv["public_messages_before_execution"]
    assert gens[0]["prompt_messages"] == read_json("docs/selftrained/examples/v2/tool_call.messages.json")
    assert gens[1]["prompt_messages"] == gens[0]["prompt_messages"] + [
        {"role": "assistant", "content": first}, stdout["tool_trace"]["tool_message"],
    ]
    assert gens[1]["raw_output"] == "結果是416。"
    metrics_path = "docs/selftrained/results/public-raw/moe/test/metrics.json"
    metrics = read_json(metrics_path)
    receipt = read_json("docs/selftrained/results/public-raw/moe/test/receipt.json")
    assert sha(ROOT / metrics_path) == next(item["sha256"] for item in receipt["files"] if item["path"] == "metrics.json")
    group = metrics["per_task_final_reply"]["tool_call"]
    assert group["count"] == 276
    assert group["tool_roundtrip"] == {"numerator": 0, "denominator": 276, "rate": 0.0}
    assert group["tool_parameters"]["numerator"] == 0
    assert group["tool_executed"]["numerator"] == 114
    assert metrics["safe_weights_sha256"] == gens[0]["model_sha256"]
    return {
        "scope": "重算保存的原始輸出與彙總；未重新載入權重生成，未取得測試逐題 outputs.jsonl",
        "cpu_command_argv": argv["argv"], "public_source": stdout["public_source"],
        "model_sha256": gens[0]["model_sha256"], "model_config": stdout["model"]["config"],
        "selected_step": stdout["model"]["selected_step"],
        "seed": stdout["model"]["origin"]["seed"],
        "raw_first_output": first, "actual_tool_trace": stdout["tool_trace"],
        "generations": [{k: g[k] for k in ("call_index", "model_sha256", "raw_output", "eos", "generation_status", "prompt_messages")} for g in gens],
        "cpu_record_matches_index_and_stdout": True,
        "test_metrics_match_execution_receipt": True,
        "test_split": metrics["split"], "test_count": metrics["count"],
        "tool_test_group": group,
        "test_complete": metrics["evaluation_complete"],
        "validation_match_record": {"saved_GPU_validation_answer": wrapper["saved_GPU_validation_answer"],
                                    "matches": wrapper["answer_matches_saved_GPU_validation"]},
    }


def main():
    section_run()
    inputs = [
        "docs/course-review-20261019/freeze-01/sources/19.7.md", "docs/course-review-20261019/freeze-04.json",
        "course/figures/p6-19-modal-tool.svg", "tiny_perceptron/selftrained/tools.py",
        "tiny_perceptron/selftrained/inference.py", "scripts/selftrained/chat.py", "scripts/selftrained/evaluate.py",
        ".python-version", "pyproject.toml", "docs/selftrained/examples/v2/tool_call.messages.json",
        "docs/selftrained/results/public-cpu-raw/tool_call-1-result.json",
        "docs/selftrained/results/public-cpu-raw/tool_call-1-stdout.txt",
        "docs/selftrained/results/public-cpu-raw/tool_call-1-argv.json",
        "docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-index.json",
        "docs/selftrained/results/public-raw/moe/test/metrics.json",
        "docs/selftrained/results/public-raw/moe/test/receipt.json",
        "docs/selftrained/results/public-raw/moe/test/evaluation-receipt.json",
        "docs/selftrained/v2-public-cpu-commands.md",
    ]
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-7/verify.py",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "bytecode_disabled": os.environ.get("PYTHONDONTWRITEBYTECODE", "")},
        "inputs": [{"path": p, "sha256": sha(ROOT / p)} for p in inputs],
        "calculator": calculator_audit(), "code": code_audit(), "protocol": protocol_audit(),
        "empirical": empirical_audit(), "result": "所有斷言通過",
    }
    save("verification.json", result)
    print(json.dumps({"result": result["result"], "operand_cases": 30000, "rejections": len(result["calculator"]["rejected"]), "tool_test": "0/276"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
