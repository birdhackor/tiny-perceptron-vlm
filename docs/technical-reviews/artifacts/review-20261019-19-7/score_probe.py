"""用人工輸出驗證評分口徑；這不是模型能力實驗。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.selftrained.evaluate import score_reply
from tiny_perceptron.selftrained.tools import execute_tool_call, run_tool_loop, serialize_tool_result
from tiny_perceptron.selftrained.tokenizer import SPECIAL_NAMES


def main():
    original = {"tool": "calculator", "operation": "multiply", "a": 26, "b": 16}
    raw = json.dumps(original, separators=(",", ":"))
    prompt = [{"role": "user", "content": "用計算器算出26×16。"}]
    record = {
        "task": "tool_call", "messages": prompt + [{"role": "assistant", "content": raw}],
        "supervision": {"expected_call": original},
    }
    rows = []
    for name, call, answer in [
        ("正確呼叫與最終數字", raw, "結果是416。"),
        ("合法呼叫但參數不符", raw.replace("26", "25"), "結果是400。"),
        ("執行正確但最終數字錯誤", raw, "結果是400。"),
        ("沒有呼叫工具", "結果是416。", None),
    ]:
        outputs = iter([call, answer])
        trace = run_tool_loop(lambda _: next(outputs), prompt)
        score = score_reply(record, trace["final_output"], trace)
        rows.append({"case": name, "trace": trace, "score": score})
    assert [r["score"]["tool_roundtrip"] for r in rows] == [True, False, False, False]
    assert rows[1]["score"]["tool_final"] and not rows[1]["score"]["tool_parameters"]
    changed = {"tool": "calculator", "ok": True, "result": 417}
    replay = prompt + [{"role": "assistant", "content": raw}, {"role": "tool", "content": serialize_tool_result(changed)}]
    assert json.loads(replay[-1]["content"])["result"] == 417
    assert execute_tool_call(raw)["result"] == 416
    saved = json.loads((ROOT / "docs/selftrained/results/public-cpu-raw/tool_call-1-stdout.txt").read_bytes())
    eos_id = SPECIAL_NAMES.index("eos")
    assert all(g["generated_ids"][-1] == eos_id for g in saved["generations"])
    value = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-7/score_probe.py",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
            "scripts/selftrained/evaluate.py", "tiny_perceptron/selftrained/tools.py",
            "tiny_perceptron/selftrained/tokenizer.py", "tiny_perceptron/selftrained/model.py",
        ]},
        "cases": rows, "test_only_replacement_history": replay,
        "genuine_executor_result": 416, "replacement_result": 417,
        "saved_cpu_generation_eos_ids": [g["generated_ids"][-1] for g in saved["generations"]],
        "scope": "人工 callback 與現行 score_reply；替換值不是實際乘法結果，也不回填正常往返成績",
        "result": "所有斷言通過",
    }
    with Path(__file__).with_name("scoring.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": value["result"], "cases": len(rows), "eos_id": eos_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
