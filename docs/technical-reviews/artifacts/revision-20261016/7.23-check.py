"""只核對 7.23 已明標假設的表格；不呼叫商用 API，也不推估模型效果。"""
import json
import platform

rows = [
    {"question": "2+3", "request": "high", "max_rounds": 5, "actual_rounds": 1, "final_answer_characters": 1},
    {"question": "多次修改後的訂單數量", "request": "high", "max_rounds": 5, "actual_rounds": 5, "final_answer_characters": 1},
]
checks = {
    "2_plus_3": 2 + 3,
    "len_decimal_5": len(str(2 + 3)),
    "both_actual_rounds_within_cap": all(0 < row["actual_rounds"] <= row["max_rounds"] for row in rows),
    "same_request_and_cap": rows[0]["request"] == rows[1]["request"] and rows[0]["max_rounds"] == rows[1]["max_rounds"],
    "different_actual_rounds": rows[0]["actual_rounds"] != rows[1]["actual_rounds"],
    "same_final_answer_length": rows[0]["final_answer_characters"] == rows[1]["final_answer_characters"],
}
assert checks == {"2_plus_3": 5, "len_decimal_5": 1, "both_actual_rounds_within_cap": True, "same_request_and_cap": True, "different_actual_rounds": True, "same_final_answer_length": True}
print(json.dumps({
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/7.23-check.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; Python scalar arithmetic only; no model call"},
    "rows": rows,
    "results": checks,
    "scope": "表格是教材人工假設；檢查其上限/耗用與答案字數的內部一致性，不驗證任何商用模型的用量、延遲或品質。訂單題沒有提供事件序列或答案內容，因此不虛構或驗算其訂單值。",
}, ensure_ascii=False, indent=2))
