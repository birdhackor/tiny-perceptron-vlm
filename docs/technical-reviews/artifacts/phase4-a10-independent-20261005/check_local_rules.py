"""Independent, bounded review of the three authored messages, not model evaluation."""

from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
section_bytes = (HERE / "section.md").read_bytes()
section = section_bytes.decode("utf-8")
rows = re.findall(r"^\| (10月\d+日) \| (.+) \|$", section, re.M)
assert len(rows) == 3
messages = []
for timestamp, text in rows:
    day = int(re.search(r"\d+月(\d+)日", timestamp).group(1))
    messages.append((date(2026, 10, day), text))


def state_at(cutoff):
    state = {}
    for timestamp, text in messages:
        if timestamp > cutoff:
            continue
        activity = re.search(r"(10月\d+日)攝影社活动|(?P<traditional>10月\d+日)攝影社活動", text)
        if activity:
            state["activity_date"] = activity.group(1) or activity.group("traditional")
        time = re.search(r"\d{2}:\d{2}", text)
        if time:
            state["time"] = time.group(0)
        place = re.search(r"活動室(M\d+)", text)
        if place:
            state["place"] = place.group(1)
        if "一小段文字" in text:
            state["format"] = "paragraph"
        if "兩個條列" in text:
            state["format"] = "two_single_sentence_bullets"
        assert "晚餐" not in text
    return state


def inspect_current_answer(answer):
    # Local authored rubric: exact facts, two bullet items, one sentence each.
    # This is deliberately not the LongMemEval LLM evaluator.
    bullets = re.findall(r"^- (.+)$", answer, re.M)
    return {
        "activity_date": "10月10日" in answer,
        "time": "16:00" in answer and "15:00" not in answer,
        "place": "活動室M17" in answer,
        "format": len(bullets) == 2 and all(item.count("。") == 1 for item in bullets),
    }


states = {day: state_at(date(2026, 10, day)) for day in (1, 2, 3, 4)}
assert states[1] == {"activity_date": "10月10日", "time": "15:00", "place": "M17", "format": "paragraph"}
assert states[2] == states[1]
assert states[3] == {"activity_date": "10月10日", "time": "16:00", "place": "M17", "format": "paragraph"}
assert states[4] == {"activity_date": "10月10日", "time": "16:00", "place": "M17", "format": "two_single_sentence_bullets"}
answer = "- 10月10日16:00集合。\n- 地點是活動室M17。"
assert answer in section
answer_cases = {
    "authored_answer": answer,
    "stale_time": answer.replace("16:00", "15:00"),
    "missing_place": "- 10月10日16:00集合。\n- 地點尚未提供。",
    "paragraph": "10月10日16:00在活動室M17集合。",
}
checks = {name: inspect_current_answer(value) for name, value in answer_cases.items()}
assert all(checks["authored_answer"].values())
assert checks["stale_time"]["time"] is False
assert checks["missing_place"]["place"] is False
assert checks["paragraph"]["format"] is False
assert states[2]["time"] == "15:00" and states[4]["time"] == "16:00"
assert all("dinner" not in state for state in states.values())

# Same key-message content remains identifiable under irrelevant interleaving.
# This verifies the comparison contract, not a model's handling of long histories.
short = [(stamp.isoformat(), text) for stamp, text in messages]
long = [short[0], ("2026-10-02", "今天討論相機電池，沒有修改攝影社活動。"), short[1], short[2]]
assert [record for record in long if record in short] == short

result = {
    "scope": "Deterministic reconstruction of authored example and local answer rubric; no model inference, training, scoring or original dataset used.",
    "environment": {"python": sys.version, "device": "CPU", "network": "none used"},
    "source_sha256": hashlib.sha256(section_bytes).hexdigest(),
    "input_rows": [{"message_date": stamp.isoformat(), "text": text} for stamp, text in messages],
    "states": {str(day): value for day, value in states.items()},
    "current_answer_checks": checks,
    "historical_cutoff": "2026-10-02 -> 15:00; event remains 10月10日",
    "missing_information": "No dinner place occurs in any of the three original messages; an unknown/clarification answer is needed by the local rubric.",
    "local_counts": {"original_messages": len(rows), "question_categories": 3, "current_answer_bullets": 2},
    "paired_content_check": {"short_key_messages": len(short), "interleaved_key_messages": 3, "short_character_count": sum(len(x[1]) for x in short), "interleaved_character_count": sum(len(x[1]) for x in long), "key_positions_zero_based": [0, 2, 3], "identical_ordered_key_messages": True},
    "denominator_limit": "No model accuracy denominator or token-length measurement is claimed in A.10. These counts are authored examples, not performance data.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
