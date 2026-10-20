"""核對凍結範例、練習、字串規則邊界與有效 EOS 目標。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, render_chat


def main():
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/8.16.md"
    raw = frozen.read_bytes()
    namespace = {}
    captured = io.StringIO()
    blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    with contextlib.redirect_stdout(captured):
        for block in blocks:
            exec(compile(block, str(frozen), "exec"), namespace)
    check = namespace["check"]
    expected = [
        [True, True, True, True],
        [False, True, True, True],
        [True, False, True, True],
        [True, True, False, True],
        [True, True, False, True],
        [True, True, True, False],
    ]
    observed = [list(check(reply, stop).values()) for reply, stop in namespace["cases"]]
    assert observed == expected
    more_expected = [
        [True, False, False, True],
        [True, False, False, True],
        [True, False, True, True],
        [False, False, True, True],
        [True, True, False, True],
        [False, True, True, True],
        [False, True, False, True],
        [False, True, True, True],
    ]
    more = [
        {"name": name, "reply": reply, "observed": list(check(reply, "EOS").values())}
        for name, reply in namespace["more"]
    ]
    assert [item["observed"] for item in more] == more_expected
    original_reply = namespace["cases"][0][0]
    namespace["target"] = ["Closed today", "Open tomorrow"]
    namespace["excluded"] = ["今日休館", "明日開放"]
    english_reply = '{"text":["Closed today","Open tomorrow"]}'
    exercise = {
        "old_answer": check(original_reply, "EOS"),
        "new_answer": check(english_reply, "EOS"),
    }
    assert list(exercise["old_answer"].values()) == [False, False, True, True]
    assert all(exercise["new_answer"].values())
    namespace["target"] = ["今日休館", "明日開放"]
    namespace["excluded"] = ["Closed today", "Open tomorrow"]
    reversed_reply = '{"text":["明日開放","今日休館"]}'
    assert list(check(reversed_reply, "EOS").values()) == [False, True, True, True]
    format_rejections = [
        '```json\n' + original_reply + '\n```',
        '{"text":["今日休館",4]}',
        '{"text":"今日休館\\n明日開放"}',
        '{"other":["今日休館","明日開放"]}',
    ]
    assert all(not check(reply, "EOS")["格式"] for reply in format_rejections)
    tok = ByteTokenizer()
    _, y = render_chat([
        {"role": "user", "content": "只抄兩行中文，依原順序，回只有text欄位的JSON。"},
        {"role": "assistant", "content": original_reply},
    ], tok)
    answer = y[y != -100].tolist()
    assert answer[-1] == tok.eos_id
    assert tok.decode(answer) == original_reply
    assert answer[:-1] == tok.encode(original_reply)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-16/verify.py",
        "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform()},
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_stdout": captured.getvalue(),
        "table_expected": expected,
        "table_observed": observed,
        "joint_pass_count": sum(all(row) for row in observed),
        "question_denominator": len(observed),
        "per_criterion_pass_count": [sum(row[i] for row in observed) for i in range(4)],
        "more": more,
        "english_exercise": exercise,
        "reverse_order": check(reversed_reply, "EOS"),
        "format_rejections": [{"reply": reply, "result": check(reply, "EOS")} for reply in format_rejections],
        "encoded_decodes_to_target": json.loads(namespace["encoded"])["text"] == namespace["target"],
        "escaped_character": {"escape": "\\u4eca", "decoded": json.loads('"\\u4eca"')},
        "training_target": {
            "valid_targets": len(answer), "eos_id": tok.eos_id,
            "last_target": answer[-1], "decoded": tok.decode(answer),
            "last_target_after_removing_eos": answer[-2],
        },
        "result": "所有算例、練習與邊界核對均符合預期；停止原因為手寫條件，沒有模型品質實測。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
