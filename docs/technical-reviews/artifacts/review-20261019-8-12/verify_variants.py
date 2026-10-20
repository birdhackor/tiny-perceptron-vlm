"""核對字串算例、練習變體與計數限制。"""

import json
import platform
import sys


def inspect(text):
    parts = text.split("。")
    return {
        "text": text,
        "length": len(text),
        "parts": parts,
        "kept_parts": parts[:-1],
        "distinct_parts": sorted(set(parts[:-1])),
        "distinct_count": len(set(parts[:-1])),
    }


short = "答案是4。"
repeated = short * 10
twice = short * 2
useful = "答案是4；兩組各兩個物件合起來共有四個。"
cases = {
    "short": inspect(short),
    "repeated": inspect(repeated),
    "twice": inspect(twice),
    "useful": inspect(useful),
    "without_final_delimiter": inspect(useful[:-1]),
    "same_meaning_different_words": inspect("答案是4。結果為四。"),
}
choice_probe = ["z", "aa"]
assert [len(short), len(repeated)] == [5, 50]
assert max([short, repeated], key=len) == repeated
assert cases["repeated"]["distinct_count"] == 1
assert [len(short), len(twice)] == [5, 10]
assert cases["twice"]["distinct_count"] == 1
assert cases["useful"]["distinct_count"] == 1
assert cases["useful"]["parts"][-1] == ""
assert cases["without_final_delimiter"]["distinct_count"] == 0
assert cases["same_meaning_different_words"]["distinct_count"] == 2
assert max(choice_probe) == "z"
assert max(choice_probe, key=len) == "aa"
assert 2 * 2 == 4

print(
    json.dumps(
        {
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "device": "CPU；純 Python 字串與整數操作",
            },
            "cases": cases,
            "lengths": {
                "repeated": [len(short), len(repeated)],
                "twice": [len(short), len(twice)],
            },
            "length_judge_selects_repeated": max([short, repeated], key=len) == repeated,
            "ordering_probe": {
                "candidates": choice_probe,
                "default_max": max(choice_probe),
                "length_max": max(choice_probe, key=len),
            },
            "group_arithmetic": {"groups": 2, "items_per_group": 2, "total": 2 * 2},
            "result": "全部斷言通過；計數依字串與分隔符號，未執行模型或 tokenizer。",
        },
        ensure_ascii=False,
        indent=2,
    )
)
