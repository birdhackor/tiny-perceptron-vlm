"""核對下一字配對、切片與嚴格 zip 的確定性行為。"""

import json
import platform
import sys


def inspect_text(text, expected_x, expected_y):
    original = list(text)
    before = original.copy()
    x = original[:-1]
    y = original[1:]
    pairs = list(zip(x, y, strict=True))
    expected_pairs = list(zip(expected_x, expected_y, strict=True))
    assert x == expected_x
    assert y == expected_y
    assert pairs == expected_pairs
    assert original == before
    assert x is not original and y is not original and x is not y
    assert original[0] == text[0]
    assert original[-1] == text[-1]
    assert len(x) == len(y) == len(text) - 1
    wrong_y = original[:-1]
    wrong_pairs = list(zip(x, wrong_y, strict=True))
    assert len(x) == len(wrong_y) == len(text) - 1
    assert all(current == answer for current, answer in wrong_pairs)
    assert wrong_pairs != pairs
    x[0] = "另"
    assert original == before
    return {
        "text": text,
        "unicode_codepoints": [f"U+{ord(char):04X}" for char in text],
        "s": original,
        "x": expected_x,
        "y": expected_y,
        "pairs": pairs,
        "question_count": len(pairs),
        "first_index": original[0],
        "last_index": original[-1],
        "source_unchanged_after_slicing_and_mutating_x": original == before,
        "separate_list_objects": True,
        "incorrect_y": wrong_y,
        "incorrect_pairs": wrong_pairs,
        "incorrect_length_check_passes": len(expected_x) == len(wrong_y) == len(text) - 1,
    }


cases = [
    inspect_text("貓看狗。", ["貓", "看", "狗"], ["看", "狗", "。"]),
    inspect_text("鳥飛。", ["鳥", "飛"], ["飛", "。"]),
]
unequal = []
for left, right in [(["貓"], ["看", "狗"]), (["貓", "看"], ["狗"])]:
    iterator = zip(left, right, strict=True)
    created_without_error = True
    prefix = []
    try:
        while True:
            prefix.append(next(iterator))
    except ValueError as error:
        unequal.append({
            "left": left,
            "right": right,
            "created_without_error": created_without_error,
            "yielded_prefix": prefix,
            "error_type": type(error).__name__,
            "error": str(error),
        })
    except StopIteration:
        raise AssertionError("不同長度的 strict zip 未回報錯誤") from None

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-3/verify_pairs.py",
    "environment": {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "device": "CPU；只使用 Python 標準函式庫",
    },
    "cases": cases,
    "strict_zip_unequal_cases": unequal,
    "result": "全部精確比較與斷言通過",
}, ensure_ascii=False, indent=2))
