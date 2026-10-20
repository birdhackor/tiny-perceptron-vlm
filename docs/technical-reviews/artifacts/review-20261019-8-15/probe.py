"""核對告示示範的目標、格式與截斷；不訓練模型。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum


def main():
    tok = ByteTokenizer()
    notice = "今日休館\nClosed today\n明日開放\nOpen tomorrow"
    cases = [
        ("全部照抄，一行一項", notice, 53),
        ("只抄中文，一行一項", "今日休館\n明日開放", 26),
        ("只抄英文，一行一項", "Closed today\nOpen tomorrow", 27),
        ("只抄中文，一行一項", notice, 53),
    ]
    records = []
    for request, target, expected_count in cases:
        user = notice + "\n要求：" + request
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ]
        x, y = render_chat(messages, tok)
        actual = y[y != IGNORE].tolist()
        expected = tok.encode(target) + [tok.eos_id]
        first = (y != IGNORE).nonzero()[0].item()
        assert actual == expected
        assert len(actual) == expected_count
        assert x.dtype == torch.long and y.dtype == torch.long
        assert x.shape == y.shape and x.ndim == 1
        assert x[first].item() == tok.assistant_id
        assert y[first].item() == tok.encode(target)[0]
        assert y[:first].eq(IGNORE).all().item()
        assert tok.decode(actual) == target
        assert x[:first].tolist() == [tok.bos_id, tok.user_id] + tok.encode(user) + [tok.eos_id]
        logits = torch.zeros((1, len(y), tok.vocab_size))
        total, count = loss_sum(logits, y.unsqueeze(0))
        assert count.item() == expected_count
        records.append(
            {
                "request": request,
                "target": target,
                "target_lines": target.splitlines(),
                "target_utf8_bytes": len(target.encode("utf-8")),
                "effective_targets": len(actual),
                "last_target": actual[-1],
                "last_is_eos": actual[-1] == tok.eos_id,
                "decode_exact": tok.decode(actual) == target,
                "first_prediction_input": x[first].item(),
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "loss_denominator": int(count.item()),
                "uniform_logit_loss_sum": float(total.item()),
                "request_target_matches": target == notice if request.startswith("全部") else (
                    target == "今日休館\n明日開放" if request.startswith("只抄中文") else target == "Closed today\nOpen tomorrow"
                ),
            }
        )

    chinese = "今日休館\n明日開放"
    x, y = render_chat(
        [{"role": "user", "content": notice + "\n要求：只抄中文，一行一項"}, {"role": "assistant", "content": chinese}],
        tok,
    )
    _, truncated_y, _ = pad_batch([(x, y)], max_length=len(x) - 1)
    remaining = truncated_y[truncated_y != IGNORE].tolist()
    assert len(remaining) == 25 and tok.eos_id not in remaining
    assert tok.decode(remaining) == chinese

    json_target = '{"text":["今日休館","明日開放"]}'
    english_json_target = '{"text":["Closed today","Open tomorrow"]}'
    wanted_chinese = ["今日休館", "明日開放"]
    wanted_english = ["Closed today", "Open tomorrow"]

    def inspect_json(text, wanted):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"parses": False, "format": False, "matches_requested_scope": False}
        valid_format = isinstance(parsed, dict) and set(parsed) == {"text"} and isinstance(parsed["text"], list)
        return {"parses": True, "format": valid_format, "matches_requested_scope": valid_format and parsed["text"] == wanted}

    json_checks = {
        "chinese_target": inspect_json(json_target, wanted_chinese),
        "plain_text_for_json_request": inspect_json(chinese, wanted_chinese),
        "english_target_for_chinese_request": inspect_json(english_json_target, wanted_chinese),
        "unchanged_chinese_target_for_english_request": inspect_json(json_target, wanted_english),
        "corrected_english_target": inspect_json(english_json_target, wanted_english),
    }
    assert json_checks["chinese_target"] == {"parses": True, "format": True, "matches_requested_scope": True}
    assert json_checks["plain_text_for_json_request"]["format"] is False
    assert json_checks["english_target_for_chinese_request"]["format"] is True
    assert json_checks["english_target_for_chinese_request"]["matches_requested_scope"] is False
    assert json_checks["unchanged_chinese_target_for_english_request"]["matches_requested_scope"] is False
    assert json_checks["corrected_english_target"]["matches_requested_scope"] is True

    print(
        json.dumps(
            {
                "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-15/probe.py",
                "result": "全部斷言通過；錯配目標可轉成訓練資料，格式與指定範圍須分別核對。",
                "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256((ROOT / "tiny_perceptron/data.py").read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256((ROOT / "tiny_perceptron/model.py").read_bytes()).hexdigest(),
                "notice_rows": [{"row": i, "text": text, "utf8_bytes": len(text.encode("utf-8"))} for i, text in enumerate(notice.splitlines(), 1)],
                "cases": records,
                "eos_truncation": {"remaining_effective_targets": len(remaining), "eos_present": tok.eos_id in remaining, "decoded_text_unchanged": tok.decode(remaining) == chinese},
                "json_checks": json_checks,
                "limitations": "手寫範例與確定性的資料處理核對；沒有模型訓練或生成品質量測。",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
