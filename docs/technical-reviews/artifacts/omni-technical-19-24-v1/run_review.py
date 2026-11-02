"""19.24 獨立技術核對：執行凍結教材原碼，不載入成品權重或訓練。"""

import contextlib
import datetime
import hashlib
import io
import itertools
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.lesson_products.data import arithmetic_worlds  # noqa: E402
from tiny_perceptron.lesson_products.inference import parse_state, rule_depth  # noqa: E402
from tiny_perceptron.lesson_products.model import Tokenizer, padded  # noqa: E402


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rejected(function, value):
    try:
        function(value)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"Unexpected acceptance: {value!r}")


def main():
    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
    freeze = json.loads(freeze_path.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "19.24")
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert sha(raw) == page["source_sha256"]
    current = (ROOT / page["source"]).read_bytes()
    current_section = re.search(rb"^## 19\.24 .*?(?=^## |\Z)", current, re.M | re.S).group()
    assert current_section == raw
    assert page["figures_sha256"] == {} and not re.search(rb"!\[[^\]]*\]\(", raw)
    blocks = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)
    assert len(blocks) == 2
    namespace = {}
    block_runs = []
    for index, code in enumerate(blocks, 1):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, f"{page['snapshot']}:python-block-{index}", "exec"), namespace)
        block_runs.append({"index": index, "code_sha256": sha(code.encode()), "stdout": stream.getvalue()})
    assert block_runs[0]["stdout"] == (
        "{'route': 'b_sum', 'state': '三數加法;x=1;y=2;z=3', 'loops': 3}\n"
        "{'route': 'handoff', 'reason': 'B輸入契約未通過'}\n"
    )
    assert block_runs[1]["stdout"] == (
        "指定正確入口： 3 / 4\nC選對入口： 2 / 4\n真路由整條成功： 1 / 4\n"
    )
    adapt, select = namespace["adapt_sum"], namespace["select_route"]
    tokenizer = Tokenizer()
    token_lengths = set()
    for numbers in itertools.product(range(7), repeat=3):
        state = adapt(list(numbers))
        assert state == f"三數加法;x={numbers[0]};y={numbers[1]};z={numbers[2]}"
        assert parse_state(state) == {"task": "sum"}
        token_lengths.add(len(tokenizer.encode(state)))
        assert padded([tokenizer.encode(state)], 128).dtype == torch.long
    invalid_values = [None, (1, 2, 3), [], [1, 2], [1, 2, 3, 4], [True, 2, 3],
                      [1, False, 3], [1.0, 2, 3], ["1", 2, 3], [-1, 2, 3], [1, 2, 7]]
    rejected_inputs = [{"input": repr(value), "error": rejected(adapt, value)} for value in invalid_values]
    request = {"text": "計算三個數", "numbers": [1, 2, 3]}
    good = {"route": "b_sum", "p_max": 0.96, "info_yes": True}
    insufficient = dict(good, info_yes=False)
    exercise = {
        "automatic": select(request, insufficient, tau=0.9),
        "manual_b_sum": select(request, insufficient, tau=0.9, manual="b_sum"),
    }
    assert exercise["automatic"]["route"] == "handoff"
    assert exercise["manual_b_sum"] == {"route": "b_sum", "state": adapt([1, 2, 3]), "loops": 3}
    assert select(request, dict(good, p_max=0.9), 0.9)["route"] == "b_sum"
    assert select(request, dict(good, p_max=0.899), 0.9)["route"] == "handoff"
    assert select(dict(request, numbers=[1, 2, 7]), insufficient, 0.9, "b_sum")["route"] == "handoff"
    routes = {
        "unknown": select(request, dict(good, route="none"), 0.9),
        "a_text": select(request, dict(good, route="a_text"), 0.9),
        "av_missing": select(request, dict(good, route="a_av"), 0.9),
        "av_present": select(dict(request, has_audio=True, has_video=True), dict(good, route="a_av"), 0.9),
    }
    assert routes["unknown"]["route"] == routes["av_missing"]["route"] == "handoff"
    assert routes["a_text"]["text"] == request["text"]
    assert routes["av_present"]["output"] == "canonical semantic result"
    assert "answer" not in exercise["manual_b_sum"]
    grammar = {
        "sum_out_of_range": rejected(parse_state, "三數加法;x=1;y=2;z=7"),
        "boxes_valid": parse_state("兩盒追蹤;A=2;B=6;事件=交換;查詢=B"),
        "boxes_unknown_event": rejected(parse_state, "兩盒追蹤;A=2;B=6;事件=加一;查詢=B"),
        "boxes_too_many_events": rejected(parse_state, "兩盒追蹤;A=2;B=6;事件=" + ",".join(["交換"] * 7) + ";查詢=B"),
        "unknown_token": rejected(tokenizer.encode, "🙂"),
        "length_rejection": rejected(lambda tokens: padded([tokens], 3), [1, 2, 3, 4]),
    }
    worlds = list(arithmetic_worlds())
    assert len(worlds) == 343
    assert len({w["family_id"] for w in worlds}) == 84 == math.comb(9, 3)
    policy_path = ROOT / "docs/lesson-products/results/b-auto-policy.json"
    policy = json.loads(policy_path.read_bytes())
    old_sum_loops = rule_depth(policy, "sum")
    assert old_sum_loops == 1
    paired_rows = namespace["rows"]
    observed = {
        "gold": [i + 1 for i, row in enumerate(paired_rows) if row["gold_backend_ok"]],
        "route": [i + 1 for i, row in enumerate(paired_rows) if row["route_ok"]],
        "joint": [i + 1 for i, row in enumerate(paired_rows) if row["route_ok"] and row["actual_backend_ok"]],
        "denominator": len(paired_rows),
    }
    assert observed == {"gold": [1, 2, 4], "route": [1, 3], "joint": [1], "denominator": 4}
    prerequisites = []
    for lesson, ranges in {"19.17": [(1, 48), (60, 60)], "19.19": [(1, 30), (52, 56)],
                           "19.21": [(3, 7), (220, 232), (239, 241)],
                           "19.22": [(3, 7), (200, 208)]}.items():
        entry = next(p for p in freeze["pages"] if p["page_id"] == lesson)
        snapshot = (ROOT / entry["snapshot"]).read_bytes()
        assert sha(snapshot) == entry["source_sha256"]
        live = re.search(rb"^## " + lesson.encode().replace(b".", rb"\.") + rb" .*?(?=^## |\Z)", current, re.M | re.S).group()
        assert snapshot == live
        lines = snapshot.splitlines(keepends=True)
        prerequisites.append({"lesson_id": lesson, "snapshot": entry["snapshot"],
                              "sha256": sha(snapshot), "current_section_equal": True,
                              "slices": [{"lines": f"{start}-{end}", "sha256": sha(b"".join(lines[start-1:end]))}
                                         for start, end in ranges]})
    result = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-24-v1/run_review.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
        "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path.read_bytes())},
        "source": {"path": page["snapshot"], "sha256": sha(raw), "current_section_equal": True, "figures": {}},
        "blocks": block_runs, "adapter": {"accepted_triples": 343, "token_lengths": sorted(token_lengths), "rejected": rejected_inputs},
        "exercise": exercise, "other_routes": routes, "grammar": grammar,
        "arithmetic_worlds": {"rows": len(worlds), "independent_family_ids": 84, "combinatoric_count": "C(7+3-1,3)=C(9,3)=84"},
        "paired_metrics": observed,
        "old_policy_comparison": {"path": str(policy_path.relative_to(ROOT)), "sha256": sha(policy_path.read_bytes()),
                                  "frozen_sum_depth": old_sum_loops, "lesson_fixture_loops": 3,
                                  "note": "19.24硬編碼三輪與固定舊Auto加法一輪不同；此處僅記觀察，判斷見報告。"},
        "prerequisites": prerequisites,
        "scope": "僅精確教材原碼、公開文法、確定性記帳；未載入A/B/C成品、未訓練、未聲稱模型品質或正式整合結果。",
        "result": "All deterministic assertions passed."
    }
    destination = Path(__file__).with_name("execution.json")
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination.read_bytes()),
                      "result": result["result"], "block_stdout": [b["stdout"] for b in block_runs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
