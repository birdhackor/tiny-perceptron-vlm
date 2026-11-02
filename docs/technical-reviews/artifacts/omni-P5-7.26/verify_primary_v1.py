"""只執行可攜短摘錄中的 Kev 資料轉換，沒有下載、資料集或模型。"""

from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import random


def main():
    source = Path(__file__).with_name("primary-inspection-v1.json")
    receipt = json.loads(source.read_bytes())
    kev = receipt["sources"][0]
    snippets = kev["supporting_excerpts"]
    namespace = {}
    # 這兩段只含常數、augment 與 none_pair，無第三方套件或資料讀取。
    exec(compile(snippets[0]["text"] + snippets[1]["text"], "pinned-kev-short-excerpts", "exec"), namespace)
    augment, none_pair = namespace["augment"], namespace["none_pair"]
    request = {
        "state": {"任務": "查卡片進度", "申請編號": "K17"},
        "questions": {"next": {
            "type": "choice",
            "instructions": "Pick the matching action.",
            "criteria": {"card": "查卡片進度", "address": "查地址變更規則", "ask_id": "詢問申請編號"},
            "label": "card",
        }},
    }
    original = deepcopy(request)
    samples = {}
    for branch, probabilities in [
        ("remove_gold_add_correct_none", (1.0, 0.0, 0.0)),
        ("keep_gold_add_wrong_none", (0.0, 1.0, 0.0)),
        ("keep_gold_add_distractor", (0.0, 0.0, 1.0)),
        ("permute_only", (0.0, 0.0, 0.0)),
    ]:
        out = augment(request, random.Random(0), *probabilities)
        q = out["questions"]["next"]
        assert out["state"] == request["state"]
        assert q["instructions"] == request["questions"]["next"]["instructions"]
        if branch == "remove_gold_add_correct_none":
            assert "card" not in q["criteria"] and q["label"] in q["criteria"]
            assert q["label"] not in request["questions"]["next"]["criteria"]
        elif branch in ("keep_gold_add_wrong_none", "keep_gold_add_distractor"):
            assert q["label"] == "card" and len(q["criteria"]) == 4
            assert q["criteria"]["card"] == "查卡片進度"
        else:
            assert q["label"] == "card"
            assert set(q["criteria"]) == {"card", "address", "ask_id"}
        samples[branch] = q
    assert request == original

    pair = none_pair(request, random.Random(0))
    assert len(pair) == 2
    present, absent = [row["questions"]["next"] for row in pair]
    assert pair[0]["state"] == pair[1]["state"] == request["state"]
    assert present["label"] == "card"
    new = set(present["criteria"]) - set(request["questions"]["next"]["criteria"])
    assert len(new) == 1
    none_key = next(iter(new))
    assert none_key in absent["criteria"] and absent["label"] == none_key
    assert list(absent["criteria"]) == [key for key in present["criteria"] if key != "card"]
    assert absent["criteria"] == {key: value for key, value in present["criteria"].items() if key != "card"}
    assert present["instructions"] == absent["instructions"]
    assert request == original

    soft = deepcopy(request)
    soft["questions"]["next"]["target"] = {"card": 1.0}
    assert none_pair(soft, random.Random(0)) == []
    soft_augmented = augment(soft, random.Random(0), 1.0, 0.0, 0.0)["questions"]["next"]
    assert soft_augmented["criteria"] == soft["questions"]["next"]["criteria"]
    assert soft_augmented["target"] == {"card": 1.0}
    two_options = deepcopy(request)
    two_options["questions"]["next"]["criteria"].pop("ask_id")
    assert none_pair(two_options, random.Random(0)) == []

    output = {
        "reviewer_task": "/root/omni_p5_technical_7_26",
        "executed_at": datetime.now(UTC).isoformat(),
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-7.26/verify_primary_v1.py",
        "environment": {"python": platform.python_version(), "device": "CPU / Python only", "model": "none"},
        "portable_input": {"path": "docs/technical-reviews/artifacts/omni-P5-7.26/primary-inspection-v1.json", "sha256": sha256(source.read_bytes()).hexdigest()},
        "original_source": {"url": kev["url"], "version": kev["version"], "sha256": kev["sha256"]},
        "forced_branches": samples,
        "none_pair": pair,
        "none_pair_changes_only_gold_presence_and_label": True,
        "none_pair_other_candidate_relative_order_identical": True,
        "soft_target_only_permuted": True,
        "soft_target_and_two_option_none_pair_excluded": True,
        "result": "PASS: pinned original source excerpt behavior matches the lesson citation",
        "limits": "只核資料操作。沒有載入 Kev 套件、公開資料集、任何權重或推論，也沒有核對 Kev 的訓練品質。",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
