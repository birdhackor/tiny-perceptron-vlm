"""7.26 獨立技術審閱：只建構人工資料與 targets，不建模型或訓練。"""

from contextlib import redirect_stdout
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from itertools import combinations, permutations
import json
from pathlib import Path
import platform
import re

import torch

from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
EXPECTED_SOURCE = "f5112679706435c6ca637d156e45ea92f2d89fc62c0c1ae7424eef09e4f5eb49"
EXPECTED_NOTEBOOK = "20cd96f8e053cbdfb18b0a617379b4a3d8381d52ee84a405d655c64a9eeede87"


def digest(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def main():
    frozen = json.loads((ROOT / FREEZE).read_bytes())
    page = next(p for p in frozen["pages"] if p["page_id"] == "7.26")
    body = (ROOT / page["snapshot"]).read_bytes()
    assert sha256(body).hexdigest() == page["source_sha256"] == EXPECTED_SOURCE
    chapter = (ROOT / page["source"]).read_bytes()
    current = re.search(rb"^## 7\.26 .+?(?=^## |\Z)", chapter, re.M | re.S).group()
    assert current == body
    codes = re.findall(r"```python\n(.*?)```", body.decode(), re.S)
    assert len(codes) == 1, "本節一個 Python 區塊，沒有折疊選讀程式"
    code = codes[0]
    stdout = StringIO()
    namespace = {}
    with redirect_stdout(stdout):
        exec(compile(code, page["snapshot"], "exec"), namespace)
    expected_lines = [
        "原題 → 查卡片進度 ／稽核： 有合適候選",
        "移除正解 → 以上皆不適合 ／稽核： 列表缺正解",
        "增加干擾 → 查卡片進度 ／稽核： 有合適候選",
        "重排 → 查卡片進度 ／稽核： 有合適候選",
        "缺編號 → 詢問申請編號 ／稽核： 有合適候選",
        "不支援 → 以上皆不適合 ／稽核： 不支援",
    ]
    assert stdout.getvalue().splitlines()[:6] == expected_lines

    make, meaning = namespace["make"], namespace["meaning"]
    full, rows = namespace["full"], namespace["rows"]
    assert len(meaning.values()) == len(set(meaning.values()))
    before, after = rows[0][1], rows[1][1]
    assert before["prompt"].split("候選：")[0] == after["prompt"].split("候選：")[0]
    assert before["prompt"].split("候選：")[1].split("／")[1:] == after["prompt"].split("候選：")[1].split("／")
    assert before["answer"] == "查卡片進度" and after["answer"] == "以上皆不適合"
    assert "以上皆不適合" in before["prompt"] and "以上皆不適合" in after["prompt"]
    assert list(reversed(full)) == ["none", "ask_id", "address", "card"]

    exercise = make({"任務": "查卡片進度"}, ["card", "address", "none"])
    assert (exercise["answer"], exercise["audit_reason"]) == ("以上皆不適合", "列表缺正解")

    # 獨立列出世界規則的真值，不從被測函式取得 needed。
    states = [
        ({"任務": "查卡片進度", "申請編號": "K17"}, "card"),
        ({"任務": "查卡片進度"}, "ask_id"),
        ({"任務": "查地址變更規則"}, "address"),
        ({"任務": "查地址變更規則", "申請編號": "K17"}, "address"),
        ({"任務": "重設密碼"}, None),
    ]
    substantive = ("card", "address", "ask_id", "weather")
    swept = 0
    for state, gold_action in states:
        for count in range(len(substantive) + 1):
            for subset in combinations(substantive, count):
                for order in permutations((*subset, "none")):
                    row = make(state, list(order))
                    expected = gold_action if gold_action in subset else "none"
                    expected_reason = (
                        "不支援" if gold_action is None else
                        "列表缺正解" if expected == "none" else "有合適候選"
                    )
                    assert row["answer"] == meaning[expected]
                    assert row["audit_reason"] == expected_reason
                    assert row["answer"] in [meaning[key] for key in order]
                    swept += 1

    tok = ByteTokenizer()
    sft = []
    for name, row in rows + [("練習", exercise)]:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        changed_audit = {**row, "audit_reason": "REVIEWER_AUDIT_SENTINEL"}
        assert messages == [
            {"role": "user", "content": changed_audit["prompt"]},
            {"role": "assistant", "content": changed_audit["answer"]},
        ]
        assert "audit_reason" not in json.dumps(messages, ensure_ascii=False)
        assert "REVIEWER_AUDIT_SENTINEL" not in json.dumps(messages, ensure_ascii=False)
        x, y = render_chat(messages)
        expected_targets = tok.encode(row["answer"]) + [tok.eos_id]
        actual_targets = y[y != -100].tolist()
        assert actual_targets == expected_targets
        first = (y != -100).nonzero()[0].item()
        assert first == len(tok.encode(row["prompt"])) + 3
        assert x[first].item() == tok.assistant_id
        assert (y[:first] == -100).all().item()
        # 只用人工零 logits 驗證 loss 的遮罩；沒有模型 forward 或參數更新。
        zeros = torch.zeros((1, len(x), tok.vocab_size))
        original_loss = masked_loss(zeros, y[None]).item()
        changed = zeros.clone()
        changed[0, y == -100, 0] = 100.0
        changed_loss = masked_loss(changed, y[None]).item()
        assert original_loss == changed_loss
        sft.append({
            "name": name,
            "input_length": len(x),
            "supervised_targets": len(actual_targets),
            "target_text": tok.decode(actual_targets),
            "last_target": actual_targets[-1],
            "first_target_input_id": x[first].item(),
            "ignored_prefix_length": first,
            "zero_logits_loss": original_loss,
            "changed_ignored_logits_loss": changed_loss,
            "audit_sentinel_does_not_enter_messages": True,
        })

    notebook = json.loads((ROOT / page["notebook"]).read_bytes())
    assert digest(page["notebook"])["sha256"] == EXPECTED_NOTEBOOK == page["notebook_sha256"]
    notebook_codes = [(i, "".join(c["source"])) for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"]
    assert len(notebook_codes) == 2
    assert notebook_codes[1][1].rstrip() == code.rstrip()
    notebook_namespace = {}
    notebook_stdout = StringIO()
    with redirect_stdout(notebook_stdout):
        for index, cell in notebook_codes:
            exec(compile(cell, f"{page['notebook']}:cell-{index}", "exec"), notebook_namespace)
    assert notebook_namespace["in_colab"] is False
    assert notebook_stdout.getvalue() == stdout.getvalue()

    result = {
        "reviewer_task": "/root/omni_p5_technical_7_26",
        "executed_at": datetime.now(UTC).isoformat(),
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-7.26/verify_inline_v1.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU"},
        "inputs": [digest(FREEZE), digest(page["snapshot"]), digest(page["source"]), digest(page["notebook"]), digest("tiny_perceptron/data.py"), digest("tiny_perceptron/model.py")],
        "current_source_equals_freeze": True,
        "inline_stdout": stdout.getvalue(),
        "expected_six_lines_match_exactly": True,
        "minimal_pair_changes_only_candidate_list": True,
        "exercise": exercise,
        "candidate_subset_permutation_checks": {"states": len(states), "cases": swept, "all_passed": True, "domain": "有效既有候選 ID；每列含唯一 none；編號只分有合法字串／欄位缺失；不含重複候選"},
        "sft_mask_checks": sft,
        "notebook": {"local_bootstrap_executed": True, "teaching_code_exactly_matches_source": True, "stdout_matches_source": True, "colab_execution": False},
        "model_forward_calls": 0,
        "parameter_updates": 0,
        "limits": ["沒有檢查模型是否學會、泛化或拒答原因辨識。", "沒有執行 Colab、pip 安裝分支、下載大型權重或正式 final-family NN。", "make 以鍵是否存在判斷申請編號，沒有號碼格式／空值驗證或一般輸入校驗。", "沒有候選語義去重或資料家族切分器；那些是正文要求，這個小程式未聲稱自動完成。"],
        "result": "PASS: all deterministic assertions passed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
