"""為19.24尚待工程的規劃保存當版文字與已公開B契約的核對結果。"""

import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    contract_path = "docs/course-revision-20261029-omni/CONTRACTS.md"
    contract = (ROOT / contract_path).read_text()
    statements = [
        "尚無新版能力結果",
        "舊任務仍走原A",
        "文字不能覆寫用來驗收或發聲的結果",
        "C核心final為60家族",
        "另有100個預先留出的Auto整合家族",
        "同一100家族成對跑gold/真C路由",
        "C路由只讀文字及附件存在資訊，不讀gold task或影音內容",
        "人工指定仍需過輸入契約",
        "原B權重、config、tokenizer與depth policy固定",
        "舊算式世界只有84個獨立三數家族",
        "保留能力面板跑新版wrapper與固定舊版",
        "已批准的GitHub Release版本化資產通道",
        "保留原結果",
    ]
    checks = [{"statement": item, "found": item in contract} for item in statements]
    assert all(item["found"] for item in checks)
    policy_path = "docs/lesson-products/results/b-auto-policy.json"
    policy = json.loads((ROOT / policy_path).read_bytes())
    assert policy["sum_depth"] == 1
    source_path = "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/19.24.md"
    source = (ROOT / source_path).read_text()
    scope_quotes = [
        'return {"route": "b_sum", "state": text, "loops": 3}',
        "這兩行只核對控制流程",
        "程式也還沒有讓B算出答案",
        "兩段程式是人工控制流程示例；新版整合與保留能力的實測仍待完成",
    ]
    assert all(quote in source for quote in scope_quotes)
    result = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-24-v1/inspect_contract.py",
        "environment": {"python": __import__("platform").python_version(), "device": "cpu; text inspection only"},
        "contract": {"path": contract_path, "sha256": digest(contract_path), "checks": checks},
        "policy": {"path": policy_path, "sha256": digest(policy_path), "sum_depth": policy["sum_depth"]},
        "source": {"path": source_path, "sha256": digest(source_path), "scope_quotes": scope_quotes},
        "scope_judgment": "人工範例回傳loops=3，但沒有執行B；固定舊Auto sum_depth=1是正式契約。三處明確範圍句已將範例與正式產品分開，故本次不將不同設定當成已完成產品違約或能力退步。",
        "result": "All current text-contract assertions passed; no model capability measured.",
    }
    path = Path(__file__).with_name("contract-inspection.json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(path.relative_to(ROOT)), "sha256": digest(str(path.relative_to(ROOT))),
                      "result": result["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
