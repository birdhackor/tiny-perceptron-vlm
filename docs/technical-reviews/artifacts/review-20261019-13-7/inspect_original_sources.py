"""以 curl 讀取原始來源，只保存來源收據與支持主張的小段摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html import unescape


def normalized(raw):
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.S)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())


specifications = [
    {
        "title": "Disentangling Length from Quality in Direct Preference Optimization",
        "url": "https://arxiv.org/html/2403.19159v2",
        "version": "arXiv:2403.19159v2，2024-09-09",
        "locator": "Introduction；Direct Preference Optimization；Building in Explicit Regularization in DPO；What Drives Length Exploitation?；Limitations",
        "excerpt": "they all show significant length bias out-of-distribution",
        "inspection_note": "已讀偏好資料的長度傾向、DPO 的整篇策略／參考比值、額外長度正則項與分布外長度偏差分析。研究顯示參考比值仍可有長度偏差；結論限於作者研究的模型、資料與配方，不是所有任務必然偏長。",
    },
    {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "url": "https://arxiv.org/html/2305.18290v3",
        "version": "arXiv:2305.18290v3",
        "locator": "Direct Preference Optimization：Deriving the DPO objective，顯示式標籤 (7)；What does the DPO update do?",
        "excerpt": "the Bradley-Terry model depends only on the difference of rewards between two completions",
        "inspection_note": "已讀 DPO 目標與梯度推導。目標內比較兩篇完整條件機率相對固定參考的 log 比值；這不是正確答案的校準信心，也不是各篇 token 分數平均。整體 margin 增加並不約束 chosen 機率必須獨自增加。",
    },
    {
        "title": "NIST/SEMATECH e-Handbook of Statistical Methods — Confounding (also called aliasing)",
        "url": "https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm",
        "version": "2026-10-20 實際讀取的網頁，以完整 HTML SHA-256 識別",
        "locator": "Confounding means we have lost the ability to estimate some effects and/or interactions",
        "excerpt": "Confounding means we have lost the ability to estimate some effects and/or interactions",
        "inspection_note": "已讀 confounding 的定義與因子效應不可分離的例子。教材把長度與品質同時改變造成的原因難分稱作混淆，符合這個廣義的設計概念；不把教材例子宣稱為完整的隨機因果實驗。",
    },
]
receipts = []
for spec in specifications:
    argv = ["curl", "-fsSL", "--retry", "2", "--max-time", "30", spec["url"]]
    result = subprocess.run(argv, check=True, capture_output=True)
    text = normalized(result.stdout.decode("utf-8"))
    assert spec["excerpt"] in text
    receipts.append({
        **spec,
        "command_argv": argv,
        "exit_code": result.returncode,
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "accessed_on": "2026-10-20",
        "original_bytes": len(result.stdout),
        "original_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "excerpt_matched_original": True,
        "checked_original": True,
        "full_text_saved": False,
    })
print(json.dumps({"sources": receipts, "result": "三份原始來源均由 curl 成功讀取，摘錄逐字匹配"}, ensure_ascii=False, indent=2))
