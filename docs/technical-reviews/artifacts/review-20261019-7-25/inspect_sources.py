import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime


specifications = [
    {
        "id": "snell",
        "url": "https://arxiv.org/html/2408.03314v1",
        "version": "arXiv:2408.03314v1，2024-08-06",
        "marker": "2408.03314v1",
        "locator": "Test-Time Compute-Optimal Scaling Strategy；Estimating Question Difficulty for Compute-Optimal Scaling；Experimental Setup",
        "excerpt": "assumes oracle access to a ground-truth correctness checking function, which is of course not available upon deployment",
        "inspection": "已讀計算預算下按題選策略的定義、oracle 與模型預測難度的區分，以及額外難度判斷成本的限制。論文承認沒有將該項成本納入自己的效益曲線，因此不把其效益數字當成完整服務成本的證據。其任務設定也假定推理所需基本知識已具備。",
    },
    {
        "id": "routellm",
        "url": "https://arxiv.org/html/2406.18665v1",
        "version": "arXiv:2406.18665v1，2024-06-26",
        "marker": "2406.18665v1",
        "locator": "Problem Formulation；Quantifying dataset and benchmark similarity；Routing Overhead",
        "excerpt": "A concern with LLM routing is the overhead of routing as compared to using a single model.",
        "inspection": "已讀由輸入選擇模型的函數定義、品質與成本目標、分布相似度與路由表現的關係，以及路由自身的吞吐與成本量測。研究對象是兩個模型之間的路由；只用來佐證選擇器機制與成本，沒有外推為本專案 Auto 的實測成績。",
    },
    {
        "id": "sklearn",
        "url": "https://scikit-learn.org/1.9/modules/cross_validation.html",
        "version": "scikit-learn 1.9.1 官方文件，1.9 版本網址",
        "marker": "1.9.1",
        "locator": "Cross-validation: evaluating estimator performance，開頭驗證資料與測試資料的說明",
        "excerpt": "there is still a risk of overfitting on the test set because the parameters can be tweaked",
        "inspection": "已讀訓練集、驗證集、測試集的分工，以及用測試結果反覆調設定造成的洩漏。原文明說有限切分的結果會依所選資料而變化；這支持本節不從四筆人工紀錄宣稱普遍排序。",
    },
    {
        "id": "python",
        "url": "https://docs.python.org/3.13/library/stdtypes.html#truth-value-testing",
        "version": "Python 3.13 官方標準型別文件；本地執行版本 3.13.16",
        "marker": "Truth Value Testing",
        "locator": "Truth Value Testing",
        "excerpt": "empty sequences and collections:",
        "inspection": "已讀空序列與空容器被視為 false 的規則；清單不是空的即為 true，與清單元素本身是否為 0 無關。另以本地程式直接核對 [] 與 [0]。",
    },
    {
        "id": "sre",
        "url": "https://sre.google/sre-book/monitoring-distributed-systems/",
        "version": "Google Site Reliability Engineering 線上書，Monitoring Distributed Systems；頁面標示 Copyright 2017",
        "marker": "Monitoring Distributed Systems",
        "locator": "Worrying About Your Tail (or, Instrumentation and Performance)",
        "excerpt": "The simplest way to differentiate between a slow average and a very slow \"tail\" of requests",
        "inspection": "已讀延遲分布與尾端會被平均值掩蓋的討論，以及原書 Bigtable 案例。僅支持本節量測延遲分布的建議，沒有將該系統的特定百分位或成本外推到教材模型。",
    },
    {
        "id": "nist",
        "url": "https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-7-rules-and-style-conventions-expressing-values",
        "version": "NIST Guide to the SI 線上版；頁面標示更新日期 2025-08-18",
        "marker": "Updated August 18, 2025",
        "locator": "Value and numerical value of a quantity",
        "excerpt": "The value of a quantity is its magnitude expressed as the product of a number and a unit",
        "inspection": "已讀量值等於數值與單位之積，以及必須清楚定義所量之量的規則。輪次與 token 雖都是計數，所計物件仍不同；沒有換算規則時，相加不能代表本節定義的輪數、token 數或耗時。",
    },
]

records = []
for spec in specifications:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", spec["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    normalized = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw.decode())))
    assert spec["marker"] in normalized, "來源版本或標題不符"
    assert spec["excerpt"] in normalized, "來源摘錄不符"
    assert len(spec["excerpt"].split()) <= 25, "摘錄過長"
    records.append({
        **{key: value for key, value in spec.items() if key != "marker"},
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "retrieval_command": command,
        "exit_code": completed.returncode,
        "full_response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "checked_original": True,
        "storage": "只保存支持主張的短摘錄與讀取收據，未保存第三方全文。",
    })
print(json.dumps({"sources": records, "result": "所有原始來源已由 curl 讀取並核對短摘錄。"}, ensure_ascii=False, indent=2))
