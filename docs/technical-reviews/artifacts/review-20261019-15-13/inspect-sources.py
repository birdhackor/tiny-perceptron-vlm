"""以 curl 讀原始來源，僅保存短摘錄與取得收據。"""

import hashlib
import json
import platform
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


SOURCES = [
    ("mixtral", "https://arxiv.org/html/2401.04088v1", "arXiv:2401.04088v1, 2024-01-08",
     ["the model’s total parameter count", "high performance specialized kernels"],
     ["2.1 Sparse Mixture of Experts", "3 Results"],
     "原文區分總參數與 active 參數，K 調節每個 token 的 expert 計算；另說明專用 kernel、expert 分派與負載平衡。本文只能用它核對方法，不能把 Mixtral 的效益轉移到本專案。"),
    ("randomness", "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/notes/randomness.rst", "PyTorch v2.9.0",
     ["the same series of random numbers will be generated each time", "Completely reproducible results are not guaranteed across PyTorch releases"],
     ["PyTorch random number generator", "Python"],
     "讀取開頭的跨版本／平台限制、manual_seed 段落與 Python RNG 段落；固定種子控制亂數序列，完整重現仍須同環境並排除其他非確定性。本節僅說初始化可重現。"),
    ("validation", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst", "scikit-learn 1.7.2",
     ["training proceeds on the training set,\nafter which evaluation is done on the validation set"],
     ["When evaluating different settings", "A solution to this problem"],
     "讀取訓練／驗證／測試切分的開頭說明：驗證資料用來比較模型設定，模型參數只在訓練資料上學習；反覆對測試集挑設定會洩漏評估資訊。"),
    ("variance", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/common_pitfalls.rst", "scikit-learn 1.7.2",
     ["For an optimal robustness of cross-validation (CV) results"],
     ["Robustness of cross-validation results", "Whether we pass"],
     "讀取 Controlling randomness、Estimators 與 Robustness of cross-validation results 段落；固定單一 RNG 狀態會遮住隨機估計器的變動，評估應考慮不同 RNG 起點。沒有把 sklearn 特定 API 當成本專案 PyTorch 的契約。"),
    ("cross_entropy", "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/loss.py", "PyTorch v2.9.0",
     ["CrossEntropyLoss", "LogSoftmax", "NLLLoss"],
     ["class CrossEntropyLoss", "class MultiLabelSoftMarginLoss"],
     "讀取 CrossEntropyLoss 的 class-index 目標公式、ignore_index、sum／mean reduction，以及等價於 LogSoftmax 後接 NLLLoss 的說明；沒有權重時是有效目標負自然對數機率的和或平均。"),
]


def main():
    records = []
    for identifier, url, version, quotes, bounds, note in SOURCES:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
        raw = subprocess.run(command, check=True, capture_output=True).stdout
        content = raw.decode()
        if identifier == "mixtral":
            parser = Text()
            parser.feed(content)
            content = " ".join(parser.parts)
        normalized = re.sub(r"\s+", " ", content)
        for quote in quotes:
            assert re.sub(r"\s+", " ", quote) in normalized, quote
        start = normalized.rfind(bounds[0]) if identifier == "mixtral" else normalized.find(bounds[0])
        end = normalized.find(bounds[1], start + len(bounds[0]))
        assert start >= 0 and end > start, identifier
        records.append({"id": identifier, "url": url, "version": version, "accessed_on": "2026-10-20",
                        "command": command, "exit_code": 0, "response_bytes": len(raw),
                        "response_sha256": hashlib.sha256(raw).hexdigest(),
                        "read_locator": bounds, "short_quotes": quotes, "inspection_note": note,
                        "full_response_saved": False})
        print(json.dumps({"source": identifier, "version": version, "read_locator": bounds, "short_quotes": quotes,
                          "inspection_note": note}, ensure_ascii=False))
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-13/inspect-sources.py",
              "environment": {"python": platform.python_version(), "transport": "curl HTTPS", "device": "cpu"},
              "sources": records, "result": "all original sources retrieved and inspected"}
    Path(__file__).with_name("original-source-receipts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
