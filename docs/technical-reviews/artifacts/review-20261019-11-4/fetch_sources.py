import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


SOURCES = [
    {
        "id": "vqa", "url": "https://arxiv.org/html/1612.00837v3",
        "version": "arXiv:1612.00837v3, 2017-05-15",
        "locator": "Abstract; Introduction; Dataset",
        "needle": "a pair of similar images that result in two different answers to the question",
        "excerpt": "a pair of similar images that result in two different answers to the question",
        "inspection_note": "同一問題配上答案不同的相近圖片，針對忽略視覺而依賴語言先驗的捷徑；只支持這類受控對照的動機，不證明本文玩具模型已有一般視覺能力。",
    },
    {
        "id": "llava", "url": "https://arxiv.org/html/2304.08485v2",
        "version": "arXiv:2304.08485v2",
        "locator": "Architecture; Training, Table 2 caption and target-answer probability equation",
        "needle": "The model is trained to predict the assistant answers and where to stop",
        "excerpt": "The model is trained to predict the assistant answers and where to stop",
        "inspection_note": "讀取影像編碼器將實際圖片轉成 visual features 的架構，以及以圖與問題作條件、只對助理回答與停止符計算自回歸目標的段落。這是回答遮罩的一個標準做法，不是所有 VQA 系統唯一的訓練方法。",
    },
    {
        "id": "groups", "url": "https://scikit-learn.org/1.7/modules/cross_validation.html",
        "version": "scikit-learn 1.7.2 documentation",
        "locator": "Cross-validation iterators for grouped data / Group K-fold",
        "needle": "the same group is not represented in both testing and training sets",
        "excerpt": "the same group is not represented in both testing and training sets",
        "inspection_note": "讀取 grouped data 的泛化對象與 GroupKFold 說明：同一群組的多筆紀錄若跨訓練與測試，可能只測到群組特徵記憶；本文把原圖視為群組符合其新圖測試目的。未在專案新增或呼叫 sklearn。",
    },
    {
        "id": "repro", "url": "https://docs.pytorch.org/docs/2.14/notes/randomness.html",
        "version": "PyTorch 2.14 documentation; page last updated 2026-05-14",
        "locator": "Reproducibility opening; Controlling sources of randomness / PyTorch random number generator",
        "needle": "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.",
        "excerpt": "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.",
        "inspection_note": "讀取跨版本／平台不保證逐位元重現的限制，以及在固定環境、排除其他非確定性後固定 RNG seed 才有相同序列的條件。正文要求保存生成器種子、版本與參數是重建材料的必要紀錄，未宣稱跨硬體訓練可精確重現。",
    },
]

receipts = []
for source in SOURCES:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", source["url"]]
    completed = subprocess.run(command, check=True, capture_output=True)
    parser = TextParser()
    parser.feed(completed.stdout.decode("utf-8"))
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    assert source["needle"].casefold() in text.casefold()
    receipts.append({
        **source, "checked_original": True, "accessed_on": datetime.now(UTC).date().isoformat(),
        "command_argv": command, "exit_code": completed.returncode,
        "downloaded_bytes": len(completed.stdout),
        "downloaded_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "excerpt_present_in_original": True,
        "storage": "全文僅在本次程序記憶體中讀取；此檔只保存短摘錄、定位與收據。",
    })
print(json.dumps({"sources": receipts}, ensure_ascii=False, indent=2))
