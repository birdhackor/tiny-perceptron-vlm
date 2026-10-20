"""以 curl 讀原文，只保存收據及支持主張的短摘錄。"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


specs = [
    {
        "id": "s_ctrl", "url": "https://arxiv.org/html/1909.05858v2",
        "version": "arXiv:1909.05858v2, 2019-09-20",
        "locator": "Language Modeling with CTRL：條件分解與前置控制碼",
        "excerpt": "by training on sequences of raw text prepended with control codes.",
        "inspection": "閱讀 Language Modeling 及 Language Modeling with CTRL：條件分布是 p(x|c)，以控制碼作前綴，再用交叉熵訓練。普通 bytes 能表示不同條件是本專案編碼器的直接性質；論文不保證此兩筆示範的泛化品質。",
        "inspect_from": "CTRL is a conditional language model", "inspect_chars": 3600,
    },
    {
        "id": "s_sft", "url": "https://arxiv.org/html/2203.02155v1",
        "version": "arXiv:2203.02155v1, 2022-03-04",
        "locator": "Methods / Models / Supervised fine-tuning (SFT)",
        "excerpt": "We fine-tune GPT-3 on our labeler demonstrations using supervised learning.",
        "inspection": "閱讀 Methods 的 Collect demonstration data 與 Models 的 SFT 段：以標註者示範微調；後續 RM 與 PPO 是另兩個步驟。本文的資料格式展示不是這套訓練或任何未公開訓練的重現。",
        "inspect_from": "We fine-tune GPT-3 on our labeler demonstrations", "inspect_chars": 1900,
    },
    {
        "id": "s_loop", "url": "https://arxiv.org/html/2502.05171v2",
        "version": "arXiv:2502.05171v2, 2025-02-17",
        "locator": "A scalable recurrent architecture / Macroscopic Design；Training Objective / Training Recurrent Models through Unrolling",
        "excerpt": "we randomly sample iteration counts during training",
        "inspection": "閱讀 Macroscopic Design 的 e=P(x)、s_i=R(e,s_(i-1))、p=C(s_r)，及 Training Objective 的 E_x E_r L(m_theta(x,r),x')：相同輸入與 next-token 目標可以配不同輪數，內部是 tensor 狀態。亦讀到截斷反傳只通過末端若干輪；沒有把本文簡短說明當作這篇大規模實驗的重現。",
        "inspect_from": "3.3 Training Objective Training Recurrent Models", "inspect_chars": 3600,
    },
    {
        "id": "s_group", "url": "https://scikit-learn.org/1.7/modules/cross_validation.html",
        "version": "scikit-learn 1.7 官方版本文件",
        "locator": "Cross-validation iterators for grouped data / Group K-fold",
        "excerpt": "GroupKFold is a variation of K-fold which ensures that the same group is not represented in both testing and training sets.",
        "inspection": "閱讀 grouped data 的相依樣本與未見群組評估說明。把原問題當 group，low/high 是同群的兩個版本；若跨 train/test，不能宣稱測試問題未見。這不禁止另行定義只測已見題新檔位的實驗。",
        "inspect_from": "3.1.2.4. Cross-validation iterators for grouped data", "inspect_chars": 2000,
    },
]
receipts = []
for spec in specs:
    command = ["curl", "-fsSL", "--max-time", "45", spec["url"]]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(result.stdout.decode("utf-8"))
    text = " ".join(" ".join(parser.parts).split())
    excerpt = spec["excerpt"]
    assert excerpt in text
    start = text.rfind(spec["inspect_from"])
    assert start >= 0
    print(json.dumps({"source_id": spec["id"], "exit_code": result.returncode, "excerpt_located": True, "original_response_sha256": hashlib.sha256(result.stdout).hexdigest()}, ensure_ascii=False))
    receipts.append({k: v for k, v in spec.items() if k not in ("inspect_from", "inspect_chars")} | {
        "command_argv": command, "exit_code": result.returncode,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_response_bytes": len(result.stdout),
        "original_response_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "checked_original": True,
    })

from pathlib import Path

Path(__file__).with_name("source-receipts.json").write_text(
    json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
