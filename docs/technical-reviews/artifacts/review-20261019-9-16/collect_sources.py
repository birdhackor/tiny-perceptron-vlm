"""用 curl 讀取原始來源，只保存雜湊、少量摘錄及核對定位。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return command, result.stdout


def main():
    revision = "d9be2f98d27aaa27f9afacb1d882ef39d3fc8211"
    sources = [
        ("launch", "https://deskmind.dev/zh/blog/launch/", "作者署名 gxcsoccer；發布日期 2026-10-04；本次讀取的動態頁面以全文 SHA 辨識。",
         ["DeskMind 是一个开源的 Computer Use Agent", "接口和 Jev 一样"],
         "已閱讀全文，重點核對『先做眼睛，再做大脑』、『做成了吗』及作者署名。作者自述用 Qwen3.5 加 LoRA、自製 Jev 式介面；自測限於自己出的桌面任務，並明說限制。頁面沒有宣稱它是 TypeSafe 官方訓練文件。"),
        ("training", f"https://raw.githubusercontent.com/deskmind-ai/brain/{revision}/docs/training.md", revision,
         ["a teacher gives each one a probability distribution", "LoRA on Qwen3.5-0.8B and 4B", "answer options are shuffled per example"],
         "已逐行閱讀 Pipeline 與 Training command：教師分布、LoRA、KL 與 CE 混合、--shuffle-options，且範例命令確實啟用 shuffle。這些是作者自己的訓練流程，並非 Jev 專有模型的完整配方。"),
        ("threshold", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/classification_threshold.rst", "scikit-learn 1.7.2",
         ["the decision problem to take concrete action based on those probability predictions"],
         "已全文閱讀開頭、Post-tuning the decision threshold 與 Important notes regarding the internal cross-validation：預測與行動可分離，門檻依效用而變，調門檻應另用驗證資料。文件亦明示改門檻可保持 predict_proba 不變。"),
        ("python", "https://docs.python.org/3.13/tutorial/floatingpoint.html", "Python 3.13.16 documentation",
         ["most decimal fractions cannot be represented exactly as binary fractions"],
         "已閱讀 Floating-Point Arithmetic: Issues and Limitations 與 Representation Error：十進位小數通常近似為 binary64，顯示四捨五入不修復底層表示。具體 0.8 的成本值另以本機 Python 驗算。"),
        ("calibration", "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf", "PMLR 70:1321–1330，ICML 2017 出版版",
         [],
         "原 PDF 另經網頁工具讀取 Definitions 的 calibration 定義 P(correct | confidence=p)=p 與 Calibration Methods 的獨立驗證資料及同分布假設；只能支持在相應分布上估算答對機會，不能支持單題保證或資料外泛化。此處 curl 取得的原 PDF 僅留 SHA，不落盤。"),
        ("selective", "https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf", "NeurIPS 2017 出版版，Geifman 與 El-Yaniv",
         [],
         "原 PDF 另經網頁工具讀取 Introduction 的誤判／拒絕成本設定及 Problem Setting 的 coverage=E[g]、selective risk=E[loss*g]/coverage，並讀取獨立保留測試資料的實驗安排；只報自動題正確率不足以描述覆蓋率與總成本。此處 curl 取得的原 PDF 僅留 SHA，不落盤。"),
    ]
    receipts = []
    for identifier, url, version, excerpts, note in sources:
        command, raw = fetch(url)
        if not url.endswith(".pdf"):
            text = raw.decode()
            if "<html" in text.lower() or "<!doctype" in text.lower():
                parser = PlainText()
                parser.feed(text)
                text = " ".join(parser.parts)
            text = re.sub(r"\s+", " ", text)
            for excerpt in excerpts:
                assert excerpt in text, "來源中找不到預期摘錄：" + identifier
        receipts.append({"id": identifier, "url": url, "version": version, "command": command,
                         "accessed_at": datetime.now(timezone.utc).isoformat(),
                         "original_sha256": hashlib.sha256(raw).hexdigest(), "original_bytes": len(raw),
                         "excerpts": excerpts, "inspection_note": note,
                         "checked_original": True})
    print(json.dumps({"environment": {"python": platform.python_version(),
                                      "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
                      "result": "來源下載成功；文字摘錄比對成功；未保存第三方全文。",
                      "receipts": receipts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
