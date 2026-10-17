# 正文內嵌實作與試讀問題修訂

本輪從 `b05ebda3527789e4b6c550ad20271ababce251cd` 開始，在 branch `course-inline-code-20261017` 進行。作者的要求有兩件：

1. 重要的程式碼放進教學正文，像 [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/first-steps/) 那樣邊看程式邊講；Colab 是想自己動手跑的學生的便利入口，不是學習必經之處。
2. 處理 19 位模擬學生（AI subagent，不是真人）試讀全書後提出的問題。

作者寫作規則見[AUTHORING.md](AUTHORING.md)，各實作放在哪一節見[PLACEMENT.md](PLACEMENT.md)。

## 修訂前的狀況

- 312 節中有 275 節含可執行的 Python 程式（共 282 段），網站把程式與 CPU 實際輸出放在同一頁（`scripts/export_course.py` 組頁時取用已執行 Notebook 的輸出）。所以問題不是「正文沒有程式」。
- 正文程式從 `tiny_perceptron` import 的頂層類別與函式共 64 個，正文展示其實作的是 0 個。盤點方法：用 `ast` 解析每段 python fence 的 import，再找該名稱的 `class`／`def` 是否出現在任何正文程式中。例如 4.6 只寫 `from tiny_perceptron.model import TinyLM, ModelConfig`，`TinyLM` 怎麼把字表、位置、block 與輸出層接起來，學生要自己打開原始碼。
- 37 節沒有任何 Python 程式；其中一部分是純文字講解的機制，一部分是成品操作或實測結果。
- 每節頁尾的主按鈕是「在 Colab 動手做 ↗」，文字是「要試做本節練習，可開啟 Colab」，把 Colab 擺成動手的主場。

## 設計決策

**D1 實作怎麼進正文。** 在第一次教某個零件的那一節，用可執行的 python fence 展示實作，分三種寫法：

- 原樣節錄：fence 第一行是 `# 節錄自 tiny_perceptron/<檔案>.py：<名稱>`，其後逐字等於函式庫中那個頂層定義。用於不含後面章節才教的參數、篇幅短的定義。
- 精簡版：第一行是 `# 精簡自 tiny_perceptron/<檔案>.py：<名稱>（省略…）`，作者重寫只含本節所需部分的版本，同一節用 `assert` 確認它與函式庫版本在本節例子上算出相同結果。用於 `TinyLM`、`Block`、`CausalAttention` 這類帶有 cache、segments、RoPE 等後章參數的類別。
- 只說明並連到原始碼：用於資料產生器、報告輔助與大型成品程式，展示原始碼不會讓本節的問題更清楚時。

不採用的做法與理由：建置時 include 需要改凍結的 `scripts/build_course.py`，GitHub 上的 Markdown 也會只剩一行指令；只供閱讀、不執行的程式框（例如加 `title` 屬性）在 Notebook 裡不能改了再跑，`docs/review-tools/section_facts.py` 也會把語言為 `python` 的 fence 當成要執行的程式。

**D2 正文程式與函式庫不漂移。** 兩層守衛：`tests/test_course_code_excerpts.py` 逐字比對原樣節錄、檢查精簡版有對應的 import 與 `assert`；精簡版的 `assert` 在每次執行 Notebook 時核對行為（發布流程逐份執行全部 Notebook）。本輪不修改 `tiny_perceptron/*.py`：技術審閱紀錄以 `repository_code` 來源綁定這些檔案的 SHA-256，改動會讓既有證據失效。

**D3 不把輸出寫進 Markdown。** Notebook 與正文須一致（`scripts/build_course.py --check`），網站已顯示實際輸出；正文只引用 1–3 個值得觀察的數字，並以實際執行核對。

**D4 Colab 的位置。** 頁尾保留 Colab 與 Notebook 下載兩個入口（`scripts/check_site.py` 要求每節都有），改成同級的一般按鈕，文字說明「程式與輸出都在上方，不開 Colab 也能讀完」；沒有程式的小節改寫成「本節沒有程式」。README、閱讀指南與 W.1 同步改口：先當書讀，想自己改再在本機或 Colab 開 Notebook。

**D5 37 節無程式小節。** 只在能用 20 行內、離線 CPU 程式呈現本節機制，且不會把成品或實測結果簡化成錯誤印象時補程式；操作命令、需要下載權重或資料包、需要實測的小節維持文字，頁尾照 D4 說明。

**D6 全書用語。** 依[TERMS.md](TERMS.md)統一；一個概念一個名稱，第一次出現時附英文與常見別名，名詞頁照它補齊。

**D7 不新增或拆分小節；小節標題只為用語統一而改。** 新增小節要改 `course/lesson-contract.json`、凍結的 `zensical.toml` 導覽與順序檢查。標題含有 TERMS.md 要統一的詞時（清單見 TERMS.md「小節標題」段），作者只改用詞，主 agent 同步更新 `zensical.toml` 導覽。一節塞太多概念時，改為把次要內容移到補充區或縮短。

**D8 本輪不建立正式審閱輪紀錄、不切換 `docs/course-reviews-active.json`、不重估閱讀時間、不發布。** 這些是發布承諾，前幾輪由作者指定審閱方式。改過正文後，三道審閱檢查會報漂移，屬預期；本輪改用下面的獨立 subagent 審閱確認內容正確，結果與待辦寫在交接文件。

## 階段

| 階段 | 範圍 | 實作者 | 驗收 |
| --- | --- | --- | --- |
| P1 基礎 | 節錄同步測試、學生頁 commit 釘選測試、頁尾 Colab 定位、20.2 與 `docs/natural-assistant/v4/STUDENT.md` 的 commit、本目錄文件 | sonnet | 新測試與突變驗證、全套 pytest |
| P2 主線（一） | 第 1–7 章、`0C.md`（C.1–C.7） | 每章一位 opus | 共同驗收 |
| P3 主線（二） | 第 8–13 章 | 每章一位 opus | 共同驗收 |
| P4 架構、壓縮、成品與延伸 | 第 14–20 章、`0A.md`、`0B.md` | 每章一位 opus | 共同驗收 |
| P5 入口頁 | `course/first-steps.md`、`course/README.md`、`README.md`、`README_en.md`、`course/training.md` | opus | 共同驗收 |
| P6 名詞頁 | `course/glossary.md` 依 TERMS.md 與各章定稿補齊 | sonnet | 共同驗收 |
| P7 收尾 | 重建 Notebook、全書執行、網站預覽建置、交接文件 | 主 agent | 全部指令 |

同一章檔只有一位作者；不同章可以平行。作者不 commit、不執行 `scripts/build_course.py`；主 agent 統一重建、驗收，並按章 commit。每階段 commit 後由未參與撰寫的 subagent 審閱：技術審閱者執行程式、核對節錄與數字；初學讀者以「沒學過機器學習的大學生」身分讀改過的節，回報看不懂的程式與說明。主 agent 逐條裁決，修正後再審。

**共同驗收**（`$V` 是含 notebook、dev、site 群組的完整環境）：

```bash
$V/python scripts/format_course_code.py course/chapters/<改過的章>.md
$V/python scripts/build_course.py && $V/python scripts/build_course.py --check
$V/python scripts/check_lesson_order.py
$V/ruff check . && $V/ruff format --check .
$V/python scripts/check_notebooks.py --mode python
$V/python -m pytest -q
```

修訂前基準（`b05ebda3`）：全套 pytest 1427 passed、3 skipped、1 failed（`tests/test_selftrained_data_publish.py` 的一項，修訂前就失敗）；`check_notebooks.py --mode python` 312 份全數通過。

## 試讀問題的處理

19 份報告的 656 條意見依章節分給各章作者；主 agent 複核過的 27 條具體錯誤必須修正，其餘由作者逐條判斷採用、部分採用或不採用並寫理由，主 agent 審核。10 項跨章主題的處理：

| 主題 | 本輪做法 |
| --- | --- |
| 同一件事多個名字、同名不同義 | 依 TERMS.md 全書統一 |
| 先用後教、提前引用 | 改順序，或在提前處加一句「第 N 章才講，這裡只需要知道…」 |
| 理解正文需要的資訊只在補充區 | 必要的定義、分母、數字出處搬回正文；補充區只放出處、重做與延伸 |
| 寫「見 X」但 X 沒有 | 逐一改正連結或刪去不成立的承諾 |
| Python 寫法超出暖身 | W.2 補「本課會用到的 Python 寫法」，首次出現處加一句白話 |
| 數字少一步、練習有歧義 | 結果數字附算式；練習寫清改哪段程式的哪一行、預期什麼 |
| 維護者文字與原始資料混進學生頁 | 刪去修訂史與給維護者的句子；JSON 連結旁說明看哪一欄 |
| 一節塞太多新概念 | 不拆節（D7），次要內容移到補充區或縮短 |
| 照著做會卡住的操作環境 | 修 20.2 commit、W.1／T.1 補 Git LFS、操作頁寫明在哪裡跑與時間是否量測過 |
| 名詞頁查不到、入口太少 | 名詞頁補詞，章節導讀連到名詞頁 |

圖：手寫 SVG 內容與數字矛盾時直接改 SVG 文字；由凍結腳本產生的圖不重畫，改圖說或正文。

## 不在本輪範圍

- 修改 `tiny_perceptron/*.py`、重新訓練或重跑 GPU 實驗；正文引用的訓練數字只能來自 repo 既有紀錄。
- 新增、拆分、重新編號小節；用語統一以外的小節標題修改。
- 正式審閱輪紀錄、閱讀時間重估、`docs/course-reviews-active.json` 切換與網站發布。
