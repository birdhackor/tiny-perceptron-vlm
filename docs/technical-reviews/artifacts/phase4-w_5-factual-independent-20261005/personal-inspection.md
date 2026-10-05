# W.5 本人獨立查證記錄

Reviewer: /root/phase4_factual_coordinator/factual_w_5
Date: 2026-10-05
Verdict: pass

派遣範圍是 course/first-steps.md#W.5。本人先以 rg --files 定位原稿及指定方法，再親讀 factual-reviewer-instructions、checker schema、section_facts.py、clear-tutorial/SKILL.md 與 references/review-protocol.md。第一次 rg --files 只顯示檔名，未讀任何舊 technical/reader/history/review-dispatch 內容。此次沒有讀到舊判定或作者額外修正摘要。

實際教材閱讀範圍：W.5 原始全文；course/first-steps.md 原檔第 1–215 行（含導言、W.1–W.4 前文、W.5、W.6 及 W.7 開頭）。判定只涵蓋 W.5。保存的完整 Markdown 是當次 frozen input，SHA-256 af340866487d162b7a14204c92705371ecb9cc35b839f184e161ea900103feb3；這個指紋不宣稱是後續全章現況。正式 source_sha256 只涵蓋原始 W.5 bytes：7825d87e0d92fecdfa28e209935cfd08e6f2b25aebdb4b0addb80f02340c4cba。

locator cache 僅用來定位：original-source-locators.json 的必要 top-level key/type、locators 第 0 項 field/type，再篩 PyTorch 對應 URL 的 url/version_label/original_path/sha256 metadata；original-paper-locators.json 僅 top-level key/type，沒有讀 records。未把 metadata 的既有版本文字當作查證。本節為 API 與手工線性代數示範，無需原論文或原實測結果 JSON。

本人重新從 PyTorch 官方 GitHub immutable commit 5c4886908584029761b579af026dcfb627c84070 的 HTTPS raw URLs 取得四份原始檔，HTTP 200。先以 AST 定位，再親讀：torch/_torch_docs.py 第 7908–7990 行（matmul）；torch/nn/modules/linear.py 第 53–140 行（Linear docstring、shape、weight/bias allocation、forward）；torch/autograd/grad_mode.py 第 22–86 行（no_grad 行為、factory/forward-mode 範圍、enter/exit）；torch/_tensor_docs.py 第 1146–1171 行（copy_）、5445–5465 行（tolist）、6769–6783 行（T）。也讀相同位置的 installed Linear/no_grad。四份下載原檔均逐檔 hash 比對 installed .venv 原檔完全一致，結果保存在 bounded-stdout.json。官方 PyTorch repo 直接擁有 API 的實作與 docstring；非搜尋摘要或候選庫內容。

本人實際執行 section_facts.py 的 W.5 兩個原始 Python fence，依原順序接續執行；退出 0，無 guard event，CPU PyTorch 2.14.1+cpu，Python 3.13.5。原 stdout 親讀：tensor([[40.,1.],[80.,6.]])、torch.Size([2,2])、[[45.0,1.0],[85.0,6.0]]。保存原始 fence、bootstrap、stdout、stderr、environment、extraction 與 execution receipts；由 /tmp 複製到 permanent original-run 後逐檔 SHA 再核相同。

本人另執行 bounded_checks.py：以 Python 整數三項點積自行計算四個值，核對 @ 等於 matmul、2x3 @ 3x2 的 2x2 形狀、權重 .T 為 2x3；只改第一杯糖量，第二杯結果保持 80/6；右矩陣減為兩列，實際 RuntimeError 文字為 mat1 and mat2 shapes cannot be multiplied (2x3 and 2x2)。核對 Linear 的 weight/bias shape、copy_ identity、no_grad 區塊內 grad-enabled=False、退出後 restored=True、parameters 仍 requires_grad=True，而 grad 均 None；forward 仍記錄梯度但並未 backward 或做訓練。核對 tolist 的巢狀 Python list/float。bias 5→8 的新結果 [[48,1],[88,6]]，差值 [[3,0],[3,0]]。

成本單位是分／材料單位乘材料用量而成的分；甜度權重明示是假想指標，沒有物理單位或模型成績。每個輸出是一列配方的三個材料乘積之和，不是平均，没有分母或百分比；矩陣包含 2 杯、3 材料、2 指標。算例中的整數與浮點結果都是 float32 可精確表示的小整數，使用精確相等而非四捨五入，容忍差 0。

圖核對為具體 NA：W.5 沒有 Markdown/HTML 圖片或 SVG 引用；兩個小矩陣的列欄、材料與指標對應已由數字和文字完整提供，不需想像圖中空間或箭頭。本次沒有圖可 render/view，沒有宣稱做視覺 render。

結論支持範圍：二維矩陣乘法與這個 Linear(3,2) 預設 bias 範例；.T 的列欄交換適用此二維矩陣；no_grad 的說明適用手動設定權重的 reverse-mode recording，不宣稱禁止 forward-mode AD 或整個後續訓練。這是手工示範，不是模型已學會的實測、最佳權重、泛化能力或工程成品驗收。未發現實質錯誤或未解疑問；未修改教材／圖、未訓練、未下載資料／模型、未 commit。
