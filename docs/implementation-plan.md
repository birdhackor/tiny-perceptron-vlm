# 教材實作與驗收

目標：完成大綱 v0.8 的 18 章、A/B/C 支線與全部 222 個小節。讀者只需大學程度的基礎數學；Python、tensor、kernel 與訓練流程從操作指南補起。正式訓練留給讀者的 GPU 環境。

## 內容與程式的單一來源

`course/chapters/` 是繁體中文正文與可執行 code fences，`scripts/build_course.py` 由它產生每節一份 Notebook。每節包含現象、預測、直覺／必要的數學、可執行程式、結果解讀與單一變數練習。完整模型留在 `tiny_perceptron/`，Notebook 以明確連結指出當節使用哪一小段。

## 驗收條件

- 每個大綱編號都有正文與 Notebook，不能靠泛用模板只重述標題。
- 前段直接展示小矩陣與公式；先學手寫因果 attention，現代架構與 GPU 加速放到後段。
- 依使用者補充，多用譬喻，加入自行製作的SVG與逐步動畫；動態圖具靜態標註，讀者可以停下來讀懂流程，術語在需要時纔出現。
- 所有預設 Notebook 離線執行，資料可重現，無權重下載或正式訓練。
- 需要已訓練能力的比較標明前置權重、可執行 GPU 入口、實際要記錄的分項指標；不寫虛構結果。
- 核心測試檢查 causal／padding／shift、cache／SDPA 等價、梯度、checkpoint 接續、量化 packing、DPO 與 KL 方向。
- 訓練／推論 CLI 與 Notebook 執行成功；新增依賴鎖檔、lint、格式、既有測試都完成。

## 進度

- [x] 確認既有大綱與資料資產，保留使用者未提交文件。
- [x] 文字、對話、架構、模態、對齊與壓縮基本件。
- [x] 222 節正文、短 Notebook 與入門指南。
- [x] 訓練、評估、模態生成、資產準備與比較入口。
- [x] 完整執行、內容審閱、README 與環境啟動指引同步。

讀者入口在 [course/README.md](../course/README.md)，實作與未執行的能力量測範圍見[訓練操作](../course/training.md)及[驗證紀錄](validation.md)。
