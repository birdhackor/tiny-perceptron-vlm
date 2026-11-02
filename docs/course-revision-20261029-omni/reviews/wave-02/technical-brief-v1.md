# 本輪單頁正確性獨立審閱契約

僅做所派 canonical 頁的獨立技術與證據審閱，不改正文、圖、Notebook、實作、checker、其他報告或 active pointer，不 commit，不啟動訓練或付費工作。每頁使用自己的真 fresh task。按 `docs/technical-review-guide.md` 與 `.agents/skills/clear-tutorial` 的通用技術準則，先讀自己的凍結正文、必要前文、實際內嵌程式及其明示依賴；不先讀作者計畫／理由、讀者答案、舊或其他 technical／continuity 結論。

先核 source freeze 的當頁 snapshot、source_sha256 與全部 figure_sha256，再逐項查正文真正提出／使用的概念、數值、軟體與實證主張。機制示意、參數更新、正式成果與泛化分開判；軟體操作與無圖頁也不能整頁假 NA。理解必需的程式、輸入、關鍵輸出與理由應能直接在正文定位，不能靠開 Colab 補當下理解。只為具體疑點做相稱的離線 CPU 驗證，不因文字問題重跑全部 Notebook 或訓練。

已有圖須真正 render 並用圖片工具看；核數字、箭頭、對象、軸與正文的對應。未驗視覺／未實跑的範圍明記。必要程式執行若做了，保留精確 command、環境、raw stdout／結果及 SHA；大型圖與執行原始輸出留 repo 外，小型自製報告與 source-specific 收據可放本人目錄。第三方全文仍留 Git 外，報告只保存來源 URL／版本／讀取日／SHA 與支持主張的小段引用。

一次新寫自己的 `technical-original-v1.json`，真 UTC created_at。既有 schema 必有：整數 `schema_version=1`、`review_stage=technical`、自己的 `lesson_id/source/reviewer_task/reviewer_context=fresh/source_sha256/figure_sha256/verdict`；`claims` 區分 concept、numeric、software、empirical，逐項具 location/status/evidence/scope；`sources` 與 `artifacts` 真實可驗。checks 依既有格式包含 factual_accuracy、numeric_verification、figure_consistency、source_verification、limitations，指向實際 claim IDs。原始 unresolved／revise 也保存，不為過 checker 改成 pass；全部 necessary／optional 問題留穩定 ID、原句、位置、缺少最小關係、已教部分、理解影響、最小修正與分級理由。

使用既有自查，reader-task 參數須用真實派工提供的名稱：

```text
uv run --extra cpu python docs/course-review-20261019/tools/check_technical_report.py ACTUAL_PAGE ACTUAL_REPORT --freeze ACTUAL_FREEZE --reader-task ACTUAL_READER_TASK
```

先保存自己的不可變原判並回報 path／SHA／實質問題，之後才接收其他角色答案／歷史收件接合。原判不能事後覆寫，修正或追加查證另存 current。對 whole-file first-steps，先完整自主核 W.1–W.7 與前言；自己的來源原判保存並收件後，才另核協調者提供的編號 sidecar 與真原判／current／收件資料，以免把彙整當成獨立發現。
