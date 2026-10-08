# 教材修補發布紀錄（2026-10-08）

本次發布採用最新版 clear-tutorial 準則下的全書診斷與實際修後複查，保留原始審閱、分級差異、封存 bytes 與已揭露的程序限制。教材的局部修補與實驗紀錄勘誤見[前一階段紀錄](../course-repair-20261008/README.md)；該紀錄中的「尚未發布」描述的是當時狀態，沒有改寫為當時已驗收。

目前325個 canonical 單位中，288個使用同準則、同來源的原全書審閱；28個使用受影響單位的修後獨立審閱；9個使用後續完整回讀。這是可追溯的組合驗證，沒有聲稱另做一次325頁盲讀。知情回查中的技術與銜接原報告均保留，並另外保存[真正技術封存後的銜接回查](../course-repair-20261008/reviews/callback-02/reports/continuity-all-post-technical-callback.json)。

新政策 `tutorial_composite_v1` 由[manifest](review-manifest.json)、[逐頁 catalog](review-catalog.json)、[目前來源凍結](current-source-freeze.json)、[原路徑索引](archive-index.json)及[當下的協調者歷史聲明](coordinator-history-attestation.json)組成。原始全書資料存於 `audit-original/`；凍結來源、準則、Notebook 與圖檔以 SHA 綁定。原論文使用固定來源、原檔 SHA 與本地快取實際核對的 receipt，不把第三方全文重新發布，也不以 receipt 取代教材、程式、圖、逐段筆記或執行證據。歷史角色與編排依現存報告、trace 和封存紀錄聲明，沒有重建不存在的 spawn／FINAL API receipt。

發布檢查會拒絕來源或準則漂移、證據缺失或竄改、作者兼任審閱者、同一審閱者跨職複用、未處理必要問題，以及缺少真正技術封存後的最後銜接回查。舊 `phase7_grouped` 政策及其拒絕原因仍保留；政策切換不會把舊結果改成通過。程式完整性檢查不能證明讀者理解、親自看圖、物理隔離或科學結論。

修後技術審閱所用的128個實作檔及147個實驗輸入另與目前檔案逐一核對。`pyproject.toml` 的唯一允許差異是四個明列的 Ruff 原始證據目錄排除；解析後的其他設定與原排除順序必須完全一致，沒有改寫舊封存 SHA。

本次另以獨立 CPU kernel 實際執行全部287份 Notebook：287通過、0失敗，詳見[執行結果](checks/notebook-kernels.json)。發布使用既有 Pages 工作流程重新執行、嚴格建站及核對全站連結。公開版本以[網站 build-info](https://birdhackor.github.io/tiny-perceptron-vlm/build-info.json)的 Git revision 與對應 GitHub Actions 結果為準。
