# 新版教材研究的無付費前置檢查

使用者已於2026-10-29批准v3主線／第19章新版模型修訂、額外Modal gross上限US$60及檢查後發布。此批只建立必要基礎設施，沒有修改學生教材、啟動訓練或宣稱新版能力。

新workflow只有三個操作：`inventory`以現有Modal client讀workspace/main的帳務、舊ledger/quota、App與Volume狀態；`transport-probe`只在Actions使用標準GitHub token，驗微型binary權重＋momentum/RNG包的draft Release上傳／下載／hash／接續運算並清理；`initialize`建立獨立GitHub CAS帳本，但必須先有真實inventory及獨立審閱、已提交的reconciliation。

本批沒有Modal容器、HF寫入或新Volume建立功能。正式GPU runner必須等教材易讀易理解／正確性／連貫性三輪及具體job費用上界核對後，另行實作與審查。既有HF權重仍固定revision；新模型與完整續訓包採已批准的Release資產通道，HF新鏡像可選。學生最終公開匿名下載尚待實際產品發布後驗證，不把draft認證傳輸當匿名發布。

兩道新round守門是不可遞減的新預約合計<=60，以及官方新gross增量＋未釋放新bounds＋舊未釋放carry＋proposed<=60；舊歷史audit總額不混成carry。每次CAS先完成才可進未來付費邊界；unknown/live未釐清停止，不修改舊policy/ledger，也不自動關閉其他工作。這是保守派工規則，不聲稱供應商帳單已最終結算。

獨立技術初判、修後複查與實際反例見[infrastructure-review/recheck-01.md](infrastructure-review/recheck-01.md)。原始證據保留實跑bytes，局部Ruff設定只排除封存驗證程式，產品scripts與tests照常檢查。本機Release草稿曾建立後因uploads路徑401中止，已刪自有draft；安全收據見[github-transport-preflight.json](github-transport-preflight.json)。Actions通道尚待本批workflow上main後實測，不預填成功。
