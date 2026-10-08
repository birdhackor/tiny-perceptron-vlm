# 凍結教材的來源獨立覆核

這是診斷教材要修多少、是否需要大改的審閱，不是修稿或發布驗收。使用者明確要求 subagent。你是新 reviewer，不參與作者工作；不要修改教材、skill、舊報告或其他人的記錄。批次根目錄為 `/workspace/work/tutorial-audit-20261008`，repo為 `/workspace/tiny-perceptron-vlm`。

先閱讀凍結的 `freeze/criteria/SKILL.md`、`references/review-protocol.md`、`references/calibration.md`。正文來源在 `freeze/sources/<page_id>.md`；manifest.json 的指定 group.pages 給出實際路線、來源及圖的指紋。全部指定頁都必須真正閱讀，不能用標題、搜尋命中、抽樣或套用全章通過模板代替。

你所在階段及 group 由派工訊息指定。只讀自己的任務來源及需要的前文；可以從 manifest 找前文ID，記實際讀了哪些。初判期間不可查看 `reports/`、`synthesis/`、`notes/`、`traces/`、checks中的人類判斷、其他reviewer結論、repo歷史reader/technical/continuity/validation-review紀錄或舊問題清單。不要把專案技術來源當成以往審閱結論；需要外部查證時用論文或官方文件，保存原文及實際讀取位置。

圖在 `freeze/original/course/figures/`，已渲染640/360的PNG在 `renders/`。補充資產fingerprints在 `checks/supplemental-visual-assets.json`，包含2.5折疊區連結式曲線圖。看圖先以view_image真正取得畫面，下一回合才記錄看圖結果，不把預寫描述算看後判斷。完整來源與實作在第二、三輪可以讀，這不是無提示逐段首讀，請不要冒稱這個角色做了該工作。

每頁保留自己的來源判斷、至少一段足以支持主要通過/缺口的具體原句、需要的依據及未驗範圍。每個新增必要方法/設計選擇分開核機制与需要/用途；不以概括好處代替缺少關係，也不自訂原文未承諾的新教學目標。正常推论有來源就保留最短推論；導覽、預告、操作例、純性質段與實測證據按各自範圍審，不要求全面演算法、最佳超參數或所有變體。

## 技術輪

查任務/答案/表示、計算/更新/評測邊界、介面狀態及數字。凍結實作/設定在 `freeze/implementation/`，fingerprints在 `checks/implementation-source-manifest.json`。小算例可直接核算，存在疑點才執行相稱小檢查。既有實測可以讀正文直接連結的原始結果JSON、資料manifest與執行紀錄，逐项記來源、版本、分母與比較設定；無法核实標未驗證，不自動判錯。不要重跑長訓練。不要用技能更新造成舊發布gate的hash失效當教材缺陷。

## 銜接輪

自行從正文建立實際依賴，涵蓋必要設計、規則、轉換與限定，而非只列術語。每頁記現在要能理解/判斷什麼、此前哪一段教會必要關係、此頁如何承接及下一步何時用。先檢只讀正文路線（details內不算正文），再補選讀，核對真正必要的說明是否錯放；連結存在不代表讀者已懂。主要通過項与待教/可選/未知也核，不只找錯。對初讀組別跳讀未讀前文與教材真的缺前文分開處理。

## 封存

保留實際逐頁工作筆記/檢查證據。完成全組後，輸出 `reports/<role>-<group>-initial.json`，欄位包含reviewer、role、group、criteria_sha256、actual_prerequisites、pages（每頁source_sha256、source_judgment、quoted_basis、checks、unverified）、issues、coverage、visual_scope、execution_scope。每項問題包含原句、已教部分、缺少的最小關係、當下影響、severity（blocker/burden/optional/unverified）、最小修補與rewrite_scale（paragraph/page/chapter/order）、發現時點與未看其他結論的來源。

真正完成才生成全頁coverage。用SHA-256封存initial報告至自己獨立的seal JSON，並向root回報。之後等待交叉覆核派工；不得自行開舊報告找漏項、改初判或宣稱前後獨立重現。這些是AI輔助審閱，不能說是真人學生測試。
