# README 獨立技術回查

先讀 technical-common.md 與現行技術規約，完整審 readme canonical page；freeze=freeze-03.json，reader task=/root/reader_readme_callback_20261027。這是完整新審，不以舊審代替閱讀。不要讀作者 PLAN、其他本輪報告或歷史問題來形成初判。已驗證的既有原始實驗資料可讀，不必重訓。現行網站 http://127.0.0.1:8000/ 可核對；圖需實際查看。重要教學內容不能依賴 Colab。不得改教材或工具。

先保存自己的 technical-original-01.json，再做以下歷史修復回查。原 task 已無法重新開啟（協調者有實際工具收據）；這裡的工作是新 reviewer 的獨立檢查，不宣稱原 reviewer。

查 docs/course-review-20261019/dispositions-continuity-01.json /items/196–200 與各自原始 issue／verification，親自核對五項修復在目前 README／程式／網站仍成立，保存 technical-historical-recheck-01.json。這份 appendix 需符合完整 technical schema（來源、claims、artifacts、checks 等），但明示只覆蓋歷史修復，不能取代完整頁面審閱。加 historical_issue_rechecks：每筆具 disposition_ref{path,sha256,json_pointer}、original_report（舊問題 report ref）、json_pointer（原問題 pointer）、historical_fix_verification（舊 verification 原樣）、issue_id、status=still_fixed、verdict=pass、reason、current_source_sha256、current_figure_sha256、current_evidence（source_id、locator、supports）、artifact_ids，指向本人真實新證據；若不成立照實報 revise。

完整 current 另存 technical-current-01.json，basis/original 指本人原始 full-page review，不用 appendix 作 full proof。兩份都須本人 owner、fresh technical、created_at、source/figure 與目前 freeze 相符。受審程式已定稿後才派工；核對所引用 repository_code 的現行 SHA。資料與模型主張的限制要明列，不把舊雲端訓練說成本次重跑。

報告存 reviews/technical-readme-final/（本輪資料夾下）；小 artifacts 存 docs/technical-reviews/artifacts/tokenizer-20261027/readme-final/；scratch/截圖存 /workspace/work/tokenizer-dynamic-20261027/readme-final/。圖與第三方全文不重複寫進 Git。禁止開代理／提交／發布。

實際作業：先保存獨立完整 original，再執行歷史檢查。完成必要複查後，先保存完整 current（包含已查證的結論），再保存歷史 appendix；helper 會核對 full/current 的 created_at 不晚於 appendix。若其後必需改判，保留舊檔，另存新的完整 current 與其後的新 appendix。不要為符合時序修改既有時間。

本次是第三波最後技術來源回查，wave tokenizer-curriculum-callback-20261027。README本文／圖未再變；整份reading-times.json回填最後curriculum實讀估時後，舊來源hash過期，因此要真正核對目前完整頁及當前來源。reader task仍為已完整讀相同版本README的 /root/reader_readme_callback_20261027。先獨立original，歷史196–200回查在original後，完整current與本人appendix按上述時序另存。這次全部351頁估時已穩定，網站strict重建已完成；不要因舊來源過期而僅更新hash，也不需要再審整本書。禁止改受審內容。
