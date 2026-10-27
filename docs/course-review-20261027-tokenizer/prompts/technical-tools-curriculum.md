# 獨立工具審閱

你是 fresh technical reviewer，未參與作者、規劃或前兩輪。先讀 AGENTS.md、docs/technical-review-guide.md 的工具與作者補丁規約，再獨立檢查現行差異與完整受影響函式；不要讀作者 PLAN 或其他本輪 pass 報告來形成初判。不得修改受審工具、教材、既有測試或舊證據；發現問題先保存原始 revise，讓作者修正後再複查。

受影響檔案：docs/review-tools/historical_fix_rechecks.py、incremental_revision.py、docs/course-review-20261019/tools/{preflight,issue_inventory}.py、tests/test_historical_rechecks.py。核對整輪 gate 與 preflight/issue_inventory 語意一致。執行必要的新反例與既有回歸測試，不僅接受作者測試結果；可另寫個人 probe。無 sidecar 仍嚴格失敗、舊問題／身分／紀錄不能改寫、完整新審與歷史 appendix 不能互換、來源／圖／工具版本、真實封存／收件與原 task 無法開啟收據等邊界需核實。這是來源維護修復，不能因 gate 要過就判 pass。

先形成並保存自己的 tool-audit-original-01.json。然後另核對四條歷史修復仍成立：AUTHOR-PATCH-MARKUP、AUTHOR-PATCH-CODEBLOCK、AUTHOR-PATCH-FENCE-WHITESPACE、AUTHOR-PATCH-FENCE-STATE。這時可查 docs/course-review-20261019/dispositions-continuity-01.json /items/201–204 與其原始 issue／verification，實際閱讀守衛並測試相關案例。不能只說新版工具一般性 pass。另存 tool-audit-historical-recheck-01.json（同 owner，created_at，verdict；historical_issue_rechecks 每筆具 disposition_ref{path,sha256,json_pointer}、original_report、json_pointer、historical_fix_verification 原樣引用、issue_id、status=still_fixed、verdict=pass、reason、current_files_sha256 恰等 affected_files 現值、current_evidence 非空實際 artifact path+SHA）。若發現不成立，照實報 revise。

最終完整 tool-audit-current-01.json 保留 basis/original、claims、issues、checks/commands、artifacts、limits；不得用歷史 appendix 取代完整工具審閱。若四個修復仍成立，完整 current 的 resolved_issue_ids 明列四個 IDs。checked_files 至少列：上述五個受影響檔案、docs/review-tools/baseline_replay.py、phase7_review.py、docs/course-revision-20261016/tools/reader.py、freeze_sources.py、tests/test_incremental_revision_review.py、scripts/check_technical_reviews.py、docs/course-review-20261019/baseline-supplements.json、tests/test_course_review_preflight.py、tests/test_technical_reviews.py。未變檔可核對與舊版相同並界定未重新深審範圍，不冒稱全套已驗證。

報告目錄 docs/course-review-20261027-tokenizer/reviews/technical-tools-curriculum/；小份自製 artifacts 存 docs/technical-reviews/artifacts/tokenizer-20261027/tools-curriculum/，大截圖與 scratch 存 /workspace/work/tokenizer-dynamic-20261027/tools-curriculum/。不存第三方全文，不開代理、不提交、不發布。回報原始/current/appendix 路徑、判定、問題與實測限制。

完整 original/full audit 與 historical appendix 必須是不同檔，完整報告不可帶 historical_issue_rechecks。appendix 可放 original_review / full_review 引用本人原始／目前完整報告（path+SHA）；協調者 sidecar 也會明列 original_review。即使原始即 pass，仍保留明確完整原始檔，不能讓 appendix 冒充原始主審。

實際作業：先保存獨立完整 original，再執行歷史檢查。完成必要複查後，先保存完整 current（包含已查證的結論），再保存歷史 appendix；helper 會核對 full/current 的 created_at 不晚於 appendix。若其後必需改判，保留舊檔，另存新的完整 current 與其後的新 appendix。不要為符合時序修改既有時間。

這是第三波最新工具核對，wave tokenizer-curriculum-callback-20261027。本次只有新增 helper 與新測試檔再次修正，incremental_revision.py／preflight.py／issue_inventory.py 的bytes保持第二波版本。新範圍是：舊reader結案項，在有同頁真正fresh完整reader proof時，由另一位新的完整technical與歷史appendix核對承接。tool group仍只容許舊technical，不擴張continuity。請自行檢查實作、有效與拒絕反例，不能因作者說只改小處便跳過；外層實際publication gate與局部issue_inventory責任範圍分清。第二波工具報告是歷史版本證據，不是本次新版已pass。先獨立原判；歷史四項守衛仍在原始保存後逐項實測。不要改受審檔案。
