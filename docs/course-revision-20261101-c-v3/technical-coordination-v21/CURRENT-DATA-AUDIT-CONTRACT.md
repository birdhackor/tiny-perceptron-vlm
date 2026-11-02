# 獨立 current-data supplement 的輸入契約

這是本輪行政資料審核的任務規格，並非正確性判定，也不是 canonical 教材頁或新的 tool primary。審核者必須是與作者、collector、reader、其他 technical primary 不同的真正 fresh actor，special lesson_id 為 `__current_data__`。

T 內先實讀原生資料契約，保存自己的完整來源初判與真收件。此時尚不存在的最終 C candidate 不在 T 原判的已驗範圍。真 C OPEN 之後，root 才交付具精確 path/SHA 的 ordinary candidate；同一位 auditor 對那份實際 bytes 另寫 exact-data supplement。auditor 不修改 candidate、DP、原始報告、收據、派工或 seal。

actual-data 審核的主要輸入是 ordinary candidate、其 source21/implementation/assignments/review-wave refs、全部新及 callback 報告與 collection receipts、author dispositions、numbered bundle/index、historical rechecks 及其原 references。必須實際 hash／parse，被引用的 path 與 SHA 對應真 bytes；不能只採用作者或 collector 說通過的摘要。

auditor 以現行 native 原始碼為契約來源，獨立核：

- source21 canonical 身分、source/figure/Notebook 凍結、implementation 與 baseline supplements 的實際 bytes。
- 任務唯一性、角色與作者／reader／technical／continuity 身分分離，以及 per-primary 真 lesson_id。bounded original-owner callbacks 保留原 assignment 身分，不冒稱 fresh 首讀。
- 真 R completed <= T started、T completed <= C started 與各 callback 的真階段；OPEN 階段不能造未來 completed seal。
- 新與 callback 報告的真 created_at、dispatch window、唯一收件及 collection_index/report_sha256s 對應；不得有未來時間、未收原判先寫 current、重收 current 再把時間改成早於 original 的證據。
- 每份 original 與 current 身分／來源／圖／native schema，以及原問題的同 owner 明確逐項 closure；保留的舊報告須核 source 與全部實際依賴 identity，不能只換 hash 或把舊結論當新閱讀。
- author dispositions 對實際問題 inventory 的完整、不重複覆蓋、必要性及 verification links；舊歷史原始 issues、prior pointers、unavailability、收據與作者必要性不被抹除。
- numbered 真 section originals/currents、retained entries、真收據、issue index、final bundle，以及四個 canonical collector current 對 final bundle 的 artifact SHA，避免互相引用循環。
- 所有新資料與 claims 的真實 scope：qualified 固定成品與 source-identical 研究的 retention、本輪新 executed probes、本機 ZIP 內容驗證與尚未完成的公開 transport 分別據實記錄。

這份獨立 exact-data supplement 補核本輪實際 ordinary candidate；它不取代保留的 core tool primary，也不宣稱 generic I01/I02 已修好。工具本身能查到的格式與身分聲明，不等於科學正確性、物理隔離、真人測試、實際圖像觀看、訓練品質或部署成功。新原始碼審閱與新測試範圍不得擴成無關的舊研究重證。

exact supplement 應綁 ordinary candidate 的實際 SHA，不要求 candidate 預先引用 auditor 自己的 report；root 之後整合 auditor reference 時，另核其差異。auditor 自己的真正 report receipt 與之後發生的 C close 由 collector/root 另驗，避免自審 collection 或 SHA 循環。若必要資料仍有未解矛盾，結論須維持 revise 並指出精確 path／JSON pointer／時序，不以 pending=0 或全過布林值代替證據。
