# curriculum 獨立銜接與最終依賴核對

你是 fresh continuity reviewer，未參與作者/reader/technical。讀 AGENTS.md、editorial-guide.md、clear-tutorial銜接規約；先不讀作者 PLAN、本輪其他結論或舊問題清單，完整讀目前 curriculum（docs/curriculum.md，freeze-03.json），從入口讀者核對地圖、前後依賴、數量與章節/指南路線，以及導覽承諾與先備負擔。明記實際讀的必要上下文，不冒稱全文逐頁閱讀。保存自己的 continuity-original-01.json，再做交叉與依賴核對、另存current，原始不可改。

最終 strict 網站 http://127.0.0.1:8000/，outputs/tokenizer-20261027/site。實看桌面1280×800和手機390×844截圖，實點 curriculum↔教材入口/readme/章節連結，核對表格/重要圖文/閱讀時間，不能只用DOM或XML取代實看。若有圖需view_image；PNG存Git外 /workspace/work/tokenizer-dynamic-20261027/continuity-curriculum-callback/，只留SHA與重現方式。

初判後再核對本輪真實 primary、readme callback、curriculum callback 的 reader/technical/continuity證據、freeze版本與來源；形成 dependency-audit-01.json。機器掃描列全351 inventory_page_ids，與上一active freeze及03比較，區分改／新增5頁與其餘346未變頁；閱讀只涵蓋實際受影響路線，不宣稱全書首讀。確認前兩波留下的 curriculum 待回查邊界是否已被本輪實際三輪收束。原始校對和你在初判後才看到的證據分開，不改稱獨立重現。若仍有未解必要問題照實revise，不為最終gate補pass。新工具完整獨立審閱與9筆歷史回查在第二波；你可核對這些版本/身分/實際收件，不重新審工具語意，也不把metadata當閱讀證據。

報告位於 docs/course-review-20261027-tokenizer/reviews/continuity-curriculum-callback/，每頁含page_id、source_sha256、figures_sha256、verdict、issues與實際依賴/版面/路徑檢查。audit含reviewer_task、created_at、verdict、inventory_page_ids與明確範圍。不得改受審內容、工具或舊報告，不開代理、不提交、不發布。最後回報判定、問題、路徑與限制。

版面範圍以來源特徵抽樣：大綱首段計數/入口、長表、必要流程圖全段、章節路線及末端返回；桌面和手機都實看，無需為每一段相同純文字另截一張。正文仍完整閱讀，不能把抽樣版面寫成每一像素檢驗。最終第三波還有一名fresh README technical，原因是最終估時表整檔hash變動；其本文／圖未變，前次reader/continuity仍同源。最終sidecar實際為README五筆、curriculum一筆、工具四筆，共十個歷史身分；核對最新主proof與原始收件，不改寫第二波舊hash或把其過期source當當前技術證據。
