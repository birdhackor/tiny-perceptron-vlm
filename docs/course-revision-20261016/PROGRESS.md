# 執行進度

本檔記錄實際完成與未完成項目，不作審閱通過證據。

- 已建立PLAN與三成品能力／切分契約；新增25節草稿。
- 13段新增Python機制片段已於CPU實際執行，全部exit 0；沒有正式新模型訓練。
- 已接回第7、9、14、19章與入口／名詞／操作／課綱。C.1–C.5移到第7章後的實際導航，C.6–C.7在第13章後。14章共享後先讀循環深度，19章先總覽再三條路線。
- 共312份逐課Notebook同步檢查通過；350頁預覽Zensical strict建置成功，尚未附本輪完整kernel輸出或新AI估時。
- 已保留開改前reader／technical／all三道實際gate結果。舊manifest不改；baseline_replay在暫存副本復原舊版本並原樣執行，實際通過。
- freeze-01共44個來源變動頁；來源不變頁保留原證據引用。6名fresh context獨立讀者已依helper逐段保存當下四題判斷，完成全組正文後才補選讀；51頁當前讀者判斷及必要／可選處置均已保存，reader增量gate通過。
- 51個受影響編號節／獨立頁的技術審閱已在首讀修正後開始；銜接階段尚未開始。新review gate工具尚待獨立核對，未切換生效pointer。
- 正式產品工程、GPU訓練、公開權重、Auto留出評估、完整網站發布均尚未開始。

- 首次reader階段於reader-seal-01保存結案；新工具獨立反例審核發現完整性缺口後，已由原reader另存44個非「已足夠」concept的實際收束，早期紀錄不改。後補屬technical階段格式補證，不倒填時間。
- 25節新增頁的實讀AI閱讀估計合計147–227分鐘；不含安裝、動手、下載或訓練。350頁目前source/figure估時核對通過（executed首頁版本）。
- 工具original與callback01保留revise；修正版manifest-reader-04的350頁reader metadata核對及117個相關測試實際通過。工具callback02獨立複查進行中；不表示完整technical/all已通過。

- 04:54 UTC 狀態補記：工具獨立callback03已pass，51個反例／正例探針與117個相關測試通過；先前revise原件保留。全套離線pytest另有1323 passed、1 skipped（103.66秒）的作者實跑收據。
- 312份Notebook的首次bulk為311通過、8.12於kernel啟動前失敗；8.12單獨重跑通過，312份目前輸出均逐份核對原始碼與執行結果。350頁executed strict預覽建置成功。這些不是正式B/C成品訓練。
- 英文首頁「three optional branches」已同步成A/B延伸與必修C，原英文全文及aux finding保留，原審閱者知情callback核實新版SHA與同步關係。技術審閱尚餘首頁任務收尾，銜接尚未開始。

- 05:36 UTC 狀態補記：51個編號節／頁的技術審閱與350頁technical metadata gate已實際通過；六組獨立銜接審閱已完成原始判斷及交叉核對。19.17的C→B必要契約缺口已修正，原讀者、原技術與原銜接審閱者另存新版本複查。
- freeze-09較08有12頁bytes變動：11頁Python排版AST不變，19.17新增兩段契約。12份受影響Notebook於各自kernel重跑通過；目前312份執行輸出逐份核對source、cell與execution，350頁含當前AI估時的strict預覽通過。19.17閱讀估時由原讀者依新文更新；此前147–227分鐘是較早版本。
- 原CDN公式資產在本環境實際503，已保存失敗。改用同一MathJax 3.2.2固定commit、授權及94份hash核對資產隨站發布；第三方bytes不進Git。原公式問題發現者及作者均於封鎖站外請求後親看桌機／手機公式正常；不是宣稱全站所有功能離線。
- 資產工具獨立original發現stage目的symlink可寫出目錄，保留revise；作者已修正並新增外部sentinel反例。stage/export/reading-time三檔51項測試通過，完整ruff check與format通過；原工具審閱者正在知情回查。第一輪all gate尚未宣稱通過，未切換active pointer。
- 本次新資產整合後全pytest曾有1 failed、1332 passed、1 skipped，失敗為估時單元測試的迷你網站fixture未隔離資產依賴；已補測試隔離並重跑全套中。保留失敗收據，不沿用先前全套pass替代新版本驗證。
- 正式B/C工程、GPU新訓練與公開新權重仍未開始；下一步是第一輪所有審閱與工具回查實際結案後進入階段5。

- 05:54 UTC 第一輪結案：all-gate-04實際核對350頁通過，啟用tutorial_incremental_v2；舊pointer、失敗all候選01–03與所有早期判斷保持。補充證據按真實continuity收件階段簽封，未倒填technical seal。
- 新core工具callback04獨立親跑120項測試、16個新收件案例、28個回歸案例及4個各350頁正向control通過；原callback03不變。舊pyproject快照補入baseline replay後，獨立完整325頁重播通過。
- 最新全套作者pytest為1340 passed、1 skipped（90.93秒）；完整ruff check及format再次通過。新25節當前閱讀估時由350頁metadata實算為149–229分鐘，不含操作與訓練。
- 現在開始階段5產品工程；會採納Astra補充諮詢的去ID觀測消重／標籤一致性、狀態追蹤題型分層、算術答案覆蓋、C有效三級標籤與Auto全考卷分母。尚無正式B/C新訓練或公開新權重結果。
