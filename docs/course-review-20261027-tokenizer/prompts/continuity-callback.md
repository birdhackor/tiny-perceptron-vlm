# README callback 獨立銜接審閱

你是 fresh continuity reviewer，未參與作者、reader 或 technical。讀 AGENTS.md、docs/editorial-guide.md 與 clear-tutorial 銜接規約。先不讀作者 PLAN 或本輪他人結論，完整閱讀目前 canonical readme（README.md；freeze-02.json），從入口讀者角度檢查操作路線、名詞回查、數字／章節入口是否一致。必要前後文可讀，但列實際路徑，不冒稱全文閱讀。

真實最終網站 http://127.0.0.1:8000/，輸出 outputs/tokenizer-20261027/site。桌面1280×800與手機390×844，實看截圖、實點 README 入口／暖身／教材首頁和返回路徑、核對重要圖、程式與內容可在本文閱讀，避免橫向溢出／被迫開 Colab。截圖在 /workspace/work/tokenizer-dynamic-20261027/continuity-readme-callback/，Git 僅存SHA與重現方式。不要只用 HTML/XML/DOM 代替實看。

先保存 continuity-original-01.json（schema1, reviewer_task實際自己, created_at, pages [{page_id:readme,source_sha256,figures_sha256,verdict,issues,concept/dependency/layout/link checks}]）；保留任何 revise。再保存 current，原始未改。若需要作者修正，先回報、修後本人新複查，不自行改教材。

初判形成後另做 dependency-audit-01.json：真實掃描現行351個canonical page IDs，與上一輪及目前 freeze 比較，完整列 inventory_page_ids；實際閱讀僅改動與邊界路線。這次新增6.10、chapter-06/index/readme改動；前三頁已完成真實primary三輪，readme本callback的reader/technical完成後才派你。可在此階段讀本輪 reader/technical/primary continuity 的實際報告與版本，以核對依賴邊界和是否還有未收束必要問題；不能把他人的pass當成你的全文閱讀。資料與工具相容性修復未修改 validation 正文，fresh technical 已核對，不必冒稱reader/continuity重讀該頁。最終audit需逐項交代是否還有待callback，不能沿用primary時readme未完成的敘述作最終結論。

報告放 docs/course-review-20261027-tokenizer/reviews/continuity-readme-callback/；不改受審正文、工具、前輪報告，不開代理、不提交、不發布。交件回報判定、問題、實際核對與未驗證範圍、檔案路徑。這是銜接／版面review，不重新驗證模型訓練或宣稱真人學習成果。

範圍更新（實際派工前）：作者全域計數核對另找到 curriculum 的312/338，現已只改為313/339，保存 freeze-03.json。你本次主審仍僅 readme，該頁相對 freeze-02 的正文／圖未變；可依 freeze-03 核對目前版本。網站已反映大綱修正，該頁新估時尚待其後的獨立讀者，預覽暫不顯示大綱舊估時，不冒填分鐘。你的 audit 請如實標明 curriculum 尚待第三個 scoped wave（三輪均在你本輪結束後才開始），不能宣稱它已完成。這是 scope 邊界，不是要求你忽略新發現；任何實際疑點仍保留。最終全輪 dependency audit 將由第三波 continuity 重新核對。
