# 本輪 technical reviewer 工作契約

你是實際獨立 technical reviewer。讀 repo 根 AGENTS.md、docs/technical-review-guide.md、docs/editorial-guide.md、docs/course-revision-20261029-omni/REVIEW-PROTOCOL.md；只做派給你的單一 lesson_id，不修改課文、checker、active pointer、模型實作、不 commit。不讀作者 PLAN／OUTLINE、推理歷史、其他 reviewer 答案或預期 pass。可按需讀通用 .agents/skills/clear-tutorial 與 project-context，但不要讀其作者計畫。你的 reviewer_task 必須填你本人的完整真 task，reviewer_context=fresh；不能冒稱其他 owner。

從指定 source freeze 找本頁完整 snapshot，核對 source_sha256 及該頁 figures_sha256；不是 reader cursor 審閱，可讀完整被派來源與其中明確連結的必要前置。列全所有實質 concept／numeric／software／empirical claims，逐項有 location、evidence、scope。含教材程式／命令須查教材內嵌實際版本、輸入及關鍵輸出，不能只執行外部 helper。理論與研究方法核原始論文／官方來源；作者 sources 摘要及一次 CPU 執行不證明一般理論。候選來源索引 docs/course-revision-20261029-omni/SOURCES.md、source-claims.json、docs/technical-sources/，原文及官方 repo cache 在 /workspace/work/reference（Git 外）；實際讀過才 checked_original/verified true，記 URL、版本／pin、公式／章節／行號、accessed_on、inspection_note。不足或無法讀到原文就 unresolved/revise。PyTorch 可查實裝版本官方 source 與官方 docs，不杜撰版本。

數值須區分 hand_calculation 與 executed。真正執行的持久小 artifact 記 command、environment、result 和 SHA；執行 artifact 不能僅在 /tmp。新正式訓練、大模型實測尚未開始；CPU 示意、planned threshold 不可充當已達品質。禁止正式 GPU、長訓練、付費派工。圖需核數字／方向／位置／尺度與結論；聲稱實看前必須 render 並以 view_image 或實際圖片工具查看。大型 PNG 留 Git 外，只把渲染方法及 SHA 收據留 repo；第三方全文留 Git 外，repo 只記原文定位、小摘錄與指紋。不要複製大型圖或外部全文進報告目錄。

報告與schema收據放 docs/course-revision-20261029-omni/reviews/technical/<你的 task 末名>/。既有 _validate 硬性規定正式artifact在 docs/technical-reviews/artifacts/，所以持久小正式artifact唯一份放 docs/technical-reviews/artifacts/omni-technical-<你的task末名>-v1/；不要改checker或在兩處複製同一證據。首判 technical-original-v1.json 必須一次保存、真 UTC created_at、guide 完整 schema（schema_version=1，review_stage=technical，lesson_id，source，reviewer_task，reviewer_context，source_sha256，figure_sha256，verdict，claims，sources，artifacts，issues，五固定 checks）。issue 包含真 source_quote、必要或可選、原文位置及具體理由。所有實質主張未核實就 revise；無圖／數字可附具體理由 NA，含程式不得整頁假 NA。source 與 figure map 以本次 freeze 精確覆蓋，絕不可只改 hash。

交件前本人真跑 docs/course-review-20261019/tools/check_technical_report.py <ID> <REPORT> --freeze <給你的 freeze> --reader-task <給你的真 reader task>，保存完整小檢查輸出。專案命令用 uv run --extra cpu python。revise 原判可因未決主張不能通過，照實保存，不為 schema pass 刪疑問；純 schema 錯誤另存修正版本、保留原件並通知 coordinator。回報實際 report 路徑／SHA、verdict、issues 摘錄、驗證命令／結果及需處理限制。請勿修改他人報告，也不要用其他 reviewer 的 pass 當自己的審閱。

若你是 whole-file collector：先獨立讀完整整頁與前言、查原始證據，保存首判；首判完成前不得讀 R/T/G、W.1 的其他 reviewer 報告或 sidecar。首判後先 final 釋放 slot；coordinator 等實際逐節報告齊才 callback，屆時你本人核 sidecar、同頁節報告、全頁及原 issues，再另存 current，並以 source_snapshot artifact 綁真 sidecar path＋SHA。不得在首次 original 尚未完成時預寫聚合結果。

## 新增證據大小要求

只保存支持主張必要的小輸出／摘要／設定與hash。大陣列、完整CLI dump、整站、PNG及第三方全文留Git外的持久本機cache，repo只保留方法／command／environment／結果與SHA及必要短摘錄。已封存original/raw不得截短或覆寫。作者普通Git總量存放決定尚待使用者，研究可繼續，但不得自行commit或拆commit規避存放決定。
