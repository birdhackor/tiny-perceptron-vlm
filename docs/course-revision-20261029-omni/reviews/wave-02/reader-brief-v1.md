# 本輪逐段首讀工作契約

只做所派頁面的獨立首次閱讀，不修改正文、圖、Notebook、程式、他人報告或 active pointer，不 commit，不啟動模型訓練或付費服務。讀者設定：中文初學者；Python、tensor 與基礎數學可依當前正文的精確連結回查，不以自己的專家常識補未教的關係。正文已明示的必要前文依計畫實讀，記清楚實際讀過與只假設具備的能力。

讀 `.agents/skills/clear-tutorial/SKILL.md`、`references/review-protocol.md` 及 `docs/editorial-guide.md` 的通用首讀準則。不要讀作者計畫、問題處置、技術／銜接答案、舊 reader 報告或其他角色原判；不要先開完整章檔、當頁 snapshot 或網站來預看尚未解鎖的後文。共用 filesystem 不是隔離，提前曝光須如實記錄。

計畫的 `source_freeze` 為真正來源。先為所有 `context_pages` 用以下命令取得前文並實讀；它是已明定先備，不是當頁首讀：

```text
uv run --extra cpu python docs/course-revision-20261016/tools/reader.py supplement PAGE --reviewer ACTUAL_TASK --plan ACTUAL_PLAN --freeze ACTUAL_FREEZE
```

依計畫 `main_pages` 的順序，每頁另開真正游標：

```text
uv run --extra cpu python docs/course-revision-20261016/tools/reader.py start PAGE --reviewer ACTUAL_TASK --plan ACTUAL_PLAN --freeze ACTUAL_FREEZE
```

每次 tool 真正回傳一段後，先理解並保存本人 checkpoint，才以 `next SESSION --checkpoint PATH` 開下一段。禁止預寫未來答案、讀 state 的未揭露 units、同一腳本模擬整段閱讀或事後重建首次理解。每筆 checkpoint 有 `unit_index`、非空 `understanding`、`materials_and_labels`、`expected_change`、`confusion_and_quote`、`missing_visuals`，及 `concept_checks` 清單。每個實際必要概念的 check 記 `concept`、`identity`、`mechanism`、`need`、`usage`、`example_scope`、逐字 `source_quote`、`verdict`。第二題的機制與用途分別核對；理由需接到當時已讀的獨立問題，不能只重述好處。合理推論記最短已教步驟，未知與正常待後文解釋先保存，後文解答另加位置，不改原筆。

圖出現時才取得它的凍結路徑，render 640／360 兩種寬度並用真正圖片工具看。保存 SHA 與實看範圍；沒有圖也查是否缺必要素材／位置／對應。正文整組游標完成後才可對 main pages 使用 supplement 補讀選讀、看整頁網站或視覺排版；分開記時點與範圍。理解所需程式、輸入及關鍵輸出必須可在正文定位，不得靠 Colab 補當下理解。

在自己的 `reviews/wave-02/ACTUAL_TASK_SUFFIX/` 一次新寫 `reader-original-v1.json`，不覆寫舊報告。需記真正 UTC、task、fresh context、freeze／plan 路徑及 SHA、實際前文路線、提前曝光、未驗證範圍。每頁至少有 page_id／source／source_sha256／figures_sha256／session／verdict、自己的整體理解、四題與原句、程式／輸入／輸出解釋、有意義小變化與理由、練習、缺圖判定、全部 necessary／optional 問題及其原句／當時單位／理解影響／最小關係／建議；沒有問題可留空，不編造。

逐頁保留真 session 的 opening 與全部原 note 路徑＋SHA 引用，以及先備 supplement 收據。保存首次閱讀 AI 分鐘區間與理由，明記是主觀估計，不能用實際 agent wall time 或原作者估時代替。必要缺口則 revise；不能用後見的專家復述抹掉首讀斷點。初判保存後先回報路徑、SHA、判定及問題，未收件前不互看他人答案。未執行程式或未看完整網站均如實寫明，不冒稱已驗。
