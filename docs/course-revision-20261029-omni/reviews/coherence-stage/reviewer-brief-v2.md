# 本輪連貫性審閱工作契約 v2（尚未派工）

這份通用工作契約不代表階段已開始。只有主代理保存真正technical階段收件與關閉紀錄後，才能派工。每位審閱者使用無作者歷史的獨立task，先保存自己的來源初判，再看其他角色的答案與舊問題。

只做指定來源的唯讀獨立審閱，不修改課文、圖、模型、checker、他人報告、active pointer，不commit，也不啟動付費工作。讀根AGENTS.md、docs/editorial-guide.md與通用clear-tutorial的銜接／圖文準則；不要讀作者PLAN／OUTLINE、作者觀察、本輪reader／technical答案、問題處置表或歷史appendix要求，直到自己的初判不可變保存完成。共用檔案系統不是真正隔離；實際提前曝光必須如實記錄。

## 實際閱讀

從派工指定的canonical source freeze核對當頁snapshot、正文SHA與圖的SHA，閱讀相關必要前文→完整當頁→範圍內下一步。自行建立依賴，不以標題、術語列表、作者概念清單代替機制與理由的銜接。必要程式、輸入與關鍵輸出必須能在正文定位，不能用打開Colab／Notebook補當下理解。

先走正文，再檢查選讀邊界。逐一說明必要對象與符號、操作或表示轉換、此刻需要它的理由、例子支持到哪個範圍。公平區分原文正常可推得、正常複習前文、需拼湊或猜測、新增未教前提；不要把合理簡化或選讀名稱自動升為必要缺口。計算、機制、設計收益與實測效果分別核對。

圖須真正以實際圖片工具查看。另實際看桌面與手機的當前網站，保存使用尺寸、路徑／錨點、必要標記／箭頭／單位、公式與程式橫向捲動、折疊狀態、固定頁首及下載／Colab操作入口。不要只以DOM尺寸、SVG文字提取、建置成功或截圖存在，宣稱自己已看圖。截圖放Git外，只在repo保存方法、SHA與必要小觀察。

## 自己的來源初判

在自己專屬的reviews/continuity/目錄一次寫入continuity-original-v1.json，記真UTC created_at。最少結構如下，內容由本人實讀填寫，範例欄位不是預期答案：

```json
{
  "schema_version": 1,
  "review_stage": "continuity",
  "reviewer_task": "/root/ACTUAL_TASK",
  "reviewer_context": "fresh",
  "created_at": "ACTUAL_UTC_TIME",
  "source_freeze": {"path": "ACTUAL_FREEZE", "sha256": "ACTUAL_SHA"},
  "verdict": "pass or revise",
  "pages": [
    {
      "page_id": "ACTUAL_CANONICAL_ID",
      "source": "ACTUAL_SOURCE",
      "source_sha256": "ACTUAL_SOURCE_SHA",
      "figures_sha256": {},
      "source_quote": "本人實際讀過、確實出現在此頁凍結正文中的原句",
      "verdict": "pass or revise",
      "checks": [
        {
          "relation": "本人所核的具體必要關係",
          "source_quote": "當前正文實際原句",
          "prerequisite_location": "精確頁節與實際已教範圍",
          "identity": "對象／符號對應",
          "mechanism": "操作／轉換如何銜接",
          "need": "此刻需要這個設計的已教理由",
          "usage_and_scope": "此例如何使用與支持邊界",
          "verdict": "pass or revise",
          "reason": "具體來源依據"
        }
      ],
      "actual_read_route": [],
      "actual_visual_observations": [],
      "issues": []
    }
  ],
  "actual_reading": [],
  "machine_scan_only": [],
  "unverified": []
}
```

每頁本身也要有非空`source_quote`，逐字出現在該頁的凍結正文；`checks`各項的引文另外保留，不能以它們或後續appendix替換自己原判的頁級來源引文。這是通用證據格式，不提供待審內容的答案。

每個必要或可選問題保存穩定id、原句、位置、必要性與理由、讀者此刻缺少的最小關係、已教部分、具體理解影響與最小修正負擔。沒有證據就明列待查；不要以概括好處、全演算法／全效能不是本課目的，代替回應真正缺少的基本關係。

初判保存後先回報精確路徑／SHA及待處理項，讓主代理收件。不得事後覆寫原判來配合其他角色；需要修訂或補查，另存current、交叉核對或appendix，記真正可見材料與時間。修後由原owner實核，未執行不記resolved。

## 全站依賴清冊的另項工作

只有明確派到全站inventory的審閱者，另存dependency-audit-original-v1.json：使用當前freeze全部canonical page IDs，自行核真路由／來源與依賴。明分完整實讀、機械掃描、抽查、沿用證據及未驗範圍；369個ID不等於369頁新實讀。auxiliary英文入口另列來源／SHA與實際範圍，不塞進canonical清冊。

這份audit不得提前稱整輪完成。若其他真階段／callback／新版成果尚未收束，照實保留pending範圍；待真正到件後另存最終audit，不把未來收到的內容填回原時間。作者對未報問題通過圖的抽查，是另一份作者證據，不代替此獨立審閱。

本v2只在派工前補齊既有工具需要的頁級原句欄位；尚無連貫性原判。先前v1保留：`docs/course-revision-20261029-omni/reviews/coherence-stage/reviewer-brief-v1.md`，SHA-256 `83108466f78a091e448de585526f71ed98a6778d95d4170a2933800bb16f5ff1`。
