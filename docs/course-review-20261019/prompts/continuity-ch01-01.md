# 第 1 章試跑的銜接審閱（正式審閱）

你是這個 wave（`pilot-ch01`）的銜接審閱者。你的任務名是 `/root/rv1019_continuity_ch01`，寫進每一份紀錄的 `reviewer_task`。

這是正式審閱：你寫的檔會以 SHA-256 綁住，之後不能改寫；修正後的複查另寫新檔。你沒有參與撰寫。最終回覆用正體中文。

## 審什麼

銜接：讀者照順序讀下來，每一頁用到的東西是不是在前面教過（或在當頁講清楚），頁與頁之間的承接、回指與預告是不是對得上；暖身頁教的寫法和正文用的是否一致；章末是否把讀者交到下一章導言需要的狀態。

- 閱讀路線（依序）：
  1. 先備：`first-steps`（操作與數學暖身）；
  2. 本 wave 的頁：`chapter-01`、`1.1`、`1.2`、`1.3`、`1.4`、`1.5`、`1.6`、`1.7`、`1.8`、`1.9`、`1.10`、`1.11`、`1.12`、`1.13`、`1.14`、`1.15`；
  3. 下一章導言：`chapter-02`。
- 原文從凍結檔取：`docs/course-review-20261019/freeze-02.json`（SHA-256 `2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1`）的 `pages[]`，每頁有 `snapshot`（原文快照路徑）、`source_sha256`、`figures_sha256`。報告裡的 SHA 照抄凍結檔。
- 圖：`outputs/course-review-20261019/figure-renders/<PNG 檔名>`，用 `view_image` 看。哪張 SVG 對應哪個 PNG，記在 `docs/course-review-20261019/figure-renders.json`（SHA-256 `88d147c0700c8e7e4ba6a20c376df56851da5410f0aa926fe326904d2209f39e`）的 `records[]`。路線上的 7 張圖都在裡面：`first-steps` 2 張、`1.1`、`1.3`、`1.5`、`1.10`、`1.12` 各 1 張。
- 判準：`.agents/skills/clear-tutorial/SKILL.md`、`.agents/skills/clear-tutorial/references/review-protocol.md`、`.agents/skills/clear-tutorial/references/calibration.md`。

## 順序與獨立性

1. **先存自己的原始判斷**：只讀上面的原文、圖與判準，寫 `continuity-original-01.json`。這一步之前，不看任何審閱報告：`docs/course-review-20261019/reviews/` 底下其他人的目錄、`docs/technical-reviews/` 底下的報告與 artifact、以前輪次的目錄（`docs/course-repair-20261017/`、`docs/course-revision-20261016/`、`docs/course-publish-20261008/` 的審閱紀錄）都不讀。也不讀 `course/`、`notebooks/`、`scripts/`、`tiny_perceptron/`、Git 歷史，不用網頁搜尋，不執行教材程式。
2. **再交叉核對**：原始判斷存好後，下一則訊息會給本 wave 的讀者報告與技術報告路徑。那時再讀它們，看它們提出的問題是否影響銜接、修正後的版本是否引入新的斷裂，寫 `continuity-current-01.json`。原始判斷不改；改變的判斷在 current 裡寫明理由。

## 寫什麼、寫在哪

- 只寫你的目錄 `docs/course-review-20261019/reviews/rv1019_continuity_ch01/`。不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不連網。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- 兩份報告都用這個格式：

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_continuity_ch01",
  "created_at": "寫檔當下的 UTC 時間，ISO 8601 帶 +00:00",
  "report_kind": "continuity original 或 continuity current",
  "source_freeze": {"path": "docs/course-review-20261019/freeze-02.json", "sha256": "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1"},
  "route": ["first-steps", "chapter-01", "1.1", "…", "1.15", "chapter-02"],
  "pages": [
    {
      "page_id": "…",
      "source_sha256": "照抄凍結檔",
      "figures_sha256": "照抄凍結檔（字典，沒有圖就是 {}）",
      "verdict": "pass 或 需補說明",
      "summary": "這頁承接了什麼、交出了什麼",
      "prerequisites_used": ["這頁用到、在前面教過的東西，附教的位置"],
      "issues": [
        {
          "id": "C-1.3-1",
          "quote": "原文一兩句",
          "problem": "斷裂在哪：用到沒教過的、回指錯、預告沒兌現、暖身頁寫法不一致…",
          "evidence": "前面哪裡教了／沒教（頁與原句）",
          "impact": "對讀者的實際影響",
          "necessary": true,
          "suggestion": "最小的修正方向"
        }
      ]
    }
  ],
  "unverified": "沒能核對的範圍，例如實際網頁的桌面與手機版面"
}
```

- `pages` 只列本 wave 的 16 頁，依閱讀順序。暖身頁與下一章導言不列成 pages，但它們造成的問題記在受影響的那一頁。
- 有任何 `"necessary": true` 的 issue，該頁 verdict 寫 `需補說明`。沒有問題就是 `"issues": []`。
- 實際網頁的版面（桌面、手機）由協調者另外抽樣截圖；你沒看到的部分，在 `unverified` 寫明，不要宣稱看過。

## 回覆

- 每頁一行：頁、verdict、必要問題數。
- 報告路徑與 SHA-256。
- 沒能核對的範圍。
