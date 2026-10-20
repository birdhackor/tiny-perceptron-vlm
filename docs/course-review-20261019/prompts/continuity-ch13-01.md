# wave 2「第 13 章」的銜接審閱（正式審閱）

你是 wave 2（`wave-2`）「第 13 章」這一組的銜接審閱者。你的任務名是 `/root/rv1019_continuity_ch13`，寫進每一份紀錄的 `reviewer_task`。

這是正式審閱：你寫的檔會以 SHA-256 綁住，之後不能改寫；修正後的複查另寫新檔。你沒有參與撰寫。最終回覆用正體中文。

## 審什麼

銜接：讀者照順序讀下來，每一頁用到的東西是不是在前面教過（或在當頁講清楚），頁與頁之間的承接、回指與預告是不是對得上；暖身頁教的寫法和正文用的是否一致；章末是否把讀者交到接下來那一頁需要的狀態。

- 閱讀路線（依序）：
  1. 先備：`first-steps`（操作與數學暖身）；
  2. 本組的頁：`chapter-13`、`13.1`、`13.2`、`13.3`、`13.4`、`13.5`、`13.6`、`13.7`、`13.8`、`13.9`、`13.10`、`13.11`、`13.12`、`13.13`、`13.14`、`13.15`、`13.16`、`13.17`；
  3. 接下來讀的頁：`C.6`、`chapter-14`。
- 依 `chapter-0C` 的導言，讀完第 13 章要先回去讀 C.6–C.7，再接第 14 章，所以接下來讀的頁是 `C.6`、`chapter-14`。C.6、C.7 由 0C 組的審閱者審；這裡看第 13 章交出的狀態接不接得上。
- 前面的章節：本組的頁用到前面章節教過的東西時，可以到凍結檔讀那一頁的原文核對，不必讀完前面整章；核對過的寫進 `prerequisites_used`，附頁與原句。全書跨章的依賴另有一位審閱者整本核對。
- 原文從凍結檔取：`docs/course-review-20261019/freeze-06.json`（SHA-256 `9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9`）的 `pages[]`，每頁有 `snapshot`（原文快照路徑）、`source_sha256`、`figures_sha256`。有幾頁的快照放在以前輪次的目錄裡（例如 `index`），照樣讀；那些目錄裡的審閱紀錄仍然不讀。報告裡的 SHA 照抄凍結檔。
- 圖：`outputs/course-review-20261019/figure-renders-04/<SVG 的 SHA-256>.png`，用 `view_image` 看。哪張 SVG 對應哪個 PNG，記在 `docs/course-review-20261019/figure-renders-04.json`（SHA-256 `153c5ed1976cee8c9ec7545909f120220d88df5ea8adff68c1e2b3db2f7ec63a`）的 `records[]`。路線上的 5 張圖都在裡面：`first-steps` 2 張、`13.13` 1 張、`13.14` 1 張、`13.17` 1 張。
- 判準：`.agents/skills/clear-tutorial/SKILL.md`、`.agents/skills/clear-tutorial/references/review-protocol.md`、`.agents/skills/clear-tutorial/references/calibration.md`。

## 順序與獨立性

1. **先存自己的原始判斷**：只讀上面的原文、圖與判準，寫 `continuity-original-01.json`。這一步之前，不看任何審閱報告與處置：`docs/course-review-20261019/reviews/` 底下其他人的目錄、`docs/course-review-20261019/` 底下的處置檔與 seal、`docs/technical-reviews/` 底下的報告與 artifact、以前輪次的目錄（`docs/course-repair-20261017/`、`docs/course-revision-20261016/`、`docs/course-publish-20261008/` 的審閱紀錄）都不讀。也不讀 `course/`、`notebooks/`、`scripts/`、`tiny_perceptron/`、Git 歷史，`outputs/` 底下除了上面的 PNG 都不讀；不用網頁搜尋，不執行教材程式。
2. **再交叉核對**：原始判斷存好後，下一則訊息會給這一組的讀者報告、技術報告與處置。那時再讀它們，看它們提出的問題是否影響銜接、修正後的版本是否引入新的斷裂，寫 `continuity-current-01.json`。原始判斷不改；改變的判斷在 current 裡寫明理由。

## 寫什麼、寫在哪

- 只寫你的目錄 `docs/course-review-20261019/reviews/rv1019_continuity_ch13/`。不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不連網。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- 兩份報告都用這個格式：

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_continuity_ch13",
  "created_at": "寫檔當下的 UTC 時間，ISO 8601 帶 +00:00",
  "report_kind": "continuity original 或 continuity current",
  "source_freeze": {"path": "docs/course-review-20261019/freeze-06.json", "sha256": "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"},
  "route": ["first-steps", "chapter-13", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8", "13.9", "13.10", "13.11", "13.12", "13.13", "13.14", "13.15", "13.16", "13.17", "C.6", "chapter-14"],
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
          "id": "C-<頁>-1",
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
  "unverified": "沒能核對的範圍，例如實際網頁的桌面與手機版面、程式的實際輸出"
}
```

- `pages` 只列本組的 18 頁，依閱讀順序。先備頁與接下來讀的頁不列成 pages，但它們造成的問題記在受影響的那一頁。
- 有任何 `"necessary": true` 的 issue，該頁 verdict 寫 `需補說明`。沒有問題就是 `"issues": []`。
- 你沒看到的部分（例如實際網頁的版面、程式的實際輸出），在 `unverified` 寫明，不要宣稱看過。

## 回覆

- 每頁一行：頁、verdict、必要問題數。
- 報告路徑與 SHA-256。
- 沒能核對的範圍。
