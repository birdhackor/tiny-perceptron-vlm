你的原始判斷 `docs/course-review-20261019/reviews/rv1019_reader_ch10/reader-original-01.json`（SHA-256 `52b84c6b8e155488717ac2b147f15ac987172935d9917be0e943f534c9442946`）已收件。這 12 頁的正文與圖在你讀過之後沒有改過。請再交兩份檔，都是正式紀錄：寫好就以 SHA-256 綁住，不能改寫；寫錯就另存 `-02` 並在回覆說明原因。

能看什麼、不能做什麼都照原指示：可以讀你自己的 session 檔與你自己的輸出目錄；只寫 `docs/course-review-20261019/reviews/rv1019_reader_ch10/`。不要重讀教材，也不要開新的 session。

## 一、目前判斷 `reader-current-01.json`

12 頁每頁一筆，表示你對目前版本的最後判斷。因為正文與圖都沒改，依據就是你的原始判斷。

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_reader_ch10",
  "created_at": "<寫檔當下的 UTC 時間，ISO 8601>",
  "report_kind": "current judgments",
  "source_freeze": <照抄原始判斷的 source_freeze>,
  "reading_plan": <照抄原始判斷的 reading_plan>,
  "pages": [
    {
      "page_id": "<頁>",
      "source_sha256": "<照抄原始判斷這一頁的值>",
      "figures_sha256": <照抄原始判斷這一頁的值>,
      "verdict": "<你對目前版本的判斷：pass 或 需補說明>",
      "basis_report": {"path": "docs/course-review-20261019/reviews/rv1019_reader_ch10/reader-original-01.json", "sha256": "52b84c6b8e155488717ac2b147f15ac987172935d9917be0e943f534c9442946"},
      "ai_estimate": {"minutes_min": <整數>, "minutes_max": <整數>, "reason": "<理由>"}
    }
  ]
}
```

`verdict` 與 `ai_estimate` 一般照原始判斷；若你現在有不同看法，照實寫，並在回覆說明哪一頁、為什麼。

## 二、收束 `reader-closure-01.json`

你的 12 個 session 的筆記（`note-NNN.json`）裡，凡是 `concept_checks` 某一項的 `verdict` 不是恰好「已足夠」，都要在這裡交代最後狀態，一項一筆，不能漏也不能重複。主 agent 從你的筆記數到 13 項；你自己列出來核對，數目不同就照你查到的寫，並在回覆說明。

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_reader_ch10",
  "created_at": "<寫檔當下的 UTC 時間，晚於 reader-current-01.json 的 created_at>",
  "report_kind": "checkpoint closure",
  "basis_report": {"path": "docs/course-review-20261019/reviews/rv1019_reader_ch10/reader-current-01.json", "sha256": "<reader-current-01.json 寫好後的 SHA-256>"},
  "pages": [
    {
      "page_id": "<有這種概念的頁>",
      "source_sha256": "<同目前判斷這一頁>",
      "figures_sha256": <同目前判斷這一頁>,
      "verdict": "<pass，或仍有必要缺口時寫 需補說明>",
      "checkpoint_resolutions": [
        {
          "note": {"path": "docs/course-revision-20261016/reader-sessions/<session>/note-NNN.json", "sha256": "<那份 note 的 SHA-256>"},
          "concept_index": <那一項在 concept_checks 裡的位置，從 0 起算>,
          "original_verdict": "<筆記裡原本的判定，照抄>",
          "final_status": "resolved | nonblocking | deferred",
          "reason": "<為什麼是這個狀態>",
          "evidence": "<支持它的原文，或你讀到的後文位置與原句>"
        }
      ]
    }
  ]
}
```

- `pages` 只列有這種概念的頁。
- `final_status`：
  - `resolved`：之後讀到的正文已經講清楚；`evidence` 引那段原文與所在頁、單元。
  - `nonblocking`：這一頁的理解與操作不需要它；`reason` 說明為什麼不需要。
  - `deferred`：正文明說之後才教，而這一頁不靠它；`evidence` 引正文指向後面的那句。
- 某一項其實是這一頁不補就讀不懂的必要缺口時，不要硬寫成上面三種：那一頁 `verdict` 寫「需補說明」，並在回覆說明。
- SHA-256 用 `shasum -a 256 <檔>` 算。

## 回覆

- 兩份檔的路徑與 SHA-256。
- 目前判斷裡和原始判斷不同的地方；沒有就說沒有。
- 收束的表：頁、note、concept_index、原判定、最後狀態，一項一行；以及你列出的總數。
