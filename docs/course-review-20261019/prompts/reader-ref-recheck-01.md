作者處置了你的原始判斷，改了 curriculum、validation。請照原指示的讀法，用新的閱讀計畫從頭重讀 `curriculum`、`validation` 這 2 頁，再交三份檔。三份都是正式紀錄：寫好就以 SHA-256 綁住，不能改寫；寫錯就另存 `-02` 並在回覆說明原因。

## 怎麼讀

- 能看什麼、不能看什麼、不能做什麼，都照原指示，只有下面幾點換成新的：
  - `R` 的 `start` 與 `supplement` 這次要帶：`--reviewer /root/rv1019_reader_ref --plan docs/course-review-20261019/reader-plan-ref-recheck-01.json --freeze docs/course-review-20261019/freeze-04.json`。
  - 圖改看 `outputs/course-review-20261019/figure-renders-04/<SHA>.png`（SHA 一樣先用 `shasum -a 256 <SVG 路徑>` 算出）；`outputs/` 底下這次只看這個目錄的 PNG。
  - 這幾頁在網站上沒有附程式輸出，不用跑 `executed_outputs.py`。
1. 依序對 `curriculum`、`validation` 各開一個新的 session：`R start <頁> …`，從第一個單元逐單元讀完。每個單元照原格式寫 checkpoint 到 `docs/course-review-20261019/reviews/rv1019_reader_ref/drafts/recheck-<頁>-<單元序號>.json`，再 `R next <session> --checkpoint <那份檔>`，直到印出 `{"complete": true, …}`。
2. 這 2 頁的正文都讀完後，才用 `R supplement <頁> …` 補讀它們的選讀。
3. 只讀這 2 頁，不讀其他頁。

## 交三份檔

### 一、`reader-recheck-01.json`

格式同你的 `reader-original-01.json`，`pages` 只有這 2 頁，依閱讀順序。

- `report_kind` 寫 `"recheck judgments after sequential rereading"`。
- `source_freeze` 與 `reading_plan` 指這次的凍結檔與新計畫（path 與用 `shasum -a 256` 算的 sha256）。
- 每頁的 `source_sha256`、`figures_sha256` 照抄這次新 session 的 `opening.json`；`session`、`session_evidence` 指這次的新 session。
- `optional_reading_observation` 寫這次補讀選讀得到的理解。
- `verdict` 寫 `pass` 或 `需補說明`；修改後仍在的問題與這次新發現的問題，照原格式列在 `issues`。
- 寫好後照原指示跑 `check_reader_original.py`，只是對象換成這份檔，把輸出貼進回覆。

### 二、`reader-current-01.json`

4 頁每頁一筆，表示你對目前版本的最後判斷。

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_reader_ref",
  "created_at": "<寫檔當下的 UTC 時間，ISO 8601，晚於 reader-recheck-01.json 的 created_at>",
  "report_kind": "current judgments",
  "source_freeze": {"path": "docs/course-review-20261019/freeze-04.json", "sha256": "<用 shasum -a 256 算出>"},
  "reading_plan": <照抄原始判斷的 reading_plan>,
  "pages": [
    {
      "page_id": "<頁>",
      "source_sha256": "<照抄 basis_report 裡這一頁的值>",
      "figures_sha256": <照抄 basis_report 裡這一頁的值>,
      "verdict": "<你對目前版本的判斷：pass 或 需補說明>",
      "basis_report": {"path": "<見下>", "sha256": "<那份檔的 SHA-256>"},
      "ai_estimate": {"minutes_min": <整數>, "minutes_max": <整數>, "reason": "<理由>"}
    }
  ]
}
```

- 這次重讀的頁：`basis_report` 是 `docs/course-review-20261019/reviews/rv1019_reader_ref/reader-recheck-01.json`；`verdict` 與 `ai_estimate` 是你對修改後版本的判斷。
- 其他頁：正文與圖在你讀過之後沒有改過，`basis_report` 是原始判斷 `docs/course-review-20261019/reviews/rv1019_reader_ref/reader-original-01.json`（SHA-256 `3de9c0495d773051f59bca0b4d7b88950c40018ea41c419938142926a3d888cf`）；`verdict` 與 `ai_estimate` 一般照原始判斷，若你現在有不同看法，照實寫，並在回覆說明哪一頁、為什麼。

### 三、`reader-closure-01.json`

你所有 session 的筆記（`note-NNN.json`）裡，凡是 `concept_checks` 某一項的 `verdict` 不是恰好「已足夠」，都要在這裡交代最後狀態，一項一筆，不能漏也不能重複：

- 沒有重讀的頁：第一次讀的 session。
- 這次重讀的頁：第一次讀的 session 與這次的新 session 都要。第一次那些項目要對修改後的版本重新判斷，不要照抄。

主 agent 從你第一次讀的筆記數到 7 項（不含這次的新 session）；你自己列出來核對，數目不同就照你查到的寫，並在回覆說明。

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_reader_ref",
  "created_at": "<寫檔當下的 UTC 時間，晚於 reader-current-01.json 的 created_at>",
  "report_kind": "checkpoint closure",
  "basis_report": {"path": "docs/course-review-20261019/reviews/rv1019_reader_ref/reader-current-01.json", "sha256": "<reader-current-01.json 寫好後的 SHA-256>"},
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

- 三份檔的路徑與 SHA-256，以及 `check_reader_original.py` 的輸出。
- 這次讀到的修改，對你的理解有沒有影響；你的判斷和第一次讀時有沒有不同。
- 收束的表：頁、session、note、concept_index、原判定、最後狀態，一項一行；以及你列出的總數。
