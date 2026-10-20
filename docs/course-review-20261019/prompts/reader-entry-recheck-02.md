銜接審閱之後，作者又改了 `training`（銜接修正；改法見各頁新版）。請照原指示的讀法，用新的閱讀計畫從頭重讀這 1 頁，再交三份檔。三份都是正式紀錄：寫好就以 SHA-256 綁住，不能改寫；寫錯就另存新編號並在回覆說明原因。

## 怎麼讀

- 能看什麼、不能看什麼、不能做什麼，都照原指示，只有下面幾點換成新的：
  - `R` 的 `start` 與 `supplement` 這次要帶：`--reviewer /root/rv1019_reader_entry --plan docs/course-review-20261019/reader-plan-entry-recheck-02.json --freeze docs/course-review-20261019/freeze-07.json`。
  - 圖改看 `outputs/course-review-20261019/figure-renders-04/<SHA>.png`（SHA 一樣先用 `shasum -a 256 <SVG 路徑>` 算出）；`outputs/` 底下這次只看這個目錄的 PNG。
  - 這幾頁在網站上沒有附程式輸出，不用跑 `executed_outputs.py`。
1. 依序對 `training` 各開一個新的 session：`R start <頁> …`，從第一個單元逐單元讀完。每個單元照原格式寫 checkpoint 到 `docs/course-review-20261019/reviews/rv1019_reader_entry/drafts/recheck-02-<頁>-<單元序號>.json`，再 `R next <session> --checkpoint <那份檔>`，直到印出 `{"complete": true, …}`。
2. 這 1 頁的正文都讀完後，才用 `R supplement <頁> …` 補讀它們的選讀。
3. 只讀這 1 頁，不讀其他頁。

## 交三份檔

### 一、`reader-recheck-02.json`

格式同你的原始判斷 `docs/course-review-20261019/reviews/rv1019_reader_entry/reader-original-01.json`，`pages` 只有這 1 頁，依閱讀順序。

- `report_kind` 寫 `"recheck judgments after sequential rereading"`。
- `source_freeze` 與 `reading_plan` 指這次的凍結檔與新計畫（path 與用 `shasum -a 256` 算的 sha256）。
- 每頁的 `source_sha256`、`figures_sha256` 照抄這次新 session 的 `opening.json`；`session`、`session_evidence` 指這次的新 session。
- `optional_reading_observation` 寫這次補讀選讀得到的理解。
- `verdict` 寫 `pass` 或 `需補說明`；修改後仍在的問題與這次新發現的問題，照原格式列在 `issues`。
- 寫好後照原指示跑 `check_reader_original.py`，只是對象換成這份檔，把輸出貼進回覆。

### 二、`reader-current-03.json`

5 頁每頁一筆，表示你對目前版本的最後判斷。格式同 `reader-current-02.json`：

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_reader_entry",
  "created_at": "<寫檔當下的 UTC 時間，ISO 8601，晚於 reader-recheck-02.json 的 created_at>",
  "report_kind": "current judgments",
  "source_freeze": {"path": "docs/course-review-20261019/freeze-07.json", "sha256": "<用 shasum -a 256 算出>"},
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

- 這次重讀的頁：`basis_report` 是 `docs/course-review-20261019/reviews/rv1019_reader_entry/reader-recheck-02.json`；`verdict` 與 `ai_estimate` 是你對修改後版本的判斷。
- 其他頁：正文與圖在你上次判斷之後沒有改過，照抄 `docs/course-review-20261019/reviews/rv1019_reader_entry/reader-current-02.json`（SHA-256 `a96fb56b97eefc353a801a6c91745d699a64b07224838454c465e6d86e3e66bd`）裡這一頁的 `source_sha256`、`figures_sha256`、`basis_report`、`verdict` 與 `ai_estimate`；若你現在有不同看法，照實寫，並在回覆說明哪一頁、為什麼。

### 三、`reader-closure-03.json`

你所有 session 的筆記（`note-NNN.json`）裡，凡是 `concept_checks` 某一項的 `verdict` 不是恰好「已足夠」，都要在這裡交代最後狀態，一項一筆，不能漏也不能重複：

- 沒有重讀的頁：照抄 `docs/course-review-20261019/reviews/rv1019_reader_entry/reader-closure-02.json`（SHA-256 `cc522b0ab033aea6643e4975640d2fc595b64965770fa8a0a00f86d0f79688f7`）裡這一頁的各筆，`source_sha256`、`figures_sha256` 同目前判斷這一頁。
- 這次重讀的頁：之前讀這一頁的所有 session（第一次讀，以及之前各次複查時重讀的那幾次，如果有）與這次的新 session 都要。之前那些項目要對這次修改後的版本重新判斷，不要照抄。

`reader-closure-02.json` 列了 26 項；`reader-closure-03.json` 要包含這些項，再加上這次新 session 的項目。你自己列出來核對，數目不對就照你查到的寫，並在回覆說明。

格式同 `reader-closure-02.json`，只有 `basis_report` 換成 `{"path": "docs/course-review-20261019/reviews/rv1019_reader_entry/reader-current-03.json", "sha256": "<reader-current-03.json 寫好後的 SHA-256>"}`，`created_at` 晚於 `reader-current-03.json` 的 `created_at`。`final_status` 的意思同上次：

- `resolved`：之後讀到的正文已經講清楚；`evidence` 引那段原文與所在頁、單元。
- `nonblocking`：這一頁的理解與操作不需要它；`reason` 說明為什麼不需要。
- `deferred`：正文明說之後才教，而這一頁不靠它；`evidence` 引正文指向後面的那句。
- 某一項其實是這一頁不補就讀不懂的必要缺口時，不要硬寫成上面三種：那一頁 `verdict` 寫「需補說明」，並在回覆說明。
- SHA-256 用 `shasum -a 256 <檔>` 算。

## 回覆

- 三份檔的路徑與 SHA-256，以及 `check_reader_original.py` 的輸出。
- 這次讀到的修改，對你的理解有沒有影響；你的判斷和上次有沒有不同。
- 收束的表：頁、session、note、concept_index、原判定、最後狀態，一項一行；以及你列出的總數。
