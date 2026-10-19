作者在技術審閱之後改了 1.4。請照原指示的讀法，用新的閱讀計畫從頭重讀 1.4，再交三份檔。這三份都是正式紀錄：寫好就以 SHA-256 綁住，不能改寫；寫錯就另存 `-02` 並在回覆說明原因。

## 怎麼讀

- 能看什麼、不能看什麼、不能做什麼，都照原指示；另外不直接讀 `docs/course-review-20261019/executed-outputs-ch01-02.json`。
- `R` 的 `start` 與 `supplement` 這次要帶：`--reviewer /root/rv1019_reader_ch01b --plan docs/course-review-20261019/reader-plan-ch01-recheck-01.json --freeze docs/course-review-20261019/freeze-02.json`。
- 程式輸出照舊用 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/executed_outputs.py show <session> --reviewer /root/rv1019_reader_ch01b`；它會從新計畫取這次的輸出清單。

1. `R start 1.4 …`：開新的 session，從第一個單元逐單元讀完。每個單元照原格式寫 checkpoint 到 `docs/course-review-20261019/reviews/rv1019_reader_ch01b/drafts/recheck-1.4-<單元序號>.json`，再 `R next <session> --checkpoint <那份檔>`，直到印出 `{"complete": true, …}`。
2. 正文讀完後，才用 `R supplement 1.4 …` 補讀 1.4 的選讀。
3. 只讀 1.4，不讀其他頁。

## 交三份檔

### 一、`reader-recheck-01.json`

格式同你的 `reader-original-01.json`，`pages` 只有 1.4 一頁。

- `report_kind` 寫 `"recheck judgments after sequential rereading"`。
- `source_freeze` 與 `reading_plan` 指這次的凍結檔與新計畫（path 與用 `shasum -a 256` 算的 sha256）。
- `source_sha256`、`figures_sha256` 照抄這次新 session 的 `opening.json`。
- `session`、`session_evidence` 指這次的新 session。
- `optional_reading_observation` 寫這次補讀選讀得到的理解。
- `verdict` 寫 `pass` 或 `需補說明`。

### 二、`reader-current-02.json`

格式同你的 `reader-current-01.json`，`pages` 只有 1.4 一頁。

- `basis_report` 寫 `{"path": "docs/course-review-20261019/reviews/rv1019_reader_ch01b/reader-recheck-01.json", "sha256": "<寫好後算的 SHA-256>"}`。
- `source_sha256`、`figures_sha256` 照抄 recheck 那一頁。
- `verdict` 與 `ai_estimate` 是你對修改後版本的判斷。
- `created_at` 晚於 recheck 的 `created_at`。

### 三、`reader-closure-02.json`

格式同你的 `reader-closure-01.json`，`pages` 只有 1.4 一頁。

- `basis_report` 寫 `{"path": "docs/course-review-20261019/reviews/rv1019_reader_ch01b/reader-current-02.json", "sha256": "<寫好後算的 SHA-256>"}`。
- `created_at` 晚於 current-02 的 `created_at`。
- `source_sha256`、`figures_sha256` 同 current-02 那一頁。
- `checkpoint_resolutions` 要涵蓋 1.4 兩個 session 裡**所有**不是恰好「已足夠」的概念，一項一筆，不漏不重：
  - 第一次讀的 session `08da1e57a25448dfa56a4a794694ce76`：你在 `reader-closure-01.json` 收束過其中一項。請對修改後的版本重新判斷它的最後狀態，不要照抄。
  - 這次的新 session：每一項都要收束。
- `final_status` 的意思、`evidence` 的寫法，照 `reader-closure-01.json` 那次的指示。某一項其實是這一頁不補就讀不懂的必要缺口時，不要硬寫成三種狀態之一：那一頁 `verdict` 寫「需補說明」，並在回覆說明。

## 回覆

- 三份檔的路徑與 SHA-256。
- 這次讀到的修改，對你的理解有沒有影響；你的判斷和第一次讀時有沒有不同。
- 收束的表：session、note、concept_index、原判定、最後狀態，一項一行；以及你列出的總數。
