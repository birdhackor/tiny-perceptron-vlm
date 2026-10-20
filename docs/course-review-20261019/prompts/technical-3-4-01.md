# 3.4 事實與技術審閱（正式審閱）

你是這一節的技術審閱者。你的任務名是 `/root/rv1019_technical_3-4`，寫進報告的 `reviewer_task`；`reviewer_context` 填 `fresh`。

這是正式審閱：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫；修正後的複查另寫新檔。
- 你沒有參與撰寫。核對的是頁面寫出的每一個實質主張是否成立、數字是否算對、圖是否表達同一件事，以及引述的來源與實驗能支持多大的結論。
- 讀者是否讀得懂，由另一位審閱者負責；你不判斷易讀性，除非某句話會讓讀者得到錯的事實。

最終回覆用正體中文。

## 審什麼

- 頁面：`3.4`，原文是 `course/chapters/03.md` 的這一節。凍結版本在 `docs/course-review-20261019/freeze-01/sources/3.4.md`，SHA-256 `6e91ec43442e8b1819db3f6ced32704bb835ff1479056c3563f0fbe6b5256a35`；報告的 `source_sha256` 必須等於它。
- 頁面的圖：`course/figures/rewrite-03-qkv-flow.svg`（SHA-256 `2686978d0393bb829546d92aaee107d760a24b0cd29d3e5fe5ea2128c7aa292f`；算繪 `outputs/course-review-20261019/figure-renders-04/2686978d0393bb829546d92aaee107d760a24b0cd29d3e5fe5ea2128c7aa292f.png`）
- 規約：先全文讀 `docs/technical-review-guide.md`，報告格式照它寫。

## 能看什麼、能做什麼

- repo 裡的教材、程式、測試、資料紀錄都可以讀，也可以執行。
- 小節程式用 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/03.md#3.4" --execute` 實跑；需要另外驗證時，自己寫小程式放進 artifact 目錄再執行。
- 可以連網找原始來源：論文、官方文件、原始碼。用 `curl` 讀原文，記下 HTTPS URL、實際讀取的版本、讀取日期與定位。第三方全文不要存進 repo；需要佐證時只保存支持主張的小段摘錄。
- 看圖：`outputs/course-review-20261019/figure-renders-04/<SVG 的 SHA-256>.png` 是凍結版 SVG 的算繪，用 `view_image` 看；也可以讀 SVG 原文。對應收據在 `docs/course-review-20261019/figure-renders-04.json`。這些 PNG 不在 Git 裡，不要列成 artifact；圖的核對寫在 `figure_consistency` 的 details，註明看的是哪個 PNG。

## 獨立性

完成並存下自己的原始判斷以前，**不看**下面這些位置。提示詞明確指出路徑的凍結原文、凍結圖與工具，照路徑直接讀，不在此限：

- 這一輪其他審閱者的紀錄：`docs/course-review-20261019/` 底下除了你自己報告目錄以外的審閱報告、處置、seal、閱讀計畫與提示詞（例如 `reviews/` 裡別人的目錄、`dispositions-*.json`、`seal-*.json`、`reader-plan-*.json`、`prompts/`），以及 `docs/technical-reviews/artifacts/` 裡別人的目錄；
- 以前各輪的審閱紀錄、問題與處置：`docs/technical-reviews/*.json`、`docs/reader-reviews/`、`docs/course-repair-20261008/`、`docs/course-repair-20261017/`、`docs/course-revision-20261005/`、`docs/course-revision-20261007-phase7/`、`docs/course-revision-20261016/`、`docs/course-publish-20261008/`、`docs/course-inline-code-20261017/`；
- `outputs/`（除了上面的 PNG）。

在 repo 裡做大範圍搜尋時，先排除上面這些位置再搜，例如 `rg` 加上 `--glob '!docs/course-review-20261019/**' --glob '!docs/technical-reviews/**' --glob '!docs/reader-reviews/**' --glob '!docs/course-repair-*/**' --glob '!docs/course-revision-*/**' --glob '!docs/course-publish-20261008/**' --glob '!docs/course-inline-code-20261017/**' --glob '!outputs/**'`。自己的報告與 artifact 目錄直接照路徑列出。若仍意外看到這些位置的內容，照實寫進報告與回覆的訊號C。

`docs/technical-sources/` 是找證據的候選庫，可以用來找線索，但它的摘要不能當證據；原文要自己讀。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_3-4/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-3-4/`。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不跑長時間訓練。
- 執行輸出要複製進 artifact 目錄，報告引用 artifact 目錄裡的檔與它的 SHA-256；不要引用 `/tmp` 或其他暫存位置的檔。
- `section_facts.py --execute` 會在 `outputs/reviewer-tools/runs/<時間>/` 寫執行紀錄（被 gitignore），這是工具本來的設計，不算越界；不要改寫或繞過它的路徑限制。報告要引用的執行結果，從那裡複製進你的 artifact 目錄。
- 原始判斷：`docs/course-review-20261019/reviews/rv1019_technical_3-4/technical-3-4-original-01.json`。格式照規約，另外：
  - `source` 寫 `course/chapters/03.md#3.4`，`lesson_id` 寫 `3.4`；
  - `source_freeze` 寫 `{"path": "docs/course-review-20261019/freeze-04.json", "sha256": "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"}`；
  - `figure_sha256` 必須恰好等於這一節的圖清單（上面列的路徑與 SHA-256）；沒有圖就是 `{}`；
  - `author_tasks` 寫 `["/root"]`；
  - 全部主張核實才寫 `verdict: pass`。有任何 `unresolved` 或 `contradicted` 就寫 `revise`，並在 `issues` 逐條寫：`claim_id`、`status`、`details`（哪一句、錯在哪、依據）、`necessary`（不改就會讓讀者得到錯的事實或做錯事時寫 true）、`suggestion`（最小的修正方向）。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_technical_report.py 3.4 docs/course-review-20261019/reviews/rv1019_technical_3-4/technical-3-4-original-01.json --freeze docs/course-review-20261019/freeze-04.json --reader-task /root/rv1019_reader_ch03
  ```
  它只檢查格式、版本與證據的雜湊，不判斷你的內容。`verdict: revise` 時它會回報「審閱要求修訂」，這是預期的；其他錯誤要修好（另存 `-02` 並說明原因）。

## 回覆

- verdict、主張數（依 kind）、未核實或矛盾的主張各一行。
- 報告與每個 artifact 的路徑與 SHA-256。
- 自我檢查的輸出。
- 你沒能驗證的範圍。
