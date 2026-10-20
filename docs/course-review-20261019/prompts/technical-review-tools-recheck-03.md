# 發布閘門工具的第三次複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_review-tools`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-02.json`（verdict pass，SHA-256 `566084e5e32cf6baa989ca5542e4129d37777c715bc7922ba41242a2938a07bf`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告和它們的 artifact 都不要改。
- 你要判斷的仍是：工具照它的註解與設計做事、沒有讓不該通過的紀錄通過、測試真的守著每一條規則。

最終回覆用正體中文。

## 自上一份報告以來的變動

作者決定：作者直接要求的小補丁不必重新審閱，紀錄寫明是作者要求、沒有經過 AI 審閱。閘門為此新增「作者補丁」規則，在兩個 commit 裡：

- `f7ede9da`：manifest 新增選用欄位 `author_patches`，指向作者補丁清單；閘門把現行檔倒退清單裡的補丁，得到的較早版本只在特定幾處承認；補充還原清單新增一筆 `scripts/check_technical_reviews.py`。
- `229bb69d`：補丁只能改 Markdown 原文；程式圍欄改照網站用的 pymdownx.superfences 認（含 `~~~` 與引用區塊裡的圍欄），`old`／`new` 不能含 `~~~`。

差異：`git diff b88b4b63 229bb69d -- <檔>`。改到的檔：

- 你上一份 `checked_files` 列的七個檔裡：`docs/review-tools/incremental_revision.py`、`tests/test_incremental_revision_review.py`、`docs/course-review-20261019/baseline-supplements.json`。
- 不在你上一份範圍、但實作同一條規則的檔：`scripts/check_technical_reviews.py`、`docs/course-review-20261019/tools/preflight.py`、`docs/course-review-20261019/tools/issue_inventory.py`、`tests/test_course_review_preflight.py`、`tests/test_technical_reviews.py`。
- 規則文字：`docs/technical-review-guide.md`、`docs/editorial-guide.md`、`AGENTS.md`。這一輪的作者補丁清單是 `docs/course-review-20261019/author-patches.json`。

## 審什麼

1. **作者補丁規則**：判準是 `docs/technical-review-guide.md` 寫的規則，以及程式裡守衛旁的註解（它們自己寫了驗證什麼、不涵蓋什麼）。要回答的是：閘門有沒有承認規則以外的東西——例如讓補丁改到程式、圖、連結或跨行的內容，讓清單以外的修改搭便車，或在規則沒說的地方承認舊版本。也要回答：缺 `author_patches` 欄位時，閘門的行為是否和改動前相同。可以自己寫小程式或 fixture 去試；試出漏洞時，附可重現的輸入與輸出。
2. **補充還原清單新增的一筆**：在新的乾淨 clone 核對：
   - 保存的副本在 repo 內、路徑與 `live_path` 對得上，SHA-256 等於舊輪次實作表為 `scripts/check_technical_reviews.py` 記的值；
   - 用現行清單跑重放，結果通過；
   - 只拿掉新增的這一筆再跑，重放回報 `scripts/check_technical_reviews.py` 的 drift。
   - 重放的做法同上一份：用 `importlib` 載入 `docs/review-tools/baseline_replay.py`，把清單的 `files` 交給它的 `replay(root, supplements)`，印出 `status` 與 `failures`。
   - clone 放在 `outputs/course-review-20261019/tool-audit/` 底下的新子目錄，不要覆寫之前用過的 clone。用 `git clone --no-local` 從這個 repo 複製，設 `GIT_LFS_SKIP_SMUDGE=1`。
3. **沒改的四個檔**：`docs/review-tools/baseline_replay.py`、`docs/review-tools/phase7_review.py`、`docs/course-revision-20261016/tools/reader.py`、`docs/course-revision-20261016/tools/freeze_sources.py`。核對現行 SHA-256 與你上一份 `checked_files` 記的相同，在 `claims` 記下。
4. **跑測試**：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_incremental_revision_review.py tests/test_course_review_preflight.py tests/test_technical_reviews.py`，記下結果。

## 能看什麼、能做什麼

- repo 裡的程式、測試、紀錄都可以讀，也可以執行；可以連網查官方文件。
- 可以讀你自己之前的報告與 artifact，以及上一份用過的設計文件。
- 完成並存下這次的判斷以前，不看其他人對這些工具的審閱報告與複審意見，例如 `outputs/course-inline-code-20261017/pub/` 底下的 `reports/`、`orders/`、`findings.tsv`、`design/consult-*.md`，以及 `docs/course-review-20261019/reviews/` 底下別人的目錄。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-review-tools/`，用新檔名（例如加 `-current-03` 字尾），不改之前的檔；
  - 上面的 clone 子目錄，以及你為了試規則自己建的暫存目錄（放在 `outputs/course-review-20261019/tool-audit/` 底下）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- artifact 只放小檔：你寫的程式、指令輸出、SHA 清單。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-03.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-02.json", "sha256": "566084e5e32cf6baa989ca5542e4129d37777c715bc7922ba41242a2938a07bf"}`；
  - `"resolved_issue_ids": []`（上一份沒有未解決的 issue）；
  - `checked_files` 列上一份的七個檔，加上「變動」一節列的五個實作同一條規則的檔，全部記現行 SHA-256。
- 四項核對都和預期一致、測試通過，才寫 `verdict: pass`；否則寫 `revise`，並在 `issues` 逐條寫明。

## 回覆

- verdict；`checked_files` 每個檔的 SHA-256。
- 作者補丁規則：你試了哪些情況、各自的結果；試出的漏洞各一行，附重現方式。
- 補充還原清單新增那一筆的三項核對各一行：做法、結果、artifact。
- issues 各一行；沒有就說沒有，並說明你怎麼確認。
- 報告與每個 artifact 的路徑與 SHA-256。
- 你沒能驗證的範圍。
