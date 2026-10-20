# 發布閘門工具的第五次複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_review-tools`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-04.json`（verdict revise，SHA-256 `54fee4fb1bd6ebc98dc2d5e3591d7f47ecd507df050927e7955a7408d1371798`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告和它們的 artifact 都不要改。
- 你要判斷的仍是：工具照它的註解與設計做事、沒有讓不該通過的紀錄通過、測試真的守著每一條規則。

最終回覆用正體中文。

## 自上一份報告以來的變動

- `bba56e8b` 改了 `docs/review-tools/incremental_revision.py` 與 `tests/test_incremental_revision_review.py`，差異用 `git diff b62ec6e7 bba56e8b -- docs/review-tools/incremental_revision.py tests/test_incremental_revision_review.py` 看。`fenced` 不再模擬網站與 Notebook 的收尾規則；作者補丁改成只用在程式圍欄寫法單純的檔（`PLAIN_FENCE` 與 `fence_blocks`），現行版本與每倒退一筆後的版本都要符合。守衛旁的「驗證」與「不涵蓋」也改寫了。
- `41651edb` 改了 `docs/technical-review-guide.md` 的作者補丁規則：例子多了「檔裡的程式圍欄寫法單純」；作者負責的範圍多了「閘門沒認出、網站或 Notebook 卻顯示成程式的字」。
- 其餘 commit 只有派工單與派工紀錄，沒有改到你上一份 `checked_files` 裡的檔。用 `git diff --stat b62ec6e7 HEAD -- <那十二個檔>` 核對。

協調者對你上一份兩條 issue 的處理：

- `AUTHOR-PATCH-FENCE-WHITESPACE`、`AUTHOR-PATCH-FENCE-STATE` 與第三次的假收尾是同一個原因：閘門對圍欄的認法和某個渲染器不同。所以不再逐條補收尾規則。改成只承認寫法單純的檔；寫法有任何不單純，作者補丁一律不適用，例如 Unicode 空白、反引號 info、選項、縮排、引用記號、`~~~`、行中間的三個反引號、檔尾停在區塊裡。
- 守衛註解的草稿原本說，單純寫法下網站、Notebook 與 `scripts/build_course.py` 認出的區塊相同。協調者提交前查到這句不成立：網站各節與 Notebook 的 Markdown 格，是 `scripts/build_course.py` 切過的片段。所以註解改成只宣稱程式格那一句，其餘寫進「不涵蓋」，並在規約寫明由作者負責。

規約怎麼定由作者與協調者決定，不在這次審閱的範圍。要判斷的是：閘門是否照現在的規約與守衛旁的註解做事，以及註解寫的「驗證」是否成立。

## 審什麼

1. **之前的四條 issue**：第三次的 `AUTHOR-PATCH-MARKUP`、`AUTHOR-PATCH-CODEBLOCK`（你第四次已判解決），與第四次的 `AUTHOR-PATCH-FENCE-WHITESPACE`、`AUTHOR-PATCH-FENCE-STATE`。在現在的程式、規約與註解下，各自是否解決。發布閘門只承認 verdict pass 的報告列在 `resolved_issue_ids` 裡的 issue，所以四條都要重新判一次：解決的列進 `resolved_issue_ids`；沒解決的照之前的格式留在 `issues`，寫明現在的規約或註解哪一句與閘門的行為不符。
2. **`fence_blocks` 的「驗證」**：
   - 程式是否照註解的規則判斷：怎麼分行、哪些行要整行符合 `PLAIN_FENCE`、怎麼開與收、檔尾。
   - 註解說：這樣的檔裡，`scripts/build_course.py` 切出的每個程式格（從 ```` ```python ```` 那行切到下一個 ```` ``` ````）都落在閘門認出的區塊裡。核對這句是否成立；試著找閘門接受、但某個程式格含有閘門判在區塊外的字的檔。
   - 一筆或幾筆補丁，能不能讓現行版本與審閱過的版本認出的區塊不同，使補丁位置在其中一個版本裡落在區塊內。
   - 用現行程式重跑你上一份的反例，以及 `fence-admission-probe-current-04.py`、`fence-semantics-probe-current-04.py` 的各個案例，記下哪些結果改變。

   試出來的情況都附可重現的輸入與輸出。
3. **「不涵蓋」**：註解列的每種情況是否真的存在、說法是否正確。
4. **沒改的檔**：上一份 `checked_files` 裡這次沒改的檔，核對現行 SHA-256 與上一份所記相同。
5. **跑測試**：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_incremental_revision_review.py tests/test_course_review_preflight.py tests/test_technical_reviews.py`，記下結果。

## 能看什麼、能做什麼

- repo 裡的程式、測試、紀錄都可以讀，也可以執行；可以連網查官方文件與規格。
- 可以讀你自己之前的報告與 artifact。
- 完成並存下這次的判斷以前，不看其他人對這些工具的審閱報告與複審意見，例如 `outputs/course-inline-code-20261017/pub/` 底下的 `reports/`、`orders/`、`findings.tsv`、`design/consult-*.md`，以及 `docs/course-review-20261019/reviews/` 底下別人的目錄。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-review-tools/`，用新檔名（例如加 `-current-05` 字尾），不改之前的檔；
  - 你為了試規則自己建的暫存目錄（放在 `outputs/course-review-20261019/tool-audit/` 底下的新子目錄）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- artifact 只放小檔：你寫的程式、指令輸出、SHA 清單。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-05.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-04.json", "sha256": "54fee4fb1bd6ebc98dc2d5e3591d7f47ecd507df050927e7955a7408d1371798"}`；
  - `"resolved_issue_ids"`：四條之中解決了的 `id`；
  - `checked_files` 列上一份的十二個檔，全部記現行 SHA-256。
- 五項核對都和預期一致、測試通過，才寫 `verdict: pass`；否則寫 `revise`，並在 `issues` 逐條寫明。

## 回覆

- verdict；四條 issue 解決與否，各一行。
- `fence_blocks` 的核對：你試了哪些情況、各自的結果；試出的漏洞各一行，附重現方式。
- 上一份探針各案例的新結果。
- issues 各一行；沒有就說沒有，並說明你怎麼確認。
- 報告與每個 artifact 的路徑與 SHA-256。
- 你沒能驗證的範圍。
