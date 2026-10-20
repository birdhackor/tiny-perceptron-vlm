# 發布閘門工具的第四次複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_review-tools`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-03.json`（verdict revise，SHA-256 `95da65dd61410fb86ae104bfa57e28e71e4585d2f7f8fbb4ca534addfb75dfea`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告和它們的 artifact 都不要改。
- 你要判斷的仍是：工具照它的註解與設計做事、沒有讓不該通過的紀錄通過、測試真的守著每一條規則。

最終回覆用正體中文。

## 自上一份報告以來的變動

只有 `b62ec6e7`，差異用 `git diff 229bb69d b62ec6e7` 看，改了三個檔：

- `docs/review-tools/incremental_revision.py`：`fenced` 改成用兩條收尾規則各認一次圍欄：`closes_superfences` 是網站的認法，行首（縮排與引用記號）要和開頭那行相同；`closes_commonmark` 是 Notebook 的認法。任一條認為在區塊裡就算。`author_patches` 旁的「不涵蓋」也改寫了。
- `tests/test_incremental_revision_review.py`：新增假收尾與兩種認法不一致的情況。
- `docs/technical-review-guide.md` 的作者補丁規則：條件改成以 `author_patches` 與 `undo_patches` 為準；另加一句：補丁內容的對錯由要求的作者負責，改到行內程式碼、數學式、連結或沒有圍欄的程式區塊裡的字時也一樣。

協調者對你上一份兩條 issue 的處理：

- `AUTHOR-PATCH-CODEBLOCK`：假收尾（縮排或引用記號不同的同一串記號）是程式錯，照上面改了。縮排四格的程式區塊與 HTML 的 `<pre>` 沒有加檢查，改成規約明寫由作者負責。
- `AUTHOR-PATCH-MARKUP`：沒有加檢查。作者要的是「改一個字不必重跑整套審閱」，不碰連結與程式是協調者另外加的限制；要讓閘門真的認出這些位置，得在閘門裡另寫一套 Markdown 解析。所以改了規約，讓規約只承諾閘門真的會擋的條件。

規約怎麼定由作者與協調者決定，不在這次審閱的範圍。要判斷的是：閘門是否照現在的規約與守衛旁的註解做事，以及註解寫的「驗證」是否成立。

## 審什麼

1. **上一份的兩條 issue**：在現在的規約與註解下，是否解決。解決的列進 `resolved_issue_ids`；沒解決的照上一份的格式留在 `issues`，寫明現在的規約或註解哪一句與閘門的行為不符。
2. **`fenced` 的新認法**：核對 `closes_superfences` 與 uv.lock 鎖定版本的 pymdownx.superfences 是否一致（包括你上一份的假收尾反例），`closes_commonmark` 與 CommonMark 規格是否一致；兩者取聯集時，有沒有網站或 Notebook 會當成程式、閘門卻判在區塊外的輸入。試出來的附可重現的輸入與輸出。
3. **你上一份的探針**：用現行程式重跑 `author-patch-probe-current-03.py` 的各個案例，記下哪些結果改變。
4. **沒改的檔**：上一份 `checked_files` 裡這次沒改的檔，核對現行 SHA-256 與上一份所記相同。
5. **跑測試**：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_incremental_revision_review.py tests/test_course_review_preflight.py tests/test_technical_reviews.py`，記下結果。

## 能看什麼、能做什麼

- repo 裡的程式、測試、紀錄都可以讀，也可以執行；可以連網查官方文件與規格。
- 可以讀你自己之前的報告與 artifact。
- 完成並存下這次的判斷以前，不看其他人對這些工具的審閱報告與複審意見，例如 `outputs/course-inline-code-20261017/pub/` 底下的 `reports/`、`orders/`、`findings.tsv`、`design/consult-*.md`，以及 `docs/course-review-20261019/reviews/` 底下別人的目錄。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-review-tools/`，用新檔名（例如加 `-current-04` 字尾），不改之前的檔；
  - 你為了試規則自己建的暫存目錄（放在 `outputs/course-review-20261019/tool-audit/` 底下的新子目錄）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- artifact 只放小檔：你寫的程式、指令輸出、SHA 清單。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-04.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-03.json", "sha256": "95da65dd61410fb86ae104bfa57e28e71e4585d2f7f8fbb4ca534addfb75dfea"}`；
  - `"resolved_issue_ids"`：解決了的上一份 issue 的 `id`；
  - `checked_files` 列上一份的十二個檔，全部記現行 SHA-256。
- 五項核對都和預期一致、測試通過，才寫 `verdict: pass`；否則寫 `revise`，並在 `issues` 逐條寫明。

## 回覆

- verdict；上一份每條 issue 解決與否，各一行。
- `fenced` 的兩種認法：你試了哪些情況、各自的結果；試出的漏洞各一行，附重現方式。
- 上一份探針各案例的新結果。
- issues 各一行；沒有就說沒有，並說明你怎麼確認。
- 報告與每個 artifact 的路徑與 SHA-256。
- 你沒能驗證的範圍。
