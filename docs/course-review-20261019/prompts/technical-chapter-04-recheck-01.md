# chapter-04 事實與技術審閱的複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_chapter-04`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_chapter-04/technical-chapter-04-original-01.json`（verdict pass，SHA-256 `25ad1b3b52ddeff2d346fbf5f036467b27dc7c675693f7f583351365d5fb9806`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告與 artifact 都不要改。
- 判準和上一份相同：頁面寫出的每一個實質主張是否成立、數字是否算對、圖是否表達同一件事，以及引述的來源與實驗能支持多大的結論。

最終回覆用正體中文。

## 自上一份報告以來的變動

- 你上一份以 SHA-256 引用的 repo 檔有變動：`course/chapters/03.md`（你記的 `df0709c102dba36fa89b3c88904f55172339a61c15c4a0583484b9b919bd2ee7`，現行 `61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09`）；`course/chapters/14.md`（你記的 `c8391918c37c6a09fc8674989989f75e295513f00b299c79af42c43e355a82bb`，現行 `40448837dfb1418b4fdfdad4544e133e6554bb80831ae2a2802ef6ab3ffc7601`）。差異用 `git diff 5ef56bc756f3 3af7f13ef959 -- course/chapters/03.md`、`git diff 5ef56bc756f3 3af7f13ef959 -- course/chapters/14.md` 看，確認你依它核對的主張仍成立，報告記現行 SHA-256。

## 審什麼

- 頁面：`chapter-04`，原文是 `course/chapters/04.md` 的這一節。凍結版本在 `docs/course-review-20261019/freeze-01/sources/chapter-04.md`，SHA-256 `54cfbe71cacf6517e3c9f85b4228fcce892569afeba8d0d3460fb72ac44de5de`；報告的 `source_sha256` 必須等於它。
- 頁面的圖：這一頁沒有圖，`figure_sha256` 寫 `{}`。
- 規約：`docs/technical-review-guide.md`，報告格式照它寫。

做法：

1. 上一份的每一條 issue：判斷現在是否解決。解決的把 `claim_id` 列進 `resolved_issue_ids`；沒解決的照上一份的格式留在 `issues`。
2. 改過的句子當成新主張核對，要有自己的依據。沒改的主張，確認上一份的核對仍成立；上一份的 artifact 路徑與 SHA-256 都沒變時可以沿用。
3. 報告引用的 repo 檔，SHA-256 要等於現行檔。
4. 這份報告要完整列出頁面全部實質主張，不只列改過的部分。

## 能看什麼、能做什麼

- repo 裡的教材、程式、測試、資料紀錄都可以讀，也可以執行；可以連網找原始來源，規則同上一份。
- 小節程式照上一份的方式實跑；`section_facts.py` 讀的是現行章檔，現行章檔與上面的凍結版本相同。
- 看圖：`outputs/course-review-20261019/figure-renders-04/<SVG 的 SHA-256>.png` 是凍結版 SVG 的算繪，用 `view_image` 看；也可以讀 SVG 原文。

## 獨立性

完成並存下這次的判斷以前，仍然不看其他角色的報告與以前的審閱紀錄（範圍同上一份），也不看 `outputs/`（上面的 PNG，以及「變動」一節明說開放的檔除外）。課文或你上一份報告明確指名路徑的證據檔，即使位在這些目錄裡，也可以只讀那一個檔，並在報告註明。

## 寫什麼、寫在哪

- 只寫你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_chapter-04/` 與 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-chapter-04/`；新檔用新檔名（例如加 `-current-01` 字尾），不改之前的檔。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不跑長時間訓練。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_chapter-04/technical-chapter-04-current-01.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_chapter-04/technical-chapter-04-original-01.json", "sha256": "25ad1b3b52ddeff2d346fbf5f036467b27dc7c675693f7f583351365d5fb9806"}`；
  - `"resolved_issue_ids"`：解決了的上一份 issue 的 `claim_id`；
  - `source_freeze` 寫 `{"path": "docs/course-review-20261019/freeze-05.json", "sha256": "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"}`；
  - `reviewer_task` 寫 `/root/rv1019_technical_chapter-04`，`reviewer_context` 仍填 `fresh`；
  - 全部主張核實才寫 `verdict: pass`；否則 `revise`，在 `issues` 逐條寫（欄位同上一份）。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_technical_report.py chapter-04 docs/course-review-20261019/reviews/rv1019_technical_chapter-04/technical-chapter-04-current-01.json --freeze docs/course-review-20261019/freeze-05.json --reader-task /root/rv1019_reader_ch04
  ```

## 回覆

- verdict；上一份每條 issue 解決與否，各一行。
- 改過的句子各一行：成立與否、依據。
- 報告與每個新 artifact 的路徑與 SHA-256；自我檢查的輸出。
- 你沒能驗證的範圍。
