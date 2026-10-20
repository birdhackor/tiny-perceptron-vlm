# 7.7 事實與技術審閱的複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_7-7`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_7-7/technical-7-7-original-01.json`（verdict revise，SHA-256 `cc6d73b23e5c8a22c87fb9d3ac784a59898433a78a0680c6a38bc7162fc83f79`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告與 artifact 都不要改。
- 判準和上一份相同：頁面寫出的每一個實質主張是否成立、數字是否算對、圖是否表達同一件事，以及引述的來源與實驗能支持多大的結論。

最終回覆用正體中文。

## 自上一份報告以來的變動

- 頁面原文改了。上一份審的是 `docs/course-review-20261019/freeze-01/sources/7.7.md`（SHA-256 `50dd698b6191bfe21c34b216aeb75d562192bb1f3fc4aac6e7828c613066571d`），新版在 `docs/course-review-20261019/freeze-05/sources/7.7.md`（SHA-256 `b59db468c89f0cbc1dd71b8d487030209f352817767abdff27e8cca1e7ba9575`）。差異用 `git diff --no-index docs/course-review-20261019/freeze-01/sources/7.7.md docs/course-review-20261019/freeze-05/sources/7.7.md` 看。
- 你上一份提的問題：`c09`。作者的修法在差異裡。
- 差異裡也可能有不是你提的修正（例如同一種寫法在別頁被指出後，作者把這頁同類的句子一起改了）。每一處改過的句子都當成新主張核對。

## 審什麼

- 頁面：`7.7`，原文是 `course/chapters/07.md` 的這一節。凍結版本在 `docs/course-review-20261019/freeze-05/sources/7.7.md`，SHA-256 `b59db468c89f0cbc1dd71b8d487030209f352817767abdff27e8cca1e7ba9575`；報告的 `source_sha256` 必須等於它。
- 頁面的圖：`course/figures/rewrite-07-07-padding-positions.svg`（SHA-256 `16931629aba76361e505777950208c4af602515fb4e2b3e1314517b43fadf72b`；算繪 `outputs/course-review-20261019/figure-renders-04/16931629aba76361e505777950208c4af602515fb4e2b3e1314517b43fadf72b.png`）
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

- 只寫你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_7-7/` 與 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-7-7/`；新檔用新檔名（例如加 `-current-01` 字尾），不改之前的檔。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不跑長時間訓練。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_7-7/technical-7-7-current-01.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_7-7/technical-7-7-original-01.json", "sha256": "cc6d73b23e5c8a22c87fb9d3ac784a59898433a78a0680c6a38bc7162fc83f79"}`；
  - `"resolved_issue_ids"`：解決了的上一份 issue 的 `claim_id`；
  - `source_freeze` 寫 `{"path": "docs/course-review-20261019/freeze-05.json", "sha256": "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"}`；
  - `reviewer_task` 寫 `/root/rv1019_technical_7-7`，`reviewer_context` 仍填 `fresh`；
  - 全部主張核實才寫 `verdict: pass`；否則 `revise`，在 `issues` 逐條寫（欄位同上一份）。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_technical_report.py 7.7 docs/course-review-20261019/reviews/rv1019_technical_7-7/technical-7-7-current-01.json --freeze docs/course-review-20261019/freeze-05.json --reader-task /root/rv1019_reader_ch07
  ```

## 回覆

- verdict；上一份每條 issue 解決與否，各一行。
- 改過的句子各一行：成立與否、依據。
- 報告與每個新 artifact 的路徑與 SHA-256；自我檢查的輸出。
- 你沒能驗證的範圍。
