# 20.12 事實與技術審閱的複查（正式審閱，原審閱者第二次複查）

你是 `/root/rv1019_technical_20-12`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_20-12/technical-20-12-current-01.json`（verdict revise，SHA-256 `6583081ee19fd89cdb74cba999a93186c5a1e358d0e27d35088509cb84560560`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告與 artifact 都不要改。
- 判準和上一份相同：頁面寫出的每一個實質主張是否成立、數字是否算對、圖是否表達同一件事，以及引述的來源與實驗能支持多大的結論。

最終回覆用正體中文。

## 自上一份報告以來的變動

- 頁面原文沒有改，凍結版本與上一份相同。
- 你上一份的 `c18`、`c19` 停在 unresolved：判準由誰撰寫、有沒有逐題經人審核，缺直接紀錄。作者把這件事寫成聲明 `docs/authorship.md`（SHA-256 `58092b2cabcac23e39b935b2aaf3716ec2a68eee8ad3d8f937483b8a5fe8fb20`），裡面有作者的原話，以及它涵蓋的判準所在的檔與寫下時的 SHA-256。這份聲明是作者對自己做法的說明；它能支持多大的結論，由你判斷。

## 審什麼

- 頁面：`20.12`，原文是 `course/chapters/20.md` 的這一節。凍結版本在 `docs/course-review-20261019/freeze-06/sources/20.12.md`，SHA-256 `45e1e1d5ec66a7a1d792c7ab76bdbfbbe5749a47cb31ac927d1759b384e52dcc`；報告的 `source_sha256` 必須等於它。
- 頁面的圖：`course/figures/natural-v4-asr-two-routes.svg`（SHA-256 `aaed40e6fc4eebfa6ba59a98b2c08ea42624a481553570313b146d8e1f0085ac`；算繪 `outputs/course-review-20261019/figure-renders-04/aaed40e6fc4eebfa6ba59a98b2c08ea42624a481553570313b146d8e1f0085ac.png`）
- 規約：`docs/technical-review-guide.md`，報告格式照它寫。

做法：

1. 上一份的每一條 issue：判斷現在是否解決。解決的把 `claim_id` 列進 `resolved_issue_ids`；沒解決的照上一份的格式留在 `issues`。
2. 沒改的主張，確認上一份的核對仍成立；上一份的 artifact 路徑與 SHA-256 都沒變時可以沿用。
3. 報告引用的 repo 檔，SHA-256 要等於現行檔。
4. 這份報告要完整列出頁面全部實質主張，不只列這次重新判斷的部分。

## 能看什麼、能做什麼

- repo 裡的教材、程式、測試、資料紀錄都可以讀，也可以執行；可以連網找原始來源，規則同上一份。
- 小節程式照上一份的方式實跑；`section_facts.py` 讀的是現行章檔，現行章檔與上面的凍結版本相同。
- 看圖：`outputs/course-review-20261019/figure-renders-04/<SVG 的 SHA-256>.png` 是凍結版 SVG 的算繪，用 `view_image` 看；也可以讀 SVG 原文。

## 獨立性

完成並存下這次的判斷以前，仍然不看其他角色的報告與以前的審閱紀錄（範圍同上一份），也不看 `outputs/`（上面的 PNG 除外）。課文或你上一份報告明確指名路徑的證據檔，即使位在這些目錄裡，也可以只讀那一個檔，並在報告註明。

## 寫什麼、寫在哪

- 只寫你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_20-12/` 與 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-20-12/`；新檔用新檔名（例如加 `-current-02` 字尾），不改之前的檔。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不跑長時間訓練。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_20-12/technical-20-12-current-02.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_20-12/technical-20-12-original-01.json", "sha256": "ae61d09277d0a8ab72472e37e44ffe15173ce66e7c2e0a234c9d51f47db9bc6c"}`；
  - `"resolved_issue_ids"`：解決了的上一份 issue 的 `claim_id`；
  - `source_freeze` 寫 `{"path": "docs/course-review-20261019/freeze-06.json", "sha256": "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"}`；
  - `reviewer_task` 寫 `/root/rv1019_technical_20-12`，`reviewer_context` 仍填 `fresh`；
  - 全部主張核實才寫 `verdict: pass`；否則 `revise`，在 `issues` 逐條寫（欄位同上一份）。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_technical_report.py 20.12 docs/course-review-20261019/reviews/rv1019_technical_20-12/technical-20-12-current-02.json --freeze docs/course-review-20261019/freeze-06.json --reader-task /root/rv1019_reader_ch20
  ```

## 回覆

- verdict；上一份每條 issue 解決與否，各一行。
- 報告與每個新 artifact 的路徑與 SHA-256；自我檢查的輸出。
- 你沒能驗證的範圍。
