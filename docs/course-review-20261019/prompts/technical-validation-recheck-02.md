# validation 事實與技術審閱的複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_validation`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-current-01.json`（verdict pass，SHA-256 `5f29a0d97d534a42532749af11667dfba86f305f08cf03d86d17476ce5ceff2f`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告與 artifact 都不要改。
- 判準和上一份相同：頁面寫出的每一個實質主張是否成立、數字是否算對、圖是否表達同一件事，以及引述的來源與實驗能支持多大的結論。

最終回覆用正體中文。

## 自上一份報告以來的變動

- 你上一份以 SHA-256 引用的 repo 檔有變動：`docs/technical-review-guide.md`（你記的 `142b3c065a89963a036cec470f31299eaa39dc2ff67f5417317703bb4e7b10bd`，現行 `b39bfc9a08098ad7b71446c34a548e57f7ba251d75bc7012f4a600ac0faef344`）。差異用 `git diff 6a683dc4d249 41651edb -- docs/technical-review-guide.md` 看，確認你依它核對的主張仍成立，報告記現行 SHA-256。
- 上一次複查開放讀的證據，這次仍然可以讀，規則同上一次；報告引用時把讀到的欄位摘錄與 SHA-256 存進 artifact 目錄：
  - c03：Notebook 環境是 `/private/tmp/claude-501/-Users-yclin-project-tiny-perceptron-vlm/24cef3c9-71fc-4407-956a-34014b925580/scratchpad/venv-nb`（裝了 notebook 依賴群組，已註冊 `tiny-perceptron` kernel）。可以自己跑 `PYTHONPATH=. /private/tmp/claude-501/-Users-yclin-project-tiny-perceptron-vlm/24cef3c9-71fc-4407-956a-34014b925580/scratchpad/venv-nb/bin/python scripts/check_notebooks.py --mode kernel --workers 1 --lesson <小節> --output <你的 artifact 目錄>/notebooks`，輸出只寫到你的 artifact 目錄。全書的執行結果在 `outputs/course-review-20261019/executed-05/validation-kernel.json`，這個檔也可以讀；`outputs/` 底下其他檔仍然不看。
  - c09、c52、c53：`docs/selftrained/results/training-raw/` 底下每一段訓練的 `receipt.json` 與 `raw/` 裡的 `train-receipt.json`、`execution.json`，以及 `docs/selftrained/results/v2-final-training-source-selection.json`、`docs/selftrained/results/v2-dense-public-final-test-stage-index.json`、`docs/selftrained/results/v2-moe-public-final-test-stage-index.json`。它們旁邊檔名含 `integrity-review` 的檔是以前的審閱紀錄，不看。
  - c45、c46：新版頁面把這兩句改成指向 `docs/lesson-products/results/final-report-v1.json`、`publication-package-v1.json`（與原本就連的 `settings-freeze-v1.json`），以及兩位審核者的重算紀錄 `docs/lesson-products/reviews/engineering-final-b-mechanisms-original.json`、`engineering-final-c-original.json`。c46 講的就是這兩份紀錄，所以這次可以讀它們與同名的 `-execution.json`，只拿來核對這一句；`docs/lesson-products/reviews/` 底下其他檔不看。`git log` 可以用來核對這些檔提交的先後。

## 審什麼

- 頁面：`validation`，原文是 `docs/validation.md`（整份檔）。凍結版本在 `docs/course-review-20261019/freeze-05/sources/validation.md`，SHA-256 `3d4456fda1dc81edb0ac2b4a7314321a4a3cbd98914a99fbc7db5af16bd79b9c`；報告的 `source_sha256` 必須等於它。
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

- 只寫你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_validation/` 與 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-validation/`；新檔用新檔名（例如加 `-current-01` 字尾），不改之前的檔。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow、不跑長時間訓練。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-current-02.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-original-01.json", "sha256": "9e3ed2f5b3ce98f6f72494b041f2996e4a427ead7296a8f681f33db2d29df072"}`；
  - `"resolved_issue_ids"`：解決了的上一份 issue 的 `claim_id`；
  - `source_freeze` 寫 `{"path": "docs/course-review-20261019/freeze-06.json", "sha256": "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"}`；
  - `reviewer_task` 寫 `/root/rv1019_technical_validation`，`reviewer_context` 仍填 `fresh`；
  - 全部主張核實才寫 `verdict: pass`；否則 `revise`，在 `issues` 逐條寫（欄位同上一份）。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_technical_report.py validation docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-current-02.json --freeze docs/course-review-20261019/freeze-06.json --reader-task /root/rv1019_reader_ref
  ```

## 回覆

- verdict；上一份每條 issue 解決與否，各一行。
- 改過的句子各一行：成立與否、依據。
- 報告與每個新 artifact 的路徑與 SHA-256；自我檢查的輸出。
- 你沒能驗證的範圍。
