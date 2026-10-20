# 發布閘門工具的第二次複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_review-tools`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-01.json`（verdict pass，SHA-256 `daa5f41067456bcd5b7f3dbf583ed0a63a1d2a12d66cf9c23fd8f5409f3cc2cb`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。之前的報告和它們的 artifact 都不要改。
- 你要判斷的仍是：工具照它的註解與設計做事、沒有讓不該通過的紀錄通過、測試真的守著每一條規則。

最終回覆用正體中文。

## 自上一份報告以來的變動

你上一份 `checked_files` 列的七個檔裡，只有 `docs/course-review-20261019/baseline-supplements.json` 變了：

- `84987d49`：同一個 commit 改了 `scripts/build_course.py`（Windows 上的索引路徑改寫成 `/`），並在補充還原清單新增一筆：把 `scripts/build_course.py` 換回 repo 內保存的副本，那份副本的 SHA-256 應等於舊輪次 freeze-01 實作表為這個檔記的值。
- 差異：`git diff cc6afc5d..HEAD -- docs/course-review-20261019/baseline-supplements.json`。

## 審什麼

判準仍是設計文件 `outputs/course-inline-code-20261017/pub/design/replay-supplements.md` 的 D-RP2〈讀取範圍〉，以及你上一份用的同一套判準。先讀那一節。

1. **新增的那一筆**：在新的乾淨 clone 核對：
   - 保存的副本在 repo 內、路徑與 `live_path` 對得上，SHA-256 等於舊輪次實作表所記的值；
   - 用現行清單跑重放，結果通過；
   - 只拿掉新增的這一筆再跑，重放回報 `scripts/build_course.py` 的 drift。
   - 重放可以自己寫小程式：用 `importlib` 載入 `docs/review-tools/baseline_replay.py`，把清單的 `files` 交給它的 `replay(root, supplements)`，印出 `status` 與 `failures`。
   - clone 放在 `outputs/course-review-20261019/tool-audit/` 底下的新子目錄，不要覆寫之前用過的 clone。用 `git clone --no-local` 從這個 repo 複製，設 `GIT_LFS_SKIP_SMUDGE=1`。
2. **沒改的六個檔**：`docs/review-tools/incremental_revision.py`、`docs/review-tools/baseline_replay.py`、`tests/test_incremental_revision_review.py`、`docs/review-tools/phase7_review.py`、`docs/course-revision-20261016/tools/reader.py`、`docs/course-revision-20261016/tools/freeze_sources.py`。核對現行 SHA-256 與你上一份 `checked_files` 記的相同，在 `claims` 記下。
3. **跑測試**：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_incremental_revision_review.py`，記下結果。

## 能看什麼、能做什麼

- repo 裡的程式、測試、紀錄都可以讀，也可以執行；可以連網查官方文件。
- 可以讀你自己之前的報告與 artifact，以及上面的設計文件。
- 完成並存下這次的判斷以前，不看其他人對這些工具的審閱報告與複審意見，例如 `outputs/course-inline-code-20261017/pub/` 底下的 `reports/`、`orders/`、`findings.tsv`、`design/consult-*.md`，以及 `docs/course-review-20261019/reviews/` 底下別人的目錄。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-review-tools/`，用新檔名（例如加 `-current-02` 字尾），不改之前的檔；
  - 上面的 clone 子目錄。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- artifact 只放小檔：你寫的程式、指令輸出、SHA 清單。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-02.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-01.json", "sha256": "daa5f41067456bcd5b7f3dbf583ed0a63a1d2a12d66cf9c23fd8f5409f3cc2cb"}`；
  - `"resolved_issue_ids": []`（上一份沒有未解決的 issue）；
  - `checked_files` 列同樣七個檔的現行 SHA-256。
- 三項核對都和預期一致、測試通過，才寫 `verdict: pass`；否則寫 `revise`，並在 `issues` 逐條寫明。

## 回覆

- verdict；`checked_files` 每個檔的 SHA-256。
- 新增那一筆的三項核對各一行：做法、結果、artifact。
- issues 各一行；沒有就說沒有，並說明你怎麼確認。
- 報告與每個 artifact 的路徑與 SHA-256。
- 你沒能驗證的範圍。
