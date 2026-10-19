# 發布閘門工具的複查（正式審閱，原審閱者複查）

你是 `/root/rv1019_technical_review-tools`，延續你上一份報告 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-original-01.json`（verdict revise，issues `READ-BOUNDARY-1`、`READ-BOUNDARY-2`）。

這是正式複查：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫。上一份報告和它的 artifact 都不要改。
- 你要判斷的仍是：工具照它的註解與設計做事、沒有讓不該通過的紀錄通過、測試真的守著每一條規則。

最終回覆用正體中文。

## 自上一份報告以來的變動

- `4ce780ba`：兩份既有清單（`FREEZE`、`OVERLAY`）改經 `repo_file` 讀；建立暫存目錄以前，先由 `checked_tracked_names` 核對每個追蹤檔名稱的上層。
- `36534cce`：`baseline_replay.py` 只改註解（`checked_tracked_names` 的「驗證」「不涵蓋」，以及載入 `composite_review.py` 的地方）；測試新增 `test_tracked_symlink_alone_in_a_redirected_folder_is_rejected_by_its_parent_check`，並把從 `links/` 指回 `AGENTS.md` 的相對 symlink target 改用平台自己的分隔符。
- 差異：`git diff cc6afc5d..36534cce -- docs/review-tools/baseline_replay.py tests/test_incremental_revision_review.py`。

## 判準的更正

上一份提示詞〈審什麼〉第 1 點要你判斷「有沒有任何輸入能讓重放寫到暫存樹以外、讀到 repo 以外，或改到硬連結的原檔」。其中「讀到 repo 以外」被寫成一條通則，但設計 D-RP2 沒有這條通則，這項要求撤回。「寫到暫存樹以外」「改到硬連結的原檔」仍是設計承諾的還原寫入限制。

這次的判準是設計文件 `outputs/course-inline-code-20261017/pub/design/replay-supplements.md` 裡 D-RP2 的〈讀取範圍〉一節，包括其中「工具審閱的判準」那一段。先讀那一節。重點：

- 前提：驗證程式可信、只讀，路徑不被並行置換。
- 承諾的讀取（兩份既有清單與三份清單逐筆的 `path`、舊身分引用鏈、每個追蹤檔的上層）沒有核到，或違反舊身分、SHA、還原寫入限制：necessary issue。
- 列為不保證的讀取：不因讀到外部合法內容、重放照樣成功，就列為 necessary。用真實驗證器重現一筆應被拒絕的紀錄被接受時，列為 issue，並寫明要重開範圍還是設計。
- 重現前提（diff、Git 狀態、checkout 條件、外部檔案從哪裡來）都要記錄。替身驗證器只能證明接線或讀取行為，不能單獨證明真實閘門誤放行。

判斷上一份的兩條 issue 是否已解決，也以〈讀取範圍〉承諾的範圍為準。

## 審什麼

1. **上一份的兩條 issue**：用你上一份的探針（可以改寫）在新的乾淨 clone 重跑，確認現在都在任何 link／copy 與暫存樹建立以前就被拒絕。確認解決的，列進 `resolved_issue_ids`；沒解決或只解決一部分的不列，寫進 `issues`。
2. **改過的兩個檔**：讀上面的 diff 與兩個檔的現行全文，對照〈讀取範圍〉判斷：
   - 承諾的每一種讀取是否都經過核對，有沒有漏掉的讀取路徑；
   - 新註解的每一句是否成立；
   - 新測試抓不抓得到「葉節點是 symlink 就跳過上層檢查」這種錯法。在 clone 裡的副本自己打進這個錯法跑測試，不要改主工作樹。
3. **沒改的五個檔**：`docs/review-tools/incremental_revision.py`、`docs/course-review-20261019/baseline-supplements.json`、`docs/review-tools/phase7_review.py`、`docs/course-revision-20261016/tools/reader.py`、`docs/course-revision-20261016/tools/freeze_sources.py`。核對現行 SHA-256 與你上一份 `checked_files` 記的相同，在 `claims` 記下。
4. **真實重放對照**：在新的乾淨 clone，對現行版本重做上一份的五種情況，以及重放前後全部追蹤檔的 SHA-256 與 mode 核對。
   - clone 放在 `outputs/course-review-20261019/tool-audit/` 底下的新子目錄，不要覆寫上一份用過的 clone。用 `git clone --no-local` 從這個 repo 複製，設 `GIT_LFS_SKIP_SMUDGE=1`。
5. **跑測試**：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_incremental_revision_review.py`，記下結果。

## 能看什麼、能做什麼

- repo 裡的程式、測試、紀錄都可以讀，也可以執行；可以連網查官方文件。
- 可以讀你自己的上一份報告與 artifact，以及上面的設計文件。
- 完成並存下這次的判斷以前，不看其他人對這些工具的審閱報告與複審意見，例如 `outputs/course-inline-code-20261017/pub/` 底下的 `reports/`、`orders/`、`findings.tsv`、`design/consult-*.md`，以及 `docs/course-review-20261019/reviews/` 底下別人的目錄。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-review-tools/`，用新檔名（例如加 `-current` 字尾），不改上一份的檔；
  - 上面的 clone 子目錄。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow。寫檔用不覆寫的方式（例如 `open(path, "x")`）。
- artifact 只放小檔：你寫的程式、指令輸出、SHA 清單。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-current-01.json`。欄位與上一份相同，另外：
  - `"initial_context": "recheck by the original reviewer"`；
  - `"original_report": {"path": "docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-original-01.json", "sha256": "0c989ef2ae525e2c71dae26e63bd60678c55a59db6d8d1ccfa1277b9b2fbd613"}`；
  - `"resolved_issue_ids": [<確認已解決的上一份 issue id>]`；
  - `checked_files` 列上一份同樣七個檔的現行 SHA-256。
- 照〈讀取範圍〉的判準全部核實、五種對照都和預期一致、上一份兩條都已解決，才寫 `verdict: pass`；否則寫 `revise`。

## 回覆

- verdict；`checked_files` 每個檔的 SHA-256；`resolved_issue_ids`。
- 上一份兩條 issue 各一行：怎麼重跑、結果、artifact。
- 五種對照各一行：預期、實際、artifact。
- issues 各一行；沒有就說沒有，並說明你怎麼確認。
- 報告與每個 artifact 的路徑與 SHA-256。
- 你沒能驗證的範圍。
