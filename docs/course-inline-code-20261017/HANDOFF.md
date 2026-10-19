# 交接：正文內嵌實作與試讀問題修訂

branch `course-inline-code-20261017`，從 `b05ebda3527789e4b6c550ad20271ababce251cd` 開始。目標與設計決策見[計畫](PLAN.md)。

## 合併到 `main` 之前要決定的事

這個 branch 現在推到 `main`，Publish course 會在第一道審閱檢查停下，網站不會更新。要發布，下面前兩項都要先完成；這兩項都是[計畫](PLAN.md)的 D8 留給作者決定的發布承諾，本輪沒有做。

1. **三道發布審閱檢查目前失敗，要等新一輪審閱生效。** `scripts/check_course_reviews.py`、`scripts/check_technical_reviews.py`、`docs/review-tools/check_review_round.py` 在 `b05ebda3` 都通過（退出碼 0），在 `827ebf74` 都失敗，第一個錯誤是 `preserved baseline failed: ['current implementation drift: scripts/fetch_natural_data.py']`；從 `84987d49` 起，第一個錯誤是 `preserved baseline failed: ['current implementation drift: scripts/build_course.py']`，因為那個 commit 也改了 `scripts/build_course.py`，而它在重放基準的實作雜湊表裡排在前面。
   - **第一層是重放。** 現行審閱工具 `docs/review-tools/incremental_revision.py` 的 `audit_manifest` 讀進 manifest、確認政策之後，先呼叫 `docs/review-tools/baseline_replay.py` 的 `replay` 重放 2026-10-08 那一輪；manifest 有 `baseline_supplements` 時，一併傳入它指定的補充還原清單。重放在暫存副本裡把 `docs/course-publish-20261008/current-source-freeze.json`、`docs/course-revision-20261016/baseline-overlay.json` 與補充還原清單（合稱還原清單）列出的檔案換回當時的版本，再用 `docs/review-tools/composite_review.py` 核對那一輪，其中實作與技術輸入逐檔比對當時記下的雜湊。這個 branch 改過的 `scripts/build_course.py`、`scripts/fetch_natural_data.py` 與 `docs/selftrained/v2-manifest.json` 都不在現行生效 manifest 的還原清單裡，所以重放失敗；三個都換回重放基準記的版本，重放才通過。
   - **第二層才是現行輪次的凍結。** 在暫存 clone 把三個都換回後，`scripts/check_course_reviews.py` 改報 `frozen figure source mismatch`：有圖檔和 `docs/course-reviews-active.json` 指向的那一輪凍結的版本不同。從 `docs/course-reviews-active.json` 可達的紀錄裡出現過 SHA-256 綁定、這個 branch 又改過的檔案，包括全部 23 個章檔、312 份 Notebook、多張圖、入口頁、名詞頁、`scripts/export_course.py` 與 `scripts/build_course.py`；這一層要靠新一輪審閱與新的凍結紀錄。
   - **`scripts/fetch_natural_data.py` 與 `docs/selftrained/v2-manifest.json` 不能改回去。** 它們原本釘的 commit `21a24124…`、`08761dac…` 已不在這個 repo 裡；換回後，`tests/test_course_commit_pins.py` 與 `tests/test_course_data_pins.py` 共 6 個測試失敗。
   - **重放的做法。** 改了 `baseline_replay.py` 與 `incremental_revision.py`：新一輪的 manifest 用 `baseline_supplements` 指向自己的補充還原清單，本輪是 `docs/course-review-20261019/baseline-supplements.json`，三筆就是上面那三個檔，來源是 repo 內保存、SHA-256 等於重放基準所記的副本。每筆的目的路徑必須是重放基準雜湊表（`implementation_sha256`、`files_sha256`）的 key，SHA-256 必須等於表上的值；清單本身也要列進新一輪工具審閱的 `checked_files`。沒有改寫既有紀錄（`docs/course-revision-20261016/baseline-overlay.json` 由現行 manifest 以 SHA-256 綁住），也沒有換審閱政策。這兩個工具檔在工具審閱的清單裡，所以新一輪要有新的獨立工具審閱。新一輪的 manifest 生效前，三道檢查仍以上面那則 `current implementation drift` 錯誤失敗。
   - **留在 `main` 的目錄。** 只要還用現行工具，`docs/course-publish-20261008/` 與 `docs/course-revision-20261016/` 就得留在 `main`：`baseline_replay.py` 與 `incremental_revision.py` 寫死讀這兩個目錄裡的檔，在暫存 clone 移走任一個，`scripts/check_course_reviews.py` 就報找不到檔案。新一輪的補充還原清單另外讀 `docs/course-repair-20261008/reviews/freeze-01/freeze/` 底下的兩個副本，它們也要留在原路徑。
   - **怎麼開一輪。** 審閱本身照 AGENTS.md 走[逐節審閱標準](../editorial-guide.md)；那份文件沒有寫 manifest、凍結紀錄、重放或 `docs/course-reviews-active.json` 怎麼操作，`docs/review-tools/README.md` 也沒有。現行工具要什麼，寫在工具程式與輪次紀錄裡：`incremental_revision.py` 的 `audit_manifest`、`baseline_replay.py`、產生新凍結紀錄的 `docs/course-revision-20261016/tools/freeze_sources.py`、2026-10-16 那一輪的證據契約 `docs/course-revision-20261016/REVIEW-CONTRACT.md`，以及現行輪次的 `docs/course-repair-20261017/manifest-final-01.json`。
2. **閱讀時間要重估。** `course/reading-times.json` 每頁綁定正文與圖檔的 SHA-256。350 頁裡有 344 頁的正文改了，只有首頁與五頁技術參考沒變。`scripts/reading_time.py` 的 `load_estimates` 讀到任何一筆過期紀錄就報錯，不只在加了 `--require-reading-times` 時，所以 `scripts/export_course.py` 用預設的估時檔也建不出網站。本機預覽的做法見下方〈網站預覽〉。
3. **模型卡。** `docs/selftrained/v2-manifest.json`（第 1 項擋住重放的兩個檔之一）的 `package.revision` 原本指向改寫歷史後已不存在的 `08761dac…`，改成同一個資料包 pointer 所在的 `4a229b85…`；清單的 SHA-256 因此從 `3443e3d3…` 變成 `3e50d133…`，`docs/selftrained/model-cards/repository-README.md` 跟著改。這份新卡沒有發布過：repo 裡最後一份卡片發布規格 `docs/selftrained/v2-jobs/release-repository-card-main-links.json` 綁的是改版前的卡與舊清單，而且 repo 裡沒有它的執行紀錄（更早的 `release-repository-card.json` 有）；線上的卡是哪一版、Hugging Face 上 main 目前是哪個 commit，本輪都沒有查。要換成新卡，得另寫一份綁新卡與新清單 SHA-256 的發布規格，其中的 `parent_commit` 要等於 Hugging Face 上 main 當下的 commit：執行 `stage` 為 `release`、`mode` 為 `repository-card` 的規格時，`scripts/selftrained/hf_transport.py` 的 `repository_card_public_gate` 讀到的 main 不等於 `parent_commit` 就拒絕發布，所以這個值要作者發布前自己查。規格提交後，由作者手動觸發 `.github/workflows/selftrained-assistant.yml`，它的輸入是 `operation`、`manifest`、`job`、`batch_id`。`release-repository-card.json` 那次的執行紀錄在 `docs/selftrained/infrastructure/v2-repository-card-publication/actual-card-publication/`：`actual-dispatch.json:4` 的 `job` 是 `docs/selftrained/v2-jobs/release-repository-card.json`，`actual-job/selftrained-run-5aa019d50a940deae8f8f1cf84bd9b11fed47121-37479108854-1/gha-37479108854-1/receipt.json:3` 的 `batch_id` 是 `selftrained-v2`。`operation` 要選 `run`：workflow 的檢查、預約、執行三步都只在 `run` 時跑，預設的 `ledger` 只讀帳本，卡片不會發布。`manifest` 要填 `docs/selftrained/v2-manifest.json`：`scripts/selftrained/modal_runner.py` 從 `--manifest` 指的檔算出 SHA-256，`scripts/selftrained/hf_transport.py` 的 `approved_repository_card` 拿它和規格的 `manifest_sha256` 比，不同就以 `Repository card differs from the exact frozen manifest` 拒絕；預設的 `docs/selftrained/manifest.json` 是 v1 的清單。上一次發卡時，同一份 `receipt.json:6` 記的 `manifest_sha256` 正是上面說的舊清單 SHA-256 `3443e3d3…`。
4. **Windows 指令由 CI 實跑；學生電腦有幾種情況仍未驗證。** [`.github/workflows/windows-student-commands.yml`](../../.github/workflows/windows-student-commands.yml) 在 GitHub 的 `windows-latest` 上，用 Windows PowerShell 5.1 與 PowerShell 7 各跑一次 README.md、README_en.md、`course/first-steps.md`、`course/training.md` 要 Windows 讀者在 PowerShell 輸入的命令：clone（先執行 `git config --global core.longpaths true`，再執行 `$env:GIT_LFS_SKIP_SMUDGE="1"` 與 `git clone …`）、`uv sync`、`.venv\Scripts\Activate.ps1` 與不啟用的 `.\.venv\Scripts\python.exe` 寫法、`check_env.py`、kernel 登記、啟動 JupyterLab、W.7 的路徑練習（含 `..\.venv\Scripts\python.exe`），以及訓練頁的本機練習與 `--device cpu` 的 `simple_models`、`posttraining` 固定實驗；刻意不跑的命令寫在 workflow 的 `Skip-Page`，記錄裡印出簡短理由，完整理由在呼叫處的註解。頁面以外，它用登記好的 `tiny-perceptron` kernel 以 nbclient 從頭執行照 W.1 改字練習改了一行的 `01/1.1.ipynb`，核對最後的文字輸出等於頁面寫的句子；也執行接在第一個程式格之後的 W.6 程式，核對它的第一行輸出。`tests/test_windows_student_commands.py` 把 workflow 的命令和頁面逐條釘在一起。仍未驗證的有：學生電腦的 `LongPathsEnabled` 登錄值、使用者資料夾含中文、沒有預裝 Python（runner 預裝了，uv 直接用它建 `.venv`）、沒裝 Git LFS、Windows PowerShell 5.1 解析 `NaN` 的行為、中文 Windows（cp950）特有的檔案編碼失敗（runner 是英文 Windows），以及瀏覽器裡的操作（選 kernel、點進程式格按 Shift+Enter、選單的 Restart Kernel and Run All Cells）。頁面要讀者核對、CI 不核的其他結果，逐項列在 `tests/test_windows_student_commands.py` 開頭的〈不涵蓋〉。

## 這個 branch 改了什麼

- **實作進正文**：55 段原樣節錄或精簡版，另有 17 個定義只在正文說明、不貼程式；位置見[配置表](PLACEMENT.md)。改版前沒有 Python 程式的 37 節裡，26 節補了 Python 程式，其餘 11 節仍沒有。
- **守衛**：`tests/test_course_code_excerpts.py` 核對節錄與精簡版；`tests/test_course_commit_pins.py`、`tests/test_course_data_pins.py` 核對學生頁釘住的 commit 與資料包。
- **頁尾**：Colab 與 Notebook 下載改成同級按鈕，說明程式與輸出都在頁面上（`scripts/export_course.py` 的 `section_footer`）。
- **用語**：依 [TERMS.md](TERMS.md) 統一全書用語，[名詞頁](../../course/glossary.md)照它補齊並加英文與縮寫索引。
- **試讀意見**：19 份模擬試讀報告（AI subagent，不是真人）的意見由各章作者逐條處理。
- **入口頁**：README、閱讀指南與暖身指南改成「先當書讀，想自己改再開 Notebook」；clone 一律跳過資料包下載，要用資料包時再照 W.1 安裝 Git LFS。

## 驗收

下表全部在 `761abe0f` 跑。環境是 macOS（Apple silicon，沒有 NVIDIA GPU）、Python 3.13.16，套件版本與 `uv.lock` 相同（torch 2.14.1、ruff 0.16.9）：

| 指令 | 結果 |
| --- | --- |
| `python scripts/check_lesson_order.py` | 通過 |
| `ruff check .`、`ruff format --check .` | 通過 |
| `python scripts/build_course.py --check` | 通過 |
| `python scripts/check_notebooks.py` | 312 份 Notebook 全部執行成功 |
| `python scripts/check_notebooks.py --mode kernel --workers 1` | 312 份 Notebook 全部執行成功（480 秒） |
| `python -m pytest` | 1 failed、1854 passed、3 skipped |
| 三道發布審閱檢查 | 失敗，見上面第 1 項 |

全套 pytest 唯一的失敗是 `tests/test_selftrained_data_publish.py::test_missing_pointer_bootstrap_uploads_native_local_lfs_object_without_commit`，`b05ebda3` 也失敗（計畫的「修訂前基準」）。這台機器有 git-lfs 3.8.0；失敗的原因是本機 `~/.gitconfig` 的 `init.templateDir` 指向一個含 `post-checkout` hook 的範本目錄，測試新建的暫存 repo 複製了這個 hook，`git lfs install --local` 因此回報 `Hook already exists: post-checkout`。

## 網站預覽

本機在 `761abe0f` 建置過。閱讀時間紀錄已過期，所以匯出時把 `--reading-times` 指向不存在的檔案：這樣各頁不顯示估時，其餘照發布流程核對。

```bash
python scripts/check_notebooks.py --mode kernel --workers 1 --output outputs/notebooks
python scripts/export_course.py --executed outputs/notebooks --reading-times outputs/no-reading-times.json --revision "$(git rev-parse HEAD)"
python scripts/check_site.py
python -m http.server 8000 --directory outputs/site
```

第一行要先照[發布說明](../publishing.md)註冊 `tiny-perceptron` kernel。前三行在 `761abe0f` 跑過，只是 Notebook 輸出與估時檔放在別的路徑：312 份 Notebook 全部執行成功，匯出成功，`check_site.py` 回報 351 頁、312 份下載與 Colab 入口、135110 個內部連結、1347 個 Notebook 閱讀連結通過。

## 審閱

每階段 commit 後，由沒參與撰寫的 subagent 審閱：技術審閱者實跑程式、核對節錄與數字；P2–P4 的第二輪與 P5 的第一輪（階段見下表）另有初學讀者試讀指定的小節，回報看不懂的地方。主 agent 逐條裁決，修正後只複查修正的 commit；每個階段結束前另由不同模型複審主 agent 的裁決。

下表是各輪接受的意見條數，從裁決表數：判定接受的才算，一列記多條時照列內寫的條數，登記不改的不算（P1–P5 的第一輪沒有區分邏輯與文字；「初學讀者必改」是初學讀者意見裡主 agent 判定必須改的條目）。邏輯類指讀者照著讀會得到錯的事實或做錯事，文字類指意思沒錯、用詞可以更好，目標缺口指違反 D1 或 D5 的決定（實作沒放在第一次教的節、該補程式的節沒補）。

| 階段 | 範圍 | 第一輪 | 第二輪 | 第三輪 | 結果 |
| --- | --- | --- | --- | --- | --- |
| P1 | 測試、釘選與頁尾 | 接受 14 | 邏輯 6、文字 3 | 邏輯 4、文字 4 | 第三輪之後的修正由複審者重跑突變並審 diff，收斂 |
| P2 | 第 1–7 章、單元 C | 接受 36，駁回 1 | 技術 8、初學讀者必改 5 | 邏輯 0、文字 2 | 收斂 |
| P3 | 第 8–13 章 | 接受 38 | 技術 7、初學讀者必改 1 | 邏輯 0、文字 4 | 收斂 |
| P4 | 第 14–20 章、單元 A、B | 接受 51 | 技術 6、初學讀者必改 6 | 邏輯 2、文字 4 | 最後一輪的邏輯類由複審者核對修正 commit，收斂 |
| P5 | 入口頁 | 接受 8 | 邏輯 2、文字 4 | 邏輯 0、文字 1 | 收斂 |
| P6 | 名詞頁 | 邏輯 6、文字 8 | 邏輯 3、文字 1 | — | 最後一輪的邏輯類由複審者核對修正 commit，收斂 |
| P7 | 交接文件與配置表 | 邏輯 5、文字 6 | 邏輯 1、文字 2 | — | 最後一輪的邏輯類由複審者核對修正 commit，收斂 |
| 總體審查 | 整個 branch（第二輪起章檔、文件分開複查修正） | 邏輯 9、目標缺口 6、文字 32 | 章檔邏輯 2；文件邏輯 5、文字 3 | 章檔 0；文件邏輯 2、文字 0 | 文件第四輪：邏輯 0，另 1 條邏輯類登記不改（守衛是 `tests/test_course_commit_pins.py`）；複審者逐列讀裁決表、自己重算跨檔數字、讀補程式與邏輯修正的 diff、實跑新補的程式，收斂 |

第二輪複查發現，修正時新寫的句子常常本身又錯：沒有對照它提到的程式、紀錄或其他節，或同一個問題只改了被點名的那一處。之後的修正要求每句新寫的事實附收據（指令輸出或原文）、泛化句先試反例、做同型掃描，同一句在本分支改過兩次以上就只能刪或縮小；第三輪複查時，章節修正帶進的新錯只剩第 19 章的 2 條。名詞頁第二輪的 3 條邏輯類出自主 agent 自己寫的修正句，所以那一輪只做刪除與縮小。

各階段之後另有總體審查（整個 branch 的跨檔審閱）。它的第一輪讀的是 `a1183e0a`，其中全書審閱者沒有逐筆讀的部分包括：本 branch 新增行裡指向小節的 1,842 筆引用，腳本篩檢後放行 1,560 筆，除了隨機抽讀的 30 筆與帶強主張（「定義」「方法見」這類字）的引用，其餘約 1,470 筆沒讀；篩出的 282 筆裡，沿用改版前寫法的 49 筆沒讀；當時節錄標記與配置表列出的 70 個定義，在展示節之後各節程式區塊裡的 336 行 import 與呼叫沒有逐行讀。

審閱報告與裁決表留在本機暫存區，沒有進 Git；它們不是 D8 所說的正式審閱輪紀錄，不能拿來通過發布檢查。

## 已知、本輪沒有處理的事

- `course/figures/index.json` 沒有跟著圖更新：這個 branch 改了 19 張手寫圖的 `<title>`，其中在索引裡的 4 張（`flash_allocated_memory`、`foundations_shift`、`multimodal_overlapping_bands`、`window_training`）索引還是舊標題。手寫圖在索引裡的 caption 取自 `<desc>`（`scripts/build_visuals.py` 的手寫圖迴圈）；索引裡 41 筆手寫圖，有 6 筆的 caption 和現在的 `<desc>` 不同，其中 2 筆在 `b05ebda3` 就已不同。203 張 SVG 裡有 113 張在改版前就不在索引裡。沒有頁面、腳本或測試讀這份索引。能重建它的 `scripts/build_visuals.py` 在 `b05ebda3` 就會在 `looped-mechanism-measured.svg`（沒有 `<title>` 與 `<desc>`）中止，而且中止前已改寫自動產生的 `frames.svg`、`normalization.svg`。README.md〈專案結構與維護〉與 README_en.md〈Maintenance〉列的維護指令，第一條就是 `python scripts/build_visuals.py`，照著做會在同一處失敗。
- `docs/` 的發布與查核紀錄仍寫舊清單的 SHA-256 `3443e3d3…`：它們記的是當時的清單，沒有改。拿這些紀錄核對現在的清單會不相符，原因是上面第 3 項的 `package.revision`。
