# 交接：正文內嵌實作與試讀問題修訂

branch `course-inline-code-20261017`，從 `b05ebda3527789e4b6c550ad20271ababce251cd` 開始，沒有 push。目標與設計決策見[計畫](PLAN.md)。

## 合併到 `main` 之前要決定的事

這個 branch 現在推到 `main`，Publish course 會在第一道審閱檢查停下，網站不會更新。要發布，下面前兩項都要先完成；這兩項都是[計畫](PLAN.md)的 D8 留給作者決定的發布承諾，本輪沒有做。

1. **三道發布審閱檢查目前失敗。** `scripts/check_course_reviews.py`、`scripts/check_technical_reviews.py`、`docs/review-tools/check_review_round.py` 在 `b05ebda3` 都通過（退出碼 0），在 `827ebf74` 都失敗，第一個錯誤是 `current implementation drift: scripts/fetch_natural_data.py`。檢查遇到第一個錯誤就停，所以這只是其中一個：從 `docs/course-reviews-active.json` 可達的紀錄裡出現過 SHA-256 綁定、這個 branch 又改過的檔案有 425 個（上界，舊輪次紀錄裡的綁定也算進去），包括全部 23 個章檔、312 份 Notebook、68 張圖、入口頁、名詞頁、`scripts/export_course.py` 與 `scripts/fetch_natural_data.py`。要讓檢查通過，得依[逐節審閱標準](../editorial-guide.md)開新一輪審閱並更新凍結紀錄；不要為了通過而改寫既有紀錄或其中的 hash。
2. **閱讀時間要重估。** `course/reading-times.json` 每頁綁定正文與圖檔的 SHA-256。350 頁裡有 342 頁的正文改了（其中 66 頁的圖也改了），只有 `chapter-01`、`chapter-02` 兩頁章首導言、首頁與五頁技術參考沒變。`scripts/reading_time.py` 的 `load_estimates` 讀到任何一筆過期紀錄就報錯，不只在加了 `--require-reading-times` 時，所以 `scripts/export_course.py` 用預設的估時檔也建不出網站。本機預覽的做法見下方〈網站預覽〉。
3. **模型卡。** `docs/selftrained/v2-manifest.json` 的 `package.revision` 原本指向改寫歷史後已不存在的 `08761dac…`，改成同一個資料包 pointer 所在的 `4a229b85…`；清單的 SHA-256 因此從 `3443e3d3…` 變成 `3e50d133…`，`docs/selftrained/model-cards/repository-README.md` 跟著改。這份新卡沒有發布過：repo 裡最後一份卡片發布規格 `docs/selftrained/v2-jobs/release-repository-card-main-links.json` 綁的是改版前的卡與舊清單，而且 repo 裡沒有它的執行紀錄（更早的 `release-repository-card.json` 有）；線上的卡是哪一版，本輪沒有查。要換成新卡，得另寫一份綁新卡與新清單 SHA-256 的發布規格，再由作者手動觸發 workflow。
4. **Windows 指令沒有實跑。** 這台機器是 macOS；入口頁新寫或改寫的 PowerShell 指令只對照了 Microsoft 與 Python 的文件：`$env:GIT_LFS_SKIP_SMUDGE="1"`（README.md、README_en.md 與 W.1 的 clone 說明）、`.venv\Scripts\Activate.ps1`（README 兩檔與 W.1）、`.\.venv\Scripts\python.exe`（W.7、`course/training.md` 開頭）與 `..\.venv\Scripts\python.exe`（W.7）。

## 這個 branch 改了什麼

- **實作進正文**：52 段原樣節錄或精簡版，另有 10 個定義只在正文說明、不貼程式；位置見[配置表](PLACEMENT.md)。改版前沒有程式的 37 節裡，20 節補了程式，其餘 17 節維持文字。
- **守衛**：`tests/test_course_code_excerpts.py` 核對節錄與精簡版；`tests/test_course_commit_pins.py`、`tests/test_course_data_pins.py` 核對學生頁釘住的 commit 與資料包。各自涵蓋什麼、不涵蓋什麼，寫在檔頭。
- **頁尾**：Colab 與 Notebook 下載改成同級按鈕，說明程式與輸出都在頁面上（`scripts/export_course.py` 的 `section_footer`）。
- **用語**：依 [TERMS.md](TERMS.md) 統一全書用語，[名詞頁](../../course/glossary.md)照它補齊並加英文與縮寫索引。
- **試讀意見**：19 份模擬試讀報告（AI subagent，不是真人）的意見由各章作者逐條處理。
- **入口頁**：README、閱讀指南、暖身指南與訓練操作改成「先當書讀，想自己改再開 Notebook」；clone 一律跳過資料包下載，要用資料包時再照 W.1 安裝 Git LFS。

## 驗收

下表在 `827ebf74` 跑，只有以 kernel 執行 Notebook 那一列在 `02608a1a` 跑；兩者之間只有 `tests/test_course_data_pins.py` 多一個測試案例。環境是 macOS（Apple silicon，沒有 NVIDIA GPU）、Python 3.13.16，套件版本與 `uv.lock` 相同（torch 2.14.1、ruff 0.16.9）：

| 指令 | 結果 |
| --- | --- |
| `python scripts/check_lesson_order.py` | 通過 |
| `ruff check .`、`ruff format --check .` | 通過 |
| `python scripts/build_course.py --check` | 通過 |
| `python scripts/check_notebooks.py` | 312 份 Notebook 全部執行成功 |
| `python scripts/check_notebooks.py --mode kernel --workers 1` | 312 份 Notebook 全部執行成功（483 秒） |
| `python -m pytest` | 1 failed、1854 passed、3 skipped |
| 三道發布審閱檢查 | 失敗，見上面第 1 項 |

全套 pytest 唯一的失敗是 `tests/test_selftrained_data_publish.py::test_missing_pointer_bootstrap_uploads_native_local_lfs_object_without_commit`，`b05ebda3` 也失敗（計畫的「修訂前基準」）。這台機器有 git-lfs 3.8.0；失敗的原因是本機 `~/.gitconfig` 的 `init.templateDir` 指向一個含 `post-checkout` hook 的範本目錄，測試新建的暫存 repo 複製了這個 hook，`git lfs install --local` 因此回報 `Hook already exists: post-checkout`。

## 網站預覽

本機在 `02608a1a` 建置過。閱讀時間紀錄已過期，所以匯出時把 `--reading-times` 指向不存在的檔案：這樣各頁不顯示估時（首頁仍有說明估時方式的段落），其餘照發布流程核對。

```bash
python scripts/check_notebooks.py --mode kernel --workers 1 --output outputs/notebooks
python scripts/export_course.py --executed outputs/notebooks --reading-times outputs/no-reading-times.json --revision "$(git rev-parse HEAD)"
python scripts/check_site.py
python -m http.server 8000 --directory outputs/site
```

第一行要先照[發布說明](../publishing.md)註冊 `tiny-perceptron` kernel。前三行在 `02608a1a` 跑過，只是 Notebook 輸出與估時檔放在別的路徑：312 份 Notebook 全部執行成功，匯出成功，`check_site.py` 回報 351 頁、312 份下載與 Colab 入口、135095 個內部連結、1344 個 Notebook 閱讀連結通過。

## 審閱

每階段 commit 後，由沒參與撰寫的 subagent 審閱：技術審閱者實跑程式、核對節錄與數字；初學讀者回報看不懂的地方。主 agent 逐條裁決，修正後只複查修正的 commit，直到只剩文字類問題；每個階段結束前另由不同模型複審主 agent 的裁決。

下表是各輪接受的意見條數（第一輪沒有區分邏輯與文字；「初學讀者必改」是初學讀者意見裡主 agent 判定必須改的條目）。邏輯類指讀者照著讀會得到錯的事實或做錯事，文字類指意思沒錯、用詞可以更好。

| 階段 | 範圍 | 第一輪 | 第二輪 | 第三輪 | 結果 |
| --- | --- | --- | --- | --- | --- |
| P1 | 測試、釘選與頁尾 | 接受 14 | 邏輯 6、文字 3 | 邏輯 4、文字 4 | 第三輪之後的修正由複審者重跑突變並審 diff，收斂 |
| P2 | 第 1–7 章、單元 C | 接受 36，駁回 1 | 技術 8、初學讀者必改 5 | 邏輯 0、文字 2 | 收斂 |
| P3 | 第 8–13 章 | 接受 38 | 技術 7、初學讀者必改 1 | 邏輯 0、文字 4 | 收斂 |
| P4 | 第 14–20 章、單元 A、B | 接受 51 | 技術 6、初學讀者必改 6 | 邏輯 2、文字 4 | 收斂 |
| P5 | 入口頁 | 接受 8 | 邏輯 2、文字 4 | 邏輯 0、文字 1 | 收斂 |
| P6 | 名詞頁 | 邏輯 6、文字 8 | 邏輯 3、文字 3 | — | 收斂 |

第二輪複查發現，修正時新寫的句子常常本身又錯：沒有對照它提到的程式、紀錄或其他節，或同一個問題只改了被點名的那一處。之後的修正要求每句新寫的事實附收據（指令輸出或原文）、泛化句先試反例、做同型掃描，同一句在本分支改過兩次以上就只能刪或縮小；第三輪複查時，章節修正帶進的新錯只剩第 19 章的 2 條。名詞頁第二輪的 3 條邏輯類出自主 agent 自己寫的修正句，所以那一輪只做刪除與縮小。

審閱報告與裁決表留在本機暫存區，沒有進 Git；它們不是 D8 所說的正式審閱輪紀錄，不能拿來通過發布檢查。

## 已知、本輪沒有處理的事

- `course/figures/index.json` 沒有跟著圖更新：這個 branch 改了 4 張手寫圖的 `<title>`（`flash_allocated_memory`、`foundations_shift`、`multimodal_overlapping_bands`、`window_training`），索引還是舊標題；203 張 SVG 裡有 113 張在改版前就不在索引裡。沒有頁面、腳本或測試讀這份索引。能重建它的 `scripts/build_visuals.py` 在 `b05ebda3` 就會在 `looped-mechanism-measured.svg`（沒有 `<title>` 與 `<desc>`）中止，而且中止前已改寫自動產生的 `frames.svg`、`normalization.svg`。
- `docs/` 的發布與查核紀錄仍寫舊清單的 SHA-256 `3443e3d3…`：它們記的是當時的清單，沒有改。拿這些紀錄核對現在的清單會不相符，原因是上面第 3 項的 `package.revision`。
