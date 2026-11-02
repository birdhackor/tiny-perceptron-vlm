# AGENTS.md

給所有 coding agent（Codex、Claude Code 等）與維護者共用的專案原則與 repo 維護準則。`CLAUDE.md` 是指向本檔案的 symlink。

## 專案概要

小小感知機（Tiny Perceptron）：用最少的程式碼，從零實作小型多模態模型的教學 repo。
第一階段是「文字 + 影像 + 音訊 → 文字」，之後擴充到「影片 + 音訊 → 文字 / 影像」。
讀者以中文使用者為主，英文使用者為輔；不預設讀者懂模型或本專案。Python、tensor 與數學工具可按需回查暖身材料。

## 常用指令

```bash
uv sync                                  # 安裝（PyPI 預設版 torch）
uv sync --extra cpu                      # 或 cu126 / cu130，見 pyproject.toml
uv run python scripts/check_env.py       # 環境檢查
uv run pytest                            # 測試（全部離線）
uv run ruff check . && uv run ruff format .
```

- 用了 `--extra` 安裝時，`uv run` 也要帶同一個 `--extra`，否則 uv 會把 torch 換回預設版。
- `uv.lock` 要和 `pyproject.toml` 一起提交。重新產生 lock 需要連得到 download.pytorch.org。

## 原則

- 教學優先：清楚比聰明重要。每個檔案都要能從頭讀到尾，寧可多幾行直白的程式，也不要難懂的抽象。
- 教材先從具體問題與例子開始，再介紹術語；前置要連到確切小節。需要圖解時使用與例子一致的 SVG。
  正文或圖解變更後，依 [逐節審閱標準](docs/editorial-guide.md) 由獨立讀者核對，再重建 Notebook 與網站。
- 從零實作：模型與前處理（patch embedding、log-mel 頻譜等）直接用 PyTorch 寫。要引入預訓練編碼器
  （SigLIP、Whisper 等）或 transformers / timm / torchaudio / torchvision 之前，先和作者確認。
- 依賴越少越好：新增套件前先說明為什麼需要。
- 要能在 CPU、Apple MPS、單張 NVIDIA GPU 上跑；玩具設定在筆電上要幾分鐘內跑完。
- 文件與註解用繁體中文，技術名詞保留英文；README_en.md 同步更新英文版。
- 原始資料快取、權重、實驗輸出不進普通 Git；使用者授權的固定教學資料包僅放在
  `assets/training/*.tar.gz`，使用 Git LFS。來源、授權、SHA-256 manifest 與生成器使用普通 Git。
  `data/` 仍是本機忽略目錄；測試不能依賴網路。

## 什麼能進普通 Git

- 第三方全文（論文 PDF、網頁、外部程式碼或文件）不提交。審閱需要讀全文時，全文留在本機並加入 `.gitignore`；Git 只存收據：來源 URL、版本、讀取日期、原檔 SHA-256 與引用位置。需要原文佐證時，只保存支持主張的小段摘錄。
- 單檔 ≥ 5 MiB，或一次提交新增 ≥ 20 MiB 時，先停下來和作者確認存放方式。提交前用下列指令看暫存區新增或修改的檔案大小，最後一行是合計：

  ```bash
  git diff --cached --name-only --diff-filter=AM -z | xargs -0 -r du -ch | sort -h | tail
  ```

- `git count-objects -vH` 的 `size-pack` 大約是 clone 要下載的量；[GitHub 建議](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) 整個 repo 小於 1 GB。
- 設計下一輪審閱的工具與證據格式時：同一份證據只存一份，用路徑＋SHA-256 引用；截圖這類大型自製證據放在 Git 之外，只留 SHA-256 與取得方式。新格式上線前，照現行審閱工具與檢查規定的方式保存，不要為了省空間自行繞過。

原因：內容一旦提交，就留在每個 clone 和整段歷史裡，事後只能改寫歷史才移得掉。2026-10 整理前，工作目錄有一半是逐位元組相同的副本，`.git` 約八成是審閱用的 PNG 截圖與論文 PDF。

## 審閱輪次的保存週期

- `docs/course-reviews-active.json` 指向的生效輪次必須留在 `main`：發布流程每次 push 到 `main` 都核對它，下一輪沿用舊審閱的頁面也從這裡取得證據。
- 新一輪生效且發布檢查通過後，依 [審閱輪次的保存規則](docs/repository-archives.md#審閱輪次的保存規則) 封存上一輪。

## 移除或搬移檔案前

- 先找出下列來源對要移除檔案的引用：課文與 notebook 的連結（含 `blob/main/…` 與 tag 連結）、README、`.github/workflows` 呼叫的腳本、`scripts/` 裡寫死的路徑、`.agents/skills/` 的審閱準則，以及生效輪次的紀錄。被這些來源引用的檔案留在原路徑。其他歷史紀錄裡的路徑可以只在 Git 歷史或封存 repo 中有效，但要記在 [開發分支與歷史版本](docs/repository-archives.md)。
- 不要只憑目錄名稱判斷用途。例如 `docs/natural-assistant/evidence/v4-research/` 看似審閱證據，實際是第 20 章資料準備腳本的輸入。
- 在暫存副本實際移除後，跑下列三道發布審閱檢查與全套 `uv run pytest`，結果要與移除前相同（安裝時用了 `--extra` 的話，`uv run` 要帶同一個 `--extra`）：

  ```bash
  uv run python scripts/check_course_reviews.py
  uv run python scripts/check_technical_reviews.py
  uv run python docs/review-tools/check_review_round.py
  ```

## 凍結檔案

- 生效輪次以 SHA-256 綁定實作檔、課文與部分文件。改到其中任何一個，發布檢查會回報類似 `implementation drift` 或 `current source/figure/notebook freeze stale` 的錯誤，網站就無法更新，直到重新審閱、更新凍結紀錄，並重估受影響頁面的閱讀時間。例外是現行工具（`docs/review-tools/baseline_replay.py`）每次檢查都會重放一個舊輪次：那一輪以 SHA-256 記下、又不在重放還原清單（重放前先換回舊版的檔）裡的實作與技術輸入（例如 `scripts/fetch_natural_data.py`、`docs/selftrained/v2-manifest.json`），改了會報 `preserved baseline failed`。這時新一輪的 manifest 要用 `baseline_supplements` 指向自己的補充還原清單（例如 `docs/course-review-20261019/baseline-supplements.json`）：每一筆把一個這類檔換回 repo 內保存、SHA-256 等於舊輪次所記的副本，清單也要列進那一輪的工具審閱。做法與限制見[交接文件](docs/course-inline-code-20261017/HANDOFF.md)〈合併到 `main` 之前要決定的事〉。
- 提交任何修改前都跑上面三道檢查。新工具或新 CI 檢查優先做成新檔案；新增的腳本與 workflow 不受凍結影響，但新增的課文頁面仍要審閱。必須改凍結檔時，當成教材修改，走 [逐節審閱標準](docs/editorial-guide.md)。
- 不要為了讓檢查通過而改寫既有審閱紀錄或其中的 hash；紀錄本身也以 SHA-256 封存。作者直接要求的小補丁照技術審閱規約的作者補丁規則記錄，不改寫既有紀錄。

## 改寫 Git 歷史

只在移除的內容於歷史中占用太多空間時才做，而且依序：

1. 把所有分支（目前只有 `main`）與 tag 以新的 ref 名稱推到私有封存 repo `birdhackor/tiny-perceptron-vlm-archive`。封存 repo 已有的 ref 保存著更早的歷史，不要覆寫，也不要用 `--mirror` 推送：

   ```bash
   git push git@github.com:birdhackor/tiny-perceptron-vlm-archive.git \
     'refs/heads/main:refs/heads/snapshot-<日期>' 'refs/tags/*:refs/tags/snapshot-<日期>/*'
   ```

   再用 `git ls-remote` 核對推上去的 hash 與原 repo 一致。
2. 在有完整歷史、不是 partial clone 的暫存 clone 用 `git filter-repo` 預演（淺 clone 時 `tests/test_course_commit_pins.py` 會跳過祖先核對）：逐檔比對保留檔案的 blob 與 mode 不變，三道審閱檢查的輸出和改寫前相同。`tests/test_course_data_pins.py` 讀舊版本檔案的核對也要完整歷史，在淺 clone 與 partial clone 裡都會跳過。全套測試裡，這兩個檔核對釘選 commit 的測試，遇到被改寫的 commit 會失敗：頁面、JSON 與 `scripts/fetch_natural_data.py` 釘的還是舊 hash，暫存 clone 裡已經沒有這些 commit；其他測試的結果要和改寫前相同。再拿 `git filter-repo` 產生的 commit-map（`.git/filter-repo/commit-map`）裡每個舊 hash 搜尋整個工作樹，命中處不是出處與審閱紀錄的，換成對應的新 hash；對到 40 個 0 的是改寫後變空、被刪掉的 commit，不能照換，要改釘仍存在、而且那處讀取的檔內容相同的 commit；GitHub 可能暫時仍回應已改寫掉的 commit，某個 hash 還在不在，要用新 clone 的 `git show` 判斷。換過 hash 的檔若有 SHA-256 被別處引用（例如 `docs/selftrained/v2-manifest.json`），引用處（出處與審閱紀錄除外）也要換成新值，這兩個測試檔才會全部通過；改到模型卡範本 `docs/selftrained/model-cards/repository-README.md` 時，線上的卡片要另外重新發布，見[交接文件](docs/course-inline-code-20261017/HANDOFF.md)〈合併到 `main` 之前要決定的事〉的模型卡那一項。

   換 hash 改到的檔若以 SHA-256 記在發布檢查核對的審閱紀錄裡（生效輪次的凍結紀錄，或現行工具重放的舊輪次紀錄；例如章檔與 Notebook、`docs/natural-assistant/v4/` 的文件、`docs/selftrained/v2-manifest.json`），三道審閱檢查會失敗，推送後 Publish course 也不會更新網站。這就是改了凍結檔；哪些檔光靠重新審閱解不開，見「凍結檔案」。
3. 課文或 notebook 以 `git clone --branch <tag>` 引用的 tag 一起改寫並保留（可用 `git grep -n -e '--branch' -- course notebooks` 找出；目前是第 19 章的 `selftrained-v2`）。
4. 推送前先照「凍結檔案」處理第 2 步改到的凍結檔；推送時對每個 ref 用 `--force-with-lease=<ref>:<預期舊值>`，並加 `--atomic`。推完確認 Publish course 與 CI 通過，線上 `build-info.json` 的 `revision` 已更新。
5. 更新 [開發分支與歷史版本](docs/repository-archives.md)。

原因：repo 內的紀錄以完整 hash 引用大量 commit；改寫後，這些 hash 只在封存 repo 有效。

## GitHub Actions

- push 到 `main` 會觸發 Publish course 與 CI；`ci.yml` 的 push 沒有限制分支，推任何 tag 也會觸發 CI。
- `windows-student-commands.yml` 在任何 branch 的 push 或 pull request 執行。改到的每個檔都落在它 `paths` 用 `!` 排除的審閱紀錄目錄裡時不跑；commit 訊息帶 `[skip ci]` 這類跳過字串，或一次改到超過 3,000 個檔、而 GitHub 先列出的 3,000 個都在排除的目錄裡時，也不跑。它在 Windows 上照入口頁跑學生指令，不用 secrets，不連 Modal 與 Hugging Face。
- 其他 workflow 只能手動觸發，可能啟動 Modal 計費運算或發布到 Hugging Face。沒有作者明確同意，不要觸發。

## 判斷失敗是否為回歸

- 測試或 CI 失敗時，先在改動前的版本跑同一個測試，或查同一 commit 原本的 CI 結果，再判斷是否由這次修改造成。例如本機沒有 git-lfs 時，`tests/test_selftrained_data_publish.py` 有一項測試必然失敗；`selftrained-v2` 快照在 Windows 上本來就有測試失敗。
- 專案腳本用 `uv run` 執行；macOS（Xcode）附的 Python 3.9 缺少審閱工具需要的 `datetime.UTC`。
