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

## 審閱與發布分工

- 教材實質修改仍核對易讀性／正確性／連貫性，按改動範圍審閱。PR 簡述發現、修正、測試與未驗證範圍；不聲稱未做的盲讀、真人測試或執行。
- 審閱是編輯流程，不是網站的執行輸入。不得恢復以歷史審閱、逐段 session、舊原稿 SHA、原審閱者回叫或全書問題鏈作為發版門檻。
- 一般 CI 記錄保留 30 天，臨時詳細審閱資料最多 90 天；需要長期支持科學主張的結果、來源、授權、切分規則和限制另行保存。PR 留簡短結論即可。
- 閱讀時間是可選的 AI 近似資訊，來源改變時省略舊估時；缺少或過期不阻擋部署，不由程式偽造新估時。

## Repo 存放與封存

- 普通 Git 保存教材、圖、Notebook、實作、測試，以及教學和實驗重現所需的小型設定、生成器、資料來源、授權和重要結果。
- 權重、完整資料、訓練原始輸出、截圖與完整審閱過程放 Release、模型／資料儲存庫或暫存 artifact，按需下載；不要複製進新的審閱目錄。
- 封存輪次的完整公開來源固定在提交 `83630ea1cea23d47c4b77162a31cbb7566686d11`，見 [歷史版本](docs/repository-archives.md)。它仍是 main 的祖先，不改寫舊紀錄。
- 搬移前辨認真正的教材程式輸入與來源／授權資料，不因路徑有 evidence 或 review 就整棵刪除，也不因舊審閱文字引用就整棵保留。
- 既有來源與結果的事實引用保持可查。移出的說明／證據連結指向固定舊提交，不能留失效的 blob/main 連結。
- 本輪不改寫 Git 歷史，不 force push。若未來要縮小完整 clone，先另作備份、處理教材固定版本連結，再提出一次性的歷史整理。

## 變更驗證與 GitHub Actions

- 修改前後核對 `python scripts/build_course.py --check`；程式、資料、共享執行輸入或依賴變更時跑相關測試及全套 CPU Notebook。純文字變更可重用相同執行輸入的成功結果。
- `scripts/course_execution.py` 使用一個粗粒度鍵管理整套 CPU 結果；鍵不同、缺檔或驗證失敗就重新執行全部。不得把失敗輸出或不同程式的輸出當成快取命中。
- 網站正式發布使用 `export_course.py --publish`：核對執行結果、固定 Notebook 版本、嚴格建站與連結／下載。不得為了快取而略過錯誤、猜結果或標示未執行的新程式已成功。
- 日常 CI 在 PR 和 main 執行；文字修改跑 Linux 檢查，實作／安裝／依賴變更另跑 Windows、macOS。Windows 學生指令按入口及共用程式的改動觸發。
- 模型訓練、Modal 計費運算與 Hugging Face 發布維持手動，沒有作者授權不觸發。網站發布不下載純為歷史審閱使用的資料包。
- 發布後核對 workflow 與線上 build-info.json 的 revision；合併成功不能代替確認網站上線。

## 判斷失敗是否為回歸

測試失敗先比對基底或同一提交的既有結果，保留原始失敗。不要刪除或跳過產品測試來掩蓋回歸；退役的只有舊審閱引擎及它的專屬檢查。
