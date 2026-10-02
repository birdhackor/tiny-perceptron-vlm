# CLAUDE.md

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
