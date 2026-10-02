# 小小感知機 Tiny Perceptron

看到「今天天氣」，電腦怎麼猜下一個字？這套教材從幾個字的編號與接字表開始，用完整例子、圖解和短程式，逐步解釋文字、圖片與聲音模型的運作。

繁體中文 | [English](README_en.md)

**[線上教材](https://birdhackor.github.io/tiny-perceptron-vlm/) · [閱讀路線](course/README.md) · [操作與數學暖身](course/first-steps.md) · [222 個小節](course/lessons.md) · [訓練操作](course/training.md)**

教材預覽版：網站提供正文、SVG 圖解與實際 CPU 輸出。每節按「在 Colab 動手做」可開啟同一課的完整 Notebook；第一個程式格會準備專案與套件，小實驗不必先租 GPU。也可下載本節 `.ipynb` 後依下方步驟在本機練習。Colab 雲端執行尚未實機驗證。

![文字與識別編號的對照](course/figures/character_ids.svg)

## 教材提供什麼

18章與A／B／C支線、每節一份獨立 Notebook，以及自製 SVG 圖解與逐步提示。前段只用小矩陣、接字表、MLP與手寫注意力；後段才加入現代架構、Dense／MoE、cache／SDPA、量化與蒸餾。可以來回跳章，不需要從頭把同一個模型訓練到底。

文字與圖音模型、對話遮罩、DPO、量化儲存與訓練基本件直接以 PyTorch 實作。資料、訓練、推論與評估有可執行入口。LoRA、QAT 與小型 RL 等概念提供局部實驗；影片目前提供逐幀切片的輔助函式，完整教學列為後續擴充。此版本沒有提供已訓練的通用影音助理。

第一次接觸本專案也能從第一節開始；Python、tensor、機率、矩陣與梯度都有可按需查閱的暖身，不必先修完所有數學。每節先說明問題與具體例子，所需背景提供可點擊的小節連結。開發時驗證了 CPU 執行與梯度，沒有正式訓練模型；權重與能力評估留給讀者之後在自己的 GPU 環境進行。詳見[驗證報告](docs/validation.md)。

## 安裝與開啟 Notebook

先安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/) 與 Git。CPU 閱讀與練習：

```bash
git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
source .venv/bin/activate
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

Windows PowerShell 啟用指令是 `.venv\Scripts\Activate.ps1`。選擇 **Tiny Perceptron** kernel，先開啟 `notebooks/01/1.1.ipynb`。詳細操作見[暖身指南](course/first-steps.md)。

| 環境 | 安裝選擇 |
| --- | --- |
| CPU | `uv sync --frozen --extra cpu --group notebook` |
| Apple Silicon | `uv sync --frozen --group notebook` |
| NVIDIA CUDA 13.0 相容硬體／驅動 | `uv sync --frozen --extra cu130 --group notebook` |
| NVIDIA CUDA 12.6 相容硬體／驅動 | `uv sync --frozen --extra cu126 --group notebook` |

各版本的顯卡與驅動限制、Colab和鏡像設定見[環境說明](docs/environment.md)。使用 extra 後，啟用 `.venv` 直接用 `python`，或每次 `uv run` 都帶相同 extra，避免更換 PyTorch。CUDA／MPS未在本次 CPU 環境實測。

只想閱讀網頁，可直接開啟線上教材；離線副本用 `python scripts/export_course.py` 匯出，再打開 `outputs/site/index.html`。網頁可搜尋小節；Notebook用來修改程式與執行。SVG保留靜態標註，支援系統減少動態效果。離線時公式保留 TeX 文字，連網後由 MathJax 排版。附帶已執行輸出的建置與 GitHub Pages 操作見[教材發布](docs/publishing.md)。

## 最小流程

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

預設只做一個 batch 的 forward/backward，不更新權重。正式訓練再明確加 `--train`；字元模型、多模態、風格與安全、偏好和蒸餾配方在[訓練操作](course/training.md)。[首批固定訓練資料](assets/training/README.md)以 Git LFS 發布，執行 `python scripts/fetch_training_assets.py --list` 查看並按需解包。原始快取與權重仍放在被 Git 忽略的 `data/`、`checkpoints/`；分工見[資產管理](docs/asset-storage.md)。

## 專案結構與維護

```text
course/chapters/     教材正文與短程式，單一來源
course/figures/      自製 SVG
notebooks/          每小節一份，由正文產生
tiny_perceptron/    模型、資料、對齊、模態與壓縮零件
scripts/            資料、訓練、推論、評估與教材建置
tests/              離線契約測試
docs/               大綱、研究、環境與驗證紀錄
```

```bash
python scripts/build_visuals.py
python scripts/format_course_code.py course/chapters/01.md
python scripts/build_course.py
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_notebooks.py --mode python
python scripts/check_notebooks.py --mode kernel --lesson 3.6
pytest -ra
ruff check .
ruff format --check .
```

修改教材請改 `course/chapters/`，先格式化該檔的 Python 範例再產生 Notebook。每節須由獨立讀者核對目前正文與 SVG，審閱協定見[編輯說明](docs/editorial-guide.md)；版本不符會阻止發布。執行輸出與網頁放在 `outputs/`。完整 kernel 檢查用 `--mode kernel`，每節開獨立工作桌。GitHub Actions 保留 Linux、macOS、Windows 的核心測試；Notebook整套驗證另行執行。

## 參考與授權

參考 [nanoGPT](https://github.com/karpathy/nanoGPT)、[nanochat](https://github.com/karpathy/nanochat)、[MiniMind](https://github.com/jingyaogong/minimind)、[MiniMind-V](https://github.com/jingyaogong/minimind-v) 與 [nanoVLM](https://github.com/huggingface/nanoVLM)。公開課與學習者回饋的來源、取捨與查證限制保留在[大綱](docs/curriculum.md)與研究筆記。

程式、教材與自製圖解使用 [MIT](LICENSE)。外部資料依各自授權。
