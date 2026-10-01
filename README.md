# 小小感知機 Tiny Perceptron

> 用最少的程式碼，從零打造一個會看、會聽、也會讀的小型多模態模型。

繁體中文 | [English](README_en.md)

> [!NOTE]
> 🚧 專案剛起步：目前只完成**環境初始化**，模型程式碼還沒開始寫。

## 這是什麼？

小小感知機是一個輕鬆的教學專案。我們用最少的程式碼從零實作一個小型多模態模型，並把每一步拆開講清楚。

- **第一階段**：文字 + 影像 + 音訊 → 文字（多模態輸入，文字輸出）
- **之後擴充**：影片 + 音訊 → 文字 / 影像，以及更複雜的多模態任務

風格參考 Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat)，以及 [MiniMind](https://github.com/jingyaogong/minimind) / [MiniMind-V](https://github.com/jingyaogong/minimind-v)。共同點是模型很小、可以從零訓練、以教學為主。

**適合誰**：有基本 Python / PyTorch 基礎，想搞懂多模態模型是怎麼從零做出來的人。

## 路線圖

- [x] 環境初始化：uv、PyTorch、環境檢查、三平台 CI
- [ ] 第一階段：文字 + 影像 + 音訊 → 文字
- [ ] 之後：影片 + 音訊 → 文字 / 影像

## 需要什麼環境？

| 硬體 | 適合做什麼 |
| --- | --- |
| 一般 CPU（筆電就行） | 跑通整個流程、玩具規模的實驗 |
| Apple Silicon（M 系列，macOS 14 以上） | 同上，用 MPS 加速 |
| NVIDIA GPU | 正式訓練；目標是單張消費級顯示卡就能訓練 |
| Google Colab | 手邊沒有 GPU 時的替代方案 |

軟體方面只需要 [uv](https://docs.astral.sh/uv/) 和 Git。uv 會自動下載 Python 3.13，並把所有套件裝進專案內的 `.venv`，不會弄亂你的系統環境。

各種選擇背後的理由（PyTorch / CUDA 版本、為什麼不用 torchaudio、候選資料集等）都整理在 [docs/environment.md](docs/environment.md)。

## 安裝

**1. 安裝 uv**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. 下載專案並安裝套件**

```bash
git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync
```

`uv sync` 預設安裝 PyPI 上的 PyTorch，但它在不同平台上的行為不一樣，請依你的環境選擇指令：

| 你的環境 | 安裝指令 | 說明 |
| --- | --- | --- |
| macOS（Apple Silicon） | `uv sync` | 預設版就支援 MPS |
| Linux + NVIDIA，驅動 ≥ 580 | `uv sync` | 預設版就是 CUDA 13.0 |
| Windows + NVIDIA，驅動 ≥ 580 | `uv sync --extra cu130` | PyPI 預設版在 Windows 上只有 CPU |
| NVIDIA 驅動 < 580，或 GTX 9 / 10 系列、V100 等舊顯卡 | `uv sync --extra cu126` | 不支援 RTX 50 系列 |
| 沒有 NVIDIA GPU 的 Linux | `uv sync --extra cpu` | 省下約 2.5 GB 的 CUDA 套件 |
| 沒有 NVIDIA GPU 的 Windows | `uv sync` | 預設版就是 CPU 版 |

驅動版本可以用 `nvidia-smi` 查。各 CUDA 版本支援哪些顯卡，見 [docs/environment.md](docs/environment.md#3-pytorch-與-cuda)。

> [!WARNING]
> 用了 `--extra` 之後，每次 `uv run` 都要帶同一個 `--extra`，例如 `uv run --extra cu130 python ...`。不然 uv 會「好心地」把 PyTorch 換回預設版。
> 嫌麻煩的話，可以先啟用虛擬環境（macOS / Linux：`source .venv/bin/activate`，Windows：`.venv\Scripts\activate`），之後直接用 `python`。

**3. 檢查環境**

```bash
uv run python scripts/check_env.py   # 印出版本與裝置（CUDA / MPS / CPU），並在裝置上實際算一次
uv run pytest                        # 冒煙測試，全部離線執行
```

`check_env.py` 發現常見問題時會直接給提示，例如「有 NVIDIA 驅動，卻裝到 CPU 版 PyTorch」。回報問題時請附上它的輸出。

### 在 Google Colab 上

Colab 已經預裝 PyTorch，用 pip 安裝其他套件即可：

```python
!git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
%cd tiny-perceptron-vlm
!pip install -e .
!python scripts/check_env.py
```

### 在中國大陸

```bash
# PyPI 鏡像（Windows PowerShell：$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"）
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# Hugging Face 鏡像，用來下載資料集與模型
export HF_ENDPOINT=https://hf-mirror.com
```

設定 PyPI 鏡像後，uv 會改寫本機的 `uv.lock`，這個改動不要提交。
ModelScope、PyTorch wheel 鏡像、W&B 的替代品等更多選擇，見 [docs/environment.md](docs/environment.md#7-中國大陸網路環境)。

## 專案結構

```text
tiny-perceptron-vlm/
├── tiny_perceptron/      # 模型與訓練的核心程式（施工中）
├── scripts/
│   └── check_env.py      # 環境檢查
├── tests/                # 冒煙測試，全部離線
├── docs/
│   └── environment.md    # 環境研究筆記：每個選擇的理由
└── pyproject.toml        # 套件與工具設定（uv）
```

## 開發

```bash
uv run pytest                                 # 測試
uv run ruff check . && uv run ruff format .   # lint 與格式化
```

GitHub Actions 會在 Linux、macOS、Windows 三個平台上跑環境檢查與測試。

## 致謝

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)、[karpathy/nanochat](https://github.com/karpathy/nanochat)
- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)、[jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v)
- [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)

## 授權

[MIT](LICENSE)
