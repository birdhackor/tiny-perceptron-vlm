# 環境研究筆記

> 整理於 2026 年 10 月。套件版本變化很快，實際安裝的版本以 `uv.lock` 為準。
> 資料集大小多半是概略數字，下載前請再看一次資料集頁面。

這份筆記記錄初始化 repo 時的調查：小小感知機需要什麼環境、每個選擇的理由，以及還沒決定的事。

## 決策一覽

| 項目 | 選擇 | 理由 |
| --- | --- | --- |
| 套件管理 | [uv](https://docs.astral.sh/uv/) | 快、有 lockfile 可重現、會自動下載 Python。nanochat 也用它 |
| Python | 3.13（支援 ≥ 3.11） | 與 Colab 相同。3.10 在 2026-10 停止維護，PyTorch 2.15 也預計不再支援 |
| PyTorch | PyPI 預設版，另用 extras 切換 `cpu` / `cu126` / `cu130` | 見[第 3 節](#3-pytorch-與-cuda) |
| 文字 | `tokenizers`，自己訓練 BPE | 可以訓練，處理中文比 GPT-2 的 BPE 有效率；MiniMind 也這樣做 |
| 影像 | `pillow` 解碼，前處理自己用 PyTorch 寫 | 不需要 torchvision |
| 音訊 | `soundfile` 解碼，log-mel 頻譜自己用 `torch.stft` 寫 | torchaudio 已進入維護模式，停在 2.11 |
| 影片（之後） | TorchCodec 或 PyAV，到時再決定 | 見[第 4 節](#4-各模態的函式庫) |
| 資料 | `datasets` + `huggingface-hub` | 音訊欄位用 `decode=False` 再交給 soundfile，避開 TorchCodec 與 FFmpeg |
| 實驗紀錄 | 先用 print；W&B / SwanLab 之後選配 | 中國大陸連不上 W&B |
| 開發工具 | ruff、pytest、GitHub Actions | CI 在 Linux / macOS / Windows 三個平台跑 |

## 1. 硬體：能在哪裡跑？

目標是讓手邊只有筆電的人也能把流程跑通，有一張消費級 GPU 的人可以完整訓練。

| 硬體 | 定位 |
| --- | --- |
| CPU | 跑通流程、玩具規模實驗（小圖、短音訊、幾 M 參數） |
| Apple Silicon（MPS） | 同上但快一些。PyTorch 2.14 的 macOS wheel 只有 arm64，而且需要 macOS 14 以上，Intel Mac 裝不到 |
| NVIDIA GPU | 正式訓練。顯卡世代與驅動版本的對照見[第 3 節](#3-pytorch-與-cuda) |
| Google Colab | 免費的 T4。目前是 Python 3.13 + PyTorch 2.11，用 `pip install -e .` 即可。T4 沒有原生 bf16，要用 fp16 |

參考專案的數字，可以拿來估算規模：

- nanochat 在 M3 Max 上用 CPU / MPS 訓練 6 層的小模型：預訓練約 30 分鐘，SFT 約 10 分鐘。
- MiniMind-V 在單張 3090 上，SFT 一個 epoch 約 2 小時。
- nanoVLM 至少要 4.5 GB VRAM。

估顯示記憶體的經驗法則：用 AdamW 加混合精度訓練時，每個參數大約要 16 bytes（權重、梯度、兩份優化器狀態），還要再加上 activation。
例如 1 億參數的模型，光是參數相關就要約 1.6 GB。

## 2. Python 與套件管理

- **uv**（目前 0.12.x）：一個指令就能建虛擬環境、裝套件、鎖版本。`uv.lock` 讓所有讀者裝到完全一樣的版本。
- **Python 3.13**：寫在 `.python-version`，uv 會自動下載，和 Colab 目前的版本相同。
  `requires-python = ">=3.11"`，3.11、3.12、3.13 都實測可以安裝並通過測試。在 3.11 上 uv 會自動改用 numpy 2.4，因為 numpy 2.5 需要 3.12。
- **建置後端**：hatchling。程式放在 repo 根目錄的 `tiny_perceptron/`（flat layout），比 `src/` layout 更容易讀。

## 3. PyTorch 與 CUDA

目前最新的是 PyTorch 2.14.1（2026-09-30）。PyPI 上的 `torch` 在三個平台上的行為不同：

| 平台 | PyPI 預設的 `torch` |
| --- | --- |
| Linux | CUDA 13.0 版，需要 NVIDIA 驅動 ≥ 580。沒有 GPU 也能用，但會多下載約 2.5 GB 的 CUDA 套件 |
| macOS | 只有 Apple Silicon（arm64），需要 macOS 14 以上，支援 MPS |
| Windows | 只有 CPU 版 |

PyTorch 官方 index 上，2.14 提供這幾種 CUDA 版本（cu128 從 2.12 起就沒有了）：

| 版本 | 需要的驅動 | 支援的顯卡 |
| --- | --- | --- |
| cu126 | ≥ 525 | Maxwell 到 Hopper：GTX 9 / 10 系列、V100、RTX 20–40 系列、A100、H100。**不支援 RTX 50 系列** |
| cu130 | ≥ 580 | Turing 到 Blackwell：T4、RTX 20–50 系列、A100、H100、B200。不支援 GTX 10 系列與 V100 |
| cu132 | ≥ 580 | 同 cu130 |

所以 `pyproject.toml` 用 uv 官方建議的寫法：預設從 PyPI 安裝，再提供幾個 extra，讓使用者改從 PyTorch 官方的 index 安裝特定版本。

| extra | 適用對象 |
| --- | --- |
| （不加） | macOS；Linux + 驅動 ≥ 580 |
| `cpu` | 沒有 GPU，想省下載量 |
| `cu126` | 驅動 < 580；GTX 9 / 10 系列、V100 等舊顯卡 |
| `cu130` | Windows + NVIDIA（驅動 ≥ 580） |

用 `nvidia-smi` 可以看到驅動版本。顯卡不在目前這版 torch 的支援範圍時，torch 2.14 也會在執行時直接提示該換哪個 index。

**使用 extra 的陷阱**：`uv run` 每次執行前都會先同步環境。
如果用 `uv sync --extra cu126` 安裝，之後卻直接 `uv run ...` 而沒帶 `--extra cu126`，uv 會把 torch 換回 PyPI 預設版。在 Windows 上就是從 CUDA 版變回 CPU 版。
這個行為已經實際驗證過。解法是每次都帶同一個 `--extra`，或啟用 `.venv` 後直接用 `python`。

其他考量：

- `--torch-backend=auto` 可以自動偵測 CUDA 版本，但目前只有 `uv pip` 能用，`uv sync` / `uv lock` 不支援，所以沒有採用。
- `torch>=2.11` 這個下限是配合 Colab：它預裝 torch 2.11，用 pip 安裝時就不會重裝 torch。TorchCodec 0.12 起也要求 torch ≥ 2.11。實際版本由 `uv.lock` 鎖定。
- PyTorch 2.15（預計 2026-10-28）有個還沒定案的提案：拿掉 cu126 與 cu130，改以 cu132 為預設。
  如果通過，GTX 10 系列與 V100 最後能用的就是 2.14，`cu130` 這個 extra 屆時也要改成 `cu132`。

### 給之後訓練程式的提醒

- **bf16**：NVIDIA 顯卡要 Ampere（sm_80）以上才有原生 bf16，T4 請用 fp16 搭配 GradScaler。
  `torch.cuda.is_bf16_supported()` 預設會把「模擬」也算成支援，要加 `including_emulation=False`。MPS 在 macOS 14 以上支援 bf16。
- **torch.compile**：Linux 可以直接用；Windows 的 GPU 要另外安裝 `triton-windows`；macOS 的支援還在原型階段。建議做成選項，預設關閉。
- **`scaled_dot_product_attention`**：Flash 和 cuDNN 後端要 sm_80 以上。在 MPS 上訓練（需要梯度）時，會走比較慢的非融合路徑。
- **`torch.accelerator`**（2.6 起）可以寫出不分 CUDA / MPS 的程式。
  呼叫 `current_accelerator()` 時要加 `check_available=True`，否則在沒有 GPU 的機器上它也會回傳 cuda。

## 4. 各模態的函式庫

原則是能用 PyTorch 自己寫的就自己寫，這本來就是教學內容；只有「解碼檔案格式」這種與模型無關的苦工才交給套件。

### 文字

用 `tokenizers` 訓練 byte-level BPE。MiniMind 用同樣的方法，詞表只有 6400。
`tiktoken` 不能訓練新詞表，而 GPT-2 的詞表切中文很沒效率，所以不用。

### 影像

`pillow` 負責解碼。縮放、正規化、切 patch 都用 PyTorch 寫。

不用 torchvision，理由如下：

- 我們用不到它的模型和資料集，而它的影像解碼功能也已經改建議用 TorchCodec。
- 它和 torch 一樣分 CPU / CUDA 版，必須從同一個 index 安裝，會讓 extras 設定變複雜。
  （0.29 起已經不必和 torch 版本完全對應。）

### 音訊

`soundfile` 負責解碼。wheel 已經內建 libsndfile（1.2.2），可以讀寫 WAV / FLAC / MP3 / OGG，不能讀 AAC / M4A。

log-mel 頻譜用 `torch.stft` 加上自己寫的 mel 濾波器組。Whisper 的設定可以當參考：16 kHz、25 ms 視窗（`n_fft=400`）、10 ms 步長（`hop_length=160`）、80 個 mel bin。

不用 torchaudio，因為它已經進入維護模式：

- 最後一版是 2.11.0（2026-03），之後就沒有再發新版，torch 已經到 2.14 了。
- 2.9 起移除了很多 API，`load` / `save` 也改成呼叫 TorchCodec。
- `MelSpectrogram`、`Resample` 雖然還在，但自己寫 log-mel 本來就是教學內容。

大部分語音辨識資料集本來就是 16 kHz。真的需要重新取樣時，可以加 `soxr` 這個很小的套件，或者自己寫。

### 影片（之後的階段）

| 選項 | 優點 | 缺點 |
| --- | --- | --- |
| [TorchCodec](https://github.com/meta-pytorch/torchcodec) 0.17 | PyTorch 官方，能解碼音訊也能解碼影片，Linux 上支援 GPU 解碼 | 需要 torch ≥ 2.11 和系統的 FFmpeg 4–9 共享函式庫；Linux 的 PyPI wheel 是 CUDA 版，必須像 torch 一樣設定 index |
| [PyAV](https://github.com/PyAV-Org/PyAV) 19 | wheel 已經內建 FFmpeg，安裝最簡單 | 要 Python ≥ 3.12；API 比較底層 |

## 5. 資料

### 下載工具

- 用 `datasets`（5.x）和 `huggingface-hub`（1.x）。datasets 5.0.1 限制 `huggingface-hub<2.0`，所以會裝到 1.x。兩者都會讀 `HF_ENDPOINT`，可以接鏡像站。
- **音訊欄位要注意**：datasets 4.0 之後，Audio / Video 欄位改用 TorchCodec 解碼，而 TorchCodec 需要系統的 FFmpeg。
  這裡改用 `ds.cast_column("audio", Audio(decode=False))` 拿原始 bytes，再交給 soundfile 解碼。實測沒有安裝 torchcodec 也能正常運作。
- 舊的、以腳本建立的資料集（例如 `openslr/librispeech_asr`）要加 `revision="refs/convert/parquet"`。

### 候選資料集

★ 表示小到可以在 CPU 上做示範。

**文字**

| 資料集 | 語言 | 大小 | 備註 |
| --- | --- | --- | --- |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 英 | 約 7.6 GB | 約 214 萬則短篇故事，CDLA-Sharing-1.0 |
| ★ [TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M) | 簡中 | 約 840 MB | 機器翻譯，授權待確認 |
| [minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) | 中為主 | `pretrain_t2t_mini.jsonl` 1.2 GB | ModelScope 上也有（`gongjy/minimind_dataset`） |

目前找不到可靠的繁體中文版 TinyStories，需要的話可以用 OpenCC（s2twp）轉換。

**影像 + 文字**

| 資料集 | 語言 | 大小 | 備註 |
| --- | --- | --- | --- |
| ★ [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) | 英 | 約 1.1 GB | 8000 張圖，每張 5 句描述 |
| ★ [COCO-CN](https://huggingface.co/datasets/AIMClab-RUC/COCO-CN) | 簡中 | 文字約 15 MB | 2 萬張 COCO 圖的人工中文描述，圖要另外從 COCO 下載 |
| [COCO Karpathy split](https://huggingface.co/datasets/yerevann/coco-karpathy) | 英 | 文字約 50 MB | 圖要另外從 COCO 下載 |
| [LLaVA-Pretrain（LCS-558K）](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 英 | 約 27 GB | 多模態對齊預訓練的常見選擇 |
| [minimind-v_dataset](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset) | 中英各半 | 約 9 GB | 圖已縮成 256×256，打包在 parquet 裡 |

**音訊 + 文字**

| 資料集 | 語言 | 大小 | 備註 |
| --- | --- | --- | --- |
| ★ [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) | 英 | 27 MB | 3000 段唸數字的錄音，8 kHz，非常適合第一個 demo |
| ★ [Mini LibriSpeech](https://www.openslr.org/31/) | 英 | 約 460 MB | LibriSpeech 的迷你版，CC BY 4.0 |
| ★ [Speech Commands v0.02](https://huggingface.co/datasets/google/speech_commands) | 英 | 約 2.3 GB | 約 10 萬段 1 秒指令詞 |
| [LibriSpeech clean-100](https://www.openslr.org/12/) | 英 | 約 6.3 GB | 100 小時，CC BY 4.0 |
| [AISHELL-1](https://www.openslr.org/33/) | 普通話 | 約 15 GB | 178 小時，Apache-2.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 多語 | 每種語言約 12 小時 | 含普通話、粵語、英語 |
| [Common Voice](https://commonvoice.mozilla.org/) | 多語 | zh-TW 約 2.9 GB | CC0。2025-10 起改由 Mozilla Data Collective 發布，要註冊帳號並同意條款 |

**影片 + 音訊（之後的階段）**

| 資料集 | 語言 | 備註 |
| --- | --- | --- |
| [MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT) | 英 | 1 萬段影片、20 萬句描述，約 2.3 GB |
| [VATEX](https://eric-xw.github.io/vatex-website/) | 中英 | 每段影片有中英文各 10 句描述；影片要從 YouTube 下載 |

## 6. 實驗紀錄

一開始只用 print 和 CSV，不加任何依賴。之後若需要儀表板，可以選配 W&B 或 SwanLab。
MiniMind 的 README 提到 W&B 在中國大陸無法連線，所以改用 SwanLab。

## 7. 中國大陸網路環境

**PyPI 鏡像**：擇一即可。

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple   # 清華 TUNA
export UV_DEFAULT_INDEX=https://mirrors.ustc.edu.cn/pypi/simple     # 中科大 USTC
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/     # 阿里雲
```

也可以寫進使用者層級的 `uv.toml`（Linux / macOS：`~/.config/uv/uv.toml`，Windows：`%APPDATA%\uv\uv.toml`），不用每次設定：

```toml
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

注意：`uv.lock` 記錄的是 pypi.org 的網址。換成鏡像後，uv 會重新解析並改寫你本機的 `uv.lock`，這個改動不要提交。
鏡像只該設定在自己的電腦上，不要寫進專案的 `pyproject.toml`。

**PyTorch wheel**：`download.pytorch.org` 在大陸通常連得上，只是可能比較慢。也可以把專案裡的 PyTorch index 換成鏡像，以 cu126 為例：

```bash
uv sync --extra cu126 --index pytorch-cu126=https://mirror.nju.edu.cn/pytorch/whl/cu126
```

`--index 名稱=網址` 會取代 `pyproject.toml` 裡同名的 index，這個機制已經驗證過；南京大學（NJU）的鏡像依其說明文件是標準的 simple index，但我們沒有實際連線測試。
這種寫法同樣會改寫本機的 `uv.lock`。

**Python 本身**：uv 預設從 GitHub 下載 Python。太慢的話，可以把 `UV_PYTHON_INSTALL_MIRROR` 設成 GitHub Release 的鏡像（例如中科大）。
也可以自己先裝好 Python 3.13，uv 會直接使用系統上已有的版本。

**Hugging Face**：

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run hf download roneneldan/TinyStories --repo-type dataset --local-dir data/TinyStories
```

**ModelScope**：很多中文資料集和模型只放在 ModelScope，例如 MiniMind 系列。

```bash
uvx --from modelscope-hub modelscope download gongjy/minimind_dataset --repo-type dataset --local-dir data/minimind
```

`uvx` 會把下載工具裝在獨立的環境裡，不會影響專案的套件。`modelscope` 這個指令現在由 `modelscope-hub` 套件提供。
不要把 `modelscope[datasets]` 裝進專案：它限制 `datasets<=4.8.4`，會和本專案的 datasets 5.x 衝突。

## 8. 參考專案的環境

| 專案 | Python | 安裝方式 | 編碼器 | 硬體 |
| --- | --- | --- | --- | --- |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 不限 | `pip install` 一行 | 純文字 | 可在 CPU / MPS 上跑小設定 |
| [nanochat](https://github.com/karpathy/nanochat) | ≥ 3.10 | uv，用 `cpu` / `gpu` extras 切換 torch，有提交 `uv.lock` | 純文字 | 8×H100 約 2 小時；也有 CPU / MPS 腳本 |
| [MiniMind](https://github.com/jingyaogong/minimind) | 3.10 | `requirements.txt`，torch 讓使用者自己裝 | 純文字 | 單張 3090 約 2 小時 |
| [MiniMind-V](https://github.com/jingyaogong/minimind-v) | 3.10 | 同上 | 凍結的 SigLIP2 | 單張 3090，SFT 每個 epoch 約 2 小時 |
| [MiniMind-O](https://github.com/jingyaogong/minimind-o) | — | 同上 | 凍結的 SenseVoice（音訊）與 SigLIP2（影像） | 單張 3090 |
| [nanoVLM](https://github.com/huggingface/nanoVLM) | 3.12 | `uv add` 一串套件 | SigLIP2，會一起微調 | VRAM ≥ 4.5 GB |

共同點：

- 大家都用 PyTorch、numpy、parquet，以及 `tokenizers` 或 tiktoken。
- 用到預訓練編碼器的專案，都透過 `transformers` 載入。
- 音訊都是 16 kHz，用 soundfile 或 librosa 讀取。

## 9. 還沒決定、但會影響環境的事

1. **編碼器要從零訓練，還是用凍結的預訓練模型？**
   上面所有多模態參考專案都用預訓練的影像 / 音訊編碼器，沒有一個從零訓練音訊編碼器。
   - 全部從零訓練：最符合「從零」的精神，也不需要新的依賴，但需要更多資料和算力，效果也會比較差。
   - 用預訓練模型：需要 `transformers`（或用 `safetensors` 自己寫載入），還要下載幾百 MB 的權重。
   - 折衷做法：玩具任務從零訓練，正式任務換成預訓練編碼器，兩條路都做成教學。
2. **語言模型要從零訓練，還是從現成的小模型初始化？** MiniMind-V 就是從 MiniMind 的權重開始訓練。
3. **資料格式**：parquet、jsonl 加圖片 / 音訊檔，或 nanoGPT 式的 numpy memmap。
4. **實驗紀錄工具**：要不要加 W&B 或 SwanLab。
5. **繁體或簡體**：公開的中文語料多半是簡體。要做繁體版本的話，需要 OpenCC 之類的轉換工具。
