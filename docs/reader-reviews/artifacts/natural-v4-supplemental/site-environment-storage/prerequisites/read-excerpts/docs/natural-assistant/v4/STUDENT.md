# 在自己的電腦開啟照片與語音助理

這份指引帶你打開第 20 章的成品：打字聊天、加入照片，或先說一句中文再送出問題。先用最簡單的文字確認程式能回答，接著加圖片，最後加語音。每次多加一種輸入，遇到問題時就容易知道卡在哪一步。

本版成品使用Qwen3-VL-2B-Instruct圖文底座與Whisper-large-v3-turbo，沒有另加微調修正。打字與照片進入同一圖文模型；語音先由Whisper聽寫成文字，再交給這位聊天助理。回答顯示為文字。公開清單固定已驗證選定的配置，下載程式照清單取得配套，不要求每位讀者自己挑模型。

原理入口是[20.1 的輸入分工](../../../course/chapters/20.md#20.1)，成品的能力及限制集中在[20.13](../../../course/chapters/20.md#20.13)。這份操作指引先帶你走一條完整路線，不要求先重新訓練模型。

## 1. 取得程式，確認 Python

以下使用Linux的Bash終端機、Git與Python 3.12。先執行`git --version`確認Git已安裝；若找不到命令，依[Git官方安裝指引](https://git-scm.com/downloads)安裝後再繼續。先把本版放在新的`tiny-perceptron-natural`資料夾，保留原有專案與練習：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/birdhackor/tiny-perceptron-vlm.git tiny-perceptron-natural
cd tiny-perceptron-natural
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 59a1eda4ed7b6e8609892ec2b9013c821ac93e69
python3.12 --version
```

第一行讓Git先取得程式與教材，跳過大型訓練資料；第三行固定實際操作核對使用的程式與公開清單。現在只使用成品，無須下載全部練習題。第四行應顯示`Python 3.12.x`；如果找不到這個命令，先安裝Python 3.12及它的`venv`支援，再繼續。這個新資料夾已有相同版本時，可直接從建立環境繼續。

接下來的指令都在專案根目錄執行。看到 `scripts/`、`requirements-natural.txt` 與 `tiny_perceptron/`，就到了正確位置。

## 2. 建立專用環境，選擇 CPU 或 NVIDIA GPU

前面的小型教學實驗使用 `.venv`；這個成品另外建立 `.venv-natural`，讓兩套配套可以各自保留：

```bash
python3.12 -m venv .venv-natural
.venv-natural/bin/python -m pip install --upgrade pip
```

每次明寫 `.venv-natural/bin/python`，就不用猜終端機正在使用哪套 Python。下面的 CPU 和 GPU 安裝選一種即可。

沒有 NVIDIA GPU，或先準備 CPU 路線時：

```bash
.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv-natural/bin/python -m pip install -r requirements-natural.txt
```

有 NVIDIA GPU，並準備用它執行圖文模型時：

```bash
nvidia-smi
.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv-natural/bin/python -m pip install -r requirements-natural.txt
.venv-natural/bin/python -c "import torch; print('GPU 可用：', torch.cuda.is_available()); print('支援 bfloat16：', torch.cuda.is_bf16_supported())"
```

`nvidia-smi` 應能列出顯示卡與驅動。`cu128` 是這套 PyTorch 使用的 CUDA 12.8 配套，驅動也要能支援它；版本來自 [PyTorch 官方安裝指引](https://pytorch.org/get-started/previous-versions/#v280)。最後兩項是 True 時，可以使用後面的預設 GPU 指令。GPU 可用但不支援 bfloat16 時，本指引另提供 float16 啟動方式。

CPU與GPU兩條路都要載入完整圖文底座。磁碟放下載檔，記憶體另外放運算中間結果、照片與對話。語音辨識器也有自己的權重，這份程式在CPU執行聽寫；只看顯示卡容量，還不能判斷整套助理的負擔。[20.3](../../../course/chapters/20.md#20.3)用一個乘法例子解釋這個差別。

下面是本版一次Linux CPU操作的實際記錄。圖文模型使用float32，也就是每個浮點數用32 bits表示；Torch把同一項運算分給5個執行緒，另設定1個管理不同運算之間工作的執行緒（interop）。模型快取已齊備，接著用實際Chromium瀏覽器走文字、照片、語音、更正送出、追問與清除對話：

| 觀察項目 | 實際數字 | 量測範圍 |
| --- | ---: | --- |
| 本專案公開配置文件 | 2份，共9,262 bytes | 匿名下載的說明與來源記錄；不含官方模型權重。 |
| 兩個官方模型快照 | 23個檔案，共5,889,111,977 bytes，約5.4847 GiB | 已有快取中的指定模型與配套檔；不含Python環境、其他快取或檔案系統額外占用。 |
| 模型服務記憶體峰值 | 13,496,104 KiB，約12.8709 GiB | Linux服務程序的最大常駐記憶體，不含瀏覽器與操作驅動程序。 |
| 啟動到介面可用 | 24.24秒 | 本機模型快取已完成，包含載入圖文模型；不含下載。 |
| 四次聊天請求 | 3.42／42.87／36.58／37.11秒 | 瀏覽器依序等待問候、看圖、送出更正語音文字與追問的時間。 |
| 第一次辨識語音 | 18.03秒 | 瀏覽器等待轉寫的時間，含這次首次載入辨識器。 |
| 完整操作工作 | 170.33秒 | 包含啟動、四次聊天、一次辨識、瀏覽器操作、清除與關閉；不是單次回答延遲。 |

這些數字描述這臺機器與這組輸入，沒有量出最低RAM、最低磁碟容量或所有問題的固定等待時間。這次重用已有官方模型快取，沒有重做模型權重下載；公開配置文件則從空目錄匿名取得並核對。完整來源見[操作核對](../evidence/v4-research/student-selected-public-ui/README.md)。這份檢查證明操作與清除接通，沒有替回答打分，品質仍看[20.13的能力卡](../../../course/chapters/20.md#20.13)。

這份完整瀏覽器操作的實測範圍是Linux CPU；上面的NVIDIA安裝選擇仍要在自己的設備確認。Windows、WSL與Apple的安裝及裝置配套各有差異，需要另外的完整驗證，不能只換一下斜線便視為相同路線。

## 3. 讀取公開清單，再下載選定配套

先查看這一版實際配對的模型與檔案：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --list
```

這份[公開清單](public-release.json)像零件表：圖文底座、圖片與文字處理工具，以及語音辨識器，都使用指定官方版本。本版沒有修正權重，先下載兩份小型配置文件，再由服務取得官方模型；不自行換成最新版，也不需要Hugging Face登入。

下載並核對：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4
.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4 --verify
```

下載程式逐檔核對配置文件的大小與SHA-256；SHA-256是從完整檔案算出的內容指紋。全部核對成功，才建立完整目錄。第二行只重新檢查這些配置文件，尚未下載完整模型，也不檢查回答能力。

中途下載失敗，可以重跑同一下載指令。已完成的目錄使用 `--verify`；如果內容不同，程式會停止並保留它，請改用新的下載目錄，後面的 `--output` 也要一起改。自己的照片和錄音不放進權重目錄，以免被當成多出的檔案。

