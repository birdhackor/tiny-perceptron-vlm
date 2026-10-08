# 環境設定與硬體選擇

讀教材和執行模型需要的設備不同。看網頁、圖解與已展示的結果，用一般瀏覽器即可；想改一小段程式，才需要Python。先選自己要做的事情，再準備對應工具，會比一開始下載所有套件、資料和權重容易。

本課有兩套程式環境。前面自己寫的小模型使用`.venv`；第20章的照片與語音助理使用`.venv-natural`。它們像兩個工具箱：各自放好配套，執行時指定正確的Python，就不必讓不同版本互相遷就。

## 1. 硬體：能在哪裡跑？

| 你想做什麼？ | 從哪裡開始？ | 需要注意什麼？ |
| --- | --- | --- |
| 讀正文、看圖與結果 | [閱讀指南](../course/README.md) | 不必安裝Python，也不必租GPU。 |
| 修改一節短程式 | 選定教材小節的Colab入口，或[本機暖身](../course/first-steps.md#W.1) | 小實驗用CPU即可；Notebook中的程式格要按順序執行。 |
| 試用自行訓練的v2小助理 | [公開CPU操作](selftrained/v2-public-cpu-commands.md) | 使用 `.venv` 的 `cpu`、`selftrained` 配套，匿名下載固定推論權重。 |
| 重做小模型的正式訓練 | [訓練操作](../course/training.md) | 先讀對應概念，再看該項資料、步數與設備設定。 |
| 從隨機權重重做v2 | [本機stage訓練指引](selftrained/TRAINING.md) | 取得固定LFS資料，逐段接自己的checkpoint；全量CPU訓練與短程式不同。 |
| 試用照片與語音助理 | [學生操作指引](natural-assistant/v4/STUDENT.md) | 需要另外取得完整圖文模型和語音辨識器；這比短程式大得多。 |
| 微調照片與語音助理 | [LoRA訓練指引](natural-assistant/v4/TRAINING.md) | 使用自己的NVIDIA GPU，按固定資料與配方建立候選，再做驗收。 |

CPU也能執行第20章成品，但等待時間和記憶體需求不同。學生操作指引保留一次Linux CPU實測的磁碟、記憶體及等待時間；這是那臺機器的觀察，沒有量出所有電腦的最低需求。Apple MPS、其他作業系統和不同顯示卡也要分別確認，不能由一次CPU成功推成全部設備都已驗證。

## 2. Python與套件管理

前面的小模型使用[uv](https://docs.astral.sh/uv/)管理Python與套件。`pyproject.toml`像工具清單，`uv.lock`把配套版本固定下來；`--frozen`要求沿用這份清單。專案的Python支援範圍寫在`requires-python`，本機預設版本寫在`.python-version`。

先依[W.1](../course/first-steps.md#W.1)取得專案。在有`pyproject.toml`的專案根目錄執行：

```bash
uv sync --frozen --extra cpu --group notebook
uv run --extra cpu --group notebook python scripts/check_env.py
uv run --extra cpu --group notebook jupyter lab
```

第一行準備CPU版套件與Notebook工具；第二行核對環境的基本運算；第三行開啟JupyterLab。環境檢查能跑通，還不代表某個模型已學會任務，模型品質要另看留出題的結果。

`uv run`會先同步環境。安裝時若選了`--extra cpu`，執行時也帶同一個extra，才不會切回另一套PyTorch。另一種方式是啟用`.venv`後直接用那套`python`；啟用方式與kernel選擇見暖身指南。

第19章v2也使用這套環境，另加 `selftrained`，提供safetensors推論權重的讀取配套：

```bash
uv sync --frozen --extra cpu --extra selftrained
```

[CPU操作頁](selftrained/v2-public-cpu-commands.md)保留文字、圖片、指定區域讀字、計算工具及語音後續對話的實際命令。所選validation示範完成8次chat呼叫與1次history append，確認公開下載、載入與介面可用；這不是未知題成功率，也不是全量訓練重跑。能力結果見[19.12](../course/chapters/19.md#19.12)。

第20章另使用Python3.12、PyTorch2.8.0與`requirements-natural.txt`中的固定配套。請按[學生操作指引第2節](natural-assistant/v4/STUDENT.md#2-建立專用環境選擇-cpu-或-nvidia-gpu)建立`.venv-natural`，並使用`.venv-natural/bin/python`。這套成品環境不沿用前面小模型的`uv.lock`。

## 3. PyTorch與CUDA

PyTorch負責張量與模型運算。CPU版在主處理器執行；NVIDIA GPU版另外需要匹配的CUDA套件與驅動。看到顯示卡型號，還不足以知道目前的Python能不能使用它。

本專案的小模型環境提供三個互斥選擇：

| extra | 用途 |
| --- | --- |
| `cpu` | CPU實驗，避免另外下載GPU配套。 |
| `cu126` | 從專案指定的CUDA12.6套件來源取得PyTorch。 |
| `cu130` | 從專案指定的CUDA13.0套件來源取得PyTorch。 |

選GPU路線前，先用`nvidia-smi`確認設備與驅動，再對照[PyTorch官方安裝指引](https://pytorch.org/get-started/locally/)和[NVIDIA相容性說明](https://docs.nvidia.com/deploy/cuda-compatibility/)選配套。`cu126`與`cu130`不是「比較慢」和「比較快」兩個按鈕；套件版本、驅動與顯示卡都要相容。

例如選用`cu126`時，安裝和執行要一致：

```bash
uv sync --frozen --extra cu126 --group notebook
uv run --extra cu126 python scripts/check_env.py
```

第20章的專用環境有自己的CPU／CUDA12.8安裝指令，請沿用那份指引，不把兩個工具箱的命令混在一起。GPU可用、數值格式可用，以及整個任務能完成，也是三個不同的檢查。

## 4. 圖片與聲音如何進入程式？

前面的短實驗由Pillow讀圖片、soundfile讀音訊，再用PyTorch自己處理像素與波形。這讓我們可以看見切小圖、頻率分析和向量配對的每一步。第20章則使用預訓練模型配套的處理器，保留它原本認得的輸入格式。

例如，[聲音入門](../course/chapters/12.md#12.1)先從波形理解聲音，再逐步看時間與頻率。這些局部實驗不等於已訓練一般語音辨識器。第20章的Whisper先把錄音轉成文字，再交給圖文模型聊天，分工見[20.1](../course/chapters/20.md#20.1)。

第19章v2由自行訓練的聲音入口直接提供特徵給同一文字核心，沒有執行Whisper或其他ASR（自動語音辨識，也就是把語音轉成文字，見[12.11](../course/chapters/12.md#12.11)）。它只學地址、App與卡片三種銀行客服主題，再產生文字回答；沒有一般逐字聽寫或語音輸出。OCR（光學字元辨識，也就是從圖片讀字，見[11.12](../course/chapters/11.md#11.12)）同樣限於12個已知字、使用者提供的一個連續1–4字區域，Sans與Serif兩種字型都已出現在訓練中。

檔案格式能否讀取與模型能力也要分開。成品目前接受的圖片、音訊格式及上限列在學生操作指引；選短而清楚的WAV錄音開始，會比較容易核對「聽到什麼」與「回答什麼」。

## 5. 資料與權重按需要取得

讀課文不需要先下載完整資料。開始一項訓練前，先讀[訓練資料入口](../assets/training/README.md)或[第20章資料說明](natural-assistant/v4/DATA.md)，確認用途、授權與切分。第19章v2按[CPU操作頁](selftrained/v2-public-cpu-commands.md)取得固定權重和示範素材；第20章則取得自己的[公開成品配套](natural-assistant/v4/STUDENT.md#3-讀取公開清單再下載選定配套)。

資料、權重、套件和運算中間結果各自占空間。壓縮包不大，不代表解包後或載入模型後也同樣小；小小的LoRA修正檔也仍需要完整底座參與計算。檔案如何保存與核對，接著讀[教材資產存放](asset-storage.md)。
