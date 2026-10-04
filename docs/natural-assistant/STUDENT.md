# 在自己的電腦開啟照片與語音助理

這份指引讓你打開本課程的實用成品：打字問問題、加入自己的照片，或先說一句中文，再把辨識出的文字送給助理。你不必先重新訓練模型。第一次先試一句文字，確定能回答，再依序加照片、加語音，遇到問題時比較容易知道卡在哪一步。

這個成品借用已經學過大量資料的 Qwen3-VL，再載入本課程訓練的小份調整權重。這份調整叫 **LoRA adapter**；可以把它想成寫在原書旁邊的修改筆記。只有筆記、沒有原書，模型就跑不起來。下載程式會取得公開的筆記；第一次開啟助理時，還會下載指定版本的原模型。兩份權重都會留在電腦上，下次可以重用。

語音會走兩站：**聲音 → 辨識出的文字 → 同一個聊天模型**。第一站是 Whisper 語音辨識，第二站才是聊天。你會看見它真正辨識出的文字，可以更正聽錯的字，再送出。這個成品以文字回答，沒有把聊天模型訓練成直接讀取聲波，也沒有合成語音輸出。

下面的指令使用 Linux 的 Bash 終端機，實用成品使用獨立的 **Python 3.12** 環境。兩條路線需要的套件配套不同，分開安裝可以避免互相改壞。前面各節的小實驗仍使用原本的 `.venv`，請保留它。

## 1. 先取得程式，暫時不抓訓練資料

如果已經有本專案，先進入專案資料夾；確認你使用的是包含這份指引的公開版本。如果尚未下載，在終端機執行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
```

第一行的設定讓 Git 先取得程式與教材，跳過大型訓練資料。現在只是使用已訓練的成品，這一步不需要整批訓練資料。

檢查 Python：

```bash
python3.12 --version
```

它應該顯示 `Python 3.12.x`。如果電腦只有其他版本，請先安裝 Python 3.12；不要直接把舊環境的套件換成這裡的版本。

## 2. 建立專用的環境

```bash
python3.12 -m venv .venv-natural
.venv-natural/bin/python -m pip install --upgrade pip
```

接下來每個指令都寫出 `.venv-natural/bin/python`，讓你不用猜目前終端機開的是哪一套環境。

有 NVIDIA GPU 的電腦，先安裝互相配對的 PyTorch 與 torchvision：

```bash
.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv-natural/bin/python -m pip install -r requirements-natural.txt
```

`cu128` 指的是這套 PyTorch 所用的 CUDA 12.8 執行環境。顯示卡驅動需要能支援它；安裝套件前可以用 `nvidia-smi` 檢查驅動與顯示卡是否正常。這組版本來自 [PyTorch 的官方歷史版本安裝指引](https://pytorch.org/get-started/previous-versions/#v280)，其他套件則固定在專案的 `requirements-natural.txt`。

安裝後確認 Python 看得到 GPU：

```bash
.venv-natural/bin/python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print('GPU 可用：', torch.cuda.is_available())"
```

前兩行應該以 `2.8.0`、`0.23.0` 開頭，最後一行應該是 `GPU 可用： True`。如果是 `False`，先處理驅動或安裝問題，尚未需要下載大型模型。

沒有 GPU 時可以安裝 CPU 版本：

```bash
.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv-natural/bin/python -m pip install -r requirements-natural.txt
```

CPU 能走同一條程式路線，但模型載入與回答會比較慢。這個實用模型約有 21 億個參數；CPU 路線用每個參數 4 bytes 的數值，光這部分權重就約占 8.5 GB，還需要圖片、對話、語音辨識與執行程式的記憶體。這個算式只是權重大小，不能拿來保證某個容量的電腦一定跑得動。GPU 的可用記憶體也會受圖片大小與對話長度影響，請依教材公開的實測條件判斷。

這輪學生操作指引以 Linux 為驗證範圍。尚未完成 Windows 的這套成品推論驗證；Windows 原生環境與 WSL 的驅動、路徑形式也不完全相同，因此這裡沒有把 Linux 指令當成已驗證的 Windows 步驟。

## 3. 下載經過核對的公開調整權重

先查看課程實際公開的版本：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --list
```

它會顯示公開權重的 Hugging Face 儲存庫與固定版本，也會列出聊天模型、語音辨識模型各自的固定版本。這些資料來自 `docs/natural-assistant/public-release.json`；沒有這份已發布清單時，程式會直接說「尚未發布」，不會自行改抓最新版或私人訓練檔。

下載到自己的成品資料夾：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release
```

這一步不需要 Hugging Face 登入，也不會下載訓練資料或可以繼續訓練的私人狀態。程式逐一核對公開檔案的大小與 SHA-256；SHA-256 就像檔案的指紋。所有檔案都核對成功，才會建立完整的成品資料夾。

如果中途斷線，可以重跑同一個下載指令。資料夾已經成功建立時，下載程式會保留它，請改用核對指令：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release --verify
```

不要把自己的圖片或其他檔案放進這個權重資料夾，核對程式會檢查它是否只有公開清單列出的檔案。

## 4. 開啟助理，先試一句文字

GPU 路線：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release --serve --device cuda
```

CPU 路線：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release --serve --device cpu
```

第一次會下載數 GB 的聊天模型檔案，所以需要網路、磁碟空間，也需要等候。語音辨識模型在第一次按「辨識語音」時才載入。終端機顯示下面的網址後，用**同一台電腦**的瀏覽器開啟：

```text
http://127.0.0.1:8766/
```

在「要送出的文字」輸入一句簡單的問題，例如「請用兩句話介紹你自己」，按「送出問題」。右邊或下方的「對話」會顯示模型的實際回答。先確認這一步成功，再加入照片。

如果程式說顯示卡不支援 `bfloat16`，可以改用：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release --serve --device cuda --dtype float16
```

這只是改變數值儲存形式，仍需要足夠的顯示卡記憶體。如果看到「記憶體不足」，先結束其他占用 GPU 的程式，並依終端機的錯誤訊息檢查；不要以為小份 LoRA 權重可以代替整個原模型。

## 5. 加一張照片，再接著問

在「照片」選擇一張 PNG、JPEG 或 WebP，檔案上限 8 MiB（約 8.39 MB；介面標為 8 MB）。先用畫面清楚、內容不複雜的照片，再問「照片裡的人在做什麼？」。回答出現後，可以直接問「他旁邊還有什麼？」；前一張照片會保留在這段對話裡，不必再選一次。

想試中文讀字，可以選一張字體清楚的圖片，問「請逐字讀出圖片裡的中文」。再試較小的字、多行文字、不同閱讀方向，對照它的回答與原圖。模型讀錯時，介面會保留錯的回答，讓你看見它目前的能力界線。

## 6. 把同一個問題改成語音

準備一段最長 30 秒、檔案同樣上限 8 MiB 的中文語音，格式使用 WAV、FLAC、MP3 或 OGG。在「中文語音」選擇它，再按「辨識語音」。

現在先停一下，讀「辨識原稿」與「要送出的文字」。如果聲音問的是「照片裡的人在做什麼」，但原稿漏了「人」，你可以直接在文字框補回來。確認後按「送出問題」，聊天模型讀到的是你最後送出的那份文字。

這樣可以分別看出兩種錯誤：語音辨識有沒有聽對，以及聊天模型在拿到文字後有沒有答對。更正逐字稿是你在介面做的動作；課程自動評估語音時會保留原本辨識結果，不會暗中換成正確答案。

## 7. 重開對話與結束

按「開始新對話」會清除目前對話與這段對話上傳的檔案。想繼續聊舊照片時，請留在原本的對話裡。

結束時回到終端機，按 `Ctrl+C`。這次介面收到的圖片與聲音會隨介面關閉而刪除，下載過的模型權重會繼續保留。介面只在本機提供服務；GitHub Pages 上的教材與你的本機聊天介面是不同網址。

如果之後完全離線使用，可以在所有模型都已下載完成的情況下，加上 `--local-files-only`。聊天模型與語音辨識模型分開下載；只試過文字，還不足以確認語音模型也已在本機。
