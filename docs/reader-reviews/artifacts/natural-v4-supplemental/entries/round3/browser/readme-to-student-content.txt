首頁
教材
第 20 章：做一位能看圖、讀字與聽問題的助理
在自己的電腦開啟照片與語音助理¶

這份指引帶你打開第 20 章的成品：打字聊天、加入照片，或先說一句中文再送出問題。先用最簡單的文字確認程式能回答，接著加圖片，最後加語音。每次多加一種輸入，遇到問題時就容易知道卡在哪一步。

本版成品使用Qwen3-VL-2B-Instruct圖文底座與Whisper-large-v3-turbo，沒有另加微調修正。打字與照片進入同一圖文模型；語音先由Whisper聽寫成文字，再交給這位聊天助理。回答顯示為文字。公開清單固定已驗證選定的配置，下載程式照清單取得配套，不要求每位讀者自己挑模型。

原理入口是20.1 的輸入分工，成品的能力及限制集中在20.13。這份操作指引先帶你走一條完整路線，不要求先重新訓練模型。

1. 取得程式，確認 Python¶

以下使用Linux的Bash終端機、Git與Python 3.12。先執行git --version確認Git已安裝；若找不到命令，依Git官方安裝指引安裝後再繼續。先把本版放在新的tiny-perceptron-natural資料夾，保留原有專案與練習：

GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/birdhackor/tiny-perceptron-vlm.git tiny-perceptron-natural
cd tiny-perceptron-natural
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 59a1eda4ed7b6e8609892ec2b9013c821ac93e69
python3.12 --version


第一行讓Git先取得程式與教材，跳過大型訓練資料；第三行固定實際操作核對使用的程式與公開清單。現在只使用成品，無須下載全部練習題。第四行應顯示Python 3.12.x；如果找不到這個命令，先安裝Python 3.12及它的venv支援，再繼續。這個新資料夾已有相同版本時，可直接從建立環境繼續。

接下來的指令都在專案根目錄執行。看到 scripts/、requirements-natural.txt 與 tiny_perceptron/，就到了正確位置。

2. 建立專用環境，選擇 CPU 或 NVIDIA GPU¶

前面的小型教學實驗使用 .venv；這個成品另外建立 .venv-natural，讓兩套配套可以各自保留：

python3.12 -m venv .venv-natural
.venv-natural/bin/python -m pip install --upgrade pip


每次明寫 .venv-natural/bin/python，就不用猜終端機正在使用哪套 Python。下面的 CPU 和 GPU 安裝選一種即可。

沒有 NVIDIA GPU，或先準備 CPU 路線時：

.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv-natural/bin/python -m pip install -r requirements-natural.txt


有 NVIDIA GPU，並準備用它執行圖文模型時：

nvidia-smi
.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv-natural/bin/python -m pip install -r requirements-natural.txt
.venv-natural/bin/python -c "import torch; print('GPU 可用：', torch.cuda.is_available()); print('支援 bfloat16：', torch.cuda.is_bf16_supported())"


nvidia-smi 應能列出顯示卡與驅動。cu128 是這套 PyTorch 使用的 CUDA 12.8 配套，驅動也要能支援它；版本來自 PyTorch 官方安裝指引。最後兩項是 True 時，可以使用後面的預設 GPU 指令。GPU 可用但不支援 bfloat16 時，本指引另提供 float16 啟動方式。

CPU與GPU兩條路都要載入完整圖文底座。磁碟放下載檔，記憶體另外放運算中間結果、照片與對話。語音辨識器也有自己的權重，這份程式在CPU執行聽寫；只看顯示卡容量，還不能判斷整套助理的負擔。20.3用一個乘法例子解釋這個差別。

下面是本版一次Linux CPU操作的實際記錄。圖文模型使用float32，也就是每個浮點數用32 bits表示；Torch把同一項運算分給5個執行緒，另設定1個管理不同運算之間工作的執行緒（interop）。模型快取已齊備，接著用實際Chromium瀏覽器走文字、照片、語音、更正送出、追問與清除對話：

觀察項目	實際數字	量測範圍
本專案公開配置文件	2份，共9,262 bytes	匿名下載的說明與來源記錄；不含官方模型權重。
兩個官方模型快照	23個檔案，共5,889,111,977 bytes，約5.4847 GiB	已有快取中的指定模型與配套檔；不含Python環境、其他快取或檔案系統額外占用。
模型服務記憶體峰值	13,496,104 KiB，約12.8709 GiB	Linux服務程序的最大常駐記憶體，不含瀏覽器與操作驅動程序。
啟動到介面可用	24.24秒	本機模型快取已完成，包含載入圖文模型；不含下載。
四次聊天請求	3.42／42.87／36.58／37.11秒	瀏覽器依序等待問候、看圖、送出更正語音文字與追問的時間。
第一次辨識語音	18.03秒	瀏覽器等待轉寫的時間，含這次首次載入辨識器。
完整操作工作	170.33秒	包含啟動、四次聊天、一次辨識、瀏覽器操作、清除與關閉；不是單次回答延遲。

這些數字描述這臺機器與這組輸入，沒有量出最低RAM、最低磁碟容量或所有問題的固定等待時間。這次重用已有官方模型快取，沒有重做模型權重下載；公開配置文件則從空目錄匿名取得並核對。完整來源見操作核對。這份檢查證明操作與清除接通，沒有替回答打分，品質仍看20.13的能力卡。

這份完整瀏覽器操作的實測範圍是Linux CPU；上面的NVIDIA安裝選擇仍要在自己的設備確認。Windows、WSL與Apple的安裝及裝置配套各有差異，需要另外的完整驗證，不能只換一下斜線便視為相同路線。

3. 讀取公開清單，再下載選定配套¶

先查看這一版實際配對的模型與檔案：

.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --list


這份公開清單像零件表：圖文底座、圖片與文字處理工具，以及語音辨識器，都使用指定官方版本。本版沒有修正權重，先下載兩份小型配置文件，再由服務取得官方模型；不自行換成最新版，也不需要Hugging Face登入。

下載並核對：

.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4
.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4 --verify


下載程式逐檔核對配置文件的大小與SHA-256；SHA-256是從完整檔案算出的內容指紋。全部核對成功，才建立完整目錄。第二行只重新檢查這些配置文件，尚未下載完整模型，也不檢查回答能力。

中途下載失敗，可以重跑同一下載指令。已完成的目錄使用 --verify；如果內容不同，程式會停止並保留它，請改用新的下載目錄，後面的 --output 也要一起改。自己的照片和錄音不放進權重目錄，以免被當成多出的檔案。

4. 啟動助理，先試一句文字¶

依安裝路線，CPU 使用：

.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4 --serve --device cpu --dtype float32


NVIDIA GPU 使用：

.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4 --serve --device cuda


第一次啟動會下載並載入指定圖文底座，請等終端機顯示服務已開始。語音辨識器在第一次按「辨識語音」時另行下載並載入；已有相同快取時可重用。因此只有文字試用成功，還沒有確認語音權重已準備好。

在執行程式的同一台電腦，用瀏覽器開啟：

http://127.0.0.1:8766/


這個網址連的是你自己的電腦，終端機必須保持開啟。GitHub Pages 上的教材則是另一個網站。若程式在桌機上執行，手機的 127.0.0.1 會指向手機本身，不能用同一網址直接接上桌機。

在「要送出的文字」輸入「你好，請用一句話回答」，按「送出問題」。等「對話」出現實際回應，確認最簡單的文字路線已通，再加下一種輸入。

GPU 可以使用、但不支援 bfloat16 時，結束原服務，再以 float16 啟動：

.venv-natural/bin/python scripts/fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --output checkpoints/natural-assistant/release-v4 --serve --device cuda --dtype float16


這是另一種數值儲存形式，仍須足夠記憶體。若看到記憶體不足，先檢查其他程式佔用、照片大小與對話長度；下載檔量不是運算所需的完整記憶體量。

5. 加一張照片，保留同一段對話¶

在「照片」選擇 PNG、JPEG 或 WebP。檔案上限是 8 MiB，約 8.39 MB；圖片最多一千六百萬像素，只接受單張靜態圖片。先選畫面清楚的照片，問「照片裡主要有哪些東西？」。回答後再問其中一個物件的位置，原照片會留在這段對話裡。

需要它讀字時，把要求說清楚。例如「請逐字抄寫招牌上的中文，保留原字，不加說明」，和「這張告示大致在說什麼」，是不同任務。拿原圖逐字核對，才能知道抄寫有沒有完成；回答提到熟悉店名，還不足以證明每字都對。照片判尺見20.9，逐字和行序見20.10及20.11。

想開始新的話題時按「開始新對話」。這會清除歷史與這段對話上傳的檔案；追問原照片時，則保留目前對話。

6. 改用語音，先檢查聽寫再送出¶

準備一段最長 30 秒、上限 8 MiB 的中文錄音，格式使用 WAV、FLAC、MP3 或 OGG。在「中文語音」選取檔案，按「辨識語音」。這一步只聽寫，尚未請聊天助理回答。

先讀「辨識原稿」與「要送出的文字」。如果你說「我不吃辣」，原稿卻寫「我吃辣」，就應先更正。按「送出問題」後，聊天模型會收到最後送出的那份文字，以及目前歷史與照片。原稿與更正是兩份不同記錄，介面會保留這個差別。

這是一條兩站路線：先分清說了什麼，再回應需求。你可以直接輸入正確原話，比較同一問題的文字與語音回答；兩路回答措辭不必完全相同，重點是有沒有符合相同要求。20.12解釋如何定位聽錯與回錯。成品只以文字回答，朗讀回覆需要另外的語音合成元件。

7. 結束、離線使用與排除問題¶

結束時回終端機按 Ctrl+C。介面這次收到的照片與聲音隨服務關閉而刪除，下載權重與模型快取仍可重用。

想在之後離線使用，先在線上完整試過文字、照片及語音，確定圖文模型和語音模型都已下載，再把原啟動指令加上 --local-files-only。程式會只讀本機快取；缺檔時會直接說明，不再連網補抓。

若下載核對失敗，先重新執行第 3 步的 --verify，分清缺檔和版本不同。若程式版本不符，回到公開清單配對的專案版本；不改清單指紋來繞過檢查。若語音格式無法讀取，可先轉成單聲道 WAV，再從短錄音開始測試。若對話太長，按「開始新對話」，確認短問題能回答後，再逐步加入前文。

到這裡，你已能使用成品，也能看見每一站真正收到和回出的內容。接下來若想自己改資料、重新訓練，先讀資料說明，再依訓練指引建立自己的新實驗。