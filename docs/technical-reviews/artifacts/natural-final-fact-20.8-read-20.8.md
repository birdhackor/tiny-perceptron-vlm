## 20.8 怎麼把整套助理交給下一位讀者？

你借給同學一張修訂便條，卻沒有交代它對應哪本筆記，對方可能把正確修正貼到錯誤版本。LoRA檔也是如此：它需要配對的底座、圖片處理方式與文字格式；語音入口還需要辨識模型和取樣設定。交付的目標，是讓下一位讀者拿到相同版本，真的打開網頁、送進自己的照片與錄音，再核對助理的回答。

前置是[19.11推論與續訓檔不同](19.md#19.11)、[20.2沿用底座](20.md#20.2)與[20.4修正分支](20.md#20.4)。固定版本指定底座與微調檔是哪一版；SHA-256指紋用來核對實際下載內容。文字切詞工具、對話模板與圖片處理工具也要配套，因為同一張圖可能因前處理不同，變成另一串模型輸入。

先用一小段人工文字示範為什麼檔名相同還不夠：

```python
from hashlib import sha256

original = b"base revision A, adapter revision B"
received = b"base revision A, adapter revision C"
expected_fingerprint = sha256(original).hexdigest()
received_fingerprint = sha256(received).hexdigest()
print("期待指紋", expected_fingerprint[:12])
print("收到指紋", received_fingerprint[:12])
print("內容符合交付清單", expected_fingerprint == received_fingerprint)
```

`b`開頭表示bytes，也就是計算內容指紋的一串位元組；它們是人工示意文字，不是模型檔。`sha256`讀這些bytes，`hexdigest()`把摘要寫成可讀字串，`[:12]`只顯示前12個字符方便比較。正式下載要核對完整指紋，不能只用12字符。兩份內容的微調版本字母不同，最後應為False；檢查指出內容不符合約定，不能判斷哪一版回答更好。

這次成品使用約21.3億個參數的圖文底座，加上約161萬個LoRA修正參數；聲音辨識另外使用約2.42億個參數的Whisper-small，精確計算見[20.2](20.md#20.2)。本課只更新LoRA，底座與辨識器沿用既有權重；學生試用時可以直接載入成品。

先看交付版實際通過了哪些檢查。下表來自[最後測試逐題核對](../../docs/natural-assistant/evidence/semantic-final/final-semantic-review.json)及[外部OCR核對](../../docs/natural-assistant/evidence/semantic-external/external-semantic-review.json)，其中聊天模型在GPU上推論，權重採用bfloat16，這是16-bit浮點數的一種格式；後面的本機試用則用CPU與float32，也就是32-bit浮點數。這些名稱延續20.2的數值儲存問題：格式會影響記憶體用量與數值精度，硬體也需支援所選格式。聊天與照片的「通過」要求回答完成、遵守題目指令，且整段內容符合已固定的標準；無依據的新增事實與被長度上限截斷的回答仍算失敗。

| 檢查 | 交付版結果 | 這份結果的範圍 |
| --- | --- | --- |
| 普通文字聊天 | 4/4題通過 | 本課自寫的四道短題 |
| 使用先前對話 | 1/2題通過 | 兩道固定歷史對話題 |
| 照片描述與短問答 | 17/36題通過；其中完整描述0/12 | 12張留出照片，每圖三題；完整描述會加錯細節或被截斷 |
| 合成中文字卡 | 抄寫與行序18/18題；有字／無字6/6題 | 本課字卡，不能據此推定街景也能讀對 |
| 外部自然中文圖片 | 1/10張逐字核對通過 | 固定的10張外部圖，沿用來源逐字稿；來源標註也有遺漏與歧義 |
| 華語錄音轉寫 | 原始字元錯誤率150/481，約31.2% | 12段真人朗讀；包含標點及簡繁字形差異 |
| 正確逐字稿送進聊天 | 4/12題通過 | 把同12段朗讀的官方逐字稿當文字輸入，作為對照 |
| 實際辨識文字送進聊天 | 1/12題通過 | 原樣使用Whisper結果；朗讀陳述，並非自然問句 |

照片36題來自12張圖，不能當成36張新照片。語音兩路也來自同12段錄音；每段聲音只辨識一次，兩路分別檢查聊天回應。字元錯誤率的150是替換、刪除及插入的總次數，481是原始參考字元數。這些小份結果不涵蓋所有物件、手寫字、口音、方言或多人重疊語音，也不能證明上游底座從未見過公開素材。

[學生CPU網頁試用](../../docs/natural-assistant/evidence/student-trial/student-trial-summary.json)確實走過8次聊天、2次語音辨識與重新開始對話；照片和先前訊息能接續保留，回應都正常送回頁面。但它同時留下了能力失敗：新字卡明明寫著「星光書店」「週三公休」，要求逐字抄寫時，助理只回「這是一張中文的圖片。」要求摘要下一段錄音時，它又回了一段制式自我介紹。

錄音的原始辨識文字是「全球有近200个跑步旅游组织大多独立运营」，原樣送出後，助理回「這句話是錯誤的。」卻沒有提出依據。使用者在文字框追加「請用更短的一句話說明。」後，才得到「全球有近200個跑步旅遊組織，大多獨立運營。」這次是追加指令後的繁體中文重述，不能當成可靠摘要的證據。可以在[實際網頁截圖](../../docs/natural-assistant/evidence/student-trial/desktop-full-history.png)看到這些原回答；操作順暢與模型可靠，需要分別判斷。

這輪CPU試用使用Linux、Python 3.12.14、5個邏輯CPU與5個PyTorch執行緒，圖文底座以float32載入。微調小包重新匿名下載；底座與Whisper重用已逐檔核對指紋的固定版本快取。整段自動瀏覽器試用共289.192秒，包含模型載入與操作，不含首次重下兩個大模型。從快取載入圖文組合花8.426秒；8次網頁回答的單次模型生成為2.160–60.071秒。兩次7.32秒錄音的辨識生成與解碼分別花3.140、3.095秒，不含讀檔、前處理及辨識器載入；本次Whisper從快取載入另花0.404秒。

Linux記錄的助理程序最高常駐記憶體約12.36 GiB，涵蓋模型載入、上述網頁操作、語音辨識及兩次附加圖片對照。這個數字是程序的實測峰值，不能當成全系統RAM用量或所有電腦的最低容量保證。底座float32權重數字本身約8.51 GB的算式，與這個實測也不是同一件事；圖片、對話長度與其他運算內容都會影響需求。

本節公開成品是[assistant-2b-v3固定版本](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/8dab26b439499188149b7ac5b861dab4f6c0755a/natural-v3/assistant-2b-v3)。公開微調小包含權重、設定、模型卡與來源紀錄四個檔案，共6,459,450 bytes，約6.46 MB，其中LoRA權重約6.44 MB。[交付清單](../../docs/natural-assistant/public-release.json)保存完整版本、指紋與大小；[公開下載核對](../../docs/natural-assistant/evidence/release/result.json)與這次學生重新匿名下載都通過了核驗。底座與語音辨識器的固定公開檔案另共23個，合計約5.24 GB，安裝套件與快取還會另占磁碟空間。

接下來用Linux的Bash終端機與Python 3.12走CPU路線。電腦需要已有Git與Python 3.12，並能建立`venv`。在準備放置專案的資料夾執行以下指令；這個固定程式版本已包含正式公開清單，兩步Git操作都跳過大型訓練資料：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
GIT_LFS_SKIP_SMUDGE=1 git checkout 1d5fccbc76a6ebbb3757f9b31df17a7c22ea171b
python3.12 --version
```

最後應顯示`Python 3.12.x`。若已經有這個公開版本的專案，就進入它的資料夾，不必再clone。以下指令都在專案資料夾執行；這輪實測以Linux為範圍，Windows操作另需驗證。

先建立獨立環境，保留前面小實驗使用的`.venv`。每個指令都寫出新的Python路徑，讓你不用猜終端機正在用哪一套套件：

```bash
python3.12 -m venv .venv-natural
.venv-natural/bin/python -m pip install --upgrade pip
.venv-natural/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv-natural/bin/python -m pip install -r requirements-natural.txt
```

第三行先安裝互相配對的CPU版PyTorch與torchvision，第四行再補齊專案固定的其他套件。安裝完成後，先查看實際公開的版本，再下載微調小包：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --list
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release
```

`--list`顯示公開成品、底座與語音模型各自的固定版本。公開成品應是`assistant-2b-v3`，版本為`8dab26b439499188149b7ac5b861dab4f6c0755a`。第二行下載微調小包，逐檔核對大小與完整SHA-256，不需要Hugging Face登入。全部核對成功後，才會建立指定的成品資料夾；兩個大模型在後面的使用步驟才下載。

中途斷線時，可以重跑同一個下載指令。資料夾已成功建立後，請改用下面的核對指令。它會重新檢查微調小包，也檢查資料夾有沒有多出不屬於清單的檔案，所以自己的照片與錄音應放在別處。

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release --verify
```

核對成功後，啟動CPU助理：

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --output checkpoints/natural-assistant/release --serve --device cpu
```

這個入口會再核對微調小包、套件與配套程式，然後載入指定底座與LoRA。第一次需要連網下載底座，也要等待載入；看到終端機顯示「照片與語音助理已啟動」後，才用**同一台電腦**的瀏覽器開啟：

```text
http://127.0.0.1:8766/
```

這是你電腦正在執行的助理頁面。GitHub Pages提供教材；模型是在剛才啟動程式的電腦上運算。終端機要繼續開著，關掉程式後，瀏覽器便不能繼續取得回答。

依序試四步，每次等結果出現再做下一步：

1. **先打字。**在「要送出的文字」輸入「你好，請用一句話回答」，按「送出問題」，查看「對話」裡真正生成的回答。能收到回答後，再加入圖片。
2. **再加照片。**在「照片」選一張清楚的PNG、JPEG或WebP，檔案最多8 MB。問「照片裡有哪些東西？」並送出。接著可以問「它們在哪個位置？」；前一張照片會跟著這段對話保留，不必再選一次。另開對話、加入中文字圖片，也能試逐字抄寫，但要逐字對照原圖。
3. **把同一問題改成錄音。**先準備最長30秒、最多8 MB的中文錄音檔，格式使用WAV、FLAC、MP3或OGG。在「中文語音」選檔，按「辨識語音」。第一次按這個按鈕，才會下載並載入固定的Whisper語音模型；這一步尚未請聊天模型回答。
4. **讀過逐字稿，再送出。**看「辨識原稿」與「要送出的文字」。若聲音問的是「照片裡有哪些東西」，卻漏掉了「照片」，先在文字框補回來，再按「送出問題」。聊天模型收到的是你最後確認的文字，也能接續先前的照片與對話。

這四步把「聽對問題」與「回答正確」分開了。更正逐字稿是你在介面做的動作；[20.7](20.md#20.7)的自動語音評估仍保留原始辨識結果。模型讀錯字、描述了圖中沒有的東西，或回答改錯原意時，也應留下原回答，再檢查錯在哪一站。

想重新開始，按「開始新對話」，會清除這段對話與上傳的檔案。結束試用時，回到終端機按`Ctrl+C`；本次上傳的圖片與聲音會刪除，下載的模型權重仍保留。之後若想離線使用，可以在啟動指令加`--local-files-only`，但底座與語音模型必須都已下載；只成功試過打字，還不能證明語音檔案也已齊全。

有NVIDIA GPU、驅動能支援CUDA 12.8時，可以把安裝指令的`https://download.pytorch.org/whl/cpu`換成`https://download.pytorch.org/whl/cu128`，啟動指令的`--device cpu`換成`--device cuda`。圖文聊天預設使用bfloat16；若程式明確說不支援，再加`--dtype float16`，改用另一種16-bit浮點數格式。Whisper辨識仍在CPU執行。完整檢查與故障處理見[學生操作指引](../../docs/natural-assistant/STUDENT.md)。

這份公開小包供推論使用。想接續訓練，還需要資料版本、更新器與進度等狀態；不能把LoRA推論便條宣稱成可以精確恢復作者訓練的完整工作桌。若另做量化或部署壓縮，也要重新測回答、時間與記憶體，第19章小模型的4-bit儲存結果不能直接當作這個底座的成績。

練習只把`received`改成與`original`完全相同，先預測最後True，再執行核對。然後說明這個True能證明什麼：收到的bytes符合我們指定的內容；它沒有證明模型會答對，也沒有證明訓練發生。文件、執行與能力，仍需各自的證據。
