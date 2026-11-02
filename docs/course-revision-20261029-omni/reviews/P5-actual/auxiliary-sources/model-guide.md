# 影音、有限音調與候選判斷：新版模型操作

這組權重接續第19章的三種工作：看已取得的完整灰階影片與聲音片段、把有限音型程式變成可播放聲音，以及在使用者列出的候選中判斷答案。它們從隨機參數訓練，沒有使用Qwen或Kev的權重、回答或機率作教師。原有MoE助理、Looped解題器與舊機率判斷器的[下載入口](../lesson-products/README.md)仍保留。

是否達到升版門檻，依[本輪結果與限制](v1/results-summary.json)；完成訓練不自動表示可以取代舊成品。機制、必要程式與關鍵輸出直接內建在[19.21–19.24](../../course/chapters/19.md#19.21)，閱讀不需要開Colab。本頁供下載後親自操作。

影音主測與有限音調達到本輪門檻；候選判斷器C的最終正確率與接受後風險未達標，這裡提供的是研究候選，舊C仍為預設。影音在換聲音內容的控制也未全對，不能把主測滿分讀成任何影片都能答對。

## 取得固定權重

固定封包已公開在[本輪版本資產](https://github.com/birdhackor/tiny-perceptron-vlm/releases/tag/omni-core-candidate-cpu-core-40988311100-1)。下方入口按清單取得選定版本。

在已取得本repo的目錄執行：

```bash
uv sync --extra cpu --locked
uv run --extra cpu python -m scripts.omni_revision.public_download --output checkpoints/omni-products-v1
```

入口按[固定公開清單](v1/public-release.json)匿名取得選定的影音、codec、Talker與候選判斷四份封包，核對下載及封包內每個檔案的大小與SHA-256，再安裝到指定目錄。SHA-256是檔案內容的指紋；內容不同就停止，不能把同名檔當成同一模型。下載命令每次都會連網取得並核對封包；完成安裝後，下方推論指令直接使用本機模型，不再下載。

預設安裝推論所需的設定、版本紀錄與`safetensors`權重。下載的原始ZIP還含訓練狀態，因此網路下載量大於最後安裝的推論檔。若要另外保留比較用的compact候選判斷器，在命令末加`--include-control`；若要保留完整最佳化器與隨機狀態，再加`--full-checkpoints`並選一個新目錄。只有推論權重不足以精確接續訓練；完整檔案存在也不表示已在另一種硬體驗證相同續算結果。

安裝入口只處理檔案，不執行訓練檢查點中的Python物件。公開操作使用`safetensors`。自己的輸出請放在新的路徑；WAV與不同版本的既有收據不會被覆寫。

## 同一段聲音放在不同時間

```bash
uv run --extra cpu python -m scripts.omni_revision.public_demo av --model-root checkpoints/omni-products-v1 --which 0 --output-wav outputs/omni-demo/early.wav
uv run --extra cpu python -m scripts.omni_revision.public_demo av --model-root checkpoints/omni-products-v1 --which 1 --output-wav outputs/omni-demo/late.wav
```

省略`--family`時，使用公開生成器的第一個訓練家族，方便檢查介面。兩次保留同一畫面與聲音內容，只移動聲音起點；預測應對應發聲當時的形狀。這個展示不是未見家族的成績，正式200家族結果另列在結果紀錄。

輸出的`answer`是影音模型真正預測的`circle`、`square`或`unknown`，分別代表圓、方、不可判定；後續聲音沿用它。`source`記錄回答來自哪個模組，並不代表開放式文字助理或持續對話記憶。圓對應220 Hz、停頓、440 Hz的音型，方把第一音改為880 Hz。若影音先答錯，後續仍按它的錯答生成；不能偷偷用正解取代。

在其中一條命令加`--missing image`或`--missing audio`可檢查刪除線索時的行為。不可判定時不寫聲音檔，不能把沒有檔案或靜音算成成功回答。

固定權重的CPU示範中，`--which 0`回傳`"answer": "circle"`，`--which 1`回傳`"answer": "square"`，兩者各寫出4,000個sample，即8 kHz下的半秒聲音。刪圖或刪聲音時都回傳`"answer": "unknown"`，沒有`sound`波形。這些是第一個訓練家族的介面核對，不能計入200個未見家族的正式得分。

## 輸入音型程式，產生有限聲音

```bash
uv run --extra cpu python -m scripts.omni_revision.public_demo sound --model-root checkpoints/omni-products-v1 --output-wav outputs/omni-demo/program.wav
```

預設輸入是220 Hz發聲200 ms、停100 ms，再440 Hz發聲200 ms。Talker自行產生聲音碼與結束符號，codec的decoder把碼還原成8 kHz波形。這是有限頻率與時值的音型生成，不是中文或英文人聲。

程式先完成全部自回歸聲音碼，再逐塊解碼。`whole_code_generation_seconds`記整列聲音碼的生成時間；`internal_first_chunk_ready_seconds`還包括第一塊解碼，沒有包含命令啟動、WAV寫入或播放器。這個介面不能用來宣稱文字到語音的串流首段延遲。

要換程式，可用`--request`指定一份JSON清單，例如：

```json
[
  {"frequency": 880, "duration_ms": 200, "rest_ms": 100},
  {"frequency": 440, "duration_ms": 200, "rest_ms": 0}
]
```

這份輸入就是方形對應的音型。可用頻率為220／440／880 Hz，發聲時值100／200／300 ms，停頓0／100／200 ms；一至四個事件且總長最多兩秒，最後一個事件的`rest_ms`必須為0。契約不接受相鄰同頻且無停頓的兩個事件，因為波形無法分辨那個邊界。規格內的輸入仍可能生成錯誤，格式合法與音型正確分開驗收。

## 判斷候選及是否轉交

```bash
uv run --extra cpu python -m scripts.omni_revision.public_demo choice --model-root checkpoints/omni-products-v1
```

預設材料描述三個裝置的任務、權限與就緒狀態，問題要求選A的動作。A有權限且已就緒，應執行它的record任務。候選各自有固定ID與文字描述；模型回傳候選機率，另一個資訊充分性問題則判斷必要欄位是否完整。

`prediction`保留原始分數及校準後機率；`accepted`和`decision`另表示是否符合固定接受規則。最高機率不等於必須接受，以上皆不適合也不自動算可接受答案。溫度T及接受門檻tau由各自的校準／策略資料先選好；下載者看到單一示範答錯時，不應把它當成正式測試調門檻的理由。

這個固定示範的實際CPU輸出為`prediction.answer = "record"`、`information.answer = "yes"`、`accepted = true`、`decision = "accept"`。這次兩題都答對，預測也符合固定接受規則；接受規則本身只讀預測、機率與資訊充分性答案，不知道正解。單一例子成功不能推翻正式測試中C未達標的結論。

自訂`--request`需提供同樣的`state`、`kind`、`text`、`candidates`，每個候選有`id`與`description`。要套固定接受規則，另須提供`information_question`，使用相同state並詢問與Choice同一對象的必要資訊；省略它時仍回傳分數，決策維持`handoff`。輸入不接受`gold`欄，避免把測試答案當作使用時資訊。模型訓練範圍是這個有限裝置規則世界，不能由可填JSON推論它已能處理任意客服、數學或安全決策。

## 版本與實驗紀錄

- [19.21](../../course/chapters/19.md#19.21)：兩種時間表示、各三個seed及200個最終家族。
- [19.22](../../course/chapters/19.md#19.22)：codec重建、gold音型與真影音預測整鏈的不同分母。
- [19.23](../../course/chapters/19.md#19.23)：共享材料、候選隔離、校準與固定接受規則。
- [19.24](../../course/chapters/19.md#19.24)：舊A／B保留、路由錯誤及100家族整合量測。
- [固定訓練配方](v1/formal-recipes.json)、[資料定義](v1/data-definition.json)、[事前門檻](v1/metrics.json)、[選模清單](v1/model-selection.json)與[成熟模型研究規格](v1/mature-research-protocol.md)。

完整原始結果與續訓封包存於各結果紀錄指向的版本資產；普通Git只保存程式、固定清單與必要的小型證據。Qwen2.5-Omni及Kev的研究不改動本組權重；Qwen3-Omni-30B-A3B只使用公開技術資料，不要求在學生電腦載入。
