# C-v3：可操作的裝置決策成品

目前推薦的產品C是C-v3。它根據3–6個裝置的任務、許可與就緒狀態，提出動作建議、判斷資訊是否齊全，或依固定規準給分。它從隨機參數開始訓練，不需要Qwen、Modal或Colab；本次公開操作已在CPU核對。

完整的問題、規則、記憶體與候選計分程式直接放在[19.25](../../course/chapters/19.md#19.25)。本頁負責下載與操作。C-v3接的是明確欄位的JSON介面；自由文字Auto、Looped解題器及C-v2研究介面仍有各自的[歷史入口](../lesson-products/README.md)與[影音／C-v2入口](../omni-products/README.md)，沒有把C-v3當成那些文字模型的替換權重。

## 下載固定配套

在已取得本repo、含`pyproject.toml`的目錄執行：

```bash
uv sync --frozen --extra cpu
uv run --extra cpu python -m scripts.course_product download
uv run --extra cpu python -m scripts.course_product judge
```

第一條準備課程的CPU環境；第二條匿名下載[固定Release](https://github.com/birdhackor/tiny-perceptron-vlm/releases/tag/c-v3-20261101)的封包，核對大小、完整SHA-256及四個推論檔的指紋，再安裝到`checkpoints/decision-product-v3`；第三條才執行本機推論。推論不會再下載。封包為6,168,350 bytes，約5.88 MiB，其中也保存原始碼、驗收及三項成品審閱；安裝器只取設定、tokenizer、固定政策與`safetensors`權重，不執行封包中的程式。

已有不同內容的目錄會停止，避免覆寫自己的模型。需要換位置時，下載加`--output checkpoints/my-c-v3`，推論對應加`--bundle checkpoints/my-c-v3`。相同版本可以再次核對。固定清單見[public-release.json](v3/public-release.json)。SHA-256只核對取得相同內容，能力仍須看驗收。

## 讀取自己的資料

預設使用repo中的`examples/decision_product/record-ready.json`：

```json
{
  "devices": [
    {"id": "主錄音室", "task": "record", "permission": true, "ready": true},
    {"id": "播放台", "task": "play", "permission": true, "ready": false},
    {"id": "封存NAS", "task": "save", "permission": false, "ready": true},
    {"id": "備援錄音室", "task": "record", "permission": false, "ready": false}
  ],
  "target": "主錄音室",
  "kind": "all",
  "language": "zh"
}
```

每個裝置的四個欄位都要明列，未知值填`null`，不能用`false`代替。裝置ID可用自己的中文或英文名稱，彼此須不同。`task`限`record/play/save`，`permission`與`ready`限`true/false/null`；本版一次接受3–6個裝置。`kind`可選`choice/info/rubric/all`；`language`是服務產生內部問題文字的`zh/en`選項，不是讓模型任意翻譯文字的指令。

使用自己的JSON檔時：

```bash
uv run --extra cpu python -m scripts.course_product judge --request examples/decision_product/record-unknown.json
uv run --extra cpu python -m scripts.course_product judge --request examples/decision_product/answer-absent.json
```

第一份只把主錄音室的`ready`改為`null`，預期動作拒答、資訊判斷為`no`、評分拒答。第二份讓正確的主錄音室`record`不在候選中，預期拒答。這些是可操作例子，不是新增的未見測試成績。複製其中一份到自己的新路徑，再用同一個`--request`選項讀取即可。

預設五個候選都指向目標裝置。自訂`candidates`時可列1–7個物件，各自明列`id/device/action`；`device`必須是已提供的裝置，`action`限`stop/wait/record/play/save`。候選ID不可重複或使用`none`；同一裝置同一動作也不可重複。範例中候選`b`雖寫`record`，卻屬於播放台，不能當成主錄音室的正確動作。

## 怎樣讀結果

結果的`product`為`C-v3`，`dispatch`為`typed-devices`，表示入口按公開欄位選擇服務。這不是另訓練的路由模型，也不代表支援任意文字請求。

`results.choice.status`是`accepted`時，`selected`才是建議採用的候選ID。`abstained`表示拒答，`selected`為`null`，`reason`說明原因。`model_choice`保留模型判斷；`none`表示資訊不足或沒有合適選項。`confidence`是固定溫度校準後的分數，不能保證任何新情境都同樣可靠。

`results.info.selected`為`yes/no`，兩種完整答案的`status`都叫`prediction`；`results.rubric.selected`是字串`0/1/2`，缺欄時也會拒答。這兩型供查看，不套用動作的採用門檻。本程式只提出建議，不會實際控制錄音、播放或儲存裝置。

動作要求許可與就緒都已知：無許可先`stop`，否則未就緒就`wait`，兩者皆真才做任務。資訊題對`record`要求兩欄、`play`只要求許可、`save`只要求就緒；已知的`false`也算資訊完整。評分的兩欄都須已知：record數true欄位，得0／1／2；play在許可為true時得2，否則按就緒給0或1；save在就緒為true時得2，否則按許可給0或1；缺欄則拒答。這是固定裝置規準，機制與完整算例在[19.25](../../course/chapters/19.md#19.25)。

## 驗收支持到哪裡

新最後測試包含1,500個未用於訓練、選模、校準或政策選擇的4–6裝置事實家族。每個家族各有三型問題，Choice、info、rubric各答對1,500／1,500；三列共用家族，不是4,500個獨立世界。1,025筆可回答的動作全被接受且零錯誤，其餘475筆正確拒答。接受比例68.3%以全部1,500筆為分母；可回答題的接受比例則是100%。

另外的手工檢查涵蓋全部27種單裝置事實、3–6裝置的各個位置、兩種服務語言、候選移除／重排與快取重用。三裝置也核對原先已看過的60個世界，屬於新介面的語義回歸；它不是另一份未見三裝置的正式測試，也不是舊文字API相容聲明。零觀察錯誤不等於風險為零；本次1,025筆接受的單側95% Wilson風險上界約0.263%，用途與近似統計限制見[19.25](../../course/chapters/19.md#19.25)。

[固定驗收摘要](v3/qualification-summary.json)保留分母、指紋與範圍；溫度及門檻在下載後的`policy.json`，並保存在ZIP的`evidence/selection.json`。完整證據與三項成品審閱在Release ZIP的`evidence/`及`MANIFEST.json`。原始公開封包的推論權重只有2,818,540 bytes；Python與PyTorch的安裝量、執行記憶體另計。本版的自由文字、任意新增任務、GPU及MPS成績均沒有由上述CPU測試建立。
