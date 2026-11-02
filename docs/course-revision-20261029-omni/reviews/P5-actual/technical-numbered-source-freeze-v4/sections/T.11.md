## T.11 留下別人能核對的實驗紀錄

隔一週回頭看「這次比較好了」，你可能忘了改哪個地方、用哪些題目。記錄要把材料、做法和結果接起來，才方便自己檢查或讓別人重做。

用量化比較作例子，開始前先記下：

| 要留下的內容 | 用途 |
| --- | --- |
| 原始模型與程式版本 | 確認兩版的共同起點 |
| 資料來源、授權、切分與指紋 | 確認同一批可使用的題目 |
| 新版的量化轉換命令，以及原版、新版各自的評估命令 | 確認主要改動是位數，兩版用同條件評估 |
| 生成上限、裝置與精度 | 確認比較條件 |
| 逐題目標、回答與停止原因 | 確認內容、格式、結束，以及題目分母 |
| 檔案大小與實際量測範圍 | 確認儲存、記憶體、時間各指什麼 |

SHA-256是檔案指紋，Git commit是程式版本；兩者讓你取得同一份內容，不會自動證明資料正確或模型有能力。未量過的欄位就填未量測。

例如先訂「4-bit檔案較小，且同批題目答對數不減」。假設原版80,000bytes、4/5題對，新版60,000bytes、3/5題對，它符合保存變小，沒有符合保住答對數。這是手寫判準例子，數字不是模型實測。

比較時也留原始逐題結果。兩版都4/5，可能錯在不同題；一版先丟掉難題，也不能和另一版的全檔比例直接比較。驗證題可以用來選設定，測試題留到選完後才開啟。

紀錄做到能回答「改了什麼、哪個條件固定、結果支持哪件事」，就能完成一次有限比較。文字、圖解或排版修訂本身不需要重跑訓練；模型、資料或能力宣稱改變時，再做相應驗收。

本輪新增的影音對齊、有限音調與C-v2，在[19.21–19.24](chapters/19.md#19.21)直接內建必要程式、輸入、關鍵輸出、資料切分及真實驗收；Colab只供重跑小例子。十個正式工作已在標準Linux CPU完成訓練並保存最佳及最後續訓狀態：六個影音模型、兩個候選判斷器、codec與Talker。看過開發資料選好權重、T及tau後才量final，沒有因C未達標而重訓或換門檻。本頁前面的既有配方不會自動訓練這些新模組。

新版配套使用固定版本的GitHub Release資產，分清「可推論權重」與「連同最佳化器、隨機狀態保存的完整續訓檢查點」，另附資料與版本清單。四個預選模組的safetensors權重合計2,192,940 bytes，約2.09 MiB；這只算參數檔，沒有量測整個Python／PyTorch的執行記憶體。以下可下載並以CPU操作，不必重跑正式訓練：

```bash
uv run --extra cpu python -m scripts.omni_revision.public_download --output checkpoints/omni-products-v1
uv run --extra cpu python -m scripts.omni_revision.public_demo choice --model-root checkpoints/omni-products-v1
```

影音與聲音操作、實際輸出及完整檢查點選項見[新版模型操作](../docs/omni-products/README.md)。新版C的固定final未過候選判斷與接受風險門檻，這裡明確下載研究候選，不更換舊版C預設；[9.18](chapters/09.md#9.18)已用這次失敗解讀固定校準的限制。自製音調／節奏的訓練沒有使用Qwen或Kev作教師，成熟模型研究另見[20.16](chapters/20.md#20.16)。既有HF固定revision與歷史結果不覆寫。

<details>
<summary>共同成品、成熟模型與完整歷史紀錄</summary>

[第19章](chapters/19.md)追蹤我們自行訓練的 MoE 助理，以及各階段載入的父檢查點（定義見[T.4](training.md#T.4)），階段接續見[19.4](chapters/19.md#19.4)。它的 v2 版已完成訓練、公開推論權重，並在固定的測試資料上評估；[19.12](chapters/19.md#19.12)列出能力及未達標項目，[公開CPU操作](../docs/selftrained/v2-public-cpu-commands.md)提供試用，[本機訓練指引](../docs/selftrained/TRAINING.md)用Git LFS保存的固定資料逐段接自己的訓練檢查點，不需要作者的私有雲端儲存空間。其結果與局部實驗分開。[第20章](chapters/20.md)是沿用成熟Qwen／Whisper的延伸，專用操作見[重訓指引](../docs/natural-assistant/v4/TRAINING.md)，上游能力不算成我們從零學會。

第 19 章另外從隨機參數訓練了 Looped 解題器與機率判斷器兩個成品；用途見[19.13](chapters/19.md#19.13)，實測成績與限制見[19.20](chapters/19.md#19.20)，下載、命令與本機重訓入口見[產品操作](../docs/lesson-products/README.md)。能跑通命令，不代表所有任務都可靠。

完整歷史結果在[實驗紀錄](../docs/course-experiments/README.md)。RAG、工具與推理的實驗紀錄（JSON）保存各自的原始輸入、輸出及分母。工具要留實際執行結果，以及把結果放回對話後模型生成的回答；只生成請求文字，還沒有完成操作。

[工具實驗紀錄（JSON）](../docs/course-experiments/results/tools.json)的每一步，`generation.input_ids` 才是模型當時真正收到的輸入；同一步的 `generation.messages` 有些混入了後來才產生的訊息，不能當成當時的輸入。逐步核對時請對照[工具紀錄勘誤](../docs/course-repair-20261008/tool-trace-errata.md)，它附有重建後的當時文字。

</details>
