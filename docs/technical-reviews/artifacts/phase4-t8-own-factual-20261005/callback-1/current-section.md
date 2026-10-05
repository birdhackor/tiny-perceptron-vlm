## T.8 架構比較一次只換一個條件

先保存基準，再只換一個零件。若同時換資料、寬度與步數，便很難知道差異來自哪裡。下面固定同一份短文與200次更新，基準是預設的LayerNorm正規化與共用FFN；RMS版只換成不先減平均的均方根縮放，方法見[14.2](chapters/14.md#14.2)。MoE版則將共用FFN換成四個專家FFN，每個文字位置由路由選兩個處理，角色見[15.1](chapters/15.md#15.1)與[15.4](chapters/15.md#15.4)。

三版使用入口預設寬度32，也就是每個文字位置保存32個特徵數字。`--experts 4`指定專家總數，`--top-k 2`指定每次選擇數：

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --output checkpoints/baseline.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --norm rms --output checkpoints/rms.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --experts 4 --top-k 2 --output checkpoints/moe.pt
```

MoE每次選兩位專家，但全部專家權重仍需保存。同寬度不等於同參數或同計算預算。至少記錄參數總數、更新次數、每批題數與有效計分位置數，再另量訓練時間；共同200步不代表各步工作量相同，記錄範圍見[5.9](chapters/05.md#5.9)。接著用同一份validation驗證題評估三版：`--tokens 32`限制每題新增生成單位，`--limit all`選全檔，但仍須查看是否有跳過題。

```bash
.venv/bin/python scripts/evaluate.py checkpoints/baseline.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/baseline-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/rms.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/rms-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/moe.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/moe-validation.json
```

把平均代價、有效位置數與逐筆生成並排。平均代價`mean_token_nll`是有效下一項目標的平均預測代價；`effective_tokens`是實際計分位置數，忽略位置不計入，結束目標可計入。`samples`讓你把同一筆目標、生成與停止原因對上，欄位用途見[T.4](training.md#T.4)。

用手寫材料`text="AB"`示範判讀：輸入A，目標後文是B再接EOS，假設只計這兩個有效位置。下表是人工欄位例，不是三條命令的實測結果：

| 版本 | 平均代價 mean_token_nll | 有效位置數 effective_tokens | 生成後文 | 停止 |
| --- | --- | --- | --- | --- |
| 原版（手設） | 0.8 | 2 | B | EOS |
| 另一版（手設） | 0.6 | 2 | B | EOS |

另一版在這兩個位置的平均預測代價較低，但兩版生成內容與結束相同；這沒有證明新題能力或速度改善。實際比較另記參數量、裝置與時間。品質、儲存和實際速度各有自己的單位，不能只用「比較快」替整個設計下結論。先在驗證側選設定，再使用尚未拿來選設定的最後測試，資料角色見[5.10](chapters/05.md#5.10)。

<details>
<summary>現代零件與效率的固定配方</summary>

按[T.1](training.md#T.1)取得TinyStories包後，可選`--experiment modern`或`--experiment moe`重做固定比較，零件分別見[14.2](chapters/14.md#14.2)與[15.4](chapters/15.md#15.4)。下面的效率組另依賴T.4的完整`sft/model.pt`和`dataset.json`；precision組沿用同一對話起點，對照FP32、BF16與FP16的計算精度，意義見[16.7](chapters/16.md#16.7)。`--device cuda`指定相容的NVIDIA GPU，是否支援某精度須看報告，未執行不能算通過：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment efficiency --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment precision --device cuda
```

### 選讀：效率實驗的量測條件

推論先暖機，再在計時兩側等待 GPU 完成；保存多次採樣與中位數。記錄提示長度、生成長度、精度、後端與工作範圍。訓練計時是否含反向和更新，也要寫清楚。

PyTorch已配置記憶體是交給張量的空間，不等於整張GPU用量。保存量測前基線、過程峰值與新增量；不要只比較新增量便宣稱整張卡節省同樣倍數。詳細操作見[16.1](chapters/16.md#16.1)與[效率報告](../docs/course-experiments/results/efficiency.json)。

FlashAttention是分塊安排注意力計算、減少中間表儲存與搬移的方法，見[16.9](chapters/16.md#16.9)。PyTorch的SDPA是呼叫這套注意力計算的介面，實際可能選不同後端，見[16.8](chapters/16.md#16.8)。要獨立核驗是否用了Flash實作，執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment flash_probe --device cuda
```

它只測固定Q/K/V的注意力前向與反傳；Q用來查詢、K供比較、V是混合的內容數字，見[3.4](chapters/03.md#3.4)，不訓練整個模型。報告的`results.verification_passed=true`只表示`supported_routes`至少有一條已核驗Flash後端的路線，而且這個集合內的路線都已完成；其他路線仍可能失敗。先查看各路線的`status`、後端核驗與數值容差，再以`results.schedule_completed`和`results.status`判讀整次探針是否完成，以及是否有不支援的路線。未執行或未完成時，先看原因，不據此聲稱使用成功。還要核對記憶體範圍，不能由SDPA這個API名稱推定已用Flash；完整條件在[原報告](../docs/course-experiments/results/flash_probe.json)。

</details>

