## T.9 真的縮小保存格式，再量品質

先讀[17.2的整數刻度](chapters/17.md#17.2)、[17.8的低位元保存](chapters/17.md#17.8)。把浮點數四捨五入後仍存在原本的浮點容器，只是改了數值，檔案不一定變小。現在用已完成的屬性問答實驗核對真正4-bit／8-bit儲存；先完成[T.4的課程SFT入口](#T.4)，讓`outputs/course-experiments/course-v1/sft/`中有`model.pt`與原始三側`dataset.json`。

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment quantization --device cpu
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/quantization/fp32.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --json
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/quantization/model.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --json
```

第一個命令載入直接SFT底座，額外更新120次，batch16、學習率0.003、seed42，再把同一份更新後權重各自轉成4-bit與8-bit。它保存`fp32.pt`、4-bit的`model.pt`、8-bit的`packed8.pt`及完整`result.json`，包含原45／5／10筆的訓練、驗證與最後測試切分。CPU可重跑流程，正式數字則來自NVIDIA L4；改用`--device cuda`需要自己的CUDA環境，裝置與時間不能混抄。後兩條命令用同一提示檢查FP32與4-bit的實際答案，`--json`保留原始ID、EOS與非法控制標記。本輪兩版都答`re`，標準答案`blue`，載入與停止都正常，內容仍錯。

4-bit指每個量化整數用四個位元。Linear把一組特徵加權混合成另一組，見[4.1](chapters/04.md#4.1)。轉換只動13個Linear的權重；嵌入查表、正規化與bias仍是FP32，scale也要保存。因此三版141,568個參數的個數相同，保存方式改了，不是學生架構縮小。

| 本次共同來源的版本 | 模型tensor bytes | 私有實跑檔案bytes | 測試完整匹配／題數 | 回答NLL／有效目標 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 566,272 | 588,358 | 6/10 | 0.4565／69 |
| 4-bit | 168,736 | 182,277 | 6/10 | 0.4278／69 |
| 8-bit | 226,336 | 242,085 | 6/10 | 0.4554／69 |

NLL是平均負對數代價，越低表示這批標準答案的機率較高；69個目標包含回答與EOS，不含問題。三版EOS皆10/10，完成匹配也都是6/10，原始生成ID逐項相同。但驗證五題的FP32／4-bit／8-bit是2／1／2題正確，不能只憑測試這一欄宣稱四位元保住所有新題。[公開完整報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/quantization.json)的`results.runs`逐版保存`storage`、`validation`、`test`與`timing`，實際更新總共讀到13,610個有效目標；`fp32-training.pt`另保存optimizer，不拿它與部署檔作大小比較。

推論目前會把緊密保存的整數還原成浮點數再計算。它可以減少檔案大小，卻不能直接宣稱推論記憶體同樣減少或運算更快。

本次底座本來就不共享輸入與輸出表，沒有額外拆開共享語意。若你另有訓練好的浮點模型，可用`.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt`獨立轉換；這條命令不做上述120次更新，不能套用本表成績。它的`input_output_sharing_removed`會指出是否拆開共享表，見[17.9](chapters/17.md#17.9)。原始checkpoint可能保存optimizer與隨機狀態；本表FP32雖沒有optimizer，仍有隨機狀態及不同metadata。只比較數值儲存用tensor bytes；私有原檔與公開剔除訓練狀態後的匯出檔大小也要各量一次。

要讓模型預先適應誤差，可看[17.14的量化感知訓練](chapters/17.md#17.14)，常縮寫 QAT。那節用模擬量化檢查前向誤差與近似梯度，真正的低位元加速還需要支援的格式與硬體運算核心。

課程報告的`storage.file_bytes`與`storage.tensor_bytes`分別對應本表的檔案與數值大小；`test.generated_samples`用`family`、`question`、`expected`、`generated_ids`與`exact`逐題核對。這套欄名與單一`quantize.py`命令的`source_file_bytes`／`quantized_file_bytes`不同，讀報告時要使用實際欄位。本輪L4固定八步decode，FP32約2.371毫秒／步、4-bit約4.882毫秒／步，參考反量化路徑較慢；完整暖機、提示長度與共存模型記憶體界線見[17.10](chapters/17.md#17.10)。

練習保存原版和4-bit版的實際檔案大小，再用相同驗證題逐題比較答案。把「少了多少 bytes」和「多少題改變」分成兩欄；bytes 是位元組；一個位元組包含八個位元，也就是 `1 byte = 8 bits`。這兩欄不能互相取代。

