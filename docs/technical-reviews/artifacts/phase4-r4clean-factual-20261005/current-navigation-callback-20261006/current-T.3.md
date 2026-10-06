## T.3 先訓練接字表，再比較固定窗口

接字表只看前一字；固定窗口 MLP 可以同時讀幾個字。先用相同短文與切分，分別保存更新前、更新後的代價，看看猜法是否改變。

預設資料是十二篇顏色與形狀短文。工具先去掉完全重複的文件、按整篇切分，再取窗口；種子 42 的訓練／驗證／測試是 9／1／2 篇。字表只看訓練側。MLP的`--context 3`表示看前三個字，`--width 16`讓每字查出16個特徵數字，中間層也使用16格。先建立輸出資料夾，再取得未更新的起點：

```bash
mkdir -p outputs checkpoints
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu > outputs/bigram-before.json
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu > outputs/mlp-before.json
```

這兩次沒有更新。報告的 `train_loss` 與 `validation_loss` 是兩側平均下一字代價；`parameters` 是可調數字總數，`split_unit` 說明切分單位。

現在先更新接字表：

```bash
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu --train --steps 200 --output checkpoints/bigram.pt > outputs/bigram-after.json
```

`--train` 開啟更新，`--steps` 是次數，`--output` 保存模型。報告中的 `train_loss`、`validation_loss` 使用最後更新後的參數；`last_batch_loss_before_update` 是最後一步更新之前，不能把兩個時點混在一起。

打開前後 JSON，分別計算兩個 loss 的「更新後減更新前」。負值表示代價下降。訓練側下降而驗證側上升，說明這批練習題改善沒有延伸到留出題；不要只挑訓練側看。

接著用相同資料訓練 MLP：

```bash
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu --train --steps 200 --output checkpoints/mlp.pt > outputs/mlp-after.json
```

核對 MLP 自己的前後報告，再比較不同模型。改窗口也會改參數量，不能把所有差異都歸因於前文長度。這些存檔屬於接字表／MLP格式，不能直接交給 Transformer 的 `infer.py`。

<details>
<summary>取得已訓練模型與固定比較</summary>

[公開模型入口](https://huggingface.co/birdhackor/tiny-perceptron-course-models)列出各獨立實驗的模型卡。下面另取`text_foundation`，它是英文規則文字Transformer，和本節中文接字表／MLP不同，使用Transformer的推論入口：

```bash
.venv/bin/python scripts/fetch_course_models.py --list
.venv/bin/python scripts/fetch_course_models.py --model text_foundation
.venv/bin/python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt "color=blue;shape=circle;" --tokens 32 --device cpu --json
```

`--tokens 32`限制新增編碼單位，遇EOS（表示生成結束的專用token）可以提前結束，兩種停止來源見[7.9](chapters/07.md#7.9)。JSON的`generated_ids`保留原始新增ID，`eos`表示是否生成EOS，`generation_status`記停止資訊。例如下面是按本課byte編碼手寫的欄位摘錄，不是上述模型的實測回答：

```json
{"answer":"A","generated_ids":[73,2],"eos":true,"generation_status":"eos"}
```

73表示A，2表示EOS，所以只新增兩個單位就結束，沒有用滿32。`answer`只顯示正文A；只看可讀文字，無法分辨模型主動結束還是到上限截停。這份摘錄也沒有證明A符合原問題；內容與停止要分開檢查。

固定比較包含兩類模型、四個設定：接字表，以及一字、三字、五字視窗的 MLP。視窗取幾個位置的差別見[2.5](chapters/02.md#2.5)；要重做這組比較，用：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu
```

結果在 `outputs/course-experiments/course-v1/simple_models/`，歷史結果與完整設定見[原始報告](../docs/course-experiments/results/simple_models.json)。公開學生包只含推論所需狀態；精確接續原訓練所需的完整檔案，見[5.7](chapters/05.md#5.7)。

</details>

