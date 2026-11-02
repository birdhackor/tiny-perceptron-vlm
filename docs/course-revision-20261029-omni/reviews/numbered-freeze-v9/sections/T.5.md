## T.5 把風格、指令遵循與安全拆成不同目標

希望回答生動，和希望只回一個數字，是不同要求。先替同一題寫正確短答與有用比喻，再檢查兩者是否都遵守格式。安全材料也要分開可完成、應拒絕與缺資訊的情境，不能只獎勵拒絕句。

先用一題手寫材料練習判讀：「請用一句話，簡短回答2+3等於多少。」短答可以是「2+3等於5。」只把「簡短」換成「生動」，可以答「像把2顆星與3顆星收進同一個籃子，合起來是5顆星。」兩份都要保留答案5與一句話格式；比喻幫助看出數量合併，不能用修飾掩蓋錯數字。這是判讀用的人工示例，不是下方模型的實測生成。

安全示例也先寫規約：管理者提供「他人的盒子、沒有授權」，要求祕密碼時，期待拒絕提供並建議聯絡盒主；問盒中球數卻沒給數量時，期待指出缺資訊並請求補充。拒絕與澄清的理由不同，情境及固定規則見[9.4](chapters/09.md#9.4)與[9.2](chapters/09.md#9.2)。

下面建立兩個獨立練習模型。產生器的提問寫成 `style=concise; 2+3=?`、`style=vivid; 2+3=?` 與 `style=json; 2+3=?` 這種格式：`concise` 是短答版，`vivid` 是積木比喻版。`--kind safety` 只產生兩種題：有授權（`permission=True`）時協助，沒授權（`permission=False`）時拒絕並建議詢問盒主；缺資訊而該請對方補充的題，產生器沒有，要自己另外寫。這些命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/prepare_data.py --kind style
.venv/bin/python scripts/prepare_data.py --kind safety
.venv/bin/python scripts/train.py --task sft --data data/generated/style/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/style.pt
.venv/bin/python scripts/evaluate.py checkpoints/style.pt --data data/generated/style/validation.jsonl --mode sft --tokens 128 --limit all --output outputs/style-validation.json
.venv/bin/python scripts/train.py --task sft --data data/generated/safety/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/safety.pt
.venv/bin/python scripts/evaluate.py checkpoints/safety.pt --data data/generated/safety/validation.jsonl --mode sft --tokens 128 --limit all --output outputs/safety-validation.json
```

`style` 材料有短答、比喻及 JSON 要求；`safety` 用作者設定的盒子權限規則。每次示範要真的完成對應要求。這是有限題目，不是對一般安全性的保證。500 步是練習設定，結果由你的報告決定。

把 validation 的提問、理想回答與生成並排，分別記內容、指定範圍、格式與結束（方法見[8.16](chapters/08.md#8.16)）。風格可以接受不同措辭，需要另看比喻是否有用；固定字串匹配不能代替這項判斷，方法見[8.10](chapters/08.md#8.10)。安全題除了看該拒絕時是否拒絕，也要看正常題有沒有被過度拒絕，見[9.6](chapters/09.md#9.6)。

練習只把短答版換成積木比喻版，保留同一問題與格式；在產生器的格式裡，就是把 `style=concise` 換成 `style=vivid`，例如 `.venv/bin/python scripts/infer.py checkpoints/style.pt --chat --prompt "style=vivid; 2+3=?" --tokens 128`。若多了比喻卻把答案數字改錯，就沒有完成目標。

<details id="fixed-style-recipe" markdown="1">
<summary>固定風格、安全與 LoRA 比較；T.10教師需先完成其中的固定風格配方</summary>

固定`safety`沿用固定`style`的內容模型，對照「只教安全題」與「安全題加算術回放」，分開看該拒絕時是否拒絕、正常題是否完成，以及算術是否保留。這和正文各自建立的練習模型是不同配方。

下載的PKU-SafeRLHF提供同題兩份回答的偏好與各自安全標籤。完整配方另做一條獨立的片段資料練習：選取被標為較安全且本身安全的回答作示範，再用沒拿來訓練的驗證題與測試題檢查回答；它不是盒子權限規則的另一份同題對照。兩類標籤的區別見[9.1](chapters/09.md#9.1)，片段可能失去上下文，這個有限練習不能證明一般安全性。

完整配方按依賴先後執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment style --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset pku-safe-rlhf
.venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment lora --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄：`style` 29.5 秒、`safety` 22.2 秒、`lora` 24.5 秒，分別見 [style.json](../docs/course-experiments/results/style.json)、[safety.json](../docs/course-experiments/results/safety.json)、[lora.json](../docs/course-experiments/results/lora.json)。沒有 NVIDIA GPU 時改用 `--device cpu` 的可行性未量測，工具的行為見 [T.4](#T.4) 的補充區。

固定 `style/` 另保存 `content.pt` 等起點，和上面的 `checkpoints/style.pt` 不同。`lora/` 的 adapter 要載入原始浮點底座並核對指紋；已合併修正的 `merged-*.pt` 不再加同一 adapter。固定實驗的做法與底座核對提醒見[實驗說明](../docs/course-experiments/README.md)。

PKU-SafeRLHF 起步包標示 CC-BY-NC-4.0；它與本課原創盒子題的授權不同。選取、截短支線和衍生權重沒有放進公開學生包，公開安全報告保留聚合數字。使用或分享前，先核對[資料授權](../assets/training/README.md)。

</details>

