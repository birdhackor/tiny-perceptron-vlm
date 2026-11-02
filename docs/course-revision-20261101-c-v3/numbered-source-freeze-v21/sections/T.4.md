## T.4 把文字訓練和對話練習分開看

屬性資料可以寫成短文，也可以寫成問題與答案。文字階段練習接續原文；對話階段則把助手回答作為示範。下面`toy-text`準備文字，`attributes-sft`準備屬性問答。固定隨機種子`seed=42`讓資料切分與模型起點可重做，42本身不是能力分數。

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42
```

產物在 `data/generated/`，各有train（更新參數的訓練題）、validation（比較前後或選設定的驗證題）、test（設定選完後才看的測試題）和manifest（記錄這次資料來源、切分與內容指紋的清單）。用途見[5.10](chapters/05.md#5.10)。下面兩筆是不同屬性的材料，只比較主要格式；`family`是分組標籤，同一家族的題目放在一起：

```jsonl
{"text":"color=red;shape=circle;side=left.","family":"color=red;shape=circle;side=left."}
{"messages":[{"role":"user","content":"color=red;shape=circle;pitch=low;shape?"},{"role":"assistant","content":"circle"}],"family":"red:circle:low"}
```

第一行練下一個 token；本課的 byte tokenizer 把一個 UTF-8 byte 或一個特殊 token（例如 EOS）算成一個 token。第二行的問題提供線索，只有助手的`circle`與EOS結束目標計回答代價，結束標記見[7.9](chapters/07.md#7.9)。同一家族的五種問法都放在同一份train、validation或test，避免教過同一組素材的一種問法，再把另一問法當全新素材。預設文字是train 9篇、validation 1篇、test 2篇；問答是train 45題、validation 5題、test 10題。

先保存同一個未訓練起點。若直接從本節開始，先依[T.3的建夾步驟](#T.3)建立`outputs/`與`checkpoints/`。用任何純文字編輯器（例如 VS Code）把下段存成 `outputs/save_start.py`，再用 `.venv/bin/python outputs/save_start.py` 執行：

```python
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything

seed_everything(42)
model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
save_checkpoint("checkpoints/start.pt", model, step=0, metadata={"seed": 42, "note": "untrained baseline"})
print("保存隨機起點；沒有更新參數")
```

這個模型特徵寬度 32、一層、一個注意力頭，最多 128 個位置。保存起點是為了前後比較；檔案裡沒有最佳化器狀態，不能拿來從中斷處精確接續訓練。

文字路線先評估、更新、再評估。本節的訓練與評估命令在你電腦上的時間未量測；`train.py` 報告的 `seconds` 欄會記下你這次的訓練迴圈時間。

以下的`evaluate.py`生成答案時固定採用**greedy（貪婪選取）**：每一步取目前機率最高的token，再接著生成下一步，不做隨機抽樣。這個工具目前沒有切換選字方式的命令列參數；前後比較使用同一份驗證題與這項選字規則。

```bash
.venv/bin/python scripts/evaluate.py checkpoints/start.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --device cpu --limit all --output outputs/text-before.json
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --checkpoint checkpoints/start.pt --seed 42 --device cpu --train --steps 200 --output checkpoints/text.pt
.venv/bin/python scripts/evaluate.py checkpoints/text.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --device cpu --limit all --output outputs/text-after.json
```

想直接練習對話，則從同一個隨機起點走另一支：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/start.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/attributes-before.json
.venv/bin/python scripts/train.py --task sft --data data/generated/attributes-sft/train.jsonl --checkpoint checkpoints/start.pt --seed 42 --device cpu --train --steps 500 --output checkpoints/attributes.pt
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/attributes-after.json
```

若要沿用文字階段再教對話，起點就換成文字路線保存的 `checkpoints/text.pt`。先評估它當作對話的起點，再從它開始 SFT，最後評估；輸出檔名另取，才不會蓋掉上面兩支的結果：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/text.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/text-then-sft-before.json
.venv/bin/python scripts/train.py --task sft --data data/generated/attributes-sft/train.jsonl --checkpoint checkpoints/text.pt --seed 42 --device cpu --train --steps 500 --output checkpoints/text-then-sft.pt
.venv/bin/python scripts/evaluate.py checkpoints/text-then-sft.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/text-then-sft-after.json
```

這是新的階段。`text.pt` 在這裡是**父檢查點**：上一階段保存、這一階段載入當起點的檢查點。工具只載入它的參數，另外建立新的最佳化器，所以不加 `--resume`；`--resume` 是同一次訓練中斷後接續用的。兩階段的理由見[7.17](chapters/07.md#7.17)，共同成品的接續見[19.4](chapters/19.md#19.4)。

評估後打開 `--output` 指定的完整 JSON。主要欄位如下：

| 欄位 | 要核對什麼 |
| --- | --- |
| `mean_token_nll`、`effective_tokens` | 有效目標的平均代價（負對數似然，negative log-likelihood，NLL），以及有效目標數 |
| `exact_match` | SFT 回答內容 ID 是否和目標一致；不刪空白或藏非法角色標記 |
| `completed_exact_match`、`eos_rate` | 內容匹配且正常結束；另記主動結束比例 |
| `samples` | 逐題的目標、實際生成、原始 ID 與停止原因 |
| `skipped`、`metric_denominators` | 哪些題未完成評估，以及各指標的實際分母 |

文字模式沒有一問一答的正確答案，因此匹配率為 `null`。SFT 的 `row=0` 對應同一份 JSONL 的第一筆；從該筆 `messages` 找問題，再和 `samples` 的目標及生成並排。`--limit all` 選全檔，仍要確認是否有跳過題。`--tokens` 算的是新增 token 數，不等於中文字數：一個常見中文字在 UTF-8 佔 3 個 byte，也就是 3 個 token，見[6.1](chapters/06.md#6.1)；預算與結束見[7.19](chapters/07.md#7.19)。

用同樣的資料、題號、生成上限比較前後。練習先指出平均代價是否改變，再從逐題回答找一個有意義的變化；不只挑一個變好的答案，也不以提高上限遮住結束問題。

<details id="fixed-sft-recipe" markdown="1">
<summary>故事、切詞與固定對話實驗；T.10教師需先完成其中的固定SFT配方</summary>

固定課程配方使用與上述小型 CPU 練習不同的設定，不要混用成績。按需要選一組：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset tinystories --asset chinese-poetry
.venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset ultrachat-sft
.venv/bin/python -m scripts.course_experiments.run --experiment sft --device cuda
```

| 配方 | 重做什麼 | 作者的紀錄與時間（Modal 上的 NVIDIA L4） |
| --- | --- | --- |
| `text_foundation` | 9 篇顏色與形狀短文的文字訓練，600 次更新 | [text_foundation.json](../docs/course-experiments/results/text_foundation.json)，16.0 秒 |
| `real_text` | TinyStories 與唐詩各自訓練 800 次 | [real_text.json](../docs/course-experiments/results/real_text.json)，23.9 秒 |
| `tokenizer` | byte 與 BPE 兩種切詞各訓練 400 次 | [tokenizer.json](../docs/course-experiments/results/tokenizer.json)，16.2 秒 |
| `sft` | 屬性問答 SFT 900 次，另含先練續寫再 SFT 的支線與 UltraChat 小型實跑 | [sft.json](../docs/course-experiments/results/sft.json)，31.7 秒 |

沒有 NVIDIA GPU 時，可以把 `--device cuda` 改成 `--device cpu`，但 CPU 可行性未量測：能不能跑完、要多久都沒有紀錄。指定 `cuda` 而電腦沒有 CUDA 時，工具會直接停下，不會自動改用 CPU；這個工具也只接受 `cpu` 與 `cuda`，不支援 Apple 的 MPS。

各組在 `outputs/course-experiments/course-v1/<組名>/` 保存結果、權重和原始切分。`sft/model.pt` 是直接對話版；`pretrain.pt` 與 `pretrain-sft.pt` 是另一條兩階段支線。後面的固定比較若依賴 `sft/`，先完成這個完整配方，不能拿本節 `checkpoints/attributes.pt` 當成相同模型。

BPE 權重必須配對它的 tokenizer JSON，推論要加 `--tokenizer`。不同切詞的 ID 不能互換，來源與完整步驟見[第 6 章](chapters/06.md)及[實驗說明](../docs/course-experiments/README.md)。要比較錯標與回放，先完成固定 `sft`，再執行 `--experiment sft_ablation`（作者在 L4 上的紀錄是 23.5 秒，見 [sft_ablation.json](../docs/course-experiments/results/sft_ablation.json)）。

</details>

