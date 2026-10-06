## T.4 把文字訓練和對話練習分開看

屬性資料可以寫成短文，也可以寫成問題與答案。文字階段練習接續原文；對話階段則把助手回答作為示範。下面`toy-text`準備文字，`attributes-sft`準備屬性問答。固定隨機種子`seed=42`讓資料切分與模型起點可重做，42本身不是能力分數。

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42
```

產物在 `data/generated/`，各有train（更新參數的題目）、validation（比較前後或選設定的驗證題）、test（設定選完後才看的最後題）和manifest（記錄這次資料來源、切分與內容指紋的清單）。用途見[5.10](chapters/05.md#5.10)。下面兩筆是不同屬性的材料，只比較主要格式；`family`是分組標籤，把同一素材衍生的問題放在一起：

```jsonl
{"text":"color=red;shape=circle;side=left.","family":"color=red;shape=circle;side=left."}
{"messages":[{"role":"user","content":"color=red;shape=circle;pitch=low;shape?"},{"role":"assistant","content":"circle"}],"family":"red:circle:low"}
```

第一行練下一文字單位；第二行的問題提供線索，只有助手的`circle`與EOS結束目標計回答代價，結束標記見[7.9](chapters/07.md#7.9)。同一屬性家族的五個要求都放在同一份train、validation或test，避免教過同一組素材的一種問法，再把另一問法當全新素材。預設文字是train 9篇、validation 1篇、test 2篇；問答是train 45題、validation 5題、test 10題。

先保存同一個未訓練起點。把下段存為 `outputs/save_start.py`，用 `.venv/bin/python outputs/save_start.py` 執行：

```python
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything

seed_everything(42)
model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
save_checkpoint("checkpoints/start.pt", model, step=0, metadata={"seed": 42, "note": "untrained baseline"})
print("保存隨機起點；沒有更新參數")
```

這個模型每個位置有 32 個特徵、一層、一個注意力頭，最多 128 個位置。保存起點是為了前後比較；沒有更新器狀態，不是中斷後的續訓檔。

文字路線先評估、更新、再評估：

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

若要沿用文字階段再教對話，把 SFT 的 `--checkpoint` 改成 `checkpoints/text.pt`，另取輸出檔名；同時先評估那份文字權重作為對話的起點。這是新的階段，載入父權重並開新更新工具，不加 `--resume`。兩階段的理由見[7.17](chapters/07.md#7.17)，共同成品的接續見[19.4](chapters/19.md#19.4)。

評估後打開 `--output` 指定的完整 JSON。主要欄位如下：

| 欄位 | 要核對什麼 |
| --- | --- |
| `mean_token_nll`、`effective_tokens` | 有效位置的平均代價，以及實際計分位置數 |
| `exact_match` | SFT 回答內容 ID 是否和目標一致；不刪空白或藏非法角色標記 |
| `completed_exact_match`、`eos_rate` | 內容匹配且正常結束；另記主動結束比例 |
| `samples` | 逐題的目標、實際生成、原始 ID 與停止原因 |
| `skipped`、`metric_denominators` | 哪些題未完成評估，以及各指標的實際分母 |

文字模式沒有標準問答，因此匹配率為 `null`。SFT 的 `row=0` 對應同一份 JSONL 的第一筆；從該筆 `messages` 找問題，再和 `samples` 的目標及生成並排。`--limit all` 選全檔，仍要確認是否有跳過題。`--tokens` 算新增 byte／結構單位，不等於中文字數；預算與結束見[7.19](chapters/07.md#7.19)。

用同樣的資料、題號、生成上限比較前後。練習先指出平均代價是否改變，再從逐題回答找一個有意義的變化；不只挑一個變好的答案，也不以提高上限遮住結束問題。

<details>
<summary>故事、切詞與固定對話實驗</summary>

固定課程配方使用與上述小型 CPU 練習不同的設定，不要混用成績。按需要選一組：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset tinystories --asset chinese-poetry
.venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset ultrachat-sft
.venv/bin/python -m scripts.course_experiments.run --experiment sft --device cuda
```

各組在 `outputs/course-experiments/course-v1/<組名>/` 保存結果、權重和原始切分。`sft/model.pt` 是直接對話版；`pretrain.pt` 與 `pretrain-sft.pt` 是另一條兩階段支線。後面的固定比較若依賴 `sft/`，先完成這個完整入口，不能拿本節 `checkpoints/attributes.pt` 當成相同模型。

BPE 權重必須配對它的 tokenizer JSON，推論要加 `--tokenizer`。不同切詞的 ID 不能互換，來源與完整步驟見[第 6 章](chapters/06.md)及[實驗入口](../docs/course-experiments/README.md)。要比較錯標與回放，先完成固定 `sft`，再執行 `--experiment sft_ablation`。

</details>

