## T.9 真的縮小保存格式，再量品質

把數字四捨五入後仍存成浮點數，未必能縮小檔案。先從同一份已訓練浮點模型，各自轉換不同位數，再檢查保存大小與相同題目的回答。

下面兩條都從 [T.4](#T.4) 對話路線保存的原始 `checkpoints/attributes.pt` 出發，不把8-bit再轉成4-bit。本節命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 8 --output checkpoints/attributes-int8.pt
```

轉換只改支援的 Linear 權重，其他部分照原樣以浮點數保存。各部分在量化裡的角色如下，概念見[2.3](chapters/02.md#2.3)與[17.7](chapters/17.md#17.7)：

- **Linear 權重**：混合輸入數字的乘數；這是被換成低位元整數碼的部分。
- **embedding 表**：用 ID 查出向量的數字表，保留浮點。
- **正規化層的參數**：調整數字尺度，保留浮點。
- **偏移（bias）**：每個輸出另加的數，保留浮點。
- **刻度（scale）**：轉換時新增的倍數，把存下的整數碼乘回浮點近似值（還原值）。

這些保留部分與刻度都要儲存。本課推論會先把碼換回浮點數再計算，所以檔案變小沒有直接證明執行時記憶體同樣變小或更快。

下面兩次推論都沿用本工具的預設 `--temperature 0`，也就是[1.15的greedy規則](chapters/01.md#1.15)：每步直接取最高分候選，不做隨機抽樣。題目、新增上限24個token與CPU裝置也相同，這樣比較原版與量化版時就固定了選字條件；是否維持回答仍要看實際輸出。

```bash
.venv/bin/python scripts/infer.py checkpoints/attributes.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py checkpoints/attributes-int4.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
```

希望答案是`blue`。這一題先讓你查看前後有沒有改變；完整能力檢查仍用同一份驗證題。原模型若本來就沒答 `blue`，量化後也不必是 `blue`：這一題比的是前後兩次的答案是否一致。原訓練檔可能含最佳化器狀態與隨機狀態，不能把訓練狀態的差異也算作量化收益。

本機練習先比較純模型張量 bytes，不要求另做原版部署匯出。用任何純文字編輯器把下段存成 `outputs/size_check.py`，再用 `.venv/bin/python outputs/size_check.py` 執行，應印出三行，每個檔案一行。它只讀剛才的三個檔案：`model` 保存模型數字，最佳化器與其他訓練狀態在它之外。每份模型張量的格數乘每格 bytes，再加總；量化版的低位元碼、刻度與保留的浮點張量都計入。

```python
from pathlib import Path
import torch

paths = [
    Path("checkpoints/attributes.pt"),
    Path("checkpoints/attributes-int4.pt"),
    Path("checkpoints/attributes-int8.pt"),
]
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    tensor_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(path.name, "模型張量bytes", tensor_bytes, "整個檔案bytes", path.stat().st_size)
```

`weights_only=True` 讓 `torch.load` 只讀出張量、數字、文字這類資料，不執行檔案裡可能夾帶的其他 Python 物件，讀別人給的檔案時比較安全。`numel()`數張量格數，`element_size()`給每格bytes。這裡按模型狀態表的每個項目加總，不計最佳化器或檔案包裝；也不是執行時記憶體或速度量測。若改比較完整部署檔大小，三版就須使用同用途、同樣不含訓練狀態的匯出方式。

接著使用T.4的同一份屬性validation題組，評估三版；每題新增上限都是24個 token，`all`選全檔：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-fp32-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int4-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int8.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int8-validation.json
```

在三份JSON中用相同`row`找回同一筆提問與目標，再並排`samples`裡的生成、原始ID與停止原因，欄位見[T.4](training.md#T.4)。逐題判內容匹配及正常結束，記「新增答對、新增答錯與不變」，相同總答對數可能是不同題目答對。先用validation選位數；設定選完，才用下面三條命令評估測試題。它們和上面三條只差資料路徑改成 `test.jsonl`、加上 `--split-label test`，以及另取輸出檔名；測試題不拿來重新選設定：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/test.jsonl --split-label test --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-fp32-test.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/test.jsonl --split-label test --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int4-test.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int8.pt --data data/generated/attributes-sft/test.jsonl --split-label test --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int8-test.json
```

練習把三版模型張量大小與逐題變化放在同一張表，分別說明「保存變小了嗎」與「相同題目是否維持回答」。尚未量到的執行記憶體與速度就留未量測。

<details>
<summary>固定量化與量化感知訓練</summary>

固定比較先完成T.4的完整`sft`，再跑：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment quantization --device cpu
```

作者的紀錄是在 Modal 的 NVIDIA L4 上用 `--device cuda` 跑的，7.8 秒；照上面用 CPU 跑的時間未量測。這個配方另做120次更新才轉換，和上面的單獨 `quantize.py` 不同；不能互換成績。它保存浮點、4-bit、8-bit與共同題目，格式見[量化實驗紀錄（JSON）](../docs/course-experiments/results/quantization.json)。[17.14](chapters/17.md#17.14)介紹QAT：讓訓練先適應模擬的量化誤差；真正的低位元 kernel（計算程式）仍需格式與硬體支援。

</details>

