## T.9 真的縮小保存格式，再量品質

把數字四捨五入後仍存成浮點數，未必能縮小檔案。先從同一份已訓練浮點模型，各自轉換不同位數，再檢查保存大小與相同題目的回答。

```bash
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 8 --output checkpoints/attributes-int8.pt
```

兩份都從T.4產生的原始`attributes.pt`出發，不把8-bit再轉成4-bit。轉換只改支援的Linear權重，也就是混合輸入數字的乘數；嵌入是ID查回的數字表，正規化調整數字尺度，偏移是每個輸出另加的數，刻度則把整數碼換回浮點近似，角色見[2.3](chapters/02.md#2.3)與[17.7](chapters/17.md#17.7)。這些保留部分與刻度都需儲存。本課推論會先反量化成浮點數，所以檔案變小沒有直接證明執行記憶體同樣變小或更快。

```bash
.venv/bin/python scripts/infer.py checkpoints/attributes.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py checkpoints/attributes-int4.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
```

希望答案是`blue`。這一題先讓你查看前後有沒有改變；完整能力檢查仍用同一份留出題。原訓練檔可能含更新器與隨機狀態，不能把訓練狀態的差異也算作量化收益。

本機練習先比較純模型張量bytes，不要求另做原版部署匯出。以下只讀剛才的三個檔案：`model`保存模型數字，更新器與其他訓練狀態在它之外。每份模型張量的格數乘每格bytes，再加總；量化版的低位元碼、刻度與保留浮點張量都計入。

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

`numel()`數張量格數，`element_size()`給每格bytes。這裡按模型狀態表的每個項目加總，不計更新器或檔案包裝；也不是執行時記憶體或速度量測。若改比較完整部署檔大小，三版就須使用同用途、同樣不含訓練狀態的匯出方式。

接著使用T.4的同一份屬性validation題組，評估三版；每題新增上限都24，`all`選全檔：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-fp32-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int4-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int8.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int8-validation.json
```

在三份JSON中用相同`row`找回同一筆提問與目標，再並排`samples`裡的生成、原始ID與停止原因，欄位見[T.4](training.md#T.4)。逐題判內容匹配及正常結束，記「新增答對、新增答錯與不變」，相同總答對數可能是不同題目答對。先用validation選位數；設定選完才把相同三條命令的資料路徑改為`test.jsonl`，加上`--split-label test`並另取輸出檔名，最後題不拿來重新選設定。

練習把三版模型張量大小與逐題變化放在同一張表，分別說明「保存變小了嗎」與「相同題目是否維持回答」。尚未量到的執行記憶體與速度就留未量測。

<details>
<summary>固定量化與量化感知訓練</summary>

固定比較先完成T.4的完整`sft`，再跑：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment quantization --device cpu
```

這個入口另做120次更新才轉換，和上面的單獨 `quantize.py` 不同；不能互換成績。它保存浮點、4-bit、8-bit與共同題目，格式見[量化報告](../docs/course-experiments/results/quantization.json)。[17.14](chapters/17.md#17.14)介紹QAT：讓訓練先適應模擬誤差；真正低位元核心仍需格式與硬體支援。

</details>

