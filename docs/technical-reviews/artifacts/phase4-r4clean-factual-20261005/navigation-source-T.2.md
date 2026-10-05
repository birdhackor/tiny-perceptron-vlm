## T.2 先讓一組資料走過模型

模型還沒更新前，先讓一批材料進去，確認它能算出有限代價，參數也收到更新方向。這像開工前先試機器的通路；它不是能力考試。

下面三行分別檢查文字接續（text）、助手示範回答（sft）與圖片問答（vision），選一條即可。沒有 `--train` 時，工具只做前向與反向，不更新、不保存訓練權重。

```bash
python scripts/train.py --task text --device cpu
python scripts/train.py --task sft --device cpu
python scripts/train.py --task vision --device cpu
```

前向用目前參數預測，反向計算代價對參數的敏感度；真正調整要由更新工具執行。

先看報告的幾個欄位：

| 欄位 | 這一步要確認什麼 |
| --- | --- |
| `mode` | 應為 `dry-run-no-weight-update` |
| `task`、`device` | 與命令選擇相符 |
| `loss` | 是有限數字，沒有 NaN 或無窮大 |
| `grad_norm` | 有限且大於零，表示至少有參數收到非零梯度 |
| `effective_tokens` | 若入口有統計，確認有效計分位置大於零 |

這個圖片入口未統計最後一欄，可能寫 `null`；它表示未提供數量，不能讀成零個目標。`grad_norm` 大於零也沒有證明每個參數都有梯度。要查特定零件，回到[5.1](chapters/05.md#5.1)或[10.7 的回答位置](chapters/10.md#10.7)看逐項檢查。

選一條命令，把模式、代價和梯度大小記下來。能解釋「通路已工作，但尚未更新」後，才進入訓練。

