## T.10 用較小學生學教師，再與普通訓練比較

先選較小的學生架構，再比較它只學正確答案、與另加教師機率的差別。兩支學生要共用起點、資料與更新安排；否則看到差異，也難知道是不是教師訊號帶來的。

這套固定配方使用三種已訓練完成的教師：屬性教師、風格教師與 MoE 教師。此路線必須先打開並完成[T.4的固定對話配方](training.md#fixed-sft-recipe)、[T.5的固定風格配方](training.md#fixed-style-recipe)及[T.8的固定MoE配方](training.md#fixed-moe-recipe)中的`sft`、`style`、`moe`三組，不能用前面小練習的`checkpoints/attributes.pt`、`style.pt`或`moe.pt`替代。

這三組前置配方都寫 `--device cuda`。沒有 NVIDIA GPU（例如用 Mac）時，工具不會自動改用 CPU，指定 `cuda` 會直接停下；可以把它們改成 `--device cpu`，但 CPU 可行性未量測：能不能跑完、要多久都沒有紀錄。這個工具只接受 `cpu` 與 `cuda`，不支援 Apple 的 MPS。作者在 Modal 的 NVIDIA L4 上的紀錄是 `sft` 31.7 秒、`style` 29.5 秒、`moe` 65.7 秒。

三組命令沿用預設輸出根目錄`outputs/course-experiments/course-v1/`。核對下列六個檔案都已產生：每份`model.pt`是該教師，配對的`dataset.json`保存它原本的訓練、驗證與測試切分，學生沿用這些切分比較：

| 教師 | 模型檔 | 配對資料切分 |
| --- | --- | --- |
| 屬性教師 | `sft/model.pt` | `sft/dataset.json` |
| 風格教師 | `style/model.pt` | `style/dataset.json` |
| MoE 教師 | `moe/model.pt` | `moe/dataset.json` |

保留同一組的模型與資料，不用只含推論權重的下載包替代完整教師產物，再執行：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset gsm8k
.venv/bin/python -m scripts.course_experiments.run --experiment distillation --device cpu
```

蒸餾命令寫的是 `--device cpu`。它要訓練多支學生，涵蓋不同任務、學生尺寸與教師訊號，CPU 時間未量測；作者的紀錄是在 Modal 的 NVIDIA L4 上用 `--device cuda` 跑的，90.1 秒。有相容環境可改成 `--device cuda`。產物在 `distillation/`，不同任務、學生尺寸與教師訊號各有檔名。GSM8K 是一份英文小學數學應用題資料集；這裡只拿它看學生遇到訓練範圍外的題型時表現如何，也就是有限的域外診斷，並不是這份屬性模型已具備解應用題的證據。

先選同寬度的兩支，用相同題目檢查。下方檔名的`w32`表示學生的特徵寬度是 32；`ce`只用正確答案的交叉熵（cross entropy，見[1.8](chapters/01.md#1.8)）訓練，`ce_kl`同時學正確答案與教師在相同位置的候選機率。KL是Kullback–Leibler divergence（KL散度），在此衡量教師、學生兩份機率分佈的差異，方法見[18.8](chapters/18.md#18.8)與[18.9](chapters/18.md#18.9)。兩行推論也固定相同題目、24個新增token上限與CPU；都沿用本工具預設的 `--temperature 0`，每步選最高分候選，不做隨機抽樣，與T.7、T.9的比較規則相同：

```bash
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce_kl.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
```

希望答案是 `square,high`。有的題改善，也可能另有題退步；核對整份相同題目，並一起記學生參數、教師成本與有效目標數。只變小是架構的效果，教師訊號是否有幫助由這個對照回答。

教師和學生還要字表對齊：同一位置、同一份字表的候選，才可以直接比較機率。多模態學生還可能減少圖像位置，做法另見[18.13](chapters/18.md#18.13)。蒸餾後再量化是另一個改動，要分開驗收。

<details>
<summary>原始資料、教師與多模態路線</summary>

完整教師、學生、對照與生成見[蒸餾實驗紀錄（JSON）](../docs/course-experiments/results/distillation.json)，不同任務有各自分母，不用單一數字代表全部能力。

多模態蒸餾使用固定 `vqa` 與 `joint` 兩份教師，以及各自的原始切分。先完成[T.4](training.md#T.4)的固定 `sft`，再按[T.6](training.md#T.6)的固定配方完成 `encoders`、`projector`、`vqa`，接著在相容的 CUDA 環境依序執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment joint --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment multimodal_distillation --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄：`joint` 33.4 秒、`multimodal_distillation` 53.0 秒；改用 CPU 的可行性未量測，而且 `joint` 屬於模態實驗，CPU 結果只會被標成通路檢查。第一行是[12.12](chapters/12.md#12.12)的固定圖音聯合訓練，直接讀取 `sft` 與 `encoders`，不承接 `vqa`。這份 `joint`、T.6 手動保存的 `checkpoints/joint.pt` 和合成小世界整合實驗的 `capstone_joint` 各有自己的配方，不能互換來源。實驗工具每次只執行指定項，不自動補齊依賴；上述命令沿用同一個預設 `outputs/course-experiments/course-v1/` 根目錄，第二行會讀取 `vqa/` 與 `joint/` 各自的 `model.pt`、`dataset.json`。先保留來源權重和原始切分；僅下載推論包不足以恢復訓練資料。操作與格式見[實驗說明](../docs/course-experiments/README.md)。

</details>

