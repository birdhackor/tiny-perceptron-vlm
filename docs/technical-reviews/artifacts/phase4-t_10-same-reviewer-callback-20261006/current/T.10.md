## T.10 用較小學生學教師，再與普通訓練比較

先選較小的學生架構，再比較它只學真值、與另加教師機率的差別。兩支學生要共用起點、資料與更新安排；否則看到差異，也難知道是不是教師訊號帶來的。

這套固定入口使用已完成的屬性、條件風格與MoE三種教師。先取得T.4完整`sft`、T.5完整`style`、T.8完整`moe`，保留各自的模型與資料切分，再執行：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset gsm8k
.venv/bin/python -m scripts.course_experiments.run --experiment distillation --device cpu
```

這是多支線的完整訓練，CPU需要時間；相容環境可改CUDA。產物在 `distillation/`，不同任務、學生尺寸與教師訊號各有檔名。GSM8K包只用作有限域外診斷，並不是這份屬性模型已具備解應用題的證據。

先選同寬度的兩支，用相同題目檢查。下方檔名的`w32`表示學生每個位置有32個特徵；`ce`只用真值回答訓練，`ce_kl`同時學真值與教師在相同位置的候選機率。KL在此衡量教師、學生兩份機率分佈的差異，方法見[18.8](chapters/18.md#18.8)與[18.9](chapters/18.md#18.9)：

```bash
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce_kl.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
```

希望答案是 `square,high`。有的題改善，也可能另有題退步；核對整份相同題目，並一起記學生參數、教師成本與有效訓練目標。只變小是架構的效果，教師訊號是否有幫助由這個對照回答。

教師和學生的文字單位也必須對齊：同一位置、同一候選字表，才可以直接比較機率。多模態學生還可能減少圖像位置，對齊方式另見[18.13](chapters/18.md#18.13)。蒸餾後再量化是另一個改動，要分開驗收。

<details>
<summary>原始資料、教師與多模態路線</summary>

完整教師、學生、對照與生成見[蒸餾報告](../docs/course-experiments/results/distillation.json)，不同任務有各自分母，不用單一數字代表全部能力。

多模態蒸餾使用固定 `vqa` 與 `joint` 兩份教師，以及各自的原始切分。先完成[T.4](training.md#T.4)的固定 `sft`，再按[T.6](training.md#T.6)的固定入口完成 `encoders`、`projector`、`vqa`，接著在相容的 CUDA 環境依序執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment joint --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment multimodal_distillation --device cuda
```

第一行是[12.12](chapters/12.md#12.12)的固定圖音聯合訓練，直接讀取 `sft` 與 `encoders`，不承接 `vqa`。這份 `joint`、T.6 手動保存的 `checkpoints/joint.pt` 和第19章的 `capstone_joint` 各有自己的配方，不能互換來源。實驗工具每次只執行指定項，不自動補齊依賴；上述命令沿用同一個預設 `outputs/course-experiments/course-v1/` 根目錄，第二行會讀取 `vqa/` 與 `joint/` 各自的 `model.pt`、`dataset.json`。先保留來源權重和原始切分；僅下載推論包不足以恢復訓練資料。操作與格式見[實驗說明](../docs/course-experiments/README.md)。

</details>

