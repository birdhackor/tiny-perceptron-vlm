## T.7 先檢查回答能力，再學較偏好的回答

偏好資料讓同一題有兩篇候選，再指出較合適的一篇。例如 `0+5=?` 的候選 5 與 6，可以依正確答案選 5；資料中的`chosen`欄保存較合適的回答，`rejected`欄保存較不合適的回答，兩欄配同一題。DPO 直接用這種同題偏好更新模型，讓它比固定起點參考更偏向較合適的回答；固定參考不更新，也不是提供正確答案的老師，角色見[13.4](chapters/13.md#13.4)。

下面是本機小型練習：使用 [T.5](#T.5) 自行訓練的 `checkpoints/style.pt`，在程式產生的偏好材料上更新 200 步，再比較前後回答。它和後面下載外部偏好包的固定完整配方是兩套做法，結果由各自報告決定。這些命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/prepare_data.py --kind preference
```

先開啟`data/generated/preference/train.jsonl`，核對`chosen`、`rejected`是否符合自己的判準。這裡用T.5的`style.pt`作起點，另保留不更新的參考副本；參考是原回答傾向，不是提供正確答案的老師。

更新前先保存這一題的起點回答。兩次生成使用同一題；下方`--temperature 0`在本工具中表示每步直接選最高分候選，沿用[1.15的greedy規則](chapters/01.md#1.15)，讓前後比較固定選字方式：

```bash
.venv/bin/python scripts/infer.py checkpoints/style.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
```

核對偏好標註並保存起點回答後，再更新200步；輸出另存`preferred.pt`，保留原`style.pt`：

```bash
.venv/bin/python scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt
```

完成後用相同方式取得新回答：

```bash
.venv/bin/python scripts/infer.py checkpoints/preferred.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
```

保存原問題、候選與兩份實際回答，再對其餘驗證題做相同檢查。候選 5 比 6 的機率高，仍不表示自由生成一定是 5；也要保留原來的內容、格式與安全題，確認新階段沒有破壞它們。[13.5](chapters/13.md#13.5)說明排序與生成的差別。

<details>
<summary>DPO 固定配方與 PPO 選卡路線</summary>

另一套固定DPO配方先完成T.5的固定`style`，再取外部偏好包。它使用自己的起點、資料與排程，不是接續上面200步的`preferred.pt`，也不能把兩套成績混用：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset ultrafeedback-dpo
.venv/bin/python -m scripts.course_experiments.run --experiment dpo --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄是 18.7 秒；改用 CPU 的可行性未量測。`dpo/` 保存策略模型、固定參考與偏好切分；`preference` 欄核對候選比較，`arithmetic` 欄核對實際生成，各自有分母。完整格式見 [DPO 實驗紀錄（JSON）](../docs/course-experiments/results/dpo.json)。

想逐步理解獎勵、價值模型與 PPO，可另跑獨立 CPU 選卡例子：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment posttraining --device cpu
```

作者在 CPU 上的紀錄是 2.1 秒，見 [posttraining.json](../docs/course-experiments/results/posttraining.json)。它選一張預寫的回答卡，不逐 token 生成；評分標籤來自作者規則。角色和更新流程見[13.10–13.17](chapters/13.md#13.10)。DPO與PPO是兩種路線，不必依序套用。

</details>

