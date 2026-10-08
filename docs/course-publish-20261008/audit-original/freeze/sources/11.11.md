## 11.11 平均九成，為什麼稀少題仍可能全錯？

常見組 90/90、稀少組 0/10，全部題合計仍是 90%。稀少需求的使用者卻一次也沒成功。材料已經給出每組答對與總數，先用兩種平均看它們的權重。

```python
groups = {"常見": {"correct": 90, "count": 90}, "稀少": {"correct": 0, "count": 10}}
micro = sum(g["correct"] for g in groups.values()) / sum(g["count"] for g in groups.values())
macro = sum(g["correct"] / g["count"] for g in groups.values()) / len(groups)
for name, group in groups.items():
    print(name, "答對/總數", group["correct"], group["count"])
print("按題平均", micro, "按組平均", macro)
```

按題平均 micro 把分子 90＋0、分母 90＋10 合起來，得 0.9；按組平均 macro 先算 1、0，再給兩組同權重，得 0.5。程式的人工反例沒有改模型，只改觀看成績的方式。

平均方式不會自動揭露所有弱點。既有視覺題顏色 6/6、形狀 3/6，兩組同大，micro 和 macro 都是 75%；再看形狀，圓 3/3、方 0/3，才知道固定猜 circle 的問題。

因此每組定義、分子、分母與具體答案都應保存。組別可以按題型、目標大小或未見組合預先安排；重疊組不能讓 micro 重複加同一題。小組只一題，即使 1/1 也不足以說穩定可靠。

兩模型都答對 90/100：一個是 90/90 與 0/10，另一個是 81/90 與 9/10。總數相同，使用經驗不同。哪種平均重要，要看是否關注實際使用比例或各類表現，不能只挑好看的那個。

練習稀少 count 改 20、correct 仍 0，micro 成 90/110 約 0.8182，macro 仍 0.5。測試組成改了，不代表模型學壞。

<details>
<summary>回顧與查證</summary>

可回顧：[11.8資料分布影響成績](11.md#11.8)。

原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/vision_ablation.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。

</details>

