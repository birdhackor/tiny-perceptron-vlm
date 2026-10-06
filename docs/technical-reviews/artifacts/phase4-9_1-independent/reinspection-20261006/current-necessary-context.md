## 9.2 不知道時怎麼回答？

盒內真有3球，助手沒看到，猜3即使撞中也沒有根據；若已明說3，卻還一律澄清，也沒有使用材料。只改「球數是否可見」，理想回答就應改變。

visible_count保存可見數量，沒有時None。作者另知hidden_count，但生成目標的規則只讀可見欄位，不能偷偷把隱值教成猜中的獎勵。

```python
examples = [
    {"visible_count": 3, "hidden_count": 3},
    {"visible_count": None, "hidden_count": 3},
]
for row in examples:
    visible = row["visible_count"]
    target = str(visible) if visible is not None else "看不到球數，請提供數量或圖片。"
    print("可見", visible, "→", target)
```

輸入兩筆的隱藏真值都為3，只有可見資訊不同。`visible`只取可見欄位，`is not None`檢查是否有值。這裡把分支縮成一行：`A if 條件 else B`表示條件成立就取A，不成立就取B。因此有值時`str`把整數轉成回答文字，無值時給出說明缺項的目標句。輸出第一行`可見 3 → 3`，第二行`可見 None → 看不到球數，請提供數量或圖片。`。關鍵是產生回答的那行從未讀取`hidden_count`。

這是人工規則產生訓練目標，不是模型已具備自我覺察的展示。真正訓練可以把「盒中有3顆球，問幾顆」與「一個未開的盒子，問幾顆」做成對話，讓模型學兩種回答情況。測試時再換球數與問法，查看它是否把缺資訊的題一律猜最常見數字，或把資訊完整的題也過度澄清。
