recipe = {"風格": 30, "格式": 30, "應拒絕題": 20, "正常題完成": 20}
metrics = {key: None for key in ["風格", "格式", "誠實", "安全", "正常題完成", "原文字任務"]}
print("每100對的計畫", recipe, "合計", sum(recipe.values()))
for key, value in metrics.items(): print(key, "尚未量測" if value is None else value)
