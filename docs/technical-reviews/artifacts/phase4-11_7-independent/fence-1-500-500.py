recipe = {"文字有效位置": 500, "圖文有效位置": 500}
recorded = {
    "訓練前": {"文字答對": 8, "圖文答對": 2},
    "訓練後": {"文字答對": 6, "圖文答對": 7},
}
total_positions = sum(recipe.values())
ratios = {}
for category, positions in recipe.items():
    ratios[category] = positions / total_positions
print("示例訓練比例", ratios)
for phase, scores in recorded.items():
    print(phase, "文字", scores["文字答對"] / 10, "圖文", scores["圖文答對"] / 10)
