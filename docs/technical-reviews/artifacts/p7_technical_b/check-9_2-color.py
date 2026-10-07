row = {"visible_count": None, "hidden_count": 3, "box_color": "紅"}
visible = row["visible_count"]
target = str(visible) if visible is not None else "看不到球數，請提供數量或圖片。"
print(target)
assert target == "看不到球數，請提供數量或圖片。"
