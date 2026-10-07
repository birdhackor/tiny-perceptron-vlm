examples = [{"visible_count": 3, "hidden_count": 3}, {"visible_count": None, "hidden_count": 3}]
for row in examples:
    visible = row["visible_count"]
    target = str(visible) if visible is not None else "看不到球數，請提供數量或圖片。"
    print("可見", visible, "→", target)
for visible, hidden, expected in [(0, 3, "0"), (None, 5, "看不到球數，請提供數量或圖片。")]:
    target = str(visible) if visible is not None else "看不到球數，請提供數量或圖片。"
    assert target == expected
    print("variation", visible, hidden, "→", target)
