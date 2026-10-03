examples = [("紅色物體是", "圓"), ("藍色物體是", "方")]
for length in [1, 3, 5]:
    visible = []
    for prefix, answer in examples:
        visible.append((prefix[-length:], answer))
    print(length, visible)
