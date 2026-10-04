examples = [("紅色的小小物體是", "圓"), ("藍色的小小物體是", "方")]
for length in [1, 3, 5, 8, 7]:
    visible = []
    for prefix, answer in examples:
        visible.append((prefix[-length:], answer))
    print(length, visible)
