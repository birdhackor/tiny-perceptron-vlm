all_pairs = [(color, shape) for color in ["紅", "藍"] for shape in ["圓", "方"]]
held_out = {("紅", "方")}
train = [pair for pair in all_pairs if pair not in held_out]
print("訓練組合", train)
print("留出組合", held_out)
assert not (set(train) & held_out)
