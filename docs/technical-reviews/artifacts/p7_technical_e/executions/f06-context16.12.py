length = 8
window = 3  # 包含目前位置

def visible(position):
    return set(range(max(0, position - window + 1), position + 1))

print("位置7直接讀", sorted(visible(7)))
two_layers = set().union(*(visible(p) for p in visible(7)))
print("兩層可能傳來的來源", sorted(two_layers))
print("完整因果配對", sum(p + 1 for p in range(length)))
print("滑窗配對", sum(len(visible(p)) for p in range(length)))
