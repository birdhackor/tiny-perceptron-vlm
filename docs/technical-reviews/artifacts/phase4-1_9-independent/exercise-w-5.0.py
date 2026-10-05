def loss(w):
    return (w - 3) ** 2


w = 5.0
h = 0.001
left = loss(w - h)
right = loss(w + h)
gradient = (right - left) / (2 * h)
print(left, right)
print(gradient, "公式計算", 2 * (w - 3))
