velocity = 0.0
position = 0.0
for gradient in [1.0, 1.0, -1.0]:
    velocity = 0.5 * velocity + gradient
    position -= 0.1 * velocity
    print("新梯度", gradient, "保留方向", round(velocity, 3), "位置", round(position, 3))
