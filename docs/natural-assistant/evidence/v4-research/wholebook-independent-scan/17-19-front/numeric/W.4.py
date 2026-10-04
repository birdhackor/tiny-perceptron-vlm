import math

for p in [0.9, 0.5, 0.1]:
    cost = -math.log(p)
    bits = cost / math.log(2)
    print(p, round(cost, 3), round(bits, 3))
