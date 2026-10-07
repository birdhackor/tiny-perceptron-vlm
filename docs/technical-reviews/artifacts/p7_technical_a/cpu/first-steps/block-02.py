import math

for p in [0.9, 0.5, 0.1]:
    cost = -math.log(p)
    print(p, round(cost, 3))
