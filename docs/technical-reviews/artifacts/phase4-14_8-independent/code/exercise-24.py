L, L_prime = 8, 24
positions = list(range(L_prime))
coordinates = [m * L / L_prime for m in positions]
print("卡片數", len(positions), "座標數", len(coordinates))
for m in (5, 23):
    print(m, "→", coordinates[m])
