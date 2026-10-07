L,L_prime=8,16
positions=list(range(L_prime))
coordinates=[m*L/L_prime for m in positions]
print("卡片數",len(positions),"座標數",len(coordinates))
for m in (0,1,2,14,15):print(m,"→",coordinates[m])
print('all_coordinates',coordinates)
