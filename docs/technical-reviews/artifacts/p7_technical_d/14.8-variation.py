L,L_prime=8,24
positions=list(range(L_prime));coordinates=[m*L/L_prime for m in positions]
print('card_and_coordinate_counts',len(positions),len(coordinates))
for m in (5,23):print(m,coordinates[m],round(coordinates[m],4))
print('distance',coordinates[-1]-coordinates[0],'all_less_than_L',all(0<=v<L for v in coordinates))
