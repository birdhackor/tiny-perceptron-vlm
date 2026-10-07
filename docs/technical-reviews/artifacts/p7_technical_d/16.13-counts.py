n=6
rules={
 'full':lambda p:list(range(p+1)),
 'sliding3':lambda p:list(range(max(0,p-2),p+1)),
 'block3':lambda p:list(range(p//3*3,p+1))}
for name,f in rules.items():
 rows=[f(p) for p in range(n)]
 assert all(k<=p for p,r in enumerate(rows) for k in r)
 print(name,'rows',rows,'pairs',sum(map(len,rows)),'p3',f(3),'p4',f(4))
assert [sum(len(f(p)) for p in range(n)) for f in rules.values()]==[21,15,12]
