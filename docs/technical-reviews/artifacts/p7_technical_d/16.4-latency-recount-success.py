import json,statistics,torch
print('torch',torch.__version__)
d=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['models']
for name in ['mha','gqa']:
 t=d[name]['cache']['cached_decode'];v=t['samples_seconds'];m=statistics.median(v);assert abs(m-t['median_seconds'])<1e-12
 print(name,'n',len(v),'median_ms',m*1000,'rounded',round(m*1000,3))
