import json,statistics,torch
print('torch',torch.__version__)
d=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['models']
for name in ['mha','gqa']:
 c=d[name]['cache']
 t=c['cached_decode'];print(name,t)
 vals=t['samples'];m=statistics.median(vals);assert abs(m-t['median_seconds'])<1e-12
 print(name,'n',len(vals),'median_ms',m*1000)
