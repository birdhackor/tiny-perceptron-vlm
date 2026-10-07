import json,statistics,torch
print('torch',torch.__version__)
d=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['models']
for name in ['mha','gqa']:
 c=d[name]['cache'];print(name,list(c))
 for key in ['prefill_timing','full_decode_timing','cached_decode_timing']:
  if key in c:
   t=c[key];print(key,t)
