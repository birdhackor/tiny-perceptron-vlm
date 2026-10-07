import json,statistics
from tiny_perceptron.data import ByteTokenizer
r=json.load(open('docs/course-experiments/results/efficiency.json'));c=r['results']['models']['mha']['cache'];print('device',r['gpu'],'prompt_tokens',1+len(ByteTokenizer().encode('Once upon a time, a girl')),'raw',c['prompt_tokens'],'generated_steps',c['generated_tokens'])
for k in ['prefill','full_recompute_decode','cached_decode']:
 d=c[k];m=statistics.median(d['samples_seconds']);assert m==d['median_seconds'];print(k,'ms',round(m*1000,3),'samples',len(d['samples_seconds']),'warmup',d['warmup_calls'],'synced',d['synchronized'])
