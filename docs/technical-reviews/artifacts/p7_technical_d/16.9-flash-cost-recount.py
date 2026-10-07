import torch,json,statistics
print('torch',torch.__version__);r=json.load(open('docs/course-experiments/results/flash_probe.json'))['results']
for n,d in r['routes'].items():
 for mode,ds in d['measurements'].items():
  for backend,x in ds.items():
   vs=x['samples_seconds'];m=statistics.median(vs);assert len(vs)==x['measured_calls']==9 and x['warmup_calls']==3;assert abs(m-x['median_seconds'])<1e-12
   b=x['allocated_before_bytes'];p=x['peak_allocated_bytes'];a=x['additional_peak_allocated_bytes'];assert p-b==a;assert x['synchronized']
   print(n,mode,backend,'n',len(vs),'medianms',round(m*1000,3),'memory_before_peak_add_MiB',[v/2**20 for v in [b,p,a]])
