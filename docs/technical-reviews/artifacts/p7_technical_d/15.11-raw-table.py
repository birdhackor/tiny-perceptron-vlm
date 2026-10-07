import json
r=json.load(open('docs/course-experiments/results/moe.json'));print('gpu',r['environment'] if 'environment' in r else r.get('hardware'));v=r['results']['variants']
for n in ['dense_active_top1','dense_active_top2','dense_total','top1_aux0.01','top2_aux0.01']:
 d=v[n];t=d['training'];print(n,'width',d['model']['config']['width'],'ms',round(t['warm_step_median_seconds']*1000,3),'beforeMiB',round(t['memory_allocated_before_bytes']/2**20,3),'peakMiB',round(t['peak_memory_allocated_bytes']/2**20,3),'additional',t['peak_additional_allocated_bytes']);assert t['peak_additional_allocated_bytes']==t['peak_memory_allocated_bytes']-t['memory_allocated_before_bytes']
