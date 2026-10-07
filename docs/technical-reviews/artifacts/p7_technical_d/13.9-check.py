import json
j=json.load(open('docs/course-experiments/results/dpo.json'))['results'];q=j['ultrafeedback_pilot']
print('exposure_counts',2*30*100,2*30*10)
print('pilot_steps',q['sft_training']['steps'],q['training']['steps']);assert q['sft_training']['steps']==q['training']['steps']==80
for split in ('validation','test'):
 p=q['evaluation'][split];rs=p['samples'];relative=sum(r['relative_margin']>0 for r in rs);absolute=sum(r['policy_margin']>0 for r in rs);print('natural',split,'n',len(rs),'relative',relative,'absolute',absolute);assert relative==p['relative_preference_improved'] and absolute==p['chosen_higher_absolute_probability']
for name,r in j['runs'].items():
 rs=r['arithmetic']['test']['samples'];print('addition',name,len(rs),sum(s['generated']==s['expected'] for s in rs))
rs=j['format_only']['arithmetic']['test']['samples'];print('format_free_generation',len(rs),sum(s['generated']==s['expected'] for s in rs))
print('ratio_change_sum',30+30+40+0)
