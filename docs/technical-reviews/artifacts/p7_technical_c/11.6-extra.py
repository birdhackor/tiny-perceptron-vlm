import pathlib
print('changed-only',2000+6000,(2000+6000)==budget,'rebalance',2000+8000,(2000+8000)==budget)
r=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vqa-raw.json').read_text())['results']
for n,t in [('alignment',r['two_stage_alignment_training']),('all',r['variants']['all']['training']),('direct',r['variants']['direct_vqa']['training'])]:
 print('target_sum',n,sum(x['effective_targets'] for x in t['history']),t['effective_targets']);assert sum(x['effective_targets'] for x in t['history'])==t['effective_targets']
print('total budget',14408+3830,18256,'gap',18256-(14408+3830))
for n in ['all','direct_vqa']:
 v=r['variants'][n];s=v['test']['samples'];print('recount',n,sum(x['generated']==x['target'] for x in s),len(s),'text_keys',[k for k in v if k.startswith('text')])
