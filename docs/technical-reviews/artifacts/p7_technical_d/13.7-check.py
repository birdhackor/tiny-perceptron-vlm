import json
from tiny_perceptron.data import ByteTokenizer,render_chat
j=json.load(open('docs/course-experiments/results/dpo.json'))['results'];f=j['format_only'];t=ByteTokenizer()
print('variation_repeat2',len(t.encode('4。'*2)),len(t.encode('4。'*2))+1)
for text in ('4','4; answer complete'):
 x,y=render_chat([{'role':'user','content':'2+2=?'},{'role':'assistant','content':text}]);print('format_target',text,int((y!=-100).sum()))
rs=f['preference']['test']['samples'];print('format_test',len(rs),'reference_short_higher',sum(r['reference_margin']>0 for r in rs),'after_short_higher',sum(r['policy_margin']>0 for r in rs),'relative_improved',sum(r['relative_margin']>0 for r in rs),'steps',f['training']['steps']);assert len(rs)==7 and all(r['reference_margin']>0 and r['policy_margin']>0 and r['relative_margin']>0 for r in rs) and f['training']['steps']==200
gs=f['arithmetic']['test']['samples'];print('generation',len(gs),sum(r['generated']==r['expected'] for r in gs),'2+2',next(r for r in gs if r['messages'][0]['content']=='2+2=?'))
r=next(r for r in f['preference']['validation']['samples'] if r['prompt']=='0+3=?');b=next(r for r in j['before']['validation']['samples'] if r['prompt']=='0+3=?');new=r['policy_chosen_logp'];old=b['policy_chosen_logp'];delta=(r['policy_chosen_logp']-r['policy_rejected_logp'])-r['reference_margin'];print('0+3_chosen_before_after',old,new,'change',new-old,'relative',delta);assert new<old and abs(delta-r['relative_margin'])<1e-10
