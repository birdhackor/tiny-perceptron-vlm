import math,json,pathlib
D=pathlib.Path(__file__).parent/'evidence'
out={'scope':'固定原始logits與已存generated_ids的短程核算；未訓練、未重新生成、未下載權重','calibration':{},'answer_token_validity':{}}
def compute(logits,labels,t):
 ps=[];pred=[]
 for z in logits:
  x=[q/t for q in z];m=max(x);e=[math.exp(q-m) for q in x];s=sum(e);ps.append([q/s for q in e]);pred.append(max(range(len(z)),key=lambda k:z[k]))
 n=len(labels);conf=[max(p) for p in ps];correct=[int(a==b) for a,b in zip(pred,labels)];bins=[]
 for k in range(5):
  ids=[i for i,c in enumerate(conf) if min(int(c*5),4)==k]
  if ids:bins.append({'bin':k,'count':len(ids),'accuracy':sum(correct[i] for i in ids)/len(ids),'mean_confidence':sum(conf[i] for i in ids)/len(ids)})
 return {'count':n,'correct':sum(correct),'accuracy':sum(correct)/n,'mean_confidence':sum(conf)/n,'nll':-sum(math.log(p[y]) for p,y in zip(ps,labels))/n,'brier':sum(sum((a-int(k==y))**2 for k,a in enumerate(p)) for p,y in zip(ps,labels))/n,'ece':sum(b['count']/n*abs(b['accuracy']-b['mean_confidence']) for b in bins),'bins':bins,'predicted_labels':pred}
j=json.load(open(D/'encoders.json'))
for kind,x in j['results'].items():
 c=x['calibration'];entry={}
 for split in ('validation','test'):
  a=c[split];entry[split]={}
  for k in ('original','calibrated'):
   expected=a[k];r=compute(a['logits'],a['labels'],expected['temperature'])
   differences={q:abs(r[q]-expected[q]) for q in ('accuracy','mean_confidence','nll','brier','ece')}
   assert max(differences.values())<2e-6,(kind,split,k,differences)
   assert r['predicted_labels']==expected['predicted_labels']
   entry[split][k]={'recomputed':r,'max_absolute_numeric_difference':max(differences.values())}
  assert entry[split]['original']['recomputed']['predicted_labels']==entry[split]['calibrated']['recomputed']['predicted_labels']
 grid=[(t,compute(c['validation']['logits'],c['validation']['labels'],t)['nll']) for t in c['temperature_grid']]
 selected=min(grid,key=lambda z:z[1])[0];assert selected==c['chosen_temperature'];entry['grid_selection_from_validation_only']=grid;entry['chosen_temperature']=selected
 out['calibration'][kind]=entry
for f in ['style','lora']:
 p=json.load(open(D/(f+'.json')))['results'];counts={'stored_generated_sequences':0,'sequences_with_control_id_before_first_EOS':0};bad=[]
 def walk(x,path):
  if isinstance(x,dict):
   if 'generated_ids' in x:
    ids=x['generated_ids'];ids=ids[:ids.index(2)] if 2 in ids else ids;counts['stored_generated_sequences']+=1
    if any(k<8 for k in ids):counts['sequences_with_control_id_before_first_EOS']+=1;bad.append(path)
   for k,v in x.items():walk(v,path+'/'+k)
  elif isinstance(x,list):
   for k,v in enumerate(x):walk(v,path+'/'+str(k))
 walk(p,'results');counts['bad_paths']=bad;out['answer_token_validity'][f]=counts
(D/'recompute-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'calibration':{k:{'chosen_temperature':v['chosen_temperature'],'test_original':{q:v['test']['original']['recomputed'][q] for q in ('count','correct','mean_confidence','ece','brier')},'test_calibrated':{q:v['test']['calibrated']['recomputed'][q] for q in ('count','correct','mean_confidence','ece','brier')}} for k,v in out['calibration'].items()},'answer_token_validity':out['answer_token_validity']},ensure_ascii=False,indent=2))
