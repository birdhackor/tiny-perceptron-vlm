import collections,contextlib,hashlib,io,json,re,sys,platform
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.natural_assistant import messages_for,score_output
N=ROOT/'docs/natural-assistant';O=ROOT/'docs/technical-reviews/artifacts';P='natural-final-fact-20.5-'
def read(path):return json.loads((N/path).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
body=(O/(P+'section.md')).read_text()
fence=re.search(r'```python\n(.*?)\n```',body,re.S)[1]
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):exec(compile(fence,'20.5 exact CPU fence','exec'),{})
expected='卡 0 物件集合吻合 True\n活動符合本題標註 True\n卡 1 物件集合吻合 True\n活動符合本題標註 False\n'
assert stdout.getvalue()==expected
(O/(P+'example.py')).write_text(fence+'\n');(O/(P+'example.txt')).write_text(stdout.getvalue())
exercise=fence.replace('for index, card in enumerate(cards):','cards[1]["objects"].add("狗")\nfor index, card in enumerate(cards):')
stdout2=io.StringIO()
with contextlib.redirect_stdout(stdout2):exec(compile(exercise,'20.5 exact exercise','exec'),{})
assert stdout2.getvalue()==expected.replace('卡 1 物件集合吻合 True','卡 1 物件集合吻合 False')
(O/(P+'exercise.py')).write_text(exercise+'\n');(O/(P+'exercise.txt')).write_text(stdout2.getvalue())
manifest=read('manifest.json');frozen={r['id']:r for r in manifest['rows']}
families={s:{r['family'] for r in manifest['rows'] if r['split']==s and r['task']=='scene'} for s in ['train','validation','test']}
assert all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
split_counts={s:{'images':len(families[s]),'questions':sum(r['split']==s and r['task']=='scene' for r in manifest['rows'])} for s in families}
assert split_counts=={'train':{'images':120,'questions':120},'validation':{'images':12,'questions':36},'test':{'images':12,'questions':36}}
file_map={f['path']:f for f in manifest['files']}
for r in manifest['rows']:
 if r['task']=='scene':assert sha(ROOT/'data/natural'/r['image'])==file_map[r['image']]['sha256']
trains={k:read(f'evidence/{k}/result.json') for k in ['train','train-gentle']}
a,b=trains.values()
settings=['model','model_revision','asr_model','asr_revision','versions','device','dtype','gpu_name','attention_implementation','max_pixels','min_pixels','max_tokens','seed','dataset_version','manifest_sha256','asset_sha256','sources','total_parameters','trainable_parameters','lora_rank','lora_targets','requested_steps','completed_steps','gradient_accumulation','trained_rows','optimizer_parameter_names','trainable_parameter_names','optimizer_only_lora','initial_adapter_tensors','frozen_parameter_samples_initial']
assert all(a[k]==b[k] for k in settings)
assert [a['learning_rate'],b['learning_rate']]==[1e-4,3e-5]
assert [(h['step'],h['row_ids'],h['supervised_tokens']) for h in a['history']]==[(h['step'],h['row_ids'],h['supervised_tokens']) for h in b['history']]
train_summary={}
for name,t in trains.items():
 ids=[i for h in t['history'] for i in h['row_ids']]
 assert len(t['history'])==t['completed_steps']==180 and len(ids)==360 and len(set(ids))==272
 assert len(t['initial_adapter_tensors'])==len(t['final_adapter_tensors'])==112
 assert sum(t['initial_adapter_tensors'][k]['sha256_values']!=t['final_adapter_tensors'][k]['sha256_values'] for k in t['initial_adapter_tensors'])==112
 assert t['frozen_parameter_samples_initial']==t['frozen_parameter_samples_final']
 train_summary[name]={'steps':len(t['history']),'visits':len(ids),'records':len(set(ids)),'answer_positions':sum(h['supervised_tokens'] for h in t['history']),'learning_rate':t['learning_rate'],'initial_tensors':len(t['initial_adapter_tensors']),'fresh_initial_hashes_identical':True}
stages={name:read(f'evidence/{name}/generations.json') for name in ['validation','validation-gentle','final']}
token_audit={};phrase={}
for name,rs in stages.items():
 token_audit[name]={};phrase[name]={}
 for variant in ['base','adapter']:
  part=[g for g in rs if g['variant']==variant]
  stop=collections.Counter()
  for g in part:
   ids=g['generated_token_ids'];eos=bool(ids and ids[-1] in g['eos_token_ids']);limit=len(ids)>=384
   assert len(ids)==g['generated_tokens']<=384
   assert eos==g['ended_with_eos'] and (limit and not eos)==g['truncated']
   assert (not eos and not limit)==g['completion_unknown']
   assert g['stop_reason']==('eos' if eos else 'max_new_tokens' if limit else 'unknown')
   stop[g['stop_reason']]+=1
   if g['id'] in frozen:
    r=frozen[g['id']];assert r['user']==g['user'] and r['image']==g['image'] and r['answer']==g['reference_answer']
    assert score_output(r,g['prediction'])==g['score']
  scene=[g for g in part if g['task']=='scene'];facts=[g for g in scene if '/fact-' in g['id']]
  phrase[name][variant]={'passed':sum(g['score']['passed'] for g in facts),'denominator':len(facts)}
  token_audit[name][variant]={'all_records':len(part),'stops':dict(stop),'scene_truncated_ids':[g['id'] for g in scene if g['truncated']],'open_scene_eos':sum(g['stop_reason']=='eos' for g in scene if g['id'].endswith('/scene'))}
 assert len([g for g in rs if g['task']=='scene'])==72
assert phrase['validation']=={'base':{'passed':21,'denominator':24},'adapter':{'passed':15,'denominator':24}}
assert phrase['final']=={'base':{'passed':23,'denominator':24},'adapter':{'passed':21,'denominator':24}}
assert token_audit['final']['adapter']['stops']=={'eos':85,'max_new_tokens':5}
assert token_audit['final']['adapter']['open_scene_eos']==7
reports={'validation':read('evidence/semantic-validation/paired-384-semantic-review.json'),'validation-gentle':read('evidence/semantic-validation/gentle-384-semantic-review.json'),'final':read('evidence/semantic-final/final-semantic-review.json')}
summaries={};report_raw_checks={}
for name,d in reports.items():
 raw={(g['id'],g['variant'],g['task']):g for g in stages[name]};recs=d.get('all_records',d.get('records'));counts=collections.defaultdict(lambda:[0,0]);category=collections.defaultdict(lambda:[0,0]);field='raw_generation_record_preserved' if name=='final' else 'raw_generation_record'
 assert len(recs)==len(raw)
 for r in recs:
  assert r[field]==raw[(r['id'],r['variant'],r[field]['task'])]
  success=r['completed_task_success']
  if success:assert raw[(r['id'],r['variant'],r[field]['task'])]['ended_with_eos'] and not raw[(r['id'],r['variant'],r[field]['task'])]['truncated']
  counts[(r['variant'],r['task'])][0]+=int(success);counts[(r['variant'],r['task'])][1]+=1
  if name=='final' and r['task']=='scene':category[(r['variant'],r['scene_primary_criterion'])][0]+=int(success);category[(r['variant'],r['scene_primary_criterion'])][1]+=1
 summaries[name]={'tasks':{str(k):v for k,v in counts.items()},'scene_categories':{str(k):v for k,v in category.items()}}
 report_raw_checks[name]={'records':len(recs),'all_embedded_raw_records_exactly_equal':True}
 if name=='final':
  for v,vals in [('base',{'object':[8,13],'activity':[0,2],'relation':[7,9],'open_description':[1,12]}),('adapter',{'object':[10,13],'activity':[1,2],'relation':[6,9],'open_description':[0,12]})]:
   assert counts[(v,'scene')]==([16,36] if v=='base' else [17,36])
   for k,n in vals.items():assert category[(v,k)]==n
 else:
  assert counts[('base','scene')]==[11,36] and counts[('adapter','scene')]==[19,36]
selection=read('selection.json');rule=read('selection-rule-gentle.json');math=read('evidence/selection/validation-decision-math.json');fe=read('evidence/final/execution.json')
assert sha(N/'selection.json')==fe['selection_sha256']=='c7e4dff552012cd18593f3708dc9f6fbf11292511894a9ac70c1bb86558e76fa'
assert selection['adapter_run_id']==a['execution']['run_id']==fe['adapter_run_id']
assert selection['adapter_sha256']==fe['adapter_sha256']=='0778668f3fd382d7fb8ef0e54c02504cf0ad517d1eca80891b0407e1101d85c5'
assert selection['validation_result_sha256']==sha(N/'evidence/validation/result.json')
assert selection['selection_rule_sha256']==sha(N/'selection-rule-gentle.json') and selection['selection_math_sha256']==sha(N/'evidence/selection/validation-decision-math.json')
means={}
for name,tgt,variant in [('base','validation','base'),('original','validation','adapter'),('gentle','validation-gentle','adapter')]:
 recs=reports[tgt]['records'];groups={k:[r for r in recs if r['variant']==variant and r['task'] in (['text_chat','text_dialogue'] if k=='text_chat_and_dialogue' else [k])] for k in rule['domains']}
 actual={k:{'completed_success':sum(r['completed_task_success'] for r in v),'denominator':len(v)} for k,v in groups.items()}
 for k,v in actual.items():assert v=={j:math['values'][name]['counts'][k][j] for j in v}
 mean=sum((Fraction(v['completed_success'],v['denominator']) for v in actual.values()),Fraction())/6
 assert mean==Fraction(math['values'][name]['mean_fraction']);means[name]={'counts':actual,'fraction':str(mean),'decimal':float(mean)}
assert [Fraction(means[k]['fraction']) for k in ['base','original','gentle']]==[Fraction(83,216),Fraction(106,216),Fraction(100,216)]
assert Fraction(101,216)<Fraction(106,216)
for name in ['original','gentle']:
 c=means[name]['counts'];bc=means['base']['counts'];guard=rule['adapter_eligibility']
 assert c['ocr']['completed_success']>=guard['ocr_minimum_completed_exact'] and c['text_presence']['completed_success']>=guard['text_presence_minimum_completed_exact']
 assert bc['scene']['completed_success']-c['scene']['completed_success']<=guard['scene_maximum_completed_success_drop_vs_base']
 assert bc['text_chat_and_dialogue']['completed_success']-c['text_chat_and_dialogue']['completed_success']<=guard['text_chat_and_dialogue_maximum_completed_success_drop_vs_base']
ctrl=read('evidence/image-control/result.json');protocol=read('image-control.json');assert ctrl['protocol_sha256']==sha(N/'image-control.json')
public=read('public-release.json');public_provenance=json.loads((O/(P+'public-release-provenance.json')).read_text())
assert (ctrl['public_repo'],ctrl['public_revision'])==(public['repo'],public['revision'])
assert ctrl['base_model']==public['base_model']==public_provenance['base_model']
assert ctrl['runtime']=={k:public['runtime'][k] for k in ctrl['runtime']}
assert next(f for f in public['files'] if f['output']=='adapter_model.safetensors')['sha256']==selection['adapter_sha256']
assert public_provenance['source']['prefix'].endswith('/'+selection['adapter_run_id']+'/adapter')
assert sha(O/(P+'public-release-provenance.json'))==next(f for f in public['files'] if f['output']=='release-provenance.json')['sha256']
cs=ctrl['cases'];assert len(cs)==2 and cs[0]['row']['user']==cs[1]['row']['user'] and cs[0]['row']['history']==cs[1]['row']['history']==[]
control=[]
for c in cs:
 r=c['row'];g=c['result'];im=c['input_image'];assert r['answer'] is None and r['split']=='train' and 'system' not in r and r['image'] in {a['image'] for a in manifest['rows'] if a['split']=='train'}
 assert sha(ROOT/'data/natural'/r['image'])==im['sha256']
 m=messages_for(r,ROOT/'data/natural');assert len(m)==1 and m[0]['role']=='user' and [p['type'] for p in m[0]['content']]==['image','text']
 assert m[0]['content'][-1]['text']==r['user'] and g['ended_with_eos'] and not g['truncated']
 control.append({'id':r['id'],'prediction':g['prediction'],'history':r['history'],'messages':m,'input_sha256':im['sha256'],'stop_reason':g['stop_reason']})
assert [c['prediction'] for c in control]==['一個藍色的蘑菇。','一列鐵路車廂。']
result={'python':platform.python_version(),'torch':torch.__version__,'device':'CPU only; no model forward/new inference','example_output':stdout.getvalue(),'exercise_output':stdout2.getvalue(),'source_sha256':hashlib.sha256(body.encode()).hexdigest(),'split_counts':split_counts,'training':train_summary,'all_settings_equal_except_learning_rate':True,'token_audit':token_audit,'phrase_proxy':phrase,'semantic_record_exact_checks':report_raw_checks,'recomputed_semantic_counts':summaries,'recomputed_selection':means,'selected_candidate':'original, already locked before final; no selection mutation','selection_sha256':sha(N/'selection.json'),'image_control':control}
(O/(P+'audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'example':stdout.getvalue(),'exercise':stdout2.getvalue(),'scene_counts':summaries['final']['scene_categories'],'means':{k:v['fraction'] for k,v in means.items()},'all_assertions':'passed'},ensure_ascii=False,indent=2))
