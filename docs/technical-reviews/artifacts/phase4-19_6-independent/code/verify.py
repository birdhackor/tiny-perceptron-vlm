import ast,copy,hashlib,json,sys
from collections import Counter
from pathlib import Path
import torch,yaml
root=Path.cwd();sys.path.insert(0,str(root));base=root/'docs/technical-reviews/artifacts/phase4-19_6-independent'
from tiny_perceptron.capstone import CapstoneModel,build_dataset,modality_tensors,prepare_batch,prompt_ids
from tiny_perceptron.multimodal import log_mel,tone
from tiny_perceptron.data import ByteTokenizer

def show(name,value):print(name,json.dumps(value,ensure_ascii=False,sort_keys=True))
torch.set_num_threads(1);torch.manual_seed(42);torch.set_default_device('cpu');assert torch.version.cuda is None and not torch.cuda.is_available()
splits,_=build_dataset();row=next(r for r in splits['train'] if r['task']=='joint');image,audio=modality_tensors(row)
f=torch.nn.functional.adaptive_avg_pool2d(image[None],(4,4))
manual=image.reshape(3,4,4,4,4).mean((2,4));assert torch.allclose(f[0],manual,atol=1e-6,rtol=0)
mel=log_mel(tone(440,seconds=.04),bands=16);assert tuple(mel.shape)==(16,5)
assert torch.allclose(mel.mean(-1),mel[:,torch.tensor([4,2,0,3,1])].mean(-1),atol=1e-6,rtol=0)
assert not torch.equal(mel,mel[:,torch.tensor([4,2,0,3,1])])
show('FEATURE_AXES',{'image':list(image.shape),'pool':list(f.shape),'flat_width':f.flatten(1).shape[1],'waveform_samples':len(tone(440,seconds=.04)),'log_mel_band_time_shape':list(mel.shape),'time_reordering_mean_equal':True,'audio_summary':list(audio.shape),'manual_pool_max_error':float((f[0]-manual).abs().max())})
model=CapstoneModel();before={k:v.clone() for k,v in model.state_dict().items()};batch,labels=prepare_batch([row])
with torch.no_grad():out=model(**batch)
assert out['logits'].shape[:2]==labels.shape and out['logits'].shape[-1]==ByteTokenizer.vocab_size==264
assert not out['logits'].requires_grad and all(p.grad is None for p in model.parameters())
assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
changed=copy.deepcopy(row);changed['user']+='請保持簡短。'
variant,y=prepare_batch([row,changed])
with torch.no_grad():v=model(**variant)
assert v['logits'].shape[:2]==y.shape and not v['logits'].requires_grad
show('FORWARD_VARIANT',{'original_logits':list(out['logits'].shape),'original_labels':list(labels.shape),'variant_logits':list(v['logits'].shape),'variant_labels':list(y.shape),'valid_lengths':variant['valid'].sum(-1).tolist(),'no_grad':not out['logits'].requires_grad,'parameters_unchanged':True,'optimizer_or_backward_called':False})
show('CHINESE_PAPER_MATERIAL',{'alphabet':'大小上下左右開關入出人口','count':len('大小上下左右開關入出人口'),'unique':len(set('大小上下左右開關入出人口')),'upper':'入口','lower':'出口','selected_lower_expected':'出口'})
readme=(base/'sources/minds14-readme.md').read_text();meta=yaml.safe_load(readme.split('---',2)[1]);zh=next(x for x in meta['dataset_info'] if x['config_name']=='zh-CN');fields=[x['name'] for x in zh['features']];intent=next(x for x in zh['features'] if x['name']=='intent_class')['dtype']['class_label']['names']
assert fields==['path','audio','transcription','english_transcription','intent_class','lang_id'];assert [intent[str(i)] for i in [1,2,6]]==['address','app_error','card_issues']
show('OFFICIAL_ZH_SCHEMA',{'fields':fields,'chosen_intents':{str(i):intent[str(i)] for i in [1,2,6]},'speaker_id_present':False,'assistant_reply_present':False,'train_examples':zh['splits'][0]['num_examples']})
rawdata=json.loads((base/'sources/pinned-data.json').read_text());test=rawdata['splits']['test'];assert test==splits['test'];assert rawdata['manifest']['families']['test']['modalities']==['modalities:green:square']
show('RAW_DATA_INSPECTION',{'pointers':['/splits/test','/manifest/seed','/manifest/counts','/manifest/families/test'],'types':{k:type(v).__name__ for k,v in rawdata.items()},'seed':rawdata['manifest']['seed'],'counts':rawdata['manifest']['counts'],'test_task_counts':dict(Counter(r['task'] for r in test)),'current_generated_test_equals_pinned_test':True,'test_modality_families':rawdata['manifest']['families']['test']['modalities']})
# Use exact original experiment methods found by AST; no training/evaluation of saved weights.
pinned=base/'sources/pinned-capstone_deployment.py';tree=ast.parse(pinned.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['image_counterfactuals','counterfactual_pairs']];ns={'copy':copy};exec(compile(ast.Module(body=selected,type_ignores=[]),str(pinned),'exec'),ns)
image_rows=ns['image_counterfactuals'](test);audio_rows=[]
for r in test:
 if r['task']=='joint':
  s=copy.deepcopy(r);s['audio']['pitch']='high' if r['audio']['pitch']=='low' else 'low';s['answer']=f"DIRECT:{r['image']['color']},{s['audio']['pitch']}";s['id']=r['id']+'-audio-swap';audio_rows.append(s)
original=json.loads((base/'sources/pinned-test-joint.json').read_text());swaps=json.loads((base/'sources/pinned-test-joint-image-swaps.json').read_text());pairs=json.loads((base/'sources/pinned-test-joint-image-pairs.json').read_text());audio_swap=json.loads((base/'sources/pinned-test-joint-audio-swaps.json').read_text())
byid={r['id']:r for r in original['records']};rawid={r['id']:r for r in test};pairmap={r['original_id']:r for r in pairs['pairs']}
raw_expected={'original':{r['id']:r['answer'] for r in test},'image':{r['id']:r['answer'] for r in image_rows},'audio':{r['id']:r['answer'] for r in audio_rows}}
for name,evaluation in [('original',original),('image',swaps),('audio',audio_swap)]:
 for r in evaluation['records']:
  if r['task'] not in ['image_color','image_shape','joint']:continue
  assert r['expected_action']==raw_expected[name][r['id']]
  action=r['action_trace']['eos'] and r['action_trace']['raw']==r['expected_action']
  final=r['expected_action'].split(':',1)[1]
  assert r['expected_final']==final and r['action_correct']==action
  assert r['end_to_end_correct']==(action and r['answer']==final)
 for task in ['image_color','image_shape','joint']:
  records=[r for r in evaluation['records'] if r['task']==task]
  if not records:continue
  assert evaluation['by_task'][task]['count']==len(records)
  assert evaluation['by_task'][task]['end_to_end_correct']==sum(r['end_to_end_correct'] for r in records)
checkpairs=ns['counterfactual_pairs'](original,swaps);assert checkpairs['pairs']==pairs['pairs']
assert checkpairs['count']==pairs['count']==36 and checkpairs['both_end_to_end_correct']==pairs['both_end_to_end_correct']==27
assert original['checkpoint_sha256']==swaps['source_checkpoint_sha256']==audio_swap['source_checkpoint_sha256']
summary={}
for task,n in [('image_color',9),('image_shape',9),('joint',18)]:
 ori=[r for r in original['records'] if r['task']==task];sw=[r for r in swaps['records'] if r['task']==task]
 assert len(ori)==len(sw)==n
 chosen=[pairmap[r['id']] for r in ori];summary[task]={'denominator_pairs':n,'original_correct':sum(r['end_to_end_correct'] for r in ori),'swapped_correct':sum(r['end_to_end_correct'] for r in sw),'both_correct':sum(r['both_end_to_end_correct'] for r in chosen),'generated_answer_changed':sum(r['generated_answer_changed'] for r in chosen),'original_generated':dict(Counter(r['action_trace']['raw'] for r in ori)),'swapped_generated':dict(Counter(r['action_trace']['raw'] for r in sw))}
assert [(summary[t]['original_correct'],summary[t]['swapped_correct'],summary[t]['both_correct']) for t in ['image_color','image_shape','joint']]==[(9,9,9),(0,9,0),(18,18,18)]
for old,new in zip([r for r in test if r['task']=='joint'],audio_rows,strict=True):
 assert old['image']==new['image'] and old['user']==new['user'] and old['audio']['pitch']!=new['audio']['pitch']
 a=byid[old['id']];b=next(r for r in audio_swap['records'] if r['id']==new['id']);assert a['end_to_end_correct'] and b['end_to_end_correct'];assert a['answer']!=b['answer']
# Regenerate only bounded synthetic test pixels/features; never generate new trained-model answers.
changed_pixels=changed_audio=0
for new in image_rows:
 old=rawid[new['id'].removesuffix('-image-swap')];assert old['user']==new['user'];a,_=modality_tensors(old);b,_=modality_tensors(new);assert not torch.equal(a,b);changed_pixels+=1
 if old['task']=='joint':assert old['audio']==new['audio']
for new in audio_rows:
 old=rawid[new['id'].removesuffix('-audio-swap')];a,_=modality_tensors(old);b,_=modality_tensors(new);assert torch.equal(a,b);_,a=modality_tensors(old);_,b=modality_tensors(new);assert not torch.equal(a,b);changed_audio+=1
show('RECOMPUTED_IMAGE_PAIRS',summary);show('RECOMPUTED_AUDIO_PAIRS',{'pairs':18,'original_correct':18,'swapped_correct':18,'both_correct':18,'image_fixed':True,'question_fixed':True,'low_high_reversed':True})
show('BOUNDED_MATERIAL_CHECK',{'image_pixel_replacements':changed_pixels,'audio_feature_replacements':changed_audio,'new_trained_model_generation':False,'checkpoint_used_only_as_recorded_sha':original['checkpoint_sha256']})
show('ENVIRONMENT',{'python':sys.version,'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda),'threads':torch.get_num_threads(),'source':'raw original JSON and explicitly selected methods; author commentary values not accessed'})
print('PASS all original numeric, feature, label, provenance and paired-answer assertions')
