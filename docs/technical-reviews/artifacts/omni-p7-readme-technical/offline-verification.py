"""Independent README checks: JSON/count/hash/AST only, no model execution."""
from pathlib import Path
import ast, collections, datetime, hashlib, json, math, re, subprocess, sys, tarfile, zipfile
ROOT=Path('/workspace/work/omni-book-integrated-v1')
BASE=ROOT/'docs/technical-reviews/artifacts/omni-p7-readme-technical'
RAW=Path('/workspace/work/omni-p7-readme-technical-v1/raw')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def ensure(x,why):
 if not x: raise AssertionError(why)
r={'reviewer_task':'/root/omni_p7_technical_readme','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_scope':'Offline stdlib JSON/AST/hash/arithmetic and read-only Notebook serialization contract; no model forward, training, teacher inference, final evaluation, publication gate, GPU, workflows or paid services.'}
readme=ROOT/'README.md'; fig=ROOT/'course/figures/rewrite-01-character-ids.svg'
ensure(sha(readme)=='4ab2217e7d40f0991844dec262ef9e27436a06b945aea649cc0094b3fdeedaac','README freeze')
ensure(sha(fig)=='61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad','figure freeze')
r['freeze']={'source_sha256':sha(readme),'figure_sha256':sha(fig)}
text='貓看狗，狗看貓。'; vocab=sorted(set(text)); encoded=[vocab.index(c) for c in text];ensure(encoded==[3,2,1,4,1,2,3,0],'figure IDs');r['figure_arithmetic']={'vocabulary':dict(zip(vocab,range(len(vocab)))),'original_text':text,'encoded':encoded,'convention':'Sorted Unicode vocabulary gives reversible identifier lookup, not semantic order.'}
chapters=sorted((ROOT/'course/chapters').glob('*.md')); inventory=[]
for p in chapters:
 for ident,title in re.findall(r'^## ([\dABC]+\.\d+) (.+)$',p.read_text(),re.M):
  n=ROOT/'notebooks'/p.stem/(ident+'.ipynb'); ensure(n.is_file(),ident+' Notebook')
  inventory.append({'id':ident,'title':title,'source':str(p.relative_to(ROOT)),'source_sha256':sha(p),'notebook':str(n.relative_to(ROOT)),'notebook_sha256':sha(n)})
front={str(p.relative_to(ROOT)):re.findall(r'^## ([RWTG]\.\d+) (.+)$',p.read_text(),re.M) for p in [ROOT/'course/README.md',ROOT/'course/first-steps.md',ROOT/'course/training.md',ROOT/'course/glossary.md']}
ensure(len(chapters)==23 and len(inventory)==331,'23 chapter files /331 lessons')
ensure(sum(map(len,front.values()))==26,'front 26')
ensure(len(list((ROOT/'notebooks').rglob('*.ipynb')))==331,'Notebook count')
proc=subprocess.run([sys.executable,'scripts/build_course.py','--check'],cwd=ROOT,capture_output=True,text=True)
ensure(proc.returncode==0,proc.stderr)
r['inventory']={'chapter_files':[str(p.relative_to(ROOT)) for p in chapters],'lessons':331,'front_counts':{k:len(v) for k,v in front.items()},'numbered_total':357,'notebooks':331,'serialization_command':[sys.executable,'scripts/build_course.py','--check'],'serialization_stdout':proc.stdout.strip(),'serialization_returncode':proc.returncode,'all_lesson_records':inventory}
ids=['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16']
ensure(len(ids)==18 and set(ids)<=set(x['id'] for x in inventory),'18 sections named')
r['current_eighteen']=[x for x in inventory if x['id'] in ids]
local=[]
for target in re.findall(r'\]\(([^)]+)\)',readme.read_text()):
 if '://' in target: continue
 file,_,frag=target.partition('#');p=ROOT/file;ensure(p.is_file(),'link file '+target)
 if frag:
  ensure(re.search(r'^## '+re.escape(frag)+r'(?: |$)',p.read_text(),re.M),'link anchor '+target)
 local.append(target)
r['readme_links']={'local_links_checked':len(local),'targets':local}
# Frozen v2 training and public evaluation receipts, not model cards.
v2={}
for arch,stage in [('moe','moe-native'),('dense','dense-weighted')]:
 train=load(ROOT/f'docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json')
 receipt=load(ROOT/f'docs/selftrained/results/training-raw/{stage}/receipt.json')
 metrics=load(ROOT/f'docs/selftrained/results/public-raw/{arch}/test/metrics.json')
 journal=load(ROOT/f'docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json')
 ensure(metrics['count']==journal['completed_count']==3734 and journal['status']=='complete',arch+'test count')
 ensure(sum(x['count'] for x in metrics['per_task_final_reply'].values())==3734,arch+' task denominator')
 ensure(metrics['teacher_forcing_used_for_generation'] is False,'generated test')
 ensure(train['origin']['kind']=='all-neural-weights-random' and train['test_used_for_selection'] is False,'from zero / selection')
 ensure(train['train_records']==28876 and train['validation_records']==2435,'train/val count')
 ensure(receipt['data_readiness']['record_files_verified']==12 and receipt['data_readiness']['asset_files_verified']==8950,'archive file count receipt')
 weights=next(x for x in receipt['files'] if x['path']=='model.safetensors')
 ensure(weights['sha256']==metrics['safe_weights_sha256']==journal['checkpoint_sha256'],'v2 selected/public pairing')
 tool=metrics['per_task_final_reply']['tool_call']['tool_final'];ensure(tool['rate']<metrics['thresholds']['tool_roundtrip'],'not all capability criteria pass')
 v2[arch]={'total_parameters':train['total_parameters'],'completed_stage_steps':train['steps'],'selection':train['selection'],'test_used_for_selection':train['test_used_for_selection'],'test_count':metrics['count'],'per_task_count':{k:v['count'] for k,v in metrics['per_task_final_reply'].items()},'tool_final':tool,'tool_threshold':metrics['thresholds']['tool_roundtrip'],'safe_weights_sha256':weights['sha256'],'train_receipt_sha256':sha(ROOT/f'docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json')}
ensure(v2['moe']['total_parameters']==5447107 and v2['dense']['total_parameters']==2288067,'parameter counts')
selection=load(ROOT/'docs/selftrained/results/v2-final-training-source-selection.json')
ensure(selection['moe_joint']['selected_step']==selection['dense_joint']['selected_step']==1000,'v2 selection steps')
r['v2']=v2
# Read the existing immutable data archive; no generation or materialization.
m=load(ROOT/'docs/selftrained/v2-manifest.json'); archive=ROOT/m['package']['path'];ensure(archive.stat().st_size==67862431 and sha(archive)==m['package']['sha256'],'v2 archive')
expected={x['path']:x for x in m['records']+m['assets']}; counts=collections.Counter();tasks=collections.Counter();members={}
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  if not member.isfile(): continue
  name=member.name.removeprefix('./');ensure(name not in members,'duplicate tar file');members[name]=member.size
  ensure(name in expected,'unlisted archive file '+name)
  ensure(member.size==expected[name]['bytes'],'archive file size '+name)
  if name in {x['path'] for x in m['records']}:
   b=tar.extractfile(member).read();ensure(hashlib.sha256(b).hexdigest()==expected[name]['sha256'],'record sha '+name)
   for line in b.splitlines():
    row=json.loads(line);counts[row['split']]+=1;tasks[row.get('task','unknown')]+=1
ensure(set(members)==set(expected) and len(members)==8962,'8962 packaged files')
ensure(dict(counts)=={'test':3734,'train':28876,'validation':2435},'v2 split counts')
r['v2_data']={'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'files':len(members),'record_files':len(m['records']),'other_files':len(m['assets']),'split_counts':dict(counts),'task_counts':dict(tasks),'checks':'Every archive member name/size matched manifest; all 12 record-file byte hashes verified; every JSONL record counted. Media bytes were not decoded.'}
# Public tree at the exact README revision. No tensor download.
tree=load(BASE/'v2-public-tree-original.json');files=[x for x in tree if x['type']=='file'];groups=collections.Counter(str(Path(x['path']).parent) for x in files)
ensure(len(files)==16 and len(groups)==4 and set(groups.values())=={4},'four paired inference groups')
r['v2_public_tree']={'files':len(files),'groups':dict(groups),'fixed_revision':'979cdfacc588ad0536f1c64fff96f264571cf054'}
# Offline recalculation from preserved final raw observations; never call evaluator/model functions.
av=load(RAW/'measurements/av/results.json')['models']['av-time-rope-s29'];rows=av['records'];paired=sum(x['answers']==x['gold'] for x in rows);swap=sum(x['waveform_swap_answers']==x['waveform_swap_gold'] for x in rows)
ensure(len(rows)==200 and paired==200 and swap==148,'AV paired score/diagnostic')
r['omni_AV']={'selected_job':'av-time-rope-s29','families':len(rows),'observations':sum(len(x['answers']) for x in rows),'paired_correct':paired,'same_metadata_waveform_swap_paired_correct':swap,'paired_rule':'both stored answers equal both stored gold labels; one family is the denominator unit'}
sound=load(RAW/'measurements/sound/results.json')['codec_and_gold_program'];rows=sound['rows'];good=sum(x.get('generation',{}).get('program_success',False) for x in rows);ensure(len(rows)==good==200,'sound full program score')
r['omni_sound']={'families':len(rows),'program_success':good,'eos_all':all(x['generation']['eos'] for x in rows),'capped_any':any(x['generation']['capped'] for x in rows),'analysis_success_all':all(x['generation']['analysis']['success'] for x in rows),'metric_scope':'Full requested frequency/order/onset/offset and rest pattern; free Talker.generate plus codec decode, not code teacher forcing; tolerances from frozen signal metric.'}
c=load(RAW/'measurements/choice/results.json')['primary'];sel=load(ROOT/'docs/omni-products/v1/model-selection.json');T=sel['temperature'];tau=sel['threshold'];correct=accepted=errors=0;computed=[]
for row in c['rows']:
 choice=row['tasks']['choice']; info=row['choice_info'];idx=min(range(len(choice['ids'])),key=lambda i:(-choice['logits'][i],choice['ids'][i]));answer=choice['ids'][idx]
 logits=[v/T for v in choice['logits']];top=max(logits);ps=[math.exp(v-top) for v in logits];ps=[v/sum(ps) for v in ps]
 infoidx=min(range(len(info['ids'])),key=lambda i:(-info['logits'][i],info['ids'][i]));accept=max(ps)>=tau and answer!='none' and info['ids'][infoidx]=='yes';error=accept and (answer!=choice['gold'] or not row['world_supported'] or not row['choice_information_sufficient'])
 correct+=answer==choice['gold'];accepted+=accept;errors+=error;computed.append({'family':row['family'],'answer':answer,'gold':choice['gold'],'accepted':accept,'error':error})
ensure(len(c['rows'])==60 and correct==39 and accepted==53 and errors==19,'Choice counts')
ensure([{k:x[k] for k in ['family','answer','accepted','error']} for x in computed]==[{k:x[k] for k in ['family','answer','accepted','error']} for x in c['policy']['rows']],'policy exact raw rows')
r['omni_C']={'families':len(c['rows']),'choice_correct':correct,'accepted':accepted,'accepted_errors':errors,'accepted_risk':errors/accepted,'temperature':T,'threshold':tau,'upgrade_minimum_choice_correct':load(ROOT/'docs/omni-products/v1/metrics.json')['C']['choice_minimum_correct'],'risk_limit':0.05,'method':'stdlib stable-logit answer selection, temperature softmax and frozen acceptance rule recomputed from stored original logits/gold; no prediction call'}
# First-generation B/C are a separate evaluation population.
old=load(ROOT/'docs/lesson-products/results/final-report-v1.json');ensure(old['b']['5']['boxes']['correct']==2928 and old['b']['5']['boxes']['count']==2956,'B boxes');ensure(old['b']['5']['sum']['correct']==2 and old['b']['5']['sum']['count']==42,'B sum')
for kind in ['choice','info','rubric']:ensure(old['c'][kind]['after']['correct']==old['c'][kind]['after']['count'],'C original fixed type')
training={}
for p in ['b','c']:
 t=load(BASE/f'lesson-{p}-training-original.json');ensure(t['weights_changed'] and t['random_initialization'] and t['environment']['gpu']=='NVIDIA L4','B/C training receipt');training[p]={'selected_step':t['selected_step'],'last_step':t['last_step'],'environment':t['environment'],'weights_changed':t['weights_changed']}
r['old_B_C']={'B_five_loops':{k:{f:old['b']['5'][k][f] for f in ['count','correct']} for k in ['boxes','sum']},'C_after':{k:old['c'][k]['after'] for k in ['choice','info','rubric']},'C_shift_primary':old['shifts'],'training':training,'scope':old['scope']}
public=load(ROOT/'docs/course-experiments/public-models.json')['models'];cap=load(ROOT/'docs/course-experiments/capstone-public.json')['models'];weightcount=sum(x['path'].endswith('.pt') for m in public for x in m['files']);capcount=sum(x['path'].endswith('.pt') for m in cap for x in m['files']);ensure(len(public)==30 and weightcount==120 and capcount==11,'public file counts')
r['public_models']={'experiments':len(public),'pt_files':weightcount,'capstone_models':len(cap),'capstone_pt_files':capcount}
toy=load(BASE/'toy-dry-run-original-extract.json');stdout=[json.loads(x) for x in toy['stdout'].splitlines()];v=stdout[-1];ensure(toy['returncode']==0 and v['mode']=='dry-run-no-weight-update' and v['device']=='cpu','toy raw mode');ensure(v['history'][0]['loss']==5.770333290100098 and v['history'][0]['grad_norm']==0.9595053195953369,'toy exact values');ensure(v['history'][0]['effective_tokens']==140,'toy effective tokens');ensure(math.isfinite(v['history'][0]['loss']) and math.isfinite(v['history'][0]['grad_norm']) and v['history'][0]['grad_norm']>0,'toy finite')
r['toy_original']={'command':toy['command'],'returncode':toy['returncode'],'report':v,'verification':'Parsed existing fixed stdout only; no model execution.'}
# AST contract: default dry run cannot enter optimizer.step/save branches.
tree=ast.parse((ROOT/'scripts/train.py').read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');calls=[]
def walk(n,guards=()):
 if isinstance(n,ast.If):
  for c in n.body:walk(c,guards+(ast.unparse(n.test),))
  for c in n.orelse:walk(c,guards+('not ('+ast.unparse(n.test)+')',))
  return
 if isinstance(n,ast.Call):
  label=ast.unparse(n.func)
  if label in ['optimizer.step','save_checkpoint','torch.save']:calls.append({'function':label,'line':n.lineno,'guards':list(guards)});ensure(any(g=='args.train' or g.startswith('args.train and ') for g in guards),'mutating call train guarded')
 for c in ast.iter_child_nodes(n):walk(c,guards)
walk(main);r['train_AST']={'weight_write_calls':calls,'dry_run_steps_expression':'steps = max(0, end - start) if args.train else 1','no_weight_update_verified':True,'grad_norm_API':'clip_grad_norm_ returns total pre-clipping norm, so display is not accuracy'}
r['result']='PASS: all listed assertions matched frozen README claims within their stated scope'
(BASE/'offline-verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['inventory','current_eighteen','readme_links','old_B_C']},ensure_ascii=False,indent=2))
