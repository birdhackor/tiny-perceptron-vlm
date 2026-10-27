import hashlib,json,re,sys,platform,urllib.request
from pathlib import Path
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
R=Path.cwd();out=R/'docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback/local-audit.json'
paths=[]
def get(p):
 paths.append(p);return json.loads((R/p).read_text())
def count_sections(p):
 paths.append(str(p));return re.findall(r'^## ([A-Z]|\d+)\.\d+\s', (R/p).read_text(),re.M)
x={"command":"uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback/local-audit.py","environment":{"python":sys.version.split()[0],"torch":torch.__version__,"device":"cpu","platform":platform.platform()}}
index=get('course/lesson-index.json')
chapters=list(Path('course/chapters').glob('*.md'))
refs=[Path('course/README.md'),Path('course/first-steps.md'),Path('course/training.md'),Path('course/glossary.md')]
x['inventory']={'lesson_index':len(index),'notebooks':len(list(Path('notebooks').rglob('*.ipynb'))),'numbered_chapters':len([p for p in chapters if p.stem.isdigit()]),'extension_units':[p.stem for p in chapters if not p.stem.isdigit()],'numbered_sections':sum(len(count_sections(p)) for p in chapters+refs),'reference_sections':{str(p):len(count_sections(p)) for p in refs}}
freeze=get('docs/course-review-20261027-tokenizer/freeze-03.json');f=next(p for p in freeze['pages'] if p['page_id']=='curriculum')
x['freeze']={'canonical_sha256':hashlib.sha256(Path('docs/curriculum.md').read_bytes()).hexdigest(),'snapshot_sha256':hashlib.sha256(Path(f['snapshot']).read_bytes()).hexdigest(),'expected':f['source_sha256'],'figure_sha256':hashlib.sha256(Path('course/figures/curriculum_learning_flow.svg').read_bytes()).hexdigest()}
res=get('docs/selftrained/results/v2-final-public-results.json');sel=get('docs/selftrained/results/v2-final-training-source-selection.json');x['v2']={'data_scope':res['data_scope'],'architectures':{},'test_used_for_selection':sel['test_used_for_selection'],'original_acceptance':sel['capability_acceptance']}
for arch,v in res['architectures'].items():
 cfg=SelftrainedConfig(**v['capacity']['config']);m=LimitedAssistant(cfg)
 n=sum(p.numel() for p in m.parameters())
 stages=sel[arch+'_joint'];ft=v['final_test']
 x['v2']['architectures'][arch]={'parameters_counted':n,'parameters_reported':v['capacity']['parameters'],'config':v['capacity']['config'],'test_count':ft['count'],'test_count_sum_tasks':sum(t['count'] for t in ft['per_task_final_reply'].values()),'selected_step':stages['selected_step'],'completed_steps':stages.get('completed_requested_steps',stages.get('completed_steps')),'thresholds':ft['thresholds'],'source_metrics_path':ft['source_metrics_path']}
 assert n==v['capacity']['parameters']
 del m
manifest=get('docs/selftrained/v2-manifest.json');x['v2']['initialization']=manifest['initialization'];x['v2']['manifest_scope']=manifest['scope'];x['v2']['manifest_sha256']=hashlib.sha256(Path('docs/selftrained/v2-manifest.json').read_bytes()).hexdigest()
oldpath='docs/course-repair-20261008/reviews/freeze-01/freeze/technical-data/docs/selftrained/v2-manifest.json'
old=get(oldpath);a=json.loads(json.dumps(manifest));b=json.loads(json.dumps(old));current_rev=a['package'].pop('revision');old_rev=b['package'].pop('revision');assert a==b
x['v2']['manifest_resolution']={'initial_raw_sha_assertion':'different; investigated rather than ignored','old_manifest_path':oldpath,'old_manifest_sha256':hashlib.sha256(Path(oldpath).read_bytes()).hexdigest(),'results_manifest_sha256':res['manifest_sha256'],'only_changed_field':'/package/revision','current_revision':current_rev,'historical_revision':old_rev,'all_other_fields_equal':a==b}
dep=get('docs/course-experiments/results/capstone_deployment.json')['results'];stu=get('docs/course-experiments/results/capstone_student.json')['results']
x['historical_results']={'deployment':{k:{z:dep['stages'][k][z] for z in ['count','end_to_end_correct']} for k in ['joint','dpo']},'student':{k:{z:stu['evaluations'][k][z] for z in ['count','end_to_end_correct']} for k in ['test_ce','test_kd']}}
models=get('docs/course-experiments/public-models.json')['models'];cap=get('docs/course-experiments/capstone-public.json')['models'];x['historical_publication']={'experiments':len(models),'weight_files':sum(1 for m in models for f in m['files'] if f['path'].endswith(('.pt','.pth','.safetensors'))),'integration_ids':[m['id'] for m in cap]}
nat=get('docs/natural-assistant/v4/public-release.json');x['natural_v4']={k:nat[k] for k in ['base_model','asr_model','selected_variant','adapter_parameters']}
prod=get('docs/lesson-products/results/final-report-v1.json');x['products']={'recorded_at':prod['recorded_at'],'scope':prod['scope'],'gates':prod['gates'],'b':prod['b'],'c_summary':{k:v for k,v in prod['c'].items() if not isinstance(v,(list,dict))},'shifts':prod['shifts'],'mechanisms':prod['mechanisms']}
x['products']['duplicate_example']=get('docs/lesson-products/results/duplicate-observation-example-v1.json')
x['products']['auto_policy']=get('docs/lesson-products/results/b-auto-policy.json')
# Local links and explicit anchors only, no building or running every linked lesson.
body=Path('docs/curriculum.md').read_text();links=[]
for dest in dict.fromkeys(re.findall(r'\]\(([^)]+)\)',body)):
 if dest.startswith('https://'):continue
 path,_,anchor=dest.partition('#');p=(R/'docs'/path).resolve() if path else R/'docs/curriculum.md'
 ok=p.is_file();anchor_found=not anchor
 if ok and anchor:
  b=p.read_text();anchor_found=bool(re.search(r'^## '+re.escape(anchor)+r'(?:\s|$)',b,re.M)) or ('id="'+anchor+'"' in b)
 links.append({'destination':dest,'file_exists':ok,'numbered_anchor_found':anchor_found})
x['local_links']=links
# Verify the figure's nonaccumulating gradient sequence on CPU; no training data.
p=torch.nn.Parameter(torch.tensor(1.0));opt=torch.optim.SGD([p],lr=0.1);p.grad=torch.tensor(99.0);loss=p.square();opt.zero_grad();cleared=p.grad is None or p.grad.item()==0;loss.backward();grad=p.grad.item();opt.step()
x['gradient_sequence']={'stale_grad':99,'zero_grad_cleared':cleared,'loss_before_step':loss.item(),'new_grad':grad,'value_after_step':p.item(),'expected':0.8,'tolerance':1e-6};assert cleared and grad==2 and abs(p.item()-.8)<1e-6
# UTF-8 denominator definition in the linked BPB explanation; no model score inferred.
x['bpb_derivation']={'raw':'貓🙂','utf8_bytes':len('貓🙂'.encode('utf-8')),'formula':'NLL_nats / (raw_utf8_bytes * ln(2))','observed_for_nll_5':5/(7*__import__('math').log(2))}
paths+=['docs/curriculum.md','course/figures/curriculum_learning_flow.svg','tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/tokenizer.py','tiny_perceptron/model.py','tiny_perceptron/modern.py','scripts/selftrained/hf_transport.py','scripts/build_course.py','scripts/export_course.py','pyproject.toml','tiny_perceptron/lesson_products/inference.py','tiny_perceptron/lesson_products/data.py','tiny_perceptron/lesson_products/delivery.py']
x['input_sha256']={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in dict.fromkeys(paths)}
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'inventory':x['inventory'],'parameters':{a:v['parameters_counted'] for a,v in x['v2']['architectures'].items()},'historical_results':x['historical_results'],'historical_publication':x['historical_publication'],'local_links_count':len(links),'broken_links':[l for l in links if not l['file_exists'] or not l['numbered_anchor_found']],'gradient_sequence':x['gradient_sequence'],'artifact':str(out)},ensure_ascii=False,indent=2))
