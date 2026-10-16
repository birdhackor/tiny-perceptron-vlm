import sys,json,hashlib,platform,tomllib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.reading_time import lesson_slices
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
root=Path.cwd();freeze=json.load(open('docs/course-revision-20261016/freeze-08.json'))
page=next(p for p in freeze['pages'] if p['page_id']=='curriculum')
index=json.load(open('course/lesson-index.json'))
numbered={str(p.relative_to(root)):len(lesson_slices(p.read_text())) for p in sorted((root/'course/chapters').glob('*.md'))}
front={s:len(lesson_slices((root/s).read_text())) for s in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
notebooks=list((root/'notebooks').rglob('*.ipynb'))
nav=(root/'zensical.toml').read_text();ids=['7.25','C.1','C.5','8.1','13.17','C.6','C.7','14.1','14.6','14.11','14.18','14.7','14.10','15.1']
navpos={i:nav.index('"'+i+'.md"') for i in ids}
p=json.load(open('docs/selftrained/results/v2-final-public-results.json'));models={}
for name,a in p['architectures'].items():
 model=LimitedAssistant(SelftrainedConfig(**a['capacity']['config']))
 models[name]={'parameters':sum(x.numel() for x in model.parameters()),'reported_parameters':a['capacity']['parameters'],'test_count':a['final_test']['count'],'capability_acceptance_passed':a['capability_acceptance_passed'],'selected_source':a['selected_source'],'final_test_version':{k:a['final_test'][k] for k in ['run_id','revision','evaluation_complete']},'task_counts':{k:v['count'] for k,v in a['final_test']['per_task_final_reply'].items()}}
old={}
for fn in ['capstone_deployment','capstone_student']:
 q=json.load(open(f'docs/course-experiments/results/{fn}.json'))
 old[fn]={'revision':q['revision'],'seed':q['seed'],'device':q['device'],'result_keys':list(q['results'])}
 def scan(x,path=''):
  if isinstance(x,dict):
   if path in ['/stages/joint','/stages/dpo','/evaluations/test_ce','/evaluations/test_kd'] and 'count' in x and 'end_to_end_correct' in x:old[fn][path]={k:x[k] for k in ['count','action_correct','end_to_end_correct'] if k in x}
   for k,v in x.items():scan(v,path+'/'+k)
 scan(q['results'])
pub=json.load(open('docs/course-experiments/public-models.json'));cap=json.load(open('docs/course-experiments/capstone-public.json'))
print(json.dumps({'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','platform':platform.platform()},'curriculum_snapshot':page,'source_match':hashlib.sha256((root/page['snapshot']).read_bytes()).hexdigest()==page['source_sha256'],'current_match':hashlib.sha256((root/'docs/curriculum.md').read_bytes()).hexdigest()==page['source_sha256'],'chapter_files':len(numbered),'chapter_lesson_total':sum(numbered.values()),'numbered_by_source':numbered,'front_counts':front,'all_numbered':sum(numbered.values())+sum(front.values()),'lesson_index':len(index),'notebooks':len(notebooks),'missing_index_notebooks':[i['notebook'] for i in index if not (root/i['notebook']).is_file()],'notebook_dependencies':tomllib.loads((root/'pyproject.toml').read_text())['dependency-groups']['notebook'],'nav_positions':navpos,'nav_correct':list(navpos.values())==sorted(navpos.values()),'new_ids_count':len([i for i in index if i['id'] in ['7.23','7.24','7.25'] or (i['id'].startswith('9.') and 11<=int(i['id'].split('.')[1])<=16) or (i['id'].startswith('14.') and 11<=int(i['id'].split('.')[1])<=18) or (i['id'].startswith('19.') and 13<=int(i['id'].split('.')[1])<=20)]),'v2_models':models,'data_scope':p['data_scope'],'old_results':old,'public_models_keys':list(pub['models'][0]) if isinstance(pub['models'],list) else list(pub['models'])[:2],'public_models_count':len(pub['models']),'public_model_weight_count':sum(f['path'].endswith('.pt') for x in pub['models'] for f in x['files']),'capstone_model_keys':list(cap['models']) if isinstance(cap['models'],dict) else [x.get('id') for x in cap['models']]},ensure_ascii=False,indent=2))
