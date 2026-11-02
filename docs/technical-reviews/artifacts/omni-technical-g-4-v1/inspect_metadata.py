"""Inspect only primary metadata/results and tiny deterministic local code; no training."""
import ast,hashlib,json,platform,re
from datetime import UTC,datetime
from pathlib import Path
from types import SimpleNamespace
import torch
from tiny_perceptron.model import ModelConfig,TinyLM
from tiny_perceptron.modern import MoEFFN

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
CACHE=Path('/workspace/work/reference/omni-technical-g-4-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'created_at':datetime.now(UTC).isoformat(),'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-g-4-v1/inspect_metadata.py',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},'teachers':[]}
for name in ['distillation','multimodal_distillation']:
 path=ROOT/f'docs/course-experiments/results/{name}.json'
 data=json.loads(path.read_text())
 for task,entry in data['results']['tasks'].items():
  t=entry['teacher_provenance']
  result['teachers'].append({'source':str(path.relative_to(ROOT)),'source_sha256':sha(path),'task':task,
   'locator':f'/results/tasks/{task}/teacher_provenance','experiment':t['experiment'],'checkpoint_sha256':t['sha256'],
   'steps':t['steps'],'config':t['config']})
result['published']={}
for kind in ['b','c']:
 manifest=json.loads((CACHE/f'{kind}-artifact-manifest.json').read_text())
 receipt_path=CACHE/f'{kind}-training-receipt.json'
 receipt=json.loads(receipt_path.read_text())
 assert sha(receipt_path)==manifest['files']['training-receipt.json']['sha256']
 result['published'][kind]={'manifest_sha256':sha(CACHE/f'{kind}-artifact-manifest.json'),
   'training_receipt_sha256':sha(receipt_path),'training_receipt_matches_published_manifest':True,
   **{k:receipt[k] for k in ['random_initialization','resume_source','weights_changed','steps_this_attempt','selected_step','initial_weights_sha256','selected_weights_sha256','product','scope']}}
a=json.loads((CACHE/'a-inference-manifest.json').read_text())
result['published']['a']={'manifest_sha256':sha(CACHE/'a-inference-manifest.json'),'origin_kind':a['origin']['kind'],
 'seed':a['origin']['seed'],'stage':a['stage'],'selected_step':a['selected_step'],
 'selected_checkpoint_sha256':a['selected_checkpoint_sha256'],'payload_files':list(a['files'])}
data=json.loads((ROOT/'docs/selftrained/results/v2-final-public-results.json').read_text())
result['a_historical_data_scope']={k:data[k] for k in ['hf_repo','hf_revision','release_run','data_scope','comparison_limit']}

# The precise local proxy implementation, extracted by AST and actually called.
path=ROOT/'scripts/course_experiments/architecture.py'
text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_parameter_budget')
code=ast.get_source_segment(text,node);ns={'MoEFFN':MoEFFN};exec(compile(code,str(path),'exec'),ns)
config=ModelConfig(vocab_size=11,width=4,layers=2,heads=1,max_length=7,experts=4,top_k=2)
model=TinyLM(config);budget=ns['_parameter_budget'](model)
expert=sum(p.numel() for b in model.blocks for p in b.ffn.experts.parameters())
nonexpert=sum(p.numel() for p in model.parameters())-expert
assert budget['active_parameter_proxy_per_token']==nonexpert+expert*2//4
result['parameter_proxy']={'function_path':str(path.relative_to(ROOT)),'function_lines':[node.lineno,node.end_lineno],
 'function_sha256':hashlib.sha256(code.encode()).hexdigest(),'config':vars(config),'budget':budget,
 'nonexpert':nonexpert,'selected_expert':expert*2//4,'whole_embedding_parameters':model.embedding.weight.numel(),
 'whole_position_parameters':model.position.weight.numel(),'theory_scope':'parameter count only; no throughput or quality measurement'}

# Record exact linked target slices, avoiding whole-chapter dependency pins.
body=(ROOT/'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/G.4.md').read_text()
targets={}
for href in re.findall(r'(?<!!)\[[^]]+\]\(([^)]+)\)',body):
 if href in targets:continue
 rel,_,fragment=href.partition('#');p=ROOT/'course'/rel;t=p.read_text()
 match=re.search(rf'^## {re.escape(fragment)}(?:\s|$).*?(?=^## |\Z)',t,re.M|re.S) if fragment else None
 section=match.group() if match else t[:t.find('\n## ')]
 targets[href]={'path':str(p.relative_to(ROOT)),'selector':f'section:{fragment}' if fragment else 'introduction',
  'sha256':hashlib.sha256(section.encode()).hexdigest(),'start_line':t[:match.start()].count('\n')+1 if match else 1}
result['necessary_target_slices']=targets
result['limits']=['No tensors downloaded, no published model forward run, no GPU use, no optimizer update.',
 'Published manifests and training receipts support historical provenance only; G.4 reports no numeric quality score.',
 'This audit does not certify new A audiovisual/tone or C-v2 training, which the text labels design and future acceptance.']
(OUT/'metadata-output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['created_at','environment','teachers','published','parameter_proxy','limits']},ensure_ascii=False,indent=2))
