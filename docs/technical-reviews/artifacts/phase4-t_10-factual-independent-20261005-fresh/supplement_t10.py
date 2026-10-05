"""Persist independently inspected source-version and raw alignment facts."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys

ART=Path(__file__).resolve().parent
ROOT=ART.parents[3]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):(ART/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

# Save only two existing original-state inputs; these are byte copies, not new weights.
copied=[]
for key in ('ce','ce_kl'):
    original=ROOT/'checkpoints/course/distillation'/f'sft-w32-{key}.pt'
    target=ART/'raw'/f'original-export-sft-w32-{key}.raw-snapshot'
    target.write_bytes(original.read_bytes())
    assert sha(original)==sha(target)
    copied.append({'original_path':str(original.relative_to(ROOT)),'permanent_copy':str(target.relative_to(ROOT)),'sha256':sha(target),'copy_kind':'unchanged existing exported inference input; no newly trained state'})

original=ast.parse((ART/'code/original/scripts/course_experiments/compression.py').read_text())
current=ast.parse((ART/'code/scripts/course_experiments/compression.py').read_text())
selected=['_teacher','_example','_dataset','_evaluate','_cache_text','_fit_text','_distill_case','run_distillation','_cache_modal','_fit_modal']
same={}
for name in selected:
    a=next(n for n in original.body if getattr(n,'name','')==name)
    b=next(n for n in current.body if getattr(n,'name','')==name)
    same[name]=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert same[name]

# Identify the current multimodal guard through source code, not an author explanation.
a=next(n for n in original.body if getattr(n,'name','')=='run_multimodal_distillation')
b=next(n for n in current.body if getattr(n,'name','')=='run_multimodal_distillation')
return_same={ast.literal_eval(k):ast.dump(v,include_attributes=False)==ast.dump(next(x for kk,x in zip(b.body[-1].value.keys,b.body[-1].value.values) if ast.literal_eval(kk)==ast.literal_eval(k)),include_attributes=False) for k,v in zip(a.body[-1].value.keys,a.body[-1].value.values)}
assert all(return_same.values())
old_text=(ART/'code/original/scripts/course_experiments/compression.py').read_text()
new_text=(ART/'code/scripts/course_experiments/compression.py').read_text()
import difflib
def operational(txt,node):return txt.splitlines()[node.lineno-1:node.body[-1].lineno-1]
diff='\n'.join(difflib.unified_diff(operational(old_text,a),operational(new_text,b),fromfile='original-5af615e',tofile='current',lineterm=''))+'\n'
(ART/'multimodal-operational-source-diff.txt').write_text(diff)

d=json.loads((ART/'raw/multimodal_distillation.json').read_text())
modal={}
for task,t in d['results']['tasks'].items():
    ce,kl=(t['runs'][m]['training'] for m in ('ce','ce_kl'))
    fields=['steps','batch_size','training_examples','effective_answer_tokens','initialization_sha256','batch_plan_sha256']
    assert all(ce[k]==kl[k] for k in fields)
    assert t['visual_tokens']=={'teacher':16,'student':4}
    assert t['teacher_provenance']['metadata']['task']==('vision' if task=='vqa' else 'joint')
    align={m:{k:t['runs'][m]['training']['alignment'][k] for k in ['teacher_prediction_rows','student_prediction_rows','answer_ids','same_answer_labels','teacher_visual_tokens','student_visual_tokens']} for m in ['ce','ce_kl']}
    for rec in align.values():
        assert len(rec['teacher_prediction_rows'])==len(rec['student_prediction_rows'])==len(rec['answer_ids'])
        assert all(x-y==12 for x,y in zip(rec['teacher_prediction_rows'],rec['student_prediction_rows'],strict=True))
    modal[task]={'data_counts':t['data']['counts'],'visual_tokens':t['visual_tokens'],'teacher_task_metadata':t['teacher_provenance']['metadata']['task'],'alignment':align,'steps':ce['steps'],'effective_answer_tokens':ce['effective_answer_tokens'],'shared_fields_checked':fields}

plan=json.loads((ART/'raw/plan.json').read_text())
entries=plan['sequence']+plan.get('supporting_experiments',[])
registry={id:{k:next(e for e in entries if e['id']==id)[k] for k in ['module','function','dependencies','assets']} for id in ['distillation','multimodal_distillation','encoders','projector','vqa','joint']}
commands=[['bash','-n',str(ART/'original-long-recipe.sh')],[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/fetch_training_assets.py'),'--help'],[str(ROOT/'.venv/bin/python'),'-m','scripts.course_experiments.run','--list-assets','distillation']]
calls=[]
for i,cmd in enumerate(commands,1):
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30,env=os.environ.copy())
    (ART/f'long-recipe-interface-{i}.stdout.txt').write_text(r.stdout)
    (ART/f'long-recipe-interface-{i}.stderr.txt').write_text(r.stderr)
    assert r.returncode==0,r.stderr
    calls.append({'command_argv':cmd,'exit_code':r.returncode,'stdout_file':f'long-recipe-interface-{i}.stdout.txt','stderr_file':f'long-recipe-interface-{i}.stderr.txt'})

raw_copies=[]
for origin,destination in [('docs/course-experiments/results/distillation.json','raw/distillation.json'),('docs/course-experiments/results/multimodal_distillation.json','raw/multimodal_distillation.json'),('outputs/private-review-artifacts/original-distillation/sft-teacher-logits.pt','raw/original-sft-teacher-logits.raw-snapshot'),('data/training/reasoning-initial/gsm8k-train-first200.jsonl','raw/gsm8k-train-first200.jsonl')]:
    assert sha(ROOT/origin)==sha(ART/destination)
    raw_copies.append({'original_path':origin,'permanent_copy':str((ART/destination).relative_to(ROOT)),'same_sha256':sha(ART/destination)})

result={'status':'all_assertions_passed','original_revision':d['revision'],'same_core_AST':same,'multimodal_return_field_AST_same':return_same,'multimodal_current_operational_difference':'Added explicit task/modality compatibility guard, and copied task into saved teacher metadata. Distillation loops, caches, student objectives and raw alignment methods are unchanged. This finding comes from operational code diff only.','multimodal_selected_raw_records':modal,'raw_multimodal_pointers':['/results/tasks/{vqa,joint}/data/counts','/results/tasks/{vqa,joint}/visual_tokens','/results/tasks/{vqa,joint}/teacher_provenance/metadata/task','/results/tasks/{vqa,joint}/runs/{ce,ce_kl}/training/{steps,batch_size,training_examples,effective_answer_tokens,initialization_sha256,batch_plan_sha256,alignment}'],'registry':registry,'unchanged_original_input_copies':copied,'unchanged_raw_copies':raw_copies,'bounded_CLI_interface_checks':calls,'long_recipe_executed':False,'scope':'Only source contract and original raw records. No download, model creation, training, or new quality scores.'}
write('supplement-verification.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
