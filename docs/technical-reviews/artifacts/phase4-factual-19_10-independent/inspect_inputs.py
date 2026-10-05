"""Only schemas and named raw-measurement/provenance JSON pointers are exposed."""
import ast
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec={
 'docs/course-experiments/results/capstone_deployment.json':['/revision','/seed','/torch_version','/step_scale','/artifacts','/results/data_manifest','/results/joint_ptq/joint-int4','/results/stages/joint','/results/stages/joint-int4','/results/comparison_dpo_model'],
 'docs/course-experiments/results/capstone_student.json':['/revision','/seed','/torch_version','/step_scale','/artifacts','/results/data_manifest','/results/teacher_checkpoint_sha256','/results/initial_student_state_sha256','/results/steps_per_branch']+[f'/results/branches/{mode}/{field}' for mode in ['ce','kd'] for field in ['steps','requested_steps','schedule_completed','parameters','effective_tokens','teacher_checkpoint_sha256']],
 'docs/course-experiments/results/capstone_preference.json':['/results/stage','/results/inference_export/sha256'],
 'docs/course-experiments/capstone-evidence/deployment/test-joint.json':['/count','/action_correct','/end_to_end_correct','/protocol','/checkpoint_sha256','/records/20'],
 'docs/course-experiments/capstone-evidence/deployment/test-joint-ptq4.json':['/count','/action_correct','/end_to_end_correct','/protocol','/source_checkpoint_sha256','/diagnostic_stage','/records/20'],
 'docs/course-experiments/capstone-evidence/student/test-ce.json':['/count','/action_correct','/end_to_end_correct','/protocol','/records/6'],
 'docs/course-experiments/capstone-evidence/student/test-kd.json':['/count','/action_correct','/end_to_end_correct','/protocol','/records/6'],
}
result={}
for rel,pointers in spec.items():
 p=OUT/'frozen'/rel; d=json.loads(p.read_text());entry={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'top_level_schema':{k:type(v).__name__ for k,v in d.items()},'pointers':{}}
 for ptr in pointers:
  v=d
  for k in ptr.strip('/').split('/'):v=v[int(k)] if isinstance(v,list) else v[k]
  entry['pointers'][ptr]=v
 result[rel]=entry
for rel in ['scripts/course_experiments/capstone_deployment.py','scripts/course_experiments/capstone_student.py','tiny_perceptron/quantization.py','tiny_perceptron/capstone_quantization.py','tiny_perceptron/alignment.py','tiny_perceptron/capstone.py','scripts/capstone_release.py']:
 p=OUT/'frozen'/rel;t=ast.parse(p.read_text());result[rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'AST_function_locations':[{'name':n.name,'first_line':n.lineno,'last_line':n.end_lineno} for n in ast.walk(t) if isinstance(n,(ast.ClassDef,ast.FunctionDef))]}
(OUT/'input-inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({p:{'sha256':e['sha256'],'checked_pointers':list(e.get('pointers',{}))} for p,e in result.items()},ensure_ascii=False,indent=2))
