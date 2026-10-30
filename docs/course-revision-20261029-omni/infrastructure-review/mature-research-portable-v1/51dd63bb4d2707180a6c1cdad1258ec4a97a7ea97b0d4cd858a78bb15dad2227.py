"""Private malformed core-manifest counterexample; no model/API/final calls."""
import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'original-source';sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_execution as rx,core_final as cf
root=BASE/'probes/original-core-gate';root.mkdir(parents=True,exist_ok=False);frozen=root/'docs/omni-products/v1';frozen.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('data-definition.json','metrics.json','formal-recipes.json'):(frozen/name).write_bytes((SOURCE/'docs/omni-products/v1'/name).read_bytes())
(frozen/'codec-selection.json').write_text('{"review_fixture":true}')
recipe=json.loads((frozen/'formal-recipes.json').read_bytes())
# Deliberately has no selected modules, temperature, threshold, sound_map, model SHA or completed per-job metadata.
selected={'final_opened':False,'formal_recipe_sha256':sha(frozen/'formal-recipes.json'),'codec_selection_sha256':sha(frozen/'codec-selection.json'),'all_formal_jobs':{n:{} for n in recipe['jobs']},'model_files':{n:{} for n in ('av','choice','choice_control','codec','talker')},'review_fixture':True}
(frozen/'model-selection.json').write_text(json.dumps(selected))
audit_path=root/'docs/course-revision-20261029-omni/infrastructure/final-evaluation-audit-v1.json';audit_path.parent.mkdir(parents=True)
expected=cf.evaluation_sources();audit={'status':'pass_for_reviewed_scope','reviewer_task':'/root/review_fixture_only','reviewed_files_sha256':expected,'review_fixture':True};audit_path.write_text(json.dumps(audit))
opened={'final_opened':True,'model_selection_sha256':sha(frozen/'model-selection.json'),'data_definition_sha256':sha(frozen/'data-definition.json'),'metrics_sha256':sha(frozen/'metrics.json'),'evaluation_sources_sha256':expected,'independent_audit_sha256':sha(audit_path),'review_fixture':True}
(frozen/'final-opening.json').write_text(json.dumps(opened))
def git(args,**kwargs):
 assert args[:2]==['git','show'];return (root/args[2].split(':',1)[1]).read_bytes()
cases=[]
with patch.object(rx,'ROOT',root),patch.object(rx,'FROZEN',frozen),patch.object(rx.subprocess,'check_output',side_effect=git):
 for mode in ('malformed_core_independent_audit_fixture','malformed_core_empty_reviewer'):
  if mode.endswith('empty_reviewer'):
   audit['reviewer_task']='';audit_path.write_text(json.dumps(audit));opened['independent_audit_sha256']=sha(audit_path);(frozen/'final-opening.json').write_text(json.dumps(opened))
  try:rx.checked_core('1'*40);cases.append({'case':mode,'accepted':True})
  except Exception as e:cases.append({'case':mode,'accepted':False,'error_type':type(e).__name__,'reason':str(e)})
with patch.object(cf,'FROZEN',frozen):
 try:cf.checked_selection(root/'NO_CORE_ARTIFACTS',sha(frozen/'model-selection.json'));strict={'accepted':True}
 except Exception as e:strict={'accepted':False,'error_type':type(e).__name__,'reason':str(e)}
try:rx.checked_core('1'*40)
except Exception as e:actual_absent={'accepted':False,'error_type':type(e).__name__,'reason':'Snapshot has no real codec-selection/model-selection/final-opening or final audit; rejected before paid code.'}
else:actual_absent={'accepted':True}
value={'scope':'Isolated manifest fixture, real checked_core/checked_selection functions. Git readonly HEAD/show transport substituted. No audit is represented as genuine and no neural/pretrained/final/paid call occurs.','malformed_selection_missing':['schema','selected','temperature','threshold','policy_feasible','sound_map','implementation_sha256','selection_evidence','weights/checkpoint SHA','real completed job metadata'],'mature_checked_core_cases':cases,'strict_final_selection_actual_comparison':strict,'snapshot_real_prerequisites':actual_absent,'formal_or_paid_or_final_calls':0}
p=BASE/'original-core-gate-result.json';p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(value,ensure_ascii=False,indent=2))
