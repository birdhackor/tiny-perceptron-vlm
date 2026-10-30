"""Review-only formal codec and immutable selection checks; no formal optimizer/final calls."""
import hashlib,json,shutil,sys
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'current-source';sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import formal_local as fl, core_selection as cs
from tiny_perceptron.omni_products import training_runtime as tr
from tiny_perceptron.omni_products.inference import load_trained
root=BASE/'probes/current-gate-selection';root.mkdir(parents=True,exist_ok=False)
frozen=root/'frozen';frozen.mkdir();recipe=json.loads((SOURCE/'docs/omni-products/v1/formal-recipes.json').read_bytes())
for n in ('formal-recipes.json','data-definition.json','metrics.json'):shutil.copyfile(SOURCE/'docs/omni-products/v1'/n,frozen/n)
output=root/'output';codec=output/'codec/training'
config={'seed':20261029,'updates':2,'training_seconds':20,'batch_size':8,'learning_rate':.002,'validate_every':1,'hop':80,'codes':256,'width':32,'variants':1,'training_dead_code_reset_every':1,'review_scope':'tiny development fixture; no production permission'}
trained=tr.train_codec(config,codec,'cpu');assert load_trained(codec,'codec') is not None
tr.json_file(frozen/'codec-selection.json',{'formal_recipe_sha256':tr.file_sha(frozen/'formal-recipes.json'),'weights_sha256':tr.file_sha(codec/'weights.safetensors'),'training_result_sha256':tr.file_sha(codec/'training-result.json'),'final_opened':False,'review_fixture':True})
entered=[]
def checked(_f,j):return recipe['jobs'][j],recipe
with patch.object(fl,'checked_job',side_effect=checked),patch.object(tr,'train_talker',side_effect=lambda *a:entered.append(True)):
 try:fl.execute('talker',frozen,output)
 except RuntimeError as e:gate_error=str(e)
 else:gate_error=None
codec_case={'development_updates':trained['updates'],'formal_updates_required':recipe['jobs']['codec']['training']['updates'],'formal_codec_origin_exists':(output/'codec/job-origin.json').exists(),'talker_trainer_reached':bool(entered),'rejected':gate_error is not None,'talker_failure_receipt_exists':(output/'talker/job-result.json').exists()}
# Metadata-only formal-looking receipt around the same genuine TWO-update codec must still fail.
expected={'schema':'omni-formal-local-job-v1','job':'codec','kind':'codec','stage':'formal-training','attempt':1,'final_opened':False,'formal_recipe_sha256':tr.file_sha(frozen/'formal-recipes.json'),'data_definition_sha256':recipe['data_definition_sha256'],'metrics_sha256':recipe['metrics_sha256'],'implementation_sha256':tr.source_binding()}
tr.json_file(output/'codec/job-origin.json',expected)
tr.json_file(output/'codec/job-result.json',{**expected,'success':True,'job_origin_sha256':tr.file_sha(output/'codec/job-origin.json'),'training_result_sha256':tr.file_sha(codec/'training-result.json')})
with patch.object(fl,'checked_job',side_effect=checked):
 try:fl.verified_formal_training(output,frozen,'codec')
 except Exception as e:codec_case.update(partial_with_formal_looking_receipts_rejected=True,partial_error_type=type(e).__name__)
 else:codec_case['partial_with_formal_looking_receipts_rejected']=False
# Exercise the actual six-way chooser with a deliberately equal-score/unequal-best-step fixture.
selection_root=root/'selection-output';selection_root.mkdir();records={}
for name in recipe['job_order']:
 p=selection_root/name/'training';p.mkdir(parents=True)
 (p/'training-result.json').write_text('{"review_fixture":true}')
 (p.parent/'job-result.json').write_text('{"review_fixture":true}')
 records[name]={'initial_parameter_sha256':'same-initial','trainable_parameters':119693,'best':{'dev_score':.5,'update':1000},'files':{'weights.safetensors':{'sha256':'0'*64,'bytes':1}}}
records['av-time-rope-s29']['best']['update']=500
with patch.object(cs,'formal_result',side_effect=lambda o,f,n:(recipe['jobs'][n],recipe,{},records[n])),patch.object(cs,'load_trained',return_value=object()),patch.object(cs,'CJudge',return_value=object()),patch.object(cs,'predict_families',return_value=[]),patch.object(cs,'select_temperature',return_value={'temperature':1.0}),patch.object(cs,'select_policy',return_value={'feasible':False,'threshold':None}):
 selected=cs.select_models(selection_root,frozen,root/'evidence')
selection_case={'scope':'Synthetic equal-score records; actual select_models control, no formal training or calibration/final predictions.','best_updates':{n:records[n]['best']['update'] for n in recipe['job_order'] if recipe['jobs'][n]['kind']=='av'},'scores_equal':True,'selected':selected['selected']['av'],'expected_by_frozen_rule':'av-time-rope-s29','selection_rule':recipe['candidate_selection']['AV'],'rule_followed':selected['selected']['av']=='av-time-rope-s29'}
result={'scope':'Private review fixtures and real tiny two-update train/dev codec; never canonical output-root.','codec_gate':codec_case,'six_way_tie':selection_case,'formal_success_claimed':False}
tr.json_file(BASE/'current-gate-selection-result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
