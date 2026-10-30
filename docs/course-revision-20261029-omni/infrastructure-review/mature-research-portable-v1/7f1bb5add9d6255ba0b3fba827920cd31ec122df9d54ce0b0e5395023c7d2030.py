"""Strict core transport/provenance gate with explicit artificial metadata and a disclosed final-selector stub."""
import copy,datetime,hashlib,json,shutil,sys,zipfile
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'current-source';sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_core_inputs as ci,core_final as cf
from tiny_perceptron.omni_products.inference import SOUND_MAP
from tiny_perceptron.omni_products.training_runtime import source_binding
scope=json.loads((BASE/'current-execution-scope.json').read_bytes());root=BASE/'probes/current-core-transport';root.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=datetime.datetime.now(datetime.UTC).isoformat();cases=[]
for mode in ('valid_transport_delegates','empty_reviewer','opening_schema','wrong_runtime','core_failed','foreign_target','corrupt_download','selection_not_original','strict_selection_refuses'):
 folder=root/mode;folder.mkdir();frozen=folder/'docs/omni-products/v1';frozen.mkdir(parents=True)
 for n in ('data-definition.json','metrics.json','formal-recipes.json'):(frozen/n).write_bytes((SOURCE/'docs/omni-products/v1'/n).read_bytes())
 (frozen/'codec-selection.json').write_text('{"review_fixture":true}')
 recipe=json.loads((frozen/'formal-recipes.json').read_bytes());selected={'schema':'omni-core-model-selection-v1','created_at':now,'final_opened':False,'formal_recipe_sha256':sha(frozen/'formal-recipes.json'),'data_definition_sha256':sha(frozen/'data-definition.json'),'metrics_sha256':sha(frozen/'metrics.json'),'implementation_sha256':source_binding(),'selected':{'av':'av-additive-s11','choice':'c-augmented','choice_control':'c-compact','codec':'codec','talker':'talker'},'model_files':{n:{} for n in ('av','choice','choice_control','codec','talker')},'all_formal_jobs':{n:{} for n in recipe['jobs']},'temperature':1.0,'policy_feasible':False,'threshold':None,'selection_evidence':{},'sound_map':{k:[e.__dict__ for e in p] for k,p in SOUND_MAP.items()},'codec_selection_sha256':sha(frozen/'codec-selection.json'),'review_fixture':True}
 (frozen/'model-selection.json').write_text(json.dumps(selected))
 audit=folder/'audit-v2.json';audit.write_text(json.dumps({'status':'pass_for_reviewed_scope','reviewer_task':'' if mode=='empty_reviewer' else '/root/review_fixture_only','reviewed_files_sha256':scope['final_sources'],'review_fixture':True}))
 opening={'schema':'omni-core-final-opening-v1','model_selection_sha256':sha(frozen/'model-selection.json'),'data_definition_sha256':sha(frozen/'data-definition.json'),'metrics_sha256':sha(frozen/'metrics.json'),'evaluation_sources_sha256':scope['final_sources'],'independent_audit_sha256':sha(audit),'final_opened':True,'created_at':now,'runtime':{'device':'cpu','torch':str(ci.torch.__version__),'python':'3.13.5'}}
 if mode=='opening_schema':opening['schema']='wrong'
 if mode=='wrong_runtime':opening['runtime']['device']='cuda'
 (frozen/'final-opening.json').write_text(json.dumps(opening))
 inputfiles={f'frozen/{n}':(frozen/n).read_bytes() for n in ('data-definition.json','metrics.json','formal-recipes.json','codec-selection.json','model-selection.json')};inputfiles.update({f'selection-evidence/{n}':b'{"review_fixture":true}' for n in ('calibration.json','temperature.json','policy-predictions.json','policy.json')});inputfiles['core/explicit-review-placeholder.json']=b'{"not_real_training_artifacts":true}'
 archive=folder/'fixture-core.zip';manifest={n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in inputfiles.items()}
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for n,v in inputfiles.items():z.writestr(n,v)
  z.writestr('package-file-manifest.json',json.dumps(manifest))
 asset={'id':2,'name':'core-checkpoints.zip','bytes':archive.stat().st_size,'sha256':sha(archive),'roundtrip_verified':True};attempt='cpu-core-123-1';revision='1'*40
 row={'attempt_id':attempt,'revision':revision,'status':'failed-stopped' if mode=='core_failed' else 'completed','purpose':json.dumps({'job':'cpu-core-formal','formal_recipe_sha256':sha(frozen/'formal-recipes.json')}),'receipt':{'success':True,'all_training_processes_exited':True,'no_modal_compute_created':True,'release_id':1,'release_tag':'omni-core-candidate-'+attempt,'assets':[asset]}}
 owned={'draft':True,'tag_name':row['receipt']['release_tag'],'body':'Owned unpublished core weights and full optimizer/RNG; '+attempt+'; source '+revision,'target_commitish':'wrong' if mode=='foreign_target' else revision,'assets':[{'id':2,'name':asset['name'],'size':asset['bytes']}]};events=[]
 if mode=='selection_not_original':
  selected['temperature']=2.;(frozen/'model-selection.json').write_text(json.dumps(selected));opening['model_selection_sha256']=sha(frozen/'model-selection.json');(frozen/'final-opening.json').write_text(json.dumps(opening))
 def git(args,**kwargs):return (folder/args[2].split(':',1)[1]).read_bytes()
 def download(arguments):
  events.append('download_owned_exact_asset');target=Path(arguments[arguments.index('--dir')+1])/asset['name'];shutil.copyfile(archive,target)
  if mode=='corrupt_download':target.write_bytes(target.read_bytes()+b'corrupt')
 def strict(core,expected):
  events.append('delegate_strict_checked_selection');assert core.name=='core' and expected==sha(frozen/'model-selection.json')
  if mode=='strict_selection_refuses':raise ValueError('explicit strict-selector stub rejects missing genuine10-job artifacts')
  return selected
 with patch.object(ci,'ROOT',folder),patch.object(ci,'FROZEN',frozen),patch.object(cf,'AUDIT',audit),patch.object(cf,'checked_audit',return_value=scope['final_sources']),patch.object(ci.subprocess,'check_output',side_effect=git),patch.object(ci,'api',return_value=owned),patch.object(ci,'command',side_effect=download),patch.object(cf,'checked_selection',side_effect=strict):
  try:value=ci.checked_core('2'*40,{'attempts':[row]});entry={'case':mode,'accepted':True,'result_flag':value['complete_core_revalidated_without_predictions']}
  except Exception as e:entry={'case':mode,'accepted':False,'error_type':type(e).__name__,'reason':str(e)}
 entry['events']=events;expected=mode=='valid_transport_delegates';assert entry['accepted'] is expected,entry;cases.append(entry)
record={'scope':'Real check_metadata/completed_core/download-SHA/unpack/manifest/original-choice comparison with explicit private source/core/audit/GitHub fixtures. checked_audit and checked_selection are disclosed stubs; no assertion that this is a genuine completed core ZIP or full strict-artifact positive test. The strict selector itself remains reviewed by its final owner.','cases':cases,'all_expected_outcomes':True,'real_provider_or_neural_or_final_or_paid_calls':0}
p=BASE/'current-core-transport-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
