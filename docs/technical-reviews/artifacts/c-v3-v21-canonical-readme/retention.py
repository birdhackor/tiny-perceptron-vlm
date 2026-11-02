from pathlib import Path
from datetime import datetime,timezone
import ast,collections,difflib,hashlib,importlib.metadata,json,math,platform,re,subprocess,tarfile,tomllib,zipfile
R=Path('/tmp/tpb').resolve()
A=R/'docs/technical-reviews/artifacts/c-v3-v21-canonical-readme'
B=R/'docs/technical-reviews/artifacts/omni-p7-readme-technical'
def h(b):return hashlib.sha256(b).hexdigest()
def hf(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def j(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):
 p=Path(p)
 return str(p.relative_to(R)) if p.is_relative_to(R) else str(p)
out={'actor':'/root/c_v3_technical_coord/canonical_readme','created_at':datetime.now(timezone.utc).isoformat(),'physical_root':str(R),'environment':{'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'platform':platform.platform(),'device':'cpu; stdlib JSON/hash/arithmetic only'},'scope':'Post-original legal retention. Read original immutable source/execution bytes and bounded metadata/logit arithmetic; no original verdict or reviewer pass flag used; no model forward, teacher, training, kernel, final opening, network, installation, publication, or full product acceptance replay.'}
receipt=R/'docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-readme-original-v1.json';rr=j(receipt)
assert hf(receipt)=='3fade71eaaa0ca9e8c3f720197f8a2d2ef83e0218482f2fcb40e8de8d1d3371e'
assert hf(R/rr['report']['path'])==rr['report']['sha256']=='da05ffd71e0165e414f4643b11919231e6f40f24d99e0c74dce573dc4984375a'
mapp=R/'docs/course-revision-20261101-c-v3/technical-coordination-v21/prior-evidence-reference-map-resume-v1.json';entry=next(x for x in j(mapp)['pages'] if x['page_id']=='readme');proof=entry['prior_technical_proof']
old=j(R/proof['original_report']['path']);oldcurrent=j(R/proof['report']['path']);src={x['id']:x for x in old['sources']}
assert hf(R/proof['original_report']['path'])==proof['original_report']['sha256']
assert hf(R/proof['report']['path'])==proof['report']['sha256']
out['intake']={'receipt':rel(receipt),'sha256':hf(receipt),'received_at':rr['received_at'],'map':rel(mapp),'map_sha256':hf(mapp),'selected_page_id':'readme'}
out['retained_original_attribution']={'actor':old['reviewer_task'],'original_created_at':old['created_at'],'current_created_at':oldcurrent['created_at'],'original_report':proof['original_report'],'current_report':proof['report'],'source18_freeze':proof['original_source_freeze'],'original_collection':proof['original_collection'],'current_collection':proof['current_collection'],'use':'Only primary source/execution locators, source identity, provenance, command/results and bounded underlying data; no pass/verdict transfer, no peer-comparison/historical-question artifacts opened.'}
snapshots=j(B/'source-snapshots.json');snapmap={x['path']:x for x in snapshots}
for x in snapshots:assert hf(R/x['snapshot'])==x['snapshot_sha256']==x['sha256']
repo_sources=[]
for s in old['sources']:
 if s['kind']!='repository_code':continue
 p=R/s['path'];current_sha=hf(p);row={'id':s['id'],'path':s['path'],'recorded_sha256':s['sha256'],'current_sha256':current_sha}
 if current_sha!=s['sha256']:
  snap=snapmap[s['path']];assert hf(R/snap['snapshot'])==s['sha256'];row['prior_exact_snapshot']=snap['snapshot']
 repo_sources.append(row)
artifact_hashes=[]
for x in old['artifacts']:
 assert hf(R/x['path'])==x['sha256'];artifact_hashes.append({'id':x['id'],'path':x['path'],'sha256':x['sha256'],'kind':x['kind']})
out['original_dependency_identity']={'prior_source_snapshot_files_verified':len(snapshots),'prior_original_artifacts_verified':artifact_hashes,'all_repository_sources':repo_sources,'current_changed_sources':[x for x in repo_sources if x['current_sha256']!=x['recorded_sha256']],'changed_source_scope':'331→332/source21 nav and newC text are covered by this actor fresh original. Zensical nav changed; retained theme/features, JS/CSS/footer implementation are separately checked below. Never blanket-retain a changed whole file.'}
ext=j(B/'original-source-excerpts.json');official=[]
for x in ext['sources']:
 p=Path(x['full_source_external_path']);exists=p.exists();match=exists and hf(p)==x['full_source_sha256']
 row={'id':x['id'],'url':x['url'],'original_accessed_at':x['accessed_at'],'locator':x['locator'],'full_source_sha256':x['full_source_sha256'],'original_path':str(p),'original_still_available':exists,'fingerprint_verified':match}
 if x['id'] in {'uv-sync','git-lfs-config','git-windows-config','public-current-license-declaration'}:
  assert match
  original=p.read_text(encoding='utf-8');row['small_original_excerpt']=x['small_original_excerpt'];row['current_inspection']='Personally read the indicated retained original section after verifying full-source bytes; original acquisition dates/URLs retained, no new network retrieval.'
  if x['id']=='git-lfs-config':assert 'GIT_LFS_SKIP_SMUDGE' in original and 'filter-process' in original
  if x['id']=='git-windows-config':assert 'core.longpaths::' in original and 'disabled by default' in original
  if x['id']=='public-current-license-declaration':assert '979cdfacc588ad0536f1c64fff96f264571cf054' in row['small_original_excerpt'] and 'MIT' in row['small_original_excerpt']
 official.append(row)
out['official_original_retention']=official
# v2 original training and evaluation records; no model construction.
v2={}
for arch,stage,expectedparams,expectedsteps in [('moe','moe-native',5447107,4000),('dense','dense-weighted',2288067,10000)]:
 trainp=R/src['r_v2train_'+arch]['path'];train=j(trainp);metrics=j(R/src['r_v2metrics_'+arch]['path']);journal=j(R/src['r_v2journal_'+arch]['path']);outer=j(R/src['r_v2outer_'+arch]['path']);evalouter=j(R/f'docs/selftrained/results/public-raw/{arch}/test/receipt.json')
 assert train['total_parameters']==expectedparams and train['steps']==expectedsteps and train['completed_requested_steps']
 assert train['origin']['kind']=='all-neural-weights-random' and train['test_used_for_selection'] is False
 assert train['train_records']==28876 and train['validation_records']==2435
 assert metrics['count']==journal['completed_count']==journal['expected_count']==3734
 assert journal['status']=='complete' and sum(x['count'] for x in metrics['per_task_final_reply'].values())==3734
 tool=metrics['per_task_final_reply']['tool_call']['tool_final'];assert tool['rate']<metrics['thresholds']['tool_roundtrip']
 public=evalouter['job']['public_export'];weights=next(x for x in public['files'] if x['path']=='model.safetensors');assert weights['sha256']==metrics['safe_weights_sha256']
 assert evalouter['data_readiness']['archive_sha256']=='0976073a3bc7c331a65cedb4c31d54f5c6e9a5014ff2443698a8e2b8cad7e78a'
 v2[arch]={'train_receipt':rel(trainp),'train_receipt_sha256':hf(trainp),'origin_kind':train['origin']['kind'],'parameters':train['total_parameters'],'completed_stage_steps':train['steps'],'train_records':train['train_records'],'validation_records':train['validation_records'],'selection':train['selection'],'test_used_for_selection':train['test_used_for_selection'],'test_count':metrics['count'],'per_task_counts':{k:v['count'] for k,v in metrics['per_task_final_reply'].items()},'tool_final':tool,'tool_threshold':metrics['thresholds']['tool_roundtrip'],'selected_public_weights':weights,'public_revision':public['revision'],'evaluation_receipt_sha256':hf(R/src['r_v2journal_'+arch]['path']),'original_evaluation_updated_at':journal['updated_at'],'original_training_run':outer.get('run_id'),'original_training_started_at':outer.get('started_at'),'original_evaluation_run':evalouter['run_id'],'original_evaluation_started_at':evalouter['started_at'],'scope':'Retained original trainer counters/evaluator records; fixed3734 questions share source assets; no new capability evaluation or full model materialization.'}
manifest=j(R/'docs/selftrained/v2-manifest.json');archive=R/manifest['package']['path'];assert archive.stat().st_size==67862431 and hf(archive)==manifest['package']['sha256']
expected={x['path']:x for x in manifest['records']+manifest['assets']};recordpaths={x['path'] for x in manifest['records']};seen={};splits=collections.Counter();recordhashes=[]
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  if not member.isfile():continue
  name=member.name.removeprefix('./');assert name in expected and name not in seen and member.size==expected[name]['bytes'];seen[name]=member.size
  if name in recordpaths:
   data=tar.extractfile(member).read();assert h(data)==expected[name]['sha256'];rows=[json.loads(x) for x in data.splitlines() if x];split=Path(name).stem.rsplit('-',1)[1];splits[split]+=len(rows);recordhashes.append({'path':name,'bytes':len(data),'sha256':h(data),'records':len(rows),'split':split})
assert len(seen)==8962 and dict(splits)=={'test':3734,'train':28876,'validation':2435}
v2['data']={'archive_path':rel(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':hf(archive),'file_count':len(seen),'record_files':recordhashes,'split_counts':dict(splits),'scope':'Read existing immutable tar in place; no extraction/materialization/copy, no media decode, no training.'}
tree=j(B/'v2-public-tree-original.json');files=[x for x in tree if x['type']=='file'];groups=collections.Counter(str(Path(x['path']).parent) for x in files);assert len(files)==16 and len(groups)==4 and set(groups.values())=={4}
selection=j(R/'docs/selftrained/results/v2-final-training-source-selection.json');assert selection['moe_joint']['selected_step']==selection['dense_joint']['selected_step']==1000
v2['public']={'original_tree_path':rel(B/'v2-public-tree-original.json'),'original_tree_sha256':hf(B/'v2-public-tree-original.json'),'revision':'979cdfacc588ad0536f1c64fff96f264571cf054','files':len(files),'four_groups':dict(groups),'moe_selected_step':1000,'dense_selected_step':1000,'selected_source_sha256':hf(R/'docs/selftrained/results/v2-final-training-source-selection.json'),'license_original':'public-current-license-declaration above','no_private_volume_basis':'Read source-identical scripts/selftrained/train.py export_inference model-only safetensors and full checkpoint optimizer/RNG path; full local checkpoint is needed, public inference weights alone cannot resume exactly.'}
out['v2_retention']=v2
# Ten original CPU jobs and fixed final package; source identity, not pass flags.
acq=j(B/'omni-training-acquisition.json')
for x in acq['records']:assert hf(R/x['path'])==x['sha256'] and (R/x['path']).stat().st_size==x['bytes']
sel=j(R/'docs/omni-products/v1/model-selection.json');jobs=[]
for job,spec in sel['all_formal_jobs'].items():
 folder=B/'omni-training-primary/core'/job;origin=j(folder/'job-origin.json');jobresult=j(folder/'job-result.json');train=j(folder/'training/training-result.json');logs=j(folder/'training/training-log.json')
 assert hf(folder/'job-result.json')==spec['job_result_sha256'] and hf(folder/'training/training-result.json')==jobresult['training_result_sha256']
 assert hf(folder/'training/training-log.json')==train['files']['training-log.json']['sha256']
 assert train['updates']>0 and train['initial_parameter_sha256']!=train['latest_parameter_sha256']
 assert logs[-1]['update']==train['updates'] and jobresult['runtime']['device']=='cpu' and jobresult['runtime']['torch']=='2.14.1+cpu'
 assert not jobresult['final_opened'] and not train['final_opened']
 jobs.append({'job':job,'original_started_at':origin['started_at'],'original_finished_at':jobresult['finished_at'],'updates':train['updates'],'selected_update':train['best']['update'],'initial_parameter_sha256':train['initial_parameter_sha256'],'latest_parameter_sha256':train['latest_parameter_sha256'],'runtime':jobresult['runtime'],'execution_host':jobresult['execution_host'],'job_result_sha256':hf(folder/'job-result.json'),'training_result_sha256':hf(folder/'training/training-result.json'),'training_log_sha256':hf(folder/'training/training-log.json')})
assert len(jobs)==10
rawacq=j(B/'omni-raw-acquisition.json');zipath=Path(rawacq['external_archive_snapshot']);assert hf(zipath)==rawacq['archive_sha256']
with zipfile.ZipFile(zipath) as z:
 members=[]
 for x in rawacq['members']:
  data=z.read(x['member']);assert h(data)==x['sha256'] and len(data)==x['bytes'];members.append({'member':x['member'],'sha256':h(data),'bytes':len(data)})
 av=json.loads(z.read('measurements/av/results.json'))['models']['av-time-rope-s29']['records'];sound=json.loads(z.read('measurements/sound/results.json'))['codec_and_gold_program']['rows'];choice=json.loads(z.read('measurements/choice/results.json'))['primary']
 pairs=sum(x['answers']==x['gold'] for x in av);swap=sum(x['waveform_swap_answers']==x['waveform_swap_gold'] for x in av);tones=sum(x['generation']['program_success'] and x['generation']['analysis']['success'] and x['generation']['eos'] and not x['generation']['capped'] for x in sound)
 assert len(av)==pairs==200 and swap==148 and len(sound)==tones==200
 correct=accepted=errors=0;policyrows=[];temp=sel['temperature'];threshold=sel['threshold']
 for row in choice['rows']:
  c=row['tasks']['choice'];info=row['choice_info'];idx=min(range(len(c['ids'])),key=lambda i:(-c['logits'][i],c['ids'][i]));answer=c['ids'][idx];scaled=[v/temp for v in c['logits']];top=max(scaled);ps=[math.exp(v-top) for v in scaled];confidence=max(ps)/sum(ps);ii=min(range(len(info['ids'])),key=lambda i:(-info['logits'][i],info['ids'][i]));accept=confidence>=threshold and answer!='none' and info['ids'][ii]=='yes';error=accept and (answer!=c['gold'] or not row['world_supported'] or not row['choice_information_sufficient']);correct+=answer==c['gold'];accepted+=accept;errors+=error;policyrows.append({'family':row['family'],'answer':answer,'accepted':accept,'error':error})
 assert len(choice['rows'])==60 and correct==39 and accepted==53 and errors==19
 assert policyrows==[{k:x[k] for k in ['family','answer','accepted','error']} for x in choice['policy']['rows']]
 parts={name:json.loads(z.read('measurements/'+name+'/part-completion.json')) for name in ['av','choice','sound','integration']}
out['omni_retention']={'retained_acquisition_path':rel(B/'omni-raw-acquisition.json'),'retained_acquisition_sha256':hf(B/'omni-raw-acquisition.json'),'original_acquired_at':rawacq['created_at'],'fixed_original_url':rawacq['download_url'],'original_zip_path':str(zipath),'original_zip_sha256':hf(zipath),'new_download':False,'original_training_acquisition_sha256':hf(B/'omni-training-acquisition.json'),'forty_original_training_records_hash_verified':len(acq['records']),'ten_original_jobs':jobs,'used_raw_member_fingerprints':members,'original_part_completions':parts,'raw_stored_observation_recount':{'av_paired_worlds':200,'av_questions':400,'av_both_correct_pairs':pairs,'waveform_swap_pairs_correct':swap,'waveform_swap_pairs_failed':200-swap,'sound_full_programs':200,'sound_full_success':tones,'c_v2_choice_examples':60,'c_v2_choice_correct':correct,'c_v2_accepted':accepted,'c_v2_accepted_errors':errors,'temperature':temp,'threshold':threshold},'scope':'Counters from original training logs and unchanged frozen raw observations only. No new model call/final opening; AV paired worlds, tones and C-v2 populations remain separate; swaps reuse same200 worlds.'}
# Existing first-generation fixed result is an original first-party measurement, not an AI review.
oldfinal=j(R/'docs/lesson-products/results/final-report-v1.json');start=j(R/'docs/lesson-products/results/final-report-v1.started.json');settings=j(R/'docs/lesson-products/results/settings-selection-v1.json')
assert oldfinal['settings_freeze_sha256']==start['settings_freeze_sha256']==settings['settings_freeze_sha256']
assert start['recorded_at']<oldfinal['recorded_at']
assert oldfinal['b']['5']['boxes']['correct']==2928 and oldfinal['b']['5']['boxes']['count']==2956 and oldfinal['b']['5']['sum']['correct']==2 and oldfinal['b']['5']['sum']['count']==42
assert all(oldfinal['c'][kind]['after']['correct']==oldfinal['c'][kind]['after']['count'] for kind in ['choice','info','rubric'])
assert oldfinal['shifts']['c_wording']['choice']['after']['correct']<oldfinal['shifts']['c_wording']['choice']['after']['count']
training={}
for p in ['b','c']:
 t=j(B/f'lesson-{p}-training-original.json');assert t['initial_weights_sha256']!=t['selected_weights_sha256'] and t['environment']['gpu']=='NVIDIA L4'
 training[p]={'original_public_receipt':rel(B/f'lesson-{p}-training-original.json'),'sha256':hf(B/f'lesson-{p}-training-original.json'),'run_identity':t['run_identity'],'selected_step':t['selected_step'],'last_step':t['last_step'],'environment':t['environment'],'initial_weights_sha256':t['initial_weights_sha256'],'selected_weights_sha256':t['selected_weights_sha256'],'mode':t['mode'],'random_initialization':t['random_initialization']}
pred=R/'docs/lesson-products/results'/oldfinal['predictions']['file'];predref={**oldfinal['predictions'],'available':pred.exists()}
if pred.exists():assert hf(pred)==oldfinal['predictions']['sha256'];predref['actual_sha256']=hf(pred)
out['legacy_b_c_retention']={'original_final_path':'docs/lesson-products/results/final-report-v1.json','original_final_sha256':hf(R/'docs/lesson-products/results/final-report-v1.json'),'original_final_recorded_at':oldfinal['recorded_at'],'settings_frozen_at':start['recorded_at'],'settings_freeze_sha256':settings['settings_freeze_sha256'],'B_five_loop_boxes':oldfinal['b']['5']['boxes'],'B_five_loop_sum':oldfinal['b']['5']['sum'],'C_fixed_after':{k:oldfinal['c'][k]['after'] for k in ['choice','info','rubric']},'C_wording_after':{k:oldfinal['shifts']['c_wording'][k]['after'] for k in ['choice','info','rubric']},'training':training,'frozen_Auto_rule':settings['auto_rule'],'current_Auto_implementation_sha256':hf(R/'tiny_perceptron/lesson_products/inference.py'),'predictions_reference':predref,'scope':'Retain first-party fixed original measurements and frozen settings. Same finite task/data, not new B/C acceptance, notC-v2/C-v3. Read auto_solve source: two C judgments then frozen1/3/5-depth rule; no model call.'}
# Minimal CPU example retains its actual command and original stdout, not a new execution.
toy=j(B/'toy-dry-run-original-extract.json');fullpath=R/toy['original_receipt_path'];full=j(fullpath);assert hf(fullpath)==toy['original_receipt_sha256']
for k in ['command','cwd','started_at','completed_at','returncode','stdout','stderr']:assert full['commands'][2][k]==toy[k]
report=json.loads(toy['stdout'].splitlines()[-1]);hist=report['history'][0];assert report['mode']=='dry-run-no-weight-update' and report['device']=='cpu' and hist['loss']==5.770333290100098 and hist['grad_norm']==0.9595053195953369 and hist['effective_tokens']==140 and toy['returncode']==0 and full['toy_checkpoint_files']==[]
toyinput=Path(toy['cwd'])/'data/generated/toy-text/train.jsonl';assert toyinput.exists()
out['toy_retention']={'original_actor':full['reviewer_task'],'original_created_at':full['created_at'],'original_receipt':toy['original_receipt_path'],'original_receipt_sha256':hf(fullpath),'command_record':{k:toy[k] for k in ['command','cwd','started_at','completed_at','returncode','stdout','stderr']},'original_environment':full['environment'],'preserved_input_path':str(toyinput),'preserved_input_sha256_at_current_read':hf(toyinput),'preserved_input_bytes':toyinput.stat().st_size,'default_seed_from_source_identical_parser':42,'effective_targets':140,'original_checkpoint_files':full['toy_checkpoint_files'],'scope':'Original execution2026-10-30 by named actor. This round only parsed its immutable stdout/command and unchanged args.train-guarded source; current read of preserved input is explicitly not a contemporaneous historical hash claim. No model rerun.'}
# Basic platform CI is retained as original collection/excerpt with job-log SHA, not a pass verdict.
compat=j(R/'docs/validation-artifacts/release-compatibility-ci.json');mac=next(x for x in compat['jobs'] if x['job']=='test (macos-latest)');win=next(x for x in compat['jobs'] if 'windows-latest' in x['job'])
for p in ['scripts/check_env.py','tests/test_env.py']:assert hf(R/p)==compat['source_sha256'][p]
assert mac['environment_check']['compute']=='OK（mps）' and mac['environment_check']['device']=='Apple Silicon（MPS）' and mac['summary']=='330 passed in 35.01s'
out['platform_retention']={'original_CI_receipt_path':'docs/validation-artifacts/release-compatibility-ci.json','original_CI_receipt_sha256':hf(R/'docs/validation-artifacts/release-compatibility-ci.json'),'original_checked_at':compat['checked_at'],'revision':compat['revision'],'run_id':compat['run_id'],'mac_job_original_excerpt':mac,'windows_job_original_excerpt':win,'current_source_hashes_equal_original':['scripts/check_env.py','tests/test_env.py'],'MPS_scope':'Retained fixed actualmacOS CI basic256x256 forward/backward plus32x32 CPU comparison in unchanged test_env.py, not full course/model training. No new MPS hardware execution or log download.','Windows_scope':'Retained basic Windows CPU CI only. This round page/workflow parser13tests ran onLinux. No assertion of actual Windows PowerShell5.1/7 clone/install/kernel/Jupyter workflow replay or universal app long-path support.'}
gpu=j(B/'fixed-course-GPU-metadata.json');gpurows=[]
for x in gpu['results']:
 assert hf(R/x['path'])==x['sha256'];raw=j(R/x['path']);assert raw['device']==x['device'] and raw.get('gpu')==x['gpu'] and raw['torch_version']==x['torch_version'];gpurows.append({'path':x['path'],'sha256':x['sha256'],'device':raw['device'],'gpu':raw.get('gpu'),'torch':raw['torch_version']})
assert sum(x['device']=='cuda' and x['gpu']=='NVIDIA L4' and x['torch']=='2.14.1+cu126' for x in gpurows)==29
out['historical_GPU_retention']={'original_metadata_actor':gpu['reviewer_task'],'original_inspected_at':gpu['inspected_at'],'original_metadata_sha256':hf(B/'fixed-course-GPU-metadata.json'),'original_result_files':gpurows,'CUDA_L4_2_14_1_cu126_entries':29,'other_CPU_entries':1,'scope':'Original fixed result environment fields only; oneCPU experiment not relabelledGPU; no new Modal/L4/training.'}
req=(R/'requirements-natural.txt').read_text();student=(R/'docs/natural-assistant/v4/STUDENT.md').read_text();assert 'torch==2.8.0' in req and 'torchvision==0.23.0' in req and 'Python 3.12' in req and '.venv-natural' in student and '/whl/cu128' in student
out['natural_environment_contract']={'requirements_sha256':hf(R/'requirements-natural.txt'),'student_guide_sha256':hf(R/'docs/natural-assistant/v4/STUDENT.md'),'python':'3.12','torch':'2.8.0','torchvision':'0.23.0','GPU_index':'cu128','venv':'.venv-natural','scope':'Exact source-identical separate recipe/pins, not new installation, mature-model test or GPU-runtime result.'}
currenttoml=tomllib.loads((R/'zensical.toml').read_text());oldtoml=tomllib.loads((R/snapmap['zensical.toml']['snapshot']).read_text());assert currenttoml['project']['theme']==oldtoml['project']['theme'] and currenttoml['project']['extra_javascript']==oldtoml['project']['extra_javascript']
for p in ['scripts/export_course.py','course/web/course.js','course/web/course.css','course/web/mathjax-assets.json']:
 s=next(x for x in repo_sources if x['path']==p);assert s['current_sha256']==s['recorded_sha256']
assert 'prefers-reduced-motion' in (R/'course/web/course.css').read_text() and 'classList.toggle("is-enlarged")' in (R/'course/web/course.js').read_text()
out['website_contract']={'current_configuration_sha256':hf(R/'zensical.toml'),'prior_exact_configuration_snapshot':snapmap['zensical.toml']['snapshot'],'unchanged_theme_and_feature_configuration':True,'unchanged_javascript_configuration':True,'features':currenttoml['project']['theme']['features'],'unchanged_implementation_paths':['scripts/export_course.py','course/web/course.js','course/web/course.css','course/web/mathjax-assets.json'],'scope':'README describes repository build/functionality contract, not a claim that this revision had a new deployed-site/browser acceptance. Original reviewer2026-10-30 explicitly reserved desktop/mobile UI for separate UI round. No UI/browser or full-site replay performed; static retained source supports download/Colab footer, search/theme/code copy, SVG zoom/reduced motion and bundledMathJax contract.'}
# Original student inference has a recorded superset closure: inspect every hash, then bound two drifts using original pinned source.
studentp=R/'docs/course-experiments/student-checks/text_foundation.20261002T215208840323Z.json';student=j(studentp);closure=[]
for p,expectedsha in student['source_hashes'].items():
 actual=hf(R/p);row={'path':p,'original_sha256':expectedsha,'current_sha256':actual}
 if actual!=expectedsha:
  raw=subprocess.run(['git','show',student['code_checkout_revision']+':'+p],cwd=R,capture_output=True,check=True).stdout;assert h(raw)==expectedsha
  row['pinned_original_sha256_verified']=h(raw);row['pinned_revision']=student['code_checkout_revision'];row['raw_source_diff']=''.join(difflib.unified_diff(raw.decode().splitlines(keepends=True),(R/p).read_text().splitlines(keepends=True),fromfile='original@'+student['code_checkout_revision']+':'+p,tofile=p))
  row['nonimpact_reason']=('All6text_foundation weights haveformat_version1. validate_export added checks forcapstone/multimodal versions only; its format1load_checkpoint path remains unchanged. Export/manifest-hardening changes are not generation/runtime weight changes.' if p=='scripts/course_release.py' else 'Only tool conversation messages changed todeepcopy. _FinitePolicy import is onlyinside finite-policy-v1checkpoint branch; format1text_foundation and scripts/infer.py never use this branch.')
 closure.append(row)
assert {x['path'] for x in closure if x['current_sha256']!=x['original_sha256']}=={'scripts/course_release.py','scripts/course_experiments/applications.py'}
assert len(student['commands'])==7 and all(x['returncode']==0 for x in student['commands'])
assert all(x['format_version']==1 for x in student['weights'])
last=student['commands'][-1];assert json.loads(last['stdout'])['answer']=='side=right.' and '--prompt' in last['command'] and 'color=blue;shape=circle;' in last['command']
assert h(last['stdout'].encode())==last['stdout_sha256']
publicspec=next(x for x in j(R/'docs/course-experiments/public-models.json')['models'] if x['id']=='text_foundation')
assert {x['output']:x['sha256'] for x in publicspec['files']}=={x['output']:x['sha256'] for x in student['verified_files']}
out['text_foundation_retention']={'original_receipt_path':rel(studentp),'sha256':hf(studentp),'original_checked_at':student['checked_at'],'original_public_source':student['public_source'],'original_environment':student['environment'],'original_anonymous_download':student['anonymous_download'],'original_commands_returncodes':[x['returncode'] for x in student['commands']],'last_original_command':last,'original_source_closure':closure,'original_weight_format1_count':len(student['weights']),'all_fixed_verified_files_match_current_manifest':True,'scope':'Retain actual2026-10-02 anonymous CPU file/load/inference receipt and literalpromptanswer; no new download or neural model call.14recorded source hashes form a superset;2drifts explicitly examined at exact pinnedcommit and do not affectformat1path. This one answer is not accuracy/generalization evidence.'}
out['completed_at']=datetime.now(timezone.utc).isoformat()
p=A/'retention.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'path':rel(p),'sha256':hf(p),'bytes':p.stat().st_size,'old_actor':old['reviewer_task'],'repository_source_count':len(repo_sources),'changed_sources':[x['path'] for x in out['original_dependency_identity']['current_changed_sources']],'v2_splits':dict(splits),'omni_recount':out['omni_retention']['raw_stored_observation_recount'],'B':out['legacy_b_c_retention']['B_five_loop_boxes'],'text_original_at':student['checked_at']},ensure_ascii=False,indent=2))
