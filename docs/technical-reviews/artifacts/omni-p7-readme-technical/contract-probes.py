"""Small offline contract probes for the canonical README, no NN imports."""
from pathlib import Path
import ast,datetime,hashlib,json,sys,tomllib
ROOT=Path('/workspace/work/omni-book-integrated-v1');BASE=ROOT/'docs/technical-reviews/artifacts/omni-p7-readme-technical'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ensure(x,why):
 if not x:raise AssertionError(why)
def text(n):return (ROOT/n).read_text()
r={'reviewer_task':'/root/omni_p7_technical_readme','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'scope':'Static AST/TOML/JSON contracts and arithmetic only. This is not an end-to-end install, CUDA/MPS execution, Colab execution, or a maintenance/publish run.'}
p=tomllib.loads(text('pyproject.toml'));u=p['tool']['uv'];ensure(set(p['project']['optional-dependencies'])>={'cpu','cu126','cu130','selftrained'},'extras');ensure('notebook' in p['dependency-groups'] and p['dependency-groups']['site']==['zensical==0.0.67'],'groups')
indices={x['name']:x['url'] for x in u['index']};ensure(indices=={'pytorch-cpu':'https://download.pytorch.org/whl/cpu','pytorch-cu126':'https://download.pytorch.org/whl/cu126','pytorch-cu130':'https://download.pytorch.org/whl/cu130'},'CPU/CUDA index URLs')
lock=tomllib.loads(text('uv.lock'));torch=[x for x in lock['package'] if x['name']=='torch'];ensure(all(x['version'].startswith('2.14.1') for x in torch),'locked torch version')
r['installation']={'requires_python':p['project']['requires-python'],'optional_dependencies':p['project']['optional-dependencies'],'dependency_groups':p['dependency-groups'],'torch_indices':indices,'lock_torch':[{k:x[k] for k in ['version','source'] if k in x} for x in torch],'extra_conflicts':u['conflicts'],'python_version_file':text('.python-version').strip() if (ROOT/'.python-version').exists() else 'not present'}
s=text('scripts/train.py');t=ast.parse(s);parser=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='parser');args={}
for n in ast.walk(parser):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant):
  args[n.args[0].value]={k.arg:ast.literal_eval(k.value) for k in n.keywords if k.arg in ['action','default','choices'] and isinstance(k.value,(ast.Constant,ast.List,ast.Tuple))}
ensure(args['--train']['action']=='store_true' and args['--experts']['default']==0 and args['--seed']['default']==42,'dryrun defaults');r['train_arguments']=args
source=text('scripts/prepare_data.py');tree=ast.parse(source);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_records');namespace={};exec(compile(ast.Module(body=[func],type_ignores=[]),'prepare_data.generate_records AST extraction','exec'),namespace);records=namespace['generate_records']('toy-text');ensure(len(records)==12 and len({x['family'] for x in records})==12,'toy generator')
manifest=json.loads((BASE/'toy-data-original/manifest.json').read_text());r['toy_data_contract']={'generated_records':len(records),'families':len({x['family'] for x in records}),'generator_function_only_execution':True,'fixed_original_splits':manifest['splits']}
for key,spec in manifest['splits'].items():ensure(sha(BASE/f'toy-data-original/{key}.jsonl')==spec['sha256'],'original toy split '+key)
# CLI script parsers and source-based guards are read without calling heavy branches.
commands={'scripts/build_course.py':['--check'],'scripts/check_notebooks.py':['--mode','--lesson'],'scripts/format_course_code.py':[],'scripts/build_visuals.py':[],'scripts/check_course_reviews.py':[],'scripts/check_technical_reviews.py':[],'docs/review-tools/check_review_round.py':[],'scripts/fetch_course_models.py':['--list'],'scripts/check_course_models.py':['--model'],'scripts/fetch_training_assets.py':['--list','--asset']}
for path,flags in commands.items():
 body=text(path);ast.parse(body);ensure(all(flag in body for flag in flags),'CLI flags '+path)
r['maintenance_parsers']={path:{'sha256':sha(ROOT/path),'flags_present':flags,'AST_parsed':True} for path,flags in commands.items()}
r['directories']={name:(ROOT/name).is_dir() for name in ['course/chapters','course/figures','notebooks','tiny_perceptron','scripts','tests','docs']};ensure(all(r['directories'].values()),'project directories')
site=tomllib.loads(text('zensical.toml'));features=site['project']['theme']['features'];ensure({'search.highlight','content.code.copy','navigation.tabs','navigation.indexes','toc.follow'}<=set(features),'site features');ensure(len(site['project']['theme']['palette'])==2,'dark/light palette')
js=text('course/web/course.js');css=text('course/web/course.css');export=text('scripts/export_course.py');ensure('.diagram-zoom' in js and 'prefers-reduced-motion: reduce' in css,'zoom/motion');ensure('MathJax' in text('course/web/mathjax.js') and 'tex-chtml.js' in text('zensical.toml'),'local MathJax');ensure('下載本節 .ipynb' in export and 'colab.research.google.com' in export and 'output-label' in export,'same-source downloads and CPU output')
r['site_contract']={'features':features,'extra_javascript':site['project']['extra_javascript'],'local_mathjax_manifest_sha256':sha(ROOT/'course/web/mathjax-assets.json'),'zoom_and_reduced_motion_source_checked':True,'footer_and_output_export_source_checked':True,'UI_scope':'The single README SVG was rendered; website desktop/mobile UI remains for the separate UI round.'}
pages=text('.github/workflows/pages.yml');required=['scripts/check_course_reviews.py','scripts/check_technical_reviews.py','docs/review-tools/check_review_round.py'];ensure(all(x in pages for x in required),'publish review gates');ensure('scripts/check_notebooks.py --mode kernel --workers 1' in pages,'separate whole Notebook validation');ci=text('.github/workflows/ci.yml');ensure(all(x in ci for x in ['ubuntu-latest','windows-latest','macos-latest']),'core CI OSes')
notebook=text('scripts/check_notebooks.py');ensure('namespace = {"__name__": "__main__"}' in notebook and 'NotebookClient(' in notebook and 'client.execute()' in notebook,'namespace/kernel distinction')
r['review_and_CI_contract']={'review_gate_names':required,'gate_execution_this_review':False,'CPU_whole_Notebook_job_present':True,'core_OSes':['Linux','macOS','Windows'],'python_mode':'one fresh namespace per file','kernel_mode':'new NotebookClient/kernel per file'}
compat=json.loads(text('docs/validation-artifacts/release-compatibility-ci.json'));ensure(compat['status']=='success','saved CI success')
for path in ['scripts/check_env.py','tests/test_env.py']:ensure(sha(ROOT/path)==compat['source_sha256'][path],'saved CI source matches '+path)
mac=next(x for x in compat['jobs'] if 'macos' in x['job']);ensure(mac['environment_check']['compute']=='OK（mps）','saved MPS matrix result');r['MPS_original_CI']={'run_id':compat['run_id'],'revision':compat['revision'],'mac_job':mac,'source_hashes_match':['scripts/check_env.py','tests/test_env.py'],'limits':'256x256 forward/backward in check_env and 32x32 comparison in test_env, not full model training/course validation on MPS.'}
old=json.loads(text('docs/course-experiments/results/text_foundation.json'));ensure(old['device']=='cuda' and old['gpu']=='NVIDIA L4' and old['torch_version']=='2.14.1+cu126','fixed course GPU result');r['fixed_course_GPU_environment']={k:old[k] for k in ['device','gpu','torch_version']}
natural=text('requirements-natural.txt');ensure('torch==2.8.0' in natural and 'Python 3.12' in natural and 'cu128' in natural,'separate mature environment');r['natural_environment']={'python':'3.12','torch':'2.8.0','CUDA_index':'cu128','scope':'Separate deployment recipe; no mature-model execution in this review.'}
smoke=text('.github/workflows/gpu-smoke.yml');ensure('name: GPU training smoke test' in smoke and 'workflow_dispatch:' in smoke and 'push:' not in smoke,'manual smoke');r['manual_GPU_workflow']={'name':'GPU training smoke test','manual_only':True,'invoked':False}
export_model=text('scripts/selftrained/train.py');start=export_model.index('def export_inference');end=export_model.index('\ndef ',start+1);body=export_model[start:end];ensure('checkpoint["model"].items()' in body and 'save_file(' in body,'weights-only export');r['safe_export_contract']={'source':'scripts/selftrained/train.py','function':'export_inference','selected_fields':'checkpoint model tensors + config/tokenizer + inference manifest','optimizer_RNG_in_export':False,'scope':'Full continuation requires the complete checkpoint, not the public safetensors.'}
readme_commands=['python scripts/prepare_data.py --kind toy-text','python scripts/train.py --task text --data data/generated/toy-text/train.jsonl'];ensure(all(c in text('README.md') for c in readme_commands),'reviewed minimal exact commands')
modelcheck=text('scripts/check_course_models.py');ensure('report["status"] = "passed" if all(' in modelcheck and 'if failed:' in modelcheck and 'raise SystemExit(1)' in modelcheck,'check_model exit contract');r['model_check_contract']={'scope':'Offline source/AST inspection only; existing raw check output supplies actual inference evidence. No new inference.'}
lic=text('LICENSE');ensure('MIT License' in lic and 'Permission is hereby granted' in lic,'MIT grant');r['license']={'path':'LICENSE','sha256':sha(ROOT/'LICENSE'),'external_data_scope':'Separate upstream licenses; no blanket MIT claim for data.'}
acquisition=json.loads((BASE/'omni-training-acquisition.json').read_text());selected=json.loads(text('docs/omni-products/v1/model-selection.json'));jobs=[]
for job,spec in selected['all_formal_jobs'].items():
 folder=BASE/'omni-training-primary/core'/job
 origin=json.loads((folder/'job-origin.json').read_text());job_result=json.loads((folder/'job-result.json').read_text());training=json.loads((folder/'training/training-result.json').read_text())
 ensure(job_result['success'] and training['complete'] and training['optimizer_updates_observed'],'training complete '+job)
 ensure(training['initial_parameter_sha256']!=training['latest_parameter_sha256'],'training changed '+job)
 ensure(job_result['runtime']['device']=='cpu' and job_result['runtime']['torch']=='2.14.1+cpu' and job_result['execution_host']['github_actions'],'actual Linux CPU runtime '+job)
 ensure(not job_result['final_opened'] and not training['final_opened'],'training preceded final '+job)
 ensure(sha(folder/'job-result.json')==spec['job_result_sha256'],'selected original job '+job)
 ensure(sha(folder/'training/training-result.json')==job_result['training_result_sha256'],'original training SHA '+job)
 ensure(sha(folder/'training/training-log.json')==training['files']['training-log.json']['sha256'],'original log SHA '+job)
 jobs.append({'job':job,'updates':training['updates'],'selected_update':training['best']['update'],'runtime':job_result['runtime'],'optimizer_updates_observed':training['optimizer_updates_observed'],'weights_changed':True})
ensure(len(jobs)==10,'ten formal jobs');r['ten_formal_CPU_jobs']=jobs
r['result']='PASS: offline installation, CLI, website-source, CI receipt, formal training receipt and export contracts matched the README scope.'
(BASE/'contract-probes.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(r['result'])
