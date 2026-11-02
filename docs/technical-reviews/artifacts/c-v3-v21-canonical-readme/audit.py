from pathlib import Path
import collections
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import zipfile
import torch

ROOT = Path('/workspace/work/omni-book-integrated-v1')
ZIP = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
TMP = Path('/tmp/c-v3-v21-canonical-readme')
TMP.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(TMP/'mplconfig')
os.chdir(ROOT)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

result = {'actor': '/root/c_v3_technical_coord/canonical_readme', 'created_at': datetime.now(timezone.utc).isoformat(), 'scope': 'Independent source/entry/retained-raw-observation audit; no installation, model training, final opening, paid compute, anonymous download or whole-course execution.', 'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git': torch.version.git_version, 'platform': platform.platform(), 'device': 'cpu'}}
freeze = load('docs/course-revision-20261101-c-v3/source-freeze-v21.json')
oldfreeze = load('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json')
entry = next(x for x in freeze['pages'] if x['page_id'] == 'readme')
oldentry = next(x for x in oldfreeze['pages'] if x['page_id'] == 'readme')
assert sha((ROOT/'README.md').read_bytes()) == entry['source_sha256']
assert (ROOT/'README.md').read_bytes() == (ROOT/entry['snapshot']).read_bytes()
assert sha((ROOT/oldentry['snapshot']).read_bytes()) == oldentry['source_sha256']
result['identity'] = {'source21_freeze_sha256':sha((ROOT/'docs/course-revision-20261101-c-v3/source-freeze-v21.json').read_bytes()), 'readme':entry, 'source18_readme':oldentry, 'source18_freeze_sha256':sha((ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json').read_bytes())}
chapter_counts={str(p.relative_to(ROOT)):len(re.findall(r'^##\s+([0-9ABC]+\.\d+)(?:\s|$)',p.read_text(encoding='utf-8'),re.M)) for p in sorted((ROOT/'course/chapters').glob('*.md'))}
extras={n:len(re.findall(r'^##\s+([A-Z]+\.\d+)(?:\s|$)',(ROOT/n).read_text(encoding='utf-8'),re.M)) for n in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
nb=list((ROOT/'notebooks').rglob('*.ipynb'))
assert sum(chapter_counts.values()) == 332 and sum(extras.values()) == 26 and len(nb)==332
result['counts']={'chapter_count':20,'extension_files':3,'chapter_lesson_total':sum(chapter_counts.values()),'numbered_guide_total':sum(extras.values()),'all_numbered_total':sum(chapter_counts.values())+sum(extras.values()),'notebooks':len(nb),'chapter_details':chapter_counts,'guide_details':extras}
links=[]
for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text(encoding='utf-8')):
    if '://' in target or target.startswith('#'):
        continue
    path,_,anchor=target.partition('#')
    exists=(ROOT/path).is_file()
    section_exists=True
    if anchor and re.fullmatch(r'[0-9ABCWTGR]+\.\d+',anchor):
        section_exists=bool(re.search(r'^##\s+'+re.escape(anchor)+r'(?:\s|$)',(ROOT/path).read_text(encoding='utf-8'),re.M))
    links.append({'target':target,'file_exists':exists,'numbered_section_exists':section_exists})
assert all(x['file_exists'] and x['numbered_section_exists'] for x in links)
result['local_links']={'checked':len(links),'all_resolve':True,'links':links}
kr=load('docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json')
kb=ROOT/'outputs/omni-c-v3-kernel-bundle-v21'; executions=[]; errors=[]
for p in nb:
    q=kb/p.relative_to(ROOT/'notebooks')
    src=json.loads(p.read_text(encoding='utf-8')); done=json.loads(q.read_text(encoding='utf-8'))
    source_cells=[(x['cell_type'],''.join(x.get('source',[]))) for x in src['cells']]
    executed_cells=[(x['cell_type'],''.join(x.get('source',[]))) for x in done['cells']]
    code=[x for x in done['cells'] if x['cell_type']=='code']
    bad=[x for x in code if x.get('execution_count') is None or any(y.get('output_type')=='error' for y in x.get('outputs',[]))]
    if source_cells != executed_cells or bad:errors.append(str(p.relative_to(ROOT)))
    executions.append({'lesson':p.stem,'executed_sha256':sha(q.read_bytes()),'code_cells':len(code)})
kernel_receipt_mismatches=[]
for x in kr['actual_executions']:
    source=next(p for p in nb if p.stem==x['lesson']); done=kb/source.relative_to(ROOT/'notebooks')
    assert sha(source.read_bytes()) == x['source_sha256']
    if sha(done.read_bytes()) != x['executed_sha256']:
        kernel_receipt_mismatches.append({'lesson':x['lesson'],'receipt_executed_sha256':x['executed_sha256'],'actual_executed_sha256':sha(done.read_bytes()),'actual_executed_path':str(done.relative_to(ROOT))})
result['retained_kernel_identity']={'receipt_path':'docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json','receipt_sha256':sha((ROOT/'docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json').read_bytes()),'source_cells_and_executed_cells_equal':len(executions)-len(errors),'notebooks_with_source_mismatch_or_missing_execution_count_or_error':errors,'source21_actual_refresh_source_notebook_hashes_verified':[x['lesson'] for x in kr['actual_executions']],'executed_hash_mismatches':kernel_receipt_mismatches,'retained_current_count':kr['retained_current_count'],'inspected_combined_bundle':str(kb.relative_to(ROOT)),'receipt_retained_bundle_scope':str(kr['retained_bundle'])+' is the retained328 subset; new4 execution hashes matched in the combined source21 bundle explicitly located by coordinator.','scope':'Raw saved executions retained and code/markdown identity inspected. No new kernel execution; this does not establish the scientific accuracy of every lesson.'}
release=load('docs/decision-product/v3/public-release.json')
assert ZIP.stat().st_size==release['bytes'] and sha(ZIP.read_bytes())==release['sha256']
with zipfile.ZipFile(ZIP) as z:
    pre='tiny-perceptron-c-v3/'
    manifest=json.loads(z.read(pre+'MANIFEST.json'))
    inspected={}
    def raw(name):
        data=z.read(pre+name);digest=sha(data)
        assert manifest['files'][name]['sha256']==digest if isinstance(manifest['files'][name],dict) else manifest['files'][name]==digest
        inspected[name]={'bytes':len(data),'sha256':digest}
        return data
    def j(name):
        return json.loads(raw(name))
    acquisition=load('docs/decision-product/v3/metadata/acquisition.json')
    meta=[]
    for item in acquisition['files']:
        data=raw(item['archive_member'][len(pre):]); local=(ROOT/item['path']).read_bytes()
        assert data==local and len(data)==item['bytes'] and sha(data)==item['sha256']
        meta.append({'path':item['path'],'sha256':item['sha256'],'byte_identical':True})
    for name,digest in release['bundle'].items():assert sha(raw('bundle/'+name))==digest
    for path,digest in release['implementation'].items():
        assert sha((ROOT/path).read_bytes())==digest and sha(raw(path))==digest
    obs=j('evidence/final-observations.json'); examples=j('evidence/final-examples.json'); families=j('evidence/final-families.json')
    final=j('evidence/final-results.json'); opening=j('evidence/final-opening.json'); selection=j('evidence/selection.json'); policy=j('bundle/policy.json'); registry=j('evidence/registry-before-final.json')
    assert len(obs)==len(examples)==4500 and len(families)==1500
    family_keys={f['key'] for f in families}
    assert len(family_keys)==1500
    partitions=collections.Counter()
    for f in families:
        key=sha(json.dumps(sorted(tuple(x) for x in f['facts']),separators=(',',':')).encode())
        assert key==f['key']
        partition=int(sha(('c-product-v3-visible-facts-1:'+key).encode()),16)%1000
        partitions[str(960<=partition<1000)]+=1
    kind_families={k:{x['family'] for x in obs if x['kind']==k} for k in ['choice','info','rubric']}
    assert all(v==family_keys for v in kind_families.values())
    row_counts=collections.Counter((x['family'],x['kind']) for x in obs)
    assert all(v==1 for v in row_counts.values())
    exposure={key for x in registry['entries'] for key in x['family_keys']}
    calibration=set(selection['families']['calibration']); policy_keys=set(selection['families']['policy'])
    assert not (family_keys & exposure or family_keys & calibration or family_keys & policy_keys)
    correct=collections.Counter(); total=collections.Counter(); accepted=errors=none=answerable=0
    for o,e in zip(obs,examples):
        assert (o['family'],o['kind'],o['gold_id'])==(e['family'],e['kind'],e['gold'])
        assert o['ids'][o['label']]==e['gold']
        logits=[-math.inf if x is None else x for x in o['logits']]; i=max(range(len(logits)),key=lambda n:logits[n]); predicted=o['ids'][i]
        correct[o['kind']]+=predicted==e['gold'];total[o['kind']]+=1
        if o['kind']=='choice':
            answerable+=e['gold']!='none';none+=e['gold']=='none'
            temperature=policy['temperatures']['choice'];m=max(logits);exp=[math.exp((a-m)/temperature) for a in logits];confidence=exp[i]/sum(exp)
            eligible=o['eligible'][i]
            accept=predicted!='none' and confidence>=policy['threshold'] and eligible
            accepted+=accept;errors+=accept and predicted!=e['gold']
    assert dict(correct)=={'choice':1500,'info':1500,'rubric':1500} and accepted==1025 and errors==0 and none==475 and answerable==1025
    manual=j('evidence/product-checks.json');legacy=j('evidence/legacy-regression.json')
    m3=collections.Counter(); m3correct=collections.Counter()
    for x in manual['observations']:
        if x.get('devices')==3 and x['kind'] in {'choice','info','rubric'}:
            m3[x['kind']]+=1;m3correct[x['kind']]+=x['prediction']['model_choice']==x['expected']
    lg=collections.Counter();lgcorrect=collections.Counter()
    for x in legacy['observations']:
        lg[x['kind']]+=1;lgcorrect[x['kind']]+=x['prediction']['model_choice']==x['gold']
    lp=j('evidence/lineage-parent.json'); plan=j('evidence/plan.json');completion=j('evidence/completion.json');sf=j('evidence/source-freeze.json');lineage=j('evidence/lineage.json')
    training=[json.loads(x) for x in raw('evidence/training.jsonl').splitlines() if x]
    assert completion['completed_updates']==1500 and lp['receipts']['completion.json']['completed_updates']==4000
    assert lp['receipts']['completion.json']['artifacts']['best.safetensors']==lineage['parent_weights_sha256']
    assert sha(raw('evidence/initial.safetensors'))==lineage['parent_weights_sha256']
    assert completion['artifacts']['best.safetensors']==release['bundle']['weights.safetensors']
    result['c_v3_fixed_raw_audit']={'archive_name':ZIP.name,'local_archive_bytes':ZIP.stat().st_size,'local_archive_sha256':release['sha256'],'public_transport_tested':False,'metadata_byte_identity':meta,'implementation_same_as_archive_count':len(release['implementation']),'inspected_member_fingerprints':inspected,'recounted_from_raw_logits':{'per_kind':{k:{'correct':correct[k],'families':total[k]} for k in total},'accepted':accepted,'accepted_errors':errors,'answerable_actions':answerable,'correct_none':none,'threshold':policy['threshold'],'choice_temperature':policy['temperatures']['choice']},'family_identity':{'unique_visible_fact_families':len(family_keys),'devices_distribution':dict(collections.Counter(len(f['facts']) for f in families)),'all_three_kind_sets_identical':True,'duplicate_family_kind_pairs':0,'partition_final_checks':dict(partitions),'excluded_registry_overlap':len(family_keys&exposure),'calibration_overlap':len(family_keys&calibration),'policy_overlap':len(family_keys&policy_keys),'training_dev_separation_basis':'Read current data.py partition: train[0,800),dev[800,880),calibration[880,920),policy[920,960),final[960,1000). Recomputed final membership for every canonical visible-fact key; generation and retained hashes do not include cross-split overlaps.'},'retained_three_device_engineering':{'manual_per_kind':dict(m3),'manual_model_choice_correct':dict(m3correct),'legacy_per_kind':dict(lg),'legacy_model_choice_correct':dict(lgcorrect),'legacy_distinct_original_families':len({x['original_family'] for x in legacy['observations']})},'training_lineage':{'parent_updates':4000,'selected_updates_after_internal_initialization':1500,'training_log_events':len(training),'last_recorded_step':training[-1]['step'],'parent_plan_initialization':lp['receipts']['plan.json']['training'],'selected_initial_weights_equal_parent_best':True,'selected_final_weights_equal_public_payload':True,'completion_runtime':completion['runtime'],'plan_created_at':plan['created_at'],'completion_created_at':completion['created_at'],'selection_created_at':selection['created_at'],'opening_created_at':opening['created_at'],'results_created_at':final['created_at']},'limitations':'Recount and fingerprint verification of pre-existing original observations, not new inference/evaluation; three kinds share1500 families;3-device regression is exposed engineering evidence; no anonymous public download was attempted.'}
v2=load('docs/selftrained/v2-manifest.json')
pm=load('docs/course-experiments/public-models.json');cp=load('docs/course-experiments/capstone-public.json')
result['retained_manifest_counts']={'v2_manifest_sha256':sha((ROOT/'docs/selftrained/v2-manifest.json').read_bytes()),'v2_package_bytes':v2['package']['bytes'],'v2_records_file_entries':len(v2['records']),'v2_asset_file_entries':len(v2['assets']),'v2_total_file_entries':len(v2['records'])+len(v2['assets']),'course_public_manifest_sha256':sha((ROOT/'docs/course-experiments/public-models.json').read_bytes()),'public_model_groups':len(pm['models']),'public_model_file_entries':sum(len(x['files']) for x in pm['models']),'capstone_manifest_sha256':sha((ROOT/'docs/course-experiments/capstone-public.json').read_bytes()),'capstone_model_groups':len(cp['models']),'scope':'Read original manifest bytes and counted entries; no remote download, data extraction, model loading or old product acceptance repeated.'}
result['executions']=[]
for args in [[sys.executable,'scripts/build_course.py','--check'],[sys.executable,'scripts/fetch_training_assets.py','--list'],[sys.executable,'scripts/fetch_course_models.py','--list'],[sys.executable,'scripts/check_env.py'],[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--basetemp','/tmp/c-v3-v21-canonical-readme-pytest','tests/test_course_product_entry.py','tests/test_windows_student_commands.py::test_page_windows_commands_match_workflow','tests/test_windows_student_commands.py::test_windows_clone_comes_after_global_long_paths'],[sys.executable,'-m','scripts.course_product','judge','--help']]:
    p=subprocess.run(args,cwd=ROOT,text=True,encoding='utf-8',capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=55)
    result['executions'].append({'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    assert p.returncode==0,(args,p.returncode,p.stdout,p.stderr)
result['platform_scope']='The Linux pytest command checks page/workflow extraction and installer fixture contracts; it does not execute PowerShell, Windows clone/install/Jupyter, Apple MPS, GPU training or Colab. check_env only executes a256x256 CPU matrix forward/backward.'
from scripts.check_notebooks import run_python
ns_a=TMP/'namespace-a.ipynb';ns_b=TMP/'namespace-b.ipynb'
ns_a.write_text(json.dumps({'cells':[{'cell_type':'code','source':['sentinel=17\nprint(sentinel)']}]}),encoding='utf-8')
ns_b.write_text(json.dumps({'cells':[{'cell_type':'code','source':['print(\"sentinel\" in globals())']}]}),encoding='utf-8')
na=run_python(ns_a);nbout=run_python(ns_b)
assert na=='17\n' and nbout=='False\n'
result['python_mode_namespace_probe']={'notebook_a_code':'sentinel=17;print(sentinel)','notebook_a_output':na,'notebook_b_code':'print(\"sentinel\" in globals())','notebook_b_output':nbout,'scope':'Two tiny temporary Notebook dictionaries passed to actual run_python; no whole-course run or kernel launched. run_kernel source inspected: one new NotebookClient per call, executes then writes output notebook.'}
result['torch_original_wheel_sources']=[]
for obj in [torch.nn.Embedding,torch.nn.CrossEntropyLoss,torch.autograd.backward,torch.nn.utils.clip_grad_norm_]:
    source=Path(inspect.getsourcefile(obj));line=inspect.getsourcelines(obj)[1]
    rel=source.relative_to(Path(torch.__file__).parent.parent)
    excerpt=inspect.getdoc(obj)[:2200]
    result['torch_original_wheel_sources'].append({'object':obj.__module__+'.'+obj.__name__,'wheel_version':str(torch.__version__),'torch_git':torch.version.git_version,'module_relative_path':str(rel),'full_module_sha256':sha(source.read_bytes()),'object_first_line':line,'url':f'https://github.com/pytorch/pytorch/blob/{torch.version.git_version}/{rel}#L{line}','inspection':'Read the original installed PyTorch wheel source/docstring; this URL is a commit locator and was not fetched remotely.','excerpt':excerpt})
result['figure_inspection']={'svg':'course/figures/rewrite-01-character-ids.svg','svg_sha256':sha((ROOT/'course/figures/rewrite-01-character-ids.svg').read_bytes()),'render_command':'inkscape course/figures/rewrite-01-character-ids.svg --export-type=png --export-filename=/tmp/c-v3-v21-canonical-readme-character-ids.png --export-width=640','render_sha256':sha(Path('/tmp/c-v3-v21-canonical-readme-character-ids.png').read_bytes()),'actual_visual_inspection':'Viewed the640x710PNG using view_image. Top row 貓 看 狗 ， maps3 2 1 4; bottom row 狗 看 貓 。 maps1 2 3 0. Each vertical arrow is bidirectional; repeated貓3/狗1 and bottom note say IDs are lookup codes, not semantic magnitude. No clipping/overlap observed.','scope':'Original full figure rendered; no image editing.'}
result['completed_at']=datetime.now(timezone.utc).isoformat()
(TMP/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'audit_path':str(TMP/'audit.json'),'bytes':(TMP/'audit.json').stat().st_size,'sha256':sha((TMP/'audit.json').read_bytes()),'counts':result['counts']['all_numbered_total'],'raw_metrics':result['c_v3_fixed_raw_audit']['recounted_from_raw_logits'],'execution_results':[{'command':x['command'],'returncode':x['returncode'],'stdout':x['stdout'][:200]} for x in result['executions']]},ensure_ascii=False,indent=2))
