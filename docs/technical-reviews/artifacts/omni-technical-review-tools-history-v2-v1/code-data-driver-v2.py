"""Independent code/data callback. Writes only evidence and isolated test roots."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
WORK = Path('/workspace/work/technical-review-tools-history-v2-code-data-v2/run')
WORK.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).with_name('code-data-execution-v2.json')
TASK = '/root/omni_technical_coordinator/technical_review_tools_history_v2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def reference(path):
    return {'path':path.relative_to(ROOT).as_posix(), 'sha256':sha(path)}

scope_path = ROOT/'docs/course-revision-20261029-omni/reviews/history-v2-review-scope-v4.json'
assert sha(scope_path) == '7ed4a16291e8a32c55391b16dd97edd8ba2034c0ba58be40d0426c440f6b1028'
scope = json.loads(scope_path.read_bytes())
assert len(scope['checked_file_scope']) == 210
assert all(sha(ROOT/x['path']) == x['sha256'] for x in scope['checked_file_scope'])
assert sha(ROOT/'scripts/omni_revision/review_setup.py') == '701a87eb97fe51d2449b88602c4814236bb73297dce1494964e6102ff6684ad0'
records = []

def probe(name, expected, action, scope_note):
    start = datetime.now(UTC).isoformat()
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            result = action()
        observed, error = 'accept', None
    except Exception as exc:
        result, observed, error = None, 'reject', type(exc).__name__+': '+str(exc)
    row = {'name':name,'expected':expected,'observed':observed,'matched':expected==observed,
           'result':result,'error':error,'stdout':stream.getvalue(), 'started_at':start,
           'completed_at':datetime.now(UTC).isoformat(),'scope':scope_note}
    records.append(row)
    print(json.dumps({k:row[k] for k in ('name','expected','observed','error')},ensure_ascii=False),flush=True)

visuals = load('reviewer_current_visuals', ROOT/'scripts/build_visuals.py')

def copy_gallery(destination):
    shutil.copytree(ROOT/'course/figures', destination/'course/figures')
    manifest = Path('docs/lesson-products/results/publication-spec-v1.json')
    (destination/manifest).parent.mkdir(parents=True)
    shutil.copy2(ROOT/manifest, destination/manifest)

def isolated_build(name, old=False):
    destination = WORK/name
    copy_gallery(destination)
    source = ROOT/('docs/course-repair-20261008/reviews/freeze-01/freeze/implementation/scripts/build_visuals.py'
                   if old else 'scripts/build_visuals.py')
    module = load('gallery_'+name, source)
    module.ROOT, module.DIRECTORY, module.VISUALS = destination, destination/'course/figures', {}
    before = {p.name:sha(p) for p in module.DIRECTORY.glob('*.svg')}
    error = None
    try:
        module.build()
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    after = {p.name:sha(p) for p in module.DIRECTORY.glob('*.svg')}
    changes = [name for name in before if before[name] != after.get(name)]
    if old:
        assert error and '手寫圖解需要標題與描述' in error
        assert 'normalization.svg' in changes
    else:
        assert error is None and before == after and len(before) == 215
        assert sha(module.DIRECTORY/'index.json') == sha(ROOT/'course/figures/index.json')
    return {'source':reference(source),'svg_count':len(before),'error':error,'changed_figures':changes,
            'index_same':sha(module.DIRECTORY/'index.json')==sha(ROOT/'course/figures/index.json') if not old else None,
            'work_root':str(destination)}

probe('old_archived_visuals_reproduces_failure_and_norm_reversion','accept',
      lambda:isolated_build('old-visuals',True),'Actual archived source, current gallery copied outside Git; failure classification only.')
probe('current_visuals_215_bytes_and_index_preserved','accept',
      lambda:isolated_build('current-visuals'),'Full actual build in isolated copy; no repository figures changed, no visual semantic review claimed.')

def gallery_case(variant):
    destination = WORK/('gallery-'+variant)
    directory = destination/'course/figures'
    directory.mkdir(parents=True)
    raw = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>'
    name = 'looped-mechanism-measured.svg' if variant.startswith('result_') else 'manual.svg'
    (directory/name).write_bytes(raw)
    module = load('gallery_case_'+variant,ROOT/'scripts/build_visuals.py')
    module.ROOT, module.DIRECTORY, module.VISUALS = destination, directory, {}
    if variant.startswith('result_'):
        manifest = destination/module.RESULT_PLOTS[name]['manifest']
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'files':{name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}}}))
    elif variant != 'unknown_missing':
        meta = {'schema_version':1,'figures':{name:{'sha256':hashlib.sha256(raw).hexdigest(),
                'bytes':len(raw),'title':'independent title','description':'independent description'}}}
        if variant == 'bad_schema': meta['schema_version'] = True
        if variant == 'empty_title': meta['figures'][name]['title'] = ' '
        (directory/'source-metadata.json').write_text(json.dumps(meta))
    module.index_existing_visuals()
    if variant in ('result_changed_after_index','manual_changed_after_index'):
        (directory/name).write_bytes(raw.replace(b'L1 1',b'L2 2'))
        module.index_existing_visuals()
    return {'bytes_unchanged':(directory/name).read_bytes()==raw,'index':module.VISUALS}

for variant in ('result_valid','result_changed_after_index','manual_valid','manual_changed_after_index',
                'unknown_missing','bad_schema','empty_title'):
    probe('visual_'+variant,'accept' if variant in ('result_valid','manual_valid') else 'reject',
          lambda v=variant:gallery_case(v),'Independent synthetic metadata/provenance boundary.')

metadata = json.loads((ROOT/'course/figures/source-metadata.json').read_bytes())['figures']
alt_checks=[]
for name,item in metadata.items():
    figure=ROOT/'course/figures'/name
    body=(ROOT/item['description_source']).read_text()
    alts=[alt for alt,url in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)',body) if Path(url).name==name]
    assert sha(figure)==item['sha256'] and figure.stat().st_size==item['bytes']
    assert item['description'] in alts
    alt_checks.append({'file':name,'bytes_sha_match':True,'registered_description_matches_existing_alt':True})

def windows_case(variant):
    case=WORK/('windows-'+variant)
    for name in ('README.md','README_en.md','course/first-steps.md','course/training.md'):
        (case/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,case/name)
    workflow=(ROOT/'.github/workflows/windows-student-commands.yml').read_text()
    test_source=ROOT/'tests/test_windows_student_commands.py'
    if variant=='before':
        workflow=(ROOT/'docs/technical-reviews/artifacts/omni-author-integration-fix01/windows-student-commands.yml.before').read_text()
        test_source=ROOT/'docs/technical-reviews/artifacts/omni-author-integration-fix01/test_windows_student_commands.py.before'
    elif variant=='drop_maintainer_command':
        workflow=workflow.replace('          python docs/review-tools/check_review_round.py\n','')
    elif variant=='restore_circle_probe':
        workflow=workflow.replace('--prompt "color=blue;shape=square;pitch=low;color?" --tokens 24 --device cpu --json',
                                  '--prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json')
    (case/'.github/workflows').mkdir(parents=True)
    (case/'.github/workflows/windows-student-commands.yml').write_text(workflow)
    (case/'tests').mkdir();copied=case/'tests/test_windows_student_commands.py';shutil.copy2(test_source,copied)
    module=load('windows_'+variant,copied)
    errors=[]
    for page in ('README.md','README_en.md','course/training.md'):
        try:module.test_page_windows_commands_match_workflow(page)
        except AssertionError as exc:errors.append({'node':page,'error':str(exc)})
    try:module.test_page_quotes_appear_on_the_pages()
    except AssertionError as exc:errors.append({'node':'quotes','error':str(exc)})
    assert bool(errors)==(variant!='current')
    if variant=='before':assert len(errors)==4
    return {'actual_static_failures':errors,'main_repo_unmodified':True}

for variant in ('before','current','drop_maintainer_command','restore_circle_probe'):
    probe('windows_'+variant,'accept',lambda v=variant:windows_case(v),
          'Static Linux tests on isolated exact source copies; expected rejected commands retained as results; no Windows workflow execution.')

before_test=WORK/'test_before_reading_time.py'
shutil.copy2(ROOT/'docs/technical-reviews/artifacts/omni-author-integration-fix01/test_reading_time.py.before',before_test)
process=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                       str(before_test)+'::test_export_uses_estimates_for_each_page_and_stops_before_writing_stale_data'],
                      cwd=ROOT,capture_output=True,text=True)
assert process.returncode==1 and '正式Colab入口需要--revision指定完整40位Git commit' in process.stdout
old_fixture={'source_sha256':sha(before_test),'command':[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(before_test)+'::test_export_uses_estimates_for_each_page_and_stops_before_writing_stale_data'],
             'exit_code':process.returncode,'failure_present':True,'output_outside_git':str(WORK/'old-fixture.log')}
(WORK/'old-fixture.log').write_text(process.stdout+process.stderr)
old_fixture['output_sha256']=sha(WORK/'old-fixture.log')

baseline=Path('/workspace/work/reference/omni-technical-validation-v1/baseline-head')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()=='80c5db17e8b077b2c0a5e4e9baa48e760ccb0532'
nodes=['tests/test_reading_time.py::test_export_uses_estimates_for_each_page_and_stops_before_writing_stale_data',
       'tests/test_windows_student_commands.py::test_page_windows_commands_match_workflow[README.md]',
       'tests/test_windows_student_commands.py::test_page_windows_commands_match_workflow[README_en.md]',
       'tests/test_windows_student_commands.py::test_page_windows_commands_match_workflow[course/training.md]',
       'tests/test_windows_student_commands.py::test_page_quotes_appear_on_the_pages']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
bp=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*nodes],cwd=baseline,env=env,capture_output=True,text=True)
assert bp.returncode==0 and '5 passed' in bp.stdout
baseline_five={'command':[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*nodes],
               'cwd':str(baseline),'exit_code':bp.returncode,'stdout':bp.stdout,'stderr':bp.stderr}

api=load('callback_native',ROOT/'docs/review-tools/incremental_revision.py')
setup=load('callback_setup',ROOT/'scripts/omni_revision/review_setup.py')
replay=load('callback_replay',ROOT/'docs/review-tools/baseline_replay.py')
supp_path=ROOT/'docs/course-revision-20261029-omni/baseline-supplements-v2.json'
supp=json.loads(supp_path.read_bytes())['files']
previous=json.loads((ROOT/'docs/course-revision-20261029-omni/baseline-supplements-v1.json').read_bytes())['files']
assert supp[:5]==previous and len(supp)==6
tables=replay.preserved_identities(ROOT)
provenance=[]
for item in supp:
    values={key:table[item['live_path']] for key,table in tables.items() if item['live_path'] in table}
    assert values and sha(ROOT/item['path'])==item['sha256'] and all(value==item['sha256'] for value in values.values())
    provenance.append({**item,'authenticated_old_tables':values})
baseline_results=[]
for label,entries in (('five',supp[:5]),('six',supp)):
    value=replay.replay(ROOT,entries)
    assert (value['status']=='passed_metadata_checks')==(label=='six')
    if label=='five':assert any('scripts/build_visuals.py' in x for x in value['failures'])
    baseline_results.append({'supplement_count':len(entries),'status':value['status'],'failures':value['failures'],'record_count':len(value.get('records',[]))})

old_driver=json.loads(Path(__file__).with_name('execution-v1.json').read_bytes())['mutation_reproduction']['script']
ns={};exec(old_driver.split('probes = [',1)[0],ns);ns['WORK']=WORK
for name,method in [('old_continuity_time','chronology'),('old_technical_time','original_technical_time'),
                    ('old_technical_shell','original_technical_shell'),('old_continuity_source','original_continuity_source')]:
    def run_case(n=name,m=method):
        target=WORK/n;target.mkdir();built=ns['fixture'].fixture(target);ns[m](target,built);ns['sync_seals'](target,built)
        return ns['fixture'].check(target,built)
    probe(name,'reject',run_case,'Exact own original mutation functions, current actual historical/helper code; synthetic receipts only.')

assert all(sha(ROOT/x['path'])==x['sha256'] for x in scope['checked_file_scope'])
result={'schema_version':1,'kind':'bounded_code_callback_execution','reviewer_task':TASK,
        'created_at':datetime.now(UTC).isoformat(),'command':'uv run --extra cpu python '+Path(__file__).relative_to(ROOT).as_posix(),
        'environment':{'python':sys.version,'platform':sys.platform,'device':'CPU/offline; no training/GPU/Windows runtime'},
        'scope':{'reference':reference(scope_path),'count':210,'before_after_matches':210},
        'targeted_tests':json.loads((WORK.parent/'targeted-tests-v2.json').read_bytes()),
        'probes':records,'existing_metadata_alt_checks':alt_checks,'old_reading_fixture':old_fixture,
        'same_five_nodes_at_committed_baseline':baseline_five,'baseline_supplement_provenance':provenance,
        'actual_baseline_replays':baseline_results,'no_repository_source_or_figure_modified':True}
with OUT.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':reference(OUT),'unexpected':[x['name'] for x in records if not x['matched']]},ensure_ascii=False),flush=True)
