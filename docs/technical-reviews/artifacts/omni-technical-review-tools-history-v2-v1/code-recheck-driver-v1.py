"""Bounded independent recheck; all mutations stay in synthetic roots outside Git."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
ARTIFACTS = Path(__file__).resolve().parent
WORK = Path('/workspace/work/technical-review-tools-history-v2-code-recheck-v1')
WORK.mkdir(parents=True, exist_ok=True)
TASK = '/root/omni_technical_coordinator/technical_review_tools_history_v2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

SCOPE = ROOT/'docs/course-revision-20261029-omni/reviews/history-v2-review-scope-v2.json'
assert sha(SCOPE) == 'c73cba5795420d0aac2cd0db2af01ef9cf4df9b7dfab43120e20119bd02cd97c'
scope = json.loads(SCOPE.read_bytes())
before = [{**entry, 'actual_sha256': sha(ROOT/entry['path'])} for entry in scope['checked_file_scope']]
assert len(before) == 202 and all(row['sha256'] == row['actual_sha256'] for row in before)
old_evidence = ARTIFACTS/'execution-v1.json'
assert sha(old_evidence) == '59bbf75f5341ef2a721834171489f8997e4f8dbe9d83321948b170076b356418'
# Reuse the reviewer's exact original mutation functions, not the author's new variants.
old_driver = json.loads(old_evidence.read_bytes())['mutation_reproduction']
original = old_driver['script']
assert hashlib.sha256(original.encode()).hexdigest() == old_driver['script_sha256']
ns = {}
exec(original.split('probes = [', 1)[0], ns)
ns['WORK'] = WORK
fixture, base, api = ns['fixture'], ns['base'], ns['api']
records, commands = [], []

def command(args):
    start = datetime.now(UTC).isoformat()
    process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    record = {'command':args,'started_at':start,'completed_at':datetime.now(UTC).isoformat(),
              'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    commands.append(record)
    print(json.dumps({'command':args,'exit_code':process.returncode,'stdout':process.stdout[-3000:],
                      'stderr':process.stderr[-800:]}, ensure_ascii=False), flush=True)

command(['uv','run','--extra','cpu','pytest','-q','tests/test_course_export.py',
         'tests/test_historical_rechecks_v2.py','tests/test_historical_rechecks.py',
         'tests/test_incremental_revision_review.py','tests/test_course_review_preflight.py'])
command(['uv','run','--extra','cpu','ruff','check','scripts/export_course.py','scripts/check_site.py',
         'scripts/omni_revision/review_setup.py','docs/review-tools/historical_fix_rechecks_v2.py',
         'tests/test_course_export.py','tests/test_historical_rechecks_v2.py'])

def probe(name, expected, action, note='Synthetic fixture only; no actual stage receipt.'):
    start = datetime.now(UTC).isoformat()
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            observed_result = action()
        observed, error = 'accept', None
    except Exception as exc:
        observed_result, observed, error = None, 'reject', f'{type(exc).__name__}: {exc}'
    row = {'name':name,'expected':expected,'observed':observed,'matched':expected==observed,
           'result':observed_result,'error':error,'stdout':stream.getvalue(),
           'started_at':start,'completed_at':datetime.now(UTC).isoformat(),'scope_note':note}
    records.append(row)
    print(json.dumps({k:row[k] for k in ('name','expected','observed','error')}, ensure_ascii=False), flush=True)

def history_case(name, mutation=None, variant=None, multi=False):
    root = WORK/'history'/name
    root.mkdir(parents=True, exist_ok=False)
    built = fixture.fixture(root, variant, multi=multi)
    if mutation:
        mutation(root, built)
    ns['sync_seals'](root, built)
    # Persist the exact synthetic input before calling the real native entry.
    base.save(root, 'current/independent-candidate.json', built[0])
    result = fixture.check(root, built)
    return {'native_entry':'incremental_revision.check_dispositions -> v2.Rechecks',
            'result':repr(result),'synthetic_root':str(root)}

for name in ('continuity_current_created_before_original','technical_original_created_after_current',
             'technical_original_header_shell_wrong_source','continuity_original_unbound_source'):
    method = {'continuity_current_created_before_original':'chronology',
              'technical_original_created_after_current':'original_technical_time',
              'technical_original_header_shell_wrong_source':'original_technical_shell',
              'continuity_original_unbound_source':'original_continuity_source'}[name]
    probe(name,'reject',lambda n=name,m=method:history_case(n,ns[m]))
probe('history_valid_current','accept',lambda:history_case('valid_current'))
probe('history_valid_two_pages','accept',lambda:history_case('valid_two_pages',multi=True))
probe('history_honest_prior_technical_revise','accept',lambda:history_case('honest_prior_revise',variant='technical_original_honest_revise'))

def continuity_doc(root, built, mutate, original_report=True):
    proof = built[0]['pages'][0]['continuity']
    ref = proof['original_report' if original_report else 'report']
    ns['mutate_document'](root, built, ref, mutate)

def prior_continuity_revise(root, built):
    top_ref = built[0]['historical_fix_rechecks']
    top = api.document(root, top_ref)
    chain = api.document(root, top['chains'][0])
    pointer = api.document(root, chain['prior_active_pointer'])
    prior = api.document(root, {'path':pointer['manifest'],'sha256':pointer['manifest_sha256']})
    old_freeze = prior['source_freeze']
    old_page = api.document(root, old_freeze)['pages'][0]
    def change(doc):
        doc.update(source_freeze=old_freeze, verdict='revise')
        doc['pages'][0].update(source_sha256=old_page['source_sha256'],verdict='revise',
                               source_quote='312 notebooks; local rendering remains fixed.')
    continuity_doc(root,built,change)
    top = api.document(root,built[0]['historical_fix_rechecks'])
    chain = api.document(root,top['chains'][0])
    chain['rechecks'][0]['original_source_freeze'] = old_freeze
    top['chains'][0] = base.save(root,top['chains'][0]['path'],chain)
    built[0]['historical_fix_rechecks'] = base.save(root,top_ref['path'],top)

probe('history_honest_prior_continuity_revise','accept',lambda:history_case('honest_prior_continuity_revise',prior_continuity_revise))
probe('history_equal_continuity_created','accept',lambda:history_case('equal_continuity_created',lambda r,b:continuity_doc(r,b,lambda d:d.update(created_at=base.at(42)),False)))
for name, change in [
    ('original_quote_absent',lambda d:d['pages'][0].update(source_quote='Absent independent original quote.')),
    ('original_checks_empty',lambda d:d['pages'][0].update(checks={})),
    ('original_issues_wrong_type',lambda d:d['pages'][0].update(issues={})),
    ('original_schema_boolean',lambda d:d.update(schema_version=True)),
    ('original_figures_wrong',lambda d:d['pages'][0].update(figures_sha256={})),
]:
    probe(name,'reject',lambda n=name,f=change:history_case(n,lambda r,b:continuity_doc(r,b,f)))

def snapshot_mutation(variant):
    def mutate(root,built):
        manifest,_,_,sealed = built
        proof = manifest['pages'][0]['technical']
        doc = copy.deepcopy(api.document(root,proof['original_report']))
        old_raw = b'print("original independent bytes")\n'
        snap = base.save(root,'docs/technical-reviews/artifacts/original-program-v1.py',old_raw)
        base.save(root,'scripts/independent_program.py',b'print("current independent bytes")\n')
        item = {'id':'independent-old-program','kind':'repository_code','title':'Independent synthetic old source',
                'path':'scripts/independent_program.py','sha256':hashlib.sha256(old_raw).hexdigest(),
                'version':'synthetic original bytes','verified':True,'inspection_note':'Exact saved synthetic bytes.'}
        doc['sources'].append(item)
        ref = base.save(root,'current/independent-original-with-source.json',doc)
        proof['original_report'] = ref
        mapping = {'source':item['path'],'snapshot':snap}
        proof['original_source_snapshots'] = [mapping]
        if variant=='missing': proof.pop('original_source_snapshots')
        elif variant=='unregistered_path': mapping['source']='scripts/unknown.py'
        elif variant=='unregistered_digest':
            doc['sources'][-1]['sha256']='e'*64
            ref=base.save(root,ref['path'],doc);proof['original_report']=ref
        elif variant=='duplicate': proof['original_source_snapshots'].append(copy.deepcopy(mapping))
        elif variant=='wrong_bytes': mapping['snapshot']=base.save(root,'current/wrong-snapshot.py',b'print(99)\n')
        elif variant=='wrong_reference': mapping['snapshot']={**snap,'sha256':'f'*64}
        elif variant=='arbitrary_earlier_files': mapping['earlier_files']={item['path']:item['sha256']}
        elif variant=='escape': mapping['source']='../outside.py'
        elif variant=='bad_dict': proof['original_source_snapshots']={}
        elif variant=='bad_bool': proof['original_source_snapshots']=False
        elif variant=='bad_string': proof['original_source_snapshots']='saved snapshot'
        elif variant=='null_old_source': proof['original_source_snapshots']=None
        elif variant=='bad_member': proof['original_source_snapshots']=[None]
        elif variant=='bad_source_type': mapping['source']=['scripts/independent_program.py']
        elif variant=='bad_snapshot_type': mapping['snapshot']='not a path/hash reference'
        elif variant=='stale_current':
            current=copy.deepcopy(api.document(root,proof['report']));current['sources'].append(item)
            current_ref=base.save(root,'current/independent-stale-current.json',current)
            ns['replace_report'](root,built,proof['report'],current_ref)
        manifest['issue_reports'].append(ref)
        sealed['new']['technical']['report_sha256s'].append(ref['sha256'])
        sealed['new']['technical']['collection_index'].append({**ref,'received_at':base.at(23)})
    return mutate

for variant in ('valid','missing','unregistered_path','unregistered_digest','duplicate','wrong_bytes',
                'wrong_reference','arbitrary_earlier_files','escape','bad_dict','bad_bool','bad_string',
                'null_old_source','bad_member','bad_source_type','bad_snapshot_type','stale_current'):
    probe('snapshot_'+variant,'accept' if variant=='valid' else 'reject',
          lambda v=variant:history_case('snapshot_'+v,snapshot_mutation(v)))

setup = load('reviewer_actual_original_snapshot_helper',ROOT/'scripts/omni_revision/review_setup.py')
r1_original_path=ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_r_1/technical-original-v1.json'
r1_original=json.loads(r1_original_path.read_bytes())
saved_export=ROOT/'docs/technical-reviews/artifacts/omni-author-colab-preview-fix01/export-before.py'
mapping={'source':'scripts/export_course.py','snapshot':{'path':saved_export.relative_to(ROOT).as_posix(),'sha256':sha(saved_export)}}
probe('actual_registered_export_snapshot','accept',lambda:setup.original_repository_versions(r1_original,[mapping],api),
      'Actual immutable R.1 original sources registration and saved bytes only; not a full R.1 re-review or approval.')

from scripts import export_course
site_checker=load('reviewer_actual_site_checker',ROOT/'scripts/check_site.py')
index=json.loads((ROOT/'course/lesson-index.json').read_bytes())
old_pin='80c5db17e8b077b2c0a5e4e9baa48e760ccb0532'
probe('actual_331_old_colab_pin','reject',lambda:export_course.verify_colab_notebooks(index,old_pin),
      'Actual local Git objects for all current indexed notebooks; no remote publication or Colab execution.')
preview=ROOT/'outputs/omni-review-site-colab-preview-fix01'
probe('actual_preview_site_check','accept',lambda:site_checker.check(preview),
      'Actual generated preview checked offline; no browser/UI or Colab runtime claim.')
info=json.loads((preview/'build-info.json').read_bytes())
actual_preview={
    'path':str(preview),'build_info':info,'build_info_sha256':sha(preview/'build-info.json'),
    'notebook_count':len(index),'same_download_bytes':0,'upload_markers':0,'upload_home_links':0,
    'unpublished_notes':0,'upload_menu_text':0,'colab_github_links':0,
}
for item in index:
    path=preview/(item['id']+'.html')
    html=path.read_text()
    parsed=site_checker.Links();parsed.feed(html)
    actual_preview['same_download_bytes']+=sha(preview/item['notebook'])==sha(ROOT/item['notebook'])
    actual_preview['upload_markers']+='colab-upload-instructions' in parsed.ids
    actual_preview['upload_home_links']+='https://colab.research.google.com/' in parsed.links
    actual_preview['unpublished_notes']+='尚未發布的預覽版' in html
    actual_preview['upload_menu_text']+='File → Upload notebook' in html
    actual_preview['colab_github_links']+=sum(link.startswith('https://colab.research.google.com/github/') for link in parsed.links)
assert all(actual_preview[name]==len(index) for name in ('same_download_bytes','upload_markers','upload_home_links','unpublished_notes','upload_menu_text'))
assert actual_preview['colab_github_links']==0 and info['colab_mode']=='upload' and info['publication_ready'] is False

def site_case(name,variant):
    root=WORK/'site-cases'/name;root.mkdir(parents=True,exist_ok=False)
    notebook={'cells':[],'metadata':{'lesson_id':'1.1'}}
    one_index=[{'id':'1.1','notebook':'notebooks/01/1.1.ipynb'}]
    (root/'course').mkdir();(root/'course/lesson-index.json').write_text(json.dumps(one_index))
    (root/'notebooks/01').mkdir(parents=True);(root/'notebooks/01/1.1.ipynb').write_text(json.dumps(notebook))
    site=root/'site';site.mkdir();(site/'notebooks/01').mkdir(parents=True)
    (site/'notebooks/01/1.1.ipynb').write_text(json.dumps(notebook))
    meta={'lessons':1,'builder':'zensical','builder_version':'test','revision':old_pin,
          'colab_mode':'upload','publication_ready':False,'colab_git_objects_verified':False}
    marker='<span id="colab-upload-instructions"></span>'
    link='https://colab.research.google.com/'
    if variant=='unknown_mode': meta['colab_mode']='unknown'
    elif variant=='formal_upload': meta['publication_ready']=True
    elif variant=='formal_unverified': meta.update(colab_mode='github',publication_ready=True)
    elif variant=='no_upload_marker': marker=''
    elif variant=='stale_direct_link': link=f'https://colab.research.google.com/github/example/repo/blob/{old_pin}/old.ipynb'
    (site/'build-info.json').write_text(json.dumps(meta));(site/'search.json').write_text(json.dumps({'items':[{'location':'1.1.html'}]}))
    (site/'1.1.html').write_text(f'<meta name="generator" content="zensical-test"><h1 id="1.1">1.1</h1>{marker}<a href="notebooks/01/1.1.ipynb">Download</a><a href="{link}">Colab</a>')
    prior_root=site_checker.ROOT;site_checker.ROOT=root
    try:site_checker.check(site)
    finally:site_checker.ROOT=prior_root

for variant in ('valid','unknown_mode','formal_upload','formal_unverified','no_upload_marker','stale_direct_link'):
    probe('site_'+variant,'accept' if variant=='valid' else 'reject',lambda v=variant:site_case(v,v))

replay=load('reviewer_actual_baseline_recheck',ROOT/'docs/review-tools/baseline_replay.py')
supp_path=ROOT/'docs/course-revision-20261029-omni/baseline-supplements-v1.json'
supp=json.loads(supp_path.read_bytes())['files']
tables=replay.preserved_identities(ROOT)
provenance=[]
for entry in supp:
    actual=sha(ROOT/entry['path'])
    authenticated={label:table[entry['live_path']] for label,table in tables.items() if entry['live_path'] in table}
    assert actual==entry['sha256'] and authenticated and all(value==actual for value in authenticated.values())
    provenance.append({**entry,'actual_sha256':actual,'authenticated_old_tables':authenticated})
assert len(supp)==5
def baseline_run(label,entries):
    result=replay.replay(ROOT,entries)
    path=WORK/(label+'.json');save_json(path,result)
    if label=='baseline-four':
        assert result['status']=='failed' and any('scripts/check_site.py' in item for item in result['failures'])
    else:
        assert result['status']=='passed_metadata_checks' and result['failures']==[]
    return {'status':result['status'],'failures':result['failures'],'records':len(result.get('records',[])),
            'raw_outside_git':str(path),'sha256':sha(path)}
probe('real_baseline_with_previous_four','accept',lambda:baseline_run('baseline-four',supp[:4]),
      'Native replay returns a failed result for missing check_site restoration; not an expected successful gate.')
probe('real_baseline_with_authenticated_five','accept',lambda:baseline_run('baseline-five',supp),
      'Actual preserved baseline replay only; does not certify this round final v2 candidate.')
after=[{**entry,'actual_sha256':sha(ROOT/entry['path'])} for entry in scope['checked_file_scope']]
assert all(row['sha256']==row['actual_sha256'] for row in after)
result={
    'schema_version':1,'kind':'bounded_independent_code_recheck_execution_not_final_intake',
    'reviewer_task':TASK,'created_at':datetime.now(UTC).isoformat(),
    'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU / std-library; no training'},
    'command':'uv run --extra cpu python '+Path(__file__).relative_to(ROOT).as_posix(),
    'scope':{'path':SCOPE.relative_to(ROOT).as_posix(),'sha256':sha(SCOPE),'count':202,'before_matches':202,'after_matches':202},
    'commands':commands,'probes':records,'actual_preview_dom_and_bytes':actual_preview,
    'baseline_supplement_provenance':provenance,
    'actual_original_snapshot':{'report':{'path':r1_original_path.relative_to(ROOT).as_posix(),'sha256':sha(r1_original_path)},'mapping':mapping},
    'limitations':['Fixture receipts are synthetic; no actual completed final manifest/v2chains/26+4whole intake has been certified.',
                   'Preview DOM/text, download bytes and local Git blobs were inspected; no visual/browser review, Colab login/runtime or publication was performed.',
                   'Two environment_status requests returned configuration changed; local executor remained available. No network or credential-dependent action was required.'],
}
output=ARTIFACTS/'code-recheck-execution-v1.json';save_json(output,result)
print(json.dumps({'output':output.relative_to(ROOT).as_posix(),'sha256':sha(output),
                  'probes':len(records),'unexpected':[r['name'] for r in records if not r['matched']]},ensure_ascii=False),flush=True)
