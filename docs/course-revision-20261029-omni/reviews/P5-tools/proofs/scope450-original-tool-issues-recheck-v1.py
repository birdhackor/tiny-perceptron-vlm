"""同一獨立reviewer對四檔修正的正負對照；不修改任何舊原判／來源。"""

import copy
import hashlib
import importlib.util
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace(value, old, new):
    if isinstance(value, dict):
        if value == old:
            value.clear()
            value.update(new)
            return
        if value.get('path') == old['path'] and value.get('sha256') == old['sha256']:
            value.update(new)
        for child in list(value.values()):
            replace(child, old, new)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if child == old['sha256']:
                value[index] = new['sha256']
            else:
                replace(child, old, new)


fixture = load('p5_current_fixture', ROOT / 'tests/test_historical_rechecks.py')


def history(variant, old_schema=False):
    with tempfile.TemporaryDirectory(prefix='p5-current-history-') as directory:
        root = Path(directory)
        built = fixture.fixture(root)
        manifest, live, roles, sealed = built
        sidecar = fixture.review.document(root, manifest['historical_fix_rechecks'])
        proof = manifest['pages'][0]['technical']
        old_full = copy.deepcopy(proof['report'])
        current = fixture.review.document(root, old_full)
        current['created_at'] = fixture.at(23)
        full_ref = fixture.save(root, old_full['path'], current)
        replace(manifest, old_full, full_ref)
        replace(sidecar, old_full, full_ref)
        replace(sealed, old_full, full_ref)
        for item in sealed['new']['technical']['collection_index']:
            if item['path'] == full_ref['path']:
                item['received_at'] = fixture.at(24)
        original = copy.deepcopy(current)
        original['created_at'] = fixture.at(22)
        original['verdict'] = 'revise'
        for claim in original['claims']:
            claim['status'] = 'unresolved'
        for check in original['checks'].values():
            check['status'] = 'revise'
        if variant == 'header_shell':
            original = {k:v for k,v in original.items() if k in {
                'schema_version','review_stage','lesson_id','reviewer_context','reviewer_task',
                'source_sha256','created_at','verdict','issues'}}
            original['source'] = 'NO_SUCH_SOURCE.md'
            original['source_sha256'] = 'f' * 64
        elif variant == 'missing_claims':
            del original['claims']
        elif variant == 'wrong_original_source':
            original['source'] = 'NO_SUCH_SOURCE.md'
        elif variant == 'original_after_current':
            original['created_at'] = fixture.at(24)
        original_ref = fixture.save(root, 'current/genuine-original.json', original)
        proof['original_report'] = original_ref
        proof['original_source_freeze'] = copy.deepcopy(manifest['source_freeze'])
        manifest['issue_reports'].append(original_ref)
        phase = sealed['new']['technical']
        phase['report_sha256s'].append(original_ref['sha256'])
        phase['collection_index'].append({**original_ref, 'received_at':fixture.at(23)})
        if variant == 'original_after_current':
            phase['collection_index'][-1]['received_at'] = fixture.at(24)
        elif variant == 'missing_original_receipt':
            phase['collection_index'].pop()
        elif variant == 'duplicate_original_receipt':
            phase['collection_index'].append(copy.deepcopy(phase['collection_index'][-1]))
        elif variant == 'missing_original_freeze':
            del proof['original_source_freeze']
        elif variant == 'wrong_freeze_sha':
            proof['original_source_freeze']['sha256'] = 'f' * 64
        elif variant == 'appendix_before_current':
            prior = copy.deepcopy(sidecar['rechecks'][0]['appendix'])
            appendix = fixture.review.document(root, prior)
            appendix['at'] = fixture.at(22)
            new = fixture.save(root, prior['path'], appendix)
            replace(manifest, prior, new)
            replace(sidecar, prior, new)
            replace(sealed, prior, new)
        chain_ref = fixture.save(root, 'current/recheck-chain.json', sidecar)
        manifest['historical_fix_rechecks'] = chain_ref if old_schema else fixture.save(
            root,'current/recheck-v2.json',{'schema_version':2,'chains':[chain_ref]})
        try:
            result = fixture.check(root, built)
            return {'variant':variant,'schema':1 if old_schema else 2,'accepted':True,'result':result}
        except (ValueError, KeyError, TypeError, OSError) as error:
            return {'variant':variant,'schema':1 if old_schema else 2,'accepted':False,
                    'error_type':type(error).__name__,'error':str(error)}


def notebook():
    exporter = load('scripts.p5_current_export', ROOT / 'scripts/export_course.py')
    source = {'cells':[
        {'cell_type':'code','metadata':{'course_setup':True},'source':['import math\n'],
         'execution_count':None,'outputs':[]},
        {'cell_type':'code','metadata':{},'source':['print("TEACHING")\n'],
         'execution_count':None,'outputs':[]},
        {'cell_type':'code','metadata':{'course_figure':{'name':'reviewed','alt':'已審圖'}},
         'source':['print("FIGURE")\n'],'execution_count':None,'outputs':[]},
    ],'metadata':{'lesson_id':'1.1','reviewed':'canonical'},'nbformat':4,'nbformat_minor':5}
    executed = copy.deepcopy(source)
    executed['metadata'] = {'lesson_id':'WRONG'}
    for cell in executed['cells']:
        cell['execution_count'] = 9
        cell['outputs'] = [{'output_type':'stream','name':'stdout','text':['actual result\n']}]
    executed['cells'][0]['metadata'] = {}
    executed['cells'][1]['metadata'] = {'course_setup':True}
    executed['cells'][2]['metadata'] = {'course_figure':{'name':'UNREVIEWED','alt':'wrong'}}
    before = copy.deepcopy(source)
    with tempfile.TemporaryDirectory(prefix='p5-current-notebook-') as directory:
        path = Path(directory) / 'executed.ipynb'
        path.write_text(json.dumps(executed))
        result = exporter.checked_notebook(source,path)
        visible = [''.join(cell['source']) for cell in result['cells']
                   if not cell['metadata'].get('course_setup') and 'course_figure' not in cell['metadata']]
        checks = {
            'old_metadata_swap_no_longer_hides_reviewed_teaching_code':visible == ['print("TEACHING")\n'],
            'original_top_level_metadata_preserved':result['metadata'] == source['metadata'],
            'original_figures_preserved':result['cells'][2]['metadata'] == source['cells'][2]['metadata'],
            'actual_outputs_counts_merged':all(x['execution_count']==9 and x['outputs'] for x in result['cells']),
            'original_not_mutated':source == before,
        }
        for variant in ['changed_source','not_executed','execution_error']:
            bad = copy.deepcopy(executed)
            if variant == 'changed_source':bad['cells'][1]['source']=['print("DIFFERENT")\n']
            elif variant == 'not_executed':bad['cells'][1]['execution_count']=None
            else:bad['cells'][1]['outputs']=[{'output_type':'error','ename':'Error','evalue':'bad','traceback':[]}]
            path.write_text(json.dumps(bad))
            try:exporter.checked_notebook(source,path)
            except ValueError:checks[variant+'_rejected']=True
            else:checks[variant+'_rejected']=False
    return checks


cases = [history(name) for name in ['honest_complete_revise','header_shell','missing_claims',
         'wrong_original_source','original_after_current','missing_original_receipt',
         'duplicate_original_receipt','missing_original_freeze','wrong_freeze_sha','appendix_before_current']]
legacy = history('header_shell',old_schema=True)
notebooks = notebook()
result = {'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),
          'python':platform.python_version(),'history_cases':cases,'legacy_schema1_unchanged_control':legacy,
          'notebook_controls':notebooks,
          'all_expected':cases[0]['accepted'] and all(not x['accepted'] for x in cases[1:])
                         and legacy['accepted'] and all(notebooks.values()),
          'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
              'docs/review-tools/historical_fix_rechecks.py','docs/review-tools/historical_fix_rechecks_v2.py',
              'scripts/export_course.py','tests/test_historical_rechecks.py',
              'tests/test_historical_rechecks_v2.py','tests/test_course_export.py']},
          'network_paid_compute_actual_final':False}
out=Path(__file__).with_name('scope450-original-tool-issues-recheck-result-v1.json')
with out.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(0 if result['all_expected'] else 1)
