"""Post-original documentary verification of two historical raw-score audit records."""
import ast
import gzip
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).with_name('historical-c21-inspection-v1.json')
ORIGINAL=ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/technical-validation/technical-original-v1.json'
BASE=ROOT/'docs/lesson-products/reviews'
COMMIT='f8188f88d205ca92a40fd53312d29e31ec17e2e7'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(p.read_text())

def main():
    assert sha(ORIGINAL)=='37e0d866854a13dfaf44cfee8154ec631ad681550719a649d03c09720d11bf94'
    b=read(BASE/'engineering-final-b-mechanisms-original.json');be=read(BASE/'engineering-final-b-mechanisms-execution.json')
    c=read(BASE/'engineering-final-c-original.json');ce=read(BASE/'engineering-final-c-execution.json')
    assert b['execution']['sha256']==sha(BASE/'engineering-final-b-mechanisms-execution.json')
    assert c['execution_artifacts'][0]['sha256']==sha(BASE/'engineering-final-c-execution.json')
    assert b['reviewer_task']==be['reviewer_task'] and c['reviewer_task']==ce['reviewer_task'] and b['reviewer_task']!=c['reviewer_task']
    source=BASE/'engineering-final-c-probe-source.txt';src=source.read_text()
    assert sha(source)==ce['probe_script_sha256']==c['execution_artifacts'][1]['sha256']
    tree=ast.parse(src)
    functions=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
    assert 'independent64' in functions and 'explicit32' in functions and "if row['product']!='c':continue" in src
    historical=[]
    for name in ['engineering-final-b-mechanisms-original.json','engineering-final-b-mechanisms-execution.json','engineering-final-c-original.json','engineering-final-c-execution.json','engineering-final-c-probe-source.txt']:
        p=BASE/name;blob=subprocess.check_output(['git','show',COMMIT+':'+str(p.relative_to(ROOT))],cwd=ROOT)
        assert blob==p.read_bytes();historical.append({'source':rec(p),'exact_in_historical_commit':True})
    raw=Path('/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz')
    uncompressed=gzip.decompress(raw.read_bytes());raw_sha=hashlib.sha256(uncompressed).hexdigest()
    assert raw_sha==b['pretest_freeze_check']['raw_sha256']==c['raw_reference']['sha256']==ce['raw_reference']['sha256']
    assert len(uncompressed)==c['raw_reference']['bytes']==56528235
    checked=[]
    aliases={'outputs/lesson-products/final-report-v1.json':'docs/lesson-products/results/final-report-v1.json',
             'outputs/lesson-products/final-report-v1.started.json':'docs/lesson-products/results/final-report-v1.started.json',
             'outputs/lesson-products/products/c/artifact-manifest.json':'docs/lesson-products/results/c-artifact-manifest.json',
             'outputs/lesson-products/products/c/calibration.json':'docs/lesson-products/results/c-calibration.json',
             'outputs/lesson-products/data-v2/manifest.json':'docs/lesson-products/data-v2-manifest.json',
             'outputs/lesson-products/products/b/auto-policy.json':'docs/lesson-products/results/b-auto-policy.json'}
    for role,report in [('B',b),('C',c)]:
        entries=[]
        for item in report['checked_files']:
            path=item['path'];p=ROOT/aliases.get(path,path)
            if path.endswith('final-report-v1.predictions.jsonl'):
                entries.append({'historical_path':path,'matched_current_raw_gzip_after_decompression':raw_sha==item['sha256']})
            elif p.exists():
                entries.append({'historical_path':path,'current_original_path':str(p.relative_to(ROOT)),'sha_matches':sha(p)==item['sha256']})
            else:entries.append({'historical_path':path,'not_present_at_old_local_path':True,'scope':'Not required for this documentary raw-score recomputation check; no checkpoint loaded.'})
        checked.append({'role':role,'entries':entries})
    bpath=Path(b['execution']['script_local_path'])
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_validation',
            'environment':{'python':platform.python_version(),'device':'CPU; JSON/AST/Git/byte checks only'},
            'original_report_unchanged':rec(ORIGINAL),'authorized_scope':'Only c21 historical documentary statement, after independent original was saved. No peer pass used to establish scientific numbers.',
            'original_actor_records':{'B':b['reviewer_task'],'C':c['reviewer_task'],'distinct_task_names':True,'C_declared_author_tasks':c['author_tasks'],
                                      'identity_limit':'Task names and scope declarations are documentary provenance, not a fresh human-witness or live historical-task execution assertion. Git committer identity does not establish independent agent identity.'},
            'historical_commit':{'commit':COMMIT,'commit_metadata':subprocess.check_output(['git','show','-s','--format=%H %cI %an <%ae>',COMMIT],cwd=ROOT,text=True).strip(),'files':historical},
            'B_actual_execution_record':{'source':rec(BASE/'engineering-final-b-mechanisms-execution.json'),'command':be['command'],'script_sha256':be['script_sha256'],
                                         'declared_script_local_path':str(bpath),'script_present':bpath.exists(),'source_available_for_inspection':False,
                                         'raw_identity':b['pretest_freeze_check']['raw_sha256'],'group_tables':b['verification']['group_tables_recomputed'],'metric_cells':b['verification']['metric_cells_compared'],
                                         'raw_result_examples':{'B_boxes':b['B_result']['boxes'],'B_sum':b['B_result']['sum'],'Auto_complete':b['Auto_result']['complete_eligible_denominator']},
                                         'recorded_boundary':be['boundary']},
            'C_actual_program_and_execution':{'execution':rec(BASE/'engineering-final-c-execution.json'),'source':rec(source),'command':ce['command'],'functions':functions,
                                               'raw_stream_then_C_filter':True,'independent_float64_and_explicit32':True,'raw_rows':ce['checks']['raw_C_row_integrity_counts']['details']['C_rows'],
                                               'metric_sets':ce['checks']['all_C_core_and_wording_before_after_metrics_recomputed']['details']['metric_sets'],
                                               'actual_raw_reference':ce['raw_reference'],'declared_boundary':ce['scope']},
            'raw_current_identity':{'gzip':rec(raw),'uncompressed_bytes':len(uncompressed),'uncompressed_sha256':raw_sha},'checked_file_bindings':checked,
            'current_independent_numeric_basis':'Own pre-original a_legacy/a_new computations remain scientific evidence; historical reports are inspected here only for the statement that separate raw-recomputation records were saved.',
            'finding':'The historical sentence is supported as documentary attribution: two distinct reviewer-task records bind the same actual preserved raw digest and retain separate commands, environments, script digests, calculation scopes and detailed computed outputs. C source is also preserved and inspected. B original source bytes are absent, so full replay of that one-time audit program is not established; the page does not promise such replayability. Own pre-original numerical checks remain separate evidence for the scientific results.',
            'adequacy_for_c21':{'historical_record_attribution':'sufficient','original_B_program_full_replay':'not_verified_source_not_preserved','reason':'The claim is that the recorded audits recomputed saved scores, not that both one-time audit implementations remain replayable. B output includes206group tables/1236metric cells and concrete B/mechanism/Auto quantities, bound to an exact raw SHA and source-script SHA; C output and exact source independently retain67154rawCrows/12metric sets. These original records predate this review in fixed Git bytes. Neither their verdict values nor file existence alone is used.'},
            'not_done':['No historical reviewer task invoked','No original probe rerun','No neural model, checkpoint, training, calibration search, Modal, workflow or publishing call','No historical review verdict used as proof']}
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'bytes':OUT.stat().st_size,'sha256':sha(OUT),'historical_actor_names_distinct':True,'C_program_matched':True,'B_program_present':False,'original_unchanged':sha(ORIGINAL)},ensure_ascii=False))

if __name__=='__main__':main()
