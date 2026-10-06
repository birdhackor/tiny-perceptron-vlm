"""Record same-reviewer callback, assert the canonical, then run one checker."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
REL=OUT.relative_to(ROOT).as_posix()
TASK='/root/phase4_factual_coordinator/factual_11_7'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda path:json.loads(path.read_text())
write=lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
inspection=read(OUT/'current-inspection.json')
preserved=read(OUT/'opaque-preservation-receipt.json')
prior_path=ROOT/preserved['canonical']['opaque_file']
assert digest(prior_path.read_bytes())==preserved['canonical']['sha256']
report=read(prior_path)
assert report['reviewer_task']==TASK==inspection['reviewer_task']
assert report['source_sha256']==inspection['primary']['source_sha256']
assert inspection['outcome']=='All fingerprint assertions passed; current necessary context supports unchanged 11.7 claims. Same-reviewer verdict remains pass.'
assert (OUT/'callback_inspection.stderr.txt').read_bytes()==b''
assert all(row['match'] for group in ['source_fingerprint_checks','prior_formal_artifact_fingerprint_checks','additional_original_code_fingerprint_checks'] for row in inspection[group])

def artifact(id,name,kind,description,**fields):
    path=OUT/name
    report['artifacts'].append({'id':id,'path':path.relative_to(ROOT).as_posix(),
                                'sha256':digest(path.read_bytes()),'kind':kind,
                                'description':description,**fields})

artifact('a-current-inspection-20261006','current-inspection.json','execution',
         'Actual same-original-reviewer narrow current callback: full current11.7 reading, necessary7.15 slices, exact source/code/proof hashes and explicitly bounded reuse.',
         command=inspection['command'],result='exit_code=0;7 current source files,30 prior formal artifacts and10 additional original code files match; required current context supports unchanged claims.',
         environment=inspection['environment'])
artifact('a-callback-code-20261006','callback_inspection.py','code','Actual executed current callback hash/slice/metadata program; no model execution.')
artifact('a-callback-update-code-20261006','update_canonical_and_check.py','code','Actual canonical update/assertion program; executes only one selected checker after assertions.')
artifact('a-callback-stdout-20261006','callback_inspection.stdout.txt','source_snapshot','Actual current callback stdout with primary/context hashes, verification counts and environment.')
artifact('a-callback-stderr-20261006','callback_inspection.stderr.txt','source_snapshot','Actual current callback stderr was empty; execution exit status was0.')
artifact('a-current-section-20261006','current-inputs/11.7.md','source_snapshot','Current own11.7 complete original raw UTF-8 bytes, unchanged; no whole-chapter current-hash claim.')
artifact('a-current-context-7_15-20261006','current-inputs/7.15.md','source_snapshot','Current7.15 raw section bytes; complete section read to identify dependencies, only four necessary slices verified for11.7.')
for i in inspection['necessary_context']['verified_slices']:
    artifact('a-'+i['id']+'-20261006',str(Path(i['snapshot']).relative_to(OUT.relative_to(ROOT))),
             'source_snapshot','Exact unnormalized raw source byte range used as necessary current7.15 context. '+i['scope'])
artifact('a-prior-canonical-opaque-20261006','prior-opaque/11.7.prior.json','source_snapshot','Exact own previous canonical saved opaquely before callback reading; prior history/proof declarations preserved.')
artifact('a-prior-proof-opaque-20261006','prior-opaque/phase4-11_7-independent.prior-proof.tar','source_snapshot','Opaque full original own proof directory archive;61 original files retained with their individual hashes in preservation receipt.')
artifact('a-prior-preservation-20261006','opaque-preservation-receipt.json','source_snapshot','Actual opaque copy/archive timestamps, exact prior canonical/archive paths and SHA-256 plus original proof file manifest.')
for item,id in zip(inspection['current_methods'],['a-current-factual-instructions-20261006','a-current-review-protocol-20261006','a-current-checker-code-20261006']):
    artifact(id,str(Path(item['snapshot']).relative_to(OUT.relative_to(ROOT))),'source_snapshot',
             'Current method/schema original bytes captured for this callback: '+item['path'])

report['sources'].append({'id':'s-current-inspection-20261006','kind':'execution','title':'2026-10-06 same-original-reviewer current inspection and bounded reuse','verified':True,'artifact_id':'a-current-inspection-20261006'})
for position,claim in enumerate(report['claims']):
    claim['artifact_ids'].append('a-current-inspection-20261006')
    claim['evidence'].append({'source_id':'s-current-inspection-20261006',
                              'locator':f'/primary, /source_fingerprint_checks, /prior_formal_artifact_fingerprint_checks, /reuse_matrix/{position}',
                              'supports':inspection['reuse_matrix'][position]['current_assessment']+' Prior executions and original-source readings retain their real2026-10-05 dates; no new model execution or new source fetch is claimed.'})
for source in report['sources']:
    if source['id']!='s-current-inspection-20261006':
        source['reused_after_fingerprint_check_on']='2026-10-06'
        source['reuse_scope_note']='Original personal reading/execution dates, locator and support scope retained; callback checks unchanged bytes and relevant current context rather than claiming a new original-source/model run.'

report['initial_reviewed_on']=report['reviewed_on']
report['reviewed_on']='2026-10-06'
report['reviewer_context_note']='Original independent11.7 task had fresh context. This current callback is by that same original reviewer; it does not claim a new independent dispatch.'
report['reading_scope']['current_callback']={
 'primary_complete_section_read':inspection['primary'],
 'necessary_context':{
    'source':'course/chapters/07.md#7.15','current_section_sha256':inspection['necessary_context']['current_section_sha256'],
    'verified_scope':'Four raw slices for shared parameters, fixed original tasks, proper before/after baseline and train/holdout separation.',
    'slice_ids':[i['id'] for i in inspection['necessary_context']['verified_slices']],
    'prior_context_snapshot_sha256':inspection['necessary_context']['prior_frozen_context']['sha256'],
    'prior_sha_meaning':inspection['necessary_context']['prior_sha_meaning'],
    'assessment':inspection['necessary_context']['semantic_assessment']},
 'original_prerequisite_list_meaning':'Original necessary_prerequisites and snapshots describe the2026-10-05 initial reading; callback does not assert a new scientific audit of every current prerequisite or whole chapter.',
 'reuse_scope':inspection['reuse_scope'],
 'no_new_whole_file_hash_claim':inspection['frozen_whole_file_hash_policy']}
report['reading_scope']['visual']='Original2026-10-05 desktop/mobile render+view is explicitly carried forward for byte-identical11.7/table. This2026-10-06 callback did not rerender; it checked no referenced SVG and unchanged primary bytes.'
report['current_dependency_sha256']={'course/chapters/07.md#7.15':inspection['necessary_context']['current_section_sha256']}
report['current_inspections']=[{'artifact_id':'a-current-inspection-20261006','path':REL+'/current-inspection.json',
                               'sha256':digest((OUT/'current-inspection.json').read_bytes()),
                               'reviewer_task':TASK,'inspected_at':inspection['inspected_at'],
                               'primary_sha256':inspection['primary']['source_sha256'],
                               'necessary_context_sha256':inspection['necessary_context']['current_section_sha256'],
                               'scope':'Same original reviewer; current primary plus necessary7.15 slices and exact unchanged proof/source reuse. No new training/model/source-fetch/render claim.'}]
report.setdefault('history',[]).append({'event':'Same-original-reviewer current callback after opaque preservation',
                                     'recorded_on':'2026-10-06','prior_canonical':preserved['canonical'],
                                     'prior_proof_archive':{k:v for k,v in preserved['prior_proof_archive'].items() if k!='files'},
                                     'current_inspection_artifact_id':'a-current-inspection-20261006',
                                     'primary_unchanged':True,'current_context7_15_changed':True,
                                     'result':'Necessary context premise preserved; unchanged original11.7 technical support retained;pass.'})
notes={
 'factual_accuracy':'2026-10-06 callback: personally re-read current11.7 and necessary7.15 context, verified unchanged scientific support scope and all retained source/proof fingerprints. No other review conclusion substitutes for this assessment.',
 'numeric_verification':'2026-10-06 callback: unchanged2026-10-05 original fence and raw-data accounting are explicitly reused; no recipe/model/measurement rerun was needed for the context wording/link change.',
 'figure_consistency':'2026-10-06 callback: zero referenced figures and unchanged11.7 content/table; prior2026-10-05 actual desktop/mobile visual proof is retained, with no new rendering claim.',
 'source_verification':'2026-10-06 callback:7 direct repository sources,30 registered proof artifacts and10 additional code files match exactly. Original HTTPS URLs/versions/access dates and scientific support limits remain intact; current7.15 textbook prose is context to verify, not authority.',
 'limitations':'2026-10-06 callback:7.15 arithmetic experiment and supplemental model-link claim are outside the11.7 numeric evidence dependency. The same original handmade/measured distinction, fixed small-task scope, unmatched token budget and probability/token/time limits remain.'}
for name,note in notes.items():
    report['checks'][name]['details']+=' '+note
    report['checks'][name]['current_inspection_artifact_id']='a-current-inspection-20261006'

canonical=ROOT/'docs/technical-reviews/11.7.json'
write(canonical,report)
# Canonical/source/task assertions must succeed before the one checker invocation.
current=read(canonical)
assert current['reviewer_task']==TASK and current['review_stage']=='technical' and current['lesson_id']=='11.7'
assert current['source']=='course/chapters/11.md#11.7' and current['verdict']=='pass'
raw=(ROOT/'course/chapters/11.md').read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 11.7 '));body=raw[heads[i].start():heads[i+1].start()]
assert digest(body)==current['source_sha256']==inspection['primary']['source_sha256']
ctx=(ROOT/'course/chapters/07.md').read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',ctx));i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 7.15 '));ctxbody=ctx[heads[i].start():heads[i+1].start()]
assert digest(ctxbody)==current['current_dependency_sha256']['course/chapters/07.md#7.15']
registered=next(a for a in current['artifacts'] if a['id']=='a-current-inspection-20261006')
assert registered['sha256']==digest((ROOT/registered['path']).read_bytes())==current['current_inspections'][0]['sha256']
assert current['current_inspections'][0]['reviewer_task']==TASK
assert digest(prior_path.read_bytes())==preserved['canonical']['sha256']
command=[str(ROOT/'.venv/bin/python'),'scripts/check_technical_reviews.py','--lesson','11.7']
checked=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
(OUT/'checker.stdout.txt').write_bytes(checked.stdout);(OUT/'checker.stderr.txt').write_bytes(checked.stderr)
receipt={'checked_at':datetime.now(UTC).isoformat(),'command_argv':command,
         'command':'.venv/bin/python scripts/check_technical_reviews.py --lesson 11.7',
         'cwd':str(ROOT),'exit_code':checked.returncode,'checker_invocations_in_callback':1,
         'canonical_and_task_assertions':'Succeeded before this actual checker invocation.',
         'stdout_path':REL+'/checker.stdout.txt','stdout_sha256':digest(checked.stdout),'stdout':checked.stdout.decode(),
         'stderr_path':REL+'/checker.stderr.txt','stderr_sha256':digest(checked.stderr),'stderr':checked.stderr.decode(),
         'checker_sha256':digest((ROOT/'scripts/check_technical_reviews.py').read_bytes()),
         'report_path':'docs/technical-reviews/11.7.json','report_sha256':digest(canonical.read_bytes()),
         'primary_source_sha256':current['source_sha256'],'context7_15_sha256':digest(ctxbody),
         'current_inspection_artifact':registered,'prior_canonical_opaque':preserved['canonical'],
         'prior_proof_opaque':{k:v for k,v in preserved['prior_proof_archive'].items() if k!='files'},
         'environment':inspection['environment'],
         'scope':'Selected checker validates schema/identity/source/artifact versions; substantive current dependency and support-scope judgment is the same reviewer inspection. No new model check.'}
write(OUT/'checker-receipt.json',receipt)
print(json.dumps({'report_path':receipt['report_path'],'report_sha256':receipt['report_sha256'],
                  'primary_source_sha256':receipt['primary_source_sha256'],
                  'context7_15_sha256':receipt['context7_15_sha256'],'prior_canonical_opaque':receipt['prior_canonical_opaque'],
                  'prior_proof_opaque':receipt['prior_proof_opaque'],'current_inspection_artifact':registered,
                  'checker_exit_code':checked.returncode,'checker_stdout':checked.stdout.decode(),
                  'checker_receipt':REL+'/checker-receipt.json','checker_receipt_sha256':digest((OUT/'checker-receipt.json').read_bytes()),
                  'events':inspection['events'],'reuse_scope':inspection['reuse_scope']},ensure_ascii=False,indent=2))
raise SystemExit(checked.returncode)
