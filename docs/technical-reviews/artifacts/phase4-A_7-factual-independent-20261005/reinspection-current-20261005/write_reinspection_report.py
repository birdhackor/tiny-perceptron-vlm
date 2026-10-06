import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PREFIX=HERE.relative_to(ROOT).as_posix()
TASK='/root/phase4_factual_coordinator/factual_a_7'
EXPECTED='32397500c886a31262e690ab72dd5ba34e5fc2936e5dbac9bebb9547d23bb58d'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior_path=HERE/'prior-report.opaque.json'
prior=json.loads(prior_path.read_bytes())
assert prior['reviewer_task']==TASK
assert digest(ROOT/'docs/technical-reviews/A.7.json')==digest(prior_path)
result=json.loads((HERE/'changed-budget-result.json').read_bytes())
render=json.loads((HERE/'changed-paragraph-render.json').read_bytes())
assert result['source_sha256']==render['source_sha256']==EXPECTED
assert all(item['hash_match'] for item in result['prior_artifact_hash_checks'])
assert all(item['unchanged'] for item in result['live_code_and_raw_hash_checks'])
for p in ['scripts/check_technical_reviews.py','docs/review-tools/section_facts.py']:
 assert digest(ROOT/p)==digest(HERE.parent/'inputs'/p)

new_artifacts=[]
def artifact(identifier,file,kind,description,**more):
 path=HERE/file
 value={'id':identifier,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),
        'kind':kind,'description':description,**more}
 new_artifacts.append(value)
 return value
cpu_environment={'python':'3.13.5','device':'cpu','framework':'Python standard library only',
                 'platform':'Linux 6.18.44; glibc 2.41'}
artifact('a7_prior_report_opaque','prior-report.opaque.json','source_snapshot',
         'Exact preserved own prior report, including original claims, pass judgment and empty issue history; never edited')
artifact('a7_current_section','current-section.md','source_snapshot','Complete current A.7 raw UTF-8 bytes, no normalization')
artifact('a7_current_context_7_19','current-context-7.19.md','source_snapshot',
         'Personally reread necessary current 7.19 length/serialization context; bytes identical to own previously read input')
artifact('a7_actual_delta','section.diff','derivation',
         'Actual byte-preserving section comparison: only opening input-over-capacity arithmetic wording changed')
artifact('a7_current_factual_method','factual-reviewer-instructions-current.md','source_snapshot',
         'Latest factual reviewer method personally read for this same-owner reinspection')
artifact('a7_changed_budget_code','verify_changed_budget.py','code',
         'AST extraction of current unchanged fence literals, changed arithmetic checks and exact unchanged-evidence hashes')
artifact('a7_changed_budget_execution','changed-budget-result.json','execution',
         'Actual successful bounded arithmetic and unchanged-evidence verification',
         command='.venv/bin/python '+PREFIX+'/verify_changed_budget.py',
         result='exit 0; P=105 even G=0 exceeds C=100 by 5; keeping G=25 needs input reduction 30; reduction 29 leaves 101. All 31 own prior artifacts and 6 live code/raw files SHA-match.',
         environment=cpu_environment)
artifact('a7_changed_budget_stdout','changed-budget.stdout.txt','source_snapshot','Actual new short CPU stdout')
artifact('a7_changed_budget_stderr','changed-budget.stderr.txt','source_snapshot','Actual new short CPU stderr; empty on exit 0')
artifact('a7_changed_render_code','render_changed_paragraph.py','code','Actual local-page changed-paragraph rendering code')
artifact('a7_changed_render_execution','changed-paragraph-render.json','execution',
         'Actual current local page render; only changed opening paragraph inspected',
         command='.venv/bin/python '+PREFIX+'/render_changed_paragraph.py',
         result='exit 0; HTTP 200; changed paragraph equals current raw source at both 1280×800 and 390×844; both screenshots personally viewed.',
         environment={'python':'3.13.5','device':'cpu','browser':'Chromium 151.0.7922.173','viewport':'1280×800 and 390×844'})
artifact('a7_changed_desktop','changed-paragraph-desktop.png','figure_render','Personally viewed current changed paragraph at desktop viewport')
artifact('a7_changed_mobile','changed-paragraph-mobile.png','figure_render','Personally viewed current changed paragraph at mobile viewport')
artifact('a7_reinspection_writer','write_reinspection_report.py','code','Same original owner explicit receipt/report update logic')

receipt={'schema_version':1,'kind':'same_owner_technical_reinspection','reviewer_task':TASK,
 'same_original_owner':True,'lesson_id':'A.7','source':'course/chapters/0A.md#A.7',
 'source_sha256':EXPECTED,'source_bytes_read':'Entire current A.7 raw bytes, lines 223–253, including unchanged paragraphs and fence',
 'intro':{'applicable':False,'reason':'A.7 is not the chapter first section','sha256':None},
 'figure_sha256':{},'current_section_has_image_references':False,
 'prior_opaque':{'artifact_id':'a7_prior_report_opaque','path':prior_path.relative_to(ROOT).as_posix(),
                 'sha256':digest(prior_path),'source_sha256':prior['source_sha256'],
                 'verdict_preserved':prior['verdict'],'issues_preserved':prior['issues']},
 'actual_changed_claim':'Opening adds P=105 after history+30; no nonnegative answer budget can make input105 fit C100; preserving G25 requires input removal30.',
 'changed_claim_id':'manual_budget',
 'needed_current_context':{'artifact_id':'a7_current_context_7_19','path':PREFIX+'/current-context-7.19.md',
   'sha256':digest(HERE/'current-context-7.19.md'),'locator':'course/chapters/07.md#7.19, complete section beginning line687',
   'read_scope':'Complete current 7.19, personally reread; shared input/output capacity and explicit full-history scope',
   'unchanged_against_own_prior_snapshot':True},
 'current_raw_result_read_scope':'Only full byte hash of original rag.json; no new JSON leaf reading required for unchanged empirical claims.',
 'original_raw_leaf_scope_reused':'Own execution/independent-audit.json/inspected_json_pointers (all 84 explicitly listed raw samples and necessary provenance) preserved and SHA-matched.',
 'primary_source_locator_reuse':[
   {'source_id':s['id'],'url':s.get('url'),'version':s.get('version'),
    'original_personal_inspection_locator':s.get('inspection_note'),
    'reuse':'Own independently read original evidence; supporting original snapshots SHA-verified; no refetch/rereading claimed.'}
   for s in prior['sources'] if s['kind'] in ('paper','official_source','official_docs')],
 'own_prior_artifacts_checked':result['prior_artifact_hash_checks'],
 'current_live_code_and_original_measurement_hashes':result['live_code_and_raw_hash_checks'],
 'unchanged_tool_contracts':[
   {'path':p,'sha256':digest(ROOT/p),'same_as_own_saved_contract':True}
   for p in ['scripts/check_technical_reviews.py','docs/review-tools/section_facts.py']],
 'new_proof':{'artifact_id':'a7_changed_budget_execution','path':PREFIX+'/changed-budget-result.json',
              'sha256':digest(HERE/'changed-budget-result.json'),'arithmetic':result['arithmetic'],
              'environment':result['environment'],'exit_code':0},
 'actual_new_evidence_artifacts':[{'id':a['id'],'path':a['path'],'sha256':a['sha256']} for a in new_artifacts],
 'visual_inspection':{'read_scope':'Only changed opening paragraph from http://127.0.0.1:8765/A.7.html',
   'artifact_id':'a7_changed_render_execution','path':PREFIX+'/changed-paragraph-render.json',
   'sha256':digest(HERE/'changed-paragraph-render.json'),'both_saved_pngs_personally_viewed':True},
 'history':'Own prior report/claims/issues/proof preserved opaque. New current wording resolves the possible implication that reducing the answer budget alone can fix P>C; old numeric check only reported combined130. New proof directly checks input105, zero-answer excess5 and deletion30 while retaining25. No author revision notes, third-reader reports or other reviewers judgments were read.',
 'verdict':'pass','unresolved_substantive_issues':[],
 'limitations':'Only the changed arithmetic is newly executed. Prior raw measurements, implementation branches and authority sources are explicitly reused after exact hashes. No full CPU audit, mature-model evaluation, paper download, GPU, training, model/training-data download or unrelated whole-page inspection.'}
receipt_path=HERE/'reinspection-receipt.json'
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
artifact('a7_same_owner_reinspection_receipt','reinspection-receipt.json','derivation',
         'Canonical same-owner current-version reinspection receipt with actual source/context/read scopes, proof and exact artifact IDs/paths/hashes')

report=prior
report['source_sha256']=EXPECTED
report['verdict']='pass'
report['read_scope']=['Complete current A.7 raw bytes, lines 223–253',
                      'Necessary current 7.19 full section; exact bytes unchanged',
                      'New AST literal arithmetic and actual changed-paragraph render',
                      'Unchanged original sources/code/raw measurements verified by exact hashes and own prior genuine evidence explicitly reused']
for a in report['artifacts']:
 if a['id']=='section':a['description']='Preserved original-review A.7 raw bytes for prior source SHA ed3ccf9b…; current source snapshot is a7_current_section'
 if a['id']=='extraction':a['description']='Preserved original-review extraction receipt and original hashes, not a current-section extraction'
 if a['id'] in ('desktop','mobile','render_receipt','render_html'):
  a['description']='Preserved original-review rendering evidence; current changed paragraph uses a7_changed_render_execution and its screenshots'
report['artifacts']+=new_artifacts
report['sources'].append({'id':'changed_budget_cpu','kind':'execution',
 'title':'Same-owner current A.7 input-over-capacity arithmetic verification','verified':True,
 'artifact_id':'a7_changed_budget_execution'})
claim=next(c for c in report['claims'] if c['id']=='manual_budget')
claim['statement']='The original arithmetic example gives input75 and answer reserve25,total100. Adding30 history makes input105 and combined130: even zero new output leaves excess5; preserving25 answer positions requires at least30 input positions removed. Changing retrieval40→10 then returns input75,total100.'
claim['location']='course/chapters/0A.md lines 225,230–241,244'
claim['evidence'].append({'source_id':'changed_budget_cpu','locator':'changed-budget-result.json/arithmetic; verify_changed_budget.py AST literals and integer assertions',
 'supports':'Independently checks the new input-only excess, proof for every nonnegative G, and minimum removal retaining G25.'})
claim['artifact_ids']+=['a7_current_section','a7_actual_delta','a7_changed_budget_code','a7_changed_budget_execution','a7_same_owner_reinspection_receipt']
claim['verification']={'method':'executed','expected':'BaseP75,G25,C100; afterhistory+30,P105;G0stillover5; preservingG25requires30inputdeletions;29deletionsleaves101;30leaves100.',
 'observed':'New bounded CPU arithmetic exactly agrees: P105>C100 for every G≥0, min removal5 at theoretical zero-output and30 retainingG25; all assertions exit0. Original unchanged fence75/25/100/130/30 proof is explicitly reused.',
 'details':'Current fence bytes SHA-match prior exact execution; only new arithmetic was executed using AST literal extraction, including 26 reserves0–25 plus monotonic proof P+G≥P105>C100 for all nonnegativeG. Unchanged original code/raw/primary evidence SHA-matches; no broad rerun or downloads.',
 'tolerance':'Exact integer equality, no rounding.'}
for c in report['claims']:
 c['current_reinspection']={'same_owner':TASK,'changed':c['id']=='manual_budget',
   'evidence_policy':'New proportional arithmetic proof for manual_budget; all other own original evidence explicitly SHA-verified and reused.',
   'receipt_artifact_id':'a7_same_owner_reinspection_receipt'}
report['checks']['factual_accuracy']['details']='All five original claim groups retained. Complete current section personally read; changed input105/excess5/deletion30 claim directly verified with bounded CPU proof; unchanged claims supported by exact-hash reused own evidence.'
report['checks']['numeric_verification']['details']='New integer proof: history+30 gives P105; even G0 exceeds100 by5; preservingG25 needs30 input deletions. Previous exact fence and84-sample raw calculations are unchanged and explicitly reused after hash checks, not newly rerun.'
report['checks']['figure_consistency']['details']='A.7 still has no image/SVG references and figure hashes{}. Actually rendered and personally viewed only changed opening paragraph from supplied current local page at1280×800 and390×844; prior unchanged section rendering preserved.'
report['checks']['source_verification']['details']='Latest factual method personally read. Original primary snapshots and all31 own artifacts verified by exact hashes;6 live code/raw files match own snapshots; current7.19 context unchanged and personally reread. No refetch or reliance on author revision notes/third-reader/other reviewers judgments.'
report['reinspection_history']=report.get('reinspection_history',[])+[{
 'reviewer_task':TASK,'same_original_owner':True,'prior_source_sha256':receipt['prior_opaque']['source_sha256'],
 'source_sha256':EXPECTED,'prior_opaque':receipt['prior_opaque'],
 'receipt_artifact_id':'a7_same_owner_reinspection_receipt','receipt_path':receipt_path.relative_to(ROOT).as_posix(),
 'receipt_sha256':digest(receipt_path),'changed_claim_ids':['manual_budget'],'unresolved_issues':[]}]
target=ROOT/'docs/technical-reviews/A.7.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
saved=json.loads(target.read_bytes())
assert saved['reviewer_task']==TASK and saved['source_sha256']==EXPECTED and saved['verdict']=='pass'
assert saved['reinspection_history'][-1]['receipt_sha256']==digest(receipt_path)
print(json.dumps({'report':target.relative_to(ROOT).as_posix(),'report_sha256':digest(target),
 'source_sha256':EXPECTED,'intro_sha256':None,'figure_sha256':{},
 'prior_opaque':receipt['prior_opaque'],'receipt_id':'a7_same_owner_reinspection_receipt',
 'receipt_path':receipt_path.relative_to(ROOT).as_posix(),'receipt_sha256':digest(receipt_path),
 'verdict':'pass','unresolved_issues':[]},ensure_ascii=False,indent=2))
