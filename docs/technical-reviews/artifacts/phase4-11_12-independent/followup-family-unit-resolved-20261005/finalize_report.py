from pathlib import Path
import datetime
import hashlib
import json
import re

ART=Path(__file__).resolve().parent
ROOT=ART.parents[4]
sha=lambda raw:hashlib.sha256(raw).hexdigest()
target=ROOT/'docs/technical-reviews/11.12.json'
prior=target.read_bytes()
assert sha(prior)=='ad8a5d658e3ec48cf9bf467180012e4e7ee4fa3104e88f6a48563774a6bfb9d8'
report=json.loads(prior) # Only my own preserved revise report.
current=json.loads((ART/'current-verification.json').read_bytes())
render=json.loads((ART/'current-artifact-render-receipt.json').read_bytes())
snapshot=json.loads((ART/'current-input-snapshot-receipt.json').read_bytes())
for name in ['current-verification-execution.json','current-render-execution.json']:
    assert json.loads((ART/name).read_bytes())['exit_code']==0
receipt=json.loads((ART/'formal-recheck-receipt-base.json').read_bytes())
assert receipt['source_sha256']==current['current_source_sha256']==snapshot['section_sha256']==render['source_sha256']
receipt['checked_at']=datetime.datetime.now(datetime.UTC).isoformat()
receipt['proof_file_hashes']={name:sha((ART/name).read_bytes()) for name in ['current-section.md','current-verification.json','current-verification-execution.json','current-artifact-render-receipt.json','current-render-execution.json','bounded-current-section.html','desktop-current-artifact.png','mobile-current-artifact.png','served-page-before-recheck.html']}
formal=ART/'formal-recheck-receipt.json'
formal.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')

artifact_ids={}
for p in sorted(ART.iterdir()):
    if not p.is_file():continue
    identifier='family_resolved_'+re.sub(r'[^A-Za-z0-9_]','_',p.name)
    if p.name=='formal-recheck-receipt.json':identifier=receipt['id']
    artifact_ids[p.name]=identifier
    kind='code' if p.suffix=='.py' else ('figure_render' if p.suffix=='.png' else 'source_snapshot')
    item={'id':identifier,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'kind':kind,
          'description':'Actual complete-current-section family-unit recheck evidence: '+p.name+'. Non-production render/public stale parity boundaries are explicit.'}
    if p.name=='current-verification-execution.json':
        item.update(kind='execution',command='.venv/bin/python '+(ART/'recheck_current.py').relative_to(ROOT).as_posix(),
                    result='Exit0 in0.115s; all65 prior artifact hashes, original inputs, exact current fence/SVG checked; raw string/character coverage matches amended source; no model/torch imported.',environment=current['environment'])
    if p.name=='current-render-execution.json':
        item.update(kind='execution',command='.venv/bin/python '+(ART/'render_current_artifact.py').relative_to(ROOT).as_posix(),
                    result='Exit0; bounded raw-current-source plus exact originalSVG artifact rendered and personally viewed desktop/mobile. All8paragraphs/fence/SVGbytes match; common8765page remains stale and failure is preserved.',environment={'python':render['python'],'chromium':render['chromium'],'device':'CPU; GPUdisabled','mode':'non-production bounded artifact'})
    report['artifacts'].append(item)

report['sources'] += [
    {'id':'current_family_unit_recheck','kind':'execution','title':'Complete revised source reread, unchanged-proof hash audit and current raw family/character coverage','verified':True,'artifact_id':artifact_ids['current-verification-execution.json']},
    {'id':'current_bounded_artifact_render','kind':'execution','title':'Current subsection bounded artifact rendering, explicitly non-production','verified':True,'artifact_id':artifact_ids['current-render-execution.json']}
]
# Old artifacts are retained unchanged. Narrow their report descriptions rather than rewriting historical proof bytes.
for a in report['artifacts']:
    if a['id']=='original_section_md':a['description']='Initial raw subsection d841... frozen history for original fence extraction; not the current changed paragraphs.'
    if a['id'] in {'desktop_page_png','mobile_page_png','desktop_page_html','mobile_page_html'}:
        a['description']+=' Initial d841... page only; current-text parity is not inferred from this historical file.'
    if a['id']=='render_execution_json':
        a['result']='Original initial-source page render exit0, matching the oldd841... text and exactSVG. Source text changed; original paragraphs parity is historical only. UnchangedSVG appearance remains supported; current text is separately viewed in non-production bounded artifact.'

family=next(c for c in report['claims'] if c['id']=='family_holdout')
family['status']='verified'
family['statement']='Identity-preserving variants keep their label and source image family together; with one original per class, holding out that entire source also holds out the class. The0-99 experiment instead groups complete strings with their offsets.'
family['scope']='The two0/1fixtures only establish pixel-label pairing. To assess new original images of known classes, use multiple independent source images per class while keeping each source and its variants together and retaining class coverage. Actual historical split is whole digit strings, uses familiar0-9characters and one fixed font; it does not assess new character classes, fonts or handwriting.'
family['evidence'].append({'source_id':'current_family_unit_recheck','locator':'current-verification.json#/two_fixture_holdout, #/coverage and #/original_method_reread; formal-recheck-receipt.json#/actual_reread','supports':'Independently rechecks the amended whole-section scope against exact two-fixture class coverage, original full-string grouping and raw familiar-character coverage.'})
family['artifact_ids'] += [artifact_ids['current-verification-execution.json'],artifact_ids['current-verification.json'],receipt['id']]
for claim_id in ['historical_fixed_font_result','historical_split_unit_and_class_coverage']:
    c=next(c for c in report['claims'] if c['id']==claim_id)
    c['evidence'].append({'source_id':'current_family_unit_recheck','locator':'current-verification.json#/coverage, #/unchanged_original_inputs and #/honest_reuse','supports':'Current text explicitly matches same-SHA original full-string family and familiar-character/fixed-font evidence; existing raw2/30provenance and score reused only for unchanged inputs.'})
    c['artifact_ids'] += [artifact_ids['current-verification-execution.json'],receipt['id']]
visual=next(c for c in report['claims'] if c['id']=='glyph_numbers_and_exercise')
visual['evidence'].append({'source_id':'current_bounded_artifact_render','locator':'current-artifact-render-receipt.json#/views; formal-recheck-receipt.json#/actual_visual_inspection','supports':'Personally viewed current raw-section artifact plus exact unchangedSVG. This is explicitly non-production; stale commonpage parity is not represented as current.'})
visual['artifact_ids'] += [artifact_ids['current-render-execution.json'],artifact_ids['desktop-current-artifact.png'],artifact_ids['mobile-current-artifact.png']]

issue=report['issues'][0]
assert issue['id']=='family-unit-scope'
issue['first_observed_source_sha256']=receipt['old_source_sha256']
issue['status']='resolved'
issue['resolution']=receipt['issue_recheck']['resolution']
issue['actual_recheck']=receipt['issue_recheck']['actual_recheck']
issue['recheck_source_sha256']=receipt['source_sha256']
issue['recheck_receipt_artifact_id']=receipt['id']
issue['original_issue_receipt']='docs/technical-reviews/artifacts/phase4-11_12-independent/followup-family-unit-20261005/issue-reread-receipt.json'

report['source_sha256']=receipt['source_sha256']
report['reviewed_at']=receipt['checked_at']
report['verdict']='pass'
report['earlier_frozen_inputs']=[report['frozen_input']]
report['frozen_input']={'path':snapshot['frozen_input_path'],'sha256':snapshot['frozen_input_sha256'],'captured_at':snapshot['captured_at'],'scope':snapshot['frozen_input_scope'],'receipt_artifact_id':artifact_ids['current-input-snapshot-receipt.json']}
report['read_scope']['course/chapters/11.md']='Complete current11.12 reread at2c27af...; initial11.12 and adjacent11.13reads are preserved historical scope. Wholechapter frozen SHA is dated input only; no full current chapter review claim.'
report['current_recheck']={'formal_receipt_artifact_id':receipt['id'],'formal_receipt_path':formal.relative_to(ROOT).as_posix(),'formal_receipt_sha256':sha(formal.read_bytes()),
                           'prior_pass_sha256':receipt['prior_pass_sha256'],'prior_revise_sha256':receipt['prior_revise_sha256'],'prior_revise_history':receipt['prior_revise_history'],
                           'honest_reuse':receipt['honest_evidence_reuse'],'production_page_boundary':receipt['actual_visual_inspection']['production_parity_failure']}
report['checks']['factual_accuracy']={'status':'pass','details':'Complete revised11.12 personally reread. Previously open family-unit-scope issue resolved by explicit sole-source class holdout, independent known-class sources and actual complete-string/familiar-character/fixed-font distinctions, each independently matched to original method/raw data. Other original claims retain exact-input support.','claim_ids':[c['id'] for c in report['claims']]}
report['checks']['numeric_verification']['details']='Initial original fence/grid/RGB/gradient/scoring numbers reused after personally checking all65prior artifacts and unchanged inputSHA. Current relevant raw string/character coverage additionally reread/recalculated with no model import:80/10/10string families,train0-9,test0unseen character classes,3offsets perstring;2/30 score remains historical.'
report['checks']['figure_consistency']={'status':'pass','details':'UnchangedSVG hash and exact fence bytes checked. Initial30-cell/grid/one-pixel numeric and visual evidence valid for unchanged diagram. Current revised text plus exactSVG bounded artifact independently rendered and viewed at1280x800/390x844;8paragraphs/fence/SVGbytes exact. Artifact is non-production; stale8765page parity failure retained and no current-production claim made.','claim_ids':['pixel_label_example','glyph_numbers_and_exercise','family_holdout','historical_split_unit_and_class_coverage']}
report['checks']['limitations']={'status':'pass','details':'Current text bounds known-class source-image testing versus full-string generation and explicitly excludes new digit classes/fonts/handwriting for the historical fixed-font experiment. InitialCPU proof not rerun; only exact unaffected parts reused, with all originalSHA checked. Current bounded artifact is not production-page validation;8765stale-parity failure remains recorded.','claim_ids':[c['id'] for c in report['claims']]}
report['checks']['source_verification']['details']+=' Complete revised-source recheck used original same-SHA method and named raw pointers plus reread official grouping section; new formal receipt records source units, history and evidence reuse.'
report['verification_boundary']=receipt['remaining_boundaries']
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_11_12' and report['reviewer_context']=='fresh'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':target.relative_to(ROOT).as_posix(),'verdict':report['verdict'],'source_sha256':report['source_sha256'],'report_sha256':sha(target.read_bytes()),
                  'formal_receipt_artifact_id':receipt['id'],'formal_receipt_path':formal.relative_to(ROOT).as_posix(),'formal_receipt_sha256':sha(formal.read_bytes())},ensure_ascii=False,indent=2))
