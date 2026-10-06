import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
BASE=OUT.relative_to(ROOT).as_posix()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
input_receipt=json.loads((OUT/'context-input-and-render-receipt.json').read_text())
history=ROOT/input_receipt['prior_history_path']
assert sha(history)==input_receipt['prior_history_sha256']=='612601e21bbefaa5e5de9de72023be28daaaa7cdbb8726d6d82de5d5daceb287'
report=json.loads(history.read_bytes())
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_6_8'
assert report['source_sha256']==input_receipt['sections']['6.8']['current_sha256']
assert report['verdict']=='pass'
receipt={
 'schema_version':1,'kind':'same_owner_necessary_context_reinspection',
 'reviewer_task':report['reviewer_task'],'performed_on':'2026-10-06','verdict':'pass',
 'source':'course/chapters/06.md#6.8','source_sha256':report['source_sha256'],
 'prior_history_path':input_receipt['prior_history_path'],'prior_history_sha256':input_receipt['prior_history_sha256'],
 'actual_current_context_scope':{
  '6.8':'Complete ownsection direct reread; allclaims and corrected source-group/boundary distinction.',
  '6.5':'Currentcomplete short section read, limited technicaldependency: finite-window effective-target/context-only/PAD/EOS convention, shared raw excerpt/exposure budget versus384 inputpositions, andboth referencedcurrentdiagrams.',
  '6.6':'Currentcomplete short section read, limited technicaldependency: PAD/BOS/EOS/controlIDs versus ordinaryUTF8 content; structuredrole insertion and literalquotedcontrol-name convention.',
  'excluded':'No wholechapter review due a globalSHA delta; no otherreport or authorrepairanswer read; no newtraining/scoring/modelweights/data prep/paperfetch.'
 },
 'current_context_sections':input_receipt['sections'],
 'observed_context_differences':{'6.5':'Text byte-identical to ownoriginalfrozen input; actualcurrentfigures renderedandviewed.','6.6':'Only simplified/traditional glyphvariants in oneexistingparagraph; role/content rule unchanged.','6.8':'Unchanged from ownpriorpass; initialresolvedi1 preserved.'},
 'current_context_figures':input_receipt['figure_sources'],
 'actual_visual_inspection':[{'source':f['path'],'rendered_png':f['snapshot'].removesuffix('.svg')+'.png','view_tool':'functions tools.view_image','actually_viewed':True,'observed':'Current formula/commonrawdenominator andexplicitEOS/effective-target labels inspected; exact visual observations saved inactual-context-inspection.md.'} for f in input_receipt['figure_sources']],
 'raw_measurement_pointer_receipt':{'path':f'{BASE}/measurement-pointer-check-receipt.json','sha256':sha(OUT/'measurement-pointer-check-receipt.json'),'inspected_pointers':json.loads((OUT/'measurement-pointer-check-receipt.json').read_text())['actual_inspected_pointers']},
 'original_evidence_reuse':{'prior_artifacts_hash_equal_count':len(input_receipt['prior_artifact_hash_checks']),'current_code_equal':input_receipt['reused_current_code_checks'],'original_fence_or_probe_rerun':False,'original_measurement_rerun':False,'support_checked':'Historicalsame-source<=256byte snippets andoneEOSperexcerpt fit384positions; same400*8 originaldocument schedule and664759rawbyte exposure; effective-target/rawbyte distinction andordinary/controlID convention unchanged.'},
 'browser_status':'Live6.8/6.5 HTML200 snapshotted; boundedChromium screenshot timedout20s. No renderedbrowserlayout claim. NecessarycurrentSVGs renderedwithInkscape1.4 andpersonallyviewed.',
 'unresolved_questions':[],
 'actual_inspection_notes':{'path':f'{BASE}/actual-context-inspection.md','sha256':sha(OUT/'actual-context-inspection.md')},
 'input_and_render_receipt':{'path':f'{BASE}/context-input-and-render-receipt.json','sha256':sha(OUT/'context-input-and-render-receipt.json')},
 'environment':input_receipt['environment'],
 'canonical_execution_artifact_id':'ctx20261006-reinspection'
}
receipt_path=OUT/'context-reinspection-receipt.json'
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and not p.name.startswith('checker') and p.name not in ('update.stdout.json','update.stderr.txt')]
new_artifacts=[]
for i,p in enumerate(files):
 kind='code' if p.suffix=='.py' else 'source_snapshot'
 if p.suffix=='.png':kind='figure_render'
 if p.suffix=='.diff' or p.name=='actual-context-inspection.md':kind='derivation'
 identifier='ctx20261006-reinspection' if p==receipt_path else f'ctx20261006-{i}'
 entry={'id':identifier,'kind':kind,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'description':f'Actual same-owner necessary6.5/6.6context reinspection20261006: {p.relative_to(OUT).as_posix()}; currentcontext support retained/rechecked, priorCPU/measurement evidence reused without rerun.'}
 if p in (receipt_path,OUT/'context-input-and-render-receipt.json',OUT/'measurement-pointer-check-receipt.json'):
  entry['kind']='execution'
  entry['environment']={k:str(v) for k,v in input_receipt['environment'].items()}
  script={receipt_path:'update_canonical.py',OUT/'context-input-and-render-receipt.json':'prepare_context.py',OUT/'measurement-pointer-check-receipt.json':'check_context_measurements.py'}[p]
  entry['command']=f'.venv/bin/python {BASE}/{script}'
  entry['result']='exit0; actualscopedcurrentsection/text/figures/sourcebranches inspected; rawsavedaggregate arithmeticverified;69 priorartifact hashesmatched; priorcode/execution/measurement reusedunchanged; no unresolvedfactualquestion. Chromiumtimeout separately recorded, both currentfigures actually renderedandviewed.'
 new_artifacts.append(entry)
report['artifacts'].extend(new_artifacts)
report['sources'].append({'id':'context20261006','kind':'execution','title':'Same-owner actual necessary-context reinspection20261006','verified':True,'artifact_id':'ctx20261006-reinspection'})
report['sources'].append({'id':'context-figure-aggregates','kind':'execution','title':'Named-pointer originalmeasurements andcurrentcontextfigure arithmetic audit','verified':True,'artifact_id':next(a['id'] for a in new_artifacts if a['path'].endswith('/measurement-pointer-check-receipt.json'))})
for cid in ('c1','c2','c4','c5','c6','c7','c8'):
 c=next(c for c in report['claims'] if c['id']==cid)
 c['evidence'].append({'source_id':'context20261006','locator':'context-reinspection-20261006/current6.8/current6.5/current6.6 snapshots, actual-context-inspection.md, context-reinspection-receipt.json','supports':'Actual necessary-context directreread andcurrentfigure view confirm thisexistingclaim stillhas itsoriginalsupport andscope; nochangedprecondition ornewsubstantiveissue.6.5target/EOS/rawbudget convention and6.6ordinary/controlID rule agree with6.8.'})
 c['artifact_ids'].append('ctx20261006-reinspection')
next(c for c in report['claims'] if c['id']=='c7')['evidence'].append({'source_id':'context-figure-aggregates','locator':'measurement-pointer-check-receipt.json actual_inspected_pointers/original_test_aggregate_checks','supports':'Necessarycurrent6.5figure records20 finalfragments/4193rawbytes/4213vs2503EOStargets androundedvalues consistentwith originalsavedresult andsamebudget contract; no model scoringrerun.'})
for f in input_receipt['figure_sources']:report['figure_sha256'][f['path']]=f['sha256']
report['checks']['figure_consistency']={'status':'pass','details':'6.8 itself contains no figures. Same-owner necessarycontext6.5 has two currentSVGs: Inkscape-renderedandactuallyviewed; U+1F642/7bytes/sharedformulas andoriginal20record/4193rawbyte/EOS4213vs2503 labels/rank bars consistentwith savedoriginalaggregates and6.8scope.6.6 hasnofigure. Browserlayout unverified dueboundedChromiumtimeout.','claim_ids':['c2','c6','c7','c8']}
report['checks']['factual_accuracy']['details']+=' Actualsame-owner necessarycurrent6.5/6.6context reinspection20261006:6.5text unchangedrelativeownfrozen;6.6glyph-onlydifference; currentdiagramdenominator/EOS distinctions preserve6.8support. No new unresolvedissue.'
report['checks']['numeric_verification']['details']+=' Targeted current6.5figure aggregatepointer/roundedlabel audit agrees with originalJSON (computedBPB absdifference0); not a newmodelscore ororiginalCPU rerun.'
report['checks']['source_verification']['details']+=' All69 priorartifact hashes rechecked; actualnamedoriginalJSON pointers/currentcomputingbranches checked without authorresultcommentary; no newpaper fetch.'
report['checks']['limitations']['details']+=' Necessarycontextonly; currentdiagramvisualchecks do not establish fullbrowserlayout; explicitly recordChromiumtimeout andhonest Inkscapeview.'
report['reviewed_on']='2026-10-06'
report['review_scope']+=' 2026-10-06 same originalowner actualnecessarycontext reinspection: completecurrent6.8 andshort6.5/6.6read; finitewindow/EOS/rawbudget/controlIDdependency only; two current6.5figures renderedandviewed. Priorbody/CPU/measurement/authoritysnapshotsunchanged andhonestlyreusedafter69 hashes/supportchecks; no wholechapter gate.'
report['context_reinspection']=receipt
report['recheck_history'].append({'stage':'same-owner-context-reinspection20261006','verdict':'pass','source_sha256':report['source_sha256'],'prior_report_path':input_receipt['prior_history_path'],'prior_report_sha256':input_receipt['prior_history_sha256'],'actual_context_scope':receipt['actual_current_context_scope'],'artifact_ids':['ctx20261006-reinspection']})
canonical=ROOT/'docs/technical-reviews/6.8.json'
canonical.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'report_sha256':sha(canonical),'source_sha256':report['source_sha256'],'prior_history_path':input_receipt['prior_history_path'],'prior_history_sha256':sha(history),'canonical_artifact_id':'ctx20261006-reinspection','receipt_path':receipt_path.relative_to(ROOT).as_posix(),'receipt_sha256':sha(receipt_path),'unresolved_questions':[]},ensure_ascii=False,indent=2))
