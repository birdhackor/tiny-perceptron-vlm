"""Publish this factual owner's completed, scoped correction recheck."""
from pathlib import Path
import copy, hashlib, json

ROOT = Path.cwd()
OUT = Path(__file__).parent
BASE = OUT.parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, value): (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
before = json.loads((OUT/'before-report.json').read_text())
assert sha(OUT/'before-report.json') == '9f17ba1a43585b3c70674073ebd3f8e7d3c152a7d3dfa5e4c5305519de1ba15b'
assert (ROOT/'docs/technical-reviews/T.8.json').read_bytes() == (OUT/'before-report.json').read_bytes()
reuse = json.loads((OUT/'reuse-byte-comparison.json').read_text())
navigation = json.loads((OUT/'browser-navigation.json').read_text())
assert json.loads((OUT/'recheck.execution.json').read_text())['native_exit_code'] == 0
assert all(a['same_retrieved_bytes'] for a in reuse['local_original_authorities'])

research = ROOT/'outputs/natural-v4/factual-research/T.8'
additional = []
for receipt_name, filename in [('adam-source-retrieval.json','adam_source.py'), ('tied-source-retrieval.json','tied.pdf')]:
    receipt = json.loads((BASE/receipt_name).read_text())
    actual = sha(research/filename)
    assert actual == receipt['sha256']
    additional.append({'url':receipt['url'],'local_retrieval_sha256':actual,'same_retrieved_bytes':True})
for receipt in json.loads((BASE/'original-code-retrieval.json').read_text()):
    actual = sha(research/f"architecture-original-{receipt['experiment']}.py")
    assert actual == receipt['recorded_sha256'] == receipt['retrieved_sha256']
    additional.append({'experiment':receipt['experiment'],'revision':receipt['revision'],'recorded_path':receipt['path'],'local_retrieval_sha256':actual,'same_retrieved_bytes':True})
save('additional-retained-byte-comparison.json', additional)

failures=[]
for ordinal, explanation in [('first','Own recheck code concatenated str and bytes in one assertion; fixed to bytes. No source finding.'),('second','Own browser attempt assumed /chapters/16.html; preview publishes separate 16.1.html. Correct path discovered from the rendered assigned-section link. No source finding.')]:
    receipt=json.loads((OUT/f'recheck.{ordinal}.execution.json').read_text())
    failures.append({'attempt':ordinal,'native_exit_code':receipt['native_exit'],'explanation':explanation,'preserved_files':[{ 'path':str((OUT/f'recheck.{ordinal}.{suffix}').relative_to(ROOT)), 'sha256':sha(OUT/f'recheck.{ordinal}.{suffix}')} for suffix in ['py','stdout.txt','stderr.txt','execution.json']]})
failures.append({'attempt':'report updater first','native_exit_code':1,'explanation':'Own updater expected native_exit_code while earlier attempt receipts used native_exit; preserved unchanged failing code and actual repeated subprocess capture before fixing the lookup. No canonical report had been changed.','preserved_files':[{'path':str((OUT/f'update_report.first.{suffix}').relative_to(ROOT)),'sha256':sha(OUT/f'update_report.first.{suffix}')} for suffix in ['py','stdout.txt','stderr.txt','execution.json']]})

notes={
    'owner':before['reviewer_task'],
    'actual_source_revision':'160fd47dede5c2453c8a08dd535290ddfd0b915d',
    'date':'2026-10-05',
    'own_full_reread':['Complete current raw UTF-8 T.8, including blank lines through the next ## boundary; no owned introduction.','Complete necessary current 16.1, including its exercise and changed link; complete new optional efficiency-condition subsection within T.8.'],
    'scope_comparison':{'T.8':'Exactly one bold-label conversion to an H3 heading; all other factual prose, commands, metrics, links and whitespace identical.','16.1':'Exactly one destination fragment replacement from #T.8 to #選讀效率實驗的量測條件; all other prose/code/numbers identical.'},
    'preserved_prior_judgment':{'report_sha256':sha(OUT/'before-report.json'),'verdict':before['verdict'],'issues_count':len(before['issues']),'path':str((OUT/'before-report.json').relative_to(ROOT))},
    'personal_browser_inspection':{'preview':'http://127.0.0.1:8789/','actual_click':True,'start_url':navigation['start_url'],'final_url':navigation['final_url'],'personally_viewed':['navigation-before-click.png','navigation-destination.png'],'observation':'The source page visibly offers T.8的效率量測方法 after its MiB and timing explanation. Clicking opens the new optional H3 near the top of the viewport. Its first five paragraphs visibly cover fixed L4/PyTorch/FP32/TF32/SFT conditions, six reset branches versus missing packing fields, allocated-tensor scope/exclusions, MiB, and synchronization/warmup. This is the promised measurement-method destination.'},
    'evidence_reuse':'Own prior substantive factual judgments, original-source reading, CPU execution, original CUDA-record audits and personal figure views retained after byte comparisons. All 18 registered implementation/record sources, 21 prior artifacts, five canonical figures and 24 prerequisite sections checked; only the 16.1 navigation fragment changed. All 18 retained external original authority files and five historical architecture files also match retrieval receipts. No fresh full original download, GPU benchmark or full training claimed.',
    'registered_current_markdown_scope':'Whole-file SHA is a checksum only. Newly registered Markdown inspection covers T.8 and 16.1, not every section of either full file.',
    'current_environment':{'python':'3.13.5','torch':'2.14.1+cpu','cuda_available':'false','browser':'Installed /usr/bin/chromium through existing Playwright'},
    'failed_attempts':failures,
    'limits':'No CUDA hardware is available. Original fixed GPU records remain an audited prior source, not own GPU replication. Literal official 2.14 max_memory_allocated URL failure and exact-version official-source fallback remain preserved. No peer verdict or author factual assessment was read.',
    'verdict':'pass','issues':[]
}
save('owner-inspection.json',notes)

r=copy.deepcopy(before)
r['source_sha256']=sha(OUT/'current-section.raw.md')
r['scope']='Complete current assigned T.8 and necessary 16.1 reread after the scoped heading/link correction; no owned introduction. Unchanged substantive evidence reused only after byte comparison; real preview navigation and destination personally inspected. Own CPU execution only.'
r['prerequisite_sha256']={p['source']:p['current_sha256'] for p in reuse['prerequisites']}
r['prerequisite_snapshot_paths']={p['source']:str((OUT/'current-prerequisite-16.1.md' if p['navigation_changed'] else BASE/('prerequisite-'+p['source'].split('#')[1]+'.md')).relative_to(ROOT)) for p in reuse['prerequisites']}
r['correction_recheck']={'source_revision':notes['actual_source_revision'],'prior_report_sha256':notes['preserved_prior_judgment']['report_sha256'],'prior_source_sha256':before['source_sha256'],'substantive_verdict_preserved':True,'current_inspection_artifact':'correction_inspection','execution_artifact':'correction_execution','evidence_reuse_artifact':'correction_reuse'}
for artifact in r['artifacts']:
    if artifact['id']=='section':
        artifact.update(path=str((OUT/'current-section.raw.md').relative_to(ROOT)),sha256=r['source_sha256'],description='Complete current assigned raw UTF-8 T.8 including blank lines; personally reread after heading correction, no owned introduction.')

def artifact(id,filename,kind,description,**extra):
    p=OUT/filename
    r['artifacts'].append({'id':id,'kind':kind,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'description':description,**extra})
artifact('correction_before_report','before-report.json','source_snapshot','Exact preservation of this owner\'s prior completed pass judgment, before current correction recheck.')
artifact('correction_before_section','before-section.raw.md','source_snapshot','Exact original assigned section bytes preserved before correction; prior judgment remains attributable to these bytes.')
artifact('correction_before_16_1','before-prerequisite-16.1.md','source_snapshot','Exact own prior necessary 16.1 snapshot retained before its link-fragment change.')
artifact('correction_current_16_1','current-prerequisite-16.1.md','source_snapshot','Complete current necessary 16.1 raw UTF-8, personally reread and clicked in actual preview.')
artifact('correction_code','recheck.py','code','Own bounded byte-scope/evidence comparisons and actual browser click/destination checks.')
artifact('correction_execution','recheck.stdout.txt','execution','Actual successful owner correction recheck; no GPU execution.',command=json.loads((OUT/'recheck.execution.json').read_text())['command'],result='Native exit 0; exact scoped changes, unchanged evidence and actual 16.1→new optional H3 navigation verified.',environment=notes['current_environment'])
artifact('correction_stderr','recheck.stderr.txt','source_snapshot','Actual successful correction stderr (empty).')
artifact('correction_receipt','recheck.execution.json','source_snapshot','Successful correction command, native exit and environment receipt.')
artifact('correction_reuse','reuse-byte-comparison.json','source_snapshot','Own current/prior byte comparison of registered source, artifact, figure, prerequisite and retained external-original evidence.')
artifact('correction_additional_reuse','additional-retained-byte-comparison.json','source_snapshot','Own byte comparison of retained original Adam/tying authorities and original-run historical architecture files.')
artifact('correction_navigation','browser-navigation.json','source_snapshot','Actual clicked link, destination H3, rendered paragraphs, viewport position and semantic inspection.')
artifact('correction_navigation_before','navigation-before-click.png','figure_render','Personally viewed actual 16.1 browser screenshot before clicking the measurement-method link.')
artifact('correction_navigation_after','navigation-destination.png','figure_render','Personally viewed actual new optional measurement-condition H3 and its rendered content after the click.')
artifact('correction_inspection','owner-inspection.json','source_snapshot','Own complete reread/scope/semantic observations, precise evidence reuse, preserved failed attempts and limits.')
artifact('correction_update_code','update_report.py','code','Own scoped report update after completed reread, byte comparisons and personally viewed real navigation.')

for id,path,inspection in [
    ('current_training_markup','course/training.md','Personally read complete current raw T.8 through next ## including the new optional H3; byte-compared the full assigned section with own prior snapshot. Whole-file checksum does not imply full-book reading.'),
    ('current_16_1_markup','course/chapters/16.md','Personally read complete necessary current 16.1 including the changed measurement-method link; byte-compared with own prior snapshot and actually clicked its preview counterpart. Whole-file checksum does not imply all Chapter 16 reread.')]:
    r['sources'].append({'id':id,'kind':'repository_code','title':path+' (scoped Markdown source inspection)','path':path,'sha256':sha(ROOT/path),'version':'Current working tree at 160fd47dede5c2453c8a08dd535290ddfd0b915d, owner correction recheck 2026-10-05','verified':True,'inspection_note':inspection})
c7=next(c for c in r['claims'] if c['id']=='c7')
c7['location']='T.8 ### 選讀：效率實驗的量測條件, specifically its first five paragraphs; reached from the current 16.1 measurement-method link.'
c7['artifact_ids']+=['correction_inspection','correction_execution','correction_navigation','correction_reuse']
c7['evidence']+=[{'source_id':'current_training_markup','locator':'T.8 new optional measurement-condition H3 and first five paragraphs','supports':'Current location and unchanged measurement conditions.'},{'source_id':'current_16_1_markup','locator':'16.1 MiB/measurement-method paragraph','supports':'Current explicit entry link points to the measurement conditions personally inspected.'}]
r['checks']['factual_accuracy']['details']+=' Correction recheck: complete current T.8/16.1 reread and actual new-heading destination inspected; unchanged facts and prior judgment retained.'
r['checks']['numeric_verification']['details']+=' Correction changes no numeric claim; current records/code/artifacts exactly match the prior executed review, so no redundant training rerun.'
r['checks']['figure_consistency']['details']+=' Correction recheck confirms all five canonical SVG and own render bytes unchanged; prior personal views reused. New source/destination browser screenshots also personally viewed.'
r['checks']['source_verification']['details']+=' Current Markdown inspected only for assigned T.8/necessary 16.1; exact one-heading/one-fragment scope verified and newly registered; prior originals and implementation/record evidence byte-identical.'
r['checks']['limitations']['details']+=' Correction recheck uses real preview navigation and byte evidence; no new GPU run or external-original retrieval claimed.'
(ROOT/'docs/technical-reviews/T.8.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'actual_source_revision':notes['actual_source_revision'],'current_source_sha256':r['source_sha256'],'current_16_1_sha256':sha(OUT/'current-prerequisite-16.1.md'),'report_sha256':sha(ROOT/'docs/technical-reviews/T.8.json'),'verdict':r['verdict'],'issues_count':len(r['issues']),'claims_count':len(r['claims']),'new_navigation_claim_atoms':0},ensure_ascii=False))
