"""This original reviewer rechecks the corrected section; preserve the initial report."""
from pathlib import Path
from datetime import datetime, UTC
import ast
import hashlib
import json
import re
import sys

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-14_1-independent'
RECHECK=BASE/'recheck'
TASK='/root/phase4_factual_coordinator/factual_14_1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def path(p):return p.relative_to(ROOT).as_posix()
def write(name,value):
    (RECHECK/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
initial=BASE/'initial-report.json'
assert sha(initial)=='38beede75b3dbf798d8a1325d7ada6513011dc3de2eb9c7de5a970f71bab51bb'
report=json.loads(initial.read_text())
assert report['reviewer_task']==TASK and report['verdict']=='revise'
original=(BASE/'section.md').read_bytes()
intro=(RECHECK/'intro.md').read_bytes()
section=(RECHECK/'section.md').read_bytes()
assert original.replace('下圖的虛線圓'.encode(),'下圖的圓'.encode())==section
assert len(original)-len(section)==6
assert intro==(BASE/'intro.md').read_bytes()
raw=(ROOT/'course/chapters/14.md').read_bytes()
heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 14.1 '))
assert raw[:heads[i].start()]==intro
assert raw[heads[i].start():heads[i+1].start()]==section
assert sha(RECHECK/'section.md')=='469732f5c61e87bea9ba4d1196ada837be9191ce9dcba643b877f872629d587c'
assert sha(RECHECK/'intro.md')=='c40390abbe944b64d0deb53e4fccb7c1d354d85520f05846e33297ee700ccb72'
def fences(body):return re.findall(rb'(?ms)^```python\r?\n(.*?)^```\s*$',body)
assert fences(section)==fences(original)
assert fences(section)==[(BASE/'original-fence.py').read_bytes()]

version_checks=[]
for a in report['artifacts']:
    assert sha(ROOT/a['path'])==a['sha256'],a['path']
    version_checks.append({'path':a['path'],'sha256':a['sha256'],'role':'initial original evidence, unchanged'})
for s in report['sources']:
    if s['kind']=='repository_code':
        assert sha(ROOT/s['path'])==s['sha256'],s['path']
    elif s['kind'] in ['paper','official_docs','official_source']:
        assert sha(ROOT/s['local_snapshot'])==s['snapshot_sha256'],s['local_snapshot']
manifest=json.loads((BASE/'snapshot-manifest.json').read_text())
for entry in manifest:
    assert sha(ROOT/entry['original_path'])==entry['sha256'],entry['original_path']
    version_checks.append({'path':entry['original_path'],'sha256':entry['sha256'],'role':'current original bytes unchanged since verified execution'})
for name,h in report['figure_sha256'].items():assert sha(ROOT/name)==h
current_arch=ROOT/'scripts/course_experiments/architecture.py'
assert sha(current_arch)=='7d91f0bce271d18cd3bfbc15e8c44fd7677c92fe5796a13835ec49ab426b3fc1'
historic_arch=BASE/'inputs/architecture-at-run.py'
assert sha(historic_arch)=='1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3'
names=['_copy_matching','_clone_config','_text_dataset','_nll','_train','_heldout','run_modern']
def methods(p):
    tree=ast.parse(p.read_text())
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
            if isinstance(n,ast.FunctionDef) and n.name in names}
assert methods(current_arch)==methods(historic_arch)
version_checks.append({'path':path(current_arch),'sha256':sha(current_arch),
                       'role':'current full hash differs from historical run as originally recorded; seven relevant computational methods are AST-equivalent to original run'})

# These are my judgments after personally rereading the whole new section and its intro,
# and personally viewing both original render images again in this recheck turn.
reassessments={
 'attention-roles':'New opening still defines Q/K roles and relative distance as before. Original attention paper and relative-position objective still support precisely this scope.',
 'rotary-mechanism':'New prose still explicitly fixes Q/K content. Orthogonal rotation and relative dot identity still support the fixed-content conclusion; adding real words is not claimed to hold whole-model contextual vectors fixed.',
 'pair-frequencies':'Hypothetical45-degree pair and actual multiple frequencies are unchanged. Original RoFormer frequency equation and original rope implementation still apply.',
 'trig-numbers':'All degrees/radians, signs, unit coordinates and450-degree common shift are unchanged. Previously executed numerical evidence and original API contracts remain version-matched.',
 'fence-and-exercises':'Full current Python fence is byte-identical to the twice-executed original, as are both exercises and printed numbers. It remains a fixed-number CPU illustration, with no update or training claim.',
 'negative-dot':'The negative-score/softmax distinction is unchanged; original attention equation and actual scaled implementation still support it. No negative probability is claimed.',
 'figure-geometry':'Personally re-viewed shared-rotation and component renders after proving both original SVG hashes unchanged. Corrected prose says「下圖的圓」and matches both actually solid circles.135-degree angles, shared scales, axes, arrow colors and component signs remain consistent.',
 'original-training-budget':'Optional paragraph still describes only the same historical TinyStories experiment,64features/two layers/four heads/240updates. Original raw provenance, reconstructed splits/sampling counters and exact historical methods remain unchanged; no rerun is claimed.',
 'byte-denominators':'Current text still identifies byte/EOS targets and distinguishes them from Chinese character count. Verified raw data, tokenizer/chunk methods and39256/41914 denominators are unchanged.',
 'heldout-losses':'Four displayed losses and their scope remain the same. Stored raw sums/denominators and original cross-entropy aggregation still support the historical average NLL values, without claiming general writing quality.',
 'parameter-counts':'128×64 table removal and141568/133376 counts are unchanged. Current models and original run method match the already counted original implementation; parameter inequality remains explicit.',
 'sample-and-limits':'Exact quoted prompt/sample and statement that short training does not establish good stories or>128 extrapolation remain unchanged. Recorded sample and actual128-position method support the same limited conclusion.'}
assert set(reassessments)=={c['id'] for c in report['claims']}
new_sentence=next(line for line in section.decode().splitlines() if line.startswith('下圖的圓'))
receipt={
 'reviewer_task':TASK,'reviewed_at_utc':datetime.now(UTC).isoformat(),
 'action':'Personally reread all revised14.1 and the actual opening intro, then personally viewed both existing original SVG render images again; independently checked current source/intro/figure/code/raw measurement/fence versions.',
 'source':report['source'],'old_source_sha256':report['source_sha256'],
 'new_source_sha256':sha(RECHECK/'section.md'),'intro_sha256':sha(RECHECK/'intro.md'),
 'original_sentence':report['issues'][0]['original'],'current_full_paragraph':new_sentence,
 'exact_change':'Only6UTF-8bytes, the two characters「虛線」, removed from「下圖的虛線圓」. All other section bytes identical.',
 'figure_sha256':report['figure_sha256'],
 'personally_viewed_images':[{'path':path(BASE/'render'/n),'sha256':sha(BASE/'render'/n)}
                            for n in ['shared-rotation.png','rotation-components.png']],
 'visual_reassessment':'Current「圓」description matches the original pale solid circles. Original SVGs and previously rendered/viewed images remain identical; I viewed both images again in the actual recheck turn.',
 'claim_reassessments':[{'claim_id':k,'status':'verified','own_reassessment':v} for k,v in reassessments.items()],
 'initial_report_path':path(initial),'initial_report_sha256':sha(initial),'initial_verdict':'revise',
 'cpu_policy':'No retraining, existing-model scoring, new model/data preparation, or repeat of unchanged CPU calculations. Prior executed proofs are retained because fence, source contracts, raw evidence and support scope were personally rechecked unchanged.',
 'current_html_page':'Not viewed by this reviewer. Prior own Chromium desktop/mobile timeouts remain limitations; coordinator HTTP status or page parity is not substituted as my visual proof.',
 'reader_status':'Original reader follow-up session has not been received or performed by this technical reviewer.'}
write('reading-and-figure-recheck-receipt.json',receipt)
write('version-verification.json',{'command':'.venv/bin/python '+path(RECHECK/'recheck_report.py'),
 'python':sys.version,'device':'cpu','source_sha256':receipt['new_source_sha256'],
 'intro_sha256':receipt['intro_sha256'],'section_only_expected_change':True,'fence_byte_identical':True,
 'version_checks':version_checks,'relevant_historical_methods_AST_equivalent':names})
new_artifacts=[]
def add(identifier,p,kind,description,**extra):
    x={'id':identifier,'path':path(p),'sha256':sha(p),'kind':kind,'description':description}
    x.update(extra);new_artifacts.append(x)
add('rechecked-section',RECHECK/'section.md','source_snapshot','Complete revised section raw UTF-8 bytes personally read in full by this original reviewer.')
add('rechecked-intro',RECHECK/'intro.md','source_snapshot','Complete chapter opening personally reread; raw bytes and SHA unchanged.')
add('recheck-code',RECHECK/'recheck_report.py','code','Actual original reviewer recheck/version-check/report-generation code; initial report is immutable.')
add('recheck-receipt',RECHECK/'reading-and-figure-recheck-receipt.json','source_snapshot','Own full rereading receipt, new full figure paragraph, actual image viewing, all12 individual support-scope rejudgments and explicit limits.')
add('recheck-versions',RECHECK/'version-verification.json','execution','Real independent byte/version checks for the corrected source and unchanged execution inputs.',
    command='.venv/bin/python '+path(RECHECK/'recheck_report.py'),
    result='Expected6-byte textual deletion only; intro unchanged; original fence and all registered evidence hashes unchanged; both SVGs match original renders; current computational inputs unchanged and seven relevant architecture methods AST-equivalent to original run.',
    environment={'python':sys.version,'device':'cpu','scope':'Byte/hash/AST verification only; no model or training execution'})
report['artifacts'].extend(new_artifacts)
report['source_sha256']=receipt['new_source_sha256']
report['intro_sha256']=receipt['intro_sha256']
report['verdict']='pass'
for c in report['claims']:
    assert c['status']=='verified'
    c['artifact_ids'].append('recheck-receipt')
    c['recheck_note']=reassessments[c['id']]
    if c['id']=='figure-geometry':
        c['scope']='Mathematical geometry and essential labels. Initial circle-style description issue is preserved in history and now resolved by the personally checked generic「圓」wording.'
issue=report['issues'][0]
issue['status']='resolved'
issue['resolution']='Single writer removed only「虛線」; this same original reviewer actually reread the whole new section/intro and viewed both original images again. Current「下圖的圓」matches both pale solid circles. New source SHA469732f5...; original complete issue/evidence and initial revise report are retained.'
issue['resolution_artifact_id']='recheck-receipt'
report['revision_history']=[{'round':'initial','verdict':'revise','source_sha256':receipt['old_source_sha256'],
                            'report_path':path(initial),'report_sha256':sha(initial),'issue_id':issue['id']},
                           {'round':'original-reviewer-after-minimal-correction','verdict':'pass',
                            'source_sha256':receipt['new_source_sha256'],'receipt_artifact_id':'recheck-receipt'}]
report['read_scope']['revision_recheck']='Original reviewer personally reread complete new14.1 and actual intro, then actually re-viewed both original render images after own SVG/hash parity checks. All12 support scopes reassessed individually in formal receipt. No other reviewer or root page observations were used as this proof.'
report['checks']['factual_accuracy']={'status':'pass','details':'All12 claim groups individually rejudged against the newly read section and unchanged original support. Sole initial circle-description issue is resolved by actual new wording and own current image view; full initial revise record remains.',
                                   'claim_ids':[c['id'] for c in report['claims']]}
report['checks']['figure_consistency']={'status':'pass','details':'New full circle paragraph says「下圖的圓」and matches personally re-viewed original solid circles. Both current SVG hashes equal exact original snapshots/rendered images; essential labels/axes/numbers remain consistent. Actual course-page desktop/mobile presentation remains unverified due prior Chromium timeout; root HTTP200 is not my visual proof.',
                                     'claim_ids':['figure-geometry','trig-numbers','rotary-mechanism']}
report['checks']['source_verification']['details']+=' Revised section/intro/current SVGs and every registered original evidence hash personally rechecked; current fence, raw results/data and relevant computational methods unchanged. New receipt states each support scope explicitly.'
report['limitations'].append('After minimal prose correction, this original technical reviewer reread the full revised section/intro and re-viewed unchanged original images. No first-reader follow-up result has been received; that is a separate session.')
canonical=ROOT/'docs/technical-reviews/14.1.json'
serialized=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
canonical.write_bytes(serialized)
assert canonical.read_bytes()==serialized
assert json.loads(canonical.read_text())['reviewer_task']==TASK
assert sha(initial)=='38beede75b3dbf798d8a1325d7ada6513011dc3de2eb9c7de5a970f71bab51bb'
print('Own revised canonical generated and byte/task-confirmed; initial report unchanged.')
print('verdict',report['verdict'],'claims',len(report['claims']),'resolved_issue',issue['id'])
print('source_sha256',report['source_sha256'])
print('intro_sha256',report['intro_sha256'])
print('report_sha256',sha(canonical))
print('receipt_sha256',sha(RECHECK/'reading-and-figure-recheck-receipt.json'))
