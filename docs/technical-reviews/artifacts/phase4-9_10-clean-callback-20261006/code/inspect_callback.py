"""Figure callback: bounded raw-measurement arithmetic and exact own-evidence fingerprint reuse."""
import ast
import difflib
import hashlib
import json
import math
import platform
import re
import signal
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

signal.alarm(30)
ROOT=Path(__file__).resolve().parents[5]
BASE=Path(__file__).resolve().parents[1]
OLD=ROOT/'docs/technical-reviews/artifacts/phase4-9_10-clean'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior_path=BASE/'prior/9.10-prior-report.json'
prior=json.loads(prior_path.read_bytes())
assert prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_9_10_clean'
assert sha(prior_path)=='5a88d6f62066cf36f36956026bb5a47e2842553c5492f79b82a736ce9187afb4'
reused=[]
for artifact in prior['artifacts']:
    assert sha(ROOT/artifact['path'])==artifact['sha256'],artifact['id']
    reused.append({'id':artifact['id'],'path':artifact['path'],'sha256':artifact['sha256'],'reuse_scope':artifact['description']})
source_reuse=[]
for source in prior['sources']:
    if source['kind']=='repository_code':
        assert sha(ROOT/source['path'])==source['sha256']
        source_reuse.append({'id':source['id'],'path':source['path'],'sha256':source['sha256'],'scope':source['inspection_note']})
    if source['kind'] in ['paper','official_source','official_docs']:
        assert sha(ROOT/source['snapshot_path'])==source['snapshot_sha256']
        source_reuse.append({'id':source['id'],'path':source['snapshot_path'],'sha256':source['snapshot_sha256'],'url':source['url'],'version':source['version'],'locator_and_scope':source['inspection_note'],'reuse':'Personal original inspection on 2026-10-05; exact immutable bytes fingerprinted on callback; not freshly fetched or reread in full.'})

body=(BASE/'inputs/current-section.md').read_bytes()
assert hashlib.sha256(body).hexdigest()==prior['source_sha256']=='05a9663a16f0b7de89890f900ae80e9284e71a345a069ebf757c2b073817cd96'
assert body==(OLD/'execution/section.md').read_bytes()
ns={'s':'http://www.w3.org/2000/svg'}
figures={};diffs={}
for name in ['rewrite-09-10-calibration-bins.svg','audio_calibration_test.svg']:
    p=ROOT/'course/figures'/name;original=OLD/'inputs/course/figures'/name
    figures['course/figures/'+name]={'current_sha256':sha(p),'own_previous_sha256':sha(original)}
    diffs[name]=''.join(difflib.unified_diff(original.read_text().splitlines(keepends=True),p.read_text().splitlines(keepends=True),fromfile='own-frozen/'+name,tofile='current/'+name))
(BASE/'execution/own-current-svg-diff.txt').write_text(''.join(diffs.values()))

INPUT=ROOT/'docs/course-experiments/results/encoders.json'
data=json.loads(INPUT.read_bytes());reads={}
def pointer(p):
    v=data
    for k in p.split('/')[1:]: v=v[k.replace('~1','/').replace('~0','~')]
    reads[p]=v
    return v
input_sha=sha(INPUT)
assert input_sha==sha(OLD/'inputs/docs/course-experiments/results/encoders.json')=='4ed0c1a384802adca0c30a77a230b1fd2e7c6cbe6de5f3d2af583395fed0ed18'
provenance={k:pointer('/'+k) for k in ['revision','seed','torch_version','python_version']}
for p in ['scripts/course_experiments/modalities.py','tiny_perceptron/alignment.py']:
    recorded=pointer('/code_sha256/'+p.replace('/','~1'))
    assert sha(ROOT/p)==recorded
chosen=pointer('/results/audio/calibration/chosen_temperature');assert chosen==.5
vc=pointer('/results/audio/calibration/validation/count')
vcorrect=pointer('/results/audio/calibration/validation/original/correct')
assert vc==vcorrect==8
testcount=pointer('/results/audio/calibration/test/count');assert testcount==14
labels=pointer('/results/audio/calibration/test/labels')
families=pointer('/results/audio/calibration/test/families')
assert len(labels)==len(families)==testcount
variants={}
for name in ['original','calibrated']:
    p='/results/audio/calibration/test/'+name
    row={k:pointer(p+'/'+k) for k in ['temperature','count','correct','accuracy','mean_confidence','nll','ece','brier','bins','confidence','probabilities','predicted_labels']}
    assert row['count']==testcount and row['correct']==11
    assert row['accuracy']==11/14
    assert len(row['confidence'])==len(row['probabilities'])==len(row['predicted_labels'])==14
    assert row['correct']==sum(a==b for a,b in zip(row['predicted_labels'],labels,strict=True))
    mean=sum(row['confidence'])/14
    ece=sum(b['count']/14*abs(b['accuracy']-b['mean_confidence']) for b in row['bins'] if b['count'])
    brier=sum(sum((p-float(c==target))**2 for c,p in enumerate(probs)) for probs,target in zip(row['probabilities'],labels,strict=True))/14
    assert math.isclose(mean,row['mean_confidence'],abs_tol=2e-7)
    assert math.isclose(ece,row['ece'],abs_tol=1e-12)
    assert math.isclose(brier,row['brier'],abs_tol=2e-7)
    assert len(row['bins'])==5 and sum(b['count'] for b in row['bins'])==14
    assert all(len(probs)==2 and abs(sum(probs)-1)<2e-7 for probs in row['probabilities'])
    variants[name]={k:row[k] for k in ['temperature','count','correct','accuracy','mean_confidence','nll','ece','brier']}
assert reads['/results/audio/calibration/test/original/predicted_labels']==reads['/results/audio/calibration/test/calibrated/predicted_labels']
assert format(variants['original']['mean_confidence']*100,'.2f')=='87.96'
assert format(variants['calibrated']['mean_confidence']*100,'.2f')=='92.73'
for name,ece,brier in [('original','0.13186','0.20207'),('calibrated','0.14154','0.25337')]:
    assert format(variants[name]['ece'],'.5f')==ece
    assert format(variants[name]['brier'],'.5f')==brier

svg=ET.parse(ROOT/'course/figures/audio_calibration_test.svg').getroot()
bars=[r for r in svg.findall('s:rect',ns) if r.attrib.get('fill')=='#357ba9']
backgrounds=[r for r in svg.findall('s:rect',ns) if r.attrib.get('fill')=='#e8edf3']
lines=svg.findall('s:line',ns)
assert len(bars)==len(backgrounds)==len(lines)==2
geometry=[]
for name,bar,background,line in zip(['original','calibrated'],bars,backgrounds,lines,strict=True):
    assert float(background.attrib['x'])==float(bar.attrib['x'])==32
    assert float(background.attrib['width'])==560
    assert float(line.attrib['x1'])==float(line.attrib['x2'])==472
    confidence_ratio=float(bar.attrib['width'])/560
    accuracy_ratio=(float(line.attrib['x1'])-32)/560
    assert math.isclose(confidence_ratio,round(variants[name]['mean_confidence'],4),abs_tol=1e-12)
    assert math.isclose(accuracy_ratio,11/14,abs_tol=1e-12)
    geometry.append({'variant':name,'bar_fraction':confidence_ratio,'fixed_accuracy_fraction':accuracy_ratio,'bar_y':bar.attrib['y'],'dash_y1':line.attrib['y1'],'dash_y2':line.attrib['y2']})
svg_text=[e.text for e in svg.findall('s:text',ns)]
for text in ['同批14段單音','答案不變，信心變大','在8段驗證錄音選溫度0.5。','驗證8/8全對。','另留14段測試，只答對11/14。','平均信心87.96%','平均信心92.73%','ECE　0.13186','Brier　0.20207','ECE　0.14154','Brier　0.25337','紫色虛線：固定正確率11/14','（78.57%）；兩卡都是0–100%尺度。','不是生成回答或安全判斷的可靠度。']:
    assert text in svg_text

# Locate and preserve just the original calculation/API contracts needed for the drawing.
source=ROOT/'scripts/course_experiments/modalities.py';raw=source.read_bytes();tree=ast.parse(raw)
metric=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='confidence_metrics')
segment=ast.get_source_segment(raw.decode(),metric)
(BASE/'code/inspected-original-confidence-metrics.py').write_text(segment+'\n')
(BASE/'execution/inspected-current-original-contract.txt').write_text('Source: scripts/course_experiments/modalities.py; SHA '+sha(source)+'; AST lines '+str(metric.lineno)+'-'+str(metric.end_lineno)+'\n'+segment+'\n')
result={'scope':'Figure-only callback. Current full section unchanged; old personal code/source/numeric evidence reused only after exact hash and scope checks. Fresh direct named raw audio measurement inspection and bounded scalar/bin/squared-error arithmetic; no torch execution, model inference, training, new calibration search, paper download or GPU work.',
    'environment':{'python':platform.python_version(),'device':'CPU scalar arithmetic','torch_used':'no','inkscape':subprocess.check_output(['inkscape','--version'],text=True).strip()},
    'prior_report_file':prior_path.relative_to(ROOT).as_posix(),'prior_report_sha256':sha(prior_path),'prior_issues_count':len(prior['issues']),'prior_issues_preserved':'Complete prior report retained byte-for-byte; existing issues remain unchanged in refreshed report.',
    'current_section_sha256':hashlib.sha256(body).hexdigest(),'figures':figures,'raw_input_path':INPUT.relative_to(ROOT).as_posix(),'raw_input_full_sha256':input_sha,'read_pointers':list(reads),'selected_raw_measurements':reads,'provenance':provenance,
    'audio_validation':{'count':vc,'correct':vcorrect,'chosen_temperature':chosen},'audio_test':variants,'geometry':geometry,
    'unchanged_artifact_hash_scope_checks':reused,'unchanged_primary_source_and_code_checks':source_reuse,
    'new_visual_claim_scope':'8 validation clips selected positive temperature .5; 14 fixed test clips, 11 correct, mean predicted-class confidence and ECE/unhalved two-class Brier values; common0–100% scales. Geometry and labels did not acquire claims about generated responses or safety.'}
(BASE/'execution/measurement-and-reuse-inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
