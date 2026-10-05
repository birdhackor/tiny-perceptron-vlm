"""Same owner's callback inputs and proof identity checks; no model execution."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import importlib.metadata
import json
import re
import sys

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-5_1-independent'
OUT=BASE/'continuity-repair-callback-20261005T225207Z'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def section(raw,lesson):
    headings=list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+',raw))
    index=next(i for i,h in enumerate(headings) if h[1].decode()==lesson)
    return raw[headings[index].start():(headings[index+1].start() if index+1<len(headings) else len(raw))]
dispatch=json.loads((OUT/'dispatch-provenance.json').read_bytes())
report=json.loads((ROOT/dispatch['prior_pass_path']).read_bytes())
assert sha((ROOT/dispatch['prior_pass_path']).read_bytes())==dispatch['prior_pass_sha256']
assert sha((ROOT/'docs/technical-reviews/5.1.json').read_bytes())==dispatch['prior_pass_sha256']
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_5_1'
(OUT/'inputs').mkdir(exist_ok=False)
whole=(ROOT/'course/chapters/05.md').read_bytes();body=section(whole,'5.1');intro=whole[:whole.index(b'## 5.1 ')]
old=(BASE/'original/section.md').read_bytes()
before='梯度為正只說這張表收到訊號，代價下降再說更新改善了固定題。'.encode()
after='梯度大小大於0，只說這張表收到訊號；代價下降才說明更新改善了固定題。'.encode()
assert old.count(before)==1 and old.replace(before,after)==body
for name,raw in [('chapter05-frozen-input.md',whole),('section-5_1.md',body),('chapter-intro.md',intro)]:
    (OUT/'inputs'/name).write_bytes(raw)
assert intro==(BASE/'original/chapter-intro.md').read_bytes()
fence=re.findall(rb'```python\n(.*?)```',body,re.S)
assert len(fence)==1 and fence[0]==(BASE/'original/fence-1.py').read_bytes()
(OUT/'inputs/fence-1.py').write_bytes(fence[0])
fig_refs=re.findall(rb'!\[[^\]]*\]\(([^)]+)\)',body)
assert not fig_refs
contexts=[]
for file,lesson in [('01.md','1.12'),('01.md','1.13'),('02.md','2.1')]:
    raw=(ROOT/'course/chapters'/file).read_bytes();part=section(raw,lesson)
    target=OUT/'inputs'/f'context-{lesson}.md';target.write_bytes(part)
    contexts.append(dict(source=f'course/chapters/{file}#{lesson}',sha256=sha(part),snapshot=target.relative_to(ROOT).as_posix(),read_scope='Entire current stable section read for method context only; no judgment of this other lesson’s empirical scores.',freeze_status='Coordinator explicitly confirmed 01 frozen; after 02 confirmation, 2.1 reread at final stable bytes.'))
context_figures=[]
(OUT/'figures').mkdir(exist_ok=False)
for name in ['rewrite-01-update.svg','rewrite-02-embedding-purpose.svg']:
    path=ROOT/'course/figures'/name;raw=path.read_bytes();target=OUT/'figures'/name;target.write_bytes(raw)
    context_figures.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(raw),snapshot=target.relative_to(ROOT).as_posix(),scope='Necessary context figure only; 5.1 itself references no figure.'))
proof=[]
for artifact in report['artifacts']:
    assert sha((ROOT/artifact['path']).read_bytes())==artifact['sha256'],artifact['id']
    proof.append(dict(id=artifact['id'],path=artifact['path'],sha256=artifact['sha256'],unchanged=True))
inputs=json.loads((BASE/'input-manifest.json').read_bytes());deps=[]
for path in ['tiny_perceptron/model.py','tiny_perceptron/data.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py','docs/course-experiments/results/text_foundation.json','docs/review-tools/section_facts.py']:
    raw=(ROOT/path).read_bytes();expected=inputs['current_inputs'][path]['sha256'];assert sha(raw)==expected,path
    deps.append(dict(path=path,sha256=expected,same_original_proof_input=True))
installed=[]
for item in json.loads((BASE/'sources/installed-identity.json').read_bytes()):
    assert sha((ROOT/item['installed_path']).read_bytes())==item['installed_sha256']
    installed.append(dict(path=item['installed_path'],sha256=item['installed_sha256'],same_official_snapshot_and_original_environment=True))
assert importlib.metadata.version('torch')=='2.14.1+cpu'
empirical=json.loads((ROOT/'docs/course-experiments/results/text_foundation.json').read_bytes())
pointers=['/revision','/device','/seed','/torch_version','/python_version','/gpu','/evidence_status','/step_scale','/results/data/train/records','/results/data/train/sha256','/results/data/validation/records','/results/data/validation/sha256','/results/training/steps','/results/training/effective_tokens','/results/training/initial_loss','/results/training/final_loss','/results/after/validation/nll','/results/after/validation/nll_sum','/results/after/validation/effective_tokens']
measurements={}
for pointer in pointers:
    value=empirical
    for key in pointer.strip('/').split('/'):value=value[key]
    measurements[pointer]=value
dump('empirical-pointers.json',dict(original_sha256=inputs['current_inputs']['docs/course-experiments/results/text_foundation.json']['sha256'],top_level_key_types={k:type(v).__name__ for k,v in empirical.items()},actually_read_pointers=measurements,scope='Only listed provenance/measurements inspected, no author notes/review values. Existing measurements, not new GPU execution.'))
dump('environment.json',dict(python=sys.version,python_executable=sys.executable,torch_distribution=importlib.metadata.version('torch'),device='metadata-only; no model execution',cwd=str(Path.cwd()),installed_source_checks=installed))
metadata=dict(read_on=datetime.now(UTC).isoformat(),reviewer_task=report['reviewer_task'],prior_pass=dispatch,source='course/chapters/05.md#5.1',source_sha256=sha(body),source_file_sha256=sha(whole),source_file_sha256_meaning='Aggregate identity of callback frozen input snapshot only, not full chapter acceptance.',intro_sha256=sha(intro),intro_summary='先用固定接字題查通梯度和更新，再組成batch認識優化器、步幅與存檔；評估另看新題，並分清有效答案、不同材料、參數格數和秒數。',figure_sha256={},context_sections=contexts,context_figures=context_figures,fence_sha256=sha(fence[0]),fence_unchanged=True,change=dict(previous=before.decode(),current=after.decode(),judgment='Magnitude > 0 correctly specifies the norm, rather than individual gradient signs. No new numeric result, method or code.'),dependency_checks=deps,prior_artifact_identity_checks=proof,unresolved_dependencies=[],pending_context_read_history='2.1 first read before freeze confirmation; coordinator confirmed it and its figure stable, then entire 2.1 was reread before these snapshots. No pending inference used for PASS.',scope='Only current 5.1 and chapter introduction reviewed, plus linked 1.12/1.13/2.1 for necessary method context. No continuity/reader/repair-intent reports used; no other section assessed.',model_runs='No new model run required: same original fence and dependencies plus word-level magnitude clarification; original bounded CPU/proof values re-read and hashes verified.')
dump('capture.json',metadata)
print(json.dumps({k:metadata[k] for k in ['source_sha256','intro_sha256','figure_sha256','fence_sha256','change','unresolved_dependencies']},ensure_ascii=False,indent=2))
