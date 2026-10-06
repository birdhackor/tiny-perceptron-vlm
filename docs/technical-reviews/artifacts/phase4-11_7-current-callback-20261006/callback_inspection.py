"""Same-reviewer narrow current callback; hashes/text only, no model execution."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from datetime import UTC, datetime

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PRIOR=ROOT/'docs/technical-reviews/artifacts/phase4-11_7-independent'
TASK='/root/phase4_factual_coordinator/factual_11_7'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
preserved=read(OUT/'opaque-preservation-receipt.json')
prior_path=ROOT/preserved['canonical']['opaque_file']
assert digest(prior_path.read_bytes())==preserved['canonical']['sha256']=='44153260df68a0fbcb323ee6fb24e19c4d721b13386ff18618f32a9c8b018743'
archive=ROOT/preserved['prior_proof_archive']['file']
assert digest(archive.read_bytes())==preserved['prior_proof_archive']['sha256']
prior=read(prior_path)  # Explicitly authorized own original report, after opaque preservation.
assert prior['reviewer_task']==TASK and prior['lesson_id']=='11.7'

def section(path,lesson):
    raw=path.read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
    i=next(i for i,m in enumerate(heads) if m[0].startswith(('## '+lesson+' ').encode()))
    start=heads[i].start();end=heads[i+1].start() if i+1<len(heads) else len(raw)
    return raw[start:end],raw[:start].count(b'\n')+1

body,first_line=section(ROOT/'course/chapters/11.md','11.7')
ctx,ctx_line=section(ROOT/'course/chapters/07.md','7.15')
assert body==(OUT/'current-inputs/11.7.md').read_bytes()
assert ctx==(OUT/'current-inputs/7.15.md').read_bytes()
assert digest(body)==prior['source_sha256']=='08e3028a1d0baa4f327679d862799580c81e7821bb6b9a69723c7fb328f2bfa2'
assert prior['figure_sha256']=={} and not re.search(rb'!\[[^\]]*\]\([^)]*\.svg\)',body)
prior_manifest=read(PRIOR/'input-manifest.json')
prior_ctx=next(i for i in prior_manifest['files'] if i['path']=='course/chapters/07.md#7.15')
assert digest((ROOT/prior_ctx['frozen_snapshot']).read_bytes())==prior_ctx['sha256']
source_checks=[];artifact_checks=[]
for group,target in [('sources',source_checks),('artifacts',artifact_checks)]:
    for item in prior[group]:
        if 'path' not in item or 'sha256' not in item:continue
        path=ROOT/item['path'];actual=digest(path.read_bytes())
        target.append({'id':item['id'],'path':item['path'],'expected_sha256':item['sha256'],
                       'current_sha256':actual,'match':actual==item['sha256']})
        assert actual==item['sha256'],item['path']
# Check imported/referenced code that is frozen in the original input manifest
# without comparing a moving whole chapter or mechanically checking other lessons.
additional_code_checks=[]
for item in prior_manifest['files']:
    if not item['path'].endswith('.py'):continue
    path=ROOT/item['path'];actual=digest(path.read_bytes())
    additional_code_checks.append({'path':item['path'],'frozen_sha256':item['sha256'],
                                    'current_sha256':actual,'match':actual==item['sha256']})
    assert actual==item['sha256'],item['path']
previous_audit=read(PRIOR/'independent_cpu_check.json')
for item in previous_audit['code_version_checks']:
    actual=digest((ROOT/item['path']).read_bytes())
    assert actual==item['recorded_sha256']==item['current_sha256']==item['revision_sha256']
    if not any(x['path']==item['path'] for x in additional_code_checks):
        additional_code_checks.append({'path':item['path'],'frozen_sha256':item['recorded_sha256'],
                                      'current_sha256':actual,'match':True})

slices=[]
selectors=[('一模型先回答',None,'Shared parameters may alter old answers; retest two fixed tasks from the same model.'),
           ('下面A題提供',None,'None denotes unmeasured cells; true evaluation needs answers and fixed generation rules.'),
           ('真正流程是',None,'Same trained start and fixed A/B holdouts; evaluation questions stay outside updates.'),
           ('若B提升、A下降','已有屬性','Compare old-task baseline and after; new improvement and old decline can coexist.')]
for number,(prefix,end_prefix,support) in enumerate(selectors,1):
    p=prefix.encode();start=ctx.index(p);end=ctx.index(b'\n',start)+1
    if end_prefix:end=ctx.index(end_prefix.encode(),start)
    raw=ctx[start:end]
    file=OUT/'current-inputs'/f'7.15-necessary-slice-{number}.raw.txt';file.write_bytes(raw)
    slices.append({'id':f'ctx-7.15-slice-{number}','source':'course/chapters/07.md#7.15',
                   'source_section_sha256':digest(ctx),'byte_start_within_section':start,
                   'byte_end_within_section_exclusive':end,'source_start_line':ctx_line+ctx[:start].count(b'\n'),
                   'snapshot':file.relative_to(ROOT).as_posix(),'slice_sha256':digest(raw),
                   'scope':support,'text_personally_read':raw.decode()})

current_methods=[]
for name in ['docs/review-tools/factual-reviewer-instructions.md',
             '.agents/skills/clear-tutorial/references/review-protocol.md','scripts/check_technical_reviews.py']:
    raw=(ROOT/name).read_bytes();snapshot=OUT/'current-inputs'/name
    assert raw==snapshot.read_bytes()
    current_methods.append({'path':name,'sha256':digest(raw),'snapshot':snapshot.relative_to(ROOT).as_posix()})

reuse=[]
for claim in prior['claims']:
    reuse.append({'claim_id':claim['id'],'original_support_scope':claim['scope'],
                  'sources_retained':[e['source_id'] for e in claim['evidence']],
                  'original_artifacts_retained':claim['artifact_ids'],
                  'current_assessment':('Current 7.15 slices preserve the shared-parameter/fixed-holdout retention premise; the same original paper and implementation remain the scientific support.' if claim['id']=='c-retention' else '11.7 text, relevant code, original raw measurements and original executed artifacts are byte-identical; this unchanged support is explicitly reused.'),
                  'new_model_or_training_execution':False})

receipt={
 'kind':'same_reviewer_narrow_current_technical_inspection','inspected_at':datetime.now(UTC).isoformat(),
 'reviewer_task':TASK,'reviewer_identity_scope':'Same original 11.7 reviewer callback, not a newly independent/fresh review identity.',
 'primary':{'source':'course/chapters/11.md#11.7','source_sha256':digest(body),
            'snapshot':(OUT/'current-inputs/11.7.md').relative_to(ROOT).as_posix(),
            'first_line':first_line,'read_scope':'Complete current own section personally read in this callback.',
            'intro':None,'figure_sha256':{}},
 'necessary_context':{'source':'course/chapters/07.md#7.15','current_section_sha256':digest(ctx),
                      'snapshot':(OUT/'current-inputs/7.15.md').relative_to(ROOT).as_posix(),
                      'actual_read_scope':'Complete current 7.15 read to identify the truly necessary dependency; verification is limited to the four preserved slices.',
                      'verified_slices':slices,'affects_claim_ids':['c-retention'],
                      'prior_frozen_context':prior_ctx,
                      'prior_sha_meaning':'90967d... names the original 2026-10-05 raw 7.15 snapshot; it is retained as historical input, not asserted as the current context hash.',
                      'semantic_assessment':'The necessary fixed-model/fixed-task/baseline-versus-after premises remain supported by original paper and unchanged evaluation code. Spelling was corrected. The supplemental A-model source link now points to retained direct SFT in 7.13; 11.7 does not use that supplemental arithmetic experiment as its VQA score evidence.',
                      'not_verified_as_new_claims':'7.15 own arithmetic-task numeric results and its supplemental model-link claim were not re-audited; no such result is substituted for 11.7 raw VQA evidence.'},
 'source_fingerprint_checks':source_checks,'prior_formal_artifact_fingerprint_checks':artifact_checks,
 'additional_original_code_fingerprint_checks':additional_code_checks,
 'original_authority_and_measurement_support':'Explicit arXiv1811.11682v1 and CPython3.13.5 bytes are unchanged; their personally inspected 2026-10-05 URLs/version/locators and support boundaries are retained. The RL paper supports forgetting/replay mechanism only, not supervised VQA scores. Raw vqa.json full hash is unchanged; previous named raw-token/target/sample/provenance inspection remains the score evidence.',
 'reuse_matrix':reuse,
 'reuse_scope':{'original_fence':'2026-10-05 exact fence CPU execution and 500/500 recipe-only variation are reused; not rerun today.',
               'original_empirical_accounting':'2026-10-05 raw-token rescore, split hashes/family disjointness, freeze flags, target history sums and deterministic sampler reconstruction are reused; no saved model was evaluated today.',
               'original_render':'2026-10-05 actual desktop/mobile screenshots and DOM table/code checks are reused for byte-identical 11.7 content. No current callback render is claimed.',
               'unchanged_claim_limits':'Handmade scores remain handmade; retention is fixed small-task evidence; unmatched target budgets and p0.5 example/token/time distinction remain explicit.',
               'no_expansion':'No other lesson technical verdicts read, no new source/model/data download, GPU, training, full recipe or agent delegation; no textbook edits.'},
 'frozen_whole_file_hash_policy':'Original input-manifest full11.md SHA continues to name its saved 2026-10-05 frozen input only. This callback has no new whole-chapter current-hash claim; formal source SHA remains raw11.7 bytes.',
 'opaque_prior':preserved,
 'current_methods':current_methods,
 'environment':{'python':sys.version,'python_executable':sys.executable,
                'torch_distribution':importlib.metadata.version('torch'),
                'device_scope':'CPU filesystem/JSON/hash inspection only; no model/tensor execution'},
 'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-11_7-current-callback-20261006/callback_inspection.py',
 'events':[{'type':'current_text_read','classification':'Current 7.15教材 is a pending context claim, not a prior review answer; authorized by latest factual instructions.'},
           {'type':'own_original_report_read','classification':'Same original reviewer read own opaque-preserved prior formal fields for version and support-scope reuse; no other reviewer conclusions were read.'}],
 'outcome':'All fingerprint assertions passed; current necessary context supports unchanged 11.7 claims. Same-reviewer verdict remains pass.'
}
write(OUT/'current-inspection.json',receipt)
print(json.dumps({'outcome':receipt['outcome'],'primary':receipt['primary'],
                  'current_context_sha256':digest(ctx),'prior_context_sha256':prior_ctx['sha256'],
                  'necessary_slice_ids':[i['id'] for i in slices],
                  'checked_source_files':len(source_checks),'checked_prior_formal_artifacts':len(artifact_checks),
                  'checked_additional_code_files':len(additional_code_checks),'environment':receipt['environment']},ensure_ascii=False,indent=2))
