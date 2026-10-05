"""Same-owner reinspection: current source, real diff, dependency and inherited-proof verification."""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PREFIX = 'docs/technical-reviews/artifacts/phase4-13_14-reinspection-format'
A = ROOT / PREFIX
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def read(name): return json.loads((A/name).read_bytes())
prior = read('prior-opaque/13.14-prior-report.json')
extractions = read('section-context-extraction.json')
proof = read('carried-evidence-verification.json')
served = read('served-code-verification.json')
render = read('page-render-result.json')
assert render['rendered'] and render['http_status'] == 200
assert served['http_status'] == 200
assert all(x['ast_equal'] for x in served['fence_code_matches'])
assert all(x['ast_identical'] for x in proof['fence_checks'])
source = next(x for x in extractions if x['section']=='13.14')
assert source['sha256']=='37faa95f02e041ded9cfbf85895630693204fe32d37c5c6a90896fe559f7a531'
assert prior['source_sha256']=='93a03d7a64e34c60439efa889e2739a37127ed98fd2337110144534aac2bab78'
assert sha('course/figures/rewrite-13-model-roles.svg')=='b18bbd67c517b81f3f215755d70c948156283de29aac2aa4f1a068381ec53459'

# This manually authored mapping records my actual current-claim judgment, not inferred results.
support = {
 'roles': 'Current opening and figure still distinguish context-only conditional mean critic from context/candidate reward. No changed inputs, outcome meaning or claim scope; PPO v2 §5 and InstructGPT v1 §3.5 plus model classes retain support.',
 'value-target': 'Current target paragraph still uses observed1 and squared error, with repeated different selections needed for average estimation. My E[(v−R)^2] derivation and PPO Eq9 support unchanged wording.',
 'mse-numbers': 'Current .36/−1.2 and exercise .16/+.8 are unchanged. Removed line was blank within fence after print; executable AST unchanged. Original executed scalarN=1 arithmetic and variants apply exactly.',
 'gradient-only': 'Both current fences still call backward with no optimizer or assignment update. AST unchanged, so my original value-unchanged check and upstream autograd.backward support apply. No answering-improvement claim added.',
 'old-and-reference': 'Current13.12 personally reread: only character correction迴答→回答. Collection-time old logs/values, within-rollout reuse and initial-SFT reference semantics remain identical. PPO Algorithm1 and DPO§3 plus original recipe lines289,338–354 retain support.',
 'small-networks': 'Current paragraph and five-role SVG unchanged. Four PPO-branch networks plus tensors remain the specified finiteMLPs; my original3×4×2 bounded update/freeze proof and parameter counts remain supported by unchanged code/measurement hashes.',
 'kl-numbers': 'Current candidate-axis complete sum, .1927, identical-distribution zero and exercise unchanged. Source fence2 bytes are identical; original exact sum in nats and CPU float64 check apply, without sampled-card reinterpretation.',
 'exact-kl-contract': 'Current details still state logits input and log probabilities for the specified normalized p/q. Unchanged exact_kl15–21 and upstream softmax/log_softmax source; my original shape/gradient/shift checks apply to the identical executable code.',
 'reference-objective': 'Current citation remains DPO2305.18290v3§3Eq3 and the same reward−referenceKL claim; scope still distinguishes finite recipe from all token-level RLHF. No new universal assertion. Original personally read PPO/DPO/InstructGPT versions and pointers remain valid.',
 'recipe-kl-target': 'Current final statement still separates exact KL in policy loss from RM-only critic target. Code352–354, normalized_scores160–162 and my bounded update contract are unchanged by exact fingerprints.',
 'original-records': 'Current role explanation has no altered empirical result. Original64-action×3-epoch trace and reference before/after hashes are byte-identical, as is provenance code. Original recomputed invariance and arithmetic evidence carries forward; this is not new training or model evaluation.'
}
assert set(support)=={x['id'] for x in prior['claims']}
claim_reinspection=[]
for c in prior['claims']:
 claim_reinspection.append({'claim_id':c['id'],'current_status':'verified','personally_reinspected':True,
    'support_scope':support[c['id']], 'carried_original_locators':c['evidence'],
    'carried_artifact_ids':c['artifact_ids'],
    'current_dependencies':['13.12'] if c['id']=='old-and-reference' else []})
primary_artifact_ids={'ppo':['ppo-pdf','ppo-abs'],'dpo':['dpo-pdf','dpo-abs'],
 'instructgpt':['instructgpt-pdf'],'torch-loss-source':['torch-loss'],
 'torch-functional-source':['torch-functional'],'torch-autograd-source':['torch-autograd'],
 'helper-code':['posttraining-code'],'recipe':['recipe-code'],'measurements':['raw-results']}
old_artifacts={x['id']:x for x in prior['artifacts']}
source_reuse=[]
for original in prior['sources']:
 source_reuse.append({'source_id':original['id'],'version':original.get('version'),
    'original_url':original.get('url'),'prior_original_inspection':original.get('inspection_note',original.get('details')),
    'original_support_locators':[e for c in prior['claims'] for e in c['evidence'] if e['source_id']==original['id']],
    'hash_verified_snapshots':[{'artifact_id':i,'path':old_artifacts[i]['path'],'sha256':old_artifacts[i]['sha256']} for i in primary_artifact_ids.get(original['id'],[])],
    'reinspection_action':'Own original verified evidence reused after source/diff/support and exact snapshot hash checks; no new paper fetch or CPU execution claimed.'})
receipt={
 'schema_version':1,'kind':'same_owner_technical_reinspection_receipt','reviewed_on':'2026-10-05',
 'reviewer_task':'/root/phase4_factual_coordinator/factual_13_14',
 'canonical_report':'docs/technical-reviews/13.14.json',
 'canonical_source_artifact':{'id':'format-current-section','path':source['path'],'sha256':source['sha256']},
 'source':'course/chapters/13.md#13.14','source_sha256':source['sha256'],
 'prior_source_sha256':prior['source_sha256'],
 'prior_report_opaque':{'path':PREFIX+'/prior-opaque/13.14-prior-report.json','sha256':sha(PREFIX+'/prior-opaque/13.14-prior-report.json')},
 'prior_history':'All prior issues=[], original factual support, completion, report and manifest preserved under prior-opaque; all original evidence files retained untouched.',
 'actual_read_scope':'Personally read complete current13.14 and necessary current13.11,13.12,13.13; compared own prior snapshots. No13.9 or chapter intro relied on.',
 'current_context_artifacts':[{'id':'format-current-context-'+x['section'].replace('.','-'),'path':x['path'],'sha256':x['sha256'],'change':'Only13.12迴答→回答; other two necessary context slices byte-identical.'} for x in extractions if x['section']!='13.14'],
 'intro_sha256':None,'intro_applicability':'Not applicable:13.14 is not the first numbered section.',
 'figure_sha256':prior['figure_sha256'],
 'actual_difference':'13.14 removes exactly one blank line after the first Python print and before closing fence; no prose, numbers, formulas, second fence or SVG changes. Current13.12 fixes迴答→回答 only.',
 'fence_verification':proof['fence_checks'],
 'proof_fingerprint_verification':proof['prior_artifacts_checked'],
 'primary_sources_and_locators':source_reuse,
 'claim_reinspection':claim_reinspection,
 'served_page':{'url':render['url'],'http_status':200,'html_path':PREFIX+'/current-page.html','html_sha256':sha(PREFIX+'/current-page.html'),
    'served_fences_ast_match':served['fence_code_matches'],
    'browser':'Playwright using /usr/bin/chromium; actual successful run, no full-page wait or unrelated page.',
    'visual_artifacts':[{'id':i,'path':PREFIX+'/'+p,'sha256':sha(PREFIX+'/'+p)} for i,p in [('format-desktop','first-fence-desktop.png'),('format-mobile','first-fence-mobile.png')]],
    'personal_visual_inspection':'Both actual first-code-block screenshots loaded via view_image. Current block shows the final print directly before bottom padding; executable text is complete in served HTML, while long lines require horizontal scrolling at narrow width. No whole-page or unchanged-SVG re-render claimed.'},
 'carried_runtime':proof['runtime'],
 'checks_not_repeated':'No paper re-fetch, CPU numeric rerun, full recipe, GPU, model download or existing-model evaluation: source claims unchanged and first-fence AST identical.',
 'dependencies_unresolved':[],'verdict':'pass'
}
(A/'reinspection-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')

report=copy.deepcopy(prior)
report['source_sha256']=source['sha256']
report['reinspection']={'kind':'same_owner_format_reinspection','reviewer_task':receipt['reviewer_task'],'reviewed_on':'2026-10-05',
 'receipt_artifact_id':'format-receipt','receipt_path':PREFIX+'/reinspection-receipt.json','receipt_sha256':sha(PREFIX+'/reinspection-receipt.json'),
 'prior_opaque_report':receipt['prior_report_opaque'],'actual_difference':receipt['actual_difference'],'dependencies_unresolved':[]}
report['read_scope']={'section':'Current13.14 complete text, both fences; unchanged SVG carried after exact fingerprint verification.',
 'prerequisites':['Current13.11','Current13.12','Current13.13'],
 'independence':'Same original fresh technical reviewer personally reinspected own new source and dependencies; only own prior report/evidence read. No third-party review conclusions used.',
 'method':'Current factual method reread; real raw diff and AST comparison; all prior artifact hashes and current implementation/measurement/figure hashes verified.'}
for entry in report['artifacts']:
 if entry['id']=='section':entry['description']='Historical original UTF-8 section before formatting-only first-fence blank-line removal. Current source is artifact format-current-section.'
new_artifact_defs=[
 ('format-current-section','13.14-current.md','source_snapshot','Complete current original UTF-8 section, personally read.'),
 ('format-current-context-13-11','13.11-current.md','source_snapshot','Necessary current advantage context, personally read, byte-identical.'),
 ('format-current-context-13-12','13.12-current.md','source_snapshot','Necessary current old/reference context, personally read; one typo correction only.'),
 ('format-current-context-13-13','13.13-current.md','source_snapshot','Necessary current PPO clipping context, personally read, byte-identical.'),
 ('format-section-diff','13.14-diff.txt','derivation','Actual own prior/current source diff: one blank line removed.'),
 ('format-context-diff','13.12-diff.txt','derivation','Actual necessary context diff:迴答→回答.'),
 ('format-current-fence1','fence-1-current.py','code','Current first fence bytes; AST equals original actual CPU-executed fence.'),
 ('format-current-fence2','fence-2-current.py','code','Current second fence bytes equal original actual CPU-executed fence.'),
 ('format-fingerprints','carried-evidence-verification.json','derivation','Actual48 exact fingerprint checks, AST equality, unchanged checker and runtime.'),
 ('format-current-method','factual-reviewer-instructions-current.md','source_snapshot','Latest factual reviewer instructions personally reread.'),
 ('format-extraction','section-context-extraction.json','source_snapshot','Current raw UTF-8 source/context hashes and line numbers; historical diff pointers.'),
 ('format-prior-opaque','prior-opaque/13.14-prior-report.json','source_snapshot','Opaque-preserved complete own prior canonical report and issues.'),
 ('format-backup','prior-opaque/backup-record.json','source_snapshot','Actual prior opaque file copy paths and hashes.'),
 ('format-current-html','current-page.html','source_snapshot','Actual HTTP200 served current section page; no external source download.'),
 ('format-page-fetch','page-fetch.json','source_snapshot','Actual local HTTP response facts.'),
 ('format-served-code','served-code-verification.json','derivation','Both served HTML fence ASTs equal current Markdown; prior issues[] retained.'),
 ('format-render-code','render_current.py','code','Actual bounded current-page affected-code-block rendering script.'),
 ('format-render-result','page-render-result.json','source_snapshot','Actual successful Playwright/Chromium response/code/block screenshot results.'),
 ('format-desktop','first-fence-desktop.png','figure_render','Actual affected current code block from1280×800 viewport; personally viewed.'),
 ('format-mobile','first-fence-mobile.png','figure_render','Actual affected current code block from390×844 viewport; personally viewed.'),
 ('format-render-stdout','page-render.stdout.json','source_snapshot','Actual current-page render stdout.'),
 ('format-render-stderr','page-render.stderr.txt','source_snapshot','Actual render stderr.'),
 ('format-render-exit','page-render.exit.txt','source_snapshot','Actual render exit code zero.'),
 ('format-receipt','reinspection-receipt.json','derivation','Own complete current claim/support/context/version/primary-locator reinspection receipt.'),
 ('format-writer','reinspect.py','code','Actual same-owner current report and receipt generator with assertions.')]
for identifier,filename,kind,description in new_artifact_defs:
 path=PREFIX+'/'+filename;report['artifacts'].append({'id':identifier,'path':path,'sha256':sha(path),'kind':kind,'description':description})
report['sources'].append({'id':'format-reinspection','kind':'derivation','title':'Same-owner actual current source/support reinspection','verified':True,
 'details':'Personally read current section and necessary context;13.14 diff removes one blank line, current13.12 fixes one typo. Both fence ASTs identical, all48 own proof/current source fingerprints exact. Receipt records individual current support and actual source/context artifact IDs/paths/SHAs.'})
for c in report['claims']:
 c['evidence'].append({'source_id':'format-reinspection','locator':'reinspection-receipt.json /claim_reinspection entry '+c['id'],
  'supports':support[c['id']]})
 c['artifact_ids'].append('format-receipt')
 c['reinspection_status']='personally_verified_current_version'
report['checks']['factual_accuracy']['details']='Same original reviewer personally read entire current section and necessary current13.11–13.13. Each of11 substantive claims retains its verified primary/implementation/math support; per-claim own judgment and original locators are in new receipt. No altered factual claim.'
report['checks']['numeric_verification']['details']='Current first fence differs only by trailing blank line, AST identical; second fence bytes identical. Exact code/measurement/primary/evidence fingerprints checked, so own original actual CPU and64×3 trace arithmetic are carried without new numerical rerun.'
report['checks']['figure_consistency']['details']='Current SVG hash unchanged and prior personally viewed Inkscape640/390 evidence exact. Actual current affected code block also rendered successfully in Chromium at desktop/mobile viewports and personally viewed; no full-page recheck claimed.'
report['checks']['source_verification']['details']='All own44 registered artifacts plus current code/SVG/raw measurements rehashed and identical. Own personally read primary versions/URLs/locators retained with explicit carried support; current source/context snapshots, raw diffs, served AST and receipt linked. No new paper fetch claimed.'
report['checks']['limitations']['details']='Pure formatting and one necessary-context typo fix add no answering-quality or universal token-level claim. Prior original bounded CPU and existing trace evidence retained; no full training, GPU or existing-model re-evaluation. Historical Chromium124 failure preserved; this current affected code-block render succeeds, while full webpage layout is outside this reinspection. No unresolved dependencies.'
assert report['issues']==prior['issues']==[]
assert report['verdict']=='pass'
assert len({x['id'] for x in report['artifacts']})==len(report['artifacts'])
current=(ROOT/'course/chapters/13.md').read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',current));i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 13.14 '));assert current[heads[i].start():heads[i+1].start()]==(A/'13.14-current.md').read_bytes()
raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode();(A/'current-report.json').write_bytes(raw);canonical=ROOT/'docs/technical-reviews/13.14.json';canonical.write_bytes(raw);assert canonical.read_bytes()==raw
print(json.dumps({'verdict':'pass','source_sha256':source['sha256'],'report_sha256':hashlib.sha256(raw).hexdigest(),
 'receipt_artifact':next(x for x in report['artifacts'] if x['id']=='format-receipt'),
 'prior_opaque':receipt['prior_report_opaque'],'figure_sha256':report['figure_sha256'],'intro_sha256':None,
 'dependencies_unresolved':[]},ensure_ascii=False,indent=2))
