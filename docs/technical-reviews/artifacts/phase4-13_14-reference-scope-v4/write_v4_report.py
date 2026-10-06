"""Same-owner V4 resolution report; preserve original accepted and revise history."""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PREFIX='docs/technical-reviews/artifacts/phase4-13_14-reference-scope-v4'
A=ROOT/PREFIX
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def load(name):return json.loads((A/name).read_bytes())
prior=load('prior-opaque/revise-13.14-report.json');facts=load('inspection-facts.json');contexts=load('source-context.json')
assert sha(PREFIX+'/prior-opaque/revise-13.14-report.json')=='77ff086690cd25ba76569f3549dfea9ae019e2606b30103970ea22ed05a0b9d2'
assert facts['source_sha256']=='ba2361526193f3321b9d8974a6889117fe4602a5e2d6d132413d0d22ab9d6ab4'
assert facts['figure_sha256']=='d7589ab9e44118540111e8f331b555d4d6c4db3f548e7d373bf7d40750eba406'
assert (A/'inspection.exit.txt').read_text().strip()=='0' and (A/'render.exit.txt').read_text().strip()=='0'
assert len(prior['issues'])==1 and prior['issues'][0]['id']=='reference-stage-scope' and prior['issues'][0]['status']=='open'
receipt={
 'schema_version':1,'kind':'same_owner_reference_scope_v4_resolution','reviewed_on':'2026-10-06','reviewer_task':prior['reviewer_task'],
 'verdict':'pass','canonical_report':'docs/technical-reviews/13.14.json',
 'canonical_source_artifact':{'id':'reference-v4-source','path':PREFIX+'/current-13.14.md','sha256':facts['source_sha256']},
 'canonical_inspection_artifact':{'id':'reference-v4-inspection','path':PREFIX+'/inspection-facts.json','sha256':sha(PREFIX+'/inspection-facts.json')},
 'source_sha256':facts['source_sha256'],'intro_sha256':None,
 'figure_sha256':{'course/figures/rewrite-13-model-roles.svg':facts['figure_sha256']},
 'prior_full_revise_opaque':facts['prior_full_revise_report'],
 'prior_failures_preserved':{'checker_stdout':PREFIX+'/prior-opaque/revise-checker.stdout.txt','checker_exit':1,'complete_history_backup':PREFIX+'/prior-opaque/backup-record.json'},
 'actual_read_scope':'Complete current13.14; changed necessary current13.12 and its real diff; current13.4 byte-identical. Original recipe273–290/326–354 after AST range inspection; own immutable original DPOv3§3 and InstructGPTv1§3.1/§3.5 necessary method paragraphs locally reread.',
 'current_context_versions':contexts,
 'original_issue':copy.deepcopy(prior['issues'][0]),
 'current_body_quote':facts['current_body_quote'],'current_visible_figure_labels':facts['current_visible_figure_labels'],
 'current_accessibility_description':facts['current_accessibility_description'],
 'primary_support_locators':facts['primary_support_locators'],
 'support_scope':facts['support_scope'],
 'issue_resolution':'V4 body now identifies completed-SFT policy atPPO start and fixes it only during thisPPO stage. Visible reference card and accessibility description use the samePPO boundary. Current13.12 no longer supplies conflictingwhole-post-training context. Original source and fixed recipe support this precise boundary.',
 'actual_visual_inspection':{'path':PREFIX+'/model-roles-v4.png','sha256':sha(PREFIX+'/model-roles-v4.png'),'render':'Inkscape640px exit0','view_image_performed':True,'observations':'All five roles visible. Updated reference titlePPO起點 and interval整段PPO期間不更新 fit the card and match body/desc.'},
 'actual_served_page':facts['actual_served_page'],
 'carried_evidence':'All97 prior registered own artifacts and current unchanged implementation/raw-measurement files rehashed exactly. Both source fences byte-identical. Original actual CPU/variants/trace and primary evidence carried; no new model or numeric execution, refetch or training.',
 'resolved_issue_id':'reference-stage-scope','unresolved_dependencies':[],
 'limitations':'Narrow semantic/source/visible-SVG reinspection and own localpageHTTP check; no newfull-page/mobile layout review, GPU, modelCPU, full training or downloads.'
}
(A/'v4-reinspection-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
report=copy.deepcopy(prior);report['verdict']='pass';report['source_sha256']=facts['source_sha256'];report['figure_sha256']=receipt['figure_sha256'];report['reviewed_on']='2026-10-06'
report['reference_scope_v4_reinspection']={'kind':receipt['kind'],'reviewer_task':receipt['reviewer_task'],'verdict':'pass',
 'receipt_artifact_id':'reference-v4-receipt','receipt_path':PREFIX+'/v4-reinspection-receipt.json','receipt_sha256':sha(PREFIX+'/v4-reinspection-receipt.json'),
 'inspection_artifact_id':'reference-v4-inspection','inspection_path':PREFIX+'/inspection-facts.json','inspection_sha256':sha(PREFIX+'/inspection-facts.json'),
 'prior_full_revise_opaque':receipt['prior_full_revise_opaque'],'unresolved_dependencies':[]}
report['read_scope']=receipt['actual_read_scope']
for e in report['artifacts']:
 if e['id'] in ['svg','render640','render390','reference-scope-svg','reference-scope-render']:
  e['description']='Historical preV4 own original figure evidence; immutable. Current actual V4 SVG/render are reference-v4-svg/reference-v4-render. '+e['description']
new_entries=[
 ('reference-v4-source','current-13.14.md','source_snapshot','Complete personally read currentV4 raw UTF-8 section.'),
 ('reference-v4-context13-4','current-13.4.md','source_snapshot','Necessary completed-demonstration reference context, byte-identical.'),
 ('reference-v4-context13-12','current-13.12.md','source_snapshot','Personally read current correctedPPO reference context; no other reviewer report used.'),
 ('reference-v4-context-versions','source-context.json','source_snapshot','Actual current source/context/SVG hashes and line numbers.'),
 ('reference-v4-diff13-14','diff-13.14.txt','derivation','Actual own priorREVISE/currentV4 section diff; explicit completedSFT/PPO boundary.'),
 ('reference-v4-diff13-12','diff-13.12.txt','derivation','Actual necessary context diff withPPO-specific reference and InstructGPT citation.'),
 ('reference-v4-svg','current-model-roles.svg','source_snapshot','Actual correctedV4 SVG bytes, including visible title/time label and accessible description.'),
 ('reference-v4-render','model-roles-v4.png','figure_render','Actual independently renderedV4 graphic; loaded with view_image and inspected.'),
 ('reference-v4-render-exit','render.exit.txt','source_snapshot','Actual Inkscape exit0.'),
 ('reference-v4-render-stderr','render.stderr.txt','source_snapshot','Actual render stderr.'),
 ('reference-v4-fence1','current-fence-1.py','code','Current first fence bytes equal own previously verified executable.'),
 ('reference-v4-fence2','current-fence-2.py','code','Current second fence bytes equal own previously verified executable.'),
 ('reference-v4-code','inspect_v4.py','code','Actual current body/SVG/fences/AST/prior-proof fingerprint inspection program.'),
 ('reference-v4-stdout','inspection.stdout.json','source_snapshot','Actual inspection stdout containing observed scope/source/AST/hash facts.'),
 ('reference-v4-stderr','inspection.stderr.txt','source_snapshot','Actual inspection stderr.'),
 ('reference-v4-exit','inspection.exit.txt','source_snapshot','Actual inspection exit0.'),
 ('reference-v4-commands','commands.json','source_snapshot','Actual narrow reinspection, rendering, viewing and localHTTP commands.'),
 ('reference-v4-analysis','own-v4-analysis.md','derivation','Own source-support comparison, visible figure observations and actual issue resolution.'),
 ('reference-v4-prior-opaque','prior-opaque/revise-13.14-report.json','source_snapshot','Complete untouched prior77ff REVISE canonical.'),
 ('reference-v4-backups','prior-opaque/backup-record.json','source_snapshot','AcceptedPASS, REVISE, failedchecker and originalproof/history opaque copy records.'),
 ('reference-v4-page','current-page.html','source_snapshot','Own actually fetched updated local13.14 page; exact corrected phrase verified.'),
 ('reference-v4-page-facts','page-fetch.json','source_snapshot','Actual own HTTP200 updatedpage fingerprint and phrase check.'),
 ('reference-v4-receipt','v4-reinspection-receipt.json','derivation','New same-owner receipt with canonical artifact IDs/paths/SHAs, issue history and current original support.'),
 ('reference-v4-writer','write_v4_report.py','code','Actual complete currentPASS report/receipt generator; priorREVISE remains intact.')]
for identifier,filename,kind,description in new_entries:
 path=PREFIX+'/'+filename;report['artifacts'].append({'id':identifier,'path':path,'sha256':sha(path),'kind':kind,'description':description})
report['artifacts'].append({'id':'reference-v4-inspection','path':PREFIX+'/inspection-facts.json','sha256':sha(PREFIX+'/inspection-facts.json'),'kind':'execution',
 'description':'ActualV4 narrow source/AST/hash facts, without model computation.',
 'command':'.venv/bin/python '+PREFIX+'/inspect_v4.py',
 'result':'exit0; body/SVG corrected scope matches exact original contract;97 inherited artifacts unchanged; bothfences identical; localpageHTTP200.',
 'environment':{'python':'3.13.5','torch':'2.14.1+cpu','device':'CPU source/AST/hash analysis only; no model computation','shell':'bash login:false'}})
report['sources'].append({'id':'reference-v4-verification','kind':'derivation','title':'OwnV4 completedSFT/PPO scope resolution','verified':True,
 'details':'Actual current body, visible SVG card and accessible description identify completedSFT startingpolicy, fixed during thisPPO. Personally compared immutable originalDPOv3§3/InstructGPTv1§3.1/§3.5 and recipe289/326; realcurrentdiff and newrender/view recorded.'})
for s in report['sources']:
 if s['id']=='reference-scope-derivation':
  s['details']='Historical preV4 narrow scope finding at sectionSHA37faa95f02e041ded9cfbf85895630693204fe32d37c5c6a90896fe559f7a531. Exact originalwhole-post-training quotes and issue remain immutable in own old analysis and77ff report. V4 completedSFT/PPO replacement is verified separately byreference-v4-verification.'
 if s['id'] in ['dpo','instructgpt','recipe']:
  s['inspection_note']+=' SameownerV4 2026-10-06: original necessary method paragraphs/contracts locally reread and exact hashes checked; correctedV4 completedSFT/PPO boundary matches support. No refetch or modelrerun.'
for c in report['claims']:
 if c['id']=='old-and-reference':
  c['status']='verified';c['statement']='PPO以收集時保存的critic值計固定優勢；old同輪固定下輪更換，reference保留PPO開始時已完成示範微調的策略，在這段PPO期間固定。'
  c['scope']='This finitePPO run: collection-time oldrecords versus completedSFT initialpolicy heldfixed as reference; no allpost-training interval.'
 if c['id']=='reference-whole-posttraining-scope':
  c['prior_claim_history']={'statement':c['statement'],'status':c['status'],'source_sha256':prior['source_sha256'],'preserved_in':PREFIX+'/prior-opaque/revise-13.14-report.json'}
  c['statement']='正文reference保留PPO開始時已完成示範微調策略、在這段PPO期間固定；可見reference卡為PPO起點、整段PPO期間不更新。'
  c['scope']='CompletedSFT policy fixed for the subsequent finitePPO stage, as supported by the original paper/recipe; prior broadwording is resolved and preserved as history.';c['status']='verified'
 if c['id'] in ['old-and-reference','reference-whole-posttraining-scope']:
  for e in c['evidence']:
   if e['source_id']=='reference-scope-derivation':e['supports']='Historical exact broaderwording and issue preserved; current resolution is separately checked byreference-v4-verification.'
  c['evidence'].append({'source_id':'reference-v4-verification','locator':'v4-reinspection-receipt.json /issue_resolution and originalrecipe289/326, DPOv3§3Eq3, InstructGPTv1§3.5Eq2','supports':'Personally verifies current completedSFT/PPO-only fixedreference boundary and corresponding visible/accessibleSVG.'})
  c['artifact_ids']+=['reference-v4-inspection','reference-v4-render','reference-v4-receipt']
issue=report['issues'][0]
issue['prior_open_version']={'source_sha256':prior['source_sha256'],'report_sha256':'77ff086690cd25ba76569f3549dfea9ae019e2606b30103970ea22ed05a0b9d2','original_status':'open','prior_opaque':receipt['prior_full_revise_opaque']}
issue['status']='resolved';issue['resolved_on']='2026-10-06'
issue['resolution']=receipt['issue_resolution']+' Independently rendered/viewed currentSVG; current sourceSHA'+facts['source_sha256']+' and SVG SHA'+facts['figure_sha256']+'. Exact prior quotes/source/support/checkerexit1 remain in ownopaque history.'
issue['current_quotes']={'body':facts['current_body_quote'],'figure_labels':facts['current_visible_figure_labels'],'accessible_description':facts['current_accessibility_description']}
issue['resolution_artifact_ids']=['reference-v4-inspection','reference-v4-receipt','reference-v4-render','reference-v4-analysis']
report['checks']['factual_accuracy'].update(status='pass',details='SameownerV4 complete currenttext and necessary13.12 personally read. The unresolvedreference boundary is now explicitly completedSFT/PPO and matches originalpaper/recipe. All other ownfact/math/API support carried after exact unchangedhash checks.',claim_ids=[c['id'] for c in report['claims']])
report['checks']['figure_consistency'].update(status='pass',details='NewcorrectedSVG independently Inkscape-rendered/viewed; reference title/time label and accessible description matchcompletedSFT/PPO-only currentbody. Required previousscopeissue resolved. Priorhistorical renders preserved, not mislabelled asV4.',claim_ids=['old-and-reference','reference-whole-posttraining-scope'])
report['checks']['source_verification']['details']='OriginalDPOv3§3, InstructGPTv1§3.1/§3.5 and recipe273–290/326–354 locally personally reread forV4 support; originalsnapshots/code/currentmeasurements hashes match.97 own priorartifact exactfingerprints; source/SVG/currentctx changedversions documented. No rootparity or thirdpartyverdict used.'
report['checks']['limitations'].update(status='pass',details='Scope now limited to completedSFT policy fixed through thisPPO run. Actual narrowSVGview/localpage check, original numeric evidence carried; no new modelCPU, GPU, fulltraining or download. PriorREVISE and failedchecker history retained; no unresolveddependencies. Fullpage/mobile layout not rereviewed.',claim_ids=['reference-whole-posttraining-scope','gradient-only','small-networks','recipe-kl-target','original-records'])
report['checks']['numeric_verification']['details']='Bothcurrentfences byte-identical and current originalhelper/recipe/rawmeasurements unchanged. Own actualCPU variants and64×3 rawtrace arithmetic reused after fingerprints; no unrelated numeric/modelrerun.'
assert all(c['status']=='verified' for c in report['claims']);assert report['issues'][0]['status']=='resolved'
assert len({e['id'] for e in report['artifacts']})==len(report['artifacts'])
for e in report['artifacts']:assert sha(e['path'])==e['sha256']
assert sha('docs/technical-reviews/13.14.json')=='77ff086690cd25ba76569f3549dfea9ae019e2606b30103970ea22ed05a0b9d2'
current=(ROOT/'course/chapters/13.md').read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',current));i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 13.14 '));assert current[heads[i].start():heads[i+1].start()]==(A/'current-13.14.md').read_bytes()
raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode();(A/'current-report.json').write_bytes(raw);canonical=ROOT/'docs/technical-reviews/13.14.json';canonical.write_bytes(raw);assert canonical.read_bytes()==raw
print(json.dumps({'verdict':'pass','source_sha256':report['source_sha256'],'figure_sha256':report['figure_sha256'],'report_sha256':hashlib.sha256(raw).hexdigest(),
 'prior_full_revise_opaque':receipt['prior_full_revise_opaque'],
 'inspection_artifact':next(e for e in report['artifacts'] if e['id']=='reference-v4-inspection'),
 'receipt_artifact':next(e for e in report['artifacts'] if e['id']=='reference-v4-receipt'),
 'scope':receipt['support_scope'],'unresolved_dependencies':[]},ensure_ascii=False,indent=2))
