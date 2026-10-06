"""Write the new unresolved scope inspection without changing prior accepted history."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PREFIX='docs/technical-reviews/artifacts/phase4-13_14-reference-scope'
A=ROOT/PREFIX
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def load(name):return json.loads((A/name).read_bytes())
prior=load('prior-opaque/accepted-13.14-report.json')
facts=load('inspection-facts.json')
contexts=load('source-context.json')
assert sha(PREFIX+'/prior-opaque/accepted-13.14-report.json')=='14a2ddf6e20cfe47334edf1b1264b0c6d006e292d9077c33265d4bd9a7c065e7'
assert facts['source_sha256']=='37faa95f02e041ded9cfbf85895630693204fe32d37c5c6a90896fe559f7a531'
assert facts['figure_sha256']=='b18bbd67c517b81f3f215755d70c948156283de29aac2aa4f1a068381ec53459'
assert (A/'inspection.exit.txt').read_text().strip()=='0'
assert (A/'render.exit.txt').read_text().strip()=='0'

issue={
 'id':'reference-stage-scope','status':'open','severity':'substantive_technical_scope',
 'location':'course/chapters/13.md#13.14 figure-explanation paragraph/source line463; course/figures/rewrite-13-model-roles.svg reference-card x50,y928 and accessibility description',
 'exact_quotes':{'body':'reference保留整段後訓練起點。','visible_figure':'整段後訓練都不更新'},
 'problem':'The actual wording extends the reference checkpoint/fixed interval to all post-training. The original papers and inspected recipe establish the completed-SFT snapshot, fixed in the subsequent PPO/preference stage. Current7.17 explicitly includes SFT in post-training; current13.4 gives the completed-demonstration boundary, while13.12 repeats the broader term.',
 'source_locators':[
   {'source_id':'dpo','locator':'arXiv2305.18290v3 §3 PDFpp3–4, three-phase paragraph and immediately after Eq3','supports':'SFT precedes reward/RL and the reference is the initial SFT model.'},
   {'source_id':'instructgpt','locator':'arXiv2203.02155v1 §3.1 three steps; §3.5 PDFp9 Eq2','supports':'Supervised-policy training occurs before PPO; KL reference is the SFT policy.'},
   {'source_id':'recipe','locator':'run_posttraining278–287, _update285, reference289, PPO initialization326 and losses352–354','supports':'Policy is updated during SFT before a completed policy is copied and frozen as reference.'}
 ],
 'impact':'The broader start can lead a reader to choose pre-SFT weights as the reference checkpoint; this changes the compared distribution and KL regularization target. Within one PPO run, fixed reference versus per-rollout old records remains supported.',
 'suggested_revision':'Explicitly preserve the completed-SFT policy and state that it stays fixed during this PPO/preference-optimization stage. For example, body「reference保留示範微調完成後的起點，PPO期間固定」 and figure last line「PPO期間固定」; align the SVG accessibility description.',
 'related_context':'Current13.12 equivalent phrase may need its own owner’s review if changed; this report does not modify or judge another canonical section report.',
 'resolution':'Unresolved; no source or SVG edit authorized or performed by this reviewer.',
 'artifact_ids':['reference-scope-inspection','reference-scope-analysis','reference-scope-current-source','reference-scope-svg','reference-scope-render','reference-scope-recipe-excerpt','reference-scope-dpo-excerpt']
}
old_artifacts={x['id']:x for x in prior['artifacts']}
receipt={
 'schema_version':1,'kind':'same_owner_narrow_reference_scope_inspection','reviewed_on':'2026-10-06',
 'reviewer_task':'/root/phase4_factual_coordinator/factual_13_14','verdict':'revise',
 'canonical_report':'docs/technical-reviews/13.14.json',
 'canonical_source_artifact':{'id':'reference-scope-current-source','path':PREFIX+'/current-13.14.md','sha256':facts['source_sha256']},
 'canonical_newinspection_artifact':{'id':'reference-scope-inspection','path':PREFIX+'/inspection-facts.json','sha256':sha(PREFIX+'/inspection-facts.json')},
 'source':'course/chapters/13.md#13.14','source_sha256':facts['source_sha256'],'intro_sha256':None,
 'figure_sha256':{'course/figures/rewrite-13-model-roles.svg':facts['figure_sha256']},
 'accepted_prior_opaque':{'path':PREFIX+'/prior-opaque/accepted-13.14-report.json','sha256':sha(PREFIX+'/prior-opaque/accepted-13.14-report.json'),'prior_verdict':'pass'},
 'history_preservation':'Complete accepted report, accepted receipt/completion/manifest and original report/support/manifest saved opaque. Original proof remains unchanged. The previous approval is not rewritten; this is a new unresolved scope finding.',
 'actual_read_scope':'Complete current13.14; necessary current13.4,13.12,7.17 and glossary33–34; current original recipe273–290 and326–355 after AST inspection; own immutable DPOv3§3 and InstructGPTv1§3.1/§3.5 locally reread. Unrelated12.1 was accidentally extracted during context location and is not used as support.',
 'context_versions':contexts,
 'source_versions_and_snapshots':[
   {'source_id':source['id'],'version':source['version'],'url':source['url'],
    'original_snapshot':{'artifact_id':artifact_id,'path':old_artifacts[artifact_id]['path'],'sha256':old_artifacts[artifact_id]['sha256']},
    'locators':[x for x in issue['source_locators'] if x['source_id']==source['id']],
    'new_action':'Exact prior snapshot fingerprint checked; necessary original methods reread locally, no refetch.'}
   for source,artifact_id in [(next(x for x in prior['sources'] if x['id']=='dpo'),'dpo-pdf'),(next(x for x in prior['sources'] if x['id']=='instructgpt'),'instructgpt-pdf')]
 ],
 'exact_body_quote':facts['exact_body_quote'],'exact_figure_quote':facts['exact_figure_quote'],
 'support_scope':facts['supported_freeze_scope'],'issue':issue,
 'visual_inspection':{'render_command':'inkscape course/figures/rewrite-13-model-roles.svg --export-type=png --export-width=640 --export-filename='+PREFIX+'/reference-roles-current.png',
    'render_path':PREFIX+'/reference-roles-current.png','render_sha256':sha(PREFIX+'/reference-roles-current.png'),
    'render_exit':0,'view_image_performed':True,'own_observation':'Current reference card visibly says整段後訓練都不更新, while other roles explicitly mentionPPO期間.'},
 'carried_original_evidence':'All69 previously registered own artifacts rehashed exactly; current code/SVG/original raw JSON fingerprints additionally checked. Prior numeric/API/measurement CPU evidence carried without new model execution.',
 'unresolved_dependencies':['Required source/reference-card wording boundary correction by authorized writer, followed by same-owner technical reinspection.'],
 'limitations':['No paper refetch, model CPU rerun, GPU, full recipe, training or model download.','This is a narrow wording/source-contract review; no new full-page/mobile layout inspection.']
}
(A/'newinspection-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
report=copy.deepcopy(prior)
report['verdict']='revise'
report['reviewed_on']='2026-10-06'
report['source_sha256']=facts['source_sha256']
report['reference_scope_reinspection']={'kind':receipt['kind'],'reviewer_task':receipt['reviewer_task'],'verdict':'revise',
 'receipt_artifact_id':'reference-scope-receipt','receipt_path':PREFIX+'/newinspection-receipt.json','receipt_sha256':sha(PREFIX+'/newinspection-receipt.json'),
 'newinspection_artifact_id':'reference-scope-inspection','newinspection_path':PREFIX+'/inspection-facts.json','newinspection_sha256':sha(PREFIX+'/inspection-facts.json'),
 'accepted_prior_opaque':receipt['accepted_prior_opaque'],'unresolved_dependencies':receipt['unresolved_dependencies']}
report['read_scope']=receipt['actual_read_scope']
new_artifacts=[
 ('reference-scope-current-source','current-13.14.md','source_snapshot','Current full section, unchanged SHA; exact scope wording personally reread.'),
 ('reference-scope-context13-4','current-13.4.md','source_snapshot','Necessary current completed-demonstration reference context.'),
 ('reference-scope-context13-12','current-13.12.md','source_snapshot','Necessary current old/reference context and equivalent broad phrase.'),
 ('reference-scope-context7-17','current-7.17.md','source_snapshot','Necessary prior post-training definition explicitly includesSFT; current textbook context, not authority.'),
 ('reference-scope-glossary','glossary-posttraining-lines.txt','source_snapshot','Current glossary33–34 demonstrating locally defined post-training scope.'),
 ('reference-scope-context-versions','source-context.json','source_snapshot','Actual raw context/source snapshot hashes and read ranges, including unused12.1 retrieval.'),
 ('reference-scope-svg','current-model-roles.svg','source_snapshot','Exact unchanged current SVG.'),
 ('reference-scope-render','reference-roles-current.png','figure_render','Actual current640px Inkscape render loaded with view_image; reference scope label visible.'),
 ('reference-scope-render-exit','render.exit.txt','source_snapshot','Actual Inkscape exit0.'),
 ('reference-scope-render-stderr','render.stderr.txt','source_snapshot','Actual Inkscape render stderr.'),
 ('reference-scope-render-version','inkscape-version.txt','source_snapshot','Actual Inkscape version.'),
 ('reference-scope-recipe-excerpt','recipe-sft-reference-lines.txt','source_snapshot','Exact original recipe273–290 with line numbers; SFT update before reference copy.'),
 ('reference-scope-dpo-excerpt','dpo-reference-lines.txt','source_snapshot','Exact original DPOv3§3 source-text methods excerpt with line locators.'),
 ('reference-scope-instructgpt-excerpt','instructgpt-pipeline-lines.txt','source_snapshot','Exact original InstructGPTv1§3.1 three-phase methods excerpt.'),
 ('reference-scope-analysis','own-scope-analysis.md','derivation','Own exact quotes, source boundary, impact, correction suggestion and honest prior-approval history.'),
 ('reference-scope-fingerprints','verified-fingerprints.json','derivation','Actual prior69 artifacts and4 current source/code/measurement/SVG checks.'),
 ('reference-scope-code','inspect_reference_scope.py','code','Actual narrow AST/source/fingerprint inspection program; no model execution.'),
 ('reference-scope-command','commands.json','source_snapshot','Actual commands and narrow source/visual read scope.'),
 ('reference-scope-env','environment.json','source_snapshot','Actual Python/PyTorch distribution/shell and non-model analysis environment.'),
 ('reference-scope-stderr','inspection.stderr.txt','source_snapshot','Actual narrow inspection stderr.'),
 ('reference-scope-exit','inspection.exit.txt','source_snapshot','Actual narrow inspection exit0.'),
 ('reference-scope-method','current-factual-method.md','source_snapshot','Latest factual method personally read.'),
 ('reference-scope-protocol','current-review-protocol.md','source_snapshot','Latest clear-tutorial review protocol personally read.'),
 ('reference-scope-prior-opaque','prior-opaque/accepted-13.14-report.json','source_snapshot','Complete accepted prior PASS report saved opaque unchanged.'),
 ('reference-scope-prior-backups','prior-opaque/backup-record.json','source_snapshot','Actual complete accepted/original history copy paths and fingerprints.'),
 ('reference-scope-receipt','newinspection-receipt.json','derivation','Canonical source/current inspection artifact IDs, paths and SHA plus actual unresolved source boundary judgment.'),
 ('reference-scope-writer','write_scope_report.py','code','Actual generator of new revise report; prior files remain immutable.')]
for identifier,filename,kind,description in new_artifacts:
 path=PREFIX+'/'+filename;report['artifacts'].append({'id':identifier,'path':path,'sha256':sha(path),'kind':kind,'description':description})
report['artifacts'].append({'id':'reference-scope-inspection','path':PREFIX+'/inspection-facts.json','sha256':sha(PREFIX+'/inspection-facts.json'),
 'kind':'execution','description':'Actual narrow source/AST/fingerprint facts, not model computation.',
 'command':'.venv/bin/python '+PREFIX+'/inspect_reference_scope.py',
 'result':'exit0; current section/SVG and all prior proof fingerprints exact; SFT optimizer update285 precedes frozen reference copy289; exact broad wording recorded.',
 'environment':{'python':'3.13.5','torch':'2.14.1+cpu','device':'No model computation; CPU source/AST/hash inspection','shell':'bash login:false'}})
report['sources'].append({'id':'reference-scope-derivation','kind':'derivation','title':'Own narrow comparison of exact wording and source training boundary','verified':True,
 'details':'Current text/visible figure saywhole post-training, while DPOv3§3 and original recipe278–289 place reference after SFT. Current7.17 includesSFT in post-training. Own-scope-analysis.md records the unresolved broader scope and its reference-checkpoint impact.'})
for source in report['sources']:
 if source['id'] in ['dpo','instructgpt','recipe']:
  source['inspection_note']+=' Same owner local reinspection2026-10-06: necessary original stage-boundary methods reread, exact immutable fingerprint checked; completed-SFT reference supports subsequent preference/PPO stage, not all post-training includingSFT.'
for claim in report['claims']:
 if claim['id']=='old-and-reference':
  claim['status']='partially_supported'
  claim['scope']='Within one PPO run, collection-time old records versus a completed-SFT fixed reference is verified. Current exact whole-post-training wording is materially broader and unresolved.'
  claim['evidence'].append({'source_id':'reference-scope-derivation','locator':'newinspection-receipt.json /issue, own-scope-analysis.md','supports':'Identifies precise broader textual/figure claim that is not covered by the prior initial-SFT paraphrase.'})
  claim['artifact_ids'].append('reference-scope-inspection')
report['claims'].append({'id':'reference-whole-posttraining-scope','kind':'concept',
 'statement':'正文「reference保留整段後訓練起點。」與可見圖「整段後訓練都不更新」。',
 'location':issue['location'],'scope':'Actual full post-training scope includes SFT in necessary current reader context; original paper/recipe only establishes completed-SFT reference fixed during subsequent PPO/preference stage.',
 'status':'needs_revision','evidence':copy.deepcopy(issue['source_locators'])+[{'source_id':'reference-scope-derivation','locator':'own-scope-analysis.md current7.17/glossary/13.4 comparison','supports':'Shows why necessary context does not make broader literal wording accurate.'}],
 'artifact_ids':issue['artifact_ids']})
report['issues'].append(issue)
report['checks']['factual_accuracy'].update(status='revise',details='New narrow exact-quote/source-stage inspection finds unresolvedwhole post-training reference scope; correct checkpoint is completed-SFT. All other own factual/math/API evidence retained with exact fingerprints.',claim_ids=['old-and-reference','reference-whole-posttraining-scope'])
report['checks']['figure_consistency'].update(status='revise',details='Actual current SVG rendered/viewed: reference card visibly stateswhole post-training, while primary methods/recipe establish completed-SFT reference forPPO. Required boundary correction remains open.',claim_ids=['reference-whole-posttraining-scope'])
report['checks']['source_verification']['details']='Own original DPOv3 and InstructGPTv1 methods locally reread for narrow stage boundary, no refetch; current original recipe AST/SFT-update/copy ranges personally checked. All prior own69 artifacts plus current sources rehashed unchanged. Exact primary locators support completed-SFT reference and expose wording scope gap.'
report['checks']['limitations'].update(status='revise',details='Literal body/figure reference interval extends beyond source support and requires explicit completed-SFT/PPO boundary. Prior accepted history preserved unchanged. No new math/modelCPU/training/GPU or download; this is narrow scope inspection.',claim_ids=['reference-whole-posttraining-scope'])
report['checks']['numeric_verification']['details']='Own previously executed numeric/API variants and original64×3 trace arithmetic remain exact by unchanged fingerprints; no new numeric/model computation required or claimed for this scope wording issue.'
assert len({x['id'] for x in report['artifacts']})==len(report['artifacts'])
for entry in report['artifacts']:assert sha(entry['path'])==entry['sha256']
assert sha('docs/technical-reviews/13.14.json')=='14a2ddf6e20cfe47334edf1b1264b0c6d006e292d9077c33265d4bd9a7c065e7'
raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
(A/'new-canonical-report.json').write_bytes(raw)
canonical=ROOT/'docs/technical-reviews/13.14.json';canonical.write_bytes(raw);assert canonical.read_bytes()==raw
print(json.dumps({'verdict':'revise','source_sha256':report['source_sha256'],'report_sha256':hashlib.sha256(raw).hexdigest(),
 'accepted_prior_opaque':receipt['accepted_prior_opaque'],
 'canonical_newinspection_artifact':next(x for x in report['artifacts'] if x['id']=='reference-scope-inspection'),
 'receipt':next(x for x in report['artifacts'] if x['id']=='reference-scope-receipt'),
 'exact_body_quote':facts['exact_body_quote'],'exact_figure_quote':facts['exact_figure_quote'],'support_scope':facts['supported_freeze_scope'],
 'unresolved_dependencies':receipt['unresolved_dependencies']},ensure_ascii=False,indent=2))
