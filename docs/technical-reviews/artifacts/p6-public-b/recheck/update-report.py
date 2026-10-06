from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((BASE/'initial-formal-before-recheck.json').read_text())
current=json.loads((OUT/'current-source-manifest.json').read_text())
probe=json.loads((OUT/'probe-result.json').read_text())
versions=json.loads((OUT/'evidence-version-check.json').read_text())
assert not versions['evidence_errors'] and probe['normalize']['documented_equals_actual']
assert probe['package']['record_declarations']==12 and probe['package']['other_declarations']==8950
assert probe['package']['actual_files']==8962 and probe['package']['actual_suffix_counts']['.jsonl']==12
assert len(report['issues'])==2 and all(x['status']=='unresolved' for x in report['issues'])
card=next(f for f in report['files'] if '/model-cards/' in f['source'])
for f,c in zip(report['files'],current):
 assert f['source']==c['path']
 if f is not card:
  assert c['unchanged'] and f['source_sha256']==c['current_scope_sha256']
  f['recheck']={'scope_sha256_unchanged':True,'full_file_sha256_unchanged':f['full_file_sha256']==c['current_full_sha256'],'original_personal_checks_retained':True,'checks_rerun':False,'evidence_versions_revalidated':True}
card.update(source_sha256=current[2]['current_scope_sha256'],full_file_sha256=current[2]['current_full_sha256'],frozen_input=current[2]['snapshot'],source_version='Current complete 183-line file personally reread and checked after revision; original initial input/report retained',verdict='pass')
card['previous_source_sha256']=versions['initial_snapshot_sha256']
card['initial_frozen_input']='docs/technical-reviews/artifacts/p6-public-b/initial-input/docs/selftrained/model-cards/repository-README.md'
line_updates={'card-licenses':'lines67,181','card-frozen-test':'lines71,88','card-continuation':'line88','card-cer-thresholds':'line90','card-controls':'line92','card-cpu-train':'lines96-128','card-v4-route':'line132','card-v4-selected':'line134','card-v4-test':'line136','card-old-index':'lines140-177','card-old-size':'line142'}
for claim in card['claims']:
 if claim['id'] in line_updates:claim['location']=line_updates[claim['id']]
 if claim['id'] in ['card-text-exact','card-ocr']:
  claim['status']='verified';claim['artifact_ids']+=['recheck-normalization']
  claim['scope']+=' Current line86 now states the exact normalization character set and separately scored format. Initial contradictory wording and counterexample remain in the original report/issues.'
 if claim['id']=='card-data':
  claim['statement']='Fixed package contains12 JSONL record files and8950 other files (8335PNG,603WAV,and12 source/license/metadata files),8962 total; split28876/2435/3734 and original recordings62/15/30 remain accurate.'
  claim['artifact_ids']+=['recheck-normalization']
  claim['scope']='Actual immutable tar member names/sizes compared to exact manifest records/assets declarations;8950 other files include metadata and licenses, not solely media. Original recordings exclude augmented derivatives; no speaker IDs claimed.'
 if claim['id']=='card-cer-thresholds':
  claim['artifact_ids']+=['recheck-normalization']
  claim['scope']+=' Reread current lines86 and90 against original normalize/score_reply and small CPU counterexamples; CER remains computed from normalized prediction and fixed twelve-character gold labels.'
card['claims'].append({'id':'card-normalization-rule','kind':'software','statement':'String exact, OCR whole-string and CER predicates remove whitespace with Unicode re\\s+ then strip only the nine documented boundary punctuation/quote characters; format uses a separate raw-text predicate.','location':'line86','status':'verified','artifact_ids':['recheck-normalization'],'evidence':[{'source_id':'scripts/selftrained/evaluate.py','locator':'normalize197-198,format_pass227-238,score_reply259-278','supports':'Exact character codepoints match current inline text; OCR edit-distance input is normalized generated string, format uses its own rule.'}],'scope':'Personally checked all nine documented strip characters against AST-resolved literal, plus five CPU cases: whitespace/boundary punctuation removed; internal punctuation and backslashes remain. This clarifies existing scoring; it does not change numbers or model output.'})
for name in ['factual_accuracy','numeric_verification','source_verification','limitations']:
 card['checks'][name]['status']='pass'
card['checks']['factual_accuracy']['details']='Full current model card reread; initial exact-predicate wording resolved by explicit normalization rule, and8950other-files wording checked directly against original manifest/tar classification.'
card['checks']['numeric_verification']['details']='Original table numerators/denominators retained with unchanged raw evidence; updated classification independently counted12recordJSONL+8335PNG+603WAV+12metadata/license/sourcefiles=8962. Current normalization CPU examples confirm original exact/CER behavior.'
card['checks']['source_verification']['details']='17repositorysourceSHA,23initialartifactSHA and120originalrawinputSHA revalidated without differences; original external snapshots also verified below. Current source and frozen copySHA checked.'
for issue in report['issues']:
 issue['status']='resolved'
 issue['resolution']='Original reviewer personally reread complete current card183lines, compared lines76/80/86/90 with original normalize/score_reply and five CPU cases, confirmed all nine strip character codepoints exactly; current wording explicitly describes normalization and separate format scoring. Numeric counts unchanged.'
 issue['resolved_source_sha256']=card['source_sha256'];issue['resolution_artifact_ids']=['recheck-normalization','recheck-versions']
for ident,name,desc in [('recheck-normalization','probe-result.json','Actual current original normalizer/score predicates and immutable tar/manifest member classification checks; no generation or training'),('recheck-versions','evidence-version-check.json','Complete reread183lines, scopeSHAidentity ofotherfivefiles and original evidence versions verified'),('recheck-source','current-source-manifest.json','Current assignedscopeSHA and complete revisedmodelcard frozen bytes'),('recheck-diff','current-card.diff','Actual initial/current source diff retained without altering initial source')]:
 artifact={'id':ident,'kind':'source_snapshot','path':str((OUT/name).relative_to(ROOT)),'sha256':sha(OUT/name),'description':desc}
 if ident=='recheck-normalization':artifact.update(kind='execution',command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-b/recheck/recheck-probe.py > docs/technical-reviews/artifacts/p6-public-b/recheck/probe.stdout 2> docs/technical-reviews/artifacts/p6-public-b/recheck/probe.stderr',result='exit0; exact documented nine-character set equals AST-resolved actual literal; five scoring cases and immutable12+8950fileclassification checked',environment={'python':probe['environment']['python'],'device':'cpu','torch':'2.14.1+cpu'})
 report['artifacts'].append(artifact)
report['verdict']='pass'
report['review_history'].append({'phase':'original-reviewer-recheck','reviewer_task':'/root/p6_fact_public_b','initial_report':'docs/technical-reviews/artifacts/p6-public-b/initial-report.json','initial_report_sha256':sha(BASE/'initial-report.json'),'opaque_formal_backup':'docs/technical-reviews/artifacts/p6-public-b/initial-formal-before-recheck.json','opaque_formal_backup_sha256':sha(BASE/'initial-formal-before-recheck.json'),'old_model_card_sha256':versions['initial_snapshot_sha256'],'current_model_card_sha256':card['source_sha256'],'current_model_card_lines_read':'1-183,fullfile','issues_resolved':['card-text-exact','card-ocr'],'additional_changed_claim':'card-data other-fileclassification checked independently','other_scopes':'Five existing scopesSHAunchanged; original personal checks retained, not all rerun','evidence_version_errors':[],'new_generations':0,'training_runs':0,'verdict':'pass','body_edits_by_reviewer':False})
report['commands'].append({'command':report['artifacts'][-4]['command'],'result':'exit0; own initial raw-string fixture syntaxerror preserved separately, corrected only probe and rerun'})
report['method_events'].append({'event':'recheck fixture raw-string escaped-backslash syntax','result':'Own CPU fixture SyntaxError retained in recheck/probe-first-fixture.stderr; replaced with explicit codepoint92 concatenation and reran successfully. No repository code/body changed.'})
tree=BASE/'official/hf-v4-tree.json'
fetch=next(x for x in json.loads((BASE/'official/fetch-tree-followup.json').read_text()) if x['name']=='hf-v4-tree.json')
assert sha(tree)==fetch['sha256'] and len(json.loads(tree.read_text()))==2
report['sources'].append({'id':'hf-v4-tree.json','kind':'official_source','title':'Original immutable v4 public bundle anonymous tree metadata','url':fetch['url'],'version':'fixed revisiond3954d6900b3cf81e593d99d9b8b1a91e6f9741d; sourceSHA256'+fetch['sha256'],'verified':True,'checked_original':True,'accessed_on':'2026-10-06','authority_reason':'Original public artifact service metadata at the exact release commit, retrieved with no Authorizationheader','inspection_note':'Original two-filetree personally inspected during initialpass: README.md and release-provenance.json only; immutable snapshot is retained and its SHA rechecked now.'})
claim=next(x for x in card['claims'] if x['id']=='card-v4-selected')
claim['evidence'].append({'source_id':'hf-v4-tree.json','locator':'all two type=file records at immutable releaseprefix','supports':'The public bundle preserves README/provenance rather than redistributing upstream weights.'})
dest=ROOT/'docs/technical-reviews/public-pages-p6-b.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'claims':sum(len(f['claims']) for f in report['files']),'source':card['source_sha256'],'report_sha256':sha(dest)}))
