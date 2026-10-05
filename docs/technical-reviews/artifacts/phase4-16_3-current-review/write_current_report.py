import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
P=Path(__file__).resolve().parent
RP=P.relative_to(ROOT)
H=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
receipt=json.loads((P/'reinspection-receipt.json').read_bytes())
meta=json.loads((P/'input/extraction.json').read_bytes())
assert receipt['current_source_sha256']==meta['source_sha256']=='180fd63a4635b089541e5084a0520f00595242ac39ebf9483672bc797edec5af'
# Original claim/source/artifact declarations are reused only after this owner's
# independent current-text/source/contract/support checks recorded in the receipt.
# The previous verdict is never used to decide the current verdict.
report=json.loads((P/'history/16.3-prior.json').read_bytes())
assert report['reviewer_task']==receipt['reviewer_task']=='/root/phase4_factual_coordinator/factual_16_3'
checks={row['claim_id']:row for row in receipt['per_claim_current_inspection']}
assert set(checks)=={row['id'] for row in report['claims']}
assert all(row['status']=='verified' for row in checks.values())
for row in report['artifacts']:
    assert H(ROOT/row['path'])==row['sha256']
artifacts=report['artifacts']
def add(identifier, filename, kind, description, **kwargs):
    path=RP/filename
    artifacts.append(dict(id=identifier,path=str(path),sha256=H(ROOT/path),kind=kind,description=description,**kwargs))
for identifier, filename, kind, description in [
 ('current-section','input/section.md','source_snapshot','Current16.3 UTF-8 bytes personally read in full.'),
 ('current-full-frozen','input/chapter16-current-frozen-input.md','source_snapshot','Actual current full chapter bytes frozen at follow-up; not chapter-introduction review.'),
 ('current-needed-prefix','input/16.2-current-needed-prefix.md','source_snapshot','Entire needed current16.2 personally read; identical bytes to prior needed previous section.'),
 ('current-source-diff','input/section-source-diff.txt','source_snapshot','Direct original-text comparison: only explicit3-prefix/fourth-ID transition sentence changed; no author revision notes.'),
 ('current-fence','input/fence-1.py','code','Current unchanged original Python fence, SHA matched to original own execution.'),
 ('current-figure-svg','input/current-cache-append.svg','source_snapshot','Served current SVG bytes personally SHA matched to current canonical and prior figure.'),
 ('current-page-html','input/current-page.html','source_snapshot','Actual HTTP200 local current page bytes used for browser screenshots.'),
 ('current-reinspection-receipt','reinspection-receipt.json','source_snapshot','This same independent owner’s actual current-version source/claim/authority/visual/reuse inspection receipt.'),
 ('current-history-report','history/16.3-prior.json','source_snapshot','Opaque preserved own previous report; previous verdict not evidence for current decision.'),
 ('current-history-archive','history/prior-proof.tar.gz','source_snapshot','Opaque archive of all38 original evidence files; no runtime symlinks or weights.'),
 ('current-history-manifest','history/prior-manifest.json','source_snapshot','Prior report/archive SHA plus all38 unchanged evidence hashes.'),
 ('current-reuse-code','verify_reuse.py','code','Actual current reuse gates and raw-measurement arithmetic; zero new model-forward/training calls.'),
 ('current-render-code','render_current_page.py','code','Actual bounded Playwright page rendering at1280×800/390×844.'),
 ('current-mobile-inspection-code','inspect_mobile_code.py','code','Actual mobile code/output DOM inspection; first PREscroll attempt recorded honestly.'),
 ('current-scroll-code','inspect_mobile_scroll.py','code','Actual mobile CODEcontainer local horizontal-scroll check and right-edge screenshot.'),
 ('current-receipt-code','write_receipt.py','code','This owner’s receipt generation code.'),
 ('current-report-code','write_current_report.py','code','This owner’s current report update code after substantive reinspection.'),
]: add(identifier,filename,kind,description)
for name in ['desktop-top','desktop-figure','desktop-code','desktop-measurement','mobile-top','mobile-figure','mobile-code','mobile-measurement','mobile-code-right','mobile-output','mobile-code-actual-right']:
 add('current-'+name,'execution/'+name+'.png','figure_render','Actual current page screenshot personally viewed: '+name+'.')
for name in ['desktop','mobile']:
 add('current-'+name+'-full-page','execution/'+name+'-full-page.png','figure_render','Current full-page capture at '+name+' width; not used as sole visual evidence, targeted viewport screenshots actually viewed.')
env={'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','Chromium':'151.0.7922.173'}
for identifier, stem, command, result in [
 ('current-reuse-run','verify-reuse','.venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/verify_reuse.py','exit0; unchanged fence/bootstrap/code/figure/rawmeasurement; originalGitmatched; two25-position12-step rawmeasurement array/maxima/EOS/decode calculations passed; all38 prior evidence hashes unchanged; zero modelforwards.'),
 ('current-render-run','render-current-page','timeout 45s .venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/render_current_page.py','exit0; current HTTP200page rendered at1280×800 and390×844; figure complete; screenshots captured and targeted views personally inspected.'),
 ('current-mobile-run','mobile-code-inspection','timeout 30s .venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/inspect_mobile_code.py','exit0; currentcode/preoutput text observed; PREscroll not effective, diagnosed without hiding the initial attempt.'),
 ('current-scroll-run','mobile-scroll','timeout 25s .venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/inspect_mobile_scroll.py','exit0; CODElocal overflow:auto,client375/scroll541,actualscrollLeft166; right-sidecache/allclose screenshot personally viewed.'),
]:
 add(identifier,'execution/'+stem+'.stdout.txt','execution',result,command=command,result=result,environment=env)
 add(identifier+'-stderr','execution/'+stem+'.stderr.txt','source_snapshot','Actual stderr for '+identifier+'.')
source_id='current-reinspection-execution'
report['sources'].append(dict(id=source_id,kind='execution',title='Current owner source/reuse/provenance and raw-measurement checks',verified=True,artifact_id='current-reuse-run'))
first=meta['section_first_line']
text=(P/'input/section.md').read_text()
line=lambda needle:first+text[:text.index(needle)].count('\n')
locations={
 'causal-cache':f'course/chapters/16.md#16.3 lines{line("模型讀完")},{line("沿用")},{line("圖中的")}',
 'original-fence-api':f'16.3 current Python fence starts source line{meta["python_fences"][0]["code_line"]}; explanation line{line("`ids[:,:3]`取")}',
 'position-and-prefix':f'16.3 current lines{line("位置也必須")},{line("練習只把")}',
 'padding-limit':f'16.3 current line{line("多段輸入")}',
 'query-kv-heads':f'16.3 current line{line("我們也逐步")}',
 'raw-cache-measurements':f'16.3 current lines{line("我們也逐步")},{line("原始ID中")}; current16.2 measurement conditions',
 'floating-scope':f'16.3 current lines{line("`ids[:,:3]`取")},{line("我們也逐步")}',
 'near-tie':f'16.3 current line{line("原始ID中")}',
}
for row in report['claims']:
 row['status']=checks[row['id']]['status']
 row['location']=locations[row['id']]
 row['current_reinspection']=dict(artifact_id='current-reinspection-receipt',inspection=checks[row['id']]['inspection'])
 row['artifact_ids'].append('current-reinspection-receipt')
 if row['id'] in {'original-fence-api','position-and-prefix','padding-limit','raw-cache-measurements'}:
  row['artifact_ids'].append('current-reuse-run')
  row['evidence'].append(dict(source_id=source_id,locator='Current verify-reuse.stdout and same-owner receipt per_claim_current_inspection',supports='Source/dependency/code/provenance/support gates for reuse; raw measurements recalculated. No new model execution claimed.'))
  row['verification']['current_execution_scope']='Model-forward values reused from this owner’s unchanged prior actual execution after current dependency/support gates; current raw measurement arithmetic separately executed.'
 if row['id']=='causal-cache':
  row['artifact_ids'].extend(['current-desktop-figure','current-mobile-figure','current-figure-svg'])
 if row['id']=='original-fence-api':
  row['statement']='現在原fence取前三ID作prefix，再送第四ID；比較讀完ID4的full最後位置與cached新位置0，兩路shape(1,264)，原assert通過；eval/no_grad/allclose契約不變，未更新參數。'
report['source_sha256']=meta['source_sha256']
report['figure_sha256']=meta['figure_sha256']
report['verdict']=receipt['verdict']
report['issues']=[]
report['initial_review_frozen_input']=report['frozen_input']
report['frozen_input']={'path':str(RP/'input/chapter16-current-frozen-input.md'),'sha256':H(P/'input/chapter16-current-frozen-input.md'),'meaning':'Actual full chapter bytes frozen at first read of this follow-up; not later current whole-chapter version or chapter-introduction-review claim.'}
report['intro_applicability']=False
report['intro_sha256']=None
report['intro_scope']='Non-first section16.3; introduction not substantively reviewed by this owner.'
report['actual_reading_scope']='This same original independent owner personally read entire current16.3 and entire needed current16.2, current AST-located contracts, original method/raw measurements and relevant original authority snapshots again; actually rendered/viewed current1280×800 and390×844 page/figure and mobilelocalcodescroll. Newmodelforwardnotrequired: exactfence,all relevantcode/dependencies,numbers/axes/unit/denominatorsunchanged. Originalownexecutionreused only after current source/support/provenance gates.'
report['reinspections']=[{'id':receipt['id'],'reviewer_task':receipt['reviewer_task'],'artifact_id':'current-reinspection-receipt','path':str(RP/'reinspection-receipt.json'),'sha256':H(P/'reinspection-receipt.json'),'source_sha256':meta['source_sha256'],'prior_report_path':str(RP/'history/16.3-prior.json'),'prior_report_sha256':receipt['previous_report']['sha256'],'meaning':'Actual same original independent-owner current-version reinspection; not a fresh replacement reviewer or hash-only refresh.'}]
report['checks']['factual_accuracy']['details']='Owner本人重讀currentcomplete16.3/needed16.2，逐項按原主源/officialAPI/currentcontracts重新查核。唯一新transition3prefix+fourthID與unchangedfence/currentfigure一致；no newalgorithm/axis/unit/denominator/result introduced。各claimcurrentreceipt有actualinspection。'
report['checks']['numeric_verification']['details']='Currentowner本人重新核raw25/12/IDarrays/maxima/EOS/decode；原modelforward數字復用前已hash/supportgate：currentfence/bootstrap/code/權威原件不變。未把priorGPU或CPU值說成currentmodelrerun。'
report['checks']['figure_consistency']['details']='CurrentlocalpageChromium151實際render+view1280×800/390×844；currentservedSVGmatchescanonicalSHA；old0/1/2、新3、length4、Queryall4labels/arrows皆可讀。開details看原測量文字；mobileCODE實際localhorizontal-scroll166px見rightedge，output可見；no wholepagehorizontaloverflow。'
report['checks']['source_verification']['details']='SameowneragainpersonallyreadHFv4.57.1、PyTorchv2.14.1及GQAv3相關原件snapshot/locator，支持範圍未變；currentrawJSON/sourceGit48a4f3e/hash复核；不是作者摘要或他人判定。'
report['checks']['limitations']['details']='NoGPU/training/data-modeldownload/newmodelscoreevaluation。Newmodelinference不需要且未run；只复用本人已執行證據afteridentity/supportgates。原FP32rawresult限定25位置/12步/每模型ownroutes；variablelengthcache不支持、relativeallclosetolerance和argmaxlimits仍記。Currentdesktop/mobilepage與figure已本人render/view；非introowner。'
path=ROOT/'docs/technical-reviews/16.3.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
own=json.loads(path.read_bytes())
assert own['reviewer_task']=='/root/phase4_factual_coordinator/factual_16_3'
assert own['source_sha256']==meta['source_sha256']
assert own['reinspections'][0]['sha256']==H(P/'reinspection-receipt.json')
print(json.dumps({'report':str(path.relative_to(ROOT)),'report_sha256':H(path),'source_sha256':own['source_sha256'],'reviewer_task':own['reviewer_task'],'own_reinspection_id':receipt['id'],'own_reinspection_sha256':H(P/'reinspection-receipt.json'),'verdict':own['verdict']},ensure_ascii=False))
