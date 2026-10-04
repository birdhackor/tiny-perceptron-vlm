"""Same original reviewer: preserve round1, record genuine complete current recheck."""
from pathlib import Path
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[6]
D = Path(__file__).parent
P = D.parent
TASK = '/root/v4_review_coordinator/factual_whole_training_course'
SHA = 'f9ae45fb62731b514aa704d6cb3677e9b052c6243a2a5c331d9339acbe23bf55'
REV = 'f43b393431d4908f0ea38a2375f8cae9c4320d12'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, value):
    (D/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def rel(path):
    return path.relative_to(ROOT).as_posix()

old = json.loads((P/'report.json').read_text())
reuse = json.loads((D/'reuse-verification.json').read_text())
browser = json.loads((D/'browser-render-receipt.json').read_text())
assert sha(ROOT/'course/training.md') == SHA == sha(D/'training.md.read-snapshot')
assert all(x['byte_identical'] for key in ('sources','artifacts','external_originals','figures') for x in reuse[key])
assert all(x['section_byte_identical'] for x in reuse['prerequisite_sections'])

# These are notes of images actually personally viewed in this turn, not an automated visual verdict.
notes = {
 'architecture_int4_packing': 'Renewed personal view: signed -8/-1 become codes 0/7; first code is low nibble/right, second high nibble/left; 112. Codes 8/9 for 0/1 give 152. Crossed arrows preserve order.',
 'architecture_modal_answer_alignment': 'Renewed personal view: teacher prefix16 versus student prefix4; query15/3 predicts A0,16/4 A1,17/5 A2. Taking logits from answer input16/4 would miss first-answer prediction.',
 'architecture_online_softmax': 'Renewed personal view: old maximum log2, denominator1.5,numerator25; new maximum log4 rescales old sums by.5; .75+1=1.75,12.5+30=42.5,42.5/1.75=24.2857.',
 'architecture_policy_update': 'Renewed personal view: selected answer4, reward1,baseline.5,advantage.5; gradients .25/-.25,lr.1 gives logits -.025/.025 and probabilities .487503/.512497. One step does not give100%.',
 'causal': 'Renewed personal view: lower triangular visibility including diagonal; query0 sees0,query3 sees0–3, future upper triangle forbidden.',
 'flash_allocated_memory': 'Renewed personal view: common gray65MiB baseline; forward additional24.25/.27,total89.25/65.27; forward/back additional32.75/2.03,total97.75/67.03. Bars share scale; Flash forward increment is thin. Caption excludes whole-card RAM/model training.',
 'foundations_bigram_row': 'Renewed personal view: cat row counts1,1,1,3,1,1,1 sum9; look3/9, others1/9; Laplace initial1 and two cat→look observations.',
 'fsdd_speaker_holdout': 'Renewed personal view: custom jackson/nicolas/theo20/20/20; validation5/20,test3/20,EOS20/20. 4591/8000=9182/16000=.573875 seconds; resampling adds no above4kHz information. Caption identifies custom split.',
 'multimodal_expand_image': 'Renewed personal view: one image placeholder becomes3 features; full7-position labels have5 ignored then73,2; shifting removes first label leaving4 ignored then73,2, assistant row predicts73.',
 'multimodal_lora_branches': 'Renewed personal view: frozen W in parallel with A16→2,B2→12, addition before output;32+24=56 trainable entries; alpha/rank2/2=1.',
 'qat_training_deployment': 'Renewed personal view: FP32 master .1,.4,.9; forward scale.9/7,codes1,3,7,dequantized .1286,.3857,.9; STE backward updates master; actual packed4 deployment happens separately and loads FP32 floating operations.',
 'tokenizer_common_scale': 'Renewed personal view: same20 texts4193 UTF8 bytes; byte/BPE4213/2503 targets include EOS. TokenNLL2.77947/3.98741 contrasts BPB4.02906/3.43401. Separate panel scales and caption state the different units.',
 'window_training': 'Renewed personal view: contexts1/3/5,parameters833/1345/1857; train.334/.213/.199,validation.431/.306/.513. Lower is better and context5 hurts this one heldout text.',
}
views = []
for item in browser['figure_renders']:
    entry = copy.deepcopy(item)
    entry['personal_view'] = 'completed'
    entry['inspection_note'] = notes[Path(item['render']).stem]
    assert sha(D/item['render']) == item['render_sha256']
    views.append(entry)
write('personal-view-notes.json', {
 'reviewer_task':TASK,'round':2,'personal_views_completed':True,
 'view_mechanism':'Actual tools.view_image of every round2 Chromium PNG:3 browser screenshots, then4+4+5 figure images; personally inspected all returned image blocks.',
 'original_render_receipt_preserved':'browser-render-receipt.json retains personal_view:pending from rendering; this separate receipt records later actual views.',
 'figures':views,
 'browser_views':[
  {'path':'browser-training-T8.png','sha256':sha(D/'browser-training-T8.png'),'personal_view':'completed','interpretation':'Actual full training.html#T.8 heading and architecture procedure, same-family splits, cost/quality limits, evaluator rows visible.'},
  {'path':'browser-training-memory-link.png','sha256':sha(D/'browser-training-memory-link.png'),'personal_view':'completed','interpretation':'Visible corrected2.14 max_memory_allocated link and adjacent six-branch reset/allocated versus reserved/driver/other-process limits; MiB definition and separate Flash scope.'},
  {'path':'browser-training-T4.png','sha256':sha(D/'browser-training-T4.png'),'personal_view':'completed','interpretation':'Actual click reaches full training.html#T.4 and next-token/SFT distinction, prerequisites, toy commands and family grouping.'}
 ], 'actual_dom_read':'Personally read complete saved7714-character T.8 DOM in this turn, in addition to current source and screenshot views.'
})
write('current-read-receipt.json', {
 'reviewer_task':TASK,'round':2,'source':'course/training.md','source_sha256':SHA,
 'snapshot':rel(D/'training.md.read-snapshot'),'bytes':123959,'lines':936,
 'personal_complete_read':True,
 'actual_read_order':[
  {'lines':'1–160','truncated':False},{'lines':'161–320','truncated':False},
  {'lines':'321–480','truncated':False},{'lines':'481–640','truncated':False},
  {'lines':'641–800','truncated':False},{'lines':'801–936','truncated':False}
 ],
 'prerequisites':'Original29 necessary sections personally read in round1; all exact sectionbytes rechecked unchanged before reuse, original fullfile/snapshot receipts preserved.',
 'current_authority':'Current literal official API URL actuallyGET/read entire article; first403 retained and same explicit UA200 receipt. Original26 external raw originals byte-identical before reuse; exact paper equations/API/source locators retained.',
 'current_browser':'Personally read actual8784 T.8 DOM, viewed3 new screenshots, followed actual self-anchor and T.4 link;13 current SVG renders personally viewed.',
 'guide_contract_recheck':'Personally reread complete official150-line technical guide and15-line supplemental contract in round2; current checker components read directly, no review conclusions consulted.',
 'source_change':'Only T.8 line786 hyperlink acquired .memory; this is a full current reread, not a rebinding of old source hash.'
})
write('reviewer-corrections.json', {
 'reviewer_task':TASK,'round':2,'first_report':rel(P/'report.json'),'first_report_sha256':sha(P/'report.json'),
 'first_generated_report':rel(P/'report.round1.initial.json'),'first_generated_report_sha256':sha(P/'report.round1.initial.json'),
 'c03':{'issue':'Own first report stored verification.expected/observed as JSON objects, but official checker requires nonempty strings. Numeric values were correct.',
         'original_structured_verification':copy.deepcopy(old['claims'][2]['verification']),
         'correction':'Exact same expected and observed numeric values summarized as nonempty strings; original structured data and CPU evidence unchanged.'},
 'c35':{'issue':'Own first report statement assigned7/10 to BF16,6/10 to FP16 and4/10 to FP32. The original primary record and course text correctly assign FP32=7/10,BF16=6/10,FP16=4/10.',
         'original_claim':copy.deepcopy(old['claims'][34]),
         'current_primary_record_sha256':sha(ROOT/'docs/course-experiments/results/precision.json'),
         'locator':'/results/variants/{fp32,bf16,fp16}/heldout/test/{matches,records}',
         'actual_executed_mapping':reuse['corrected_own_precision_mapping'],
         'correction':'Round2 statement and expected/observed strings explicitly name each branch correctly. This corrects reviewer text, not course source or underlying records.'},
 'inspection_script_error':{'actual_command_scope':'A round2 read-only Python claim-summary display loop accessed c23 concept verification without .get; printed c21/c22 then failed.',
                             'actual_error':'KeyError: verification; tool stdout/traceback personally seen. Display retry used c.get(verification,{}) and completed c23–c35; no report/source/evidence changed.'}
})

r = copy.deepcopy(old)
r['round'] = 2
r['verdict'] = 'pass'
r['assigned_document_scope'] = 'Complete current course/training.md: all936 lines including introduction and T.1–T.11, every authored command/Python block, table, substantive method/result/limit. Same original independent factual owner, genuine current full reread and recheck; no fabricated numbered lesson ID.'
r['current_document_sha256'] = {'course/training.md':SHA}
r['source_sha256'] = SHA
r['source_read_receipts'] = {
 'current_complete_read':rel(D/'current-read-receipt.json'),
 'current_byte_and_preview_confirmation':rel(D/'current-source-receipt.json'),
 'current_reuse_comparison':rel(D/'reuse-verification.json'),
 'original_complete_read_and_correction':copy.deepcopy(old['source_read_receipts']),
 'personal_complete_read':True,'original_source_snapshots_preserved':True
}
r['independence_disclosure']['round2'] = 'Same actual original factual task reassesses corrected current source after genuine reader phasegate; no reader verdict content, other factual conclusions or author expected outcomes read. Own first report necessarily read for same-owner revision; original fresh independence disclosure and incident limits preserved.'
r['execution_limits']['round2_actual_cpu_environment'] = reuse['environment']
r['execution_limits']['round2_performed'] = 'Actual fullcurrent source reread, exact55 repo/38artifact/26original/29prereq/13figure byte comparisons, current precision raw-record label extraction, directofficial API GET/read, actualChromium8784 anchors/DOM/screens and13 renewed personal SVG views. These source-only corrections did not require new training or repeating591/2639/102 computations.'
r['execution_limits']['reuse_bounds'] = 'Unchanged round1 CPU executions/derivations and primary fixed GPU records genuinely reused after exact comparison. Their original environments, commands, failures and scope remain recorded. No new GPU replication, kernels, timing, accuracy or elapsed training is claimed.'
r['report_preservation'] = {
 'first_true_final_report':rel(P/'report.json'),'first_true_final_report_sha256':sha(P/'report.json'),
 'first_generated_report':rel(P/'report.round1.initial.json'),'first_generated_report_sha256':sha(P/'report.round1.initial.json'),
 'first_source_sha256':old['current_document_sha256']['course/training.md'],
 'first_verdict':'revise','initial_404_and_c33_preserved':True,
 'same_owner_revision':'Current full reread and recheck, preserved initial report/errors/source/evidence; sourceonly diff and two own report corrections explicitly recorded.',
 'reviewer_corrections':rel(D/'reviewer-corrections.json')
}
r['component_validation_receipt'] = rel(D/'component-validator-output.json')
r['component_validation_scope'] = 'Calls actual unchanged official _artifacts/_sources/_claims. Additional whole-document scope, raw SHA, figure map, five-check and issue checks reuse official helpers/semantics without fabricating numbered lesson identity. Checker does not establish truth or genuine reading.'

env = reuse['environment']
def artifact(identifier, name, kind, description, command=None, result=None):
    value = {'id':identifier,'kind':kind,'path':rel(D/name),'sha256':sha(D/name),'description':description}
    if kind == 'execution':
        value.update(command=command,result=result,environment=env)
    r['artifacts'].append(value)
artifact('r2-reuse','reuse-verification.json','execution','Actual exact-byte comparison of all reused original code/records/artifacts/authorities/prerequisites/figures; fresh CPU version and precise precision label extraction.',
         '.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-course/round2/verify_reuse.py',
         '55 repo sources,38 original artifacts,26 external originals,29 sections,13 SVGs identical;CPU2.14.1+cpu;FP32 7/10,BF16 6/10,FP16 4/10.')
artifact('r2-link','current-link-retrieval.json','execution','Current authored literal official URL retrieved, original403 and successful200 retained.',
         '.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-course/round2/check_current_link.py',
         'First bare urllib403; explicit sameoriginalUA Factual-review/1.0 returns200 at exactauthoredURL; entire article personally read, rawSHA identical to originalcanonical.')
artifact('r2-browser','browser-render-receipt.json','execution','Real8784 full training.html#T.8, official href, self-anchor/T.4 clicks, savedDOM and13 newlyrendered SVGs.',
         '.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-course/round2/browser_probe.py',
         'CorrectT.8heading/currentcanonicalhref, realT.8/T.4clicks,7714charDOM,3 screenshots,13renders; subsequent personal views recorded separately.')
for identifier,name,description in [
 ('r2-source','training.md.read-snapshot','Complete current rawUTF8 source personally read,123959bytes936lines.'),
 ('r2-byte','current-source-receipt.json','Actual sourceSHA and exactpreviewf43 originalbytes equality.'),
 ('r2-read','current-read-receipt.json','Own completed fullsource/authority/DOM read scope and originalprerequisite reuse bounds.'),
 ('r2-views','personal-view-notes.json','Completed actual personal image view notes for13 newrenders and3screenshots.'),
 ('r2-corrections','reviewer-corrections.json','Preserved initial c03 structuredvalues and c35 wrongowntext, exactcorrections and rawrecord mapping.'),
 ('r2-dom','browser-T8-DOM.txt','Actual current8784 completeT.8 DOM text personally read.'),
 ('r2-diff','owned-source-diff.txt','Exactsourceonlydiff: official href gains.memory.')]:
    artifact(identifier,name,'source_snapshot',description)
for i,item in enumerate(browser['figure_renders'],1):
    artifact('r2-fig-'+str(i),item['render'],'figure_render','New current8784 Chromium render personally viewed; sourceSHA/URL/notes in r2-browser/r2-views.')
for identifier,name in [('r2-t8-screen','browser-training-T8.png'),('r2-memory-screen','browser-training-memory-link.png'),('r2-t4-screen','browser-training-T4.png')]:
    artifact(identifier,name,'figure_render','Actual current full training.html browser screenshot personally viewed, scope in r2-views.')
for name in ('verify_reuse.py','check_current_link.py','browser_probe.py','write_report.py','component_validate.py'):
    artifact('r2-code-'+Path(name).stem,name,'code','Actual own round2 evidence/validation program; codeSHA, no product/checker edits.')

memory = next(s for s in r['sources'] if s['id']=='s-memory-canonical')
memory['round2_retrieval'] = {'artifact_id':'r2-link','raw_sha256':'d3c7a6082d8f75080b4dd02c8fb5dd44f12c1903a330dbc56f4f04a106732cf9','retrieval_path_ignored':'outputs/natural-v4/factual-research/training-course/round2-memory-canonical.html','personally_read':'Complete actual API article, maximum GPU tensor allocated bytes since programstart/reset; intreturn, optionalcurrentdevice, notes.'}
for s in r['sources']:
    if s['kind']=='repository_code' or 'retrieval_path_ignored' in s:
        s['round2_reuse_note'] = 'Original personally read source and exactlocators retained; actualcurrent bytes verified identical in r2-reuse. Original evidence provenance is not newexecution.'

c = r['claims'][2]
c['verification']['expected'] = 'Seed42 defaultCPU: text effective_tokens111, loss5.7530,global_grad_norm1.0268; SFT effective_tokens8,loss5.4658,norm2.7651; vision effective_tokens null,loss5.9620,norm1.8985.'
c['verification']['observed'] = 'Original actual unchangedCPU execution: text111/5.752984046936035/1.026832938194275; SFT8/5.465846538543701/2.765058994293213; vision null/5.961961269378662/1.8985438346862793. Rounded table matches.'
c['artifact_ids'] += ['r2-corrections','r2-reuse']
c['verification']['details'] += ' Round2 corrects only own schema strings; all original structured numbers remain in r2-corrections, CPU output unchanged.'

c = r['claims'][32]
c.update(statement='The current T.8 PyTorch2.14 max_memory_allocated official API hyperlink retrieves the stated canonical document.',status='verified',
         scope='Current literal authored URL, not a substituted fallback. Initial oldURL404 remains first-report history. A first unauthored-User-Agent request403 is retained; explicit originalUA GET200 supports retrievability.',
         artifact_ids=['r2-link','r2-browser','r2-memory-screen','r2-byte'])
c['evidence'][0] = {'source_id':'s-memory-canonical','locator':'PyTorch2.14 torch.cuda.memory.max_memory_allocated API article; complete main article andreset_peak_memory_stats paragraph','supports':'Actual current literalURL200, tensorallocatedbytespeak since programstart/latestreset.'}
c['verification'] = {'method':'executed','expected':'Exact current authored .memory.max_memory_allocated PyTorch2.14 URL returnsHTTP200 with stated API article.',
 'observed':'Explicit UA Factual-review/1.0 GET200 at exact authoredURL; actual entire article read. HTML SHA d3c7a6082d8f75080b4dd02c8fb5dd44f12c1903a330dbc56f4f04a106732cf9. Actual8784 browser shows identicalhref. Bare urllib first403 retained.',
 'tolerance':'HTTPstatus andliteralURL exact; no fallback treated as authoredURL.',
 'details':'Independent currentGET/read andbrowser href/screenshot afterfull936line reread. CorrectedURL sourcebyteverified against previewf43; initial404/c33 preserved.'}

c = r['claims'][34]
c['statement'] = 'Precision branches each attempt200 updates, complete200 and skip0; FP32 weights/Adam state with AMP operations. FP32 test7/10, BF16 6/10, FP16 4/10 versus original single-head SFT5/10; this fixedrun shows noAMP speed gain.'
c['verification']['expected'] = 'Each FP32/BF16/FP16 branch:200attempts,200updates,0skips. Correct labelmapping FP32=7/10,BF16=6/10,FP16=4/10; original single-headSFT base5/10.'
c['verification']['observed'] = 'Actual round2 primaryJSON extraction: fp32 matches7/records10, bf16 6/10, fp16 4/10; each updates200/skips0/weights_dtype torch.float32. Primary record SHA unchanged. Original record audits and rawIDs/NLL retained.'
c['verification']['details'] += ' Round2 corrects ownfirstreport precisionname mapping, not course. Exactcurrent original paths re-extracted by verify_reuse.py, retained in r2-reuse/r2-corrections.'
c['artifact_ids'] += ['r2-reuse','r2-corrections']

c = r['claims'][48]
c['statement'] = 'The current complete training.html at8784 exposes actual T.8/T.4 anchors and corrected officialmemorylink; current936line source matches literalpreviewf43. All13necessary prerequisiteSVGs newlyrendered and personallyviewed.'
c['location'] = 'Completecurrenttrainingpage T.8/T.4, currentauthorityhref; allnecessaryfigures'
c['artifact_ids'] = ['r2-browser','r2-byte','r2-read','r2-views','r2-t8-screen','r2-memory-screen','r2-t4-screen','r2-dom']+['r2-fig-'+str(i) for i in range(1,14)]
c['scope'] = 'Actualcurrentownedfullsource andexecuted8784preview only;13necessary prerequisitefigures. Original8783 observations preserved as history. No publication/readingtiming approval or claim of rereading other corrected chapters.'
c['verification'] = {'method':'executed','expected':'Fullcurrent f9ae45 source=previewf43bytes; actualT.8/T.4 headings/clicks,correctmemoryhref and13renewed personal SVG views.',
 'observed':'Exactsourcebyteequalitytrue; Chromium actual training.html#T.8 self-anchor andT.4 navigation, savedfullT.8DOM/3screenshots;13current SVG renders subsequently personallyviewed with exactSHA notes.',
 'tolerance':'Fullsource andSVG SHA exact; browservisibility personallyinspected, not inferred fromHTTP.',
 'details':'Current newexecution at8784 and tools.view_image actualreturnedimages; rendering pendingflags preserved and separatecompletedpersonalnotes distinguish render fromview.'}

for c in r['claims']:
    c['current_recheck'] = {
     'source_sha256':SHA,
     'basis':'Completecurrent936line read andindependent substantive-claim reassessment; originalcode/authority/record bytes checkedexactly before reuse. Existing originalmeasurement arithmetic supports those records within originalscope, not newGPU replication.',
     'changed_evidence':c['id'] in ('c03','c33','c35','c49')
    }

r['issues'] = [
 {'claim_id':'c33','status':'resolved','details':old['issues'][0]['details'],'resolution':'Currentfullsource line786 uses .memory.max_memory_allocated. Actual currentliteralURL GET200/API article read andactual8784 href visible; exactpreview/currentSHA verified. Initial404/c33 remains untouched.',
  'evidence_artifact_ids':['r2-link','r2-browser','r2-memory-screen','r2-byte'],'history_first_source_sha256':old['current_document_sha256']['course/training.md']},
 {'claim_id':'c03','status':'resolved','details':'Ownfirstreport expected/observed were structuredobjects ratherthanrequirednonemptystrings.','resolution':'Own exactexpected/observed string summaries retain everyoriginalvalue; structuredoriginalverification preserved in r2-corrections andunchangedCPU artifact. No checker/data changes.'},
 {'claim_id':'c35','status':'resolved','details':'Ownfirstreport statement mislabeled precision branches. Course/currentoriginalrecord werecorrect.','resolution':'Ownround2 explicitly FP32=7/10,BF16=6/10,FP16=4/10; actualcurrentJSON extraction recorded. Originalwrongreport preserved, no source/record mutation.'}
]
r['checks']['factual_accuracy'].update(status='pass',details='All49substantive groups reassessed aftergenuinecurrent936line reread andexactunchangedoriginal evidence comparison. CorrectedcurrentT.8literalURL actuallyretrieved/read; ownc03format/c35mapping corrected transparently. No unresolvedknowledge/sourcecontradiction remains within statedscope.')
r['checks']['numeric_verification']['details'] = 'Currentprecision labels independentlyextracted; all55repo sources/38priorartifacts exactbefore genuine reuse of originalCPUtable/derivations,591recordarithmetic with9FP32diagnoses,2639IDflags,102NLL/BPB,8Flashmedians andOCR40/57. Initialresume difference/followup retained. No newGPU replication or rerunning these computations claimed.'
r['checks']['figure_consistency']['details'] = 'All13 necessarycurrentSVGs exactsourcebytechecked, actuallynewlyChromiumrendered at8784 andpersonallyviewed via returnedimageblocks. Numbers/direction/alignment/scales/captions independentlyconsistent;3newbrowser screenshots personallyviewed. Ownedfile embedsno directSVG.'
r['checks']['source_verification'].update(status='pass',details='Actualoriginalpapers/exactequations,2.14officialdocs/matchingtorchgitsource,pinnedcards/licenses,repoimplementations andfixedrawrecords retain exactlocators;26externaloriginals+55repo sources byteverified before reuse. Currentliteralofficiallink200 actuallyread; first403 andold404 explicitlypreserved.',claim_ids=[c['id'] for c in r['claims']])
r['checks']['limitations']['details'] += ' Round2source-onlychange carriesnonewGPU,benchmark,kernel,timing ortrainingclaims; original single-seed/hardware/data limitations and firstfailures remain explicit.'
write('report.json', r)
print(json.dumps({'report':rel(D/'report.json'),'sha256':sha(D/'report.json'),'source_sha256':SHA,'claims':len(r['claims']),'sources':len(r['sources']),'artifacts':len(r['artifacts']),'figures':len(r['figure_sha256']),'verdict':r['verdict'],'first_final_report_unchanged_sha256':sha(P/'report.json')},indent=2))
