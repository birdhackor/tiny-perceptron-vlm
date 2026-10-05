from pathlib import Path
import copy
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[6]
D=Path(__file__).parent
P=D.parent
TASK='/root/v4_review_coordinator/factual_whole_training_course'
SHA='255191c925373d8d1de312cd64eb9c74fa8486479815420205826cdf440e29ea'
REV='160fd47dede5c2453c8a08dd535290ddfd0b915d'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rel=lambda p:p.relative_to(ROOT).as_posix()
def write(name,data):
    (D/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

old=json.loads((P/'round2/report.json').read_text())
reuse=json.loads((D/'reuse-verification.json').read_text())
browser=json.loads((D/'browser-receipt.json').read_text())
assert sha(P/'round2/report.json')=='85860239fcf5f89de24164edac26fbb070ec8b6a4d44be0a2a3d4b10e0686f66'
assert sha(ROOT/'course/training.md')==SHA==sha(D/'training.md.read-snapshot')
assert all(z['byte_identical'] for k in ('sources','artifacts','external_originals','figures','prerequisite_sections') for z in reuse[k])
assert reuse['preview_current_byte_equal']
raw=(ROOT/'course/training.md').read_bytes().splitlines(keepends=True)
(D/'current-T8.md.read-snapshot').write_bytes(b''.join(raw[700:811]))
chapter=ROOT/'course/chapters/16.md'
chapterbytes=chapter.read_bytes()
section=b''.join(chapterbytes.splitlines(keepends=True)[4:47])
(D/'prerequisite-16.1.md.read-snapshot').write_bytes(section)
literalchapter=subprocess.run(['git','show',REV+':course/chapters/16.md'],capture_output=True,check=True).stdout
assert chapterbytes==literalchapter

views=[]
for s in browser['screenshots']:
    v=copy.deepcopy(s)
    v['personal_view']='completed'
    v['inspection_note']=(
      'Personally viewed new screenshot: actual16.1 page has a visible T.8效率量測方法 link after MiB/timing discussion. CPU profile and weightbytes remain different units; link is an optional route to original measurement conditions.'
      if s['path'].startswith('browser-16.1') else
      'Personally viewed new screenshot: actualfulltraining page opens at visibleH3 選讀：效率實驗的量測條件. Hardware/FP32/TF32,SFT source/familysplit,sixbranchreset andmissingpadded/packedfields,allocatedvsreserved/driver/separateprocess,MiB andinference/update timing limits visible. Below are distinctFlashprobe andprecision protocols, not a combinedmodel-speedclaim.'
    )
    assert sha(D/s['path'])==s['sha256']
    views.append(v)
write('personal-view-notes.json',{'reviewer_task':TASK,'round':3,'personal_views_completed':True,'view_mechanism':'Actual tools.view_image of both newly saved8789 Chromium screenshots; personally inspected returnedimages, then personally read complete2237-character methodDOM.','browser_views':views,'scope':'Only newmethodtarget/necessary16.1. Prior13SVG sourceandrender bytes verified identical andprior personalviews reused honestly; no new13SVG render/view claimed.'})
write('delta-read-receipt.json',{
 'reviewer_task':TASK,'round':3,'current_complete_document_sha256':{'course/training.md':SHA},
 'current_lines':938,'current_snapshot_bytes':123957,
 'new_personal_source_reads':[{'source':'course/training.md','lines':'701–811','count':111,'scope':'EntirecurrentT.8 including newlypromotedH3 andallmeasurement/Flash/precision paragraphs; tooloutputnottruncated.','snapshot':rel(D/'current-T8.md.read-snapshot'),'sha256':sha(D/'current-T8.md.read-snapshot')},{'source':'course/chapters/16.md','lines':'5–47','count':43,'scope':'Entire necessary16.1 subsection, includingdirectnewH3 methodlink; tooloutputnottruncated.','snapshot':rel(D/'prerequisite-16.1.md.read-snapshot'),'section_sha256':sha(D/'prerequisite-16.1.md.read-snapshot'),'fullfile_sha256':sha(chapter),'literalpreview_fullfile_byte_equal':True}],
 'actual_delta_read':'Personally read complete savedunifieddiff: existingboldmethodlabel became H3 plusblankline. Quantitative/proceduralbody unchanged. Current938lines are not claimed as a newfullreread.',
 'prior_complete_read_reused':'Original936-line fullread andround2 genuine936-line fullreread preserved withsourceSHA/snapshots/actualreadorders. CurrentoutsideT.8 bytes unchanged; exactreuse before currentjudgment, not inheritedPASS.',
 'byte_proof':rel(D/'reuse-verification.json'),
 'browser':'Actual8789 training.html T.8→actual16.1.html→newH3 methodanchor→T.8入口 clicks;2237-character currentmethodDOM personallyread and2screenshots personallyviewed.',
 'methodscope_judgment':'Directanchor names thealreadyverifiedmethodbody. Sixmainbranch CUDAallocatedpeaks remainseparatefrompadded/packedmissingfields,unusedreserved/driver/independentprocessmemory,standaloneFlashQKVfixture,precisionbranches andinference/updates. Itdoesnotturnthecostobservations into GPUgeneralization or aCPUvalidation ofCUDA.',
 'related_original_claim_ids':['c28','c29','c30','c31','c32','c33','c34','c35','c49'],
 'extra_prerequisite_boundary':'16.1 is necessary toactualmethodlink context. Itscompletechapter wasnotpersonallyreread; other16 sectionsretain originalread/evidence with exactsectionbytecomparisons. No new independent review of all16.1 CPUprofiler claims asserted.'
})

r=copy.deepcopy(old)
r['round']=3
r['source_sha256']=SHA
r['current_document_sha256']={'course/training.md':SHA}
r['assigned_document_scope']='Complete course/training.md under same original factualowner: current938line source, originalcomplete936line reads/evidence preserved; authorized narrowround3 reread ofallcurrentT.8/newH3 andnecessary16.1 plus real8789methodclick. Otherfullfilebytes,original49claim evidence and13SVGs exactlyunchanged before reuse. No newcomplete938line reread or numberedlessonID claimed.'
r['source_read_receipts']={'current_delta_read':rel(D/'delta-read-receipt.json'),'current_full_rawsnapshot':rel(D/'training.md.read-snapshot'),'before_current_source_preservation':rel(D/'source-preservation-receipt.json'),'unchanged_evidence_byte_proof':rel(D/'reuse-verification.json'),'previous_complete_read_history':copy.deepcopy(old['source_read_receipts']),'new_full938line_read':False,'complete_scope_basis':'Preservedgenuinecomplete936line reads plusactualcurrentT.8/16.1 delta recheck, completebefore/currentbyteproof andnecessarybrowser scope.'}
r['report_preservation']={'round2_final_report':rel(P/'round2/report.json'),'round2_final_report_sha256':sha(P/'round2/report.json'),'round2_fullsource_sha256':sha(P/'round2/training.md.read-snapshot'),'earlier_history':copy.deepcopy(old['report_preservation']),'same_owner_revision':'Originalclaims/authority/CPUfailures/source/errors/verdicts untouched. NarrownewH3 navigation/methodscope recheck; currentwholeSHA is supportedby delta andexactfullbytecomparison.'}
r['independence_disclosure']['round3']='Sameoriginalfactualowner. Readerclosure isphasegateonly; nopeerreader/factualjudgments orauthoroutcomes read. Ownpreviousreports necessarily usedfor versioned recheck. No subagents/source/checker/Git/environment/time edits.'
r['execution_limits']['round3_actual_cpu_environment']=reuse['environment']
r['execution_limits']['round3_performed']='ActualnarrowcurrentT.8/16.1 sourcedelta read; exact55repo/69artifact/26externaloriginal/29prereq/13SVG comparison; currentCPUversioninspection; actual8789 methodlink browserclick/2237charDOM/twoimages. ExistingCPUresults andoriginalGPUrecords reusedafterexactbytes, no newtraining/profiler/benchmark/download/externalretrieval.'
r['execution_limits']['round3_reuse_bounds']='Prior49claim knowledge/numerics/limits unchanged: promotionofexistinglabeltoH3 changesnavigation only. OriginalCPU andGPUrecord provenances/failures kept. No newGPUkernel/timing/modelaccuracy or938linefullreread claimed;13priorSVGviews reused aftersource/render-byteverification.'
r['component_validation_receipt']=rel(D/'component-validator-output.json')
r['component_validation_scope']='Actual unchanged official _artifacts/_sources/_claims plusfullfile source/figure/issues/fivechecks semantics withofficialhelpers. No numberedchecker identity fabricated; schemacheckdoesnotprovetruth/reading.'
env=reuse['environment']
def artifact(id,name,kind,description,command=None,result=None):
    a={'id':id,'kind':kind,'path':rel(D/name),'sha256':sha(D/name),'description':description}
    if kind=='execution':a.update(command=command,result=result,environment=env)
    r['artifacts'].append(a)
for id,name,desc in [
 ('r3-source','training.md.read-snapshot','Actualcompletecurrentrawbytes preserved; narrowread scope separatelyrecorded.'),
 ('r3-T8','current-T8.md.read-snapshot','ActualentirecurrentT.8 111lines personallyread.'),
 ('r3-16.1','prerequisite-16.1.md.read-snapshot','Actualnecessary16.1 43lines personallyread; exactmethodlink context.'),
 ('r3-preservation','source-preservation-receipt.json','Before/currentSHA/fullbytes/linecounts and untouchedround2report proof.'),
 ('r3-diff','owned-source-diff.txt','Actualsourceonly unifieddiff personallyread: boldlabel→H3.'),
 ('r3-reuse','reuse-verification.json','ActualinlineCPU bytecomparison outputs before reuse;55sources69artifacts26externaloriginals29sections13SVGs unchanged,freshruntime,previewbyteequality. Notnewtraining.'),
 ('r3-read','delta-read-receipt.json','Ownactualnarrowread/methodscope judgment and honestfullreadreuse boundaries.'),
 ('r3-views','personal-view-notes.json','Completedactualtwoimage/DOM personalinspection, old13SVGreuse explicitlybounded.'),
 ('r3-DOM','browser-method-DOM.txt','Actual2237char methodDOM personallyread.'),
 ('r3-first-errors','browser-initial-errors.json','Actualfirst guessedpath404/locatorTimeout retained; actualsitehref retry separatelyverified.')]:
    artifact(id,name,'source_snapshot',desc)
artifact('r3-browser','browser-receipt.json','execution','Actual8789training→16.1→newH3→T.8 clicks andcurrentmethodDOM/screens.',
 '.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-course/round3/browser_probe.py',
 'Actual16.1href is/16.1.html; methodclick reachesfulltraining.html#選讀效率實驗的量測條件,H3visible;2237charDOM/twoscreens,canonicalAPIhref unchanged;realT.8returnclick.')
for i,s in enumerate(browser['screenshots'],1):artifact('r3-screen-'+str(i),s['path'],'figure_render','Newcurrent8789 browser screenshot personallyviewed; actualnotesin r3-views.')
for name in ('browser_probe.py','browser_probe.initial.py','write_report.py','component_validate.py'):artifact('r3-code-'+Path(name).stem,name,'code','Actualownround3program orpreservedinitialfailedprogram; no product/checkeredit.')
r['sources'].append({'id':'r3-necessary16source','kind':'repository_code','title':'Necessary current16.1 authored methodlink context','path':'course/chapters/16.md','sha256':sha(chapter),'version':REV,'verified':True,'inspection_note':'Personallyreadexact16.1 lines5–47 only inround3; fullfileSHA pins sourcewithoutclaimingcompletechapterread. Currentliteralpreview chapterbytes identical; actualdirectmethodtarget browserverified. This document isnavigationcontext, not an external authority forCUDA results.'})
c=r['claims'][48]
c.update(statement='The necessary16.1 efficiency-method link actually opens the current fulltraining page at the new visibleH3, with the existingmeasurement protocols/limits intact; itsreturnlink reachesT.8.',location='T.8 lines780–809newH3 methodscope;16.1 line42methodlink',scope='Narrowactualnavigation/methodscope at8789, no newquantitativeclaim. Originalcomplete936line reads/49claim facts/13personalSVGviews honestlyreusedafterexactbytes; current938line source/preview match independentlyverified.',artifact_ids=['r3-browser','r3-screen-1','r3-screen-2','r3-DOM','r3-read','r3-views','r3-reuse','r3-preservation'])
c['evidence']=[{'source_id':'r3-necessary16source','locator':'16.1 line42 hyperlink ../training.md#選讀效率實驗的量測條件; actual16.1.html browserlink','supports':'Exactactualsourcecontext/target verified byclick, not guessed chapterroute.'},{'source_id':'derive-figures','locator':'OriginalpersonalSVGnotes/transparentcalculations;currentexactsource/renderbyteproof r3-reuse','supports':'Existingnecessaryfigurefacts retained, no newrender/view claimed.'}]
c['verification']={'method':'executed','expected':'Actual16.1 methodhyperlink lands onnewH3 選讀：效率實驗的量測條件 withinfulltraining.html,withunchangedmethodbody;return本節入口→T.8.',
 'observed':'Actual8789 trainingT.8link→16.1.html;methodclick→training.html#%E9%81%B8%E8%AE%80%E6%95%88%E7%8E%87%E5%AF%A6%E9%A9%97%E7%9A%84%E9%87%8F%E6%B8%AC%E6%A2%9D%E4%BB%B6;H3id選讀效率實驗的量測條件 visible.2237charactualDOMread,2newimages personallyviewed;returnclick→training.html#T.8.',
 'tolerance':'Literalcurrenthref/headingtarget/fullsourceSHA exact; personallyinspectedvisibility/methodcontext, noHTTP-onlyinference.',
 'details':'Actualcorrectedscript navigatesfromsitehyperlinks. Firstguessedchapters/16.html404/locatorTimeout transparentlypreserved asreviewerrouteerror. Bodyquantities unchanged per sourceonlydiff; officialmemoryURL remainscanonical withoutnewHTTP200claim.'}
r['claims'][32]['location']='T.8 currentline788officialAPIhyperlink; originalline786 retrievalhistory retained'
for c in r['claims']:
    c['round2_recheck_history']=copy.deepcopy(c['current_recheck'])
    c['current_recheck']={'source_sha256':SHA,'round':3,'basis':'AuthorizednarrownewH3/methodlink reassessment afteractualcurrentT.8/16.1 read andfullbyteproof. Originalsubstantivefacts/authority/code/record/CPU results unchanged andgenuinelyreused. Currentnewbrowserclaimc49 rewritten; priorreports preserved.','new_execution':c['id']=='c49'}
r['issues'].append({'claim_id':'c49','status':'resolved','details':'Firstownbrowserprobe guessed nonexistent/chapters/16.html andtimedout; directrequestconfirmed404. Thiswasnotanactualsitehyperlink.','resolution':'Preservedinitialscript/errorreceipt. ActualtrainingT.8 href/16.1.html works; currentmethodlinkclickreachesnewvisibleH3 andT.8returnlink works. No sourcecorrection required.'})
r['checks']['factual_accuracy']['details']='CurrentnarrowH3/methodlink deltajudgedafteractualT.8/16.1 read andrealbrowser. Existing49substantiveclaims remainwithin originalscope after exactoriginal evidence/fullsource comparison; methodconditions unchanged and correctlytargeted. No unresolvedfactualissue.'
r['checks']['numeric_verification']['details']='No newnumericalstatement inbold→H3delta; currentT.8numbers andbranchlimits unchanged. OriginalCPUderivations/recordarithmetic/denominators and failures genuinelyreusedafterexact55sources/69artifacts comparison, not rerun or inferrednewGPU replication.'
r['checks']['figure_consistency']['details']='13necessarySVGsource hashes andpreviousactualrenders/viewnotes byte-identical; originalpersonalviews/derivations genuinelyreused. No newSVG or changedfigure inthisdelta. Twoactualnewmethodnavigation screenshots personallyviewed.'
r['checks']['source_verification']['details']='Currentfullsource255191 matchesliteralpreview160; necessary16.1 fullfileSHA/sourcecontext andactualtarget pinnednewly. Alloriginal55registeredrepo sources,26externaloriginals,29prerequisite sections unchanged before reuse. No newexternaldownload or substitutedauthority; currentcanonicalmemoryhref unchanged.'
r['checks']['limitations']['details']='Methodanchor preserves sixmainbranch tensorallocatedpeak/reset scope,padded/packed missingfields,unusedreserved/driver/independentprocess exclusions,MiB,inference warm3/measure9 vsupdate median,standaloneFlashfixture andindependentprecisionSFTbase. These continue original c28–c35 boundaries; nonewGPU/profiler/training/benchmark orfull938line reread asserted.'
r['verdict']='pass'
write('report.json',r)
print(json.dumps({'report':rel(D/'report.json'),'sha256':sha(D/'report.json'),'current_source_sha256':SHA,'verdict':r['verdict'],'claims':len(r['claims']),'sources':len(r['sources']),'artifacts':len(r['artifacts']),'figures':len(r['figure_sha256'])},indent=2))
