"""Create a distinct informed owner callback; never overwrite the original."""
import copy
import hashlib
import json
import platform
import sys
from datetime import UTC,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
from scripts.check_technical_reviews import _validate
ART=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
proof=json.loads((ART/'current-inspection.json').read_text())
original_path=ROOT/proof['original_report']['path']
original=json.loads(original_path.read_text())
assert sha(original_path)==proof['original_report']['sha256']=='15761713e59ec3af6278a6e3ff5f0fb20d770541058b8b619a96d81010e6d0e4'
report=copy.deepcopy(original)
page=proof['current_page']
report.update(current_context='informed',source_sha256=page['source_sha256'],figure_sha256=page['figures_sha256'])
report['callback']={'kind':'informed_original_owner_order_recheck','reviewed_at':datetime.now(UTC).isoformat(),'freeze':proof['freeze'],'original':proof['original_report'],'current_source_snapshot':page['snapshot'],'current_source_sha256':page['source_sha256'],'scope':'Entire current canonical curriculum reread, plus its sole changed route paragraph, actual chapter14/19 heading order and relevant opening/transition dependencies, guide R2/R3, index/navigation/routes and same-source notebook/publisher contracts. Original40claims assessed:39page statements unchanged; c15replaced with thecurrent route contract. c01currentcounts and c32synchronization additionally rechecked. All inherited primary source/code/data/score artifact pins checked.','unread_material':'No new reader/continuity reviews, other technical reviews, author plan or known-issue list read. Original personal review and its evidence were read as authorized owner carryforward.','not_repeated':['original external source retrieval/reading','figure render/view alreadyperformed personally inoriginal review','rawscore recomputation','CPUmodeldemo','modeltraining','formaltest','GPUbenchmark','HFdownloads/publication','Colabexecution'],'issues_or_optional_suggestions':[]}
report['scope_note']='Current freeze18 whole canonical curriculum verdict by its original independent technical owner. reviewer_context=fresh records the original independent context only; current_context=informed is this callback. The original report is preserved byte-for-byte. The full current page has one paragraph changed; source/figure/data/implementation identities and prior evidence were checked before inheriting claims. New order does not strengthen model capability claims.'
report['verification_boundary']='This callback personally reread the entire current curriculum and relevant route dependencies, executed read-only order/count/hash checks and the two actual structural synchronization commands. The prior owner\'s measured/GPU/publication/rawscore/figure evidence is inherited with unchanged pins and explicitly kept historical; no claims that those executions or original external reads were repeated now. No formaltest, training, HF/Modal operation or broad benchmark was run.'
for source in report['sources']:
 if source['id']=='sBody':
  source.update(path=page['snapshot'],sha256=page['source_sha256'],title=page['snapshot'],version='freeze18canonical snapshot (stored underfreeze17); independently reread2026-10-16',inspection_note='Whole current curriculum page read; single paragraph diff compared with original. Updated14/19order verified against current original guide/config/index and authored headings; unchanged technical/empirical passages inherit this owner\'s original checked evidence. No author plan read.')
 else:source['callback_provenance']='Inherited from this same reviewer\'s original personally checked evidence; original recorded version/access date retained. Repository source SHA was checked current; no external refetch or new experiment asserted.'
for claim in report['claims']:
 claim['owner_callback']={'original_claim_id':claim['id'],'original_report_path':proof['original_report']['path'],'original_report_sha256':proof['original_report']['sha256'],'disposition':'inherited_unchanged_statement_and_evidence'}
 if 'verification' in claim:claim['verification']['details']+=' Owner callback: the original execution evidence is carried from the same reviewer and was not rerun unless separately specified by aOrder evidence.'
claim15=next(c for c in report['claims'] if c['id']=='c15')
claim15.update(statement='目前C.1–C.5接第7章、C.6–C.7接第13章；第14章依14.1–14.18、第19章依19.1–19.20順讀，與正文、索引、導航及整合章路線一致。',location='§2閱讀與實作路線；§3目前閱讀順序段',evidence=[{'source_id':'sOrderExec','locator':'current-inspection.orders;C_navigation_order_verified;capstone_route;needed_dependencies','supports':'Actual authored/index/nav14/19ascending order, C insertion points,19capstone order and necessary transition roles independently checked.'},{'source_id':'sOrderGuide','locator':'R.2topic routes and R.3firstparagraph','supports':'R.3 sequential route agrees; R.2still permits deliberate topic jumpreading without reversing the canonical sequential route.'},{'source_id':'sOrderNav','locator':'project.nav chapter7/C/8,13/C/14 andchapter19 groups','supports':'Actual website navigation order follows the stated sequence.'},{'source_id':'sOrderRoutes','locator':'routes[id=capstone].page_ids;thinking-looped-decisions scope','supports':'Capstone ischapter19then19.1through19.20; thematic subset route is distinct from full sequential reading.'}],artifact_ids=['aOrderInspection','aOrderCode','aOrderCommand'],scope='Current technical navigation/ordering contract. Reads14.6/14.7/14.10/14.11 openings and19.12end→19.13purpose→19.14common-split handoff; does not claim new model facts, revalidate whole dependency lessons, or measure reader comprehension. The deleted25-new-ID/historical-order wording is retired from the current claim, not relabeled as fresh proof.',verification={'method':'executed','expected':'14.1–14.18and19.1–19.20 inascending order inauthoredsource/index/nav;Cblocks after7/13;capstonechapterintrothen19.1–20','observed':'Exact expected authored/index/nav orders;Cpositions true;capstone route exact. Current order check passed23chapters/312lessons/failures[].','details':'Personally read current source and relevant dependency passages, parsed actual configs and executed inspect-current.py plus check_lesson_order.py; no modelrun. 19.12now explicitly sends reader to19.13purposes before19.14shared data.'})
claim15['owner_callback']['disposition']='replaced_current_order_claim_after_actual_reread;originalclaimandreportpreserved'
for id in ['c01','c32']:
 c=next(c for c in report['claims'] if c['id']==id);c['artifact_ids'].extend(['aOrderInspection','aOrderBuild']);c['owner_callback']['disposition']='unchanged_statement_evidence_carried_plus_fresh_narrowcheck'
 c['verification']['details']+=' This callback additionally rechecked current counts and source/notebook/index synchronization; see aOrderInspection/aOrderBuild.'
 if id=='c32':c['evidence'].extend([{'source_id':'sOrderExport','locator':'export_course.py lesson loop andColab/download generation near292–307;chapterindex generation near308–322','supports':'Actual current same-section notebook paths andpublishedlesson/chapter order still derive fromindex; sourceSHAchecked unchanged fromoriginal.'},{'source_id':'sOrderBuild','locator':'build_course.py175–236','supports':'Current check mode generates expectednotebook/index/lessons representations withoutwriting andcompares exactbytes.'}])

def repo(id,path,note):
 report['sources'].append({'id':id,'kind':'repository_code','title':path,'path':path,'sha256':sha(ROOT/path),'version':'Current freeze18callback bytes inspected2026-10-16','verified':True,'inspection_note':note})
repo('sOrderGuide',proof['guide_R3_snapshot'],'Only necessary R.2topic routing andR.3sequential order directly read; exact snapshot/livebyteidentity checked. No whole-guide technical verdict implied.')
repo('sOrderNav','zensical.toml','Parsed actual complete nav IDs; personally read7/C/8,13/C/14,chapter14andchapter19 blocks. Verifies order only, not rendering or all website settings.')
repo('sOrderRoutes','course/reading-time-routes.json','Read complete current route definitions;capstoneintrothen19.1–20checked. The thematic thinking subset is not presented as thefullchapter14route; no timing estimates evaluated.')
repo('sOrderBuild','scripts/build_course.py','Read current175–236source-order generation andread-onlycheck behavior; ran--checkonce inthis callback,312synchronized. No kernelsrerun.')
repo('sOrderExport','scripts/export_course.py','Read currentlesson/indexloop, same-notebookColab/download paths andchapter14/19index generation; fullfileSHAalso matches ownoriginal pinnedinput. No websitebuild/render performed.')
repo('sOrderCheckCode','scripts/check_lesson_order.py','Read completechecker andpersonally executed it: validates source/index/navIDs/orderandcapstone route. Structuralevidence,notconceptualaccuracyorreadability proof.')
report['sources'].append({'id':'sOrderExec','kind':'execution','title':'Current owner callback read-only evidence inspection','verified':True,'artifact_id':'aOrderInspection'})

def artifact(id,name,kind,description,command=None,result=None):
 p=ART/name;a={'id':id,'kind':kind,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'description':description}
 if command:a.update(command=command,result=result,environment={'python':platform.python_version(),'device':'cpu','operation':'read-only structural/source inspection'})
 report['artifacts'].append(a)
# Persist exact prior independent tool-call results, not a new execution claim.
dump(ART/'order-command-result.json',{'command':'.venv/bin/python scripts/check_lesson_order.py','exit_code':0,'result':{'status':'passed','chapters':23,'lessons':312,'failures':[]},'environment':{'python':platform.python_version(),'device':'cpu'},'source_sha256':{'scripts/check_lesson_order.py':sha(ROOT/'scripts/check_lesson_order.py'),'zensical.toml':sha(ROOT/'zensical.toml'),'course/lesson-index.json':sha(ROOT/'course/lesson-index.json'),'course/reading-time-routes.json':sha(ROOT/'course/reading-time-routes.json')},'scope':'Actual independent tool invocation inthiscallback,not a modelbenchmark.'})
dump(ART/'build-command-result.json',{'command':'.venv/bin/python scripts/build_course.py --check','exit_code':0,'stdout':'312 個小節：同步檢查通過\n','environment':{'python':platform.python_version(),'device':'cpu'},'source_sha256':{'scripts/build_course.py':sha(ROOT/'scripts/build_course.py'),'scripts/export_course.py':sha(ROOT/'scripts/export_course.py')},'scope':'Actual independent read-only invocation inthiscallback. No notebookkernelorColabexecution.'})
artifact('aOrderInspection','current-inspection.json','execution','Current whole-page diff, order/count/dependencyinspection and unchanged originalsource/artifact/inputpins.','.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/curriculum/inspect-current.py','Passed:singlechangedparagraph;312notebooks/338sections;14and19ascending;Cplacementsandcapstoneroutecorrect;alloriginalsource/artifact/measured-inputpinsunchanged;only19.12lesson/notebookchangedfromfreeze16.')
artifact('aOrderCode','inspect-current.py','code','Exact narrowinspectioncode; no training/testoperation.')
artifact('aOrderCommand','order-command-result.json','execution','Actual source/index/nav/capstoneroutecommandresult.','.venv/bin/python scripts/check_lesson_order.py','Passed;23chapters,312lessons,failures[].')
artifact('aOrderBuild','build-command-result.json','execution','Actual current source/notebook/index synchronization check.','.venv/bin/python scripts/build_course.py --check','312 個小節：同步檢查通過;exit0.')
report['checks']={
 'factual_accuracy':{'status':'pass','details':'Entire current curriculum reread.39unchangedtechnicalstatementscarrysameowneroriginalprimaryevidence;currentc15routeclaimreplacedandindependentlycheckedagainstactualsource/index/nav/guide. Noneoftheorderingwordsstrengthensmeasuredcapability.','claim_ids':[c['id'] for c in report['claims']]},
 'numeric_verification':{'status':'pass','details':'Thiscallbackpersonallyrecounted23chapterfiles,312lessons/notebooksand338numberedsections. Allothernumeric/empiricalstatementsunchanged;originalreceipts/rawscorearithmeticandtheirdata/implementationpinsmatchcurrent. Theywereinherited,notrerun.','claim_ids':['c01','c02','c03','c04','c06','c08','c09','c12','c13','c35','c36','c37','c38']},
 'figure_consistency':{'status':'pass','details':'SamecurriculumSVGhashandfiguremapasinoriginalpersonallyrendered/viewedreview. Currentchangedparagraphdoesnottouchdiagramoritsmask/updateinterpretation. Originalfigureinspectioninherited;noadditionalrenderorwebsite/mobileinspectionclaimed.','claim_ids':['c31']},
 'source_verification':{'status':'pass','details':'Alloriginalrepositorysources,artifactsandmeasuredimplementation/data/scoreinputpinscheckedunchanged. UpdatedsBodyonlyafterwholecurrentpageread;newroute/guide/config/publisherentriespinactuallyinspectedcurrentbytes. Originalfixedexternalreadsremainhistorical,notrefetched. Nootherreviewerconclusionsused.','claim_ids':[c['id'] for c in report['claims']]},
 'limitations':{'status':'pass','details':'Informedcallbackkeepsoriginalcapability/resource/denominator/provenancelimits. Numericorderdoesnotimplynewabilityorreaderunderstanding;oldexecutionreceiptsdonotpretendtobecurrentreruns.14.6stillprecedesloopedsharing;19.12hands19.13purposebefore19.14data. Noformaltest/training/GPU/HF/Colab/timingverificationperformed.','claim_ids':['c05','c06','c07','c10','c11','c12','c14','c15','c18','c23','c27','c28','c29','c30','c32','c33','c34','c35','c36','c37','c38','c39','c40']}}
report['issues']=[]
path=ROOT/'docs/course-revision-20261016/reviews/technical-order-curriculum-callback-01.json'
assert not path.exists(),'Refusing to overwrite an existingcallback.'
dump(path,report)
# Required: reread theactualsavedreportandvalidateitscurrentassignedwholepage.
actual=json.loads(path.read_text())
body=(ROOT/page['snapshot']).read_text()
errors=_validate(ROOT,ROOT/'docs/curriculum.md','curriculum',body,actual,{'/root/backfill_tech_curriculum':['curriculum']},set())
assert not errors,errors
assert sha(original_path)==proof['original_report']['sha256']
assert sha(ROOT/'docs/curriculum.md')==page['source_sha256']
seal={'reviewer_task':'/root/backfill_tech_curriculum','current_context':'informed','completed_at':datetime.now(UTC).isoformat(),'command':'.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/curriculum/seal-callback.py','environment':{'python':platform.python_version(),'device':'cpu'},'format_validation_errors':errors,'report':{'path':str(path.relative_to(ROOT)),'sha256':sha(path)},'original_report_unchanged':True,'claims':len(actual['claims']),'sources':len(actual['sources']),'artifacts':len(actual['artifacts']),'current_source_sha256':page['source_sha256'],'scope':'Technical ownercallbackcompletedandselfvalidatedafteractualsourcechecks;notafreshreaderrevieworformalmodeltest.'}
dump(ART/'validation-seal.json',seal)
print(json.dumps(seal,ensure_ascii=False,indent=2))
