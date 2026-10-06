"""Owner fixes introduction field placement after verifying the original current bytes."""
from pathlib import Path
import hashlib,importlib.util,json,re,shutil,sys

root=Path('/workspace/selftrained-v2')
out=root/'docs/technical-reviews/artifacts/p6-19.1'
path=root/'docs/technical-reviews/19.1.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads(path.read_text())
prior_sha=sha(path)
backup=out/'before-intro-top-level-metadata-report.json'
assert not backup.exists()
shutil.copyfile(path,backup)
assert sha(backup)==prior_sha
validation=out/'evidence/report-validation.json'
shutil.copyfile(validation,out/'evidence/report-validation-before-intro-metadata.json')

intro=report['chapter_introduction']
raw=(root/intro['path']).read_bytes()
intro_bytes=raw[:re.search(rb'(?m)^## ',raw).start()]
current_sha=hashlib.sha256(intro_bytes).hexdigest()
assert intro_bytes==(root/intro['snapshot']).read_bytes()
assert current_sha==intro['sha256']
owner_recheck_path=out/'evidence/recheck/recheck-execution.json'
owner_recheck=json.loads(owner_recheck_path.read_text())
assert owner_recheck['current_intro_sha256']==current_sha
assert '都由本專案從零訓練，沒有沿用既有預訓練神經權重' in intro_bytes.decode()
assert intro['reviewer_summary'].strip()
for source in report['sources']:
    if source['kind']=='repository_code':assert sha(root/source['path'])==source['sha256']
for artifact in report['artifacts']:assert sha(root/artifact['path'])==artifact['sha256']
for p,h in report['figure_sha256'].items():assert sha(root/p)==h

fields={'intro_source':intro['path'],'intro_sha256':current_sha,'intro_summary':intro['reviewer_summary']}
report.update(fields)
record={'reviewer_task':'/root/p6_fact_19_1','command_argv':sys.argv,'before_report_sha256':prior_sha,'before_report_snapshot':backup.relative_to(root).as_posix(),'change':'Standard top-level introduction metadata added by original reviewer; no knowledge claim or verdict changed.','fields_added':fields,'raw_byte_scope':'Original UTF-8 chapter bytes before first ## heading; no newline normalization.','matched_original_current_snapshot':intro['snapshot'],'original_owner_recheck_path':owner_recheck_path.relative_to(root).as_posix(),'original_owner_recheck_sha256':sha(owner_recheck_path),'preserved_prior_summary':True,'source_figure_artifact_versions_unchanged':True,'model_rerun':False,'result':'Current raw bytes, own nested hash/summary and actual owner recheck artifact match; standard field placement correction verified.'}
record_path=out/'evidence/intro-top-level-metadata-correction.json'
record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
report['artifacts'].extend([
    {'id':'a_before_intro_metadata','kind':'source_snapshot','path':backup.relative_to(root).as_posix(),'sha256':sha(backup),'description':'本人更正標準頂層導言欄位前的完整pass報告opaque副本；保留既有nested導言實查資訊及問題解決紀錄。'},
    {'id':'a_intro_metadata_code','kind':'code','path':Path(__file__).relative_to(root).as_posix(),'sha256':sha(__file__),'description':'本人實際執行的導言標準欄位更正腳本；核原始bytes、本人先前摘要/複查artifact及源碼/圖/證據版本。'},
    {'id':'a_intro_metadata_correction','kind':'execution','path':record_path.relative_to(root).as_posix(),'sha256':sha(record_path),'description':'本人查證nested current intro和標準頂層欄位相同的真實格式更正紀錄。','command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.1/standardize_intro_metadata.py','result':'exit 0；原始intro SHA與本人原snapshot/recheck一致，intro_source/intro_sha256/intro_summary補為標準顶層欄位；沒有重模型。','environment':{'python':sys.version,'device':'CPU byte/hash metadata verification only'}},
])
report.setdefault('metadata_corrections',[]).append({'reviewer_task':'/root/p6_fact_19_1','field_names':list(fields),'evidence_id':'a_intro_metadata_correction','scope':'Own previously substantively reviewed introduction; standard field placement only.'})
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

spec=importlib.util.spec_from_file_location('check_technical_reviews',root/'scripts/check_technical_reviews.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
source=root/'course/chapters/19.md'
body=next(b for lesson,b in checker.sections(source) if lesson=='19.1')
errors=checker._validate(root,source,'19.1',body,report,{'/root/p6_fact_19_1':['19.1']},set())
assert not errors,errors
validated={'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.1/standardize_intro_metadata.py (own checker._validate only after full report write)','report_sha256':sha(path),'verdict':report['verdict'],'errors':errors,'intro_top_level_checked':fields,'result':'PASS: standard introduction fields match actual raw bytes and prior owner summary; own full schema/source/figure/artifact/status check valid.','independence_check_scope':'Own report only; no other reader or technical report accessed. Full dispatch uniqueness held by coordinator.'}
validation.write_text(json.dumps(validated,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'intro_sha256':current_sha,'intro_source':report['intro_source'],'canonical_report_sha256':sha(path),'errors':errors,'model_rerun':False},ensure_ascii=False,indent=2))
