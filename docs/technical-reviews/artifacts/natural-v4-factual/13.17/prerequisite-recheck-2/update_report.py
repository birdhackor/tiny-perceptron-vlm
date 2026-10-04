"""Update own current report only after genuine affected prerequisite recheck."""
from pathlib import Path
import copy
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[6]
ART=Path(__file__).resolve().parent
REL=ART.relative_to(ROOT)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
prior_path=ART/'prior-current-pass.json';prior=json.loads(prior_path.read_text())
assert sha(prior_path)=='5385c09b343ed2b3ab640ef9d91643a815e44548a775234ff8e7dd57132e6920'
result=json.loads((ART/'probe-result.json').read_text())
assert result['current_and_prior_python_block_identical'] and result['remaining_issues']==[]
assert result['main_bytes_identical'] and result['all_original_code_and_record_fingerprints_unchanged']
assert len(prior['claims'])==22
report=copy.deepcopy(prior)
command='.venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/13.17/prerequisite-recheck-2/probe.py > docs/technical-reviews/artifacts/natural-v4-factual/13.17/prerequisite-recheck-2/probe.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/13.17/prerequisite-recheck-2/probe.stderr.txt'
environment={k:str(v) for k,v in result['environment'].items()}
def artifact(id,kind,name,description):
    value={'id':id,'kind':kind,'path':str(REL/name),'sha256':sha(ART/name),'description':description}
    if kind=='execution':value.update(command=command,environment=environment,
       result='Exit0; exact current prerequisite code and exercise executed; range[0,1],scores[1,14],chosen1/False; exercise[1,0],chosen0/True; all main/source/evidence assertions passed. No model training or new figure viewing.')
    report['artifacts'].append(value)
artifact('a-r2-prior-pass','source_snapshot','prior-current-pass.json','保全本人真原当前pass5385c09b…及22项原判断，不改旧报告。')
artifact('a-r2-prior-prereq','source_snapshot','prior-closure-current-prerequisite-13.16.md','本人先前真正读过的必要13.16原raw12f128d…保全。')
artifact('a-r2-current-prereq','source_snapshot','current-13.16.md','本人真正完整读取的新必要13.16 raw99b7866…，包括全部空行。')
artifact('a-r2-diff','source_snapshot','prerequisite.diff.txt','本人原current必要前置原字节的实际diff，定位新增range/key/lambda解释。')
artifact('a-r2-integrity','source_snapshot','prior-evidence-integrity.json','原程序/原正式record/本人旧artifact SHA未变的逐项比对；章13整文件因必要前置修改改变。')
artifact('a-r2-authority-retrieval','source_snapshot','authority-retrieval-receipts.json','本人实际下载CPython v3.13.5官方原文的版本、URL、date、retrievalSHA收据；完整文档只存ignored research。')
artifact('a-r2-inspection','source_snapshot','inspection.md','本人新必要前置完整读取、官方语法原定位与main受影响范围裁决；诚实沿用原五图view及CPU证据。')
artifact('a-r2-code','code','probe.py','实际执行完整current13.16块和练习的短CPU核对代码，无模型训练。')
artifact('a-r2-execution','execution','probe-result.json','真实新语法/原current代码相同/未变main五图和程序证据核对结果。')
artifact('a-r2-stdout','execution','probe.stdout.txt','实际短Python语法核对stdout，含预期/观察/环境和scope。')
artifact('a-r2-stderr','execution','probe.stderr.txt','实际短Python语法核对stderr；exit0且空。')
receipts=json.loads((ART/'authority-retrieval-receipts.json').read_text())
def authority(id,index,title,note):
    receipt=receipts[index]
    report['sources'].append({'id':id,'kind':'official_docs','title':title,'url':receipt['url'],
       'version':receipt['version'],'verified':True,'checked_original':True,'accessed_on':receipt['accessed_on'],
       'authority_reason':'CPython官方项目v3.13.5文档tag，原始语言/API文档与实际Python3.13.5执行环境匹配。',
       'inspection_note':note,'retrieval_sha256':receipt['retrieval_sha256']})
authority('s-r2-python-functions',0,'CPython3.13.5 len/max/key documentation',
  '本人实际下载并读Doc/library/functions.rst lines1127–1131与1211–1232：len返回项目数，max按单参数key排序并返回原项目而非key值。此例无并列。当前13.16完整块实际执行验证。')
authority('s-r2-python-range',1,'CPython3.13.5 range documentation',
  '本人实际下载并读Doc/library/stdtypes.rst Ranges lines1385–1411：range为不可变序列，start0/step1默认，正步长内容<stop。range(2)提供0/1；正文未声称为list。')
authority('s-r2-python-lambda',2,'CPython3.13.5 lambda expressions documentation',
  '本人实际下载并读Doc/reference/expressions.rst Lambdas lines1881–1907：lambda参数列表与单表达式创建匿名函数，相当于return expression。本例按编号取scores值。')
report['sources'].append({'id':'s-r2-execution','kind':'execution','title':'本人必要前置2实际短Python执行','verified':True,'artifact_id':'a-r2-execution'})
for s in report['sources']:
    if s['id']=='s-context13':
        current=sha(ROOT/s['path']);s['sha256']=current;s['version']='current full-file SHA256 '+current
        s['inspection_note']+=' 必要前置recheck2：本人完整读取current13.16 raw99b7866…，逐字比对先前已读12f128d…；实际执行未变代码块和新range/key/lambda解释，官方CPython文档与观察相符；main及所有程序/数值/SVG不变，非机械换SHA。原报告/前置/22claims/权威/CPU/5图记录保全见a-r2-prior-pass/a-r2-inspection。'
for entry in report['scope_provenance']['current_prerequisites']:
    if entry['source']=='course/chapters/13.md#13.16':
        entry.update(sha256=result['current_prerequisite13_16_sha256'],bytes=(ART/'current-13.16.md').stat().st_size,
           recheck2_original_sha256=result['prior_prerequisite13_16_sha256'],actual_full_reread=True,
           evidence_artifact_id='a-r2-execution')
for c in report['claims']:
    if c['id'] in result['closest_affected_main_claims']:
        c['artifact_ids'].append('a-r2-execution')
        c['scope']+=' 必要前置recheck2仅新增range/key/lambda语法说明；原代码/反例/结果未变，短执行及官方Python定位核实该说明，本项原结论不变。'
        if c['id']=='c21':
            c['evidence'].extend([
              {'source_id':'s-r2-python-functions','locator':'Doc/library/functions.rst lines1211–1232 (max/key)',
               'supports':'必要前置人工长度反例按分数取原编号，不能把所选分数当任务正确性。'},
              {'source_id':'s-r2-python-range','locator':'Doc/library/stdtypes.rst lines1385–1411 (Ranges)',
               'supports':'未变两候选反例提供编号0/1；新增说明准确。'},
              {'source_id':'s-r2-python-lambda','locator':'Doc/reference/expressions.rst lines1881–1907 (Lambdas)',
               'supports':'新句解释的lambda按编号取分数，原反例语义不变。'}])
report['scope_provenance']['prerequisite_recheck_2']={
    'reviewer_task':report['reviewer_task'],'artifact_id':'a-r2-execution',
    'prior_pass_sha256':result['prior_pass_sha256'],'actual_full_new_prerequisite_read':'course/chapters/13.md#13.16',
    'prior_prerequisite_sha256':result['prior_prerequisite13_16_sha256'],
    'current_prerequisite_sha256':result['current_prerequisite13_16_sha256'],
    'main_full_read_reused':True,'main_bytes_unchanged':True,'original_22_claims_preserved':True,
    'all_5_original_figure_views_reused':True,'new_figure_render_or_view':False,
    'original_model_training_and_derivations_reused':True,'new_model_training':False,
    'short_current_prerequisite_python_execution':True,'new_issues':result['remaining_issues'],
    'scope_judgment':result['scope_judgment']}
report['checks']['factual_accuracy']['details']+=' 必要前置recheck2本人真full读current13.16并核新增语法，22项main判断不变。'
report['checks']['numeric_verification']['details']+=' 本次只执行未变13.16块与正确性练习：scores1/14选1(False)，规则1/0选0(True)；原训练与main数值证据沿用。'
report['checks']['source_verification']['details']+=' 新Python语法核CPythonv3.13.5官方原docs实际阅读定位/retrieval SHA；必要前置current原字节保存，非机械hash替换。'
report['checks']['figure_consistency']['details']+=' 本次五SVG current原字节逐项相同，沿用本人既有五图actual view，不声称新render/view。'
report['checks']['limitations']['details']+=' 本次是已闭合原main的affected prerequisite recheck：main旧full-read与五图view/CPU仍适用，未制造新整页阅读、benchmark或训练。'
report['issues'].extend(result['remaining_issues'])
report['verdict']='pass' if not report['issues'] and all(c['status']=='verified' for c in report['claims']) else 'revise'
assert report['source_sha256']==prior['source_sha256']==result['main_source_sha256']
assert report['figure_sha256']==prior['figure_sha256'] and len(report['claims'])==22
path=ROOT/'docs/technical-reviews/13.17.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(ART/'updated-report.json').write_bytes(path.read_bytes())
print('Own affected-prerequisite judgment',report['verdict'],'mainSHA unchanged',report['source_sha256'],
      'claims',len(report['claims']),'issues',report['issues'],'reportSHA',sha(path))
