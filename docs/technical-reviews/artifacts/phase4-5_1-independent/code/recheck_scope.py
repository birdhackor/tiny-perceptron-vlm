"""Own 5.1 administrative scope recheck; no model run or other-lesson judgment."""
from pathlib import Path
from datetime import datetime, UTC
import ast
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT/'docs/technical-reviews/artifacts/phase4-5_1-independent'
RECHECK = OUT/'scope-recheck'
RECHECK.mkdir(exist_ok=False)
REL = OUT.relative_to(ROOT).as_posix()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
def sections(raw):
    headings = list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+',raw))
    return {match[1].decode():raw[match.start():(headings[i+1].start() if i+1<len(headings) else len(raw))] for i,match in enumerate(headings)}
def function_bytes(raw, names):
    lines = raw.splitlines(keepends=True)
    return {node.name:b''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(raw).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names}
now = datetime.now(UTC).isoformat()
manifest = json.loads((OUT/'input-manifest.json').read_bytes())
report_path = ROOT/'docs/technical-reviews/5.1.json'
old_report_raw = report_path.read_bytes()
old_report = json.loads(old_report_raw)
history = ROOT/'docs/technical-reviews/history/phase4-5_1-own-pass-before-whole-file-scope-da6c4d75525501e553baa78b67954e9d5dd1489d7f0b0361466ae985c43710bb.json'
assert digest(old_report_raw) == 'da6c4d75525501e553baa78b67954e9d5dd1489d7f0b0361466ae985c43710bb'
assert history.read_bytes() == old_report_raw
old_full = subprocess.check_output(['git','show','ae8b2795c918b70773e664b6d6731c054ecff74e:course/chapters/05.md'],cwd=ROOT)
assert digest(old_full)==manifest['source_file_sha256']
(RECHECK/'initial-chapter-05-from-git.md').write_bytes(old_full)
current_full=(ROOT/'course/chapters/05.md').read_bytes()
current_sections=sections(current_full)
old_sections=sections(old_full)
current_body=current_sections['5.1']
intro=current_full[:current_full.index(b'## 5.1 ')]
assert current_body==(OUT/'original/section.md').read_bytes()
assert intro==(OUT/'original/chapter-intro.md').read_bytes()
assert digest(current_body)==old_report['source_sha256']
assert digest(intro)==old_report['intro_sha256']
assert current_sections.keys()==old_sections.keys()
changed_ids=[name for name in current_sections if current_sections[name]!=old_sections[name]]
assert changed_ids==['5.4']
(RECHECK/'current-chapter-05.md').write_bytes(current_full)
(RECHECK/'current-section-5_1.md').write_bytes(current_body)
(RECHECK/'current-chapter-intro.md').write_bytes(intro)
fences=list(re.finditer(rb'```python\n(.*?)```',current_body,re.S))
assert len(fences)==1 and fences[0][1]==(OUT/'original/fence-1.py').read_bytes()
dependency_checks=[]
for name in ['tiny_perceptron/model.py','tiny_perceptron/data.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py','scripts/build_course.py','docs/review-tools/section_facts.py','scripts/check_technical_reviews.py','docs/course-experiments/results/text_foundation.json']:
    raw=(ROOT/name).read_bytes()
    snapshot=(OUT/manifest['current_inputs'][name]['snapshot']).read_bytes()
    assert raw==snapshot,name
    dependency_checks.append(dict(path=name,sha256=digest(raw),same_initial_snapshot=True))
source=ast.parse((ROOT/'scripts/build_course.py').read_bytes())
bootstrap=ast.literal_eval(next(n.value for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))).encode()
assert bootstrap==(OUT/'original-execution/bootstrap.py').read_bytes()
function_checks=[]
selected={'scripts/course_experiments/common.py':['new_lm','split_records','records_sha256','text_examples','_nll','fit_lm'],'scripts/course_experiments/text.py':['_steps','_save_splits','_evaluations','run_text_foundation'],'scripts/prepare_data.py':['generate_records']}
for name,names in selected.items():
    original_raw=(OUT/'original/experiment-version'/name).read_bytes()
    current_raw=(ROOT/name).read_bytes()
    a=function_bytes(original_raw,names);b=function_bytes(current_raw,names)
    assert set(a)==set(names)==set(b)
    for fn in names:
        assert a[fn]==b[fn],(name,fn)
        function_checks.append(dict(path=name,function=fn,sha256=digest(a[fn]),same_original_experiment_function_bytes=True))
    # Whole common/text files can change outside these functions; those areas are not reviewed.
proof_checks=[]
for artifact in old_report['artifacts']:
    raw=(ROOT/artifact['path']).read_bytes()
    assert digest(raw)==artifact['sha256'],artifact['id']
    proof_checks.append(dict(id=artifact['id'],path=artifact['path'],sha256=artifact['sha256'],unchanged=True))
installed=ROOT/'.venv/lib/python3.13/site-packages/torch'
installed_checks=[]
for item in json.loads((OUT/'sources/installed-identity.json').read_bytes()):
    path=ROOT/item['installed_path'];raw=path.read_bytes()
    assert digest(raw)==item['installed_sha256']
    installed_checks.append(dict(path=item['installed_path'],sha256=digest(raw),same_original_verified_official_bytes=True))
assert importlib.metadata.version('torch')=='2.14.1+cpu'
receipt=dict(
    kind='own_section_reading_and_evidence_reuse_receipt',reviewer_task=old_report['reviewer_task'],read_on=now,
    source='course/chapters/05.md#5.1',current_section_sha256=digest(current_body),current_intro_sha256=digest(intro),
    initial_source_file_sha256=digest(old_full),current_source_file_sha256=digest(current_full),
    initial_snapshot=dict(path=f'{REL}/scope-recheck/initial-chapter-05-from-git.md',sha256=digest(old_full),initial_read_on='2026-10-05',recovered_on=now,recovery_command='git show ae8b2795c918b70773e664b6d6731c054ecff74e:course/chapters/05.md',recovery_check='Exact bytes match previously observed whole-file SHA; not reconstructed by editing another section.'),
    current_snapshot=dict(path=f'{REL}/scope-recheck/current-chapter-05.md',sha256=digest(current_full),meaning='Aggregate whole-file identity only; reviewed text range remains intro and 5.1.'),
    actual_reading=[
      dict(path='course/chapters/05.md',range='lines1–49: entire chapter title/intro and 5.1, including fence, interpretation, variation, empirical paragraph',finding='親讀目前全文：固定ID/三個shifted答案、四步更新、norm訊號、all-ignore guard與既有GPU數字全部和自己原文snapshot一致；未讀/未評5.4正文。'),
      dict(path='tiny_perceptron/model.py',range='ModelConfig15–28, Block31–50, TinyLM53–105',finding='親讀目前lookup、causal blocks、max_length及loss_sum/count契約；原CPU實測inputs未改。'),
      dict(path='tiny_perceptron/attention.py',range='attention_mask/manual_attention/CausalAttention lines10–88',finding='親讀本例manual causal分支；內容與當時snapshot完全相同。'),
      dict(path='tiny_perceptron/modern.py',range='DenseFFN lines38–56',finding='親讀本例GELU FFN，不引入train/eval隨機差；原bytes未改。'),
      dict(path='tiny_perceptron/data.py',range='IGNORE10, ByteTokenizer14–28, shifted46–51, pad_batch71–86',finding='親讀有效目標、byte/EOS/−100和padding；原實測分母解讀未改。'),
      dict(path='scripts/course_experiments/common.py',range='new_lm/split_records/text_examples/_nll/fit_lm referred functions',finding='親讀目前必要函式並對照原實測版本；這些函式原bytes逐項一致。整個檔案別處不同，未宣稱整檔同版本。'),
      dict(path='scripts/course_experiments/text.py',range='_steps/_save_splits/_evaluations/run_text_foundation referred functions',finding='親讀目前600步/two-layer入口與必要計分資料流程；必要函式逐bytes同原實測版本。'),
      dict(path='scripts/build_course.py',range='BOOTSTRAP lines26–62',finding='親讀原fence執行的CPU準備/import/seed區塊，實際bootstrap bytes未改。'),
      dict(path='docs/review-tools/factual-reviewer-instructions.md',range='current method including frozen-input scope rule',finding='親讀最新方法。此文件changed，不是技術答案或程式依賴；按其要求保留初次全檔snapshot並明標範圍。'),
      dict(path=f'{REL}',range='own original section/intro/fence, input-manifest, execution/environment/stdout, variation, boundary and empirical-audit proof files',finding='親讀自己原proof並驗所有原report artifact SHA；trace與計分分母仍支持六個原claim，不複製其他審閱的結論。'),
    ],
    own_intro_summary_after_rereading='章首先把一題接字的更新流程跑通，再組成批次並認識優化器、步幅與存檔；評估時另看新題，區分有效目標、不同材料、參數格數和秒數。',
    changed_section_ids_by_raw_bytes_only=changed_ids,
    reviewed_scope='5.1 and chapter intro only, plus actually necessary implementation and own frozen proof; changed 5.4 content detected by bytes only and not reviewed.',
    dependency_checks=dependency_checks,empirical_function_checks=function_checks,original_artifact_checks=proof_checks,installed_source_checks=installed_checks,
    environment=dict(python=sys.version,python_executable=sys.executable,torch_distribution=importlib.metadata.version('torch'),device_policy='No model execution; original CPU proof reused after identical-input checks'),
    prior_pass=dict(path=history.relative_to(ROOT).as_posix(),sha256=digest(old_report_raw)),
    judgment='pass preserved; all section/intro/fence/necessary dependency inputs unchanged. Original claims remain supported. Whole-file SHA now explicitly means initial frozen input, not current whole-chapter acceptance.',
    rerun_scope='Only this receipt metadata/byte checks and subsequent check_technical_reviews --lesson 5.1; no numerical model probe, training, GPU, downloads, figure changes or other reports.',
)
receipt_path=RECHECK/'reading-reuse-receipt.json'
write_json(receipt_path,receipt)
updated=old_report
del updated['source_file_sha256']
updated['initial_snapshot']=dict(source_file='course/chapters/05.md',source_file_sha256=digest(old_full),initial_read_on='2026-10-05',snapshot_artifact_id='phase4-5_1-initial-chapter-frozen-input',snapshot_path=receipt['initial_snapshot']['path'],scope='Whole-file identity of the initial frozen input; technical acceptance covers only chapter intro and 5.1.',provenance='Recovered exact original bytes from recorded initial HEAD git object; SHA matches original input-manifest and preserved first PASS.')
updated['version_scope_recheck']=dict(read_on=now,receipt_artifact_id='phase4-5_1-scope-reading-reuse',source_sha256=digest(current_body),intro_sha256=digest(intro),current_source_file_sha256=digest(current_full),current_file_hash_meaning='Identity of current aggregate input snapshot, not review of other 05 lessons.',same_section_intro_fence_and_necessary_dependencies=True,original_proof_reused=True,model_execution_rerun=False)
updated['read_scope']['administrative_recheck']='親讀新版5.1與導言及真正依賴；所有本節inputs與原proof相同。5.4只由raw-byte差異偵測，不作其他節審閱。全檔initial/current指紋在explicit scope metadata區分。'
updated['checks']['source_verification']['details']+=' 本次由同位審閱者亲读自己的当前范围并核所有原proof SHA；初次全檔指紋改置明標initial_snapshot，沒有把舊全檔SHA冒充目前整章。'
def register(identifier,path,kind,description):
    updated['artifacts'].append(dict(id=identifier,path=path.relative_to(ROOT).as_posix(),sha256=digest(path.read_bytes()),kind=kind,description=description))
register('phase4-5_1-scope-reading-reuse',receipt_path,'source_snapshot','同位審閱者親讀目前5.1／導言／真正依賴與原proof，逐bytes核對並記錄支持範圍與reuse界限。')
register('phase4-5_1-initial-chapter-frozen-input',RECHECK/'initial-chapter-05-from-git.md','source_snapshot','初次整個Markdown frozen input的精確原bytes；全檔SHA只代表當次輸入身分，不代表其他小節通過。')
register('phase4-5_1-current-chapter-scope-input',RECHECK/'current-chapter-05.md','source_snapshot','本次current whole-file aggregate輸入快照；實際審閱範圍是intro和5.1，其他小節不作判斷。')
register('phase4-5_1-scope-recheck-code',OUT/'code/recheck_scope.py','code','實際有界讀取/bytes/依賴/proof比對與metadata更新程式；沒有model/GPU/train執行。')
updated['review_history'].append(dict(round='phase4-own-whole-file-scope-administrative-recheck',read_on=now,verdict='pass',source_sha256=digest(current_body),intro_sha256=digest(intro),prior_pass_sha256=digest(old_report_raw),prior_pass_path=history.relative_to(ROOT).as_posix(),receipt_artifact_id='phase4-5_1-scope-reading-reuse',scope='Own unchanged lesson/intro/dependencies and original proof only; initial frozen whole-file provenance clarified.',issues=[]))
write_json(report_path,updated)
print(json.dumps(dict(verdict=updated['verdict'],report_sha256=digest(report_path.read_bytes()),receipt_artifact_id='phase4-5_1-scope-reading-reuse',receipt_sha256=digest(receipt_path.read_bytes()),initial_source_file_sha256=digest(old_full),current_source_file_sha256=digest(current_full),source_sha256=digest(current_body),intro_sha256=digest(intro),changed_section_ids_by_bytes=changed_ids,scope=receipt['reviewed_scope']),ensure_ascii=False,indent=2))
