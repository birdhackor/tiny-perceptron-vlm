"""Read-only callback facts: current sections/contracts and unchanged evidence hashes.

No torch import, inference, training, external retrieval, preview or upload.
"""
import ast
import difflib
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def section(path, label):
    raw = path.read_bytes()
    headers = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
    pos = next(i for i, h in enumerate(headers) if h[0].startswith(b'## ' + label.encode() + b' '))
    end = headers[pos + 1].start() if pos + 1 < len(headers) else len(raw)
    start = headers[pos].start()
    body = raw[start:end]
    return body, raw[:start].count(b'\n') + 1

archive = json.loads((OUT / 'archive.receipt.json').read_bytes())
prior_path = ROOT / archive['prior_opaque_file']
assert digest(prior_path.read_bytes()) == archive['prior_sha256']
prior = json.loads(prior_path.read_bytes())  # Only this reviewer's own report.
assert prior['reviewer_task'] == '/root/phase4_factual_coordinator/factual_1_14'
body, first = section(ROOT / 'course/chapters/01.md', '1.14')
assert digest(body) == prior['source_sha256'] == '3bd128983d6aa0c7a60405d0fa679d6db83ecf811c4c1dffcc2f3030191fd5e1'
assert body == (BASE / 'original/section.md').read_bytes()
(OUT / 'section.current.raw.md').write_bytes(body)
t3, t3first = section(ROOT / 'course/training.md', 'T.3')
(OUT / 'T.3.current.raw.md').write_bytes(t3)
previous_t3_path = BASE / 'scope-recheck/T.3.current.raw.md'
previous_t3 = previous_t3_path.read_bytes()
t3delta = ''.join(difflib.unified_diff(previous_t3.decode('utf-8').splitlines(keepends=True), t3.decode('utf-8').splitlines(keepends=True), fromfile='own previously read T.3 raw', tofile='current T.3 raw'))
(OUT / 'T.3.source-delta.txt').write_text(t3delta)

artifact_checks = []
for artifact in prior['artifacts']:
    path = ROOT / artifact['path']
    actual = digest(path.read_bytes())
    assert actual == artifact['sha256'], artifact['id']
    artifact_checks.append({'id': artifact['id'], 'path': artifact['path'], 'expected_sha256': artifact['sha256'], 'actual_sha256': actual, 'unchanged': True, 'reused_execution': artifact['kind'] == 'execution'})
local_source_checks = []
for source in prior['sources']:
    if source['kind'] == 'repository_code':
        actual = digest((ROOT / source['path']).read_bytes())
        assert actual == source['sha256'], source['id']
        local_source_checks.append({'id':source['id'], 'path':source['path'], 'actual_sha256':actual, 'unchanged':True})
assert prior['figure_sha256'] == {}
assert not re.findall(rb'!\[[^\]]*\]\([^)]+\.svg\)', body)

code_paths = ['tiny_perceptron/simple.py', 'tiny_perceptron/data.py', 'scripts/course_experiments/text.py', 'scripts/infer_simple.py']
contracts = {}
for path in code_paths:
    raw = (ROOT / path).read_bytes()
    previous = (BASE / 'code' / path).read_bytes()
    assert raw == previous, path
    contracts[path] = {'sha256':digest(raw), 'matches_prior_code_snapshot':True}

text_tree = ast.parse((ROOT / 'scripts/course_experiments/text.py').read_bytes())
fn = next(n for n in text_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_simple_models')
loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
configurations = ast.literal_eval(loop.iter)
assert configurations == (('bigram', 1), ('mlp1', 1), ('mlp3', 3), ('mlp5', 5))

new_snapshots = {}
for path in ['scripts/train_simple.py', 'scripts/course_experiments/run.py', 'docs/course-experiments/plan.json', 'docs/review-tools/factual-reviewer-instructions.md', '.agents/skills/clear-tutorial/references/review-protocol.md']:
    raw = (ROOT / path).read_bytes()
    target = OUT / 'source-snapshots' / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    new_snapshots[path] = {'sha256':digest(raw), 'snapshot':target.relative_to(ROOT).as_posix()}

plan = json.loads((ROOT / 'docs/course-experiments/plan.json').read_bytes())
items = [(i,item) for i,item in enumerate(plan['sequence']) if item.get('id') == 'simple_models']
assert len(items) == 1
index, item = items[0]
plan_pointer = '/sequence/' + str(index)
plan_fields = {key:item[key] for key in ('id','module','function','assets')}
assert plan_fields == {'id':'simple_models','module':'text','function':'run_simple_models','assets':[]}

note = {
    'schema_version':1, 'kind':'same_owner_narrow_callback_facts', 'reviewer_task':prior['reviewer_task'], 'callback_date':'2026-10-06',
    'prior_opaque_file':archive['prior_opaque_file'], 'prior_opaque_sha256':archive['prior_sha256'],
    'current_section':{'source':'course/chapters/01.md#1.14','sha256':digest(body),'first_line':first,'last_line':first + body.count(b'\n') - 1,'actual_inspection':'親自重讀全文；四字手動概率、抽樣/current回填、句號不停、歷史合成欄位例子及T.3連結均未改；與自己原CPU輸入snapshot bytes完全相同。'},
    'current_T3':{'source':'course/training.md#T.3','sha256':digest(t3),'first_line':t3first,'last_line':t3first + t3.count(b'\n') - 1,'actual_read_scope':'親自重讀完整目前T.3；這是實際讀取範圍，不表示整節technical PASS。','necessary_scope':'1.14 L553的完整配方/原結果入口：T.3開頭12篇合成短文/9–1–2切分、中文simple字元模型身份與--train/200步配方；末段simple_models固定比較入口及原始結果JSON連結。Transformer/text_foundation/byte JSON只為確認沒有和1.14混淆；不重審其模型能力/命令實測。','dependency_reason':'T.3是1.14引用的重做/證據定位入口，不是原碼13字生成原理的必要前置；C5對既有實驗的證明仍來自原始data/result/實作及自己原有CPU證據，而非T.3文字自證。','prior_declared_read_scope':prior['read_scope']['linked_recipe'],'prior_frozen_T3_raw':{'path':previous_t3_path.relative_to(ROOT).as_posix(),'sha256':digest(previous_t3)},'delta':'固定比較句由四種模型改為兩類模型、四個設定，加入2.5導覽。從實作tuple獨立確認bigram/context1，MLP/context1、3、5；其他實測/字元配方內容未因該句改動。','explicit_boundary':'不延伸到2.5全節、不替T.3整節判定，不讀作者repair或別人的報告。'},
    'actual_contract_inspection':[
        {'source':'scripts/course_experiments/text.py','locator':'run_simple_models L128–143, L181–202；AST tuple L136','supports':'固定比較確是bigram+MLP兩類、1/3/5 MLP三個窗口設定；9文件train/vocabulary與_samples兩prompt/結果JSON入口仍由原實作決定。'},
        {'source':'scripts/train_simple.py','locator':'main L26–63, L64–90；AST定位main L26–90','supports':'--model選bigram/mlp；默認seed42/steps200/context3/width16；無--train只有梯度且optimizer.step僅在args.train分支。只讀recipe契約，不執行完整配方。'},
        {'source':'scripts/course_experiments/run.py','locator':'execute L58–79；main L165–204','supports':'module/function從具名spec讀取，context seed42；--experiment simple_models/--device cpu由現CLI映到run_simple_models；不執行長訓練。'},
        {'source':'docs/course-experiments/plan.json','locator':plan_pointer,'fields_read':plan_fields,'supports':'精確的simple_models module=text/function=run_simple_models，assets=[]；僅讀上層key型別與該項具名必要欄位，沒有notes/review或作者结果評語。'},
    ],
    'configurations_observed_from_AST':configurations, 'contracts_unchanged':contracts,
    'prior_artifacts_actual_sha256_check':artifact_checks, 'prior_local_sources_actual_sha256_check':local_source_checks,
    'new_source_snapshots':new_snapshots,
    'no_figure_analysis':{'referenced_figures':0,'need_to_imagine_materials':False,'reason':'字表、機率列、current回填以四字列表/具體數字/短碼直接展示；相同等號末字可由兩個prefix文字核對，不需未提供的圖片內容、空間素材或標籤對應。','current_rendering':'未重新檢查preview桌面/手機呈現；本回呼沒有可呼叫browser工具，也沒有以rootparity替代本人證據。source-body/圖引用未變，這次只作必要引用/契約版本查核。'},
    'environment':{'python':platform.python_version(),'python_executable':sys.executable,'platform':platform.platform(),'installed_torch_distribution':importlib.metadata.version('torch'),'device':'no model execution; hash/AST/file checks only'},
    'reused_evidence_scope':'原樣CPU fence、seed/length/punctuation變化及public simple推論/無更新NLL仍是2026-10-05本人執行的舊證據；本次逐檔SHA核對後沿用，不寫成2026-10-06的新推論/訓練。官方原來源snapshot與版本未變，不重抓。',
    'verdict':'pass','unresolved_substantive_issues':[],
}
(OUT / 'callback-facts.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'kind':note['kind'],'reviewer_task':note['reviewer_task'],'section_sha256':digest(body),'current_T3_sha256':digest(t3),'prior_T3_sha256':digest(previous_t3),'unchanged_artifacts_verified':len(artifact_checks),'unchanged_local_sources_verified':len(local_source_checks),'configs':configurations,'verdict':'pass','environment':note['environment']},ensure_ascii=False))
