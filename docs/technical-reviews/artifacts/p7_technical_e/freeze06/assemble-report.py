import copy
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path('docs/technical-reviews/artifacts/p7_technical_e')
CALLBACK = BASE / 'freeze06'
OLD_REPORT = Path('docs/technical-reviews/phase7-freeze03-e-p7_technical_e-initial.json')
NEW_REPORT = Path('docs/technical-reviews/phase7-freeze06-e-p7_technical_e-recheck.json')
MANIFEST = Path('docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json')
OLD_MANIFEST = Path('docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json')
TRACE = 'docs/course-revision-20261007-phase7/reviews/freeze-06/traces/technical/e/3de5625c9e5c4a5e840afae7e0d294d8.jsonl'
OWNER = '/root/p7_technical_e'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), str(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

assert sha(OLD_REPORT) == '60efd82dd26f75c39cc1c8a54554e106d8348bce9a9e11bb00f734f70dc4c4fd'
assert sha(MANIFEST) == '0bbf4113fbd8288786bca540330fa9e2109e4fe848304a43b0321efa7bd23709'
old = json.loads(OLD_REPORT.read_text())
manifest = json.loads(MANIFEST.read_text())
old_manifest = json.loads(OLD_MANIFEST.read_text())
pages = {p['page_id']: p for p in manifest['pages']}
old_pages = {p['page_id']: p for p in old_manifest['pages']}
report = copy.deepcopy(old)
report['manifest_sha256'] = sha(MANIFEST)
report['trace_files'].append({'path': TRACE, 'sha256': sha(TRACE)})
comparisons = []
for page in report['pages']:
    pid = page['page_id']
    now, previous = pages[pid], old_pages[pid]
    assert sha(now['snapshot']) == now['source_sha256']
    assert sha(previous['snapshot']) == previous['source_sha256']
    same = (now['source_sha256'] == previous['source_sha256'] and
            now['figures_sha256'] == previous['figures_sha256'])
    assert same == (pid not in ['18.4', '19.9']), pid
    comparisons.append({'page_id': pid, 'source_sha256_before': previous['source_sha256'],
                        'source_sha256_after': now['source_sha256'],
                        'figures_sha256_before': previous['figures_sha256'],
                        'figures_sha256_after': now['figures_sha256'],
                        'same_canonical_and_figures': same,
                        'evidence_reused_without_changes': same})
comparison_path = CALLBACK / 'unchanged-pages.json'
if comparison_path.exists():
    assert json.loads(comparison_path.read_text()) == comparisons
else:
    write_new(comparison_path, comparisons)

def execution(name, aid, description, result):
    prefix = BASE / 'executions' / name
    meta = json.loads(Path(str(prefix) + '.execution.json').read_text())
    out = Path(str(prefix) + '.stdout.txt')
    assert sha(out) == meta['stdout_sha256'] and meta['exit_code'] == 0
    assert sha(Path(str(prefix) + '.py')) == meta['code_sha256']
    return {'id': aid, 'kind': 'execution', 'path': str(out), 'sha256': sha(out),
            'description': description, 'result': result,
            'command': meta['command'], 'environment': meta['environment']}

def exec_source(aid, title):
    return {'id': aid, 'kind': 'execution', 'title': title, 'verified': True, 'artifact_id': aid}

def official(aid, title, relative, local, locator):
    return {'id': aid, 'kind': 'official_source', 'title': title, 'verified': True,
            'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/' + relative,
            'version': 'v2.5.1', 'authority_reason': 'PyTorch 原維護者的版本化 API 原碼與 docstring。',
            'checked_original': True, 'inspection_note': locator + ' 本人真讀；此歷史 API 支持機制，本 installed 2.14.1+cpu 的行為另外用本人新 CPU 核對。',
            'accessed_on': '2026-10-07', 'snapshot_path': str(BASE / 'sources' / local),
            'snapshot_sha256': sha(BASE / 'sources' / local)}

def ev(source, locator, support):
    return {'source_id': source, 'locator': locator, 'supports': support}

def page_view(page, directory, names, observation):
    arts = []
    for name in names:
        path = Path(directory) / name
        width = int(name.split('-')[0])
        arts.append({'path': str(path), 'sha256': sha(path),
                     'viewport': [width, 800 if width == 1280 else 844]})
    receipt = CALLBACK / (page['page_id'] + '-page-view.json')
    write_new(receipt, {'tool': 'tools.view_image', 'reviewer_task': OWNER,
                       'received_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'source_sha256': page['source_sha256'], 'artifacts': arts,
                       'observation': observation})
    page['prior_page_visual_check'] = copy.deepcopy(page['page_visual_check'])
    page['page_visual_check'] = {'status': 'verified', 'required': True,
                               'source_sha256': page['source_sha256'], 'artifacts': arts,
                               'receipt': {'path': str(receipt), 'sha256': sha(receipt)},
                               'details': observation, 'observation': observation}

for page in report['pages']:
    pid = page['page_id']
    if pid not in ['18.4', '19.9']:
        continue
    page['source_sha256'] = pages[pid]['source_sha256']
    page['figures_sha256'] = pages[pid]['figures_sha256']
    page['trace_file'] = TRACE
    page['callback'] = {'prior_report': str(OLD_REPORT), 'prior_report_sha256': sha(OLD_REPORT),
                        'same_owner': True, 'new_blind_reader': False,
                        'scope': '整頁本人逐段回查；所有本人原證據、限制和更正保留；沒有原 issue ID。'}
    if pid == '18.4':
        page['summary'] += ' Freeze06 整頁回查：新 detach/clone/requires_grad_ 說明及零直接 logit 梯度不保固定機率，原 API 與本人新 CPU 均支持。'
        page['artifacts'].extend([
            execution('f06-18.4-example', 'callback_example', '本次逐字執行教材三路 CE/logit 梯度', 'hard CE .6931、soft .8987；兩 hard 同梯度、soft[-.2,.2,0]；未更新模型。'),
            execution('f06-18.4-api-variation', 'callback_api', '本次 storage/leaf/upstream 與 softmax 分母變式', 'detach 同 storage、clone 獨立；上游 grad None，新 leaf 有梯度；third score 不變但機率 .1→.09978246033317684；三分類 q−p 逐項相符。')])
        page['sources'].extend([
            exec_source('callback_example', '本人新原例 CPU 真輸出'),
            exec_source('callback_api', '本人新 API／機率變式 CPU 真輸出'),
            official('detach', 'Tensor.detach 原 API', 'torch/_tensor.py', 'pytorch-v2.5.1-tensor.py', '725–741：脫離反向／前向梯度圖，但共享 storage。'),
            official('clone', 'torch.clone 原 API', 'torch/_torch_docs.py', 'pytorch-v2.5.1-torch-docs.py', '2717–2734：複製 tensor，clone 自身可微分，斷連結要 detach。'),
            official('requires_grad', 'Tensor.requires_grad_ 原 API', 'torch/_tensor_docs.py', 'pytorch-v2.5.1-tensor-docs.py', '4198–4211：in-place 更改 requires_grad，開始記錄此 tensor 的操作。'),
            official('log_softmax', 'log_softmax 原 API', 'torch/nn/functional.py', 'pytorch-v2.5.1-functional.py', '2220–2254：log-softmax 與穩定形式／gradient。')])
        page['claims'].extend([
            {'id': 'c5', 'kind': 'software', 'statement': '原 q 不記錄梯度；detach 保留值而切斷上游梯度，clone 複製獨立資料，requires_grad_ 使新分數 leaf 記錄梯度；每路重建可避免梯度累加。',
             'location': 'unit1–2 新 API 說明', 'scope': '本例和有上游的單 leaf 變式，未更新模型。', 'status': 'verified',
             'evidence': [ev('detach', '725–741', 'detach 切圖且共享原 storage。'), ev('clone', '2717–2734', 'clone copy 與梯度條件。'), ev('requires_grad', '4198–4211', '開始記錄新輸入。'), ev('log_softmax', '2220–2254', '穩定 log-softmax 定義。'), ev('callback_api', 'script storage/assert/upstream probe 和 stdout', '真 CPU leaf/storage/上游無梯度與新 leaf 有梯度。')],
             'artifact_ids': ['callback_example', 'callback_api'],
             'verification': {'method': 'executed', 'expected': 'detach 同 storage、clone 異 storage；q 原先 requires_grad=False；上游切斷、新 leaf 有梯度。', 'observed': '所有 assertion 真 CPU 通過。', 'details': 'PyTorch2.14.1+cpu；v2.5.1 原 API 與 installed 行為分開。'}},
            {'id': 'c6', 'kind': 'numeric', 'statement': 'third logit 的直接梯度近零時，改其他 logits 可改 third 機率；本變式 score 不變而 .1→.09978246033317684。',
             'location': 'unit2 新分數／機率區別', 'scope': 'FP64 人工 logits 步幅 .1，非模型 optimizer step 或訓練。', 'status': 'verified',
             'evidence': [ev('callback_api', 'zero_direct_gradient / same_third_score / third_probability stdout', 'third grad 2.7755575615628892e−18，score 相同、機率變動且 sum=1。')],
             'artifact_ids': ['callback_api'], 'verification': {'method': 'executed', 'expected': 'grad≈0、third score相同，但 softmax denominator 隨其他 score 改變。', 'observed': '2.7755575615628892e−18；score exact same；.1→.09978246033317684；sum1。', 'details': '亦對三組分類梯度核 q−p；未預測測試品質。', 'tolerance': 'q−p/assert allclose atol1e−15, rtol0；third score exact equality。'}}])
        for key in ['factual_accuracy', 'numeric_verification', 'source_verification', 'limitations']:
            page['checks'][key]['claim_ids'].extend(['c5', 'c6'])
        page['checks']['factual_accuracy']['details'] += ' 新 API 個別功能與機率/score 區別本人核實。'
        page['checks']['numeric_verification']['details'] += ' 本次重跑原例及 FP64 分母變式，梯度 tolerance 1e−15。'
        page['checks']['source_verification']['details'] += ' 本人官方版本化 Tensor/clone/requires_grad/log_softmax 原 API 真定位。'
        page['checks']['limitations']['details'] += ' 明示人工 logits 計算，不把 backward 或機率変化當訓練成果。'
        page['missing_visuals'] = '無 SVG；四張新段落 desktop/mobile 本人真view；數值與文字對應足夠。'
        page_view(page, 'outputs/phase7-render/pages/p7_technical_e/18.4/expanded/20261007T073624761952Z-96653e7c',
                  ['1280-text0-match0-0.png', '1280-text1-match0-0.png', '390-text0-match0-0.png', '390-text1-match0-0.png'],
                  '本人真 tools.view_image 看四張新定位：兩 viewport 的 detach/clone/requires_grad_ 新段落、直接零梯度與機率分母說明均完整易讀；桌面可見展開推理／歷史 cache 補充，手機第二圖只看到分類練習局部，不宣稱整頁每個位置全view。')
    else:
        page['summary'] += ' Freeze06 整頁回查：新 eval/no_grad 段，原官方 API、本人原模型 layer source 與新 CPU 模式 probe 支持；沒有時間或品質宣稱。'
        page['artifacts'].extend([
            execution('f06-19.9-example', 'callback_example', '本次逐字執行教材 full5/cache3+2', '最大差4.76837158203125e−7、allclose True。'),
            execution('f06-19.9-mode-variation', 'callback_mode', '本次模式、反向圖、state hash 與 cache 小變式', 'train/eval與eval/no_grad差0；requires_grad true/true/false；state hash不變；split2/wrongpos/manual數值逐項核。')])
        page['sources'].extend([
            exec_source('callback_example', '本人新 cache 原例真 CPU'), exec_source('callback_mode', '本人新模式與 cache 變式真 CPU'),
            official('module_mode', 'Module.train/eval 原 API', 'torch/nn/modules/module.py', 'pytorch-v2.5.1-module.py', '2827–2864：train 設 mode 並遞迴子 module；eval=train(False)，僅特定 module 有 mode-dependent 計算。'),
            official('no_grad', 'torch.no_grad 原 API', 'torch/autograd/grad_mode.py', 'pytorch-v2.5.1-grad-mode.py', '21–85：停記反向梯度計算；factory requires_grad 與 forward AD 例外，本例未用。')])
        for aid, source_path, locator in [
            ('mode_selftrained', 'tiny_perceptron/selftrained/model.py', '25–185、190–280、295–418：config、MoE、LM、三 encoder、forward/generate 本人逐段原碼查模式分支與首步素材。'),
            ('mode_attention', 'tiny_perceptron/attention.py', '全檔1–74：SDPA dropout_p=0、cache offset/absolute mask；沒有 self.training-dependent branch。'),
            ('mode_layers', 'tiny_perceptron/modern.py', '8–96：RMSNorm/RoPE/Dense/MoE本人原碼，無 dropout 或 mode branch。'),
            ('mode_base', 'tiny_perceptron/model.py', '1–84：Block/TinyLM組合及tied共表，本人原碼查。')]:
            snapshot = CALLBACK / 'sources' / (aid + '.py')
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            assert not snapshot.exists()
            snapshot.write_bytes(Path(source_path).read_bytes())
            page['sources'].append({'id': aid, 'kind': 'repository_code', 'title': source_path + ' 本人本次原码快照', 'verified': True,
                                    'path': str(snapshot), 'sha256': sha(snapshot), 'version': '實際 inspected working-tree bytes / SHA256 ' + sha(snapshot), 'inspection_note': locator})
        page['claims'].append(
            {'id': 'c3', 'kind': 'software', 'statement': '本例eval切評估模式但各層沒有會因mode改變的計算；no_grad使此計算不建立求梯度圖；兩者都不更新權重或量速度。',
             'location': 'unit3 新模式／反向圖段', 'scope': '教材寬16/1層/random/CPU例，不能泛指所有模型或所有API工廠。', 'status': 'verified',
             'evidence': [ev('module_mode', '2827–2864', 'eval等同train(False)，mode影響限特定層。'), ev('no_grad', '21–85', '關閉反向梯度計算及明列例外。'), ev('mode_selftrained', '25–185/190–280/295–418', '實際LM/encoder/forward組合及mode-independent計算。'), ev('mode_attention', '48–73', 'SDPA dropout=0，manual/cache沒有mode分支。'), ev('mode_layers', '8–96', 'RMS/FFN/MoE不依training flag分岔。'), ev('mode_base', '1–84', 'base模型module組合。'), ev('callback_mode', 'stdout/train_eval_max_diff/requires_grad/parameters hash', '差0、gradflags true/true/false、before/after sameSHA及沒有dropout/BatchNorm。')],
             'artifact_ids': ['callback_mode'], 'verification': {'method': 'executed', 'expected': 'train/eval 相同，eval gradon 與 no_grad 分數相同但後者無反向圖；state不更新。', 'observed': '兩差0、requires_grad[true,true,false]、所有 module eval true；state hash前後同8fcd21ea49c2df8097d5e560cd11443b30fe21cb576a1b3ad7901116569c49c0。', 'details': 'CPU torch2.14.1+cpu；registered mode-dependent dropout/BN為空，原程式另排查依mode分支；沒有optimizer/backward或計時。'}})
        for key in ['factual_accuracy', 'numeric_verification', 'source_verification', 'limitations']:
            page['checks'][key]['claim_ids'].append('c3')
        page['checks']['factual_accuracy']['details'] += ' 新eval/no_grad区分，原layer碼與CPU吻合。'
        page['checks']['numeric_verification']['details'] += ' 本次原例及模式/3變式真正exit0。'
        page['checks']['source_verification']['details'] += ' 本回官方Module/no_grad與實際四原程式另定位。'
        page['checks']['limitations']['details'] += ' no_grad未實測記憶體收益，沒有GPU/backend選擇/速度新量測。'
        page_view(page, 'outputs/phase7-render/pages/p7_technical_e/19.9/expanded/20261007T074054552817Z-4dbce096',
                  ['1280-table-0-h0-0.png', '1280-text0-match0-0.png', '1280-text1-match0-0.png', '390-table-0-h0-0.png', '390-text0-match0-0.png', '390-text1-match0-0.png'],
                  '本人真 tools.view_image 看六張新PNG：桌面1280×800／手機390×844六row表完整可讀，新增eval/no_grad段完整，首步素材與packing/draft/練習段在必要定位畫面可讀。桌面code長行有水平scroll，未宣稱橫向逐行view或整頁所有位置檢視。')
    write_new(CALLBACK / 'pages' / (pid + '.json'), page)

for page, original in zip(report['pages'], old['pages']):
    if page['page_id'] not in ['18.4', '19.9']:
        assert page == original, page['page_id']
report['callback_scope'] = {'same_original_owner': OWNER, 'fresh_initial_owner_retained': True,
                            'not_new_blind_reader': True, 'changed_complete_pages': ['18.4', '19.9'],
                            'reused_identical_pages': 42, 'primary_pages_total': 44,
                            'original_report': {'path': str(OLD_REPORT), 'sha256': sha(OLD_REPORT)},
                            'source_comparison': {'path': str(CALLBACK / 'unchanged-pages.json'), 'sha256': sha(CALLBACK / 'unchanged-pages.json')},
                            'mandatory_context_reread': ['16.11', '16.12', '16.13'],
                            'scope_note': '只有兩改頁替換本回實讀trace與新增證據；另42頁整個page物件與原報告完全相同。原全部traces/證據/限制/失敗probes及更正保留。'}
report['review_limits'].extend([
    'Freeze06同原owner回查，非新盲讀者；兩頁本人完整逐段，另外42頁實際canonical+figure SHA一致才原page物件精確沿用；沒有重造舊checkpoint。',
    '本回v2.5.1官方API定位支持機制，installed PyTorch2.14.1+cpu行為由本人新CPU另證；未聲稱是相同文件版本。',
    '必要context16.11歷史L4 timing原始測量本回未新增獨立核查；僅成本算式與CPU compile eager機制示例，沒有用其數字支持主審兩页新主張。',
    '本回只觀看18.4四張新增說明定位、19.9六張表/模式/素材生成定位；未宣稱整頁所有code水平區域全view。完整PDF/全文仍只在本機cache，不作CI必需artifact。'])
metas = []
for p in sorted((BASE / 'executions').glob('f06-*.execution.json')):
    metas.append({'path': str(p), 'sha256': sha(p), 'execution': json.loads(p.read_text())})
write_new(CALLBACK / 'execution-inventory.json', metas)
report['callback_execution_inventory'] = {'path': str(CALLBACK / 'execution-inventory.json'), 'sha256': sha(CALLBACK / 'execution-inventory.json')}
report['callback_recovery_notes'] = ['依本人已存summary/current-only確認18.4 unit3；不讀raw state或他人judgments。', '原model snapshot第一次read誤檔名失敗，改本人既存正確selftrained-model-3f9.py後真讀；未當作教材問題。', '自己讀inventory時誤以dict作list[-1]發生KeyError，後直接核dict metadata；未更改任何舊記錄。', '第一次report組裝誤用Path.with_suffix而把18.4檔名截成18，讀execution metadata失敗；改成追加副檔名才成功，先寫出的42頁比較原樣核相同、不覆寫。']
write_new(NEW_REPORT, report)
print(json.dumps({'path': str(NEW_REPORT), 'sha256': sha(NEW_REPORT), 'verdict': report['verdict'], 'pages': len(report['pages']), 'changed': ['18.4', '19.9'], 'unchanged_exact_page_objects': 42, 'trace_sha256': sha(TRACE)}, ensure_ascii=False))
