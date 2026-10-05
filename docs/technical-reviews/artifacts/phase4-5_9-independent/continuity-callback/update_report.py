import hashlib
import json
from pathlib import Path

A = Path(__file__).resolve().parent
ROOT = A.parents[4]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
archive = json.loads((A / 'prior-pass-archive.json').read_text())
prior = ROOT / archive['history_path']
assert sha(prior) == archive['prior_sha256']
# Same-owner report mutation after independently completing the current review.
# The archived conclusion is not a source used to establish current correctness.
report = json.loads(prior.read_text())
for item in report['artifacts']:
    assert sha(ROOT / item['path']) == item['sha256'], item['path']
current = json.loads((A / 'current-input-provenance.json').read_text())
probe = json.loads((A / 'callback-probe-results.json').read_text())
context = json.loads((A / 'context-version-check.json').read_text())
original_json = json.loads((A / 'original-json-pointer-check.json').read_text())
assert current['source_sha256'] == probe['source_sha256']
assert current['fences'][0]['same_as_initial']
assert all(x['same_as_initial'] for x in current['files'] if x['path'].startswith('tiny_perceptron/'))
assert context['unchanged'] and original_json['unchanged_from_own_initial_input']
assert probe['parameters'] == 6104 and probe['raw_FP32_parameter_bytes'] == 24416
assert probe['float32_bits'] == 32 and probe['bits_per_byte'] == 8
assert probe['fence_executed'] and probe['optimizer_updates'] == probe['backward_calls'] == 0

commands = {
    'cwd': str(ROOT), 'shell': 'bash', 'login': False,
    'probe': {'command': "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 15 .venv/bin/python " + A.relative_to(ROOT).as_posix() + '/callback_probe.py', 'exit_code': 0},
    'source_retrieval': [],
    'input_extraction': 'Python section_facts.original_section on current course/chapters/05.md#5.9 raw bytes; original_section also selects current and own initial T.4; raw copies and hashes in current-input-provenance/context-version-check.',
    'AST_inspection': 'Python ast.parse on exact TinyLM/attention/modern and own fixed historical common/text sources; function ranges in reinspection.md.',
    'JSON_inspection': 'Python top-level key/type inventory then only original-json-pointer-check.json named pointers; no review/notes/HF author explanation values read.',
    'archive': 'Opaque read_bytes/write_bytes copy; exact path/hash/order in prior-pass-archive.json.'}
for script, status in [('fetch_dtype.py',0),('fetch_dtype_retry.py',1),('fetch_dtype_rst.py',1),('fetch_dtype_docs.py',1),('fetch_dtype_reference.py',1)]:
    commands['source_retrieval'].append({'command': '.venv/bin/python '+(A/script).relative_to(ROOT).as_posix(),
        'exit_code': status, 'scope': 'Only actually retrieved primary bytes file is evidence; failed dtype table attempts are retained, not verified sources.'})
(A / 'commands.json').write_text(json.dumps(commands, ensure_ascii=False, indent=2) + '\n')
reinspection = {'id': 'callback_reinspection', 'reviewer_task': report['reviewer_task'],
    'continuity': 'Same original technical owner; no author or third-reader answer consulted.',
    'reviewed_on': '2026-10-05', 'independent_current_verdict': 'pass',
    'source': report['source'], 'source_sha256': current['source_sha256'], 'figure_sha256': {},
    'change_scope': 'Only section second paragraph adds byte=8 bits, FP32=32 bits=4 bytes, and dtype wording.',
    'personal_inspection_record': (A/'reinspection.md').relative_to(ROOT).as_posix(),
    'personal_inspection_sha256': sha(A/'reinspection.md'),
    'probe_results_path': (A/'callback-probe-results.json').relative_to(ROOT).as_posix(),
    'probe_results_sha256': sha(A/'callback-probe-results.json'),
    'prior_pass': archive,
    'unchanged_dependencies': ['Original fence and model/attention/modern/data hashes', 'T.4 bytes', 'Original text_foundation JSON bytes'],
    'mechanism_scope': 'Gradient memory/no_grad/eval re-read and checked; no new norm/clipping claim in this section and no backward/update run.',
    'unresolved_dependencies': [], 'GPU_training_rerun': False,
    'network_limitation': 'Several external dtype-table path attempts returned HTTP503. Narrow format/type claims verified from original installed official API source, CPU bits/element-size, existing official element_size source, and successfully retrieved official CPython bytes documentation.'}
(A / 'reinspection.json').write_text(json.dumps(reinspection, ensure_ascii=False, indent=2) + '\n')

ids = {}
env = {k:str(v) for k,v in probe['environment'].items()}
for path in sorted(A.rglob('*')):
    if not path.is_file() or path.name.startswith(('checker','report-update.')) or path.name == 'manifest.json':
        continue
    name = path.relative_to(A).as_posix()
    identifier = 'callback_reinspection' if name == 'reinspection.json' else 'callback_' + name.replace('/','_').replace('.','_').replace('-','_')
    ids[name] = identifier
    item = {'id':identifier, 'kind':'code' if path.suffix in {'.py','.pyi','.h'} else 'source_snapshot',
        'path':path.relative_to(ROOT).as_posix(), 'sha256':sha(path),
        'description':'本原技術owner continuity callback 的原始輸入、親讀定位、命令、官方源或保留的真取得結果：'+name}
    if name == 'callback-probe-results.json':
        item.update(kind='execution',command=commands['probe']['command'],result='exit0；current frozen fence真正執行；float32 bits32、bytes4、dtype型別，6104/24416；no_grad、零backward/update。',environment=env)
    if name in {'reinspection.md','reinspection.json'}:
        item['kind']='derivation'
    report['artifacts'].append(item)

report['sources'] += [
    {'id':'callback_python_bytes','kind':'official_source','title':'Official CPython bytes primary documentation',
     'url':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst','version':'CPython v3.13.5',
     'authority_reason':'Official python/cpython primary documentation, actually retrieved and personally read.',
     'accessed_on':'2026-10-05','verified':True,'checked_original':True,
     'inspection_note':'Personally read lines2721–2775, bytes values0<=x<256 and two hexadecimal digits/byte; current saved file hash and URL receipt. 256=2^8 makes eight-bit unit explicit.'},
    {'id':'callback_dtype_source','kind':'official_source','title':'Original installed official PyTorch dtype/finfo API declarations and scalar type mapping',
     'url':'https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070/torch',
     'version':'Installed official PyTorch2.14.1+cpu; reported git 5c4886908584029761b579af026dcfb627c84070',
     'authority_reason':'PyTorch official installed wheel original generated API declarations and bundled scalar-type header, not repository author commentary or search summary.',
     'accessed_on':'2026-10-05','verified':True,'checked_original':True,
     'inspection_note':'Personally read original installed _C/__init__.pyi198–235 (dtype,float32:dtype,finfo.bits) and bundled header ScalarType.h64–77 (Byte/Float mapping); copied full original files byte-identically with hashes/original paths in installed-primary-source-receipt. URL identifies official maintainer/source version; no online dtype table retrieved/read is claimed.'},
    {'id':'callback_cpu','kind':'execution','title':'Current original fence and byte/FP32/dtype bounded CPU callback check',
     'verified':True,'artifact_id':ids['callback-probe-results.json']}]
report['claims'].append({'id':'c8','kind':'numeric',
    'statement':'一byte有8個bit；FP32浮點格式32 bits故每參數4 bytes；這種每元素數值格式在PyTorch中以dtype描述。',
    'location':'5.9 current 第二段新增byte/bit/FP32/dtype定義',
    'scope':'標準byte與本課PyTorch float32參數的數值格式；不把4 bytes當完整checkpoint/總RAM，不論述其他格式。',
    'status':'verified','evidence':[
        {'source_id':'callback_python_bytes','locator':'stdtypes.rst2721–2775 Bytes Objects','supports':'單byte值範圍0..255，共256=2^8；two hex digits/byte的原主源規約。'},
        {'source_id':'callback_dtype_source','locator':'Installed _C/__init__.pyi198–235 and bundled ScalarType.h64–77','supports':'dtype API型別、float32:dtype、finfo.bits與Byte/Float scalar映射。'},
        {'source_id':'tensor_bytes','locator':'Official Tensor.element_size1728–1742','supports':'element_size以bytes計量每元素，不是模型總記憶體。'},
        {'source_id':'callback_cpu','locator':'callback-probe-results.json float32_bits/element_size/dtype/unit fields','supports':'float32為torch.dtype、bits32、每element4bytes、32/8=4；原current模型6104/24416。'}],
    'artifact_ids':[ids['callback-probe-results.json'],ids['python-stdtypes.rst'],ids['installed-primary-source/_C/__init__.pyi'],ids['reinspection.json']],
    'verification':{'method':'executed','expected':'byte256=2^8可能值；float32 dtype bits32/size4；6104*4=24416。',
        'observed':'全部assert精確符合；current fence exit0，6104/24416，no_grad logits不求導。',
        'details':'CPU1thread/.venv2.14.1+cpu、15秒有界；原主源親讀與原package檔案副本hash精確相同；無訓練/GPU。',
        'tolerance':'units/bitwidth/element_size/參數/bytes均整數精確相等；前向秒數無固定標準答案。'}})
report['source_sha256']=current['source_sha256']
report['verdict']='pass'
report['figure_sha256']={}
report['continuity_callback']=reinspection
report['summary']='同原技術owner親核完整current5.9；新增byte/bit/FP32/dtype由原主源與真短CPU確認；原fence/實作/T.4/原JSON版本不變，原proofs/issues保留，無未決依賴。'
for name in ['factual_accuracy','numeric_verification','source_verification','limitations']:
    report['checks'][name]['claim_ids'].append('c8')
    report['checks'][name]['details'] += ' Same-owner callback新增byte/bit/FP32/dtype已核實，詳continuity_callback與c8。'
# Correct one initially imprecise range using the personally re-read AST locations.
for c in report['claims']:
    for evidence in c['evidence']:
        if evidence['source_id']=='historical_common':
            evidence['locator']=evidence['locator'].replace('new_lm43–45','new_lm45–47').replace('fit_lm125–201','fit_lm122–202')
path=ROOT/'docs/technical-reviews/5.9.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert json.loads(path.read_text())['source_sha256'] == sha(A/'section-current.md')
print(json.dumps({'report_path':path.relative_to(ROOT).as_posix(),'report_sha256':sha(path),
    'source_sha256':report['source_sha256'],'figure_sha256':{},'verdict':'pass',
    'prior_pass_history_path':archive['history_path'],'prior_pass_sha256':archive['prior_sha256'],
    'reinspection_artifact_id':'callback_reinspection','reinspection_path':(A/'reinspection.json').relative_to(ROOT).as_posix(),
    'reinspection_sha256':sha(A/'reinspection.json'),'unresolved_dependencies':[]},ensure_ascii=False,indent=2))
