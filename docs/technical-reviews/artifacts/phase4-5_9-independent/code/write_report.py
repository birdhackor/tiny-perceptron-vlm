import hashlib
import json
from pathlib import Path

A = Path(__file__).resolve().parents[1]
ROOT = A.parents[3]
P = A.relative_to(ROOT).as_posix()
def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path):
    return path.relative_to(ROOT).as_posix()

extraction = json.loads((A / 'original/extraction.json').read_text())
checks = json.loads((A / 'bounded-check-results.json').read_text())
env = {k: str(v) for k, v in checks['environment'].items()}
original_execution = json.loads((A / 'original/execution.json').read_text())
cmds = {
    'original': '.venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.9 --output outputs/reviewer-tools/phase4-5_9-independent-original --execute --timeout 30',
    'bounded': f"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 30 .venv/bin/python {P}/code/check_costs.py > {P}/bounded-check.stdout.json 2> {P}/bounded-check.stderr.txt",
    'six': f"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. timeout 15 .venv/bin/python {P}/code/six-position-original-fence.py > {P}/six-position.stdout.txt 2> {P}/six-position.stderr.txt",
    'fetch': f'.venv/bin/python {P}/code/fetch_sources.py > {P}/source-fetch.stdout.json 2> {P}/source-fetch.stderr.txt',
    'autograd_fetch': ".venv/bin/python - <<'PY'\n" + '\n'.join((A / 'code/fetch_autograd_original_stdin.py').read_text().splitlines()[1:]) + '\nPY',
}
commands = {'cwd': str(ROOT), 'shell': 'bash', 'login': False,
    'commands': [{'id': key, 'command': value, 'exit_code': 0} for key,value in cmds.items()],
    'original_execution_facts': original_execution,
    'historical_source_retrieval': 'historical-retrieval.json records exact git show argv, source revision, and hash verification',
    'scope': 'Original and changed CPU fences, bounded attention/count inspections, external official sources only; no training/GPU/model or data downloads/uploads.'}
(A / 'commands.json').write_text(json.dumps(commands, ensure_ascii=False, indent=2) + '\n')

artifacts = []
ids = {}
def artifact(name, kind='source_snapshot', description=None, execution=None):
    path = A / name
    identifier = 'a_' + name.replace('/', '_').replace('.', '_').replace('-', '_')
    ids[name] = identifier
    item = {'id': identifier, 'kind': kind, 'path': rel(path), 'sha256': h(path),
            'description': description or f'本輪原始輸入或來源快照：{name}；由 inspection.md 指明親讀範圍。'}
    if execution:
        item.update(command=execution[0], result=execution[1], environment=env)
    artifacts.append(item)

artifact('original/execution.json', 'execution', '原始 fence 真正 CPU 執行的 exit、裝置與輸出指紋。',
         (cmds['original'], 'exit=0；原始 fence 6104參數、24416 bytes；10次平均前向0.00043802549917018043秒，非固定標準答案。'))
artifact('bounded-check-results.json', 'execution', '有界CPU形狀、參數、求導模式、重複計時與原JSON分母核對。',
         (cmds['bounded'], 'exit=0；全部assert通過；3/6位置參數不變、attention權重3x3/6x6；重建有效目標342462，未訓練。'))
artifact('six-position.stdout.txt', 'execution', '只把原 fence x 改成六位置、同樣暖身1次計時10次的真輸出。',
         (cmds['six'], 'exit=0；6104參數、24416 bytes；10次平均前向0.0003278203002992086秒，不支持六位置更快的結論。'))
for path in sorted(A.rglob('*')):
    if not path.is_file():
        continue
    name = path.relative_to(A).as_posix()
    if name in ids or name.startswith(('checker', 'report-write.')) or name == 'manifest.json':
        continue
    kind = 'code' if path.suffix == '.py' else 'derivation' if name == 'inspection.md' else 'source_snapshot'
    artifact(name, kind)

sources = []
def repo(identifier, name, title, version, note):
    path = A / name
    sources.append({'id': identifier, 'kind': 'repository_code', 'title': title, 'verified': True,
                    'path': rel(path), 'sha256': h(path), 'version': version, 'inspection_note': note})

repo('model', 'inputs/tiny_perceptron/model.py', 'TinyLM/ModelConfig 原始實作', '本輪目前原稿 helper 實際載入版本；SHA鎖定',
     '親讀ModelConfig lines14–29、Block 32–50、TinyLM 53–89；預設字表264/位置128、width8、參數numel與forward回傳logits。')
repo('attention', 'inputs/tiny_perceptron/attention.py', 'manual_attention/CausalAttention 原始實作', '本輪CPU helper實際載入SHA版本',
     '親讀lines21–28 q@k^T與weights、31–74投影與B,H,T形狀；實際hook看真weights，未改正文原碼。')
repo('modern', 'inputs/tiny_perceptron/modern.py', 'DenseFFN 原始實作', '本輪CPU helper實際載入SHA版本',
     '親讀DenseFFN lines38–55，hidden=width*4與up/down含bias；用於逐表參數推導。')
rev = 'b52935d99f58b694cd932ac158c10c4d8d92d4c2'
for sid,name,note in [
    ('historical_common','common','親讀new_lm lines43–45、split_records 48–68、text_examples77–99、fit_lm125–201；計時在_synchronize後開/關，有效目標數排除IGNORE。'),
    ('historical_text','text','親讀run_text_foundation262–314與_resume_probe；width64/layers2主600步、四組150步與完整組包括resume/causal probes。'),
    ('historical_run','run','親讀execute入口、torch threads=2與lines71–107/結果timing_scope；整組函數與同步範圍，不含雲啟動及上傳。')]:
    repo(sid, f'historical/scripts/course_experiments/{name}.py', f'歷史{ name }.py 原始實作', rev+'；hash與原JSON code_sha256精確相符',note)
repo('historical_generator', 'historical/scripts/prepare_data.py', '歷史toy-text資料產生器', rev+'；git版本鎖定',
     '親讀generate_records toy-text分支；本檔不在原JSON code_sha256 map內，故實際重建三切分JSONL並逐hash與原JSON精確核對。')
repo('data', 'historical/tiny_perceptron/data.py', '歷史ByteTokenizer/shifted/pad_batch', rev+'；hash與原JSON code_sha256精確相符',
     '親讀IGNORE、ByteTokenizer、shifted、pad_batch；byte+EOS標籤、padding排除；現行與此历史SHA一致，只重播計數。')
repo('original_result','inputs/docs/course-experiments/results/text_foundation.json','原始text_foundation結果JSON',rev+'；原始NVIDIA L4執行，非本輪新訓練',
     '親讀top-level device/gpu/torch/python/elapsed/timing_scope/code_sha256，results.data/training/scaling/resume及artifacts；用自寫CPU程式核對所有本節數值與分母。JSON來源非程式，使用repository_code型作版本鎖定本地原始實驗文件。')

fetch = json.loads((A / 'source-fetch-receipt.json').read_text())
official_ids = {
 'torch-grad-mode.py': ('no_grad', 'PyTorch no_grad 原始API'),
 'torch-module.py': ('module_eval', 'PyTorch Module.eval/train 原始API'),
 'torch-adam.py': ('adam', 'PyTorch Adam optimizer state 原始實作'),
 'torch-tensor-docs.py': ('tensor_bytes', 'PyTorch Tensor element_size/grad/nbytes 原始文件'),
 'torch-cuda.py': ('cuda_sync', 'PyTorch CUDA synchronize 原始API'),
 'torch-benchmark-timer.py': ('benchmark', 'PyTorch benchmark Timer 原始說明'),
 'python-time.rst': ('python_time', 'CPython perf_counter 原始文件'),
}
notes = {
 'no_grad':'親讀lines22–86；reverse-mode no_grad與factory/forward AD例外，TinyLM通常前向在範圍內。',
 'module_eval':'親讀lines2894–2932；eval=train(False)，遞迴設training旗標，僅特定module受影響，不關閉梯度。',
 'adam':'親讀lines148–192；grad與exp_avg/exp_avg_sq lazy tensors，支持額外訓練狀態，不給固定總記憶體倍率。',
 'tensor_bytes':'親讀lines1728–1742 element_size bytes、6555–6563 grad、6753–6768 nbytes=numel*element_size。',
 'cuda_sync':'親讀lines1271–1281；等待裝置所有streams/kernels完成；未實跑CUDA。',
 'benchmark':'親讀lines67–136；warmup、thread controls、async synchronize、replicates/noise。',
 'python_time':'親讀lines321–347；高解析度單調perf_counter，fractional seconds、差值量間隔。',
}
for item in fetch:
    if item['status'] != 'retrieved':
        continue
    sid,title = official_ids[item['path']]
    sources.append({'id': sid, 'kind':'official_source', 'title':title,
        'url':item['url'], 'version':'CPython v3.13.5' if sid=='python_time' else 'PyTorch git '+env['torch_git_version']+' (installed 2.14.1+cpu matches)',
        'authority_reason':'官方維護者的原始repository固定版本，直接查讀API/文件內容。',
        'accessed_on':'2026-10-05','verified':True,'checked_original':True,
        'inspection_note':notes[sid]+' 快照hash見本輪artifacts/receipt。'})
ar=json.loads((A/'autograd-fetch-receipt.json').read_text())
sources.append({'id':'autograd','kind':'official_source','title':'PyTorch Autograd mechanics 原始文件',
    'url':ar['url'],'version':'PyTorch git '+env['torch_git_version'],
    'authority_reason':'官方pytorch/pytorch固定版本的原始文件。','accessed_on':'2026-10-05',
    'verified':True,'checked_original':True,'inspection_note':'親讀lines12–54：forward建立求導圖、為backward保存必要中間tensor；原rst404後查同commit的md成功。'})
sources += [
 {'id':'cpu_exec','kind':'execution','title':'自寫有界CPU與歷史JSON核對','verified':True,'artifact_id':ids['bounded-check-results.json']},
 {'id':'original_exec','kind':'execution','title':'原始fence CPU執行','verified':True,'artifact_id':ids['original/execution.json']},
 {'id':'six_exec','kind':'execution','title':'原始fence只改六位置的CPU執行','verified':True,'artifact_id':ids['six-position.stdout.txt']},
 {'id':'parameter_derivation','kind':'derivation','title':'逐參數表計數與FP32 bytes手算','verified':True,
  'details':'inspection.md逐表：2112+1024+256+552+48+2112=6104；6104*4=24416；歷史141568*4=566272。'}]

def e(sid,locator,supports):
    return {'source_id':sid,'locator':locator,'supports':supports}
def claim(sid,kind,statement,location,scope,evidence,anames,verification=None):
    item={'id':sid,'kind':kind,'statement':statement,'location':location,'scope':scope,
          'status':'verified','evidence':evidence,'artifact_ids':[ids[name] for name in anames]}
    if verification:
        item['verification']=verification
    return item
claims=[]
claims.append(claim('c1','numeric','width8預設模型有6104參數；FP32純參數數據24416 bytes，格數乘4。','5.9 前兩段/原始fence/參數數量段',
    '僅該TinyLM預設vocab264/max_length128/layers1/heads1且未綁權重的FP32參數data；不宣稱存檔或總RAM大小。',
    [e('model','ModelConfig14–29; TinyLM53–89','模型預設配置與sum(numel)契約。'),e('modern','DenseFFN38–55','4倍hidden含bias的參數表。'),e('tensor_bytes','element_size1728–1742; nbytes6753–6768','每元素bytes與numel*element_size。'),e('parameter_derivation','inspection.md Counts and scopes','逐表總6104與6104*4。'),e('cpu_exec','checks.parameter_storage; parameters','真dtype float32/element_size4與總數。')],
    ['bounded-check-results.json','inspection.md','original/execution.json'],
    {'method':'executed','expected':'6104個FP32參數、24416 raw bytes。','observed':'各參數numel總6104、element_size4、raw bytes24416。','details':'原fence加逐表獨立核對；未以checkpoint檔案大小當參數bytes。','tolerance':'整數精確相等。'}))
claims.append(claim('c2','concept','訓練除參數外還有梯度、Adam歷史與求導所需中間特徵。','5.9 第二段',
    '一般反向求導/Adam資源項目；沒有承諾固定倍率，no_grad只抑制本次反向求導記錄。',
    [e('tensor_bytes','Tensor.grad6555–6563','backward產生/累積grad tensor。'),e('adam','Adam._init_group148–192','參數grad與exp_avg/exp_avg_sq額外狀態。'),e('autograd','Autograd mechanics12–54','計算圖與backward必要中間tensor的保存。')],['sources/torch-adam.py','sources/torch-autograd.md','sources/torch-tensor-docs.py']))
claims.append(claim('c3','software','原fence以一筆3位置、width8模型做CPU前向，eval設定評估模式，no_grad關閉本次求導記錄，暖身1次後計10次秒數除10；不更新參數。','5.9 Python fence與緊接API解釋段',
    '逐项cover TinyLM(ModelConfig(width=8))/tensor [[1,2,3]]、eval/no_grad/model(x)/description及perf_counter；普通reverse-mode前向，不涵蓋factory/forward-AD例外。',
    [e('model','ModelConfig; TinyLM.forward68–86; description88–89','输入B=1,T=3与输出logits的實作。'),e('module_eval','Module.train/eval2894–2932','eval=train(False)，不會自行關梯度。'),e('no_grad','no_grad22–86','本次運算結果不追反向求導，離開恢復。'),e('python_time','time.rst321–347','fractional seconds差值時計間隔。'),e('original_exec','original/execution.json; stdout; fence-1.py','原碼實際成功执行1暖身+10计时。'),e('cpu_exec','checks.eval_and_no_grad','eval logits requires_grad=true、no_grad=false；無grad/更新。')],
    ['original/execution.json','original/stdout.txt','original/fence-1.py','bounded-check-results.json'],
    {'method':'executed','expected':'原fence成功；參數/bytes正确、forward秒數正且非固定標準答案；no_grad結果不求導。','observed':'exit0；6104/24416；10次平均0.00043802549917018043秒；eval本身保留求導，no_grad關閉，未backward/update。','details':'Python3.13.5/PyTorch2.14.1+cpu、離線CPU、原helper guard無攔截；計時值只描述這次。'}))
claims.append(claim('c4','numeric','只把x改成六位置，參數數量與bytes不變，注意力表由3×3變6×6；運算增加不保證本次總耗時必定上升。','5.9 三位置變六位置段與練習',
    '固定同一default manual attention模型、B=H=1；矩陣元素9變36，有限時長計時不建立長序列快慢定律。',
    [e('attention','manual_attention21–28; forward48–74','q/k序列軸產生T×T權重而非增加權重參數。'),e('six_exec','six-position-original-fence.py/stdout','只改x的原fence實測。'),e('cpu_exec','checks.six_position_variation','hook真weights形狀、相同參數與重複測量波動。'),e('benchmark','Timer67–89','replicates量noise，單次數值不能下結論。')],
    ['bounded-check-results.json','six-position.stdout.txt','code/six-position-original-fence.py'],
    {'method':'executed','expected':'兩長度6104/24416；weights[1,1,3,3]/[1,1,6,6]；不要求秒數排序。','observed':'精確符合形狀與整數；六位置原10次平均0.0003278203002992086秒，較本次三位置低；5×100另測有波動。','details':'原x唯一變化之10次計時另存；有界hook與五組100次重複支持量測流程/波動，不把短测写成速度优势。','tolerance':'shape/參數/bytes精確相等；wall seconds不設跨執行容忍/排序。'}))
claims.append(claim('c5','concept','正式前向速度比較須控制設置、增加重複並報波動；GPU非同步工作要在計時邊界等待，前向秒數與整組實驗秒數不可混比。','5.9 正式速度比較及GPU段、補充最後段',
    '計時方法與比較範圍；本輪CPU檢查不實測GPU、不建立硬體加速倍數。',
    [e('benchmark','Timer67–136','warmup、threads、async synchronization與replicates/noise。'),e('cuda_sync','synchronize1271–1281','等待裝置所有kernels/streams完成。'),e('python_time','perf_counter321–347','elapsed計時包含實際牆鐘經過時間。'),e('historical_common','fit_lm125–168','主訓練區間有同步邊界。'),e('historical_run','execute71–107','整組額外工作與主訓練範圍不同。')],['sources/torch-benchmark-timer.py','sources/torch-cuda.py','inspection.md']))
claims.append(claim('c6','empirical','T.4原L4模型有141568個FP32參數/566272 raw bytes；600次更新每次16篇，342462有效目標是9篇材料重複曝光。','5.9 補充第一段',
    '只核對固定text_foundation原始JSON與其精確源版本；重播抽樣/標籤計數不重訓，不稱342462不同文字/篇。',
    [e('original_result','results.data; results.training; revision/gpu/torch_version','9篇/600步/141568參數/342462目標記錄。'),e('historical_common','new_lm43–45; text_examples77–99; fit_lm125–201','max_length128、batch_size16、choices有放回、排IGNORE的有效target語義。'),e('historical_text','run_text_foundation262–270','width64/layers2与主600步。'),e('historical_generator','generate_records toy-text16–24','12短文原來源與split材料。'),e('data','ByteTokenizer/shifted/pad_batch13–99','每byte+EOS計分，padding不計分。'),e('cpu_exec','checks.original_json_audit','三split hash匹配、600次16抽樣重建342462與9,600曝光；count/bytes141568/566272。')],
    ['bounded-check-results.json','historical-retrieval.json','inputs/docs/course-experiments/results/text_foundation.json'],
    {'method':'executed','expected':'9 train records、600updates、每batch16、有效目標342462；141568 FP32参数及566272 bytes。','observed':'三split hash及records_sha256精確一致；9 examples target lengths34–37；600×16=9600曝光，重播total342462；count141568、bytes566272。','details':'來源revision b52935d...、NVIDIA L4、torch2.14.1+cu126/Python3.13.3；本輪CPU版本僅復核抽樣計數與相同模型契約，未重訓/載權重。',
     'denominators':{'unique_training_documents':9,'training_examples':9,'optimizer_updates':600,'examples_per_update':16,'record_exposures':9600,'effective_targets_including_repeated_bytes_and_EOS':342462,'target_count_excludes':'IGNORE=-100 padding','parameters':141568,'bytes_per_FP32_parameter':4}}))
claims.append(claim('c7','empirical','原NVIDIA L4記錄主600次更新區間約6.1885秒，整組實驗約16.0354秒，包含記錄/週期存檔及額外比較評估保存，均不含雲啟動、映像建置或上傳備份。','5.9 補充第二段',
    '原始單次實驗記錄，不能變成純GPU kernel訓練速度或與上方CPU前向直接比較；讀原版計時位置確定邊界。',
    [e('original_result','training.seconds=6.1885463640000005; elapsed_seconds=16.035376792999998; timing_scope; gpu','原秒數/硬體/排除範圍。'),e('historical_common','fit_lm125–168','同步包住600步、含batch/log/periodic saves，initial/final NLL及final save在區間外。'),e('historical_text','run_text_foundation262–314; _resume_probe','整組主訓練前後評估、四組比較、local saves、因果/續訓probe；正文「還包括」非窮舉。'),e('historical_run','execute71–107; timing_scope129','整函數至最後同步、排除startup/build/uploads；final JSON serialization在elapsed後。'),e('cpu_exec','checks.original_json_audit.*seconds*','從原JSON計算四位小數round，非重新量GPU。')],
    ['bounded-check-results.json','historical-retrieval.json','inspection.md','inputs/docs/course-experiments/results/text_foundation.json'],
    {'method':'executed','expected':'round(training.seconds,4)=6.1885；round(elapsed_seconds,4)=16.0354；裝置NVIDIA L4，範圍分開。','observed':'精確符合；歷史common/text/run SHA吻合原JSON，邊界含排除/額外項與正文一致。','details':'只核對原JSON數值與原來源位置；沒有GPU/完整訓練重跑，不將原秒數稱本輪重現。',
     'denominators':{'timed_main_optimizer_updates':600,'main_examples_per_update':16,'main_unique_documents':9,'main_effective_targets':342462,'whole_experiment_runs':1,'additional_scaling_model_settings':4,'additional_scaling_updates_each':150,'GPU':'NVIDIA L4','original_torch':'2.14.1+cu126','original_python':'3.13.3','timing_units':'wall seconds'}}))

report={'schema_version':1,'review_stage':'technical','lesson_id':'5.9',
    'source':'course/chapters/05.md#5.9','source_sha256':extraction['source_sha256'],
    'figure_sha256':{},'reviewer_task':'/root/phase4_factual_coordinator/factual_5_9',
    'reviewer_context':'fresh','author_tasks':[],'verdict':'pass',
    'reviewed_on':'2026-10-05','reading_scope':'完整目前5.9、T.4、指定方法/helper/schema/protocol、涉及之目前與歷史實作與原JSON、固定版本官方來源；無舊review。5.9不是章首。',
    'summary':'7項實質主張獨立核對通過；原CPU fence與六位置變化、重複短測、歷史hash與有效目標重建全部通過。未GPU重訓、未讀舊review，沒有圖需render。',
    'sources':sources,'artifacts':artifacts,'claims':claims,'issues':[],
    'checks':{
        'factual_accuracy':{'status':'pass','details':'FP32/raw storage與training state、eval/no_grad、計時流程及不可混比範圍由原始API/程式支持。','claim_ids':['c1','c2','c3','c4','c5','c6','c7']},
        'numeric_verification':{'status':'pass','details':'6104/24416、141568/566272、3x3/6x6、9篇/600×16=9600曝光/342462target及秒數round皆實際CPU核對。','claim_ids':['c1','c4','c6','c7']},
        'figure_consistency':{'status':'not_applicable','details':'原始bytes/helper與親讀確認本節零圖/零SVG引用；沒有render或view可聲稱。形狀與計時範圍由文字/程式足夠表達。','claim_ids':[]},
        'source_verification':{'status':'pass','details':'直接取官方PyTorch exact installed git與CPython v3.13.5並親读定位；原b52935d版本historical SHA逐項匹配，未用候選庫摘要作證。','claim_ids':['c1','c2','c3','c4','c5','c6','c7']},
        'limitations':{'status':'pass','details':'本機秒數不可固定；一暖身十前向為流程示例；額外5×100仍非性能優劣定論；GPU同步只原碼查證；原JSON是舊實測核對不是重訓；未存檔大小/總RAM測量、未下載或載weights。','claim_ids':['c1','c3','c4','c5','c6','c7']}}}
(ROOT/'docs/technical-reviews/5.9.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':'docs/technical-reviews/5.9.json','verdict':'pass','claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'source_sha256':extraction['source_sha256']},ensure_ascii=False))
