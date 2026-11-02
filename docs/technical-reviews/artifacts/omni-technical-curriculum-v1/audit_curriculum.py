"""curriculum 數量、歷史收據、來源定位與微小 CPU 機制核對；不訓練模型。"""

from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import platform
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-curriculum-v1')
sys.path.insert(0, str(ROOT))
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.lesson_products.data import family, render, split_of
from tiny_perceptron.lesson_products.inference import rule_depth

torch.set_num_threads(1)
torch.manual_seed(42)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


READFILES = {}


def load(path):
    READFILES[path]=sha(ROOT/path)
    return json.loads((ROOT / path).read_text())


records = []


def record(name, expected, observed, scope):
    passed = expected == observed
    records.append(dict(name=name, expected=expected, observed=observed, passed=passed, scope=scope))
    assert passed, name


freeze_path = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
freeze = load(freeze_path)
page = next(p for p in freeze['pages'] if p['page_id'] == 'curriculum')
record('frozen whole page', page['source_sha256'], sha(ROOT / page['snapshot']), '完整指定 snapshot；沒有正規化 bytes。')
record('current whole page', page['source_sha256'], sha(ROOT / page['source']), '僅 curriculum 本頁。')
record('unique figure', page['figures_sha256'], {p: sha(ROOT / p) for p in page['figures_sha256']}, '圖已另以 Inkscape render 後實看。')
text = (ROOT / page['snapshot']).read_text()
record('curriculum executable code fences', 0, text.count('```'), '本頁無 executable code fence；圖含 API 名稱，下面另核其機制。')

chapter_paths = sorted((ROOT / 'course/chapters').glob('*.md'))
headings = lambda p: re.findall(r'^## ([A-Z\d]+\.\d+) (.+)$', p.read_text(), re.M)
record('numeric chapters', 20, sum(p.stem.isdigit() for p in chapter_paths), '章數，不是 canonical 網站頁數。')
record('chapter numbered lessons', 331, sum(len(headings(p)) for p in chapter_paths), '只核數量，不代表重審這331節。')
front = ['README.md', 'first-steps.md', 'training.md', 'glossary.md']
record('all numbered sections', 357, 331 + sum(len(headings(ROOT / 'course' / p)) for p in front), '章節331＋前置26，不與網站頁數混用。')
baseline = load('docs/course-review-20261027-tokenizer/freeze-03.json')
added = sorted({p['page_id'] for p in freeze['pages']} - {p['page_id'] for p in baseline['pages']})
expected_new = sorted(['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16'])
record('18 added sections', expected_new, added, '與固定前輪 source freeze 的 ID 差集；不沿用其 reviewer 答案。')

route_ids=set('R.2 R.3 W.1 6.2 6.5 7.11 7.17 7.18 7.19 7.20 7.21 7.22 7.23 7.24 7.25 7.26 8.14 8.15 8.16 9.11 9.12 9.13 9.14 9.15 9.16 9.17 9.18 10.12 11.17 11.18 12.15 12.16 12.17 12.18 13.10 13.11 14.7 14.8 14.9 14.10 14.11 14.12 14.13 14.14 14.15 14.16 14.17 14.18 14.19 16.12 16.13 16.14 16.15 16.16 17.16 17.17 18.1 18.2 18.3 18.4 18.10 19.12 19.13 19.14 19.15 19.16 19.17 19.18 19.19 19.20 19.21 19.22 19.23 19.24 20.2 20.13 20.14 20.15 20.16 A.8 A.9 A.10 B.1 B.2 B.3 B.4 C.1 C.2 C.3 C.4 C.5 C.6 C.7'.split())
pins=[]
for path in chapter_paths+[ROOT/'course/README.md',ROOT/'course/first-steps.md']:
    raw=path.read_text();matches=list(re.finditer(r'^## ([A-Z\d]+\.\d+) (.+)$',raw,re.M))
    for i,m in enumerate(matches):
        if m[1] not in route_ids:continue
        section=raw[m.start():matches[i+1].start() if i+1<len(matches) else len(raw)]
        intro=section.split('\n\n')[1] if '\n\n' in section else section
        pins.append(dict(lesson_id=m[1],path=path.relative_to(ROOT).as_posix(),file_sha256=sha(path),
                         selector='heading inclusive through next ## exclusive',section_sha256=hashlib.sha256(section.encode()).hexdigest(),
                         heading=m[0],first_paragraph_excerpt=intro[:180],
                         inspection_scope='導航語義只核標題與開頭例子，不宣稱重新技審此頁全部主張；R.2/R.3及必要BPB／操作與產品段落另已實讀。'))
record('currently linked route slices', sorted(route_ids), sorted(p['lesson_id'] for p in pins), '逐一綁精確slice與本版語義；只驗curriculum指路，不以此替整本書的新審閱。')
pin_payload=dict(source_freeze=dict(path=freeze_path,sha256=sha(ROOT/freeze_path)),
                 assigned_page=dict(path=page['source'],sha256=page['source_sha256'],selector='whole-file',snapshot=page['snapshot']),
                 scope='本人完整讀curriculum；以下是明連內容的定位核對。作者PLAN/OUTLINE及其他review答案未讀。',slices=pins)
(OUT/'source-slice-pins.json').write_text(json.dumps(pin_payload,ensure_ascii=False,indent=2)+'\n')

results = load('docs/selftrained/results/v2-final-public-results.json')
selection = load('docs/selftrained/results/v2-final-training-source-selection.json')
record('v2 data sizes', [28876,2435,3734], [results['data_scope'][k] for k in ['train_rows','validation_rows','test_rows']], '固定資料包的題數，不是獨立人數／錄音數。')
stage_rows=[]
for architecture in ['moe','dense']:
    for stage_name in ['pretrain','sft','vision','ocr','audio','joint']:
        raw_stage=load(f'docs/selftrained/results/training-raw/{architecture}-{stage_name}/raw/train-receipt.json')
        stage_rows.append([architecture,stage_name,raw_stage['completed_requested_steps'],raw_stage['origin']['kind']])
record('twelve historical from-random completed stages', True, len(stage_rows)==12 and all(r[2:] == [True,'all-neural-weights-random'] for r in stage_rows), '讀12筆原始收據，確認各階段完成與自家隨機起點；不代表新增稿已正式訓練。')
capacity = {}
historical = {}
for arch, steps, parameters, stage in [('moe',4000,5447107,'moe-native'),('dense',10000,2288067,'dense-weighted')]:
    config = SelftrainedConfig(**results['architectures'][arch]['capacity']['config'])
    model = LimitedAssistant(config)
    capacity[arch] = dict(parameters=sum(p.numel() for p in model.parameters()), top_k=config.top_k,
                          active_ffns=config.top_k if arch == 'moe' else 1, device=str(next(model.parameters()).device))
    record(f'{arch} current parameter count', parameters, capacity[arch]['parameters'], '只建立隨機CPU權重、直接相加numel；未載公開權重或做訓練。')
    raw = load(f'docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json')
    record(f'{arch} raw training count', [steps,28876,2435,parameters,True,False],
           [raw['steps'],raw['train_records'],raw['validation_records'],raw['total_parameters'],raw['completed_requested_steps'],raw['test_used_for_selection']], '讀歷史原始訓練收據；本次沒有重訓。')
    selected=selection[arch+'_joint']
    completed=selected['completed_requested_steps'] if arch == 'moe' else selected['completed_steps']
    record(f'{arch} completed versus selected', [steps,1000], [completed,selected['selected_step']], '最終段完成步數與 validation-selected checkpoint 分開。')
    metrics = load(f'docs/selftrained/results/public-raw/{arch}/test/metrics.json')
    receipt = load(f'docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json')
    record(f'{arch} final test receipt', [3734,3734,True,False,0,None],
           [metrics['count'],receipt['completed_count'],metrics['evaluation_complete'],metrics['teacher_forcing_used_for_generation'],metrics['resumed_completed_count'],receipt['resumed_from']], '同一固定考卷、單次已完成final evaluation；不外推一般能力。')
    t = metrics['per_task_final_reply']['tool_call']['tool_roundtrip']
    record(f'{arch} original criteria still unmet', True, t['rate'] < metrics['thresholds']['tool_roundtrip'], '只需一個未達項已足以否證全部通過；MoE0/276、Dense7/276。')
    historical[arch] = dict(steps=steps,selected_step=1000,tool_roundtrip=t,threshold=metrics['thresholds']['tool_roundtrip'],seed=raw['seed'],test_count=metrics['count'],weight_sha256=metrics['safe_weights_sha256'])
    del model

hf_tree = json.loads((CACHE / 'hf-v2-tree.json').read_text())
files = [x for x in hf_tree if x['type'] == 'file']
record('HF v2 inference export count', 16, len(files), '匿名Hub API，固定979cdf…；不下載或執行tensor。')
record('HF v2 export groups', ['dense-joint','moe-joint','moe-pretrain','moe-sft'], sorted({Path(x['path']).parts[-2] for x in files}), '四組各manifest/config/tokenizer/safetensors。')
record('HF v2 payload names', ['inference-manifest.json','model-config.json','model.safetensors','tokenizer.json'], sorted({Path(x['path']).name for x in files}), '沒有optimizer、RNG或sampler續訓檔。')
exports=load('docs/selftrained/v2-jobs/release-final-four-exports.json')['release']['exports']
record('four-group sixteen-file explicit MIT grant',True,len(exports)==4 and sum(len(e['files']) for e in exports)==16 and all(f['license']=='MIT' and f['redistribution_approved'] for e in exports for f in e['files']), '以作者逐資產發布契約為自身權重授權來源，不把Hub根README的程式MIT外推。')
record('published inference manifests match release specification',True,all(sha(CACHE/('hf-'+e['name']+'-manifest.json'))==next(f['sha256'] for f in e['files'] if f['path']=='inference-manifest.json') for e in exports), '實取四份manifest，SHA逐筆對release job；全文留Git外。')
for arch in ['moe','dense']:
    f=next(x for x in files if x['path']==f'selftrained/v2/{arch}-joint/model.safetensors')
    record(f'{arch} published versus evaluated weights', historical[arch]['weight_sha256'], f['lfs']['oid'], 'Hub LFS SHA與既有final metrics的safe weights SHA相同。')

models = load('docs/course-experiments/public-models.json')['models']
record('historical formal experiments', 30, len(models), '舊公開清單，不代表新增訓練。')
record('historical exported checkpoints', 120, len({f['path'] for m in models for f in m['files'] if f['path'].endswith('.pt')}), '只計獨立路徑的.pt。')
capstone = load('docs/course-experiments/capstone-public.json')['models']
record('historical capstone groups', [8,3], [sum(not m['id'].startswith('student') for m in capstone),sum(m['id'].startswith('student') for m in capstone)], '8個階段／量化項及3個學生項。')
deployment=load('docs/course-experiments/results/capstone_deployment.json')['results']['stages']
student=load('docs/course-experiments/results/capstone_student.json')['results']['evaluations']
record('historical 90-question counts', [[78,90],[78,90],[62,90],[61,90]], [[deployment[k]['end_to_end_correct'],deployment[k]['count']] for k in ['joint','dpo']]+[[student[k]['end_to_end_correct'],student[k]['count']] for k in ['test_ce','test_kd']], '讀歷史單seed原始結果；不視為v2分數或一般優劣證據。')
flash=load('docs/course-experiments/results/flash_probe.json')
ppo=load('docs/course-experiments/results/posttraining.json')
record('historical CUDA Flash and finite PPO records', ['cuda','complete_run','cpu','complete_run',True], [flash['device'],flash['evidence_status'],ppo['device'],ppo['evidence_status'],'ppo' in ppo['results'] and 'candidate_source' in ppo['results']], '只核歷史機制紀錄存在與設備；不轉為新版品質或CPU加速證據。')
natural=load('docs/natural-assistant/v4/public-release.json')
record('chapter20 released models', ['Qwen/Qwen3-VL-2B-Instruct','openai/whisper-large-v3-turbo','base',0], [natural['base_model']['repo'],natural['asr_model']['repo'],natural['selected_variant'],natural['adapter_parameters']], '固定v4公開配置；本次沒有執行大模型。')

products=load('docs/lesson-products/results/final-report-v1.json')
record('B retained arithmetic failure', [2,42], [products['b']['5']['sum']['correct'],products['b']['5']['sum']['count']], '歷史五輪、10個家族42題；未重跑模型。')
errors=load('docs/lesson-products/results/observed-failure-examples-v1.json')['examples']
record('C retained wording failures', True, all(e['candidates'][max(range(len(e['raw_scores'])), key=e['raw_scores'].__getitem__)] != e['gold'] for e in errors), '從三個保存分數向量重算argmax；沒有把集中度當準確率。')
for product in ['b','c']:
    run=load(f'docs/lesson-products/results/initial-{product}-execution.json')
    record(f'{product} historical GPU training', ['completed','NVIDIA L4',True], [run['status'],run['gpu_name'],run['training']['random_initialization']], '只讀歷史原始execution；未啟動GPU。')

card=[]
for i in range(40):
    w={'task':'card','application':100000+i,'family_id':family('card',{'application':100000+i})}
    card.append(dict(state=render(w,omitted=('application',)),family=w['family_id'],split=split_of(w['family_id'],'probe')))
record('family isolation does not imply unique wording', [1,40,True], [len({r['state'] for r in card}),len({r['family'] for r in card}),len({r['split'] for r in card})>1], '新造40個情境，只驗生成器會收斂為同句，非新模型能力或既有重複數量重算。')
policy=load('docs/lesson-products/results/b-auto-policy.json')
record('Auto rule finite budgets', True, set(rule_depth(policy,'boxes',n) for n in range(20)) <= {1,3,5}, '直接執行現行規則；auto_solve先問C choice/info，再套規則的程式另已閱讀。')

project=tomllib.loads((ROOT/'pyproject.toml').read_text())
record('notebook dependencies', ['ipykernel','jupyterlab','matplotlib','nbclient'], sorted(re.split(r'[<>=!]',x)[0] for x in project['dependency-groups']['notebook']), '直接查本版dependency group。')
spec=importlib.util.spec_from_file_location('curriculum_build',ROOT/'scripts/build_course.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
sample='先看輸入。\n\n```python\nx = 2\nprint(x + 3)\n```\n\n輸出5。\n'
nb=build.notebook('audit','audit',sample,ROOT/'course/chapters/01.md')
record('Markdown generates identical teaching cell', ['x = 2\nprint(x + 3)'], [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and not c['metadata'].get('course_setup')], '只驗當前generator，不宣稱每份Notebook已重跑或Colab已實測。')

# 圖的API順序：forward/loss可先建立graph，zero_grad在backward之前即可。
w=torch.nn.Parameter(torch.tensor(2.0))
optimizer=torch.optim.SGD([w],lr=0.1)
w.grad=torch.tensor(99.0)
loss=(w*3.0-1.0).square()
optimizer.zero_grad()
loss.backward()
grad=float(w.grad)
optimizer.step()
record('figure one nonaccumulating update', [30.0,-1.0], [grad,round(float(w.detach()),6)], 'CPU float32，舊grad99被清掉；loss=(3w−1)^2，初值w2，SGD lr.1。')
logits=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True)
labels=torch.tensor([-100,0])
masked=F.cross_entropy(logits,labels,ignore_index=-100)
masked.backward()
record('loss selection is not attention visibility', [0.0,True], [float(logits.grad[0].abs().sum()),bool(logits.grad[1].abs().sum()>0)], '只驗計分遮罩排除某列代價；attention是另一個API參數，非同一遮罩。')

# 上游完整檔不進Git。直接抽其原始AST中的小函式，避開大模型載入。
def upstream_node(relative, name):
    tree=ast.parse((CACHE/relative).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.Module(body=[node],type_ignores=[])

namespace=dict(torch=torch,F=F)
exec(compile(upstream_node('minimind/train_dpo.py','dpo_loss'),'pinned-minimind-dpo','exec'),namespace)
v=namespace['dpo_loss'](torch.zeros(2,3),torch.zeros(2,3),torch.ones(2,3),0.1)
record('pinned hand-written DPO equal-policy case', round(float(torch.log(torch.tensor(2.))),6), round(float(v),6), '直接執行原版dpo_loss；ref=policy=0 shape[2,3]、mask全1、beta.1；不是重跑其完整訓練。')
exec(compile(upstream_node('minimind/train_distillation.py','distillation_loss'),'pinned-minimind-distillation','exec'),namespace)
v=namespace['distillation_loss'](torch.zeros(2,3),torch.zeros(2,3),temperature=1.5)
record('pinned distillation equal-distribution case', 0.0, float(v), '直接執行原版loss；teacher/student全0 shape[2,3]、T1.5，KL應0；CE混合另查train_epoch原文。')
exec(compile(upstream_node('minimind-v/lm_dataset.py','generate_labels'),'pinned-minimind-v-labels','exec'),namespace)
v=namespace['generate_labels'](SimpleNamespace(bos_id=[9],eos_id=[8],max_length=10),[2,3,9,4,5,8,6])
record('pinned MiniMind-V assistant labels', [-100,-100,-100,4,5,8,-100],v,'執行原版generate_labels；9為assistant起始、8為結束；回覆與結束計分，其他-100。')

class MockVisionLanguageModel(torch.nn.Module):
    def __init__(self,*unused,**kwargs):
        super().__init__()
        self.vision_proj=torch.nn.Linear(2,2)
        self.vision_encoder=torch.nn.Linear(2,2)
        self.model=torch.nn.Module()
        self.model.layers=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(4)])
        self.processor=None

mockspace=dict(AutoTokenizer=SimpleNamespace(from_pretrained=lambda p:None),MiniMindVLM=MockVisionLanguageModel,
               get_model_params=lambda *a:None,Logger=lambda *a:None)
exec(compile(upstream_node('minimind-v/trainer_utils.py','init_vlm_model'),'pinned-minimind-v-freezing','exec'),mockspace)
mock,_,_=mockspace['init_vlm_model'](SimpleNamespace(num_hidden_layers=4),from_weight='none',device='cpu',freeze_llm=1)
actual={name:parameter.requires_grad for name,parameter in mock.named_parameters()}
expected={name:('vision_proj' in name or 'layers.0.' in name or 'layers.3.' in name) for name in actual}
record('pinned MiniMind-V freezing',expected,actual,'原版init_vlm_model＋四層微小mock、from_weight=none；只核requires_grad政策，不載預訓練模型。')

upstream_checks={
    'nanochat/runcpu.sh':['--depth=6','scripts.base_eval'],
    'nanochat/speedrun.sh':['scripts.tok_train','scripts.tok_eval','scripts.base_train','scripts.base_eval','scripts.chat_sft'],
    'nanochat/gpt.py':['F.relu(x).square()','self.lm_head = Linear','self.transformer.wte'],
    'nanochat/loss_eval.py':['total_nats / (math.log(2) * total_bytes)','num_bytes2d > 0'],
    'nanochat/tok_eval.py':['decoded = tokenizer.decode(encoded)'],
    'minimind-v/model_minimind.py':["self.hidden_act = kwargs.get(\"hidden_act\", 'silu')",'self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)','self.act_fn(self.gate_proj(x)) * self.up_proj(x)','class MOEFeedForward'],
    'minimind-v/model_vlm.py':['param.requires_grad = False','self.vision_proj = MMVisionProjector','self.count_vision_proj'],
    'minimind-v/README.md':['--from_weight llm','纯文本对话'],
    'smol/instruction.md':['Supervised Fine-Tuning'],
    'smol/preference.md':['Direct Preference Optimization'],
    'safe-rlhf/README.md':['high reward','low cost'],
    'ao/quantization_overview.rst':['packing format','Efficient kernels'],
    'ao/qat.md':['prepare inserts','convert transforms'],
}
record('pinned upstream configuration checks',True,all(all(needle in (CACHE/path).read_text() for needle in needles) for path,needles in upstream_checks.items()),'對已逐段實讀的官方固定碼版做字面契約核對；不視為上游訓練、品質或速度重現。')

result=dict(created_at=datetime.now(UTC).isoformat(),reviewer_task='/root/omni_technical_coordinator/technical_curriculum',
            command='uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-curriculum-v1/audit_curriculum.py',
            environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',threads=str(torch.get_num_threads())),
            scope='curriculum本頁數量／來源／歷史收據核對及極小機制；沒有新訓練、GPU、成熟模型或全書重審。',
            source_freeze=dict(path=freeze_path,sha256=sha(ROOT/freeze_path)),
            records=records,capacity=capacity,historical=historical,added_sections=added,stage_rows=stage_rows,input_sha256=READFILES,
            source_slice_pins_sha256=sha(OUT/'source-slice-pins.json'),
            figure_mechanism=dict(loss_before_update=25.0,stale_grad=99.0,gradient=grad,weight_after=float(w.detach())),
            result='all checks passed')
(OUT/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(result=result['result'],checks=len(records),environment=result['environment'],execution='docs/technical-reviews/artifacts/omni-technical-curriculum-v1/execution.json'),ensure_ascii=False))
