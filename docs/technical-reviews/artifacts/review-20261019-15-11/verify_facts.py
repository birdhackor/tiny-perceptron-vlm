"""核對凍結教材的形狀、路由、參數、重複計時與實驗表格。"""
import ast
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from tiny_perceptron.modern import DenseFFN, MoEFFN
from tiny_perceptron.model import Block, ModelConfig, TinyLM
from scripts.course_experiments.architecture import _parameter_budget


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(length):
    torch.manual_seed(0)
    x = torch.randn(2, length, 8)
    timings = {}
    for name, layer in [('Dense', DenseFFN(8, hidden=16)), ('MoE', MoEFFN(8, experts=4, top_k=2, hidden=4))]:
        with torch.no_grad():
            for _ in range(5):
                layer(x)
            started = time.perf_counter()
            for _ in range(30):
                layer(x)
        elapsed_ms = (time.perf_counter()-started)/30*1000
        timings[name] = elapsed_ms
    return {'length':length,'milliseconds_per_call':timings,'moe_dense_ratio':timings['MoE']/timings['Dense']}


torch.set_num_threads(1)
result = {'command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-11/verify_facts.py',
          'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':torch.version.git_version,
                         'device':'cpu','platform':platform.platform(),'processor':platform.processor(),'intra_op_threads':str(torch.get_num_threads()),
                         'cuda_available':str(torch.cuda.is_available())}}
freeze = ROOT/'docs/course-review-20261019/freeze-01/sources/15.11.md'
text = freeze.read_text()
section = text.split('```python\n')[2].split('```')[0]
class_from_section = ast.parse(section).body[0]
class_from_module = next(n for n in ast.parse((ROOT/'tiny_perceptron/modern.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MoEFFN')
result['freeze']={'source_sha256':digest(freeze),'inline_class_ast_matches_repository':ast.dump(class_from_section)==ast.dump(class_from_module),
                  'figure_count':text.count('.svg')}
result['parameter_arithmetic']={'dense_matrix':8*16+16*8,'dense_bias':16+8,'expert_matrix':8*4+4*8,
                                'expert_bias':4+8,'router':8*4,'moe_active_matrix':2*(8*4+4*8)+8*4}
torch.manual_seed(0)
x = torch.randn(2,8,8)
dense = DenseFFN(8, hidden=16)
moe = MoEFFN(8,experts=4,top_k=2,hidden=4)
flat=x.reshape(-1,x.shape[-1])
p=moe.router(flat).softmax(-1)
w,chosen=p.topk(2,dim=-1)
w=w/w.sum(-1,keepdim=True)
y,aux,actual_chosen=moe(x)
manual=torch.stack([sum(moe.experts[int(chosen[row,slot])](flat[row])*w[row,slot] for slot in range(2)) for row in range(16)])
load=F.one_hot(chosen,4).float().mean((0,1))
importance=p.mean(0)
expected_aux=4*(load.detach()*importance).sum()
result['dispatch']={'x_shape':list(x.shape),'flat_shape':list(flat.shape),'probabilities_shape':list(p.shape),
                    'weights_shape':list(w.shape),'chosen_shape':list(chosen.shape),'output_shape':list(y.shape),
                    'dense_parameters':sum(v.numel() for v in dense.parameters()),'moe_parameters':sum(v.numel() for v in moe.parameters()),
                    'expert_parameter_counts':[sum(v.numel() for v in expert.parameters()) for expert in moe.experts],
                    'input_min':x.min().item(),'input_max':x.max().item(),'router_has_bias':moe.router.bias is not None,'experts_registered':isinstance(moe.experts,nn.ModuleList),
                    'per_expert_assignments':[(chosen==i).sum().item() for i in range(4)],'load':load.tolist(),'load_sum':load.sum().item(),
                    'importance':importance.tolist(),'weights_row_sums':w.sum(-1).tolist(),
                    'output_max_error':(y.reshape(16,8)-manual).abs().max().item(),'auxiliary':aux.item(),
                    'auxiliary_formula_error':(aux-expected_aux).abs().item(),'chosen_matches':bool(torch.equal(actual_chosen,chosen)),
                    'broadcast_weight_shapes': [list(w[torch.where(chosen==i)[0],torch.where(chosen==i)[1],None].shape) for i in range(4)],
                    'output_is_contiguous':y.is_contiguous()}
rows,slots=torch.where(torch.tensor([[1,2],[0,1]])==1)
result['where_example']={'rows':rows.tolist(),'slots':slots.tolist()}
for k in (0,5):
    try:
        MoEFFN(8,experts=4,top_k=k)
    except ValueError as e:
        result[f'invalid_top_k_{k}']=str(e)
result['default_hidden']=DenseFFN(8).up.out_features
with torch.no_grad():
    result['no_grad']={'dense_requires_grad':dense(x).requires_grad,'moe_requires_grad':moe(x)[0].requires_grad}
seed_runs=[]
for _ in range(2):
    torch.manual_seed(0)
    sx=torch.randn(2,8,8)
    sd=DenseFFN(8,hidden=16)
    sm=MoEFFN(8,4,2,4)
    seed_runs.append((sx,sd.up.weight.detach().clone(),sm.router.weight.detach().clone()))
result['seed_repeats_equal']=[bool(torch.equal(a,b)) for a,b in zip(*seed_runs)]
for normalized in (False,True):
    torch.manual_seed(7)
    test=MoEFFN(8,4,1,4)
    tx=torch.randn(2,8,8)
    if not normalized:
        loss=test(tx)[0].square().sum()
    else:
        tf=tx.reshape(-1,8)
        tp=test.router(tf).softmax(-1)
        tw,tc=tp.topk(1,-1)
        tw=tw/tw.sum(-1,keepdim=True)
        out=torch.zeros_like(tf)
        for i,expert in enumerate(test.experts):
            tr,sl=torch.where(tc==i)
            if len(tr):
                out.index_add_(0,tr,expert(tf[tr])*tw[tr,sl,None])
        loss=out.square().sum()
    loss.backward()
    result[f'top1_task_router_gradient_normalized_{normalized}']=test.router.weight.grad.norm().item()
block=Block(ModelConfig(vocab_size=264,width=8,layers=1,heads=1,max_length=8,experts=4,top_k=2))
with torch.no_grad():
    block_out,cache,block_aux=block(x)
result['block']={'output_shape':list(block_out.shape),'auxiliary_shape':list(block_aux.shape),'auxiliary':block_aux.item()}
result['cpu_benchmark']={'warmup_calls_per_variant':5,'measured_calls_per_variant':30,'independent_seed_restarts_per_length':7,'batch_size':2,'feature_width':8,'seed':0,
                         'runs':[benchmark(length) for repeat in range(7) for length in (8,64)]}
for length in (8,64):
    samples=[r for r in result['cpu_benchmark']['runs'] if r['length']==length]
    result['cpu_benchmark'][f'summary_length_{length}']={'dense_median_ms':statistics.median(r['milliseconds_per_call']['Dense'] for r in samples),
         'moe_median_ms':statistics.median(r['milliseconds_per_call']['MoE'] for r in samples),
         'ratio_median':statistics.median(r['moe_dense_ratio'] for r in samples),'ratio_min':min(r['moe_dense_ratio'] for r in samples),'ratio_max':max(r['moe_dense_ratio'] for r in samples)}
record_path=ROOT/'docs/course-experiments/results/moe.json'
record=json.loads(record_path.read_text())
result['recorded_gpu']={k:record[k] for k in ('revision','device','seed','torch_version','python_version','gpu','step_scale','evidence_status')}
result['recorded_gpu']['source_sha256']=digest(record_path)
result['recorded_gpu']['dataset']=record['results']['dataset']
result['recorded_gpu']['variants']={}
for name in ('dense_active_top1','dense_active_top2','dense_total','top1_aux0.01','top2_aux0.01'):
    v=record['results']['variants'][name]
    cfg=ModelConfig(**v['model']['config'])
    model=TinyLM(cfg)
    t=v['training']
    result['recorded_gpu']['variants'][name]={'config':v['model']['config'],'budget':v['budget'],'recomputed_budget':_parameter_budget(model),
        'budget_matches':_parameter_budget(model)==v['budget'],
        'training':{k:t[k] for k in ('batch_size','requested_steps','steps','optimizer_updates','effective_tokens','warm_step_median_seconds','memory_allocated_before_bytes','peak_memory_allocated_bytes','peak_additional_allocated_bytes','auxiliary_weight')},
        'table_ms':round(t['warm_step_median_seconds']*1000,3),'table_before_MiB':round(t['memory_allocated_before_bytes']/2**20,3),
        'table_peak_MiB':round(t['peak_memory_allocated_bytes']/2**20,3),
        'peak_additional_consistent':t['peak_memory_allocated_bytes']-t['memory_allocated_before_bytes']==t['peak_additional_allocated_bytes']}
variants=record['results']['variants']
result['recorded_gpu']['matched_pairs']={}
for k in (1,2):
    d=variants[f'dense_active_top{k}']; m=variants[f'top{k}_aux0.01']
    result['recorded_gpu']['matched_pairs'][str(k)]={'parameter_proxy_difference':m['budget']['active_parameter_proxy_per_token']-d['budget']['active_parameter_proxy_per_token'],
          'moe_dense_latency_ratio':m['training']['warm_step_median_seconds']/d['training']['warm_step_median_seconds']}
paths=['tiny_perceptron/modern.py','tiny_perceptron/model.py','scripts/course_experiments/architecture.py','scripts/course_experiments/common.py','tiny_perceptron/data.py']
result['experiment_code_integrity']={p:{'current_sha256':digest(ROOT/p),'recorded_sha256':record['code_sha256'].get(p),'matches':digest(ROOT/p)==record['code_sha256'].get(p)} for p in paths}
old_commit='a5133f444e0dca2347635f74f8ac83a27bb203d4'
old_command=['git','show',old_commit+':scripts/course_experiments/architecture.py']
old_raw=subprocess.check_output(old_command,cwd=ROOT)
old_tree=ast.parse(old_raw)
current_tree=ast.parse((ROOT/'scripts/course_experiments/architecture.py').read_bytes())
names=['_sync','_forward','_train','_parameter_budget','run_moe','_text_dataset','_clone_config']
function_checks={}
for name in names:
    old_node=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    current_node=next(n for n in current_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    function_checks[name]={'ast_equal':ast.dump(old_node)==ast.dump(current_node),
        'source':ast.get_source_segment(old_raw.decode(),old_node) if name in ('_train','_forward','_parameter_budget') else None}
result['experiment_recorded_method']={'command':old_command,'resolved_commit':old_commit,'recorded_file_sha256':record['code_sha256']['scripts/course_experiments/architecture.py'],
    'git_show_file_sha256':hashlib.sha256(old_raw).hexdigest(),'file_hash_matches_record':hashlib.sha256(old_raw).hexdigest()==record['code_sha256']['scripts/course_experiments/architecture.py'],
    'function_checks':function_checks,'note':'全檔差異是其他實驗入口及其 import；本頁使用的函式 AST 均相同。'}
result['limitations']=['未在本機重跑 CUDA 訓練；GPU 數字核對既有完整實驗 JSON 及其相符程式。','本機 CPU 計時受同時執行的工作與系統排程影響；只核對當次行為，不保證特定比值。','實驗 JSON 沒有逐步 latency 陣列；中位數的取樣範圍由相符的量測程式核對。']
DEST=Path(__file__).with_name('verification.json')
DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(DEST.relative_to(ROOT)),'dispatch':result['dispatch'], 'benchmark_summaries':{k:v for k,v in result['cpu_benchmark'].items() if k.startswith('summary')},
                  'gpu_table':{n:{k:v for k,v in d.items() if k.startswith('table') or k=='budget_matches'} for n,d in result['recorded_gpu']['variants'].items()},
                  'gpu_pairs':result['recorded_gpu']['matched_pairs'],'code_integrity':result['experiment_code_integrity']},ensure_ascii=False,indent=2))
