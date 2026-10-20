import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

torch.set_num_threads(1)
torch.manual_seed(42)
config = SelftrainedConfig(vocab_size=32, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32, top_k=2)
core = LimitedAssistant(config).eval()
ids = torch.tensor([[1, 11, 12, 13, 14]])
before = {k: v.clone() for k, v in core.state_dict().items()}
with torch.no_grad():
    full = core(ids)["logits"][:, -1]
    splits = []
    for split in (3, 2):
        prefix = core(ids[:, :split])
        result = core(ids[:, split:], cache=prefix["cache"])
        scores = result["logits"][:, -1]
        splits.append({"split": split, "max_abs_difference": (full-scores).abs().max().item(),
                       "allclose": torch.allclose(full, scores, atol=1e-4, rtol=1e-4),
                       "candidate_shape": list(scores.shape),
                       "prefix_cache_shape": list(prefix["cache"][0][0].shape),
                       "final_cache_shape": list(result["cache"][0][0].shape),
                       "requires_grad": scores.requires_grad})
    core.train()
    train_logits = core(ids)["logits"]
    core.eval()
    eval_logits = core(ids)["logits"]
    mode_equal = torch.equal(train_logits, eval_logits)
with torch.enable_grad():
    graph_enabled = core(ids)["logits"].requires_grad
unchanged = all(torch.equal(before[k], v) for k, v in core.state_dict().items())

published = json.loads((ROOT/'docs/selftrained/results/v2-final-public-results.json').read_text())
final_config = published['architectures']['moe']['capacity']['config']
final_core = LimitedAssistant(SelftrainedConfig(**final_config)).eval()
sdpa_calls = []
original_sdpa = F.scaled_dot_product_attention
def traced_sdpa(q, k, v, **kwargs):
    sdpa_calls.append({"query_shape":list(q.shape),"key_shape":list(k.shape),
                       "mask_shape":list(kwargs['attn_mask'].shape),"dtype":str(q.dtype),
                       "device":str(q.device),"dropout_p":kwargs.get('dropout_p')})
    return original_sdpa(q, k, v, **kwargs)
F.scaled_dot_product_attention = traced_sdpa
valid = torch.tensor([[True, True, True, False]])
with torch.no_grad():
    final_result = final_core(torch.tensor([[1, 11, 12, 0]]), attention_mask=valid)
F.scaled_dot_product_attention = original_sdpa
routes = [{"expert_count":len(block.ffn.experts),"top_k":block.ffn.top_k,
           "valid_tokens":route['valid_tokens'],"counts":route['counts'].tolist(),
           "chosen":route['chosen'].tolist()} for block,route in zip(final_core.lm.blocks,final_result['routing'])]

payloads = [[{"kind":"image","values":torch.rand(2,1,32,32),
              "coordinates":torch.tensor([[.25,.5],[.75,.5]])},
             {"kind":"audio","values":torch.rand(20,40)}]]
multimodal_ids = torch.tensor([[1,7,7]+[9]*16+[11]])
generation = []
for use_cache in (True,False):
    calls = {"image":0,"audio":0,"input_lengths":[]}
    def image_hook(*args):
        calls['image'] += 1
    def audio_hook(*args):
        calls['audio'] += 1
    def model_hook(module,args):
        calls['input_lengths'].append(args[0].shape[1])
    hooks = [core.vision_encoder.register_forward_hook(image_hook),
             core.audio_encoder.register_forward_hook(audio_hook),core.register_forward_pre_hook(model_hook)]
    kwargs = {} if use_cache else {"use_cache":False}
    generated = core.generate(multimodal_ids,modalities=payloads,max_new_tokens=3,eos_id=-1,**kwargs)
    for hook in hooks:
        hook.remove()
    generation.append({"use_cache":use_cache,"calls":calls,"generated":generated.tolist()})

tokenizer = CharacterTokenizer.build(['短','比較長'])
encoder = RecordEncoder(tokenizer, ROOT, context=128)
records = [{"id":str(i),"task":"text","messages":[{"role":"user","content":s},
            {"role":"assistant","content":"短"}]} for i,s in enumerate(('短','比較長'))]
batch = encoder.batch(records)
encoded = [encoder.encode(r) for r in records]
batch_summary = {"shape":list(batch['input_ids'].shape),"row_lengths":[len(r['input_ids']) for r in encoded],
                 "ids":batch['input_ids'].tolist(),"mask":batch['attention_mask'].tolist(),
                 "labels":batch['labels'].tolist(),"segments_argument_present":'segments' in batch,
                 "one_record_per_row":all(torch.equal(batch['input_ids'][i,:len(r['input_ids'])],
                    torch.tensor(r['input_ids'])) for i,r in enumerate(encoded))}

source_paths = ['tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/dataset.py',
                'tiny_perceptron/selftrained/inference.py','tiny_perceptron/attention.py',
                'tiny_perceptron/model.py','tiny_perceptron/modern.py','scripts/selftrained/train.py',
                'docs/selftrained/results/v2-final-public-results.json']
result = {"command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-9/probe.py",
          "environment":{"python":platform.python_version(),"torch":str(torch.__version__),
                         "torch_git_version":str(torch.version.git_version),"device":"cpu","dtype":"float32"},
          "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
          "toy_config":vars(config),"splits":splits,"train_eval_exact_equal":mode_equal,
          "parameters_unchanged":unchanged,"logits_require_grad_when_enabled":graph_enabled,
          "mode_dependent_modules":[name for name,module in core.named_modules()
                                    if isinstance(module,(torch.nn.Dropout,torch.nn.modules.batchnorm._BatchNorm))],
          "published_config":final_config,"sdpa_calls":sdpa_calls,
          "attention":{"query_weight_shape":list(final_core.lm.blocks[0].attention.q.weight.shape),
                       "key_weight_shape":list(final_core.lm.blocks[0].attention.k.weight.shape),
                       "value_weight_shape":list(final_core.lm.blocks[0].attention.v.weight.shape),
                       "cache_key_shapes":[list(k.shape) for k,v in final_result['cache']],
                       "tied_weight_identity":final_core.lm.output.weight is final_core.lm.embedding.weight,
                       "rotary":[b.attention.rotary for b in final_core.lm.blocks],
                       "norm_types":[type(b.norm1).__name__ for b in final_core.lm.blocks]},
          "routing":routes,"generation_signature":str(inspect.signature(LimitedAssistant.generate)),
          "generation":generation,"batch":batch_summary,
          "timing_measured":False,"trained_weights_loaded":False}
assert all(r['allclose'] for r in splits), '完整與快取計算須在容差內相同'
assert mode_equal and unchanged, '評估模式與權重不變性須成立'
assert generation[0]['calls'] == {'image':1,'audio':1,'input_lengths':[20,1,1]}, '快取應沿用感知前文'
assert generation[1]['calls'] == {'image':3,'audio':3,'input_lengths':[20,21,22]}, '完整計算應重算感知前文'
assert batch_summary['one_record_per_row'], '每列應保留一筆完整紀錄'
print(json.dumps(result,ensure_ascii=False,indent=2))
