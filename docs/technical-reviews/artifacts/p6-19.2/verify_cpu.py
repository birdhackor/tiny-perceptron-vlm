import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import torch

from tiny_perceptron.selftrained.dataset import read_records, train_tokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant, PaddingSafeMoE, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS, SPECIALS

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(20261006)
print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': torch.get_num_threads()}, ensure_ascii=False))
def count(module):
    return sum(p.numel() for p in module.parameters())

cfg = SelftrainedConfig(vocab_size=550, width=256, layers=4, heads=4, kv_heads=2, ffn_hidden=512, experts=4, top_k=2, max_length=512)
for arch in ('moe', 'dense'):
    model = LimitedAssistant(replace(cfg, architecture=arch))
    report = model.description()
    expert = sum(p.numel() for n, p in model.lm.named_parameters() if '.ffn.experts.' in n)
    shared = count(model.lm) - expert
    independent_active = shared + cfg.layers * cfg.top_k * count(model.lm.blocks[0].ffn.experts[0]) if arch == 'moe' else count(model.lm)
    row = {key: report[key] for key in ['parameters', 'language_parameters', 'active_language_parameters_per_token', 'perception_parameters', 'expert_parameters', 'router_parameters']}
    row.update(architecture=arch, independently_counted_total=count(model), shared_lm=shared, independently_active=independent_active,
               tied_identity=model.lm.embedding.weight is model.lm.output.weight,
               embedding_shape=list(model.lm.embedding.weight.shape),
               unique_parameter_ids=len({id(p) for p in model.parameters()}),
               named_parameter_entries=len(list(model.named_parameters(remove_duplicate=False))))
    assert independent_active == report['active_language_parameters_per_token']
    assert sum(report['perception_parameters'].values()) == 306883
    assert count(model) == (5447107 if arch == 'moe' else 2288067)
    assert count(model.lm) == (5140224 if arch == 'moe' else 1981184)
    assert model.lm.embedding.weight is model.lm.output.weight
    print('COUNT', json.dumps(row, ensure_ascii=False))
    with torch.no_grad():
        result = model(torch.tensor([[11, 12, 13]]))
    print('MODEL_FORWARD', arch, json.dumps({'logits_shape': list(result['logits'].shape), 'routing_layers': len(result['routing']), 'chosen_shapes': [None if r is None else list(r['chosen'].shape) for r in result['routing']]}))
    untied = LimitedAssistant(replace(cfg, architecture=arch, tied=False))
    delta = count(untied) - count(model)
    assert delta == 550 * 256 and untied.lm.embedding.weight is not untied.lm.output.weight
    print('UNTIED_VARIATION', arch, json.dumps({'delta_parameters': delta, 'tied_identity': False}))

for k in (1, 2, 4):
    model = LimitedAssistant(replace(cfg, top_k=k))
    r = model.description()
    print('TOPK_VARIATION', json.dumps({'top_k': k, 'total': count(model), 'logical_active': r['active_language_parameters_per_token']}))
    assert count(model) == 5447107

mini = PaddingSafeMoE(4, experts=4, top_k=2, hidden=6)
with torch.no_grad():
    mini.router.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.]]))
x = torch.tensor([[[3., 1., .2, .1], [-3., -1., .2, .1], [20., 20., 20., 20.]]], requires_grad=True)
valid = torch.tensor([[True, True, False]])
y, aux, route = mini(x, valid)
probs = mini.router(x[0, :2]).float().softmax(-1)
gates, choices = probs.topk(2, dim=-1)
gates = gates / gates.sum(-1, keepdim=True)
manual = torch.stack([sum(mini.experts[int(choices[i,j])](x[0,i]) * gates[i,j] for j in range(2)) for i in range(2)])
error = float((y[0,:2] - manual).abs().max().detach())
assert error < 1e-6 and torch.equal(y[0,2], torch.zeros(4))
assert route['chosen'].tolist() == [[[0, 1], [2, 3], [-1, -1]]]
(y.square().sum() + .01 * aux).backward()
gradnorms = [sum(float(p.grad.abs().sum()) for p in e.parameters() if p.grad is not None) for e in mini.experts]
assert min(gradnorms) > 0 and mini.router.weight.grad is not None
print('ROUTING_VARIATION', json.dumps({'choices': route['chosen'].tolist(), 'counts': route['counts'].tolist(), 'valid_tokens': route['valid_tokens'], 'gates': gates.tolist(), 'manual_max_abs_error': error, 'expert_gradient_l1': gradnorms, 'router_gradient_l1': float(mini.router.weight.grad.abs().sum()), 'optimizer_updates': 0}))

manifest_path = ROOT / 'docs/selftrained/v2-manifest.json'
manifest = json.loads(manifest_path.read_bytes())
data = ROOT / 'outputs/selftrained-v2/data'
paths, verified = [], []
for row in manifest['records']:
    p = data / row['path']; raw = p.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    assert observed == row['sha256'] and len(raw) == row['bytes']
    paths.append(p); verified.append({'path': row['path'], 'bytes': len(raw), 'sha256': observed})
records = read_records(paths)
tokenizer = train_tokenizer(records)
ordinary = set(''.join(chr(i) for i in range(32, 127)) + OCR_CHARACTERS)
train_rows, train_messages = 0, 0
witnesses = {}
for row in records:
    if row['split'] != 'train':
        continue
    train_rows += 1
    for i, msg in enumerate(row['messages']):
        train_messages += 1
        ordinary.update(msg['content'])
        for ch in msg['content']:
            if ch not in witnesses:
                witnesses[ch] = {'record_id': row['id'], 'split': row['split'], 'message_index': i, 'role': msg['role'], 'character': ch}
assert set(tokenizer.characters) == ordinary
assert tokenizer.vocab_size == 550 and len(tokenizer.characters) == 539
public_path = Path('/tmp/p5-native-public-cpu-smoke-actual/public-model/tokenizer.json')
public_raw = public_path.read_bytes()
(ART / 'public-tokenizer.json').write_bytes(public_raw)
public = CharacterTokenizer.from_dict(json.loads(public_raw))
assert public.to_dict() == tokenizer.to_dict()
receipt_tokenizer_sha = hashlib.sha256(json.dumps(tokenizer.to_dict(), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
for stage in ['moe-pretrain', 'dense-pretrain', 'moe-native', 'dense-weighted']:
    receipt = json.loads((ROOT/'docs/selftrained/results/training-raw'/stage/'raw/train-receipt.json').read_bytes())
    assert receipt['tokenizer_sha256'] == receipt_tokenizer_sha
print('TOKENIZER_FIXED_DATA', json.dumps({'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'verified_record_files': verified, 'total_rows': len(records), 'train_rows': train_rows, 'train_messages': train_messages, 'ordinary_characters': len(ordinary), 'special_ids': len(SPECIALS), 'total_vocab': tokenizer.vocab_size, 'ascii_required': 95, 'ocr_required': len(OCR_CHARACTERS), 'tokenizer_compact_json_sha256': tokenizer.sha256, 'receipt_sorted_json_sha256': receipt_tokenizer_sha, 'public_tokenizer_file_sha256': hashlib.sha256(public_raw).hexdigest()}, ensure_ascii=False))
(ART/'train-character-witnesses.json').write_text(json.dumps({'provenance': 'First occurrence character metadata extracted from fixed SHA-verified train messages; no validation/test witness used.', 'witnesses': list(witnesses.values())}, ensure_ascii=False, indent=2) + '\n')
sample = [{'split': 'train', 'messages': [{'content': 'App商品'}]}, {'split': 'validation', 'messages': [{'content': '🦄'}]}, {'split': 'test', 'messages': [{'content': '🛸'}]}]
base = train_tokenizer(sample)
changed = train_tokenizer(sample + [{'split': 'test', 'messages': [{'content': '💡'}]}])
assert base.to_dict() == changed.to_dict()
assert base.encode('🦄') == [base.unk_id]
assert base.encode('<user>') != [base.user_id]
print('TOKENIZER_SPLIT_VARIATION', json.dumps({'vocab': base.vocab_size, 'test_addition_changes_vocab': False, 'unknown_id': base.unk_id, 'unknown_encoding': base.encode('🦄'), 'unknown_decoding': base.decode(base.encode('🦄')), 'literal_user_encoding': base.encode('<user>'), 'control_user_id': base.user_id}, ensure_ascii=False))
print('PASS: all bounded CPU assertions; no training or generation; no optimizer update.')
