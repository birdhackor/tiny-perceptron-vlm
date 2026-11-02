"""Bounded glossary arithmetic, tokenizer, masking and navigation inspection."""

import hashlib
import json
import math
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer, SPECIALS, pad_batch, render_chat, shifted
from tiny_perceptron.tokenization import generation_report

ROOT = Path('/tmp/tpb').resolve()
DEST = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-glossary/probe-output.json'
torch.set_num_threads(1)
out = {'created_at': datetime.now(UTC).isoformat(),
       'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                       'device': 'cpu', 'threads': str(torch.get_num_threads())},
       'scope': 'Deterministic tiny examples only; no training, product load, final data or performance claim.'}
out['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                        ['course/glossary.md', 'tiny_perceptron/data.py', 'tiny_perceptron/tokenization.py']}
z = torch.tensor([2., 1., 0.], dtype=torch.float64)
p = z.softmax(-1)
out['softmax'] = {'input': z.tolist(), 'probabilities': p.tolist(),
                  'sum': p.sum().item(), 'rounded_3dp': [round(v, 3) for v in p.tolist()],
                  'third_target_nll': -math.log(p[2].item())}
tok = ByteTokenizer()
out['bytes_and_tokenizer'] = {'specials': list(SPECIALS),
                            'special_ids': [tok.pad_id, tok.bos_id, tok.eos_id, tok.user_id,
                                            tok.assistant_id, tok.image_id, tok.audio_id, tok.system_id],
                            'cat_codepoints': len('貓'), 'cat_utf8_bytes': list('貓'.encode()),
                            'cat_ids': tok.encode('貓'), 'cat_roundtrip': tok.decode(tok.encode('貓')),
                            'thumb_codepoints': [hex(ord(c)) for c in '👍🏽'],
                            'thumb_bytes': len('👍🏽'.encode())}
try:
    chr(0xD800).encode('utf-8')
except UnicodeEncodeError:
    out['bytes_and_tokenizer']['strict_surrogate_encode'] = 'UnicodeEncodeError'
ids = [2, 1, 0, 2]
x, y = shifted(ids)
out['shift'] = {'original': ids, 'input': x.tolist(), 'targets': y.tolist(), 'T': len(x)}
e1 = render_chat([{'role': 'user', 'content': '貓'}, {'role': 'assistant', 'content': '狗'}])
e2 = render_chat([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}])
bx, by, valid = pad_batch([e1, e2])
out['chat_masks'] = {'input': bx.tolist(), 'labels': by.tolist(), 'valid': valid.tolist(),
                     'answer_eos_is_target': [int((yy == tok.eos_id).sum()) for _, yy in [e1, e2]],
                     'bos_is_context_only': all(int(xx[0]) == tok.bos_id and int(yy[0]) == -100
                                                for xx, yy in [e1, e2]),
                     'padding_labels_are_ignored': bool((by[~valid] == -100).all()),
                     'padding_ids_are_zero': bool((bx[~valid] == 0).all()),
                     'nonpadding_context_can_have_ignored_target': bool((valid & (by == -100)).any()),
                     'eos_report': generation_report(tok, tok.encode('狗') + [tok.eos_id])}
masked_logits = torch.stack([z, torch.tensor([100., -100., -100.], dtype=torch.float64), z])
masked_targets = torch.tensor([2, -100, 0])
nll_sum = F.cross_entropy(masked_logits, masked_targets, reduction='sum')
mean_nll = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
out['loss_denominator'] = {'targets': masked_targets.tolist(), 'valid_targets': 2,
                           'nll_sum': nll_sum.item(), 'mean_nll': mean_nll.item(),
                           'sum_divided_by_2': (nll_sum / 2).item()}
inp = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
w = torch.tensor([[1., 2., -1.], [-1., 0., 1.]], dtype=torch.float64)
b = torch.tensor([.5, -.5], dtype=torch.float64)
linear = F.linear(inp, w, b)
out['linear_last_axis'] = {'input_shape': list(inp.shape), 'weight_shape': list(w.shape),
                           'bias_shape': list(b.shape), 'output_shape': list(linear.shape),
                           'output': linear.tolist(), 'matches_x_w_transpose_plus_b':
                               bool(torch.equal(linear, inp @ w.T + b))}
out['nan_row'] = {'input': '-inf, -inf, -inf',
                   'all_softmax_outputs_nan': bool(torch.isnan(torch.full((3,), -torch.inf).softmax(0)).all())}
vectors = torch.tensor([[3., 4.], [0., 0.], [1e-20, 0.]], dtype=torch.float64)
normalized = F.normalize(vectors, p=2, dim=-1, eps=1e-12)
out['l2_eps'] = {'vectors': vectors.tolist(), 'eps': 1e-12,
                 'normalized': normalized.tolist(), 'norms': normalized.norm(dim=-1).tolist()}
za = torch.tensor([2., 0., 0.], dtype=torch.float64)
zb = torch.tensor([1., 0., -100.], dtype=torch.float64)
out['temperature_order'] = [{'T': t, 'a_confidence': (za/t).softmax(0).max().item(),
                            'b_confidence': (zb/t).softmax(0).max().item(),
                            'argmax_a': int((za/t).argmax()), 'argmax_b': int((zb/t).argmax())}
                           for t in [1., 2.]]
body = (ROOT / 'course/glossary.md').read_text()
links = re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', body)
problems = []
targets = {}
for link in links:
    file, _, frag = link.partition('#')
    if '://' in file:
        continue
    target = (ROOT / 'course' / file).resolve()
    if not target.is_file():
        problems.append({'link': link, 'problem': 'missing file'})
        continue
    raw = target.read_text()
    targets[str(target.relative_to(ROOT))] = hashlib.sha256(target.read_bytes()).hexdigest()
    if frag and not re.search(r'^##\s+' + re.escape(frag) + r'(?:\s|$)', raw, re.M):
        problems.append({'link': link, 'problem': 'missing numbered H2'})
out['navigation'] = {'link_count': len(links), 'unique_targets': targets, 'problems': problems,
                     'table_definition_rows': sum(line.startswith('| ') and not line.startswith('| ---')
                                                  and line.split('|')[1].strip() != '名詞'
                                                  for line in body.splitlines()),
                     'numbered_sections': re.findall(r'^##\s+(G\.\d+)\s+(.+)$', body, re.M)}
index_terms = []
for line in body.splitlines()[224:]:
    if not line.startswith('**'):
        continue
    entries = line.split('**', 2)[2].strip().split(' · ')
    for entry in entries:
        if entry.startswith('decode（'):
            index_terms.append('decode')
        else:
            match = re.match(r'\[([^\]]+)\]', entry)
            if match:
                index_terms.append(match.group(1))
sort_keys = [re.sub(r'[^a-z0-9]', '', term.lower()) for term in index_terms]
out['navigation']['index_term_count'] = len(index_terms)
out['navigation']['index_duplicate_keys'] = sorted({key for key in sort_keys if sort_keys.count(key) > 1})
out['navigation']['index_out_of_order'] = [index_terms[i:i+2] for i in range(len(sort_keys)-1)
                                            if sort_keys[i] > sort_keys[i+1]]
checks = {'softmax_rounding': out['softmax']['rounded_3dp'] == [.665, .245, .09],
          'special_ids': out['bytes_and_tokenizer']['special_ids'] == list(range(8)),
          'byte_token_count': len(tok.encode('貓')) == 3,
          'shift_T_plus_1': x.tolist() == [2, 1, 0] and y.tolist() == [1, 0, 2] and ids == [2, 1, 0, 2],
          'masked_mean': abs(nll_sum.item()/2 - mean_nll.item()) < 1e-12,
          'chat_mask_checks': all(out['chat_masks'][k] for k in
              ['bos_is_context_only', 'padding_labels_are_ignored', 'padding_ids_are_zero',
               'nonpadding_context_can_have_ignored_target']),
          'linear': out['linear_last_axis']['matches_x_w_transpose_plus_b'],
          'nan': out['nan_row']['all_softmax_outputs_nan'],
          'temperature_confidence_order_can_flip': out['temperature_order'][0]['a_confidence'] >
              out['temperature_order'][0]['b_confidence'] and out['temperature_order'][1]['a_confidence'] <
              out['temperature_order'][1]['b_confidence'],
          'all_navigation_targets_exist': not problems,
          'index_order_and_uniqueness': not out['navigation']['index_out_of_order']
              and not out['navigation']['index_duplicate_keys']}
out['checks'] = checks
DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(DEST.relative_to(ROOT)), 'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
                  'checks': checks, 'all_pass': all(checks.values()), 'environment': out['environment']},
                 ensure_ascii=False))
if not all(checks.values()):
    raise SystemExit(1)
