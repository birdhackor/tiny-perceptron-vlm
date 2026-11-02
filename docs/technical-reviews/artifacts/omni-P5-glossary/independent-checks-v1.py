"""Small CPU examples and static inspection; never loads product weights or trains."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, SPECIALS, IGNORE, render_chat, pad_batch, shifted
from tiny_perceptron.model import loss_sum, TinyLM, ModelConfig
from tiny_perceptron.modern import DenseFFN, MoEFFN

PROOF = Path(__file__).resolve().parent
RESULTS = {}


def record(name, value):
    RESULTS[name] = value


def extracted_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


def main():
    body = (ROOT / 'course/glossary.md').read_bytes()
    freeze_path = 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v6.json'
    frozen = next(p for p in json.loads((ROOT / freeze_path).read_text())['pages'] if p['page_id'] == 'glossary')
    assert hashlib.sha256(body).hexdigest() == frozen['source_sha256']
    assert (ROOT / frozen['snapshot']).read_bytes() == body
    table_rows = [line for line in body.decode().splitlines() if line.startswith('| ') and not line.startswith(('| ---', '| 名詞'))]
    assert len(table_rows) == 136
    record('freeze', {'body_bytes': len(body), 'body_sha256': frozen['source_sha256'],
                       'canonical_freeze': freeze_path, 'canonical_freeze_sha256': hashlib.sha256((ROOT / freeze_path).read_bytes()).hexdigest(),
                       'table_definition_rows': len(table_rows), 'figures': frozen['figures_sha256'],
                       'fenced_code_blocks': body.decode().count('```') // 2})

    links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', body.decode())
    broken = []
    for target in links:
        if target.startswith('https:'):
            continue
        path, _, fragment = target.partition('#')
        local = ROOT / 'course' / path
        if not local.is_file() or (fragment and re.search(r'^## ' + re.escape(fragment) + r' ', local.read_text(), re.M) is None):
            broken.append(target)
    assert not broken, broken
    index = body.decode().split('### 英文與縮寫索引', 1)[1]
    labels = []
    order_issues = []
    for line in index.splitlines():
        if not line.startswith('**'):
            continue
        terms = re.findall(r'\[([^\]]+)\]\(', line)
        if line.startswith('**D**'):
            terms = ['decode'] + terms[2:]
        keys = [re.sub('[^a-z0-9]', '', x.lower().replace(' 表', '')) for x in terms]
        if keys != sorted(keys):
            order_issues.append({'line': line, 'keys': keys})
        labels.extend(terms)
    assert not order_issues, order_issues
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    assert not duplicates
    record('link_index_inspection', {'link_occurrences_checked': len(links), 'unique_destinations_checked': len(set(links)),
                                    'broken_paths_or_numbered_anchors': broken, 'index_terms': len(labels),
                                    'index_order_issues': order_issues, 'duplicate_index_labels': duplicates,
                                    'decode': 'one index term with two definition destinations'})

    vocabulary = {'狗': 0, '看': 1, '貓': 2}
    ids = [vocabulary[c] for c in '貓看狗']
    assert ids == [2, 1, 0]
    embedding = torch.tensor([[0., 0.], [0., 0.], [.2, -.1]], dtype=torch.float64)
    assert embedding[2].tolist() == [.2, -.1]
    tok = ByteTokenizer()
    assert list(range(8)) == [tok.pad_id, tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id, tok.image_id, tok.audio_id, tok.system_id]
    assert len('貓'.encode('utf-8')) == 3 and len(tok.encode('貓')) == 3
    assert len('👍🏽') == 2 and len('👍🏽'.encode('utf-8')) == 8
    assert len(tok.encode('👍🏽')) == 8 and tok.decode(tok.encode('貓👍🏽')) == '貓👍🏽'
    assert all(x >= 8 for x in tok.encode('<assistant>'))
    record('ids_utf8_specials', {'example_ids': ids, 'id2_embedding': embedding[2].tolist(), 'specials': list(SPECIALS),
                               'cat_utf8_bytes': list('貓'.encode('utf-8')), 'thumbs_skin_codepoints': [f'U+{ord(c):04X}' for c in '👍🏽'],
                               'byte_values': '2**8 = 256 possibilities, inclusive range 0..255',
                               'surrogate': 'RFC3629 excludes D800..DFFF; Python strict UTF-8 rejects these values'})
    try:
        '\ud800'.encode('utf-8')
        raise AssertionError('surrogate accepted')
    except UnicodeEncodeError:
        pass

    scores = torch.tensor([2., 1., 0.], dtype=torch.float64)
    p = scores.softmax(0)
    assert torch.allclose(p, torch.tensor([.665, .245, .090], dtype=torch.float64), atol=.0005, rtol=0)
    assert torch.softmax(torch.full((3,), -torch.inf), dim=0).isnan().all()
    total, count = loss_sum(torch.stack([scores, scores])[None], torch.tensor([[2, IGNORE]]))
    assert count.item() == 1 and abs(total.item() + p[2].log().item()) < 1e-12
    record('softmax_cross_entropy', {'logits': scores.tolist(), 'probabilities': p.tolist(),
                                    'third_class_nll': -p[2].log().item(), 'three_decimal_tolerance': .0005,
                                    'all_negative_infinity_softmax': 'all NaN', 'ignored_labels': 'second target -100 contributes neither NLL nor count'})

    x, y = shifted([8, 9, 10, 11])
    assert x.tolist() == [8, 9, 10] and y.tolist() == [9, 10, 11]
    chat = render_chat([{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': '貓'}])
    assert (chat[1] != IGNORE).sum().item() == 4 and chat[1][-1].item() == tok.eos_id
    padded_x, padded_y, valid = pad_batch([chat, render_chat([{'role': 'assistant', 'content': 'a'}])])
    assert (padded_x[~valid] == 0).all() and (padded_y[~valid] == IGNORE).all()
    assert ((padded_y == IGNORE) & valid).any()
    record('shift_bos_eos_padding', {'shifted_inputs': x.tolist(), 'shifted_targets': y.tolist(),
                                   'chat_inputs': chat[0].tolist(), 'chat_targets': chat[1].tolist(),
                                   'assistant_eos_is_scored': True, 'bos_is_context_only': True,
                                   'padding_id': 0, 'padding_loss_target': -100,
                                   'real_context_can_be_excluded_from_loss': True})

    torch.manual_seed(123)
    layer = nn.Linear(3, 2)
    assert layer.bias is not None and nn.Linear(3, 2, bias=False).bias is None
    assert list(layer(torch.ones(2, 4, 3)).shape) == [2, 4, 2]
    assert F.relu(torch.tensor([-1., 0., 1.])).tolist() == [0., 0., 1.]
    ffn = DenseFFN(4)
    assert (ffn.up.in_features, ffn.up.out_features, ffn.down.out_features) == (4, 16, 4)
    tiny = torch.tensor([[0., 0.], [1e-20, 0.], [3., 4.]])
    normed = F.normalize(tiny, dim=-1)
    assert normed[0].norm().item() == 0 and normed[1].norm().item() < 1 and abs(normed[2].norm().item() - 1) < 1e-6
    record('linear_ffn_norm', {'linear_input': [2, 4, 3], 'linear_output': [2, 4, 2], 'ffn_width_hidden': [4, 16],
                               'relu_neg_zero_pos': [0., 0., 1.], 'l2_output_norms': normed.norm(dim=-1).tolist()})

    parameter = nn.Parameter(torch.tensor([1., 1.], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    parameter.grad = torch.tensor([1., 100.], dtype=torch.float64)
    optimizer.step()
    state = optimizer.state[parameter]
    assert torch.allclose(state['exp_avg'], torch.tensor([.1, 10.], dtype=torch.float64))
    assert torch.allclose(state['exp_avg_sq'], torch.tensor([.001, 10.], dtype=torch.float64))
    optimizer.zero_grad()
    assert parameter.grad is None
    record('adam_artificial_step', {'purpose': 'two synthetic scalar parameters; no task, data, trained model or product weights',
                                   'm': state['exp_avg'].tolist(), 'v': state['exp_avg_sq'].tolist(),
                                   'step_count': state['step'].item(), 'new_parameters': parameter.detach().tolist(),
                                   'state_dict_top_keys': list(optimizer.state_dict()), 'zero_grad': 'grad is None'})

    probs = torch.tensor([.6, .2, .2], dtype=torch.float64)
    concentration = (probs.max().item() - 1 / 3) / (1 - 1 / 3)
    assert abs(concentration - .4) < 1e-12
    a = torch.tensor([1., 0., 0.], dtype=torch.float64)
    b = torch.tensor([1., .9, -10.], dtype=torch.float64)
    ranking = []
    for temperature in [1., 2.]:
        pa, pb = (a / temperature).softmax(0), (b / temperature).softmax(0)
        assert a.argmax() == pa.argmax() and b.argmax() == pb.argmax()
        ranking.append({'T': temperature, 'A_pmax': pa.max().item(), 'B_pmax': pb.max().item()})
    assert ranking[0]['A_pmax'] > ranking[0]['B_pmax'] and ranking[1]['A_pmax'] < ranking[1]['B_pmax']
    record('confidence_temperature', {'jev_choice_concentration': concentration, 'pmax': .6, 'T_ranking_counterexample': ranking,
                                      'A_logits': a.tolist(), 'B_logits': b.tolist(), 'argmax_preserved': True,
                                      'not_a_fit': 'T values are illustrative constants, no calibration or policy selection performed'})
    accept = [True, False, True, True, False]
    wrong = [False, False, True, False, False]
    coverage = sum(accept) / len(accept)
    risk = sum(a and e for a, e in zip(accept, wrong)) / sum(accept)
    assert coverage == .6 and risk == 1 / 3
    record('coverage_risk', {'artificial_rows': 5, 'accepted': 3, 'accepted_errors': 1, 'coverage': coverage,
                             'risk': risk, 'zero_accepted_risk': None, 'empirical_quality_inference': 'none'})

    p0 = torch.tensor([.5, .5], dtype=torch.float64)
    kl = (p0 * (p0.log() - p0.log())).sum().item()
    entropy = -(p0 * p0.log()).sum().item()
    assert kl == 0 and abs(entropy - math.log(2)) < 1e-15
    snr = 10 * math.log10((1.**2 + (-1.)**2) / ((1. - .9)**2 + (-1. + .9)**2))
    assert abs(snr - 20) < 1e-12
    record('kl_entropy_snr_frequency', {'KL_p_p': kl, 'H_half_half_nats': entropy, 'reference': [1, -1],
                                        'reconstruction': [.9, -.9], 'snr_db': snr, 'frequency': '440 Hz means 440 cycles per second'})
    records = []
    for advantage in [-1., 1.]:
        for ratio in [.5, 1., 1.5]:
            clipped = min(1.2, max(.8, ratio))
            records.append({'advantage': advantage, 'ratio': ratio, 'surrogate': min(ratio * advantage, clipped * advantage)})
    record('ppo_surrogate', {'epsilon': .2, 'artificial_terms': records, 'scope': 'scalar objective evaluation, no policy update'})
    w = torch.tensor([[1., 2.], [3., 4.]])
    aa, bb, xx = torch.tensor([[1., 2.]]), torch.tensor([[1.], [2.]]), torch.tensor([2., -1.])
    assert torch.equal(w @ xx + bb @ (aa @ xx), (w + bb @ aa) @ xx)
    record('lora_and_rvq', {'lora': 'W x + B(Ax) = (W + BA)x; inner width r=1 in a 2x2 example',
                           'rvq': {'source': [1.2, -.4], 'first_vector': [1., 0.], 'residual': [.2, -.4],
                                   'second_vector': [.25, -.5], 'sum_reconstruction': [1.25, -.5],
                                   'remaining_error': [-.05, .1]}})

    budget = extracted_function('scripts/course_experiments/architecture.py', '_parameter_budget', {'MoEFFN': MoEFFN})
    toy = TinyLM(ModelConfig(vocab_size=8, width=4, layers=1, heads=1, max_length=5, experts=3, top_k=2))
    result = budget(toy)
    expert_one = sum(p.numel() for p in toy.blocks[0].ffn.experts[0].parameters())
    expected = sum(p.numel() for p in toy.parameters()) - 3 * expert_one + 2 * expert_one
    assert result['active_parameter_proxy_per_token'] == expected
    record('parameter_proxy', {**result, 'independent_formula': expected, 'whole_embedding_and_position_counted': True,
                               'does_not_measure_FLOPs_or_speed': True, 'formal_model_loaded': False})
    reward = extracted_function('scripts/course_experiments/applications.py', '_policy_reward',
                                {'_POLICY_ACTIONS': [str(i) for i in range(16)] + [' '.join(map(str, range(16)))], 're': re})
    assert all(reward(16, truth, proxy=True) == 1 and reward(16, truth, proxy=False) == 0 for truth in range(16))
    record('strict_loose_reward', {'truths_checked': list(range(16)), 'enumeration_loose': 'all 1.0', 'enumeration_strict': 'all 0.0',
                                  'source_function': 'scripts/course_experiments/applications.py::_policy_reward', 'training_performed': False})

    source_paths = [
        'tiny_perceptron/data.py', 'tiny_perceptron/model.py', 'tiny_perceptron/modern.py',
        'tiny_perceptron/attention.py', 'tiny_perceptron/omni_products/choice.py',
        'tiny_perceptron/omni_products/reliability.py', 'tiny_perceptron/omni_products/signal_metrics.py',
        'tiny_perceptron/omni_products/inference.py', 'tiny_perceptron/omni_products/training_runtime.py',
        'tiny_perceptron/omni_products/sound.py', 'tiny_perceptron/omni_products/data.py',
        'tiny_perceptron/lesson_products/model.py', 'tiny_perceptron/lesson_products/training.py',
        'tiny_perceptron/selftrained/model.py', 'tiny_perceptron/selftrained/tokenizer.py',
        'scripts/course_experiments/architecture.py', 'scripts/course_experiments/applications.py',
        'scripts/course_experiments/compression.py',
        'docs/selftrained/v2-manifest.json', 'docs/lesson-products/data-v2-manifest.json',
        'docs/lesson-products/results/duplicate-observation-example-v1.json',
        'docs/course-experiments/results/distillation.json', 'docs/course-experiments/results/multimodal_distillation.json',
    ]
    inspections = []
    for path in source_paths:
        raw = (ROOT / path).read_bytes()
        if path.endswith('.py'):
            tree = ast.parse(raw)
            definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        else:
            definitions = list(json.loads(raw))
        inspections.append({'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'top_level_definitions_or_keys': definitions})
    record('repository_source_inventory', inspections)
    record('execution_boundaries', {'formal_NN_inference': False, 'product_weights_loaded': False, 'training_jobs': False,
                                    'T_or_tau_tuned': False, 'dispatch_or_publish': False,
                                    'allowed_execution': 'artificial scalar/tensor examples and deterministic source/JSON inspection only'})
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer_task': '/root/omni_p5_technical_glossary_whole',
              'command': '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/omni-P5-glossary/independent-checks-v1.py',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'CPU'},
              'status': 'passed', 'checks': RESULTS}
    target = PROOF / 'independent-checks-v1.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(RESULTS), 'output': str(target.relative_to(ROOT)),
                      'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
