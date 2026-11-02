"""19.16 frozen inline-code and historical-receipt verification; CPU only."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.lesson_products import data
from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.model import candidate_batch, grouped_scores, validate_candidates
from tiny_perceptron.lesson_products.training import state_digest


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-19-16-v1')
FREEZE = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'
EXPECTED_SOURCE = '0f36f932b20323884dc778ac8c51724d79191953a515e7582b46a7b0469e7ca0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def class_ast(code, name):
    return ast.dump(next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    started_at = datetime.now(UTC).isoformat()
    freeze = load(ROOT / FREEZE)
    page = next(p for p in freeze['pages'] if p['page_id'] == '19.16')
    path = ROOT / page['snapshot']
    assert sha(path) == page['source_sha256'] == EXPECTED_SOURCE
    raw = path.read_text()
    assert page['figures_sha256'] == {} and not re.search(r'!\[[^]]*\]\(', raw)
    snippets = re.findall(r'```python\n(.*?)```', raw, flags=re.S)
    assert len(snippets) == 2
    namespace = {'__name__': '__technical_inline__'}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(snippets[0], page['snapshot'] + ':17-87', 'exec'), namespace)
    printed = stdout.getvalue()
    expected_output = re.search(r'```text\n(.*?)```', raw, flags=re.S).group(1)
    assert printed.startswith(expected_output)
    assert "{'none': 0.2472, 'app': 0.2524, 'address': 0.2484, 'card': 0.2519}" in printed
    model = namespace['model']
    tokenizer = namespace['tokenizer']
    tokens = namespace['tokens']
    model_source = (ROOT / 'tiny_perceptron/lesson_products/model.py').read_text()
    assert class_ast(snippets[1], 'CandidateScorer') == class_ast(model_source, 'CandidateScorer')

    pre = next(p for p in freeze['pages'] if p['page_id'] == '19.15')
    pre_path = ROOT / pre['snapshot']
    assert sha(pre_path) == pre['source_sha256']
    block_code = next(s for s in re.findall(r'```python\n(.*?)```', pre_path.read_text(), flags=re.S) if 'class Block' in s)
    assert class_ast(block_code, 'Block') == class_ast(model_source, 'Block')
    namespace['F'] = torch.nn.functional
    exec(compile(block_code, pre['snapshot'] + ':56-76', 'exec'), namespace)
    exec(compile(snippets[1], page['snapshot'] + ':106-126', 'exec'), namespace)
    inline_model = namespace['CandidateScorer'](model.config).eval()
    inline_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = model(tokens)
        actual = inline_model(tokens)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        one = candidate_batch([{'state': namespace['state'], 'question': 'choice', 'candidates': ['card', 'address', 'app', 'none']}], tokenizer, model.config)
        forward_scores = model(one[0])
        reverse_scores = model(tokens).flip(0)
        torch.testing.assert_close(forward_scores, reverse_scores, atol=1e-6, rtol=1e-6)
        permutation_difference = float((forward_scores - reverse_scores).abs().max())
        probability_difference = float((forward_scores.softmax(-1) - reverse_scores.softmax(-1)).abs().max())
        # Extra padding changes only masked source positions; valid CLS output stays close.
        more_padding = torch.nn.functional.pad(one[0], (0, 4))
        padding_difference = float((model(more_padding) - forward_scores).abs().max())
        torch.testing.assert_close(model(more_padding), forward_scores, atol=1e-6, rtol=1e-6)
    components = {
        'token_embedding': 144 * 64,
        'position_embedding': 128 * 64,
        'one_block': 2 * (2 * 64) + (64 * 192 + 192) + (64 * 64 + 64) + (64 * 128 + 128) + (128 * 64 + 64),
        'final_norm': 2 * 64,
        'score_head': 64 + 1,
    }
    total = components['token_embedding'] + components['position_embedding'] + 2 * components['one_block'] + components['final_norm'] + components['score_head']
    assert total == sum(p.numel() for p in model.parameters()) == 84545
    assert len(model.blocks) == 2 and model.blocks[0].qkv.weight.data_ptr() != model.blocks[1].qkv.weight.data_ptr()
    rows = [
        {'state': namespace['state'], 'question': 'choice', 'candidates': ['card', 'none']},
        {'state': namespace['state'], 'question': 'choice', 'candidates': ['app', 'address', 'none']},
    ]
    batch_tokens, slices, maximum, labels = candidate_batch(rows, tokenizer, model.config)
    with torch.no_grad():
        grouped = grouped_scores(model(batch_tokens), slices, maximum)
        probs = grouped.softmax(-1)
    assert slices == [(0, 2), (2, 5)] and maximum == 3 and labels.numel() == 0
    assert torch.isneginf(grouped[0, 2]) and probs[0, 2].item() == 0
    rejections = {}
    for name, question, candidates in [
        ('one_candidate', 'choice', ['none']),
        ('duplicate', 'choice', ['none', 'none']),
        ('unknown', 'choice', ['other', 'none']),
        ('missing_none', 'choice', ['card', 'app']),
    ]:
        try:
            validate_candidates(question, candidates)
        except ValueError as error:
            rejections[name] = str(error)
        else:
            raise AssertionError(name)
    world_card = {'task': 'card', 'application': '123456'}
    world_address = {'task': 'address'}
    world_sum = {'task': 'sum', 'x': 1, 'y': 2, 'z': 3}
    rule_probes = {
        'card_missing': data.judgment(world_card, ('application',), 'info', ['yes', 'no']),
        'card_grade0': data.judgment(world_card, ('application',), 'rubric', ['grade0', 'grade1', 'grade2']),
        'address_no_required_number': data.judgment(world_address, (), 'info', ['yes', 'no']),
        'sum_partial_grade1': data.judgment(world_sum, ('x',), 'rubric', ['grade0', 'grade1', 'grade2']),
        'sum_complete_grade2': data.judgment(world_sum, (), 'rubric', ['grade0', 'grade1', 'grade2']),
        'correct_candidate_absent': data.judgment(world_card, (), 'choice', ['address', 'none']),
    }
    assert list(rule_probes.values()) == ['no', 'grade0', 'yes', 'grade1', 'grade2', 'none']

    history_paths = [
        'docs/lesson-products/results/initial-c-execution.json',
        'docs/lesson-products/results/c-artifact-manifest.json',
        'docs/lesson-products/results/settings-freeze-v1.json',
        'docs/lesson-products/results/pretest-freeze-record-v1.json',
        'docs/lesson-products/results/final-report-v1.json',
        'docs/lesson-products/results/c-calibration.json',
        'docs/lesson-products/results/observed-failure-examples-v1.json',
        'docs/lesson-products/data-v2-token-audit.json',
    ]
    historic = {p: load(ROOT / p) for p in history_paths}
    train_record = historic[history_paths[0]]
    settings = historic[history_paths[2]]
    pretest = historic[history_paths[3]]
    final = historic[history_paths[4]]
    failure_record = historic[history_paths[6]]
    assert final['settings_freeze_sha256'] == sha(ROOT / history_paths[2]) == pretest['settings_freeze_sha256']
    assert failure_record['report_sha256'] == sha(ROOT / history_paths[4])
    assert failure_record['raw_predictions_sha256'] == final['predictions']['sha256']
    assert settings['recorded_at'] < pretest['recorded_at'] < final['recorded_at']
    assert settings['test_outcomes_observed'] is False
    local_matches = {name: sha(ROOT / 'tiny_perceptron/lesson_products' / name) == value for name, value in settings['source_sha256s'].items()}
    assert all(local_matches.values())
    for question, denominator in [('choice', 11797), ('info', 10890), ('rubric', 10890)]:
        assert final['c'][question]['after']['correct'] == final['c'][question]['after']['count'] == denominator
    assert final['c_reorder'] == {'count': 11797, 'consistent': 11797, 'consistency': 1.0}
    assert final['c_missing_best'] == {'count': 849, 'correct': 849}
    assert [final['shifts']['c_wording'][q]['after']['correct'] for q in ['choice', 'info', 'rubric']] == [11493, 10773, 10773]
    assert settings['calibration']['temperatures'] == {'choice': .25, 'info': .25, 'rubric': .25}
    assert train_record['training']['random_initialization'] is True
    assert train_record['training']['seed'] == 9041
    assert train_record['training']['steps_this_attempt'] == train_record['training']['selected_step'] == 800
    assert train_record['training']['parameters'] == 84545 and train_record['gpu_name'] == 'NVIDIA L4'

    # Original public tiny model, pinned by the repo release manifest. No training.
    product_dir = CACHE / 'product-c'
    release = load(ROOT / 'tiny_perceptron/lesson_products/release.json')
    assert sha(product_dir / 'artifact-manifest.json') == release['products']['c']['manifest_sha256']
    product = Product(product_dir, 'c', device='cpu', verify=True)
    assert state_digest(product.model.state_dict()) == train_record['training']['selected_weights_sha256']
    assert product.config == model.config
    assert product.tokenizer.characters == tokenizer.characters
    failure = failure_record['examples'][0]
    judged = product.judge(failure['state'], 'choice', failure['candidates'])
    assert judged['selected'] == failure['predicted'] == 'boxes' and failure['gold'] == 'none'
    assert math.isclose(judged['concentration'], failure['concentration'], abs_tol=1e-6)
    scores = failure['raw_scores']
    weights = [math.exp((x - max(scores)) / .25) for x in scores]
    probabilities = [w / sum(weights) for w in weights]
    p_max = max(probabilities)
    concentration = (len(scores) * p_max - 1) / (len(scores) - 1)
    assert math.isclose(p_max, failure['probability_selected'], abs_tol=1e-12)
    assert math.isclose(concentration, failure['concentration'], abs_tol=1e-12)
    tied = torch.zeros(4)
    forward_names = ['card', 'address', 'app', 'none']
    reverse_names = list(reversed(forward_names))
    choose = lambda names: min(zip(names, tied.tolist()), key=lambda pair: (-pair[1], pair[0]))[0]
    assert choose(forward_names) == choose(reverse_names) == 'address'

    prerequisites = []
    for page_id, spans in [('9.11', [(3, 5), (36, 44)]), ('9.14', [(23, 25)]), ('9.15', [(3, 11), (22, 30)]), ('19.14', [(40, 55)]), ('19.15', [(54, 78)]), ('19.23', [(3, 9), (50, 107)])]:
        p = next(p for p in freeze['pages'] if p['page_id'] == page_id)
        snapshot = ROOT / p['snapshot']
        assert sha(snapshot) == p['source_sha256']
        lines = snapshot.read_bytes().splitlines(keepends=True)
        slices_meta = [{'lines': f'{a}-{b}', 'sha256': hashlib.sha256(b''.join(lines[a-1:b])).hexdigest()} for a, b in spans]
        prerequisites.append({'page_id': page_id, 'snapshot': p['snapshot'], 'page_sha256': sha(snapshot), 'slices': slices_meta})
    result = {
        'started_at': started_at,
        'completed_at': datetime.now(UTC).isoformat(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git_version': torch.version.git_version, 'device': 'cpu', 'dtype': 'float32', 'threads': str(torch.get_num_threads())},
        'source': {'path': page['snapshot'], 'sha256': sha(path), 'freeze': FREEZE, 'freeze_sha256': sha(ROOT / FREEZE), 'figure_sha256': {}},
        'inline_blocks': [{'lines': loc, 'sha256': hashlib.sha256(s.encode()).hexdigest(), 'executed': True} for loc, s in zip(['17-87', '106-126'], snippets)],
        'inline_stdout': printed,
        'inline_expected_output_exact': True,
        'candidate_scorer_and_prerequisite_block_ast_match_current_library': True,
        'inline_excerpt_output_max_abs_difference': float((actual - expected).abs().max()),
        'permutation_score_max_abs_difference': permutation_difference,
        'permutation_probability_max_abs_difference': probability_difference,
        'additional_masked_padding_max_abs_difference': padding_difference,
        'parameter_count': {'components': components, 'blocks': 2, 'block_parameters_distinct': True, 'total': total},
        'multi_question_padding': {'slices': slices, 'max_candidates': maximum, 'unused_score': '-inf', 'unused_probability': probs[0, 2].item()},
        'candidate_rejections': rejections,
        'rule_probes': rule_probes,
        'historical_record_integrity': {'records': [{'path': p, 'sha256': sha(ROOT / p)} for p in history_paths], 'source_sha256s_match_current': local_matches, 'freeze_before_test': True, 'seed': 9041, 'steps': 800, 'selected_step': 800, 'historical_gpu': train_record['gpu_name'], 'raw_predictions_available_locally': False, 'scope': 'This run checks original historical receipts, hashes and reported values; it does not rerun historical GPU training or the full 56,528,235-byte raw predictions.'},
        'existing_tiny_product_cpu_inference': {'release_revision': release['revision'], 'manifest_sha256': sha(product_dir / 'artifact-manifest.json'), 'model_sha256': sha(product_dir / 'model.pt'), 'state_digest_matches_training_receipt': True, 'config_matches_inline': True, 'state': failure['state'], 'gold': failure['gold'], 'selected': judged['selected'], 'concentration': judged['concentration'], 'probability_selected': judged['probabilities']['boxes'], 'float64_recalculation_from_historical_raw_scores': {'probability_selected': p_max, 'concentration': concentration}, 'scope': 'One preserved tiny-model failure was rerun on CPU; no parameter updates and no new quality/generalization result.'},
        'prerequisite_slices': prerequisites,
        'result': 'All assertions passed; two frozen inline Python blocks executed, prerequisite Block re-executed, small original C failure reproduced on CPU; historical record audit kept separate.',
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': args.output, 'sha256': sha(out), 'result': result['result'], 'inline_stdout': printed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
