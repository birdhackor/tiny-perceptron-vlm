"""T.11 reviewer-owned read-only inspection; no training or final NN inference."""

import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
CORE = Path('/workspace/work/omni-core-completed-40988311100/verified')
FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')
CACHE = Path('/workspace/work/omni-P5-T11-review-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(name, value):
    path = HERE / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def decode_input(ids):
    assert ids[0] == 1 and ids[-1] == 4
    cursor, messages = 1, []
    roles = {3: 'user', 4: 'assistant', 7: 'system'}
    while cursor < len(ids) - 1:
        role = roles[ids[cursor]]
        end = ids.index(2, cursor + 1)
        messages.append({'role': role, 'content': bytes(x - 8 for x in ids[cursor+1:end]).decode('utf-8')})
        cursor = end + 1
    assert cursor == len(ids) - 1
    return messages


def softmax(logits, temperature):
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    e = [math.exp(x - peak) for x in scaled]
    return [x / sum(e) for x in e]


def best_id(row):
    return min(zip(row['ids'], row['logits']), key=lambda x: (-x[1], x[0]))[0]


def main():
    env = {'python': sys.version, 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'CPU; checkpoint tensors inspected only'}
    freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v3.json'
    page = next(x for x in read(freeze_path)['pages'] if x['page_id'] == 'T.11')
    snapshot = ROOT / page['snapshot']
    assert sha(snapshot) == page['source_sha256']
    current = (ROOT / 'course/training.md').read_text()
    current = current[current.index('## T.11 '):]
    assert hashlib.sha256(current.encode()).hexdigest() == page['source_sha256']
    links = []
    for link in re.findall(r'\]\(([^)]+)\)', snapshot.read_text()):
        path, _, anchor = link.partition('#')
        target = (ROOT / 'course' / path).resolve()
        assert target.is_file(), target
        if anchor and re.fullmatch(r'[A-Z\d]+\.\d+', anchor):
            assert re.search(r'^## ' + re.escape(anchor) + r' ', target.read_text(), re.M), (target, anchor)
        links.append({'link': link, 'target': target.relative_to(ROOT).as_posix(), 'target_sha256': sha(target), 'numbered_anchor_present': bool(anchor)})
    inventory = {'created_at': datetime.now(timezone.utc).isoformat(), 'scope': 'Only numbered T.11 including its details block; link destinations inspected as evidence, not independently reviewed lessons.', 'freeze': {'path': freeze_path.relative_to(ROOT).as_posix(), 'sha256': sha(freeze_path)}, 'snapshot': {'path': page['snapshot'], 'bytes': snapshot.stat().st_size, 'sha256': sha(snapshot)}, 'figure_sha256': page['figures_sha256'], 'fenced_blocks': re.findall(r'```(\w*)', snapshot.read_text()), 'links': links}
    outputs = [save('source-inventory.json', inventory)]

    selected_path = ROOT / 'docs/omni-products/v1/model-selection.json'
    selected = read(selected_path)
    recipes = read(ROOT / 'docs/omni-products/v1/formal-recipes.json')
    jobs = []
    required_keys = {'model', 'optimizer', 'step', 'python_rng', 'torch_rng', 'cuda_rng', 'sampler_rng', 'metadata', 'auxiliary'}
    for name in recipes['job_order']:
        job_dir = CORE / 'core' / name
        result_path = job_dir / 'job-result.json'
        result = read(result_path)
        train_path = job_dir / 'training/training-result.json'
        train = read(train_path)
        assert result['success'] and train['complete']
        assert train['updates'] == recipes['jobs'][name]['training']['updates']
        assert train['optimizer_updates_observed'] and train['initial_parameter_sha256'] != train['latest_parameter_sha256']
        assert sha(train_path) == result['training_result_sha256']
        assert result['runtime']['device'] == 'cpu' and result['runtime']['torch'] == '2.14.1+cpu'
        assert result['execution_host']['run_id'] == '40988311100'
        assert result['attempt'] == 1 and not result['final_opened']
        assert result['formal_recipe_sha256'] == sha(ROOT / 'docs/omni-products/v1/formal-recipes.json')
        assert selected['all_formal_jobs'][name]['job_result_sha256'] == sha(result_path)
        file_rows, states = {}, {}
        for fn, binding in train['files'].items():
            p = train_path.parent / fn
            assert p.stat().st_size == binding['bytes'] and sha(p) == binding['sha256']
            file_rows[fn] = binding
        for fn, expected_step in [('best-continuation.pt', train['best']['update']), ('latest-continuation.pt', train['updates'])]:
            state = torch.load(train_path.parent / fn, weights_only=True, map_location='cpu')
            assert required_keys <= state.keys() and state['step'] == expected_step
            assert state['optimizer']['state'] and state['python_rng'] and state['torch_rng'].numel() > 0 and state['sampler_rng'].numel() > 0
            assert state['metadata']['implementation_sha256'] == train['implementation_sha256']
            states[fn] = {'step': state['step'], 'keys': sorted(state), 'optimizer_parameter_states': len(state['optimizer']['state']), 'torch_rng_elements': state['torch_rng'].numel(), 'sampler_rng_elements': state['sampler_rng'].numel(), 'python_rng_saved': True, 'loaded_weights_only': True}
            del state
        jobs.append({'job': name, 'kind': result['kind'], 'job_result_sha256': sha(result_path), 'training_result_sha256': sha(train_path), 'runtime': result['runtime'], 'started_at': result['started_at'], 'finished_at': result['finished_at'], 'updates': train['updates'], 'selected_update': train['best']['update'], 'optimizer_updates_recorded': train['optimizer_updates_observed'], 'initial_and_latest_parameter_digests_differ': True, 'files_independently_hashed': file_rows, 'checkpoint_contents': states})
    assert len(jobs) == 10
    model_files = {}
    for kind in ('av', 'choice', 'codec', 'talker'):
        p = CORE / 'core' / selected['selected'][kind] / 'training/weights.safetensors'
        binding = selected['model_files'][kind]['weights']
        assert p.stat().st_size == binding['bytes'] and sha(p) == binding['sha256']
        model_files[kind] = {'job': selected['selected'][kind], **binding}
    total = sum(x['bytes'] for x in model_files.values())
    assert total == 2192940
    opening_path = ROOT / 'docs/omni-products/v1/final-opening.json'
    opening = read(opening_path)
    assert selected['created_at'] < opening['created_at'] and not selected['final_opened']
    assert opening['model_selection_sha256'] == sha(selected_path)
    assert sha(FINAL / 'frozen/model-selection.json') == sha(selected_path)
    assert sha(FINAL / 'frozen/final-opening.json') == sha(opening_path)
    binding_checks = {}
    for name, digest in selected['implementation_sha256'].items():
        binding_checks['tiny_perceptron/omni_products/' + name] = sha(ROOT / 'tiny_perceptron/omni_products' / name) == digest
    assert all(binding_checks.values())
    core = {'environment': env, 'reviewer_operation': 'Read all ten original job/training receipts; independently hash their files; weights_only read of 20 saved state dictionaries; no model construction, optimizer step, resumption, training or final inference.', 'source_run': 40988311100, 'jobs': jobs, 'counts': {kind: sum(x['kind'] == kind for x in jobs) for kind in ['av', 'choice', 'codec', 'talker']}, 'selected_inference_files': model_files, 'total_selected_safetensors_bytes': total, 'MiB_divisor': 1048576, 'total_selected_safetensors_MiB': total / 1048576, 'measurement_scope': 'Four complete safetensors files only; Python process RAM, torch working buffers, continuation files and downloaded ZIP bytes excluded.', 'selection': {'sha256': sha(selected_path), 'created_at': selected['created_at'], 'temperature': selected['temperature'], 'threshold': selected['threshold'], 'opening_sha256': sha(opening_path), 'opening_created_at': opening['created_at'], 'selection_and_opening_equal_original_final_bundle_bytes': True, 'implementation_binding_matches_current': binding_checks}, 'limits': 'Recorded fixed run and hashes support no post-final reselection within these results; no claim to prove the absence of unrecorded external experiments, cross-device bit exact continuation, or native Windows execution.'}
    outputs.append(save('core-inspection.json', core))

    choice_path = FINAL / 'measurements/choice/results.json'
    raw = read(choice_path)['primary']
    rows = raw['rows']
    assert len(rows) == 60 and len({x['family'] for x in rows}) == 60
    counts = {kind: sum(best_id(x['tasks'][kind]) == x['tasks'][kind]['gold'] for x in rows) for kind in ('choice', 'info', 'rubric')}
    decisions = []
    maximum_probability_error = 0.0
    for x in rows:
        c, info = x['tasks']['choice'], x['choice_info']
        probabilities = softmax(c['logits'], selected['temperature'])
        maximum_probability_error = max(maximum_probability_error, max(abs(a-b) for a,b in zip(probabilities,c['probabilities'])))
        answer = best_id(c)
        accepted = max(probabilities) >= selected['threshold'] and answer != 'none' and best_id(info) == 'yes'
        error = accepted and (answer != c['gold'] or not x['world_supported'] or not x['choice_information_sufficient'])
        decisions.append({'family': x['family'], 'answer': answer, 'gold': c['gold'], 'p_max': max(probabilities), 'accepted': accepted, 'error': error})
    accepted, errors = sum(x['accepted'] for x in decisions), sum(x['error'] for x in decisions)
    assert counts == {'choice': 39, 'info': 55, 'rubric': 35} and (accepted, errors) == (53, 19)
    assert maximum_probability_error < 1e-12
    metrics_path = ROOT / 'docs/omni-products/v1/metrics.json'
    metrics = read(metrics_path)['C']
    assert counts['choice'] < metrics['choice_minimum_correct']
    assert counts['rubric'] < metrics['rubric_minimum_correct']
    choice = {'environment': env, 'operation': 'Pure Python stable-ID argmax, exp/logit softmax and acceptance arithmetic from preserved original rows; no neural forward or parameter selection.', 'source': {'archive_url': 'https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/omni-final-candidate-cpu-final-41004229262-1/final-results.zip', 'archive_sha256': '50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3', 'member': 'measurements/choice/results.json', 'member_sha256': sha(choice_path)}, 'selection_sha256': sha(selected_path), 'metrics_sha256': sha(metrics_path), 'temperature': selected['temperature'], 'threshold': selected['threshold'], 'families': len(rows), 'counts': counts, 'requirements': {key: val for key,val in metrics.items() if key in ['choice_minimum_correct', 'info_minimum_correct', 'rubric_minimum_correct', 'minimum_accepted', 'maximum_empirical_risk']}, 'accepted': accepted, 'accepted_errors': errors, 'risk': errors / accepted, 'maximum_probability_error_vs_saved': maximum_probability_error, 'decisions': decisions, 'conclusion': 'Fixed C fails choice/rubric and empirical accepted-risk criteria. The example command cannot promote it.'}
    outputs.append(save('choice-recomputation.json', choice))

    tools_path = ROOT / 'docs/course-experiments/results/tools.json'
    tools = read(tools_path)['results']
    records = []
    for key in ['samples', 'one_step_cap_samples']:
        for i, episode in enumerate(tools[key]):
            prior = None
            for k, event in enumerate(episode['trace']):
                if 'generation' not in event:
                    continue
                g = event['generation']
                messages = decode_input(g['input_ids'])
                assert len(g['input_ids']) == g['input_tokens']
                if prior is None:
                    assert len(messages) == 2 and messages[1]['content'] == episode['question']
                else:
                    assert messages == prior
                records.append({'pointer': f'/results/{key}/{i}/trace/{k}/generation', 'input_tokens': len(g['input_ids']), 'input_ids_sha256': digest_json(g['input_ids']), 'decoded_messages_sha256': digest_json(messages), 'saved_messages_match_actual_ids': messages == g['messages']})
                if event.get('executed') and event.get('finite_result'):
                    prior = messages + [{'role': 'assistant', 'content': g['samples'][0]['generated']}, {'role': 'user', 'content': 'TOOL_RESULT:' + str(event['tool_result'])}]
    mismatches = sum(not x['saved_messages_match_actual_ids'] for x in records)
    assert len(records) == 56 and mismatches == 36
    historical = {'environment': env, 'operation': 'Independently decode actual saved UTF-8 byte IDs; compare message metadata and verify chronological tool-result insertion. No inference or reconstruction-file writes.', 'tools_source_sha256': sha(tools_path), 'normal_episodes': len(tools['samples']), 'one_step_cap_episodes': len(tools['one_step_cap_samples']), 'generations': len(records), 'messages_metadata_mismatches': mismatches, 'actual_input_and_step_transitions_all_checked': True, 'records': records}
    rag = read(ROOT / 'docs/course-experiments/results/rag.json')['results']
    assert all('input_ids' in x['generation'] and 'samples' in x['generation'] and 'expected' in x for x in rag['samples'])
    historical['rag'] = {'source_sha256': sha(ROOT / 'docs/course-experiments/results/rag.json'), 'saved_samples': len(rag['samples']), 'samples_all_have_actual_inputs_outputs_targets': True, 'test_examples': rag['test']['examples'], 'per_condition_explicit_denominators': {k: v['exact_match']['denominator'] for k,v in rag['metrics'].items()}}
    reasoning = read(ROOT / 'docs/course-experiments/results/reasoning.json')['results']
    historical['reasoning'] = {'source_sha256': sha(ROOT / 'docs/course-experiments/results/reasoning.json'), 'comparison_branches': {k: {'test_examples': v['test']['examples'], 'candidate_counts': [x['candidate_count'] for x in v['budgets']], 'saved_samples_per_budget': [len(x['samples']) for x in v['budgets']], 'oracle_denominators': [x['oracle_coverage']['denominator'] for x in v['budgets']]} for k,v in reasoning['comparison'].items()}, 'limits': reasoning['limits']}
    outputs.append(save('historical-record-inspection.json', historical))

    toy = {'operation': 'Transparent arithmetic for labelled hypothetical values; not model measurement.', 'original_bytes': 80000, 'new_bytes': 60000, 'saved_bytes': 80000-60000, 'size_reduction': (80000-60000)/80000, 'original_correct_fraction': '4/5', 'new_correct_fraction': '3/5', 'original_accuracy': 4/5, 'new_accuracy': 3/5, 'size_criterion': 60000 < 80000, 'quality_criterion': 3 >= 4, 'joint_criterion': 60000 < 80000 and 3 >= 4, 'same_total_distinct_errors_example': {'targets': ['a','b','c','d','e'], 'version_one_correct': [True,True,True,True,False], 'version_two_correct': [False,True,True,True,True]}, 'denominator_example': 'Dropping an incorrect fifth question converts 4/5 to 4/4 without fixing a prediction; the two fractions answer different evaluation questions.'}
    outputs.append(save('arithmetic.json', toy))
    print(json.dumps({'outputs': outputs, 'environment': env, 'operation_completed_at': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
