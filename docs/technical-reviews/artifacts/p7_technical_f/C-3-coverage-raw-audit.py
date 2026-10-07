"""Recompute historical coverage from raw IDs/text, ignoring stored grades."""
import json
import re
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
from scripts.course_experiments.applications import _prompt_ids

raw = json.loads(Path('docs/course-experiments/results/reasoning.json').read_text())['results']
tokenizer = ByteTokenizer()
output = {}
for mode, result in raw['comparison'].items():
    output[mode] = []
    for budget in result['budgets']:
        k = budget['candidate_count']
        rows = []
        candidates_total = eos_total = control_total = correct_candidates = 0
        for sample in budget['samples']:
            a, b, c = map(int, re.fullmatch(r'\((\d+)\+(\d+)\)\+(\d+)=\?', sample['question']).groups())
            truth = a + b + c
            assert len(sample['candidates']) == k
            generation = sample['generation']
            assert _prompt_ids([{'role': 'user', 'content': sample['question']}], tokenizer) == generation['input_ids']
            assert generation['candidate_count'] == k and generation['temperature'] == 0.7
            assert generation['max_new_tokens'] == (8 if mode == 'direct' else 48)
            finals, correct = [], []
            for candidate in sample['candidates']:
                ids, text = candidate['generated_ids'], candidate['generated']
                assert tokenizer.decode(ids) == text
                assert len(ids) <= generation['max_new_tokens']
                invalid = [value for value in ids if value < 8 and value != tokenizer.eos_id]
                stripped = text.strip()
                match = re.fullmatch(r'-?[0-9]+', stripped) if mode == 'direct' else re.search(r';answer=(-?[0-9]+)$', stripped)
                final = int(match[0] if mode == 'direct' else match[1]) if match and not invalid else None
                finals.append(final)
                correct.append(final == truth)
                candidates_total += 1
                eos_total += bool(ids and ids[-1] == tokenizer.eos_id)
                control_total += len(invalid)
                correct_candidates += final == truth
            rows.append({'question': sample['question'], 'truth': truth, 'finals': finals, 'contains_correct_final': any(correct)})
        output[mode].append({'k': k, 'sets': len(rows), 'coverage': sum(row['contains_correct_final'] for row in rows), 'candidates': candidates_total, 'correct_final_candidates': correct_candidates, 'eos_candidates': eos_total, 'invalid_control_tokens': control_total, 'rows': rows})
output['scope'] = 'Original 720 candidate raw IDs/text and metadata; own final parsing and arithmetic. Coverage only; no stored oracle/majority/verifier flags used, no new generation/training.'
print(json.dumps(output, ensure_ascii=False, indent=2))
