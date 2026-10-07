"""Owner audit of original raw routing outputs; no model training or generation."""
import json
from pathlib import Path
from collections import Counter
from scripts.course_experiments.applications import _prompt_ids
from scripts.course_experiments.tool_choice import build_records, split_records
from scripts.course_experiments.common import records_sha256
from tiny_perceptron.data import ByteTokenizer

raw = json.loads(Path('docs/course-experiments/results/tool_choice.json').read_text())
result = raw['results']
tokenizer = ByteTokenizer()
splits = split_records(build_records(), result['reproducibility']['seed'])
test_families = {row['family'] for row in splits['test']}
diagnostic = [row for row in build_records(paraphrase=True) if row['family'] in test_families]
output = {}
for name, records in [('validation', splits['validation']), ('test', splits['test']), ('paraphrase_diagnostic', diagnostic)]:
    samples = result[name]['samples']
    assert len(records) == len(samples)
    assert [sample['record'] for sample in samples] == records
    rows = []
    for record, sample in zip(records, samples, strict=True):
        generation = sample['generation']
        model_output = generation['samples'][0]
        ids = model_output['generated_ids']
        assert tokenizer.decode(ids) == model_output['generated']
        assert _prompt_ids(record['messages'][:-1], tokenizer) == generation['input_ids']
        invalid = [value for value in ids[:-1] if value < 8]
        valid = ids[-1] == tokenizer.eos_id and not invalid and model_output['generated'] in ('DIRECT', 'TOOL', 'ASK')
        predicted = model_output['generated'] if valid else None
        intent, available = record['intent'], record['calculator_available']
        expected = ('TOOL' if available else 'ASK') if intent == 'numerical_addition' else ('ASK' if intent == 'missing_quantity' else 'DIRECT')
        assert expected == record['expected_action']
        rows.append({'family': record['family'], 'question': record['messages'][1]['content'], 'available': available, 'intent': intent, 'expected': expected, 'predicted': predicted, 'correct': expected == predicted, 'eos': ids[-1] == 2, 'invalid_control_count': len(invalid)})
    needed = [row for row in rows if row['expected'] == 'TOOL']
    not_needed = [row for row in rows if row['expected'] != 'TOOL']
    unavailable = [row for row in rows if not row['available']]
    output[name] = {
        'count': len(rows), 'records_sha256': records_sha256(records),
        'records_sha_matches': records_sha256(records) == result['data'][name]['sha256'],
        'correct': sum(row['correct'] for row in rows), 'valid': sum(row['predicted'] is not None for row in rows),
        'needed': len(needed), 'needed_selected': sum(row['predicted'] == 'TOOL' for row in needed),
        'needed_missed': sum(row['predicted'] != 'TOOL' for row in needed),
        'not_needed': len(not_needed), 'extra': sum(row['predicted'] == 'TOOL' for row in not_needed),
        'unavailable': len(unavailable), 'unavailable_tool_selected': sum(row['predicted'] == 'TOOL' for row in unavailable),
        'eos': sum(row['eos'] for row in rows), 'invalid_control_count': sum(row['invalid_control_count'] for row in rows),
        'by_intent': {intent: {'correct': sum(row['correct'] for row in rows if row['intent'] == intent), 'count': sum(row['intent'] == intent for row in rows)} for intent in sorted({row['intent'] for row in rows})},
        'always_direct': sum(row['expected'] == 'DIRECT' for row in rows),
        'always_tool': sum(row['expected'] == 'TOOL' for row in rows),
        'mismatches': [row for row in rows if not row['correct']],
        'pair_1_2': [row for row in rows if row['family'] == 'pair:1:2'],
    }
sets = {name: {row['family'] for row in records} for name, records in splits.items()}
output['family_overlaps'] = {a + '-' + b: sorted(sets[a] & sets[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
output['single_digits_in_train'] = sorted({digit for row in splits['train'] for digit in row['operands']})
output['scope'] = 'Original historical raw generated IDs, prompts, and deterministic datasets/policy; no saved metric/chosen_action/unchanged conclusion used, no new model generation.'
print(json.dumps(output, ensure_ascii=False, indent=2))
