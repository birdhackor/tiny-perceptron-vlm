"""Own exact-match checks of only currently read RAG paired conditions."""
import collections
import json
import random
from pathlib import Path

from scripts.course_experiments.applications import _prompt_ids, _rag_splits, _split_summary
from tiny_perceptron.data import ByteTokenizer

raw = json.loads(Path('docs/course-experiments/results/rag.json').read_text())['results']
splits = _rag_splits(raw['seed'])
assert _split_summary(splits) == raw['split']
keys = {name: {row['key'] for row in rows} for name, rows in splits.items()}
assert not (keys['train'] & keys['test'])
rng = random.Random(raw['seed'] + 100)
facts = []
for key in sorted(keys['test']):
    source = f'D{rng.randrange(10)}'
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    facts.append({'family': key, 'key': key, 'address': address, 'source': source})
tokenizer = ByteTokenizer()
rows = [row for row in raw['samples'] if row['mode'] in {'correct_context', 'changed_address_context', 'changed_source_context'}]
by_mode = collections.defaultdict(dict)
records = []
for row in rows:
    fact = row['context_fact']
    expected = f"{fact['address']}[{fact['source']}]"
    sample = row['generation']['samples'][0]
    assert _prompt_ids(row['generation']['messages'], tokenizer) == row['generation']['input_ids']
    assert tokenizer.decode(sample['generated_ids']) == sample['generated']
    assert sample['generated_ids'][-1] == tokenizer.eos_id and sample['eos']
    assert sample['generated_tokens'] == len(sample['generated_ids'])
    assert row['generation']['temperature'] == 0.0
    assert expected == row['expected']
    if row['mode'] == 'correct_context':
        assert row['fact'] == next(fact for fact in facts if fact['key'] == row['family'])
    correct = sample['generated'] == expected
    by_mode[row['mode']][row['family']] = correct
    records.append({'family': row['family'], 'mode': row['mode'], 'expected': expected, 'actual': sample['generated'], 'correct': correct, 'EOS': True})
baseline = by_mode['correct_context']
print(json.dumps({
    'scope': 'Original historical raw generation ID decoding and exact comparisons, no new training or model generation',
    'split_recreated_hash_matches': True,
    'family_counts': {name: len(values) for name, values in keys.items()},
    'test_keys_absent_from_training': True,
    'counts': {mode: [sum(values.values()), len(values)] for mode, values in by_mode.items()},
    'paired_both_correct': {mode: [sum(baseline[key] and correct for key, correct in values.items()), len(values)] for mode, values in by_mode.items() if mode != 'correct_context'},
    'rows': records,
}, ensure_ascii=False, indent=2))
