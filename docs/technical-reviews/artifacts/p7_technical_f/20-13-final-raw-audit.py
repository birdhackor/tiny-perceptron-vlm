"""Own recomputation from original historical generations/transcripts/grades.

Historical manual passed fields are measurement inputs, not this reviewer's
semantic judgments. No scored scores, closure or audit summaries are read.
"""
import collections
import json
import unicodedata
from pathlib import Path

root = Path('docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review')
generations = json.loads((root / 'generations-base.json').read_text())
transcripts = json.loads((root / 'transcripts.json').read_text())
grades = json.loads((root / 'combined/grades.json').read_text())['grades']
mapping = json.loads((root / 'private-map.json').read_text())['aliases']
subsets = json.loads((root / 'descriptive-subsets.json').read_text())
assert mapping == {'A': 'base'}
passed = {row['case_id']: row['passed'] for row in grades if row['candidate'] == 'A'}
assert len(passed) == len(grades) == 147

def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]

def normal(s):
    return ''.join(unicodedata.normalize('NFKC', s).split())

counts = collections.defaultdict(lambda: [0, 0])
incomplete = []
for row in generations:
    ids = row['generated_token_ids']
    assert len(ids) == row['generated_tokens']
    ended = bool(ids and ids[-1] in row['eos_token_ids'])
    assert ended == row['ended_with_eos']
    assert (row['stop_reason'] == 'eos') == ended
    assert row['generated_tokens'] <= 384
    complete = ended and not row['truncated'] and not row['completion_unknown']
    task = row['task']
    key = task
    if task == 'scene':
        key = 'scene' if row['id'].endswith('/scene') else 'fact'
    if task in {'scene', 'chat', 'typed_chat', 'speech_chat'}:
        ok = passed[task + ':' + row['id']] and complete
    elif task in {'ocr', 'text_presence'}:
        ok = normal(row['prediction']) == normal(row['reference_answer']) and complete
    elif task == 'ocr_order':
        ok = unicodedata.normalize('NFKC', row['prediction']).strip() == unicodedata.normalize('NFKC', row['reference_answer']).strip() and complete
    else:
        raise AssertionError(task)
    counts[key][0] += int(ok)
    counts[key][1] += 1
    if not complete:
        incomplete.append({'id': row['id'], 'task': task, 'tokens': row['generated_tokens'], 'ended_with_eos': ended})

raw_edits = raw_chars = normalized_edits = normalized_chars = 0
for row in transcripts:
    reference, prediction = row['reference_transcript'], row['transcript']
    raw_edits += distance(reference, prediction)
    raw_chars += len(reference)
    normalized_edits += distance(normal(reference), normal(prediction))
    normalized_chars += len(normal(reference))
    assert normal(reference) == row['normalized_reference']
    assert normal(prediction) == row['normalized_prediction']

verified_subsets = {}
for label in ['action', 'relation']:
    cases = subsets['photo_fact_subsets'][label]['cases']
    assert all(passed[c['case_id']] == c['passed'] for c in cases)
    verified_subsets[label] = {'correct': sum(passed[c['case_id']] for c in cases), 'denominator': len(cases), 'case_ids': [c['case_id'] for c in cases]}
context_cases = subsets['context_dependent_text_chat']['cases']
assert all(passed[c['case_id']] == c['passed'] for c in context_cases)
presence = [row for row in generations if row['task'] == 'text_presence']
result = json.loads((root / 'result.json').read_text())
print(json.dumps({
    'scope': 'Own CPU aggregation of original historical manual grades and deterministic metrics; not new semantic grading, training, ASR or generation',
    'historical_counts': dict(counts),
    'answers': len(generations), 'completed': len(generations) - len(incomplete), 'incomplete': incomplete,
    'subsets_from_frozen_case_ids': verified_subsets,
    'context_chat': {'correct': sum(passed[c['case_id']] for c in context_cases), 'denominator': len(context_cases)},
    'presence_reference': dict(collections.Counter(r['reference_answer'] for r in presence)),
    'ASR_sources': dict(collections.Counter(r['source'] for r in transcripts)),
    'ASR_raw': {'edits': raw_edits, 'reference_characters': raw_chars, 'percent': round(100 * raw_edits / raw_chars, 2)},
    'ASR_normalized': {'edits': normalized_edits, 'reference_characters': normalized_chars, 'percent': round(100 * normalized_edits / normalized_chars, 2)},
    'runtime': {k: result[k] for k in ['model', 'model_revision', 'asr_model', 'asr_revision', 'total_parameters', 'trainable_parameters', 'split', 'selected_only', 'max_tokens', 'seed']},
}, ensure_ascii=False, indent=2))
