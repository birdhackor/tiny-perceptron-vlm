from pathlib import Path
import ast
import collections
import datetime
import hashlib
import json
import platform
import re

ART = Path(__file__).resolve().parent
ROOT = ART.parents[4]
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (ROOT / 'docs/course-experiments/results/ocr.json').read_bytes()
assert sha(raw) == '9b08fb8ad9007160cb5ad4448e924ce1df81c9edab92487035dc71ea71a41fb7'
assert raw == (ART.parent / 'original/raw-ocr.json').read_bytes()
j = json.loads(raw)
pointers = []
def read_pointer(pointer):
    pointers.append(pointer)
    value = j
    for key in pointer.split('/')[1:]:
        value = value[key.replace('~1', '/').replace('~0', '~')]
    return value

split_units = {}
for split in ('train', 'validation', 'test'):
    records = read_pointer('/results/data/splits/' + split + '/records')
    count = read_pointer('/results/data/splits/' + split + '/count')
    records_sha = read_pointer('/results/data/splits/' + split + '/sha256')
    assert count == len(records)
    assert sha(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()) == records_sha
    by_family = collections.defaultdict(list)
    class_frequency = collections.Counter()
    for row in records:
        assert row['family'] == row['digits'] == row['answer']
        by_family[row['family']].append(row['offset'])
        class_frequency.update(row['answer'])
    assert all(sorted(offsets) == [-1, 0, 1] for offsets in by_family.values())
    split_units[split] = {'question_count': count, 'digit_string_family_count': len(by_family),
                          'families': sorted(by_family), 'digit_character_classes': sorted(class_frequency),
                          'digit_character_occurrences': dict(sorted(class_frequency.items())),
                          'records_sha256': records_sha, 'offsets_per_family': [-1, 0, 1]}

train_strings = set(split_units['train']['families'])
test_strings = set(split_units['test']['families'])
train_chars = set(split_units['train']['digit_character_classes'])
test_chars = set(split_units['test']['digit_character_classes'])
assert train_chars == set('0123456789')
assert not train_strings & test_strings
assert not test_chars - train_chars

code = (ART.parent / 'original/fence-1.py').read_bytes()
tree = ast.parse(code)
assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'glyphs' for t in n.targets))
glyphs = ast.literal_eval(assignment.value)
assert set(glyphs) == {'0', '1'}
two_fixture_counterexamples = []
for train_label, heldout_label in [('0', '1'), ('1', '0')]:
    two_fixture_counterexamples.append({
        'train_source_family_and_character_class': train_label,
        'heldout_source_family_and_character_class': heldout_label,
        'heldout_character_class_seen_in_train': False,
        'basis': 'Each original dictionary label has exactly one original glyph; all identity-preserving variants of that glyph remain in that family. This is a hypothetical split consequence, not an executed training/evaluation experiment.'})

markdown = (ROOT / 'course/chapters/11.md').read_bytes()
headings = list(re.finditer(rb'(?m)^## [^\r\n]+', markdown))
i = next(i for i, h in enumerate(headings) if h[0].startswith(b'## 11.12 '))
end = headings[i+1].start() if i+1 < len(headings) else len(markdown)
section = markdown[headings[i].start():end]
assert sha(section) == 'd841a163328c6644676206c3f5194f79f4821ee8804596d30f6fe8d2e799394f'
(ART / 'section-rechecked.md').write_bytes(section)

sources = {}
for path in ['scripts/course_experiments/modalities.py', 'scripts/prepare_ocr.py']:
    raw_code = (ROOT / path).read_bytes()
    sources[path] = {'sha256': sha(raw_code)}
    recorded = read_pointer('/code_sha256/' + path.replace('/', '~1')) if path != 'scripts/prepare_ocr.py' else None
    if recorded is not None:
        assert recorded == sha(raw_code)
    else:
        assert raw_code == (ART.parent / 'original/prepare_ocr-22a0bb1.py').read_bytes()
    parsed = ast.parse(raw_code)
    for n in parsed.body:
        if isinstance(n, ast.FunctionDef) and n.name in {'_manifest', 'run_ocr', 'draw_digits'}:
            sources[path].setdefault('reread_locators', []).append({'function': n.name, 'start': n.lineno,
                                                                  'end': 853 if n.name == 'run_ocr' else n.end_lineno})

result = {
    'reviewer_task': '/root/phase4_factual_coordinator/factual_11_12',
    'checked_at': datetime.datetime.now(datetime.UTC).isoformat(),
    'environment': {'python': platform.python_version(), 'device': 'CPU stdlib-only data inspection; no model or torch imported'},
    'source_sha256': sha(section), 'original_result_sha256': sha(raw),
    'original_fence_sha256': sha(code), 'inspected_json_pointers': pointers,
    'split_units': split_units,
    'actual_test_novelty': {'unseen_full_digit_strings': sorted(test_strings - train_strings),
                          'unseen_digit_character_classes': sorted(test_chars - train_chars),
                          'training_character_class_count': len(train_chars), 'font_changed': False,
                          'font_basis': 'draw_digits selects each character from the same fixed GLYPHS lookup for every split; run_ocr varies entire digit string and offsets, not the font'},
    'hypothetical_two_fixture_family_holdout': two_fixture_counterexamples,
    'source_reread': sources,
    'unit_distinctions': {
        'original_image_or_glyph_source_family': 'An original image/render and its identity-preserving crops/shifts belong together when testing unseen source images; with only one source per class, holding out that source also holds out that class.',
        'font_family': 'Holding out a font/style asks about unseen fonts, which requires independent font sources and class coverage; the original fixed-font experiment has no such axis.',
        'digit_string_family': 'Original OCR run holds out complete digit strings and keeps their three offsets together. Characters 0-9 occur in training; test sequences are unseen combinations using those familiar character classes and fixed font.'},
    'scope': 'Fresh raw data/AST method reread and exact combinatorial check of split units only. No new training, model inference, model/data downloads, GPU work or neural .pt file.'
}
(ART / 'split-unit-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
