"""Bounded CPU audit: original fence, edit variants, raw provenance and route contract.

No ASR/chat model is loaded. Contract check substitutes a manual ASR string and
captures generation inputs; it is not a new model-quality measurement.
"""
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as core
from tiny_perceptron import natural_ui as ui
from tiny_perceptron.natural_concepts import text_error_report


def read(name):
    return json.loads((OUT / 'inputs' / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


environment = {
    'python': sys.version,
    'torch': importlib.metadata.version('torch'),
    'numpy': importlib.metadata.version('numpy'),
    'pillow': importlib.metadata.version('pillow'),
    'device': 'CPU',
    'platform': platform.platform(),
    'models_loaded': False,
}
(OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
print('ENVIRONMENT', json.dumps(environment))
print('ORIGINAL_FENCE')
exec(compile((OUT / 'fence-1.py').read_bytes(), 'course/chapters/20.md#20.12:fence-1', 'exec'))
reference = '請推薦不辣的晚餐。'
variants = [
    (reference, '請推薦辣的晚餐。', 1, 9),
    (reference, '請推薦不辣的晚餐！', 1, 9),
    (reference, reference, 0, 9),
    ('甲', '甲乙丙', 2, 1),
    ('', '甲', 1, 0),
]
for original, prediction, edits, count in variants:
    observed = text_error_report(original, prediction)
    assert observed['edits'] == edits
    assert observed['reference_characters'] == count
    assert observed['cer'] == (edits / count if count else None)
    assert observed['exact'] == (original == prediction)
    print('EDIT_VARIANT', json.dumps(observed, ensure_ascii=False))

manifest = read('docs/natural-assistant/v4/manifest.json')
source = read('docs/natural-assistant/v4/data/voice-question-sources.json')
prefix = 'docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/'
transcripts = read(prefix + 'transcripts.json')
generations = read(prefix + 'generations-base.json')
result = read(prefix + 'result.json')
scores = read(prefix + 'scored/scores.json')
grades = read(prefix + 'combined/grades.json')
selection = read('docs/natural-assistant/v4/selection.json')
asr_selection = read('docs/natural-assistant/v4/asr-selection.json')
binding = scores['artifact_binding']
bindings = {
    'manifest_sha256': 'docs/natural-assistant/v4/manifest.json',
    'protocol_sha256': 'docs/natural-assistant/v4/validation-protocol-lower-lr.json',
    'selection_sha256': 'docs/natural-assistant/v4/selection.json',
    'asr_selection_sha256': 'docs/natural-assistant/v4/asr-selection.json',
    'test_result_sha256': prefix + 'result.json',
    'transcripts_sha256': prefix + 'transcripts.json',
    'manual_grades_sha256': prefix + 'combined/grades.json',
}
for key, filename in bindings.items():
    expected = binding[key] if key in binding else scores[key]
    assert sha(OUT / 'inputs' / filename) == expected
assert sha(OUT / 'inputs' / prefix / 'generations-base.json') == binding['generation_file']['sha256']
assert selection['selected_variant'] == binding['selected_variant'] == 'base'
assert binding['model'] == result['model'] == selection['base_model']['model']
assert binding['model_revision'] == result['model_revision'] == selection['base_model']['revision']
assert result['asr_model'] == asr_selection['model']
assert result['asr_revision'] == asr_selection['revision']
assert result['execution']['selection_sha256'] == binding['selection_sha256']
assert result['execution']['revision'] == binding['execution_revision']
assert result['selected_only'] is True and result['status'] == 'completed'
for item in binding['runtime_sources']:
    assert sha(OUT / 'inputs' / item['path']) == item['sha256'] if (OUT / 'inputs' / item['path']).exists() else sha(ROOT / item['path']) == item['sha256']
for path, item in binding['scoring_sources'].items():
    assert sha(OUT / 'inputs' / path) == item['sha256']
print('BINDINGS', json.dumps({'selected_variant': binding['selected_variant'], 'model': result['model'], 'model_revision': result['model_revision'], 'asr_model': result['asr_model'], 'asr_revision': result['asr_revision'], 'execution_revision': binding['execution_revision'], 'matched_file_bindings': len(bindings) + 1}))

questions = [r for r in manifest['audio_rows'] if r['split'] == 'test' and r['task'] == 'speech_chat']
assert len(questions) == 4
assert len(transcripts) == 22
assert Counter(r['task'] for r in transcripts) == {'speech_transcription': 18, 'speech_chat': 4}
assert scores['denominators']['voice_typed_reference_chat'] == scores['denominators']['voice_actual_asr_chat'] == 4
upstream = dict(line.split(maxsplit=1) for line in (OUT / 'sources/aishell1-transcript-first-1MiB.raw.txt').read_text().splitlines()[:-1])
copies = []
for row in questions:
    source_row = next(r for r in source['audio_rows'] if r['id'] == row['id'])
    assert row['user'] == ''.join(upstream[row['source_utterance_id']].split())
    assert row['user'] == source_row['user']
    assert row['references']['rubric'] == source_row['semantic_rubric']
    assert row['answer'] is None and source_row['answer'] is None
    assert row['synthetic'] is False and row['history'] == []
    audio = ROOT / source['audio_root'] / source_row['audio']
    assert sha(audio) == row['sha256'] == source_row['sha256']
    retained = OUT / 'raw-audio' / audio.name
    retained.parent.mkdir(exist_ok=True)
    retained.write_bytes(audio.read_bytes())
    assert sha(retained) == sha(audio)
    with wave.open(str(retained), 'rb') as wav:
        assert wav.getframerate() == row['sample_rate'] == 16000
        assert wav.getnchannels() == row['channels'] == 1
        assert wav.getnframes() == row['num_samples']
    copies.append({'original_path': str(audio.relative_to(ROOT)), 'retained_path': str(retained.relative_to(ROOT)), 'sha256': sha(retained), 'bytes': retained.stat().st_size, 'wav_frames': row['num_samples']})
    transcript = next(r for r in transcripts if r['id'] == row['id'])
    assert transcript['audio_sha256'] == row['sha256']
    assert transcript['reference_transcript'] == row['user']
    typed = next(r for r in generations if r['id'] == row['id'] and r['task'] == 'typed_chat')
    spoken = next(r for r in generations if r['id'] == row['id'] and r['task'] == 'speech_chat')
    assert typed['user'] == row['user']
    assert spoken['user'] == spoken['transcript'] == transcript['transcript']
    assert spoken['reference_user'] == row['user']
    assert typed['family'] == spoken['family'] == row['family']
    assert typed['image'] == spoken['image'] == row.get('image')
    assert typed['variant'] == spoken['variant'] == 'base'
    assert typed['generated_token_ids'] == spoken['generated_token_ids']
    assert typed['prediction'] == spoken['prediction']
    assert transcript['raw_errors'] == core.edit_distance(row['user'], transcript['transcript']) == 0
    pair = {'id': row['id'], 'reference_characters': len(row['user']), 'asr_exact': True, 'identical_generated_suffix': True}
    for task in ('typed_chat', 'speech_chat'):
        key = task + ':' + row['id']
        grade = next(r for r in grades['grades'] if r['case_id'] == key)
        decision = next(r for r in scores['case_decisions'] if r['case_id'] == key)
        record = typed if task == 'typed_chat' else spoken
        eos = record['generated_token_ids'][-1] in record['eos_token_ids'] and record['ended_with_eos'] and not record['truncated']
        assert decision['decoder_complete'] == eos
        assert decision['rubric_passed'] == grade['passed']
        assert decision['passed'] == (eos and grade['passed'])
        pair[task + '_original_annotation_passed'] = grade['passed']
    print('RAW_PAIR_AUDIT', json.dumps(pair, ensure_ascii=False))
(OUT / 'raw-audio-copy-manifest.json').write_text(json.dumps(copies, indent=2) + '\n')
print('DENOMINATORS', json.dumps({'paired_recordings': 4, 'typed_responses': 4, 'spoken_responses': 4, 'paired_reference_codepoints': sum(len(r['user']) for r in questions), 'all_asr_recordings': 22, 'human_grade_scores_not_retraining': True}))

# Execute real run_evaluate orchestration with a bounded handwritten fixture.
row = {'id': 'manual-route-check', 'split': 'test', 'task': 'speech_chat', 'audio': 'raw-audio/aishell1-BAC009S0031W0414.wav', 'user': reference, 'system': 'same system', 'history': [{'role': 'user', 'content': 'same history'}], 'image': None}
calls = []
def capture(model, processor, entered, data_root, options):
    calls.append({'model_object_id': id(model), 'processor_object_id': id(processor), 'user': entered['user'], 'system': entered['system'], 'history': entered['history'], 'image': entered['image'], 'task': entered['task']})
    return {'task': entered['task']}
options = SimpleNamespace(selected_only=True, comparison_adapters=[], max_seconds=10, manifest='fixture', data_root=OUT, split='test', adapter=None, output=OUT / 'route-fixture-output')
manual_asr = '請推薦辣的晚餐。'
with patch.multiple(core, load_manifest=lambda *_: ({'rows': [], 'audio_rows': [row]}, OUT), load_core=lambda *_args, **_kwargs: (object(), object()), provenance=lambda *_: {}, parameter_counts=lambda *_: {}, write_json=lambda *_: None, load_asr=lambda *_: (SimpleNamespace(parameters=lambda: []), None), transcribe=lambda *_: {'transcript': manual_asr}, generate=capture, summarize=lambda *_: {}, transcription_summary=lambda *_: {}):
    checked = core.run_evaluate(options, baseline=True)
assert [c['task'] for c in calls] == ['speech_chat', 'typed_chat']
assert calls[0]['user'] == manual_asr and calls[1]['user'] == reference
assert {k: v for k, v in calls[0].items() if k not in {'user', 'task'}} == {k: v for k, v in calls[1].items() if k not in {'user', 'task'}}
assert checked['requested_audio_chat_rows'] == 1 and checked['status'] == 'completed'
print('MANUAL_CONTRACT_VARIANT', json.dumps(calls, ensure_ascii=False))
session = {'history': [], 'assets': {}, 'transcriptions': {'manual-speech': {'transcript': manual_asr, 'raw_marker': 'preserved'}}}
server = SimpleNamespace(session=lambda _: session, model=object(), processor=object(), data_root=OUT, options=SimpleNamespace())
entered = []
def ui_capture(_model, _processor, row, *_):
    entered.append(row['user'])
    return {'prediction': 'manual fixture output'}
with patch.object(core, 'generate', ui_capture), patch.object(core, 'load_asr', side_effect=AssertionError('Unexpected ASR model loading')):
    corrected = ui.NaturalServer.operation(server, '/api/chat', {'session': 'manual-session', 'prompt': reference, 'speech': 'manual-speech'})
    typed_only = ui.NaturalServer.operation(server, '/api/chat', {'session': 'manual-session', 'prompt': reference})
assert corrected['asr']['transcript'] == manual_asr
assert corrected['asr']['submitted_text'] == reference and corrected['asr']['corrected'] is True
assert session['transcriptions']['manual-speech']['transcript'] == manual_asr
assert typed_only['asr'] is None and entered == [reference, reference]
print('UI_MANUAL_CONTRACT', json.dumps({'original_transcript': corrected['asr']['transcript'], 'submitted_text': corrected['asr']['submitted_text'], 'corrected': corrected['asr']['corrected'], 'typed_only_asr': typed_only['asr'], 'asr_model_loading_calls': 0}, ensure_ascii=False))
print('SUCCESS: exact original fence, five edit variants, four raw paired records, bindings, WAV hashes/metadata, one manual route-contract fixture and UI correction/typed-only contract. No ASR/chat model inference or training executed.')
