"""Bounded independent 20.12 check: no ASR/LM inference or training."""
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import soundfile as sf
import torch

from tiny_perceptron import natural_assistant as core
from tiny_perceptron.natural_concepts import audio_order_report, text_error_report
from tiny_perceptron.natural_ui import NaturalServer

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent
RESEARCH = ROOT / 'outputs/natural-v4/factual-research/20.12'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(left, right):
    """Independent full matrix, unit insertion/deletion/substitution costs."""
    grid = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        grid[i][0] = i
    for j in range(len(right) + 1):
        grid[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            grid[i][j] = min(grid[i - 1][j] + 1, grid[i][j - 1] + 1,
                             grid[i - 1][j - 1] + int(left[i - 1] != right[j - 1]))
    return grid[-1][-1]


reference = '請推薦不辣的晚餐。'
recognized = '請推薦辣的晚餐。'
report = text_error_report(reference, recognized)
print('最少編輯次數', report['edits'])
print('參考字數', report['reference_characters'])
print('CER百分比', round(report['cer'] * 100, 1))
print('兩份問題相同', reference == recognized)
assert list(reference) == ['請', '推', '薦', '不', '辣', '的', '晚', '餐', '。']
# Length differs by one, so distance >= 1; inserting 不 achieves 1.
assert reference == recognized[:3] + '不' + recognized[3:]
assert report['edits'] == distance(reference, recognized) == 1
assert report['reference_characters'] == 9
assert round(report['cer'] * 100, 1) == 11.1
corrected = recognized[:3] + '不' + recognized[3:]
assert text_error_report(reference, corrected)['cer'] == 0
assert text_error_report(reference, reference[:-1])['edits'] == 1

# Execute actual server operation using a fixed ASR record and capture-only LM
# boundary. This is a protocol/correction exercise, not model inference.
original = {'transcript': recognized, 'stop_reason': 'eos'}
server = object.__new__(NaturalServer)
server.sessions = {'exercise': {'history': [], 'assets': {},
                              'transcriptions': {'recording': original.copy()}}}
server.model = server.processor = None
server.data_root = ROOT
server.options = SimpleNamespace()
captured = []
def capture(model, processor, row, root, options):
    captured.append(json.loads(json.dumps(row, ensure_ascii=False)))
    return {'prediction': '固定協定測試回應；沒有執行聊天模型。'}
with patch.object(core, 'generate', capture):
    correction_result = server.operation('/api/chat', {
        'session': 'exercise', 'prompt': corrected, 'speech': 'recording'})
assert captured[0]['user'] == corrected
assert correction_result['asr']['transcript'] == recognized
assert correction_result['asr']['submitted_text'] == corrected
assert correction_result['asr']['corrected'] is True
assert server.sessions['exercise']['transcriptions']['recording'] == original

base = ROOT / 'docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153'
manifest_path = ROOT / 'docs/natural-assistant/v4/manifest.json'
manifest = json.loads(manifest_path.read_text())
sources_path = ROOT / 'docs/natural-assistant/v4/data/voice-question-sources.json'
sources = json.loads(sources_path.read_text())
records_path = base / 'blind-review/generations-base.json'
transcripts_path = base / 'blind-review/transcripts.json'
result_path = base / 'blind-review/result.json'
records = json.loads(records_path.read_text())
transcripts = json.loads(transcripts_path.read_text())
result = json.loads(result_path.read_text())
index = json.loads((base / 'actual-artifact-index.json').read_text())
for name, path in [('review/generations-base.json', records_path),
                   ('review/transcripts.json', transcripts_path),
                   ('review/result.json', result_path)]:
    entry = next(x for x in index['files'] if x['path'] == name)
    assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
assert sha(manifest_path) == result['manifest_sha256']
annotations = {}
for line in (RESEARCH / 'aishell-transcripts-first-1MiB.txt').read_text().splitlines()[:-1]:
    identifier, text = line.split(maxsplit=1)
    annotations[identifier] = text
paired = []
audio_checks = []
source_rows = {r['id']: r for r in sources['audio_rows']}
for row in [x for x in manifest['audio_rows'] if x['task'] == 'speech_chat']:
    source = source_rows[row['id']]
    assert annotations[row['source_utterance_id']] == row['original_spaced_source_transcription']
    assert ''.join(annotations[row['source_utterance_id']].split()) == row['user']
    assert row['references']['rubric'] == source['semantic_rubric']
    audio = ROOT / 'outputs/natural-v4/data' / row['audio']
    info = sf.info(audio)
    assert sha(audio) == row['sha256']
    assert row['sha256'] == source['sha256']
    assert info.samplerate == 16000 and info.channels == 1 and info.frames == row['num_samples']
    assert result['asset_sha256'][row['audio']] == sha(audio)
    audio_checks.append({'id': row['id'], 'split': row['split'], 'speaker': row['source_speaker_id'],
                         'sha256': sha(audio), 'frames': info.frames, 'sample_rate': info.samplerate,
                         'source_transcript_exact_after_space_removal': True,
                         'predeclared_rubric_matches_manifest': True})
    if row['split'] != 'test':
        continue
    transcript = next(t for t in transcripts if t['id'] == row['id'])
    spoken = next(r for r in records if r['id'] == row['id'] and r['task'] == 'speech_chat')
    typed = next(r for r in records if r['id'] == row['id'] and r['task'] == 'typed_chat')
    assert typed['user'] == row['user']
    assert spoken['user'] == spoken['transcript'] == transcript['transcript']
    assert spoken['reference_user'] == transcript['reference_transcript'] == row['user']
    assert spoken['image'] == typed['image'] == row.get('image')
    assert spoken['family'] == typed['family'] == row['family']
    assert spoken['variant'] == typed['variant'] == 'base'
    assert transcript['audio_sha256'] == sha(audio)
    assert transcript['raw_errors'] == distance(row['user'], transcript['transcript'])
    assert transcript['raw_reference_characters'] == len(row['user'])
    paired.append({'id': row['id'], 'reference': row['user'], 'actual_asr': transcript['transcript'],
                   'reference_characters': len(row['user']),
                   'raw_edits': distance(row['user'], transcript['transcript']),
                   'same_image': spoken['image'] == typed['image'],
                   'system': row.get('system'), 'history': row.get('history'),
                   'same_generated_tokens': typed['generated_token_ids'] == spoken['generated_token_ids'],
                   'typed_answer': typed['prediction'], 'spoken_answer': spoken['prediction'],
                   'rubric': row['references']['rubric'],
                   'both_complete': all(r['stop_reason'] == 'eos' and r['ended_with_eos']
                                        and not r['truncated'] and r['generated_token_ids'][-1] in r['eos_token_ids']
                                        for r in [spoken, typed])})
assert len(paired) == 4 and sum(p['reference_characters'] for p in paired) == 42
assert sum(p['raw_edits'] for p in paired) == 0
assert len({r['source_speaker_id'] for r in sources['audio_rows']}) == 8
assert result['asr_model'] == 'openai/whisper-large-v3-turbo'
assert result['asr_revision'] == '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
plan_path = ROOT / 'docs/natural-assistant/v4/experiment-plan.json'
plan = json.loads(plan_path.read_text())
guard_path = base / 'source-selection-guard-before-test.json'
guard = json.loads(guard_path.read_text())
assert plan['dataset']['sha256'] == sha(manifest_path)
assert next(x for x in plan['source_hashes'] if x['path'].endswith('voice-question-sources.json'))['sha256'] == sha(sources_path)
for label in ['tiny_perceptron/natural_assistant.py', 'docs/natural-assistant/v4/manifest.json',
              'docs/natural-assistant/v4/validation-protocol-lower-lr.json']:
    assert guard['files'][label]['sha256'] == sha(ROOT / label)
assert plan['final_freeze_utc'] < guard['observed_at_utc'] < result['observed_at']
ambiguous = next(p for p in paired if p['id'].endswith('BAC009S0042W0480'))
assert ambiguous['reference'] == ambiguous['actual_asr']
assert '应询问具体发生什么事' in ambiguous['rubric']
assert ambiguous['typed_answer'] == '我會先靜下來想一想，了解事情的來龍去脈，然後根據情況決定是採取什麼行動。'

output = {
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
                    'cuda_available': torch.cuda.is_available()},
    'section_example': report, 'explicit_characters': list(reference),
    'lower_bound_and_witness': 'Length difference is 1; inserting 不 at index 3 yields reference, hence exact minimum 1.',
    'corrected_text': text_error_report(reference, corrected),
    'punctuation_probe': text_error_report(reference, reference[:-1]),
    'manual_correction_operation': correction_result,
    'original_asr_remained_unchanged': server.sessions['exercise']['transcriptions']['recording'] == original,
    'prerequisite_audio_order': audio_order_report(),
    'source_audio_checks': audio_checks, 'test_pairs': paired,
    'denominators': {'validation_question_recordings': 4, 'test_question_recordings': 4,
                     'test_reference_characters': 42, 'test_chat_answers_per_route': 4,
                     'project_seed': result['seed'], 'own_model_updates': 0},
    'record_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                      [manifest_path, sources_path, records_path, transcripts_path, result_path]},
    'record_environment': {k: result[k] for k in ['observed_at', 'model', 'model_revision', 'asr_model',
                                                'asr_revision', 'versions', 'device', 'seed']},
    'predeclaration_binding': {'plan_sha256': sha(plan_path), 'guard_sha256': sha(guard_path),
                              'frozen_manifest_and_question_source_hashes_match': True,
                              'runtime_source_bytes_match_pretest_guard': True,
                              'freeze_utc': plan['final_freeze_utc'], 'guard_utc': guard['observed_at_utc'],
                              'generation_provenance_utc': result['observed_at']},
    'independent_question_diagnosis': {'id': ambiguous['id'],
                                     'observation': 'Correct actual ASR and gold text yield the same complete answer; the answer offers a generic reaction without asking what happened, violating the frozen required clarification. This is a second-stage failure with no first-stage character errors.'},
    'scope': 'Independent CPU arithmetic and correction-protocol execution; audit of fixed original GPU/CPU records, not own ASR/LM inference, training, or performance replication. System/history reconstructed from frozen manifest and inspected runtime; raw records do not independently log rendered template tokens.'}
(OUT / 'probe-results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'corrected_CER': output['corrected_text']['cer'], 'punctuation_edits': output['punctuation_probe']['edits'],
                  'test_pairs': len(paired), 'test_raw_CER': '0/42', 'source_audio_checks': len(audio_checks),
                  'original_ASR_preserved': output['original_asr_remained_unchanged'],
                  'environment': output['environment'], 'scope': output['scope']}, ensure_ascii=False, indent=2))
