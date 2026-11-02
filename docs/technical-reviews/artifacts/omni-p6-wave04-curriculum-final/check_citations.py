"""Small post-original source-citation correction; no model code imported."""
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-p6-wave04-curriculum-final')
kev = CACHE / 'kev-model-2074cc4.py'
card = CACHE / 'minds14-README-40ce77c.md'
assert kev.read_bytes() == Path('/workspace/work/reference/kev/kev/model.py').read_bytes()
record_keys, supervision_keys, speaker_fields = set(), set(), []
count = 0
with tarfile.open(ROOT / 'assets/training/selftrained-v2.tar.gz') as archive:
    for name in ['voice-train.jsonl', 'voice-validation.jsonl', 'voice-test.jsonl']:
        for line in archive.extractfile(name):
            row = json.loads(line)
            count += 1
            record_keys.update(row)
            supervision_keys.update(row['supervision'])
            speaker_fields.extend(k for k in [*row, *row['supervision']] if 'speaker' in k.lower())
    package_card = archive.extractfile('licenses/minds14/README.md').read()
assert not speaker_fields and package_card == card.read_bytes()
freeze_path = ROOT / 'docs/lesson-products/results/settings-freeze-v1.json'
freeze = json.loads(freeze_path.read_bytes())
result = json.loads((ROOT / 'docs/lesson-products/results/final-report-v1.json').read_bytes())
assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == result['settings_freeze_sha256']
assert freeze['recorded_at'] < result['recorded_at'] and freeze['test_outcomes_observed'] is False
out = {
    'reviewer_task': '/root/omni_p6_technical_curriculum_final',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'command': 'python docs/technical-reviews/artifacts/omni-p6-wave04-curriculum-final/check_citations.py',
    'environment': {'python': sys.version.split()[0], 'device': 'CPU; metadata and JSONL only; no torch/model imported'},
    'original_unchanged': {'path': 'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-curriculum-final/original-v1.json', 'sha256': '1d318fdd74edc4edc7d778687a117e3feb7e354334daf6fff1b93fae22de9045'},
    'root_original_receipt': {'received_at': '2026-10-30T17:36:22.915173+00:00', 'intake_v16_sha256': '790613ccd6590ca689a5f3990946b232d131ef7fb85c1847de80d0a9f8b43243'},
    'Kev': {'correction': 'Original source URL owner was incorrectly written. Correct owner is jaredpalmer; substantive cached code bytes were already correct.', 'url': 'https://raw.githubusercontent.com/jaredpalmer/kev/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py', 'outside_git_path': str(kev), 'sha256': hashlib.sha256(kev.read_bytes()).hexdigest(), 'bytes': kev.stat().st_size, 'equal_to_previously_read_original_cache': True, 'actually_reread_after_fetch': 'encode87–128; branch_mask154–180; DecisionHead206–240; forward_rows_batch364–394'},
    'minds14': {'url': 'https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md', 'outside_git_path': str(card), 'sha256': hashlib.sha256(card.read_bytes()).hexdigest(), 'bytes': card.stat().st_size, 'equal_to_data_package_license_source': True, 'actually_read': 'dataset_info.all.features lines114–145', 'feature_excerpt': '\n'.join(card.read_text().splitlines()[112:146])},
    'voice_records': count, 'record_keys': sorted(record_keys), 'supervision_keys': sorted(supervision_keys), 'speaker_fields': speaker_fields,
    'bounded_reasoning': 'Different recordings may belong to the same speaker. The fixed course records and official data feature schema have no speaker identifier. Recording-level separation therefore supplies no proof of speaker separation. This does not assert measured speaker overlap or any speaker count.',
    'old_B_C_freeze_check': {'settings_path': str(freeze_path.relative_to(ROOT)), 'settings_sha256': result['settings_freeze_sha256'], 'freeze_recorded_at': freeze['recorded_at'], 'test_report_recorded_at': result['recorded_at'], 'test_outcomes_observed_at_freeze': False, 'result': 'Exact settings SHA matches and pre-test frozen timestamp precedes test report.'},
    'result': 'Correct Kev fixed source directly retrieved and reread;3015 voice records/schema inspected; c67 support limited to available metadata and explicit reasoning. No manuscript, original report, or original artifact modified.'
}
(Path(__file__).parent / 'source-citation-corrections.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'Kev_original_bytes_match': True, 'voice_records': count, 'speaker_fields': speaker_fields, 'old_B_C_freeze_hash_match': True}, ensure_ascii=False))
