"""Owner-performed byte-exact metadata relocation; never edits teaching/data/model files."""
import copy
import hashlib
import json
import platform
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'docs/technical-reviews/20.5.json'
ARTIFACTS = ROOT / 'docs/technical-reviews/artifacts'
PREFIX = 'natural-final-fact-20.5-'
OUT = ARTIFACTS / (PREFIX + 'data-source-snapshots')
HISTORY = ROOT / 'docs/technical-reviews/history/20.5'
ORIGINAL_PATH = 'data/natural/vision/selected-descriptions.jsonl'
MANIFEST_PATH = 'docs/natural-assistant/manifest.json'
EXPECTED_MANIFEST_SHA = '7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91'
EXPECTED_SOURCE_SHA = '77f13cdfeb80907c27b932bc4e0f7a95df221d2313929e3b7aa5739ec93ca343'
ACTOR = '/root/natural_factual_final_20_5'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write_new_or_equal(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f'Preserve existing different artifact: {path}'
    else:
        path.write_bytes(data)

prior_bytes = REPORT.read_bytes()
prior = json.loads(prior_bytes)
assert prior['reviewer_task'] == ACTOR and prior['verdict'] == 'pass'
assert next(s for s in prior['sources'] if s['id'] == 's-original-desc')['path'] == ORIGINAL_PATH
assert next(s for s in prior['sources'] if s['id'] == 's-original-desc')['sha256'] == EXPECTED_SOURCE_SHA
for item in prior['artifacts']:
    assert digest((ROOT / item['path']).read_bytes()) == item['sha256']
body = (ROOT / 'course/chapters/20.md').read_text()
body = body[body.index('## 20.5'):body.index('## 20.6')]
assert digest(body.encode()) == prior['source_sha256']
for path, sha in prior['figure_sha256'].items():
    assert digest((ROOT / path).read_bytes()) == sha

source_bytes = (ROOT / ORIGINAL_PATH).read_bytes()
assert len(source_bytes) == 178642 and digest(source_bytes) == EXPECTED_SOURCE_SHA
manifest_bytes = (ROOT / MANIFEST_PATH).read_bytes()
assert digest(manifest_bytes) == EXPECTED_MANIFEST_SHA
manifest = json.loads(manifest_bytes)
assert len(manifest['files']) == len({f['path'] for f in manifest['files']}) == 345
member_path = 'vision/selected-descriptions.jsonl'
member_entry = next(f for f in manifest['files'] if f['path'] == member_path)
assert member_entry['bytes'] == len(source_bytes) and member_entry['sha256'] == digest(source_bytes)
archive_entry = next(a for a in manifest['archives'] if a['path'] == 'assets/training/natural-vision-v3.tar.gz')
archive_path = ROOT / archive_entry['path']
archive_bytes = archive_path.read_bytes()
assert len(archive_bytes) == archive_entry['bytes'] == 25775400
assert digest(archive_bytes) == archive_entry['sha256'] == 'eece1f34ddc38f2538c2a7f8dcf257639d2388b0a3fc313ebf0a3875fca9e09f'
assert next(f for f in archive_entry['files'] if f['path'] == member_path) == member_entry
with tarfile.open(archive_path, 'r:gz') as archive:
    member = archive.getmember(member_path)
    assert member.isfile() and member.size == 178642
    archived_metadata = archive.extractfile(member).read()
assert archived_metadata == source_bytes

# The publisher's original complete annotation metadata is verified in memory;
# only the already-reviewed 178642-byte selected metadata is relocated to disk.
vision_manifest_bytes = (ROOT / 'data/natural/vision/manifest.json').read_bytes()
assert digest(vision_manifest_bytes) == next(f['sha256'] for f in manifest['files'] if f['path'] == 'vision/manifest.json')
vision_manifest = json.loads(vision_manifest_bytes)
source_info = vision_manifest['sources'][0]
upstream = next(f for f in source_info['files'] if f['path'] == 'sources/docci-descriptions.jsonl')
url = 'https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines?generation=1714384012999810'
assert upstream['url'] == url
assert source_info['description_generation'] == '1714384012999810'
assert upstream['sha256'] == source_info['description_sha256'] == 'c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800'
official_local = (ROOT / 'data/natural/vision/sources/docci-descriptions.jsonl').read_bytes()
assert digest(official_local) == upstream['sha256']
with urllib.request.urlopen(url, timeout=30) as response:
    official_fetched = response.read()
    status = response.status
    fetched_generation = response.headers.get('x-goog-generation')
assert status == 200 and digest(official_fetched) == upstream['sha256']
assert official_fetched == official_local
official_by_id = {r['example_id']: r for r in map(json.loads, official_fetched.splitlines())}
selected_rows = list(map(json.loads, source_bytes.splitlines()))
assert len(selected_rows) == len({r['example_id'] for r in selected_rows}) == 144
upstream_field_checks = []
for row in selected_rows:
    official = official_by_id[row['example_id']]
    assert all(row[field] == official[field] for field in ['example_id', 'description', 'split', 'image_file'])
    upstream_field_checks.append({'example_id': row['example_id'], 'description_sha256': digest(row['description'].encode()), 'all_four_original_annotation_fields_equal': True})

snapshot = OUT / 'selected-descriptions.jsonl'
history = HISTORY / 'before-data-source-relocation-report.json'
write_new_or_equal(snapshot, source_bytes)
write_new_or_equal(history, prior_bytes)
assert snapshot.read_bytes() == source_bytes and history.read_bytes() == prior_bytes
proof_path = OUT / 'relocation-proof.json'
proof = {
    'schema_version': 1,
    'actor_task': ACTOR,
    'actor_role': 'Original fresh correctness reviewer performing relocation personally',
    'performed_at_utc': datetime.now(timezone.utc).isoformat(),
    'operation': 'byte-exact copy of previously reviewed selected DOCCI description metadata; source registration path only',
    'original': {'path': ORIGINAL_PATH, 'bytes': len(source_bytes), 'sha256': digest(source_bytes)},
    'durable_snapshot': {'path': str(snapshot.relative_to(ROOT)), 'bytes': snapshot.stat().st_size, 'sha256': digest(snapshot.read_bytes()), 'byte_exact_equal': True},
    'prior_pass_report': {'path': str(history.relative_to(ROOT)), 'sha256': digest(prior_bytes), 'byte_exact_equal': True, 'source_sha256': prior['source_sha256']},
    '345_file_bundle': {'manifest_path': MANIFEST_PATH, 'manifest_sha256': digest(manifest_bytes), 'file_count': 345, 'matching_member_entry': member_entry},
    'archive_member_verification': {'archive_path': archive_entry['path'], 'archive_bytes': len(archive_bytes), 'archive_sha256': digest(archive_bytes), 'member_path': member_path, 'member_bytes': len(archived_metadata), 'member_sha256': digest(archived_metadata), 'member_bytes_equal_original_and_snapshot': True},
    'official_original_source': {'url': url, 'generation': '1714384012999810', 'fetched_header_generation': fetched_generation, 'http_status': status, 'bytes': len(official_fetched), 'sha256': digest(official_fetched), 'fetched_bytes_equal_original_bundle_official_metadata': True, 'selected_rows': 144, 'verified_fields': ['example_id', 'description', 'split', 'image_file'], 'row_checks': upstream_field_checks, 'publisher': 'Google DOCCI', 'license': source_info['license'], 'image_creator': source_info['image_creator'], 'annotation_creator': source_info['annotation_creator'], 'downloaded_metadata_not_saved_as_additional_dataset': True},
    'unchanged': {'lesson_body_sha256': prior['source_sha256'], 'figures': prior['figure_sha256'], 'claims': True, 'scores': True, 'verdict': 'pass', 'reviewer_task': ACTOR, 'GT': True, 'selection': True, 'raw_outputs': True, 'runtime': True, 'models': True},
    'gpu_used': False,
    'images_or_weights_copied': False,
    'python': platform.python_version(),
    'scope': 'Publication durability fix, not a new correctness review, score reinterpretation, or model inference. Original report and every original review artifact remain unchanged.'
}
proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')

updated = copy.deepcopy(prior)
for registry in ['sources', 'primary_sources']:
    item = next(s for s in updated[registry] if s['id'] == 's-original-desc')
    item['original_reviewed_path'] = ORIGINAL_PATH
    item['path'] = str(snapshot.relative_to(ROOT))
    item['sha256'] = EXPECTED_SOURCE_SHA
    item['relocation_evidence_path'] = str(proof_path.relative_to(ROOT))
    item['inspection_note'] += '；原reviewer親自將已讀metadata byte-exact搬至可發布artifact，同SHA；另實核345清單對應member、原vision archive及固定GCS原annotation144筆字段，proof與原pass歷史保留。'
def add_artifact(identifier, path, kind, description, **extra):
    updated['artifacts'].append({'id': identifier, 'path': str(path.relative_to(ROOT)), 'kind': kind, 'sha256': digest(path.read_bytes()), 'description': description, **extra})
add_artifact('a-relocation-source-snapshot', snapshot, 'source_snapshot', '先前已讀DOCCI selected description metadata原bytes；可發布，178642 bytes，原SHA不變；沒有照片或weights。')
add_artifact('a-before-data-source-relocation-report', history, 'source_snapshot', '本次搬移之前的原pass報告完整bytes歷史，並非重新生成舊報告。')
add_artifact('a-relocation-code', Path(__file__), 'code', '原reviewer親自執行的metadata固定來源／archive member／bytecopy／source path更新程式。')
add_artifact('a-relocation-proof', proof_path, 'execution', '實際原reviewer搬移proof：原178642 bytes同SHA、345manifest entry、archive原member及匿名固定GCS原annotation字段核驗。', command='python docs/technical-reviews/artifacts/' + PREFIX + 'relocate-data-source.py', result='Exit 0；metadata／archive member／官方原annotation144筆四fields一致；原report/body/figures/claims/scores不變；只live source path與新增proof登記。', environment={'python': platform.python_version(), 'device': 'CPU only; no GPU or neural inference', 'network': 'Anonymous fixed-generation Google DOCCI annotation metadata read only'})
updated.setdefault('relocation_history', []).append({'actor_task': ACTOR, 'kind': 'durable metadata source relocation', 'original_source_path': ORIGINAL_PATH, 'new_source_path': str(snapshot.relative_to(ROOT)), 'source_sha256_unchanged': EXPECTED_SOURCE_SHA, 'proof_artifact_id': 'a-relocation-proof', 'prior_report_artifact_id': 'a-before-data-source-relocation-report', 'claims_and_scores_unchanged': True})
updated['independent_checks']['data_source_snapshot_relocation'] = {'actor_task': ACTOR, 'proof_artifact_id': 'a-relocation-proof', 'new_correctness_review_or_inference': False}
assert updated['claims'] == prior['claims'] and updated['checks'] == prior['checks']
assert updated['source_sha256'] == prior['source_sha256'] and updated['figure_sha256'] == prior['figure_sha256']
assert updated['verdict'] == prior['verdict'] == 'pass'
REPORT.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'actor_task': ACTOR, 'snapshot': str(snapshot.relative_to(ROOT)), 'bytes': len(source_bytes), 'sha256': EXPECTED_SOURCE_SHA, 'bundle_files': 345, 'selected_original_field_matches': 144, 'official_http_status': status, 'claims_scores_body_unchanged': True}, ensure_ascii=False, indent=2))
