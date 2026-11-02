"""Publish three already verified original releases; preserve all assets and targets."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/work/omni-book-integrated-v1')
sys.path.insert(0, str(ROOT))
from scripts.omni_revision.github_transport_probe import api

proposal = Path('/workspace/work/omni-publication-proposal-final-v1.json')
marker = Path('/workspace/work/omni-publication-mutation-receipt-v1.json')
audit = ROOT / 'docs/course-revision-20261029-omni/infrastructure/qwen25-actual-execution-verification-v1.json'
assert audit.is_file(), 'Wait for actual Qwen execution verification'
spec = json.loads(proposal.read_bytes())
receipt = {
    'started_at': datetime.now(UTC).isoformat(),
    'author_task': '/root',
    'authority': 'User-approved GO includes model publication; this operation does not publish the textbook.',
    'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
    'qwen_actual_report_sha256': hashlib.sha256(audit.read_bytes()).hexdigest(),
    'preserve': 'Original tags, target commits, asset IDs, sizes and digests. Prerelease; make_latest=false.',
    'operations': [],
    'complete': False,
}
with marker.open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)

def identity(release):
    return {
        'id': release['id'], 'tag_name': release['tag_name'],
        'target_commitish': release['target_commitish'],
        'assets': sorted((a['id'], a['name'], a['size'], a['digest']) for a in release['assets']),
    }

def save():
    marker.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')

for row in spec['releases']:
    old = json.loads(Path(row['original_file']).read_bytes())
    rid = row['release_id']
    current = api(f'releases/{rid}')
    assert identity(current) == identity(old)
    assert current['draft'] is True and current['body'] == old['body']
    operation = {'release_id': rid, 'started_at': datetime.now(UTC).isoformat(), 'write_outcome': 'unknown'}
    receipt['operations'].append(operation)
    save()
    result = api(f'releases/{rid}', 'PATCH', row['payload'])
    assert identity(result) == identity(old)
    assert result['draft'] is False and result['prerelease'] is True
    assert result['body'] == row['payload']['body']
    operation.update(write_outcome='success', finished_at=datetime.now(UTC).isoformat(),
                     html_url=result['html_url'], published_at=result['published_at'],
                     identity=identity(result))
    save()
receipt.update(complete=True, finished_at=datetime.now(UTC).isoformat())
save()
print(json.dumps({'published': len(receipt['operations']), 'all_original_asset_identities_preserved': True}))
