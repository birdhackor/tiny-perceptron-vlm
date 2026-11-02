"""Bounded offline asset audit: counts, digests, parser/help and tiny fixtures only."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args, *, cwd=ROOT, env=None, input=None):
    result = subprocess.run(args, cwd=cwd, env=env, input=input, capture_output=True, text=True, timeout=30)
    return {'argv': args, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def archive_binding(path, size, digest):
    path = ROOT / path
    actual = {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    assert actual['bytes'] == size and actual['sha256'] == digest, actual
    actual['matches_manifest'] = True
    pointer = subprocess.run(['git', 'show', f'HEAD:{actual["path"]}'], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    assert f'oid sha256:{digest}\n' in pointer and f'size {size}\n' in pointer, pointer
    actual['git_head_pointer'] = pointer
    return actual


def tiny_lfs_fixture():
    """Exercise the documented skip-smudge environment with a 20-byte local payload."""
    with tempfile.TemporaryDirectory(prefix='assets-review-lfs-') as temporary:
        base = Path(temporary)
        origin, clone = base / 'origin', base / 'clone'
        origin.mkdir()
        for args in [
            ['git', 'init', '-q'],
            ['git', 'config', 'user.email', 'asset-review@example.invalid'],
            ['git', 'config', 'user.name', 'Asset fixture'],
            ['git', 'lfs', 'install', '--local'],
        ]:
            assert run(args, cwd=origin)['returncode'] == 0
        (origin / '.gitattributes').write_text('*.dat filter=lfs diff=lfs merge=lfs -text\n')
        payload = b'tiny-assets-fixture\n'
        (origin / 'tiny.dat').write_bytes(payload)
        for args in [['git', 'add', '.gitattributes', 'tiny.dat'], ['git', 'commit', '-qm', 'fixture']]:
            assert run(args, cwd=origin)['returncode'] == 0
        environment = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}
        result = run(['git', 'clone', '--no-local', str(origin), str(clone)], cwd=base, env=environment)
        assert result['returncode'] == 0, result
        obtained = (clone / 'tiny.dat').read_text()
        expected = hashlib.sha256(payload).hexdigest()
        assert obtained.startswith('version https://git-lfs.github.com/spec/v1\n')
        assert f'oid sha256:{expected}\n' in obtained
        assert f'size {len(payload)}\n' in obtained
        objects = [p for p in (clone / '.git/lfs/objects').glob('**/*') if p.is_file()]
        assert not objects
        return {'operation': 'GIT_LFS_SKIP_SMUDGE=1 git clone --no-local <local fixture> <new path>', 'returncode': result['returncode'], 'pointer': obtained, 'payload_bytes': len(payload), 'lfs_objects_downloaded': len(objects), 'scope': 'Linux Bash environment/local clone with Git LFS 3.6.1; no public repository clone or external object transfer.'}


def v2_archive_counts(document):
    expected = {row['path']: row for row in document['records'] + document['assets']}
    counts, by_file, seen, expanded = Counter(), {}, set(), 0
    with tarfile.open(ROOT / document['package']['path'], 'r|gz') as archive:
        for member in archive:
            assert member.isfile() and member.name not in seen and member.name in expected, member.name
            row = expected[member.name]
            assert row['bytes'] == member.size
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            if member.name.endswith('.jsonl'):
                file_counts = Counter()
                for line in stream:
                    digest.update(line)
                    if line.strip():
                        value = json.loads(line)
                        file_counts[value['split']] += 1
                by_file[member.name] = dict(file_counts)
                counts.update(file_counts)
            else:
                for block in iter(lambda: stream.read(1024**2), b''):
                    digest.update(block)
            assert digest.hexdigest() == row['sha256'], member.name
            expanded += member.size
            seen.add(member.name)
    assert seen == set(expected)
    assert expanded == document['package']['unpacked_bytes']
    return {'files_verified': len(seen), 'record_files': len(document['records']), 'other_files': len(document['assets']), 'unpacked_payload_bytes': expanded, 'row_counts_by_split': dict(counts), 'row_counts_by_file': by_file, 'scope': 'All already-present local archive members read and checked against fixed manifest; no downloads, extraction, model execution or training.'}


def main():
    from scripts.omni_revision.public_download import descriptor, selected_files
    from scripts.fetch_natural_release import student_options, validate_manifest

    started = datetime.now(UTC).isoformat()
    baseline = json.loads((ROOT / 'assets/training/manifest.json').read_bytes())
    v2 = json.loads((ROOT / 'docs/selftrained/v2-manifest.json').read_bytes())
    natural = json.loads((ROOT / 'docs/natural-assistant/v4/manifest.json').read_bytes())
    rows = []
    for item in baseline['assets']:
        rows.append(archive_binding(item['archive'], item['archive_bytes'], item['archive_sha256']))
    rows.append(archive_binding(v2['package']['path'], v2['package']['bytes'], v2['package']['sha256']))
    for item in natural['archives']:
        rows.append(archive_binding(item['path'], item['bytes'], item['sha256']))
    expected = [item['archive'] for item in baseline['assets']] + [v2['package']['path']] + [item['path'] for item in natural['archives']]
    attributes = run(['git', 'check-attr', 'filter', '--', *expected])
    assert attributes['returncode'] == 0 and attributes['stdout'].count(': filter: lfs') == 12
    ignored = run(['git', 'check-ignore', 'data/assets-review-placeholder', 'checkpoints/assets-review-placeholder', 'outputs/assets-review-placeholder'])
    assert ignored['returncode'] == 0 and len(ignored['stdout'].splitlines()) == 3
    version = run(['git', 'lfs', 'version'])
    pinned_pointer = run(['git', 'show', '4a229b850289a5b8954092d71fcb05686988d526:assets/training/selftrained-v2.tar.gz'])
    assert pinned_pointer['returncode'] == 0
    assert f'oid sha256:{v2["package"]["sha256"]}\n' in pinned_pointer['stdout']

    published = descriptor(ROOT / 'docs/omni-products/v1/public-release.json')
    public_binding = {key: {'job': item['job'], 'bytes': item['bytes'], 'asset_id': item['asset_id'], 'files_bound_to_selection': sorted(selected_files(published['_model_selection'], key))} for key, item in published['components'].items()}
    natural_public = json.loads((ROOT / 'docs/natural-assistant/v4/public-release.json').read_bytes())
    validate_manifest(natural_public)
    options = student_options(natural_public, Path('checkpoints/not-created'), device='cpu')
    assert options.adapter is None
    audio_counts = Counter(row['split'] for row in natural['audio_rows'])
    train_rows = [row for row in natural['rows'] if row['split'] == 'train']
    assert audio_counts['train'] == 0 and all(not row.get('audio') for row in train_rows)
    cli = []
    for args in [
        [sys.executable, 'scripts/fetch_training_assets.py', '--list'],
        [sys.executable, '-m', 'scripts.omni_revision.public_download', '--help'],
        [sys.executable, 'scripts/selftrained/train_local_stage.py', '--help'],
        [sys.executable, 'scripts/selftrained/train.py', '--help'],
        [sys.executable, 'scripts/selftrained/chat.py', '--help'],
        [sys.executable, 'scripts/fetch_natural_release.py', '--manifest', 'docs/natural-assistant/v4/public-release.json', '--list'],
    ]:
        result = run(args)
        if args[1:3] == ['scripts/selftrained/train.py', '--help']:
            # Preserve an ancillary help-formatting failure; this page does not
            # quote that command. Inspect the parser contract without training.
            assert result['returncode'] == 1 and 'ValueError: incomplete format' in result['stderr'], result
            result['audit_note'] = 'Ancillary --help formatting failure, outside this page\'s quoted commands; not hidden or repaired.'
        else:
            assert result['returncode'] == 0, result
        cli.append(result)
    from scripts.selftrained.train import parser as trainer_parser
    trainer_actions = {action.dest: {'options': action.option_strings, 'help': action.help} for action in trainer_parser()._actions if action.dest in {'resume', 'init_checkpoint', 'steps', 'device', 'output_dir'}}
    plan = json.loads((ROOT / 'docs/course-experiments/plan.json').read_bytes())
    result = {
        'started_at': started, 'completed_at': datetime.now(UTC).isoformat(),
        'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU; no model tensors allocated', 'git': run(['git', '--version'])['stdout'].strip(), 'git_lfs': version['stdout'].strip(), **{name: importlib.metadata.version(name) for name in ['torch','safetensors','huggingface-hub']}},
        'source': {'path': 'docs/asset-storage.md', 'bytes': (ROOT/'docs/asset-storage.md').stat().st_size, 'sha256': sha(ROOT/'docs/asset-storage.md')},
        'archive_bindings': rows,
        'totals': {'baseline_packages': len(baseline['assets']), 'baseline_bytes': sum(item['archive_bytes'] for item in baseline['assets']), 'v2_bytes': v2['package']['bytes'], 'natural_packages': len(natural['archives']), 'natural_bytes': sum(item['bytes'] for item in natural['archives']), 'decimal_MB': [round(sum(item['archive_bytes'] for item in baseline['assets'])/1e6,2), round(v2['package']['bytes']/1e6,2), round(sum(item['bytes'] for item in natural['archives'])/1e6,2)]},
        'v2_archive_inspection': v2_archive_counts(v2),
        'v2_pinned_git_pointer': pinned_pointer,
        'git_attributes': attributes, 'git_ignored_directories': ignored,
        'skip_smudge_tiny_fixture': tiny_lfs_fixture(),
        'public_descriptor_validated': public_binding,
        'natural_student_options': {'model': options.model, 'model_revision': options.model_revision, 'asr_model': options.asr_model, 'asr_revision': options.asr_revision, 'adapter': options.adapter, 'dtype': options.dtype},
        'natural_training_selection': {'train_rows': len(train_rows), 'train_audio_rows': audio_counts['train'], 'audio_split_counts': dict(audio_counts), 'train_tasks': dict(Counter(row['task'] for row in train_rows)), 'selection_contract': 'run_train selects manifest[rows] where split == train; audio_rows are not selected'},
        'course_experiment_sequence_length': len(plan['sequence']),
        'cli': cli,
        'trainer_parser_contract': trainer_actions,
        'scope': 'Existing asset bytes inspected offline. Tiny local LFS fixture and help/list calls only. No training/final/teacher or neural forward call; no large weight download, GPU, Modal, paid compute, publication, or platform-wide claim.',
    }
    (OUT / 'local-inspection-v1.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['totals','v2_archive_inspection','natural_training_selection','course_experiment_sequence_length']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
