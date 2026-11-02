"""Verify exact retained asset-storage evidence offline; do not rerun old experiments."""

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

ROOT = Path('/tmp/tpb').resolve()
HERE = Path(__file__).resolve().parent
ROUND = 'docs/course-revision-20261101-c-v3/'
CACHE = Path('/workspace/work/reference/omni-p6-wave04-assets')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(relative):
    return json.loads((ROOT / relative).read_bytes())


def main():
    started = datetime.now(UTC).isoformat()
    mapping = load(ROUND + 'technical-coordination-v21/prior-evidence-reference-map-resume-v1.json')
    row = next(x for x in mapping['pages'] if x['page_id'] == 'asset-storage')
    proof = row['prior_technical_proof']
    old = load(proof['original_report']['path'])
    old_current = load(proof['report']['path'])
    original_path = ROUND + 'technical-v21/canonical_asset_storage/original-v1.json'
    original = load(original_path)
    result = {'actor': original['reviewer_task'], 'started_at': started,
              'kind': 'offline immutable retention audit; no new old-model/public-authority GET, weights, unpack or training',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'device': 'CPU; hashes/metadata/arithmetic and one mocked eight-selection branch only'}}
    references = [proof['original_report'], proof['report'], proof['original_source_freeze'],
                  {'path': original_path, 'sha256': 'c0ff09cc9369bbc7cd86e26ba7d332f48fb1178f80bd6451875af950affc46f2'},
                  {'path': ROUND + 'technical-coordination-v21/receipts/canonical-asset-storage-original-v1.json',
                   'sha256': '4d0c4122ed357c5f989b24eb28fcee712e8f3123e493384b5fafb6d642067de0'},
                  {'path': ROUND + 'source-freeze-v22.json', 'sha256': '0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59'},
                  {'path': ROUND + 'evidence/format-only-source22-equivalence-v1.json', 'sha256': '16a9e24b2a20b62662e060f82fdcd1808f9995c2312a6f6b7bbacaafb2740f33'}]
    result['references'] = []
    for ref in references:
        observed = sha((ROOT / ref['path']).read_bytes())
        assert observed == ref['sha256'], ref['path']
        result['references'].append({**ref, 'matches': True})
    result['retained_owner'] = {'task': old['reviewer_task'], 'original_created_at': old['created_at'],
                                'current_created_at': old_current['created_at'],
                                'original_received_at': proof['original_collection']['received_at'],
                                'current_received_at': proof['current_collection']['received_at']}
    result['old_original_current_evidence_identical'] = all(old[k] == old_current[k] for k in ['claims', 'sources', 'artifacts'])
    current_raw = (ROOT / 'docs/asset-storage.md').read_bytes()
    old_raw = (ROOT / old['reviewed_snapshot']['path']).read_bytes()
    assert sha(old_raw) == old['source_sha256']
    c3_line = next(line for line in current_raw.splitlines(keepends=True) if b'C-v3' in line)
    assert current_raw.replace(c3_line, b'', 1) == old_raw
    f21 = next(x for x in load(ROUND + 'source-freeze-v21.json')['pages'] if x['page_id'] == 'asset-storage')
    f22 = next(x for x in load(ROUND + 'source-freeze-v22.json')['pages'] if x['page_id'] == 'asset-storage')
    assert f21 == f22 and sha(current_raw) == f22['source_sha256'] == original['source_sha256']
    result['source_identity'] = {'legacy_sha256': sha(old_raw), 'source21_and_22_sha256': sha(current_raw),
                                  'source21_and_22_page_records_equal': True,
                                  'only_addition_from_legacy': c3_line.decode().rstrip('\n'),
                                  'all_other_bytes_identical': True}
    result['repository_dependencies'] = []
    for origin, sources in [('retained_2026_10_30', old['sources']), ('own_original_2026_11_01', original['sources'])]:
        for source in sources:
            if source['kind'] == 'repository_code':
                observed = sha((ROOT / source['path']).read_bytes())
                assert observed == source['sha256'], source['path']
                result['repository_dependencies'].append({'origin': origin, 'path': source['path'],
                                                           'sha256': observed, 'matches': True})
    result['retained_artifacts'] = []
    for ref in row['prior_artifact_refs']:
        assert sha((ROOT / ref['path']).read_bytes()) == ref['sha256']
        result['retained_artifacts'].append({k: ref[k] for k in ('id', 'path', 'sha256')})
    for artifact in original['artifacts']:
        assert sha((ROOT / artifact['path']).read_bytes()) == artifact['sha256']
    receipt_rows = []
    legacy_art = 'docs/technical-reviews/artifacts/omni-p6-wave04-assets/'
    for name in ['primary-receipts-v1.json', 'primary-extra-receipts-v1.json', 'legacy-primary-receipts-v1.json']:
        receipt_rows.extend(load(legacy_art + name)['sources'])
    result['original_get_bytes'] = []
    for receipt in receipt_rows:
        raw = Path(receipt['cache_path']).read_bytes()
        assert sha(raw) == receipt['sha256'] and len(raw) == receipt['bytes']
        assert receipt['status'] == 200
        result['original_get_bytes'].append({k: receipt[k] for k in ['name', 'url', 'status', 'accessed_at', 'bytes', 'sha256']})
    def cached(name):
        return json.loads((CACHE / name).read_bytes())
    selected = load('docs/omni-products/v1/public-release.json')
    assert sha((ROOT / selected['model_selection']['path']).read_bytes()) == selected['model_selection']['sha256']
    release = cached('github-release-api.json')
    assert release['tag_name'] == selected['release_tag'] and release['draft'] is False
    for item in selected['components'].values():
        remote = next(x for x in release['assets'] if x['id'] == item['asset_id'])
        assert (remote['name'], remote['size'], remote['digest'], remote['state']) == (item['name'], item['bytes'], 'sha256:' + item['sha256'], 'uploaded')
    result['old_release_metadata'] = {'id': release['id'], 'tag': release['tag_name'], 'assets_checked': len(selected['components']),
                                      'default_families': ['av', 'choice', 'codec', 'talker'],
                                      'scope': '2026-10-30 original provider metadata retained; no new Release ZIP payload download'}
    v2tree = {x['path']: x for x in cached('hf-v2-tree.json') if x['type'] == 'file'}
    groups = ['moe-pretrain', 'moe-sft', 'moe-joint', 'dense-joint']
    filenames = {'model.safetensors', 'model-config.json', 'tokenizer.json', 'inference-manifest.json'}
    assert set(v2tree) == {f'selftrained/v2/{g}/{n}' for g in groups for n in filenames}
    exports = {x['name']: x for x in load('docs/selftrained/v2-jobs/release-final-four-exports.json')['release']['exports']}
    for group in groups:
        manifest = cached(f'hf-v2-{group}-inference-manifest.json')
        assert not {'optimizer', 'rng', 'sampler'} & manifest.keys()
        for item in exports[group]['files']:
            remote = v2tree[f'selftrained/v2/{group}/{item["path"]}']
            assert remote['size'] == item['bytes'] and item['license'] == 'MIT'
            observed = remote['lfs']['oid'] if item['path'] == 'model.safetensors' else sha((CACHE / f'hf-v2-{group}-{item["path"]}').read_bytes())
            assert observed == item['sha256']
    card = (CACHE / 'hf-current-root-readme.md').read_text()
    license_line = next(line for line in card.splitlines() if '共16個檔案' in line and 'MIT' in line)
    assert '979cdfacc588ad0536f1c64fff96f264571cf054' in license_line
    result['v2_retained_metadata'] = {'groups': groups, 'files': len(v2tree), 'bytes': sum(x['size'] for x in v2tree.values()),
                                      'license_card_commit': '0b23616ba220de6e9a09943d145f5c5013694c90',
                                      'scope': '12 original small metadata payload GETs plus provider OIDs for 4 weights; no weight GET or fresh inference'}
    public = load('docs/course-experiments/public-models.json')
    checks, oids = 0, 0
    for model in public['models']:
        tree = {x['path']: x for x in cached(f'hf-course-{model["id"]}-tree.json') if x['type'] == 'file'}
        for item in model['files']:
            assert tree[item['path']]['size'] == item['bytes']
            checks += 1
            if 'lfs' in tree[item['path']]:
                assert tree[item['path']]['lfs']['oid'] == item['sha256']
                oids += 1
    capstone = load('docs/course-experiments/capstone-public.json')
    capfiles = {x['path']: x for model in capstone['models'] for x in model['files']}
    captree = {x['path']: x for x in cached('hf-legacy-tree.json') if x['type'] == 'file'}
    for name, item in capfiles.items():
        assert captree[name]['size'] == item['bytes']
        if 'lfs' in captree[name]:
            assert captree[name]['lfs']['oid'] == item['sha256']
    result['historical_hf_metadata'] = {'models': len(public['models']), 'file_sizes_checked': checks,
                                        'provider_lfs_oids_checked': oids, 'capstone_models': len(capstone['models']),
                                        'capstone_unique_files': len(capfiles), 'scope': 'Retained original fixed-revision provider subtree bytes; non-LFS files size-only, no all-model weight roundtrip'}
    natural = load('docs/natural-assistant/v4/public-release.json')
    natural_tree = {x['path']: x for x in cached('hf-natural-v4-tree.json')}
    assert set(natural_tree) == {x['path'] for x in natural['files']}
    for item in natural['files']:
        assert natural_tree[item['path']]['size'] == item['bytes']
        assert sha((CACHE / ('hf-natural-v4-' + item['output'])).read_bytes()) == item['sha256']
    provenance = cached('hf-natural-v4-release-provenance.json')
    assert provenance['base_model'] == natural['base_model'] and provenance['asr_model'] == natural['asr_model']
    result['natural_pairing'] = {'files': len(natural_tree), 'bytes': sum(x['size'] for x in natural_tree.values()),
                                 'base_model': provenance['base_model'], 'asr_model': provenance['asr_model'],
                                 'selected_variant': provenance['selected_variant'], 'scope': 'Retained two metadata payloads; upstream large weights not downloaded'}
    local = load(legacy_art + 'local-inspection-v1.json')
    byfile = local['v2_archive_inspection']['row_counts_by_file']
    totals = Counter()
    for rowcounts in byfile.values():
        totals.update(rowcounts)
    assert dict(totals) == local['v2_archive_inspection']['row_counts_by_split'] == {'test': 3734, 'train': 28876, 'validation': 2435}
    result['v2_split_arithmetic'] = {'per_file_counts': byfile, 'sum': dict(totals),
                                     'basis': 'Retained 2026-10-30 8962-member SHA-verified read-only inspection bound to same manifest/archive SHA; not reopened in this run'}
    plan = load('docs/course-experiments/plan.json')
    assert len(plan['sequence']) == 30 == local['course_experiment_sequence_length']
    natdata = load('docs/natural-assistant/v4/manifest.json')
    train = [x for x in natdata['rows'] if x['split'] == 'train']
    result['natural_audio_exclusion'] = {'train_rows': len(train), 'train_audio_rows': sum(bool(x.get('audio')) for x in train),
                                         'audio_splits': dict(Counter(x['split'] for x in natdata['audio_rows']))}
    assert result['natural_audio_exclusion']['train_rows'] == 2077 and result['natural_audio_exclusion']['train_audio_rows'] == 0
    assert result['natural_audio_exclusion']['audio_splits'] == {'validation': 16, 'test': 22}
    # One narrowly necessary software branch probe: no unpack, no LFS pull, no download.
    spec = importlib.util.spec_from_file_location('retention_fetch_assets', ROOT / 'scripts/fetch_training_assets.py')
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    selected_ids = []
    def no_unpack(archive, asset, destination):
        selected_ids.append(asset['id'])
        return {'asset': asset['id'], 'probe_only': True}
    with mock.patch.object(sys, 'argv', ['fetch_training_assets.py', '--asset', 'all']), mock.patch.object(fetch, 'unpack_asset', no_unpack), mock.patch.object(fetch.subprocess, 'run', side_effect=AssertionError('network/LFS operation forbidden')), contextlib.redirect_stdout(io.StringIO()):
        fetch.main()
    expected_ids = [x['id'] for x in load('assets/training/manifest.json')['assets']]
    assert selected_ids == expected_ids and len(selected_ids) == 8
    result['all_selection_branch_probe'] = {'selected_ids': selected_ids, 'count': len(selected_ids), 'unpack_performed': False,
                                             'network_performed': False, 'scope': 'Actual main --asset all branch, only unpack_asset replaced by recorder; real files read first128 bytes only'}
    own = load(HERE.relative_to(ROOT).as_posix() + '/audit-result.json')
    assert len(own['training_packages']) == 8 + 1 + 3 == 12
    result['original_verified_claim_correction'] = {'claim_id': 'c14', 'original_misstatement': '13 packages in verification/result prose',
                                                   'correct_count': 12, 'raw_evidence_unchanged': True,
                                                   'details': 'Original audit-result contains exactly8+1+3=12 entries; textbook table was already correct; original report is not rewritten.'}
    c3 = load('docs/decision-product/v3/public-release.json')
    for path, expected in c3['implementation'].items():
        assert sha((ROOT / path).read_bytes()) == expected
    result['own_c3_retention'] = {'audit_sha256': sha((HERE / 'audit-result.json').read_bytes()),
                                  'public_release_sha256': sha((ROOT / 'docs/decision-product/v3/public-release.json').read_bytes()),
                                  'qualified_python_files_matched': len(c3['implementation']),
                                  'new_public_GETs': 0, 'new_real_archive_reads': 0,
                                  'same_original_transport_receipt': own['c3_anonymous_transport']}
    result['completed_at'] = datetime.now(UTC).isoformat()
    (HERE / 'retention-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed_at': result['completed_at'], 'old_owner': result['retained_owner'],
                       'repository_hashes': len(result['repository_dependencies']), 'retained_artifact_hashes': len(result['retained_artifacts']),
                       'original_GET_cache_hashes': len(result['original_get_bytes']), 'v2_split_counts': dict(totals),
                       'v2_file_count': len(v2tree), 'all_branch_selection_count': len(selected_ids),
                       'real_package_count': len(own['training_packages']), 'new_GETs': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
