"""Bounded CPU audit of existing source metadata; no inference or new scores."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, hashlib, json, platform, sys, types

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def load(name): return json.loads((BASE / 'inputs' / name).read_text())
def lines(name):
    with (BASE / 'inputs' / name).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

manifest = load('manifest.json')
protocol = load('validation-protocol.json')
assert manifest['dataset_version'] == 'natural-assistant-v4'
rows = manifest['rows'] + manifest['audio_rows']
assert len({r['id'] for r in rows}) == len(rows)
groups = defaultdict(set)
for row in rows: groups[row['family']].add(row['split'])
assert all(len(splits) == 1 for splits in groups.values())
image_splits = defaultdict(set)
for row in rows:
    if row.get('image'): image_splits[row['image']].add(row['split'])
assert all(len(splits) == 1 for splits in image_splits.values())

sources = manifest['sources']
for source in sources:
    source_path = ROOT / source['path']
    assert digest(source_path) == source['sha256'], source['path']
    assert json.loads(source_path.read_text()) == source['metadata'], source['path']

# Execute the real asset/source/family validator with existing immutable metadata.
builder = BASE / 'inputs/build_natural_v4_assets.py'
tree = ast.parse(builder.read_text())
selected_names = {'sha256', 'relative', 'safe_destination', 'regular_file', 'validate_assets'}
selected = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)): selected.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in selected_names: selected.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VOICE_PINS' for t in node.targets): selected.append(node)
namespace = {'__name__': 'reviewer_asset_contract'}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(builder), 'exec'), namespace)
selected_assets = namespace['validate_assets'](ROOT / 'outputs/natural-v4/data', manifest['rows'], manifest['audio_rows'], sources)
# Check the tracked immutable bundles, so the physical inputs remain retrievable.
archive_checks = []
for entry in manifest['archives']:
    original = ROOT / entry['path']
    assert original.stat().st_size == entry['bytes']
    assert digest(original) == entry['sha256']
    archive_checks.append({'path': entry['path'], 'bytes': entry['bytes'], 'sha256': entry['sha256']})

docci = {r['example_id']: r['cluster_id'] for r in lines('docci-clusters.jsonl')}
expected_docci_hash = load('vision-sources.json')['upstream']['clusters']['sha256']
assert digest(BASE / 'inputs/docci-clusters.jsonl') == expected_docci_hash
photos, selected_photos, clusters, original_splits, rgb_splits = {}, {}, defaultdict(set), defaultdict(set), defaultdict(set)
for entry in sources:
    if Path(entry['path']).name not in ['vision-sources.json', 'vision-activity-sources.json', 'vision-activity-heldout-sources.json']: continue
    for row in entry['metadata']['rows']:
        assert docci[row['example_id']] == row['cluster_id']
        assert row['family'] == 'vision:docci:cluster:' + row['cluster_id']
        assert row['example_id'] not in photos
        photos[row['example_id']] = row['split']
        if row['image'] in selected_assets: selected_photos[row['example_id']] = row['split']
        clusters[row['cluster_id']].add(row['split'])
        original_splits[row['sha256']].add(row['split'])
        if row.get('rgb_sha256'): rgb_splits[row['rgb_sha256']].add(row['split'])
assert all(len(v) == 1 for v in clusters.values())
assert all(len(v) == 1 for v in original_splits.values())
assert all(len(v) == 1 for v in rgb_splits.values())

chat_source = load('chat-sources.json')
raw_messages = lines('oasst2-selected-original-messages.jsonl')
assert digest(BASE / 'inputs/oasst2-selected-original-messages.jsonl') == chat_source['source_message_snapshot_sha256']
messages = {r['message_id']: r for r in raw_messages}
assert len(messages) == len(raw_messages)
chat = [r for r in manifest['rows'] if r['task'] == 'chat' and 'source_tree_id' in r]
trees = defaultdict(set)
for row in chat:
    trees[row['source_tree_id']].add(row['split'])
    ids = row['source_message_ids']
    assert len(ids) >= 2
    for i, identifier in enumerate(ids):
        message = messages[identifier]
        assert message['source_tree_id'] == row['source_tree_id']
        assert message['parent_id'] == (ids[i-1] if i else None)
assert all(len(v) == 1 for v in trees.values())
ledger = lines('chat-curation-ledger.jsonl')
assert digest(BASE / 'inputs/chat-curation-ledger.jsonl') == chat_source['ledger']['sha256']
assert len({r['source_tree_id'] for r in ledger}) == len(ledger)

# Run actual denominator contracts by extracting only calculation/API nodes.
def contract(filename, names, constants, extra):
    path = BASE / 'inputs' / filename
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names: nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets): nodes.append(node)
    ns = {'Counter': Counter, **extra}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

validation = contract('score_natural_v4_validation.py', {'require', 'case_id', 'expected_cases'}, {'WEIGHTS', 'DENOMINATOR'}, {})
vcases, vaudio = validation['expected_cases'](manifest, protocol)
test = contract('score_natural_v4_test.py', {'expected_cases'}, {'COUNTS'},
                {'require': validation['require'], 'case_id': validation['case_id'], 'MANUAL': {'photo_summary','photo_fact','text_chat','voice_typed_reference_chat','voice_actual_asr_chat'}})
tcases, taudio = test['expected_cases'](manifest)

result = {
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'cpu', 'scope': 'Existing data integrity and denominator contracts; no model loading, inference, training, or scoring'},
    'manifest_sha256': digest(BASE / 'inputs/manifest.json'),
    'read_pointers': ['manifest:/rows/*/{id,family,split,task,image,source_tree_id,source_message_ids,references,answer,predeclared_semantic_rubric}', 'manifest:/audio_rows/*/{id,family,split,task,audio}', 'manifest:/sources/*/{path,sha256,metadata/rows,metadata/sources,metadata/artifacts,metadata/audio_rows}', 'manifest:/archives/*/{path,bytes,sha256}', 'vision-sources:/upstream/clusters/sha256', 'chat-sources:/{source_message_snapshot_sha256,ledger/sha256}', 'oasst-originals:/*/{source_tree_id,message_id,parent_id}', 'docci-clusters:/*/{example_id,cluster_id}', 'chat-ledger:/*/{source_tree_id,source_message_ids,split}', 'protocol:/validation_denominators'],
    'row_counts': dict(Counter(r['split'] for r in manifest['rows'])),
    'audio_counts': dict(Counter(r['split'] for r in manifest['audio_rows'])),
    'family_counts': dict(Counter(next(iter(v)) for v in groups.values())),
    'family_cross_split_count': 0, 'image_path_cross_split_count': 0,
    'actual_source_asset_contract': {'validated_selected_assets': len(selected_assets), 'original_function': 'build_natural_v4_assets.validate_assets', 'bytes_and_sha256_matched': True},
    'immutable_archives': archive_checks,
    'docci': {'declared_source_photos': dict(Counter(photos.values())), 'selected_photos': dict(Counter(selected_photos.values())), 'unused_declared_source_photos': len(photos) - len(selected_photos), 'clusters': dict(Counter(next(iter(v)) for v in clusters.values())), 'cluster_mapping_matches_official_snapshot': True, 'cluster_cross_split_count': 0, 'declared_rgb_hash_cross_split_count': 0},
    'oasst': {'question_counts': dict(Counter(r['split'] for r in chat)), 'tree_counts': dict(Counter(next(iter(v)) for v in trees.values())), 'selected_raw_messages': len(messages), 'parent_paths_verified': True, 'tree_cross_split_count': 0},
    'denominators': {'validation_lm_cases': len(vcases), 'validation_recordings': len(vaudio), 'test_lm_cases': len(tcases), 'test_recordings': len(taudio), 'validation_groups': dict(Counter(r['group'] for r in vcases.values())), 'test_groups': dict(Counter(r['group'] for r in tcases.values()))},
    'limitations': ['Physical bytes checked against frozen source declarations; declared RGB hashes compared, not all image pixels decoded again.', 'Grouping does not establish capture-session, photographer, semantic independence, or absence from upstream pretraining.', 'No existing model result or accuracy claim was read or recomputed; no new model evaluation was performed.']
}
(BASE / 'execution/manifest-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
