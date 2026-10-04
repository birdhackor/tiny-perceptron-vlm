"""Bounded independent verification of the current section, without a model."""
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.multimodal import patchify, scene, unpatchify
from tiny_perceptron.natural_concepts import picture_order_report, swapped_picture

print('environment', json.dumps({'python': platform.python_version(), 'executable': sys.executable, 'torch': torch.__version__, 'device': 'cpu', 'cuda_available': torch.cuda.is_available()}))
report = picture_order_report()
print('different_pixels', report['different_pixels'])
print('same_4_by_4_summary', report['same_4_by_4_summary'])
print('same_pixel_sequence', report['same_pixel_sequence'])
assert report == {'different_pixels': 128, 'same_4_by_4_summary': True, 'same_pixel_sequence': False}
first, second = swapped_picture()
positions = (first != second).any(dim=0)
print('shape_dtype', list(first.shape), str(first.dtype))
print('different_color_positions', int(positions.sum()))
print('changed_channels', [(first[c] != second[c]).sum().item() for c in range(3)])
assert int(positions.sum()) == 64
assert [int((first[c] != second[c]).sum()) for c in range(3)] == [64, 0, 64]
assert torch.equal(first.sum((1, 2)), second.sum((1, 2)))
print('channel_totals', first.sum((1, 2)).tolist())

# Independent Python sums, without adaptive pooling or flattening logic.
def cell_means(image):
    values = image.tolist()
    return [[[sum(values[c][y][x] for y in range(8*r, 8*r+8) for x in range(8*q, 8*q+8))/64 for q in range(4)] for r in range(4)] for c in range(3)]

manual_first = cell_means(first)
manual_second = cell_means(second)
assert manual_first == manual_second
assert F.adaptive_avg_pool2d(first[None], (4, 4))[0].tolist() == manual_first
print('nonzero_pool_cells', [{'row': r, 'column': q, 'rgb_mean': [manual_first[c][r][q] for c in range(3)]} for r in range(4) for q in range(4) if any(manual_first[c][r][q] for c in range(3))])
assert [manual_first[c][1][0] for c in range(3)] == [0.5, 0.0, 0.5]

# Exercise: move red into the adjacent 8x8 cell, retaining blue in the first.
moved = first.clone()
moved[0, 8:16, 0:4] = 0
moved[0, 8:16, 8:12] = 1
manual_moved = cell_means(moved)
assert manual_moved != manual_first
assert [manual_moved[c][1][0] for c in range(3)] == [0.0, 0.0, 0.5]
assert [manual_moved[c][1][1] for c in range(3)] == [0.5, 0.0, 0.0]
print('exercise_adjacent_means', [[manual_moved[c][1][q] for c in range(3)] for q in (0, 1)])
assert moved.sum((1, 2)).tolist() == [32.0, 0.0, 32.0]
print('exercise_global_totals_unchanged', moved.sum((1, 2)).tolist())

# Read-prerequisite sequence checks include independently known locations.
image = scene()[None]
patches = patchify(image, 4)
assert tuple(patches.shape) == (1, 16, 48)
assert torch.equal(image, unpatchify(patches, 3, 16, 16, 4))
ramp = torch.arange(3*16*16).reshape(1, 3, 16, 16)
indexed = patchify(ramp, 4)
for row, col in ((0, 0), (0, 1), (1, 0), (3, 3)):
    expected = [ramp[0, c, 4*row+y, 4*col+x].item() for c in range(3) for y in range(4) for x in range(4)]
    assert indexed[0, 4*row+col].tolist() == expected
print('prerequisite_patch_shape', list(patches.shape), 'direct_row_channel_mapping_verified', True)

# Only the scene/action/relation original run records linked from 11.14.
scores_path = Path('docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json')
subsets_path = scores_path.parent.parent / 'descriptive-subsets.json'
scores = json.loads(scores_path.read_text())
subsets = json.loads(subsets_path.read_text())
assert hashlib.sha256(scores_path.read_bytes()).hexdigest() == subsets['scores_sha256']
decisions = {r['case_id']: r for r in scores['case_decisions']}
photo_rows = [r for r in decisions.values() if r['group'] in ('photo_summary', 'photo_fact')]
assert len(photo_rows) == 126
for row in photo_rows:
    assert row['passed'] == (row['decoder_complete'] and row['rubric_passed'])
recomputed = {}
for group in ('photo_summary', 'photo_fact'):
    rows = [r for r in photo_rows if r['group'] == group]
    recomputed[group] = {'passed': sum(r['passed'] for r in rows), 'denominator': len(rows), 'complete': sum(r['decoder_complete'] for r in rows)}
    assert recomputed[group]['passed'] == scores['correct_counts'][group]
    assert len(rows) == scores['denominators'][group]
for group in ('action', 'relation'):
    record = subsets['photo_fact_subsets'][group]
    rows = record['cases']
    assert len(rows) == len({r['case_id'] for r in rows})
    assert all(decisions[r['case_id']]['group'] == 'photo_fact' and r['passed'] == decisions[r['case_id']]['passed'] for r in rows)
    recomputed[group] = {'passed': sum(r['passed'] for r in rows), 'denominator': len(rows)}
    assert recomputed[group] == {'passed': record['correct'], 'denominator': record['denominator']}
photo_ids = [r['case_id'].rsplit('/', 1)[0] for r in photo_rows]
assert len(set(photo_ids)) == 42 and Counter(photo_ids).most_common(1)[0][1] == 3
print('fixed_run_photo_counts', json.dumps(recomputed, sort_keys=True))
print('fixed_run_shared_photos', len(set(photo_ids)), 'not_independent_answers', len(photo_rows))
manifest_path = Path('docs/natural-assistant/v4/manifest.json')
generations_path = scores_path.parent.parent / 'generations-base.json'
manifest = json.loads(manifest_path.read_text())
generations = {r['id']: r for r in json.loads(generations_path.read_text())}
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == subsets['manifest_sha256']
assert hashlib.sha256(generations_path.read_bytes()).hexdigest() == scores['artifact_binding']['generation_file']['sha256']
inputs = {r['id']: r for r in manifest['rows']}
selected_input_rows = [inputs[r['case_id'].removeprefix('scene:')] for r in photo_rows]
for row in photo_rows:
    rid = row['case_id'].removeprefix('scene:')
    source, gen = inputs[rid], generations[rid]
    assert source['split'] == gen['split'] == 'test'
    assert source['image'] == gen['image'] and source['user'] == gen['user']
    assert row['decoder_complete'] == gen['ended_with_eos'] and not gen['truncated']
for typ in ('action', 'relation'):
    direct_ids = {'scene:' + r['id'] for r in selected_input_rows if r['references']['qa_type'] == typ}
    listed_ids = {r['case_id'] for r in subsets['photo_fact_subsets'][typ]['cases']}
    assert direct_ids == listed_ids
    print('independent_manifest_subset', typ, 'rows', len(direct_ids), 'passed', sum(decisions[r]['passed'] for r in direct_ids))
test_images = {r['image'] for r in selected_input_rows}
test_families = {r['family'] for r in selected_input_rows}
non_test_images = {r.get('image') for r in manifest['rows'] if r['split'] != 'test'}
non_test_families = {r['family'] for r in manifest['rows'] if r['split'] != 'test'}
assert not test_images & non_test_images and not test_families & non_test_families
print('selected_test_photos_no_current_train_validation_image_family_overlap', len(test_images))
print('selected_original_file_sha256', json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (scores_path, subsets_path, manifest_path, generations_path)}, sort_keys=True))
print('fixed_run_identity', json.dumps({k: scores['artifact_binding'][k] for k in ('test_run_id', 'execution_revision', 'model', 'model_revision', 'selected_variant', 'adapters')}, sort_keys=True))
print('all_bounded_checks_passed', True)
