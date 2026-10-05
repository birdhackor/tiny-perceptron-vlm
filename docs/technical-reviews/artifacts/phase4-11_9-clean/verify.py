"""Independent, bounded CPU inspection for lesson 11.9; no model evaluation/training."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from tiny_perceptron.multimodal import scene
from tiny_perceptron.data import ByteTokenizer

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def pointer(obj, path):
    for key in path.strip('/').split('/'):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj

raw_path = ROOT / 'docs/course-experiments/results/vision_ablation.json'
raw_bytes = raw_path.read_bytes()
raw = json.loads(raw_bytes)
selected = {}
paths = ['/revision', '/device', '/seed', '/torch_version', '/python_version', '/gpu', '/step_scale']
for name in ['config', 'modal_config', 'steps', 'effective_tokens', 'effective_targets', 'weights_changed',
             'nonzero_gradient_seen', 'cpu_smoke']:
    paths.append('/results/training/' + name)
for split in ['train', 'validation', 'test']:
    paths.extend('/results/data/splits/' + split + '/' + name for name in ['count', 'sha256', 'records'])
sample_fields = ['row', 'family', 'question', 'target', 'generated', 'generated_ids', 'exact_match', 'eos',
                 'generation_error', 'invalid_special_tokens', 'donor_row']
for intervention in ['none', 'edge_cropped']:
    base = '/results/interventions/' + intervention
    paths.extend(base + '/' + name for name in ['examples', 'correct', 'exact_match', 'effective_tokens', 'groups'])
    for i in range(len(pointer(raw, base + '/samples'))):
        paths.extend(base + '/samples/' + str(i) + '/' + name for name in sample_fields)
for path in paths:
    selected[path] = pointer(raw, path)

# Read only the source functions that construct inputs; never inspect result note constants.
source = ROOT / 'scripts/course_experiments/modalities.py'
tree = ast.parse(source.read_bytes())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['_vision_records', '_media']]
namespace = {'scene': scene, 'torch': torch, 'np': np, 'Image': Image}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source) + ':selected-functions', 'exec'), namespace)
splits = namespace['_vision_records'](('shape?', 'color?'))
for split, records in splits.items():
    assert records == selected['/results/data/splits/' + split + '/records']
    assert len(records) == selected['/results/data/splits/' + split + '/count']
    serialized = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    assert sha(serialized) == selected['/results/data/splits/' + split + '/sha256']
for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
    assert not ({r['family'] for r in splits[first]} & {r['family'] for r in splits[second]})

counts = []
for offset in [7, 0, 8]:
    image = scene('red', 'square', offset=offset)
    crop = image[:, 4:12, 4:12]
    occupied = torch.nonzero(image[0] > 0)
    measured = {'offset': offset, 'center_x': 8 + offset, 'original_shape': list(image.shape),
                'crop_shape': list(crop.shape), 'original_red_pixels': int((image[0] > 0).sum()),
                'crop_red_pixels': int((crop[0] > 0).sum()),
                'original_y_extent': [int(occupied[:, 0].min()), int(occupied[:, 0].max())],
                'original_x_extent': [int(occupied[:, 1].min()), int(occupied[:, 1].max())]}
    counts.append(measured)
assert [r['original_red_pixels'] for r in counts] == [45, 81, 36]
assert [r['crop_red_pixels'] for r in counts] == [8, 64, 0]
assert list(range(16))[4:12] == list(range(4, 12))

# Original crop material for each original test row: shifted offset 7, then crop and RGB resize.
materials = []
ctx = SimpleNamespace(device='cpu')
for i, row in enumerate(splits['test']):
    image, waveform = namespace['_media'](dict(row, offset=7), ctx)
    assert waveform is None
    crop = image[:, 4:12, 4:12]
    pixels = (crop.permute(1, 2, 0).numpy() * 255).astype('uint8')
    resized = Image.fromarray(pixels).resize((16, 16))
    loaded = torch.from_numpy(np.array(resized.convert('RGB').resize((16, 16)), copy=True)).permute(2, 0, 1).float() / 255
    active = ['red', 'green', 'blue'].index(row['color'])
    assert (crop[active] > 0).any() and (loaded[active] > 0).any()
    assert not loaded[[c for c in range(3) if c != active]].any()
    materials.append({'row': i, 'color': row['color'], 'shape': row['shape'], 'target': row['answer'],
                      'question': row['question'], 'original_manifest_offset': row['offset'],
                      'intervention_offset': 7, 'crop_positive_pixels': int((crop[active] > 0).sum()),
                      'resized_input_shape': list(loaded.shape)})
assert sorted(set(m['crop_positive_pixels'] for m in materials)) == [1, 8]

tok = ByteTokenizer()
aggregates = {}
for intervention in ['none', 'edge_cropped']:
    z = raw['results']['interventions'][intervention]
    groups = {}
    targets = 0
    for row, sample in zip(splits['test'], z['samples'], strict=True):
        assert sample['question'] == row['question'] and sample['target'] == row['answer']
        ids = sample['generated_ids']
        answer_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        answer = tok.decode(answer_ids)
        exact = answer_ids == tok.encode(row['answer'])
        assert answer == sample['generated'] and exact == sample['exact_match']
        assert (tok.eos_id in ids) == sample['eos']
        group = groups.setdefault(row['question'], {'correct': 0, 'count': 0})
        group['correct'] += int(exact)
        group['count'] += 1
        targets += len(tok.encode(row['answer'])) + 1
    correct = sum(g['correct'] for g in groups.values())
    assert groups == z['groups'] and len(z['samples']) == z['examples'] == 12
    assert correct == z['correct'] == 9 and correct / 12 == z['exact_match'] == 0.75
    assert targets == z['effective_tokens'] == 72
    assert all(s['generated'] == 'circle' for s in z['samples'] if s['question'] == 'shape?')
    aggregates[intervention] = {'groups': groups, 'correct': correct, 'examples': 12, 'exact_match': correct / 12,
                                'target_tokens_including_eos': targets, 'shape_outputs_all_circle': True}

# Constructive information-loss checks: distinct source images can have the same crop.
red = scene('red', 'square', offset=8)
blue = scene('blue', 'square', offset=8)
assert not torch.equal(red, blue)
red_crop, blue_crop = red[:, 4:12, 4:12], blue[:, 4:12, 4:12]
assert torch.equal(red_crop, blue_crop) and not red_crop.any()
red_padded = torch.nn.functional.pad(red_crop, (4, 4, 4, 4))
blue_padded = torch.nn.functional.pad(blue_crop, (4, 4, 4, 4))
assert torch.equal(red_padded, blue_padded) and red_padded.shape == red.shape
red_resized = Image.fromarray((red_crop.permute(1, 2, 0).numpy() * 255).astype('uint8')).resize((16, 16))
blue_resized = Image.fromarray((blue_crop.permute(1, 2, 0).numpy() * 255).astype('uint8')).resize((16, 16))
assert red_resized.tobytes() == blue_resized.tobytes()
white_line = Image.fromarray(np.array([[255, 0], [255, 0]], dtype='uint8'))
checker = Image.fromarray(np.array([[255, 0], [0, 255]], dtype='uint8'))
line_small = white_line.resize((1, 1), Image.Resampling.BOX)
checker_small = checker.resize((1, 1), Image.Resampling.BOX)
assert white_line.tobytes() != checker.tobytes() and line_small.tobytes() == checker_small.tobytes()
assert line_small.getpixel((0, 0)) == 128

# Match the figure's raw per-pixel geometry with the scene and slice.
svg = ROOT / 'course/figures/rewrite-11-crop-evidence.svg'
root = ET.fromstring(svg.read_bytes())
rects = root.findall('{http://www.w3.org/2000/svg}rect')
left = [r for r in rects if r.get('width') == '14' and r.get('height') == '14']
right = [r for r in rects if r.get('width') == '22' and r.get('height') == '22']
assert len(left) == 256 and len(right) == 64
original = scene('red', 'square', offset=7)[0]
for r in left:
    x, y = (int(r.get('x')) - 53) // 14, (int(r.get('y')) - 129) // 14
    assert (r.get('fill') == '#ff0000') == bool(original[y, x] > 0)
for r in right:
    x, y = (int(r.get('x')) - 398) // 22, (int(r.get('y')) - 183) // 22
    assert (r.get('fill') == '#ff0000') == bool(original[4:12, 4:12][y, x] > 0)
box = next(r for r in rects if r.get('stroke') == '#ffdc67')
assert [int(box.get(k)) for k in ['x', 'y', 'width', 'height']] == [109, 185, 112, 112]

save('raw-selected-pointers.json', {'source_path': str(raw_path.relative_to(ROOT)), 'source_full_sha256': sha(raw_bytes),
                                  'inspection': 'Only named pointers below; results/crop_note and author notes/review/scope corrections were not inspected.',
                                  'pointers': selected})
result = {'environment': {'python': sys.version, 'torch': str(torch.__version__), 'pillow': pillow_version,
                          'numpy': np.__version__, 'device': 'cpu', 'cuda_build': str(torch.version.cuda),
                          'cuda_available': str(torch.cuda.is_available()), 'platform': platform.platform()},
          'source_sha256': sha(raw_bytes), 'counts': counts, 'materials': materials, 'aggregates': aggregates,
          'information_loss': {'different_original_colors_same_zero_crop': True, 'padding_cannot_distinguish': True,
                               'resize_cannot_distinguish': True, 'BOX_two_distinct_2x2_patterns_same_1x1_value': 128},
          'figure': {'original_cells_exact': 256, 'cropped_cells_exact': 64, 'box_xywh': [109, 185, 112, 112]},
          'scope': 'Raw stored results recalculated; original input rules executed. No trained model loaded, retested, or trained; no model/data download or .pt save.'}
save('verification.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
