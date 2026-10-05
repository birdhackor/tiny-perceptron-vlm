from pathlib import Path
import hashlib
import json
import platform
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.multimodal import VisionEncoder
from scripts.prepare_ocr import draw_digits, GLYPHS

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device('cpu')
sha = lambda x: hashlib.sha256(x).hexdigest()
raw = (ROOT / 'docs/course-experiments/results/ocr.json').read_bytes()
# Preserve the complete original input without displaying unrelated fields or notes.
(ART / 'original/raw-ocr.json').write_bytes(raw)
j = json.loads(raw)
inspected = []
def pointer(path):
    inspected.append(path)
    value = j
    for key in path.split('/')[1:]:
        value = value[key.replace('~1', '/').replace('~0', '~')]
    return value

glyphs = {'0': ['111', '101', '101', '101', '111'], '1': ['010', '110', '010', '010', '111']}
images = {label: torch.tensor([[int(c) for c in row] for row in rows], dtype=torch.float32) for label, rows in glyphs.items()}
for label, image in images.items():
    assert tuple(image.shape) == (5, 3) and image.dtype == torch.float32
    assert not image.requires_grad and image.grad_fn is None
    assert set(image.flatten().tolist()) == {0., 1.}

svg = ET.fromstring((ROOT / 'course/figures/rewrite-11-glyph-labels.svg').read_bytes())
rects = [q.attrib for q in svg.findall('{http://www.w3.org/2000/svg}rect') if q.attrib.get('width') == '44']
assert len(rects) == 30
svg_matrix = {}
for label, start_x in [('0', 67), ('1', 367)]:
    matrix = [[None] * 3 for _ in range(5)]
    for q in rects:
        x, y = int(q['x']), int(q['y'])
        if start_x <= x < start_x + 132:
            assert q['fill'] in {'#fff', '#111820'}
            matrix[(y - 85) // 44][(x - start_x) // 44] = int(q['fill'] == '#fff')
    assert matrix == images[label].int().tolist()
    svg_matrix[label] = matrix

changed_rows = list(glyphs['1']); changed_rows[0] = '110'
changed = torch.tensor([[int(c) for c in row] for row in changed_rows], dtype=torch.float32)
delta = changed - images['1']
assert delta.nonzero().tolist() == [[0, 0]] and delta[0, 0].item() == 1
assert changed.shape == images['1'].shape

canvas = torch.zeros(16, 16)
canvas[5:10, 6:9] = images['1']
rgb = canvas.unsqueeze(0).repeat(3, 1, 1)
assert tuple(rgb.shape) == (3, 16, 16)
assert all(torch.equal(rgb[c, 5:10, 6:9], images['1']) for c in range(3))
encoder = VisionEncoder(width=8)
with torch.no_grad():
    features = encoder(rgb.unsqueeze(0))
assert tuple(features.shape) == (1, 16, 8)
try:
    images['1'].reshape(3, 16, 16)
    raise AssertionError('reshape should reject 15 != 768 entries')
except RuntimeError as e:
    reshape_error = str(e)
try:
    encoder(images['1'].unsqueeze(0))
    raise AssertionError('encoder should reject monochrome 5x3 input')
except ValueError as e:
    entry_error = str(e)

# Provided class targets drive differentiation even when a human says the target is wrong.
# This is one bounded gradient calculation; no optimizer or parameter update.
logits = torch.tensor([[2., -1.]], requires_grad=True)
before = logits.detach().clone()
loss = F.cross_entropy(logits, torch.tensor([1]))
loss.backward()
assert logits.grad is not None and logits.grad.norm().item() > 0
assert torch.equal(logits.detach(), before)
assert logits.argmax(-1).item() == 0
assert ('0' == '0') and not ('00' == '0')

manifest = {}
groups = {}
for split in ['train', 'validation', 'test']:
    records = pointer(f'/results/data/splits/{split}/records')
    count = pointer(f'/results/data/splits/{split}/count')
    expected_hash = pointer(f'/results/data/splits/{split}/sha256')
    assert count == len(records)
    assert sha(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()) == expected_hash
    groups[split] = {r['family'] for r in records}
    by_family = {}
    for r in records:
        assert r['digits'] == r['answer'] == r['family']
        assert r['question'] == 'read digits' and r['modality'] == 'vision'
        by_family.setdefault(r['family'], []).append(r['offset'])
        fixture = draw_digits(r['digits'], r['offset'])
        assert fixture.mode == 'RGB' and fixture.size == (16, 16)
        assert set(fixture.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert all(sorted(v) == [-1, 0, 1] for v in by_family.values())
    manifest[split] = {'samples': count, 'families': len(groups[split]), 'offsets': [-1, 0, 1], 'records_sha256': expected_hash}
assert all(not groups[a] & groups[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
assert set.union(*groups.values()) == {str(v) for v in range(100)}
seed = pointer('/seed')
assert pointer('/results/data/seed') == seed == 42
families = list(range(100)); random.Random(seed).shuffle(families)
assert groups['train'] == set(map(str, families[:80]))
assert groups['validation'] == set(map(str, families[80:90]))
assert groups['test'] == set(map(str, families[90:]))
test_records = j['results']['data']['splits']['test']['records']
samples = pointer('/results/test/samples')
tok = ByteTokenizer()
computed_exact = []
sample_receipt = []
for index, (sample, record) in enumerate(zip(samples, test_records, strict=True)):
    assert sample['row'] == index
    assert sample['family'] == record['family'] and sample['target'] == record['answer']
    ids = sample['generated_ids']
    raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    exact = raw_ids == tok.encode(record['answer'])
    assert exact == sample['exact_match']
    assert tok.decode(raw_ids) == sample['generated']
    assert (tok.eos_id in ids) == sample['eos']
    assert sum(t < 8 for t in raw_ids) == sample['invalid_special_tokens']
    assert sample['generation_error'] is None
    computed_exact.append(exact)
    sample_receipt.append({'row': index, 'family': sample['family'], 'offset': record['offset'], 'target': sample['target'],
                           'generated': sample['generated'], 'generated_ids': ids, 'raw_token_exact': exact, 'eos': sample['eos']})
assert len(samples) == pointer('/results/test/examples') == 30
assert sum(computed_exact) == pointer('/results/test/correct') == 2
assert sum(computed_exact) / len(samples) == pointer('/results/test/exact_match')
initial = pointer('/results/training/initial_loss'); final = pointer('/results/training/final_loss')
assert final < initial
history = pointer('/results/training/history')
steps = pointer('/results/training/steps')
assert len(history) == steps == 500 and [r['step'] for r in history] == list(range(1, 501))
effective = sum(r['effective_targets'] for r in history)
assert effective == pointer('/results/training/effective_targets') == pointer('/results/training/effective_tokens') == 11584
assert all(16 <= r['effective_targets'] <= 24 for r in history)
assert any(r['grad_norm'] > 0 for r in history) == pointer('/results/training/nonzero_gradient_seen')
assert pointer('/results/training/weights_changed') is True
config = pointer('/results/training/config'); modal_config = pointer('/results/training/modal_config')
assert modal_config['image_size'] == 16 and modal_config['patch_size'] == 4
assert pointer('/results/training/cpu_smoke') is False
device = pointer('/device'); revision = pointer('/revision')
experiment_torch = pointer('/torch_version'); experiment_python = pointer('/python_version')
assert device == 'cuda'
input_code = {}
for f in ['scripts/course_experiments/modalities.py', 'tiny_perceptron/multimodal.py', 'tiny_perceptron/data.py']:
    expected_hash = pointer('/code_sha256/' + f.replace('~','~0').replace('/','~1'))
    assert sha((ROOT / f).read_bytes()) == expected_hash
    input_code[f] = {'sha256': expected_hash, 'matches_reported_historical_code': True}
historical_generator = subprocess.run(['git', 'show', revision + ':scripts/prepare_ocr.py'], cwd=ROOT, capture_output=True, check=True).stdout
assert historical_generator == (ROOT / 'scripts/prepare_ocr.py').read_bytes()
assert historical_generator == (ART / 'original/prepare_ocr-22a0bb1.py').read_bytes()
assert all(len(g) == 7 and all(len(row) == 5 for row in g) for g in GLYPHS)
result = {
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version),
                    'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'cuda_build': str(torch.version.cuda), 'threads': str(torch.get_num_threads())},
    'original_result_sha256': sha(raw), 'inspected_json_pointers': inspected,
    'glyph_matrices': {k: v.int().tolist() for k, v in images.items()}, 'glyph_white_counts': {k: int(v.sum()) for k,v in images.items()},
    'svg_grid_matches': svg_matrix, 'exercise': {'label': '1', 'changed_pixel_yx': delta.nonzero().tolist(), 'white_count_before': 8, 'white_count_after': 9},
    'rgb_entry': {'pixel_entries_before': 15, 'pixel_entries_rgb': rgb.numel(), 'shape': list(rgb.shape), 'features_shape': list(features.shape),
                  'reshape_rejection': reshape_error, 'encoder_rejection': entry_error},
    'bounded_gradient': {'given_target': 1, 'argmax_class': 0, 'loss': loss.item(), 'gradient': logits.grad.tolist(), 'values_unchanged': True, 'optimizer_steps': 0},
    'classification_vs_generation': {'class_0_equals_target_0': True, 'string_00_equals_target_0': False},
    'historical_experiment': {'revision': revision, 'device': device, 'torch': experiment_torch, 'python': experiment_python, 'seed': seed,
                              'config': config, 'modal_config': modal_config, 'split_counts': manifest, 'test_samples': len(samples),
                              'test_distinct_digit_strings': len(groups['test']), 'test_raw_token_exact': sum(computed_exact),
                              'test_exact_match': sum(computed_exact)/len(samples), 'initial_fixed_probe_loss': initial,
                              'final_fixed_probe_loss': final, 'optimizer_steps': steps, 'effective_targets_including_eos': effective,
                              'matching_code_hashes': input_code, 'generator_sha256_at_reported_revision': sha(historical_generator),
                              'font': 'fixed 5x7 GLYPHS, same glyph lookup for all splits',
                              'scope': 'Recalculated existing result measurements and raw-token scoring; no model checkpoint loaded, trained, re-evaluated or saved. No alternate font or handwriting in the original generator.'},
    'sample_receipt': sample_receipt,
    'scope': 'Original fence data and SVG pixels, one-pixel font variant, bounded RGB/shape forward and one supplied-label gradient calculation; no learned OCR accuracy claim from new CPU checks.'
}
(ART / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['sample_receipt', 'inspected_json_pointers', 'svg_grid_matches']}, ensure_ascii=False, indent=2))
