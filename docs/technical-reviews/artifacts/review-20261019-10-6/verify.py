"""核對接頭算例、特徵寬度契約與相同輸入的限制。"""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(0)
vision = torch.randn(1, 16, 8)
projector = nn.Linear(8, 12)
projected = projector(vision)
manual = vision @ projector.weight.T + projector.bias
per_position = torch.stack([projector(vision[:, i, :]) for i in range(16)], dim=1)
torch.testing.assert_close(projected, manual, rtol=1e-6, atol=1e-6)
torch.testing.assert_close(projected, per_position, rtol=1e-6, atol=1e-6)
assert projected.shape == (1, 16, 12)
assert projector.weight.shape == (12, 8)

toy = nn.Linear(2, 3)
with torch.no_grad():
    toy.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
    toy.bias.zero_()
toy_result = toy(torch.tensor([1., 2.]))
assert toy_result.tolist() == [1., 2., 3.]
assert toy_result[2].item() == (toy_result[0] + toy_result[1]).item()

same_input = vision.clone()
same_output = torch.equal(projector(vision), projector(same_input))
assert same_output

short_projector = nn.Linear(8, 10)
short_output = short_projector(vision)
assert short_output.shape == (1, 16, 10)
assert short_projector.weight.shape == (10, 8)
lm = TinyLM(ModelConfig(width=12, max_length=32))
positions = lm.position(torch.arange(16))
assert positions.shape == (16, 12)
failures = {}
for name, operation in {
    'position_addition': lambda: short_output + positions,
    'language_forward': lambda: lm(embeddings=short_output),
}.items():
    try:
        operation()
    except RuntimeError as error:
        failures[name] = {'type': type(error).__name__, 'message': str(error)}
    else:
        raise AssertionError('特徵寬度不符時應失敗')

with torch.no_grad():
    valid_logits = lm(embeddings=projected)['logits']
    encoder_output = VisionEncoder(width=8)(torch.zeros(1, 3, 16, 16))
assert encoder_output.shape == (1, 16, 8)
assert valid_logits.shape == (1, 16, lm.config.vocab_size)
wrapped = MultiModalLM(lm, vision_width=8)
assert wrapped.image_projector.in_features == 8
assert wrapped.image_projector.out_features == lm.config.width == 12

source = ROOT / 'course/chapters/10.md'
section = re.search(rb'(?m)^## 10\.6 .*?(?=^## |\Z)', source.read_bytes(), re.S).group(0)
frozen = ROOT / 'docs/course-review-20261019/freeze-01/sources/10.6.md'
assert section == frozen.read_bytes()
assert hashlib.sha256(section).hexdigest() == '4dcd7219828d7ce586a409fe3760b9f0578b04ea3815db84f5f25c0e630bf981'

repository = {}
for name, names in {
    'tiny_perceptron/model.py': ['class TinyLM', 'self.position = None', 'x = x + self.position(positions)'],
    'tiny_perceptron/multimodal.py': ['class MultiModalLM', 'self.image_projector = nn.Linear(vision_width, language_model.config.width)'],
    'course/chapters/10.md': ['self.image_projector = nn.Linear(vision_width, language_model.config.width)'],
    'course/chapters/04.md': ['class TinyLM', 'x = self.embedding(ids) + self.position(positions)'],
}.items():
    path = ROOT / name
    lines = path.read_text().splitlines()
    repository[name] = {
        'sha256': digest(path),
        'excerpts': [line for line in lines if any(needle in line for needle in names)],
    }

print(json.dumps({
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-6/verify.py',
    'result': '所有形狀、逐位置計算、算例、相同輸入與寬度契約核對通過。',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'device': str(vision.device), 'dtype': str(vision.dtype)},
    'source_sha256': hashlib.sha256(section).hexdigest(),
    'linear': {'input_shape': list(vision.shape), 'output_shape': list(projected.shape), 'weight_shape': list(projector.weight.shape), 'bias_shape': list(projector.bias.shape), 'explicit_formula_max_error': (projected-manual).abs().max().item(), 'per_position_max_error': (projected-per_position).abs().max().item()},
    'toy': {'input': [1, 2], 'weight': toy.weight.tolist(), 'bias': toy.bias.tolist(), 'output': toy_result.tolist(), 'third_is_sum': True},
    'same_input_same_output': same_output,
    'width_mismatch': {'input_shape': list(short_output.shape), 'weight_shape': list(short_projector.weight.shape), 'position_shape': list(positions.shape), 'failures': failures},
    'matched_width': {'logits_shape': list(valid_logits.shape), 'encoder_shape': list(encoder_output.shape), 'projector_in_features': wrapped.image_projector.in_features, 'projector_out_features': wrapped.image_projector.out_features, 'language_width': lm.config.width},
    'repository': repository,
}, ensure_ascii=False, indent=2))
