"""T.10 supplement: two CPU modal forwards, tagging fixture, archive inspection."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import modalities, run
from scripts.course_experiments.common import Context
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder

torch.set_num_threads(2)
torch.manual_seed(42)
CACHE = Path('/workspace/work/reference/omni-technical-t-10-v1/modal-supplement')
CACHE.mkdir(parents=True, exist_ok=True)
summary = {'created_at': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
    'scope': 'Untrained CPU forward/alignment and isolated metadata tagging only; no formal training.'}
ctx = Context('cpu', CACHE, CACHE, ROOT / 'assets/training')
teacher = MultiModalLM(TinyLM(ModelConfig(width=64, layers=2, heads=1, max_length=128)))
student = MultiModalLM(TinyLM(ModelConfig(width=32, layers=1, heads=2, max_length=128)), vision_width=8, audio_width=8)
student.vision = VisionEncoder(width=8, image_size=16, patch_size=8)
row = {'modality': 'joint', 'color': 'blue', 'shape': 'square', 'offset': 0,
       'frequency': 180, 'question': 'joint?', 'answer': 'square,low', 'family': 'blue:square:0:180'}
ids, labels, image, waveform, _ = modalities.modal_inputs(row, ctx)
aligned = {}
for name, model in [('teacher', teacher), ('student', student)]:
    with torch.no_grad():
        output = model(ids, labels, image=image, waveform=waveform)
    mask = output['labels'][0] != -100
    aligned[name] = {'vision_positions': (model.vision.image_size // model.vision.patch_size) ** 2,
        'parameters': sum(p.numel() for p in model.parameters()), 'answer_rows': mask.nonzero().flatten().tolist(),
        'answer_ids': output['labels'][0, mask].tolist(), 'logit_shape': list(output['logits'].shape)}
assert aligned['teacher']['answer_ids'] == aligned['student']['answer_ids']
assert aligned['teacher']['vision_positions'] == 16 and aligned['student']['vision_positions'] == 4
assert aligned['student']['parameters'] < aligned['teacher']['parameters']
summary['modal_alignment'] = aligned

# Execute _modal's dependency calls with explicit local weight loaders, no IO.
dependencies = []
def dependency(identifier, filename='model.pt'):
    dependencies.append([identifier, filename])
    return CACHE / identifier / filename
def fake_load(path, **kwargs):
    return {'encoder': getattr(teacher, Path(path).stem).state_dict()}
with patch.object(Context, 'dependency', lambda self, *a, **k: dependency(*a, **k)), \
     patch.object(modalities, 'load_checkpoint', return_value=(teacher.language, {})), \
     patch.object(modalities.torch, 'load', side_effect=fake_load):
    modalities._modal(ctx, 'joint')
assert dependencies == [['sft', 'model.pt'], ['encoders', 'vision.pt'], ['encoders', 'audio.pt']]
summary['joint_model_constructor_dependencies'] = dependencies
fake_module = SimpleNamespace(run_joint=lambda context: {'audit_fixture': 'No training, no evaluation'})
with patch.object(run.importlib, 'import_module', return_value=fake_module):
    result = run.execute('joint', 'cpu', CACHE / 'tagging-only', CACHE, ROOT / 'assets/training', revision='audit-fixture')
assert result['evidence_status'] == 'interface_smoke_only'
summary['cpu_joint_tagging_fixture'] = {'evidence_status': result['evidence_status'],
    'scope': 'Real runner metadata branch, experiment body replaced before execution; not a joint run.'}

summary['archive_metadata'] = {}
for name in ('sft', 'style', 'moe', 'distillation', 'joint', 'multimodal_distillation'):
    path = ROOT / f'docs/course-experiments/results/{name}.json'
    d = json.loads(path.read_text())
    entry = {'modal': d['modal'], 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if name in ('sft', 'style', 'joint'):
        entry['main_training'] = {k: v for k, v in d['results']['training'].items() if k in (
            'steps', 'records', 'batch_size', 'effective_tokens', 'effective_targets', 'parameters')}
    if name == 'moe':
        key = d['results']['teacher_variant']
        entry['teacher_variant'] = key
        entry['main_training'] = {k: v for k, v in d['results']['variants'][key]['training'].items() if k in (
            'steps', 'batch_size', 'effective_tokens', 'optimizer_updates', 'requested_steps')}
    if name in ('distillation', 'multimodal_distillation'):
        entry['pairwise_test_transitions'] = {}
        for task_name, task in d['results']['tasks'].items():
            prefix = 'w32_' if name == 'distillation' else ''
            left, right = (task['runs'][prefix + method]['test'] for method in ('ce', 'ce_kl'))
            key = 'generated_samples' if name == 'distillation' else 'samples'
            left_rows, right_rows = left[key], right[key]
            exact_key = 'exact' if name == 'distillation' else 'exact_match'
            changes = {'newly_correct': 0, 'newly_wrong': 0, 'unchanged_correctness': 0}
            for a, b in zip(left_rows, right_rows, strict=True):
                if a[exact_key] == b[exact_key]: changes['unchanged_correctness'] += 1
                elif b[exact_key]: changes['newly_correct'] += 1
                else: changes['newly_wrong'] += 1
            entry['pairwise_test_transitions'][task_name] = changes
    summary['archive_metadata'][name] = entry
summary['public_bundle_file_names'] = {}
for name in ('sft', 'style', 'moe', 'distillation', 'vqa', 'joint', 'multimodal_distillation'):
    path = ROOT / f'docs/course-experiments/releases/{name}.json'
    names = [item['path'] for item in json.loads(path.read_text())['files']]
    assert 'dataset.json' not in names
    summary['public_bundle_file_names'][name] = {'files': names, 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
OUT = Path(__file__).with_name('modal-and-archive-summary.json')
assert not OUT.exists()
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(OUT.relative_to(ROOT)), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}))
