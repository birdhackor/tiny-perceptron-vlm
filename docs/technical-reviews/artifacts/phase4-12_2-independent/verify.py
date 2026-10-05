import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.audio_utils import load_mono_audio, resample_waveform

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def peak_frequency(values, rate):
    spectrum = np.abs(np.fft.rfft(values))
    return float(np.fft.rfftfreq(len(values), d=1 / rate)[spectrum.argmax()])

values = 0.5 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)
resampled = resample_waveform(values, 16000, 8000)
assert len(resampled) == 800
assert peak_frequency(values, 16000) == 440
assert peak_frequency(values, 8000) == 220
assert peak_frequency(resampled, 8000) == 440
assert len(values) / 16000 == len(resampled) / 8000 == 0.1

n = np.arange(1600)
high = np.cos(2 * np.pi * 6000 * n / 8000)
low = np.cos(2 * np.pi * 2000 * n / 8000)
alias_error = float(np.max(np.abs(high - low)))
assert alias_error < 5e-12

raw = ROOT / 'docs/course-experiments/results/real_modal.json'
historical = json.loads(raw.read_bytes())
conversion = historical['results']['fsdd']['resampling'][0]
source = ROOT / 'data/training/fsdd-initial/recordings' / conversion['source']
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == conversion['source_sha256']
wave, rate = sf.read(source, dtype='float32')
assert wave.ndim == 1 and rate == 8000 and len(wave) == 4591
converted, metadata = load_mono_audio(source, resample=True)
assert metadata['source_samples'] == conversion['samples_before'] == 4591
assert metadata['effective_samples'] == conversion['samples_after'] == 9182
assert metadata['source_duration_seconds'] == metadata['duration_seconds'] == 0.573875

# Execute only the AST-selected original historical resampling function.
code_path = ROOT / 'scripts/course_experiments/modalities.py'
code = code_path.read_text()
tree = ast.parse(code)
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_resample_8_to_16')
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(code_path), 'exec'), namespace)
old_converted = namespace['_resample_8_to_16'](wave)
assert np.array_equal(converted, old_converted)

items = historical['results']['fsdd']['resampling']
for item in items:
    assert item['source_rate'] == 8000 and item['target_rate'] == 16000
    assert item['samples_after'] == 2 * item['samples_before']
    assert item['samples_before'] / item['source_rate'] == item['samples_after'] / item['target_rate']

report = {
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__, 'soundfile': sf.__version__, 'device': 'CPU', 'cuda_build': str(torch.version.cuda)},
    'fence_expected': {'16000': {'seconds': 0.1, 'nyquist_Hz': 8000, 'samples_per_cycle_440Hz': 36.36}, '8000': {'seconds': 0.2, 'nyquist_Hz': 4000, 'samples_per_cycle_440Hz': 18.18}},
    'playback_and_resampling': {'original_count': len(values), 'resampled_count': len(resampled), 'original_peak_Hz': peak_frequency(values,16000), 'relabelled_peak_Hz': peak_frequency(values,8000), 'resampled_peak_Hz': peak_frequency(resampled,8000), 'original_duration_seconds': len(values)/16000, 'relabelled_duration_seconds': len(values)/8000, 'resampled_duration_seconds': len(resampled)/8000},
    'alias_non_uniqueness': {'sampling_rate_Hz':8000, 'high_Hz':6000, 'low_Hz':2000, 'samples':len(n), 'max_absolute_sample_difference':alias_error, 'tolerance':5e-12},
    'historical_record_checked': {'json_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(), 'pointer':'/results/fsdd/resampling/0', 'source':str(source.relative_to(ROOT)), 'source_sha256':source_hash, 'source_rate':rate, 'samples_before':len(wave), 'samples_after':len(converted), 'duration_seconds':metadata['duration_seconds'], 'amplitude_min':metadata['amplitude_min'], 'amplitude_max':metadata['amplitude_max'], 'current_vs_original_resampling_equal':bool(np.array_equal(converted,old_converted)), 'historical_conversion_entries_checked':len(items), 'denominator':'One existing waveform verified against its recorded file SHA; all 60 raw conversion records checked for count/rate/time arithmetic; no model run.'},
    'exercise': {'source_count':8000,'source_rate':8000,'duration_seconds':1,'target_rate':16000,'target_count':16000},
    'scope': 'Bounded CPU arithmetic, FFT illustration and resampling contract verification only. No training, evaluation of an existing model, download or checkpoint write.'
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
