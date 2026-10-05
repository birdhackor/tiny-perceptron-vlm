"""Independent bounded CPU check of the current 12.8 contracts; no training."""
import ast
import hashlib
import json
import math
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.multimodal import AudioEncoder, log_mel, mel_filter_bank, tone

torch.set_num_threads(1)
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
observations = {"environment": {"python": sys.version, "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git": str(torch.version.git_version),
    "device": "cpu", "cuda_build": str(torch.version.cuda), "threads": "1", "seed": "42"}}

# The 0.1-second, 16-kHz tone uses exactly 1600 samples. None is a view axis.
one = tone()
wave = one[None]
assert one.shape == (1600,) and wave.shape == (1, 1600)
assert torch.equal(wave[0], one)
assert wave.data_ptr() == one.data_ptr()
observations["wave"] = {"samples": one.numel(), "duration_seconds": one.numel()/16000,
    "shape_before_None": list(one.shape), "shape_after_None": list(wave.shape),
    "values_and_storage_preserved": True}

# Reconstruct the centered STFT by reflection padding, frame extraction and FFT.
n_fft, hop = 400, 160
padded = F.pad(wave, (n_fft//2, n_fft//2), mode="reflect")
frames = padded.unfold(-1, n_fft, hop)
window = torch.hann_window(n_fft)
manual_stft = torch.fft.rfft(frames * window, n=n_fft, dim=-1).transpose(-1, -2)
api_stft = torch.stft(wave, n_fft, hop, window=window, return_complex=True)
torch.testing.assert_close(manual_stft, api_stft, atol=1e-6, rtol=1e-6)
expected_frames = 1 + ((1600 + 2*200 - 400)//160)
assert frames.shape == (1, expected_frames, 400) and expected_frames == 11

# Recompute HTK-mel triangular weights independently with Python scalar math.
max_mel = 2595*math.log10(1+8000/700)
hz = [700*(10**((max_mel*j/17)/2595)-1) for j in range(18)]
reference_bank = torch.tensor([[max(0., min((40*k-hz[j])/(hz[j+1]-hz[j]),
    (hz[j+2]-40*k)/(hz[j+2]-hz[j+1]))) for k in range(201)] for j in range(16)])
bank = mel_filter_bank()
torch.testing.assert_close(bank, reference_bank, atol=4e-6, rtol=4e-5)
reference_log = (reference_bank @ manual_stft.abs().square()).clamp(min=1e-8).log()
spectrogram = log_mel(wave)
torch.testing.assert_close(spectrogram, reference_log, atol=6e-5, rtol=3e-5)
assert spectrogram.shape == (1,16,11) and bool(spectrogram.isfinite().all())
observations["preprocessing"] = {"sample_rate_hz":16000,"n_fft_samples":400,
    "hop_samples":160,"reflect_padding_each_side_samples":200,"formula":"1 + floor((1600+400-400)/160) = 11",
    "fft_frequency_bins":201,"mel_bands":16,"stft_shape":list(api_stft.shape),
    "log_mel_shape":list(spectrogram.shape),"stft_max_abs_error":float((manual_stft-api_stft).abs().max()),
    "bank_max_abs_error":float((bank-reference_bank).abs().max()),
    "log_mel_max_abs_error":float((spectrogram-reference_log).abs().max()),
    "log_mel_min":float(spectrogram.min()),"log_mel_max":float(spectrogram.max()),
    "power_floor":1e-8,"log_base":"natural"}

encoder = AudioEncoder(bands=16, width=8)
before = {k:v.detach().clone() for k,v in encoder.state_dict().items()}
features = encoder(wave)
manual = encoder.feature(encoder.projection(spectrogram.transpose(-1,-2)))
assert torch.equal(features, manual)
# Every frame independently receives precisely the same learned function.
for i in range(11):
    row = encoder.feature(encoder.projection(spectrogram[0,:,i]))
    torch.testing.assert_close(features[0,i],row,atol=2e-6,rtol=2e-6)
assert all(torch.equal(before[k],v) for k,v in encoder.state_dict().items())
assert all(p.requires_grad for p in encoder.parameters())
observations["encoder"] = {"shape":list(features.shape),"projection_weight_shape":list(encoder.projection.weight.shape),
    "layers":[type(encoder.projection).__name__]+[type(x).__name__ for x in encoder.feature],
    "internal_vs_external_log_mel_max_abs_error":float((features-manual).abs().max()),
    "same_function_per_frame":True,"all_parameters_trainable":True,"parameters_unchanged_by_forward":True}

wide = AudioEncoder(bands=16,width=12)(wave)
more_bands = AudioEncoder(bands=32,width=8)
external32 = log_mel(wave,bands=32)
internal32 = more_bands(wave)
manual32 = more_bands.feature(more_bands.projection(external32.transpose(-1,-2)))
assert wide.shape == (1,11,12) and external32.shape == (1,32,11) and internal32.shape == (1,11,8)
assert torch.equal(internal32,manual32)
try:
    more_bands.projection(spectrogram.transpose(-1,-2))
except RuntimeError as exc:
    mismatch = str(exc)
else:
    raise AssertionError("An explicitly mismatched 16-band tensor must fail a 32-band Linear")
batched = encoder(torch.stack([tone(440),tone(200)]))
assert batched.shape == (2,11,8)
rows = torch.arange(11*16,dtype=torch.float32).reshape(1,11,16)/10
ordered = encoder.feature(encoder.projection(rows))
reversed_rows = encoder.feature(encoder.projection(rows.flip(1)))
torch.testing.assert_close(reversed_rows,ordered.flip(1),atol=1e-6,rtol=1e-6)
torch.testing.assert_close(ordered.mean(1),reversed_rows.mean(1),atol=1e-6,rtol=1e-6)
observations["variants"] = {"width12_shape":list(wide.shape),"bands32_preprocessing_shape":list(external32.shape),
    "bands32_encoder_shape":list(internal32.shape),"encoder_bands32_sets_internal_preprocessing_automatically":True,
    "mismatched_explicit_projection_error":mismatch,"batch2_shape":list(batched.shape),
    "frame_reversal_reverses_features":True,"mean_after_frame_reversal_unchanged":True}

# Read only named raw history pointers; never inspect /scope or author notes.
raw_path = ROOT/'docs/course-experiments/results/encoders.json'
raw = json.loads(raw_path.read_bytes())
def pointer(value,p):
    for part in p.lstrip('/').split('/'):
        value = value[part] if isinstance(value,dict) else value[int(part)]
    return value
pointers = ['/revision','/device','/seed','/torch_version','/python_version',
    '/code_sha256/tiny_perceptron~1multimodal.py', '/results/audio/classes','/results/audio/config',
    '/results/audio/data/splits','/results/audio/training/steps','/results/audio/training/trainable_parameters',
    '/results/audio/training/effective_targets','/results/audio/training/weights_changed',
    '/results/audio/training/nonzero_gradient_seen','/results/audio/before/count',
    '/results/audio/before/correct','/results/audio/before/samples','/results/audio/validation/count',
    '/results/audio/validation/correct','/results/audio/validation/samples',
    '/results/audio/test/count','/results/audio/test/correct','/results/audio/test/samples']
selected = {}
for p in pointers:
    if '~1' in p:
        selected[p]=raw['code_sha256']['tiny_perceptron/multimodal.py']
    else:selected[p]=pointer(raw,p)
splits=selected['/results/audio/data/splits']
observations['history_pointer_types']={p:type(v).__name__ for p,v in selected.items()}
(BASE/'raw-history-selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
observations['history_raw_sha256']=hashlib.sha256(raw_path.read_bytes()).hexdigest()
observations['history_selected_pointers']=pointers
# The stored manifest shape is recorded here; detailed sample checks follow after inspection.
observations['manifest_split_types']={k:type(v).__name__ for k,v in splits.items()}
method_path=ROOT/'scripts/course_experiments/modalities.py'
tree=ast.parse(method_path.read_bytes())
definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_audio_records')
namespace={}
exec(compile(ast.Module(body=[definition],type_ignores=[]),str(method_path)+':_audio_records','exec'),namespace)
expected_records=namespace['_audio_records']()
history_counts={}
families={}
for split,manifest in splits.items():
    records=manifest['records']
    assert records == expected_records[split]
    assert manifest['count']==len(records)
    digest=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    assert manifest['sha256']==digest
    families[split]={r['family'] for r in records}
    for r in records:
        assert r['answer']==('high' if r['frequency']>300 else 'low')
    result_name=split if split!='train' else None
    if result_name:
        samples=selected[f'/results/audio/{result_name}/samples']
        assert len(samples)==len(records)==selected[f'/results/audio/{result_name}/count']
        for i,(sample,r) in enumerate(zip(samples,records,strict=True)):
            assert sample['row']==i and sample['family']==r['family'] and sample['target']==r['answer']
            assert sample['correct']==(sample['predicted']==sample['target'])
        assert sum(s['correct'] for s in samples)==selected[f'/results/audio/{result_name}/correct']
    history_counts[split]={'examples':len(records),'frequency_families':len(families[split])}
for first,second in [('train','validation'),('train','test'),('validation','test')]:
    assert not (families[first]&families[second])
before_samples=selected['/results/audio/before/samples']
assert len(before_samples)==selected['/results/audio/before/count']==len(splits['test']['records'])
assert sum(s['correct'] for s in before_samples)==selected['/results/audio/before/correct']
current_module_hash=hashlib.sha256((ROOT/'tiny_perceptron/multimodal.py').read_bytes()).hexdigest()
observations['historical_method_and_samples']={'counts':history_counts,'same_frequency_families_stay_together':True,
    'all_frequency_threshold_labels_match':True,'stored_raw_samples_and_counts_match':True,
    'steps':selected['/results/audio/training/steps'],
    'effective_targets':selected['/results/audio/training/effective_targets'],
    'trainable_parameters':selected['/results/audio/training/trainable_parameters'],
    'historical_torch':selected['/torch_version'],'historical_device':selected['/device'],
    'historical_source_hash':selected['/code_sha256/tiny_perceptron~1multimodal.py'],
    'current_source_hash':current_module_hash,
    'historical_and_current_multimodal_same_bytes':current_module_hash==selected['/code_sha256/tiny_perceptron~1multimodal.py'],
    'no_model_loading_or_training':'Only stored records and measurements were parsed; no .pt read or created.'}
print(json.dumps(observations,ensure_ascii=False,indent=2))
(BASE/'bounded-results.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
