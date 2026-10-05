from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import mel_filter_bank

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
OUT = Path(__file__).resolve().parent
facts = {}

# No corpus or model downloads: construct a byte-level vocabulary with no merges.
alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
tok.decoder = decoders.ByteLevel()
encoded = tok.encode("它")
assert len(encoded.ids) == len("它".encode("utf-8")) == 3
assert tok.decode(encoded.ids) == "它"
torch.manual_seed(42)
embedding = torch.nn.Embedding(256, 4)
emb = embedding(torch.tensor(encoded.ids))
assert emb.shape == (3, 4)
assert torch.equal(emb, embedding.weight[torch.tensor(encoded.ids)])
facts["tokens_embedding"] = {"text": "它", "configuration": "ByteLevel BPE, all 256 bytes, no merges", "ids": encoded.ids, "count": len(encoded.ids), "embedding_shape": list(emb.shape)}

# Uniform query/key compatibility makes each causal row a transparent mean.
q = torch.ones(3, 1, dtype=torch.float64)
k = torch.zeros_like(q)
v = torch.tensor([[1.], [4.], [9.]], dtype=torch.float64, requires_grad=True)
valid = torch.ones(3, 3, dtype=torch.bool).tril()
weights = (q @ k.T).masked_fill(~valid, -torch.inf).softmax(-1)
out = weights @ v
assert torch.allclose(out[:, 0], torch.tensor([1., 2.5, 14./3.], dtype=torch.float64))
changed = v.detach().clone()
changed[2, 0] = 99.
assert torch.equal((weights @ changed)[:2], out.detach()[:2])
labels = torch.tensor([-100, -100, 0])
logits = torch.cat([out, -out], dim=1)
per_position = F.cross_entropy(logits, labels, ignore_index=-100, reduction="none")
assert torch.equal(per_position[:2], torch.zeros(2, dtype=torch.float64))
loss = F.cross_entropy(logits, labels, ignore_index=-100)
grad = torch.autograd.grad(loss, v)[0]
assert bool((grad[:2] != 0).all())
pad = F.pad(torch.tensor([10, 11]), (0, 2), value=0)
assert pad.tolist() == [10, 11, 0, 0]
facts["masks_padding"] = {"causal_weights": weights.tolist(), "causal_output": out.detach().tolist(), "future_change_preserves_past": True, "per_position_loss": per_position.detach().tolist(), "prompt_value_gradients": grad[:2].tolist(), "padding": pad.tolist(), "meaning": "Ignored prompt targets contribute no direct loss, but their values remain readable and affect the supervised answer. Backward only; no optimizer/update."}

# Execute the original context 4.4 fence without edits; then perturb one position.
context = (OUT.parent / "inputs/context-4.4.md").read_text()
fence = re.search(r"```python\n(.*?)```", context, flags=re.S).group(1)
(OUT / "original-context4.4-fence.py").write_text(fence)
ns = {"__name__": "__main__"}
exec(compile(fence, "context-4.4-original-fence", "exec"), ns)
layer, x, baseline = ns["layer"], ns["x"], ns["y"].detach()
x[0, 1, 0] = 2
perturbed = layer(x).detach()
assert torch.equal(perturbed[:, 0], baseline[:, 0])
assert not torch.equal(perturbed[:, 1], baseline[:, 1])
norm = torch.nn.LayerNorm(4, elementwise_affine=False, eps=0).double()
features = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
normalized = norm(features)
assert torch.allclose(normalized.mean(-1), torch.zeros(1, dtype=torch.float64), atol=1e-12)
assert torch.allclose(normalized.var(-1, unbiased=False), torch.ones(1, dtype=torch.float64), atol=1e-12)
assert torch.allclose(normalized, norm(features * 100), atol=1e-12)
facts["ffn_normalization"] = {"original_fence_sha256": hashlib.sha256(fence.encode()).hexdigest(), "up_shape": list(layer.up.weight.shape), "down_shape": list(layer.down.weight.shape), "output_shape": list(baseline.shape), "same_input_same_output": True, "only_changed_position_changes": True, "normalization_axis": "last feature axis", "normalized_mean": normalized.mean(-1).tolist(), "normalized_variance": normalized.var(-1, unbiased=False).tolist(), "scale_invariance_fixture": "100x, eps=0, affine=false; arithmetic demonstration only"}

# One second of a sine: 440 upward zero crossings and FFT peak at 440 Hz.
sr = 16000
frequency = 440
t = torch.arange(sr, dtype=torch.float64) / sr
wave = torch.sin(2 * torch.pi * frequency * t)
crossings = int(((wave[:-1] <= 0) & (wave[1:] > 0)).sum())
freq_axis = torch.fft.rfftfreq(sr, d=1/sr)
peak = float(freq_axis[torch.fft.rfft(wave).abs().argmax()])
assert crossings == 440 and peak == 440.
power = torch.stft(wave, n_fft=400, hop_length=160, win_length=400, window=torch.hann_window(400, dtype=torch.float64), center=False, return_complex=True).abs().square()
bank = mel_filter_bank(sample_rate=16000, n_fft=400, bands=16).double()
mel = bank @ power
overlap = int(((bank > 0).sum(0) >= 2).sum())
assert power.shape[0] == 201 and mel.shape == (16, power.shape[1])
assert bool((bank >= 0).all()) and overlap > 0
hand = torch.tensor([[1., .5, 0.], [0., .5, 1.]]) @ torch.tensor([4., 2., 1.])
assert hand.tolist() == [5., 2.]
facts["audio"] = {"samples": sr, "duration_seconds": 1, "frequency_hz": frequency, "upward_zero_crossings": crossings, "fft_peak_hz": peak, "stft_axes": "frequency bins, time frames", "power_shape": list(power.shape), "mel_shape": list(mel.shape), "nonnegative_bank": True, "frequency_bins_contributing_to_multiple_bands": overlap, "overlap_hand_example": hand.tolist(), "time_frames_preserved": True, "scope": "Numeric signal fixture and original CPU filter-bank helper; no auditory test, learning, quality measurement or signal reconstruction claim"}

files = ["tiny_perceptron/modern.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py"]
environment = {"python": platform.python_version(), "python_executable": sys.executable, "torch": torch.__version__, "torch_git": torch.version.git_version, "tokenizers": importlib.metadata.version("tokenizers"), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "repository_files": {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
(OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"environment": environment, "facts": facts, "status": "all assertions passed"}, ensure_ascii=False))
