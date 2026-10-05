"""Bounded CPU-only review: no training, downloads, GPU, or parameter updates."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch.profiler import ProfilerActivity, profile
from tiny_perceptron.model import ModelConfig, TinyLM

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(161)
assert torch.version.cuda is None and not torch.cuda.is_available()
model = TinyLM(ModelConfig(width=8)).eval()
parameters = [
    dict(name=n, shape=list(p.shape), numel=p.numel(), bytes_per_element=p.element_size(),
         dtype=str(p.dtype), bytes=p.numel()*p.element_size())
    for n, p in model.named_parameters()
]
numel = sum(p['numel'] for p in parameters)
weight_bytes = sum(p['bytes'] for p in parameters)
assert numel == 6104 and weight_bytes == 24416
before = {n: p.detach().clone() for n, p in model.named_parameters()}
trials = []
with torch.no_grad():
    for trial in range(3):
        for length in (3, 12):
            ids = torch.tensor([[1, 2, 3]]) if length == 3 else torch.arange(1, 13)[None]
            for _ in range(3):
                model(ids)
            with profile(activities=[ProfilerActivity.CPU]) as p:
                for _ in range(5):
                    result = model(ids)
            assert tuple(result['logits'].shape) == (1, length, 264)
            assert not result['logits'].requires_grad
            events = p.key_averages()
            record = dict(trial=trial+1, input_shape=list(ids.shape),
                          output_shape=list(result['logits'].shape), warmup_calls=3,
                          measured_forward_calls=5, parameter_bytes=weight_bytes,
                          requires_grad=result['logits'].requires_grad,
                          rows=[dict(name=e.key, calls=e.count,
                                     self_cpu_us=e.self_cpu_time_total,
                                     cpu_total_us=e.cpu_time_total)
                                for e in events])
            trials.append(record)
            print(f'trial={trial+1} length={length} shape={tuple(result["logits"].shape)} bytes={weight_bytes}')
            print(events.table(sort_by='self_cpu_time_total', row_limit=20))
assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
assert all(p.grad is None for p in model.parameters())

# Check API semantics without reading uninitialized storage.
x = torch.arange(6.).view(2, 3)
assert torch.equal(x.new_zeros((2, 3)), torch.zeros(2, 3))
assert x.new_empty((1, 2)).shape == (1, 2)
assert torch.empty(1, 2).shape == (1, 2)
assert torch.empty_like(x).shape == x.shape
for view in (x.view(6), x.unsqueeze(0), torch.as_strided(x, (3,), (1,))):
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
assert torch.equal(x.index_select(0, torch.tensor([1, 0])), x[[1, 0]])
weights = x.softmax(-1)
assert torch.allclose(weights.sum(-1), torch.ones(2))
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
expected = torch.tensor([[19., 22.], [43., 50.]])
assert torch.equal(torch.mm(a, b), expected)
assert torch.equal(torch.matmul(a, b), expected)
assert torch.equal(torch.addmm(torch.ones(2, 2), a, b), expected + 1)
assert torch.nn.functional.gelu(x).shape == x.shape
assert statistics.median([1, 2, 3, 20]) == 2.5
assert statistics.mean([1, 2, 3, 20]) == 6.5
assert statistics.median(range(1, 10)) == 5
assert 2**20 == 1048576
summary = dict(environment=dict(python=sys.version, torch=str(torch.__version__),
    torch_git=str(torch.version.git_version), device='cpu', cuda_build=str(torch.version.cuda),
    threads=str(torch.get_num_threads())), parameters=parameters, parameter_count=numel,
    parameter_bytes=weight_bytes, trials=trials, unchanged_weights=True,
    gradients_created=False, updates=0, arithmetic=dict(median_even=2.5,
    mean_even=6.5, median_nine=5, MiB_bytes=1048576), api_semantic_assertions='passed')
(OUT/'cpu-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print('PASS parameter count/bytes, shapes, unchanged weights, API semantics, arithmetic')
for length in (3, 12):
    for name in ('aten::mm','aten::matmul','aten::addmm'):
        selected = [e for t in trials if t['input_shape'][1]==length for e in t['rows'] if e['name']==name]
        print(length, name, [(e['calls'],e['self_cpu_us']) for e in selected])
