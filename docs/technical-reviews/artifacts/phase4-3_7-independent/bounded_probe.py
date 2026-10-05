"""Fresh, bounded CPU checks of section 3.7; no training or downloads."""
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.attention import CausalAttention, manual_attention

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
facts = {"environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype_default": str(torch.get_default_dtype()), "cuda_build": str(torch.version.cuda)}, "tolerance": "float32 comparison: atol=1e-6, rtol=1e-5; hand Fraction arithmetic exact; float64 synthetic attention atol=1e-12, rtol=0"}

# Original fence, unmodified, exactly as extracted from current Markdown.
ns = {"__name__": "__main__"}
exec(compile((HERE / "original/fence-1.py").read_bytes(), "original/fence-1.py", "exec"), ns)
layer, x, out, cache = (ns[k] for k in ("layer", "x", "out", "cache"))
before = {n: p.detach().clone() for n, p in layer.named_parameters()}
facts["original"] = {"x": list(x.shape), "out": list(out.shape), "K": list(cache[0].shape), "V": list(cache[1].shape), "out_requires_grad": out.requires_grad, "parameter_gradients": {n: p.grad is not None for n,p in layer.named_parameters()}, "parameter_counts": {n:p.numel() for n,p in layer.named_parameters()}, "total_parameters": sum(p.numel() for p in layer.parameters())}
assert sum(p.numel() for p in layer.parameters()) == 4 * 8 * 8 == 256
assert all(p.grad is None for p in layer.parameters())

# Exact fractions: normalization denominator is the two allowed key positions.
head0 = [Fraction(3,4)*a + Fraction(1,4)*b for a,b in zip([10,0,0,0],[0,20,0,0])]
head1 = [Fraction(1,4)*a + Fraction(3,4)*b for a,b in zip([4,8,0,0],[12,0,0,0])]
joined = head0 + head1
assert joined == [Fraction(15,2),5,0,0,10,2,0,0]
facts["specified_hand_example"] = {"head0": [float(a) for a in head0], "head1": [float(a) for a in head1], "concatenation": [float(a) for a in joined], "ratio_sum_each_head": "3/4+1/4 = 1", "axes": "one query at position 1; allowed keys 0 and 1; 2 heads x 4 value features", "units": "toy feature values, not probabilities; attention ratios dimensionless"}

with torch.no_grad():
    # A compatible synthetic Q/K realizes the stated ratios; not the course random input.
    q = torch.zeros(1,2,2,4,dtype=torch.float64)
    k = torch.zeros_like(q)
    q[0,:,1,0] = 1
    k[0,0,0,0] = 2*math.log(3)
    k[0,1,1,0] = 2*math.log(3)
    v = torch.tensor([[[[10.,0,0,0],[0,20.,0,0]],[[4.,8.,0,0],[12.,0,0,0]]]],dtype=torch.float64)
    allowed = torch.ones(2,2,dtype=torch.bool).tril()[None,None]
    mixed, weights = manual_attention(q,k,v,allowed)
    target = torch.tensor([[float(a) for a in head0],[float(a) for a in head1]],dtype=torch.float64)
    assert torch.allclose(mixed[0,:,1],target,atol=1e-12,rtol=0)
    facts["synthetic_ratio_check"] = {"row1_weights":weights[0,:,1].tolist(), "row1_output":mixed[0,:,1].tolist(), "max_abs_error":float((mixed[0,:,1]-target).abs().max()), "normalization_axis":"last axis: key positions, never feature axis"}

    # Independent per-head, per-query calculation using slices of projection rows.
    # Each head receives every input position and the full input feature width.
    def independent_reference(candidate, data):
        all_heads = []
        hd = candidate.head_dim
        for h in range(candidate.heads):
            rows = slice(h*hd,(h+1)*hd)
            qh = F.linear(data,candidate.q.weight[rows])
            kh = F.linear(data,candidate.k.weight[rows])
            vh = F.linear(data,candidate.v.weight[rows])
            rows_out = []
            for t in range(data.shape[1]):
                logits = (qh[:,t:t+1] @ kh[:,:t+1].transpose(1,2)) / math.sqrt(hd)
                rows_out.append(torch.softmax(logits,dim=-1) @ vh[:,:t+1])
            all_heads.append(torch.cat(rows_out,dim=1))
        combined = torch.cat(all_heads,dim=-1)
        return F.linear(combined,candidate.out.weight), combined

    ref, concat = independent_reference(layer,x)
    assert torch.allclose(out,ref,atol=1e-6,rtol=1e-5)
    q_raw = layer.q(x)
    q_split = q_raw.view(1,4,2,4)
    q_heads = q_split.transpose(1,2)
    facts["two_head_axes"] = {"q_projection":list(q_raw.shape), "q_split":list(q_split.shape), "q_transpose":list(q_heads.shape), "concat":list(concat.shape), "attention_scores_shape":[1,2,4,4], "score_axis_labels":["batch","heads","query_positions","key_positions"], "K_axis_labels":["batch","heads","key_positions","features_per_head"], "independent_reference_max_abs_error":float((out-ref).abs().max()), "projection_changes_concat_max_abs":float((out-concat).abs().max())}

    changed = x.clone()
    changed[:,3] += 100
    changed_out,_ = layer(changed)
    assert torch.equal(out[:,:3],changed_out[:,:3])
    assert not torch.allclose(out[:,3],changed_out[:,3])
    facts["causal_check"] = {"changed_input_position":3, "unchanged_prefix_positions":[0,1,2], "prefix_max_abs_error":float((out[:,:3]-changed_out[:,:3]).abs().max()), "last_position_max_abs_change":float((out[:,3]-changed_out[:,3]).abs().max())}

    incremental_cache = None
    incremental_outputs = []
    cache_lengths = []
    for t in range(x.shape[1]):
        yt,incremental_cache = layer(x[:,t:t+1],cache=incremental_cache)
        incremental_outputs.append(yt)
        cache_lengths.append(incremental_cache[0].shape[2])
    increment = torch.cat(incremental_outputs,dim=1)
    assert torch.allclose(out,increment,atol=1e-6,rtol=1e-5)
    assert torch.allclose(cache[0],incremental_cache[0],atol=1e-6,rtol=1e-5)
    assert torch.allclose(cache[1],incremental_cache[1],atol=1e-6,rtol=1e-5)
    facts["cache_reuse"] = {"one_token_calls":4, "K_sequence_lengths":cache_lengths, "final_K":list(incremental_cache[0].shape), "output_max_abs_error":float((out-increment).abs().max()), "K_max_abs_error":float((cache[0]-incremental_cache[0]).abs().max()), "V_max_abs_error":float((cache[1]-incremental_cache[1]).abs().max()), "scope":"same CPU float32 layer/input; no padding, packing, RoPE or GQA; default manual backend"}

    torch.manual_seed(42)
    four = CausalAttention(width=8,heads=4)
    x4 = torch.randn(1,4,8)
    out4,cache4 = four(x4)
    ref4,_ = independent_reference(four,x4)
    assert list(cache4[0].shape) == [1,4,4,2]
    assert list(out4.shape) == [1,4,8]
    assert four.head_dim == 2
    assert sum(p.numel() for p in four.parameters()) == 256
    assert torch.allclose(out4,ref4,atol=1e-6,rtol=1e-5)
    facts["four_heads"] = {"head_dim":four.head_dim, "out":list(out4.shape), "K":list(cache4[0].shape), "parameters":sum(p.numel() for p in four.parameters()), "independent_reference_max_abs_error":float((out4-ref4).abs().max())}

    try:
        CausalAttention(width=8,heads=3)
    except ValueError as e:
        facts["three_heads"] = {"error_type":"ValueError","message":str(e)}
    else:
        raise AssertionError("8/3 must be rejected")

    torch.manual_seed(42)
    repeated = CausalAttention(width=8,heads=2)
    x_repeated = torch.randn(1,4,8)
    assert torch.equal(x,x_repeated)
    assert all(torch.equal(before[n],p) for n,p in repeated.named_parameters())
    facts["seed_repetition"] = {"same_environment_input_equal":True, "same_environment_initial_weights_equal":True, "scope":"same ordered calls, PyTorch commit and CPU runtime; not a cross-version or cross-device guarantee"}

assert all(torch.equal(before[n],p) for n,p in layer.named_parameters())
facts["parameter_update"] = "No backward, optimizer or in-place parameter writes; all original parameters unchanged after forward/probes. Initial forward only builds an autograd graph."
facts["source_hashes"] = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"tiny_perceptron/attention.py",HERE/"bounded_probe.py",HERE/"original/fence-1.py"]}
(HERE/"probe-results.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(facts,ensure_ascii=False,indent=2))
