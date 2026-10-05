"""Bounded mechanism verification; no training, weights, downloads, or scoring."""
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
from tiny_perceptron.attention import attention_mask

BASE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
scope = {"__name__": "__main__"}
exec(compile((BASE / "fence-1.py").read_bytes(), "original-19.9-fence-1", "exec"), scope)
core, ids = scope["core"], scope["ids"]
before = {name: value.detach().clone() for name, value in core.state_dict().items()}
full = scope["full"]
records = []
with torch.no_grad():
    for prefix_length, suffix_start in [(3, 3), (2, 2), (2, 3), (3, 2)]:
        prefill = core(ids[:, :prefix_length])
        cache = prefill["cache"]
        saved_cache = [(k.clone(), v.clone()) for k, v in cache]
        continued = core(ids[:, suffix_start:], cache=cache)
        cached = continued["logits"][:, -1]
        diff = (full - cached).abs()
        bound = 1e-4 + 1e-4 * cached.abs()
        sequence = torch.cat((ids[:, :prefix_length], ids[:, suffix_start:]), -1)
        combined_full = core(sequence)["logits"][:, -1]
        queries = torch.arange(prefix_length, prefix_length + ids.shape[1] - suffix_start)
        keys = torch.arange(sequence.shape[1])
        mask = attention_mask(queries, keys)
        record = {
            "prefix_length": prefix_length,
            "suffix_start": suffix_start,
            "processed_ids": sequence.tolist(),
            "full_logit_shape": list(full.shape),
            "continuation_logit_shape": list(continued["logits"].shape),
            "final_candidate_shape": list(cached.shape),
            "prefill_cache_shapes": [[list(k.shape), list(v.shape)] for k, v in cache],
            "continued_cache_shapes": [[list(k.shape), list(v.shape)] for k, v in continued["cache"]],
            "query_positions": queries.tolist(),
            "key_positions": keys.tolist(),
            "causal_mask": mask[0, 0].to(torch.int32).tolist(),
            "maximum_abs_logit_difference_vs_original": diff.max().item(),
            "allclose_vs_original": torch.allclose(full, cached, atol=1e-4, rtol=1e-4),
            "elementwise_tolerance_violations": int((diff > bound).sum()),
            "max_error_vs_its_actual_combined_sequence": (combined_full - cached).abs().max().item(),
            "allclose_vs_its_actual_combined_sequence": torch.allclose(combined_full, cached, atol=1e-4, rtol=1e-4),
            "input_cache_unmodified": all(torch.equal(k, sk) and torch.equal(v, sv) for (k, v), (sk, sv) in zip(cache, saved_cache)),
            "output_has_gradient": cached.requires_grad,
        }
        if prefix_length == suffix_start:
            assert record["processed_ids"] == ids.tolist()
            assert record["allclose_vs_original"]
            assert record["elementwise_tolerance_violations"] == 0
            assert record["final_candidate_shape"] == [1, 264]
        else:
            assert record["processed_ids"] != ids.tolist()
        assert record["allclose_vs_its_actual_combined_sequence"]
        assert record["input_cache_unmodified"]
        records.append(record)
weights_unchanged = all(torch.equal(value, core.state_dict()[name]) for name, value in before.items())
assert weights_unchanged
result = {
    "environment": {"python": platform.python_version(), "python_executable": os.sys.executable,
                    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
                    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                    "device": str(full.device), "dtype": str(full.dtype), "threads": str(torch.get_num_threads())},
    "original_fence_sha256": hashlib.sha256((BASE / "fence-1.py").read_bytes()).hexdigest(),
    "random_model_config": vars(core.config),
    "seed": 42, "original_ids": ids.tolist(), "records": records,
    "optimizer_updates": 0, "weights_unchanged": weights_unchanged,
    "scope": "Original random CPU mechanism example plus both-boundary exercise and one-boundary negative controls; no model capability or timing benchmark.",
}
(BASE / "cpu-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
