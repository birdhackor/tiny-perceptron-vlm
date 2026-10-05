"""Independent bounded CPU checks for 16.2; no training or downloads."""
import ast
import hashlib
import json
import pathlib
import platform
import statistics

import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, generate

OUT = pathlib.Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(162)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
checks = {
    "environment": {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "device": "cpu", "cuda_build": str(torch.version.cuda),
        "seed": 162, "threads": torch.get_num_threads(),
    },
    "scope": "Original fence was executed separately; this check uses short CPU forward variations and reads original existing GPU measurements. No backward, parameter updates, training, or model-performance reevaluation.",
    "shape_variants": [],
}
model = TinyLM(ModelConfig(width=8)).eval()
weights_before = [p.detach().clone() for p in model.parameters()]
for length in (4, 6):
    projections = []
    handles = []
    for name in ("q", "k", "v"):
        layer = getattr(model.blocks[0].attention, name)
        handles.append(layer.register_forward_hook(
            lambda module, inputs, output, name=name: projections.append(
                {"projection": name, "input_shape": list(inputs[0].shape), "output_shape": list(output.shape)}
            )
        ))
    with torch.no_grad():
        prefill = model(torch.arange(1, length + 1)[None])
        prefill_projections = projections.copy()
        projections.clear()
        decode = model(torch.tensor([[length + 1]]), cache=prefill["cache"])
        assert list(prefill["logits"].shape) == [1, length, 264]
        assert list(decode["logits"].shape) == [1, 1, 264]
        assert list(prefill["cache"][0][0].shape) == [1, 1, length, 8]
        assert list(decode["cache"][0][0].shape) == [1, 1, length + 1, 8]
        assert torch.equal(prefill["cache"][0][0], decode["cache"][0][0][:, :, :length])
        assert not prefill["logits"].requires_grad and not decode["logits"].requires_grad
    for h in handles:
        h.remove()
    checks["shape_variants"].append({
        "prefix_length": length, "prefill_logits": list(prefill["logits"].shape),
        "decode_logits": list(decode["logits"].shape),
        "prefill_K": list(prefill["cache"][0][0].shape),
        "decode_K": list(decode["cache"][0][0].shape),
        "prefill_projections": prefill_projections, "decode_projections": projections,
        "old_K_exactly_retained": True, "logits_require_grad": False,
    })
assert all(torch.equal(old, current) for old, current in zip(weights_before, model.parameters()))
checks["parameters_unchanged"] = True

# eval changes module.training; gradient recording is a separate mechanism.
assert not model.training
eval_logits = model(torch.tensor([[1]]))["logits"]
assert eval_logits.requires_grad
checks["eval_alone_requires_grad"] = bool(eval_logits.requires_grad)

tok = ByteTokenizer()
prompt = "Once upon a time, a girl"
prompt_ids = [tok.bos_id] + tok.encode(prompt)
assert len(prompt_ids) == 25
checks["token_example"] = {"chinese_character": "字", "byte_token_ids": tok.encode("字"), "decoded": tok.decode(tok.encode("字"))}
assert len(tok.encode("字")) == 3

raw_path = OUT / "raw-efficiency.json"
raw = json.loads(raw_path.read_text())
pointers = ["/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu", "/results/source", "/results/runtime", "/results/models/mha/model/config"]
base = "/results/models/mha/cache/"
pointers += [base + k for k in ["prompt_tokens", "generated_tokens", "generated_ids_full", "generated_ids_cached", "generated_text", "prefill", "full_recompute_decode", "cached_decode"]]
pointers += ["/code_sha256/" + k.replace("~", "~0").replace("/", "~1") for k in ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/architecture.py"]]
def pointer_value(pointer):
    value = raw
    for key in pointer.strip("/").split("/"):
        value = value[key.replace("~1", "/").replace("~0", "~")]
    return value
checks["raw_pointers_read"] = pointers
checks["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
checks["raw_provenance"] = {p: pointer_value(p) for p in pointers if p in ["/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu", "/results/source", "/results/runtime", "/results/models/mha/model/config"]}
cache = raw["results"]["models"]["mha"]["cache"]
assert cache["prompt_tokens"] == len(prompt_ids) == 25
assert cache["generated_tokens"] == 12
checks["original_measurements"] = []
for field, expected_ms in [("prefill", 2.421), ("full_recompute_decode", 29.034), ("cached_decode", 27.915)]:
    record = cache[field]
    samples = record["samples_seconds"]
    median_s = statistics.median(samples)
    assert len(samples) == record["measured_calls"] == 9
    assert record["warmup_calls"] == 3 and record["synchronized"] is True
    assert median_s == record["median_seconds"]
    assert abs(median_s * 1000 - expected_ms) <= 0.0005
    checks["original_measurements"].append({"field": field, "raw_pointer": base + field, "samples_seconds": samples, "median_seconds": median_s, "median_ms": median_s * 1000, "rounded_ms": round(median_s * 1000, 3), "warmup_calls": 3, "measured_calls": 9, "synchronized": True})
checks["raw_generation"] = {
    "prompt": prompt, "prompt_ids": prompt_ids, "prompt_tokens_including_BOS": 25,
    "generated_tokens": 12, "generated_ids_full": cache["generated_ids_full"],
    "generated_ids_cached": cache["generated_ids_cached"], "generated_text_recorded": cache["generated_text"],
    "full_text_decoded_independently": tok.decode(cache["generated_ids_full"]),
    "full_EOS_count": cache["generated_ids_full"].count(tok.eos_id),
    "cached_EOS_count": cache["generated_ids_cached"].count(tok.eos_id),
}
assert len(cache["generated_ids_full"]) == len(cache["generated_ids_cached"]) == 12
assert tok.decode(cache["generated_ids_full"]) == cache["generated_text"]

# Read/extract only the actual decoding method from the exact historical source.
original = OUT / "code-snapshots/architecture-at-raw-revision.py"
assert hashlib.sha256(original.read_bytes()).hexdigest() == raw["code_sha256"]["scripts/course_experiments/architecture.py"]
tree = ast.parse(original.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fixed_decode")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), "exec"), namespace)
class AlwaysEOS:
    def __init__(self):
        self.training = True
        self.config = ModelConfig()
        self.calls = []
    def eval(self):
        self.training = False
        return self
    def train(self, mode=True):
        self.training = mode
        return self
    def __call__(self, ids, cache=None):
        self.calls.append(int(ids.shape[1]))
        total = ids.shape[1] + (0 if cache is None else cache[0][0].shape[2])
        logits = torch.zeros((1, ids.shape[1], 264))
        logits[..., tok.eos_id] = 1
        K = torch.zeros((1, 1, total, 1))
        return {"logits": logits, "cache": [(K, K.clone())]}
checks["fixed_cost_control"] = []
for cached in (False, True):
    fake = AlwaysEOS()
    result = namespace["_fixed_decode"](fake, torch.tensor([prompt_ids]), 12, cached)
    assert result.shape[1] == 37 and result[0, 25:].tolist() == [2] * 12
    assert fake.calls == ([25] + [1] * 11 if cached else list(range(25, 37)))
    checks["fixed_cost_control"].append({"cached": cached, "model_input_lengths": fake.calls, "new_ids": result[0, 25:].tolist(), "prefill_included": True})
fake = AlwaysEOS()
with torch.no_grad():
    normal = generate(fake, torch.tensor([prompt_ids]), 12, eos_id=tok.eos_id)
assert normal.shape[1] == 26 and fake.calls == [25]
checks["normal_generation_EOS_control"] = {"model_input_lengths": fake.calls, "new_ids": normal[0, 25:].tolist(), "stopped_at_first_EOS": True}
(OUT / "independent-results.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(checks, ensure_ascii=False, indent=2))
