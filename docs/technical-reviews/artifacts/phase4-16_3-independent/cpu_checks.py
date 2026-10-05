import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.data import ByteTokenizer

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
print(json.dumps({"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "default_dtype": str(torch.get_default_dtype()), "float32_bits": torch.finfo(torch.float32).bits, "threads": torch.get_num_threads()}}))

def difference(a, b):
    return (a - b).abs().max().item()

torch.manual_seed(0)
model = TinyLM(ModelConfig(width=8)).eval()
ids = torch.tensor([[1, 2, 3, 4]])
weights_before = [p.detach().clone() for p in model.parameters()]
with torch.no_grad():
    prefix = model(ids[:, :3])
    full_all = model(ids)
    old_k = prefix["cache"][0][0]
    extended_k = full_all["cache"][0][0][:, :, :3, :]
    assert torch.allclose(old_k, extended_k, atol=1e-6)
    base = model(ids[:, 3:], cache=prefix["cache"])
    assert base["logits"].shape == (1, 1, 264)
    assert base["cache"][0][0].shape == (1, 1, 4, 8)
    wrong_position = model(ids[:, 3:], positions=torch.tensor([0]), cache=prefix["cache"])
    wrong_error = difference(full_all["logits"][:, -1], wrong_position["logits"][:, 0])
    assert wrong_error > 1e-3
    last_changed = torch.tensor([[1, 2, 3, 5]])
    last_full = model(last_changed)["logits"][:, -1]
    last_cached = model(last_changed[:, 3:], cache=prefix["cache"])["logits"][:, 0]
    assert torch.allclose(last_full, last_cached, atol=1e-6)
    first_changed = torch.tensor([[6, 2, 3, 4]])
    changed_full = model(first_changed)["logits"][:, -1]
    stale = model(first_changed[:, 3:], cache=prefix["cache"])["logits"][:, 0]
    rebuilt = model(first_changed[:, 3:], cache=model(first_changed[:, :3])["cache"])["logits"][:, 0]
    assert torch.allclose(changed_full, rebuilt, atol=1e-6)
    stale_error = difference(changed_full, stale)
    assert stale_error > 1e-3
    grad_disabled = not base["logits"].requires_grad
    assert grad_disabled
    padding_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    padding_prefix = model(padding_ids, valid=valid)
    try:
        model(torch.tensor([[6], [7]]), cache=padding_prefix["cache"], valid=torch.ones((2, 4), dtype=torch.bool))
    except ValueError as error:
        padding_rejection = str(error)
    else:
        raise AssertionError("expected explicit padding cache rejection")
assert all(torch.equal(a, b) for a, b in zip(weights_before, model.parameters()))
print(json.dumps({"variants": {"base_logits_shape": list(base["logits"].shape), "appended_k_shape": list(base["cache"][0][0].shape), "old_k_max_difference": difference(old_k, extended_k), "wrong_position_max_difference": wrong_error, "last_id_changed_max_difference": difference(last_full, last_cached), "changed_prefix_stale_cache_max_difference": stale_error, "changed_prefix_rebuilt_cache_max_difference": difference(changed_full, rebuilt), "padding_cache_rejection": padding_rejection, "gradient_recording_disabled": grad_disabled, "optimizer_updates": 0, "weights_unchanged": True}}, ensure_ascii=False))

left = torch.tensor([1.0, 1.0 + 1e-7])
right = torch.tensor([1.0 + 1e-7, 1.0])
assert torch.allclose(left, right, atol=1e-6)
assert left.argmax().item() != right.argmax().item()
print(json.dumps({"near_tie": {"maximum_absolute_error": difference(left, right), "argmax_left": left.argmax().item(), "argmax_right": right.argmax().item(), "allclose_atol_1e_6": True}}))

raw_path = ROOT / "docs/course-experiments/results/efficiency.json"
raw = raw_path.read_bytes()
d = json.loads(raw)
print(json.dumps({"raw_measurement_provenance": {"path": str(raw_path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "revision": d["revision"], "device": d["device"], "gpu": d["gpu"], "torch_version": d["torch_version"], "python_version": d["python_version"], "runtime": d["results"]["runtime"]}}))
tok = ByteTokenizer()
prompt_count = len([tok.bos_id] + tok.encode("Once upon a time, a girl"))
assert prompt_count == 25
for name in ["mha", "gqa"]:
    record = d["results"]["models"][name]
    c = record["cache"]
    config = record["model"]["config"]
    assert config["heads"] == 4
    assert config["kv_heads"] == (4 if name == "mha" else 1)
    assert config["backend"] == "manual"
    assert c["prompt_tokens"] == prompt_count
    assert c["generated_tokens"] == 12
    assert len(c["generated_ids_full"]) == len(c["generated_ids_cached"]) == len(c["per_step_logit_max_error"]) == 12
    assert c["generated_ids_full"] == c["generated_ids_cached"]
    assert c["identical_greedy_ids"] is True
    maximum = max(c["per_step_logit_max_error"])
    assert maximum == 1.9073486328125e-6 and maximum < 1e-5
    assert abs(maximum - 1.90735e-6) <= 5e-12
    assert tok.decode(c["generated_ids_full"]) == c["generated_text"]
    eos_indices = [i for i, x in enumerate(c["generated_ids_full"]) if x == tok.eos_id]
    assert eos_indices and eos_indices[0] < len(c["generated_ids_full"]) - 1
    print(json.dumps({"record": name, "checked_json_pointers": [f"/results/models/{name}/model/config", *[f"/results/models/{name}/cache/{field}" for field in ["prompt_tokens", "generated_tokens", "generated_ids_full", "generated_ids_cached", "generated_text", "identical_greedy_ids", "per_step_logit_max_error"]]], "prompt_tokens": prompt_count, "generation_steps": 12, "maximum_logit_error": maximum, "rounded_display": format(maximum, ".5e"), "ids_equal": True, "generated_text": c["generated_text"], "eos_indices_zero_based": eos_indices}, ensure_ascii=False))

revision = d["revision"]
for filename in ["scripts/course_experiments/architecture.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py"]:
    expected = d["code_sha256"][filename]
    result = subprocess.run(["git", "show", revision + ":" + filename], cwd=ROOT, capture_output=True, check=True)
    actual = hashlib.sha256(result.stdout).hexdigest()
    assert actual == expected
    saved = ROOT / "docs/technical-reviews/artifacts/phase4-16_3-independent/code" / ("architecture-at-raw-revision.py" if filename.startswith("scripts/") else Path(filename).name)
    if filename.endswith("data.py"):
        saved.write_bytes(result.stdout)
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
    print(json.dumps({"raw_code_provenance": {"revision": revision, "source_path": filename, "sha256": actual, "recorded_sha256_equal": True, "snapshot_equal": True}}))
