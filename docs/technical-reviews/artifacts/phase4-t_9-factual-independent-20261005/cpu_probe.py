import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import torch
from scripts import evaluate as evaluation
from scripts import quantize
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import pack_int4, unpack_int4
from tiny_perceptron.training import save_checkpoint

torch.set_num_threads(1)
torch.manual_seed(9)
assert torch.version.cuda is None
result = {"scope": "CPU interface, packing and accounting checks only; no training or model capability benchmark", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available())}}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def accounting(state):
    assert all(isinstance(t, torch.Tensor) for t in state.values())
    groups = {}
    for name, tensor in state.items():
        role = "codes" if name.endswith(".values") else "scale" if name.endswith(".scale") else "bias" if name.endswith(".bias") else "retained_or_native"
        groups[role] = groups.get(role, 0) + tensor.numel() * tensor.element_size()
    return {"state_type": type(state).__name__, "value_types": sorted({type(x).__name__ for x in state.values()}), "entry_count": len(state), "tensor_bytes": sum(t.numel() * t.element_size() for t in state.values()), "role_tensor_bytes": groups, "non_tensor_float_count": sum(isinstance(x, float) for x in state.values())}

with tempfile.TemporaryDirectory(prefix="phase4-t9-probe-") as tmp:
    tmp = Path(tmp)
    (tmp / "checkpoints").mkdir()
    native = tmp / "checkpoints/attributes.pt"
    model = TinyLM(ModelConfig(width=4, layers=1, heads=1, max_length=128))
    save_checkpoint(native, model, optimizer=torch.optim.AdamW(model.parameters()))
    cli = []
    for bits in [4, 8]:
        target = tmp / f"checkpoints/attributes-int{bits}.pt"
        argv = ["scripts/quantize.py", str(native), "--bits", str(bits), "--output", str(target)]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            quantize.main()
        record = json.loads(stdout.getvalue())
        cli.append({"argv": argv, "selected_report_fields": {k: record[k] for k in ["bits", "model_tensor_storage_bytes", "input_output_sharing_removed"]}})
    result["synthetic_quantize_cli"] = cli
    original_cwd = Path.cwd()
    fence_stdout = io.StringIO()
    try:
        os.chdir(tmp)
        with contextlib.redirect_stdout(fence_stdout):
            exec(compile((BASE / "fence-3.py").read_bytes(), "T.9:original-fence-3", "exec"), {})
    finally:
        os.chdir(original_cwd)
    result["original_fence_stdout"] = fence_stdout.getvalue()
    result["synthetic_payloads"] = {}
    for name in ["attributes.pt", "attributes-int4.pt", "attributes-int8.pt"]:
        payload = torch.load(tmp / "checkpoints" / name, map_location="cpu", weights_only=True)
        result["synthetic_payloads"][name] = {"format_version": payload["format_version"], "top_keys_types": {k: type(v).__name__ for k, v in payload.items()}, **accounting(payload["model"])}
    rejection = []
    for source, output in [(native, native), (tmp / "checkpoints/attributes-int8.pt", tmp / "repeated.pt")]:
        with patch.object(sys, "argv", ["scripts/quantize.py", str(source), "--bits", "4", "--output", str(output)]):
            try:
                quantize.main()
            except ValueError as error:
                rejection.append(str(error))
            else:
                raise AssertionError("quantize should reject overwrite or quantized input")
    result["quantize_rejection_messages"] = rejection
    result["infer_chat_cli"] = []
    for name in ["attributes.pt", "attributes-int4.pt"]:
        argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/infer.py"), str(tmp / "checkpoints" / name), "--chat", "--prompt", "color=blue;shape=circle;pitch=low;color?", "--tokens", "24", "--device", "cpu", "--json"]
        completed = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        response = json.loads(completed.stdout)
        assert len(response["generated_ids"]) <= 24
        result["infer_chat_cli"].append({"argv": argv, "exit_code": completed.returncode, "generated_id_count": len(response["generated_ids"]), "generation_status": response["generation_status"], "support_scope": "CLI and JSON contract only; untrained synthetic model, no answer-quality claim"})

    # Exercise the actual CLI metadata branch against a temporary test.jsonl.
    data = tmp / "test.jsonl"
    data.write_text(json.dumps({"question": "a?", "answer": "blue"}) + "\n")
    output = tmp / "test-output.json"
    argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/evaluate.py"), str(native), "--data", str(data), "--mode", "sft", "--tokens", "24", "--limit", "all", "--device", "cpu", "--output", str(output)]
    completed = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    report = json.loads(output.read_bytes())
    result["test_cli_as_in_text"] = {"argv": list(argv), "exit_code": completed.returncode, "data_basename": data.name, "declared_split": report["declared_split"], "records_read": report["records_read"], "records_selected": report["records_selected"], "limit": report["limit"]}
    argv += ["--split-label", "test"]
    completed = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    report = json.loads(output.read_bytes())
    result["test_cli_with_explicit_label"] = {"argv": list(argv), "exit_code": completed.returncode, "declared_split": report["declared_split"]}

# Tensor API and odd int4 padding, using hand chosen numbers rather than trained weights.
x = torch.tensor([-7, 0, 7, -1, 3], dtype=torch.int8)
packed = pack_int4(x)
roundtrip = unpack_int4(packed, x.shape)
assert torch.equal(roundtrip, x)
floating = x.float()
result["packing_and_float_rounding"] = {"source_int8": x.tolist(), "packed_uint8": packed.tolist(), "unpacked_int8": roundtrip.tolist(), "packed_bytes": packed.numel() * packed.element_size(), "float_bytes_before_rounding": floating.numel() * floating.element_size(), "float_bytes_after_rounding": floating.round().numel() * floating.round().element_size()}
tied = TinyLM(ModelConfig(width=4, tied=True)).state_dict()
unique = {}
for name, t in tied.items():
    storage = t.untyped_storage()
    unique[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
result["tied_state_accounting"] = {"embedding_output_shared": tied["embedding.weight"].untyped_storage().data_ptr() == tied["output.weight"].untyped_storage().data_ptr(), "per_entry_tensor_bytes": sum(t.numel() * t.element_size() for t in tied.values()), "unique_storage_bytes": sum(unique.values()), "duplicate_embedding_bytes": tied["embedding.weight"].numel() * tied["embedding.weight"].element_size(), "applicability": "T.4 native model uses tied=False; T.9 explicitly counts each state-table entry"}

# Evaluate's row and answer contracts, with scripted IDs to avoid capability claims.
tok = ByteTokenizer()
records = [{"question": "a?", "answer": "blue"}, {"question": "x" * 140, "answer": "blue"}, {"question": "b?", "answer": "circle"}, {"question": "c?", "answer": "red"}]
tails = [tok.encode("blue") + [tok.eos_id], tok.encode("circle"), [tok.user_id] + tok.encode("red") + [tok.eos_id]]
def scripted_generate(model, ids, max_new_tokens, eos_id):
    assert max_new_tokens == 24
    tail = tails.pop(0)
    return torch.cat([ids, torch.tensor([tail], device=ids.device)], dim=1)
with patch.object(evaluation, "generate", scripted_generate):
    report = evaluation.evaluate(TinyLM(ModelConfig(width=4, max_length=64)), records, mode="sft", max_new_tokens=24)
assert [s["row"] for s in report["samples"]] == [0, 2, 3]
assert [s["row"] for s in report["skipped"]] == [1]
assert report["samples"][0]["completed_exact_match"] is True
assert report["samples"][1]["exact_match"] is True and report["samples"][1]["completed_exact_match"] is False
assert report["samples"][2]["exact_match"] is False
result["scripted_evaluate_contract"] = {"records_selected": report["records_selected"], "generation_evaluated_records": report["generation_evaluated_records"], "metric_denominators": report["metric_denominators"], "samples": [{k: s[k] for k in ["row", "target", "generated", "generated_ids", "generation_status", "exact_match", "completed_exact_match"]} for s in report["samples"]], "skipped_rows": [s["row"] for s in report["skipped"]], "parse_limit_all": evaluation.parse_limit("all")}

# Read-only inspection of existing public exports; hashes differ from original training
# artifacts, so no claim that these are the original .pt files in quantization.json.
result["readonly_public_exports"] = {}
for filename in ["fp32.pt", "model.pt", "packed8.pt"]:
    path = ROOT / "outputs/natural-r4-public-checkpoints/public-models/quantization" / filename
    payload = torch.load(path, map_location="cpu", weights_only=True)
    result["readonly_public_exports"][filename] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "file_bytes": path.stat().st_size, "format_version": payload["format_version"], "bits": payload.get("bits"), **accounting(payload["model"]), "support_scope": "state-dict types and tensor accounting only; public-export file SHA differs from the original-run artifact SHA"}

result["loaded_repo_module_sha256"] = {str(Path(m.__file__).resolve().relative_to(ROOT)): sha(m.__file__) for name, m in sorted(sys.modules.items()) if name.startswith(("tiny_perceptron.", "scripts.")) and getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(ROOT)}
(BASE / "cpu-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
