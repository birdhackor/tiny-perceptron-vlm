"""Independent 18.6 checks; CPU only, no training, model loading, or generation."""
import ast
import copy
import hashlib
import io
import json
import platform
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_functions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return {n.name: {"first_line": n.lineno, "last_line": n.end_lineno} for n in selected}


report = {"environment": {"python": platform.python_version(), "executable": sys.executable,
          "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
          "device": "cpu", "cuda_build": str(torch.version.cuda), "threads": str(torch.get_num_threads())}}

# Run the exact extracted original fence, with its original UTF-8 bytes.
code = (BASE / "inputs/fence-1.py").read_bytes()
ns = {}
stdout = io.StringIO()
with redirect_stdout(stdout):
    exec(compile(code, "course/chapters/18.md#18.6:fence-1", "exec"), ns)
assert ns["order"] == [1, 0]
assert torch.equal(ns["p"], ns["reordered"])
report["original_fence"] = {"sha256": hashlib.sha256(code).hexdigest(), "stdout": stdout.getvalue(),
    "order": ns["order"], "raw_mae": (ns["p"] - ns["q"]).abs().mean().item(),
    "aligned_mae": (ns["p"] - ns["reordered"]).abs().mean().item(),
    "probability_axis": 0, "candidate_count_and_mae_denominator": 2}

# The exercise changes both the identity table and the probabilities.
teacher_vocab, student_vocab = ["貓", "狗"], ["貓", "狗"]
p, q = torch.tensor([0.8, 0.2]), torch.tensor([0.8, 0.2])
order = [student_vocab.index(token) for token in teacher_vocab]
assert order == [0, 1]
assert (p - q).abs().mean().item() == (p - q[order]).abs().mean().item() == 0
q_wrong = torch.tensor([0.2, 0.8])
wrong_mae = (p - q_wrong[order]).abs().mean().item()
assert abs(wrong_mae - 0.6) < 1e-7
report["exercise"] = {"order": order, "direct_mae": 0, "reordered_mae": 0,
    "only_vocab_changed_mae": wrong_mae, "tolerance": "absolute 1e-7 for float32 MAE; exact indices"}
try:
    [ ["小", "貓", "狗"].index(token) for token in ["小貓", "狗"] ]
except ValueError:
    report["granularity_variant"] = "No candidate 小貓 in student vocabulary; permutation cannot split one token into two."
else:
    raise AssertionError("Expected absence of one-to-one token identity mapping")

# Execute the original 5af615e source methods, without importing result explanation strings.
namespace = {"torch": torch, "F": F, "time": time}
data_code = BASE / "code/original-data.py"
exec(compile(data_code.read_bytes(), str(data_code), "exec"), namespace)
locations = selected_functions(BASE / "code/original-compression.py",
    ["_sync", "_chat", "_example", "_examples", "_prompt", "_cache_text"], namespace)
locations.update(selected_functions(BASE / "code/original-alignment.py", ["distillation_kl"], namespace))
tok = namespace["ByteTokenizer"]()
IGNORE = namespace["IGNORE"]
assert tok.vocab_size == 256 + 8 == 264
assert [tok.pad_id, tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id,
        tok.image_id, tok.audio_id, tok.system_id] == list(range(8))
assert tok.decode(tok.encode("小貓追")) == "小貓追"
assert set(tok.encode("小貓追")).isdisjoint(range(8))
report["tokenizer"] = {"vocab_size": tok.vocab_size, "specials": list(namespace["SPECIALS"]),
    "ids": list(range(8)), "byte_id_range": [8, 263], "chinese_roundtrip": True}

# Inspect the original cache only, not a saved model. Verify result-version identities.
raw = json.loads((BASE / "sources/distillation.json").read_bytes())
task = raw["results"]["tasks"]["attributes"]
artifacts = {x["path"]: x for x in raw["artifacts"]}
for filename in ["sft-teacher-logits.pt", "sft-hard-targets.json"]:
    assert sha(BASE / "sources" / filename) == artifacts[filename]["sha256"]
hard = json.loads((BASE / "sources/sft-hard-targets.json").read_bytes())
cache_path = BASE / "sources/sft-teacher-logits.pt"
cache = torch.load(cache_path, map_location="cpu", weights_only=True)
report["cache_key_types"] = {k: type(v).__name__ for k, v in cache.items()}
assert set(cache) == {"logits", "teacher_sha256", "alignment", "temperature"}
assert cache["teacher_sha256"] == task["teacher_provenance"]["sha256"]
assert len(cache["logits"]) == len(hard["records"]) == len(hard["audit"]) == 45
assert task["teacher_provenance"]["config"]["width"] == 64
assert task["teacher_provenance"]["config"]["vocab_size"] == 264
counts, gold_records = [], []
for i, (record, audit, logits) in enumerate(zip(hard["records"], hard["audit"], cache["logits"], strict=True)):
    assert record["_hard_ids"] == audit["teacher_ids"]
    assert audit["teacher_ids"] == tok.encode(audit["gold_answer"]) + [tok.eos_id]
    assert audit["question"] == record["messages"][-2]["content"]
    gold = copy.deepcopy(record)
    gold["messages"][-1]["content"] = audit["gold_answer"]
    gold.pop("_hard_ids"); gold.pop("_hard_eos")
    x, y = namespace["_example"](gold, 128)
    hard_x, hard_y = namespace["_example"](record, 128)
    assert torch.equal(x, hard_x) and torch.equal(y, hard_y)
    count = int((y != IGNORE).sum())
    assert logits.shape == (count, 264)
    assert logits.device.type == "cpu" and torch.isfinite(logits).all()
    counts.append(count); gold_records.append(gold)
report["original_cache"] = {"cache_sha256": sha(cache_path), "records": len(counts),
    "supervised_token_rows": sum(counts), "row_counts_in_record_order": counts,
    "vocabulary_columns": 264, "alignment": cache["alignment"], "temperature": cache["temperature"],
    "teacher_width": task["teacher_provenance"]["config"]["width"],
    "student_widths_from_run_names": [16, 32], "all_cache_row_shapes_match_gold_answer_labels": True,
    "hard_ids_kept_separately": True, "model_evaluation_or_training_performed": False}

# A tiny deterministic stand-in isolates the actual cache slicing/order method.
class MarkerTeacher:
    config = type("Config", (), {"max_length": 128})()
    def eval(self): return self
    def requires_grad_(self, enabled): assert not enabled; return self
    def __call__(self, x, valid):
        self.last_x, self.last_valid = x.clone(), valid.clone()
        marker = x.cumsum(1).float()
        return {"logits": marker[:, :, None].expand(-1, -1, 264).clone()}

records = gold_records[:2]
fake = MarkerTeacher()
synthetic_cache, _ = namespace["_cache_text"](fake, records, "cpu")
examples = namespace["_examples"](records, 128)
x, y, valid = namespace["pad_batch"](examples)
for i, logits in enumerate(synthetic_cache):
    assert torch.equal(logits[:, 0], x.cumsum(1)[i, y[i] != IGNORE].float())
assert sum(len(t) for t in synthetic_cache) == int((y != IGNORE).sum())
assert int((y != IGNORE).sum()) < int(valid.sum())
report["cache_method_variant"] = {"records": 2, "supervised_rows": int((y != IGNORE).sum()),
    "visible_input_positions": int(valid.sum()), "padding_positions": int((~valid).sum()),
    "question_and_padding_excluded": True, "prefix_cumulative_markers_preserved_in_row_order": True,
    "scope": "Synthetic stand-in checks cache algorithm only; no teacher quality claim."}

# Same shapes are insufficient: wrong-prefix distributions can still compare numerically.
kl = namespace["distillation_kl"]
teacher = torch.tensor([[[2., 0.], [0., 2.]]])
student = teacher.clone()
labels = torch.tensor([[8, IGNORE]])
base_kl = kl(student, teacher, labels, 1.).item()
ignored_change = student.clone(); ignored_change[0, 1] = torch.tensor([100., -100.])
assert abs(kl(ignored_change, teacher, labels, 1.).item() - base_kl) < 1e-7
wrong_prefix = teacher.flip(-1)
wrong_kl = kl(wrong_prefix, teacher, labels, 1.).item()
assert wrong_kl > 1
try:
    kl(torch.zeros(1, 2, 2), torch.zeros(1, 3, 2), labels)
except ValueError:
    mismatch_rejected = True
else:
    raise AssertionError("Shape mismatch must fail before KL calculation")
report["kl_mask_and_shape_variant"] = {"base_kl": base_kl, "ignored_position_change_has_no_effect": True,
    "same_shape_wrong_conditioning_kl": wrong_kl, "shape_mismatch_rejected": mismatch_rejected,
    "scope": "Toy logits expose semantics and masking, not actual model learning or accuracy."}
report["executed_original_method_locators"] = locations
report["result_revision"] = raw["revision"]
report["original_run_environment"] = {k: raw[k] for k in ["device", "seed", "torch_version", "python_version", "step_scale"]}
(BASE / "cpu-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
