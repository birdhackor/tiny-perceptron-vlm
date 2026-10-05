"""Fresh 18.4 CPU verification: derivatives and inspection of existing raw evidence.

No optimizer, trained model inference, training, or network operation is used.
"""
import ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, IGNORE, pad_batch, render_chat

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


result = {
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "dtype_numeric": "float64",
        "scope": "raw artifact inspection plus bounded synthetic tensor calculations",
    },
    "numerics": [],
    "input_hashes": {},
}
q = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float64)
for name, p in [
    ("hard", [1.0, 0.0, 0.0]),
    ("soft", [0.7, 0.2, 0.1]),
    ("wrong_teacher_exercise", [0.2, 0.7, 0.1]),
]:
    target = torch.tensor(p, dtype=torch.float64)
    assert abs(target.sum().item() - 1.0) < 1e-14
    z = q.log().detach().clone().requires_grad_()
    initial = z.detach().clone()
    loss = -(target * z.log_softmax(0)).sum()
    loss.backward()
    expected = q - target
    torch.testing.assert_close(z.grad, expected, atol=1e-14, rtol=0)
    analytic_ce = -sum(pi * math.log(qi) for pi, qi in zip(p, q.tolist()))
    assert abs(loss.item() - analytic_ce) < 1e-14
    fd = []
    for i in range(3):
        step = torch.zeros_like(z)
        step[i] = 1e-6
        plus = -(target * (initial + step).log_softmax(0)).sum()
        minus = -(target * (initial - step).log_softmax(0)).sum()
        fd.append(((plus - minus) / (2e-6)).item())
    torch.testing.assert_close(torch.tensor(fd, dtype=torch.float64), expected, atol=2e-10, rtol=0)
    assert torch.equal(z.detach(), initial), "backward does not update the logits"
    result["numerics"].append({
        "case": name, "target": p, "ce_nats": loss.item(),
        "gradient": z.grad.tolist(), "finite_difference": fd,
        "gradient_sum": z.grad.sum().item(), "unchanged_after_backward": True,
    })

# A highest-ranked teacher candidate need not be pushed upwards at the current
# student distribution. Its own target probability must exceed the student's.
counter_target = torch.tensor([0.34, 0.35, 0.31], dtype=torch.float64)
counter_z = q.log().detach().clone().requires_grad_()
counter_loss = -(counter_target * counter_z.log_softmax(0)).sum()
counter_loss.backward()
assert counter_target.argmax().item() == 1
assert abs(counter_z.grad[1].item() - 0.05) < 1e-14
counter_step = counter_z.detach() - 0.1 * counter_z.grad
assert counter_step[1] < counter_z.detach()[1]
result["counterexample_teacher_argmax"] = {
    "student": q.tolist(), "teacher": counter_target.tolist(),
    "teacher_highest_candidate": "3", "gradient": counter_z.grad.tolist(),
    "candidate3_score_before": counter_z.detach()[1].item(),
    "candidate3_score_after_one_step_lr_0_1": counter_step[1].item(),
    "implication": "Highest teacher probability alone does not imply a negative score gradient for that candidate.",
}

# Ordinary API group: detach severs a real graph, clone has independent storage,
# requires_grad_ creates a leaf; backward accumulates only when the leaf is reused.
upstream = q.clone().requires_grad_()
logged = upstream.log()
detached = logged.detach()
z = detached.clone().requires_grad_()
assert detached.data_ptr() == logged.data_ptr()
assert z.data_ptr() != detached.data_ptr() and z.is_leaf
(-(torch.tensor([0.7, 0.2, 0.1]) * z.log_softmax(0)).sum()).backward()
assert upstream.grad is None
first = z.grad.clone()
(-(torch.tensor([0.7, 0.2, 0.1]) * z.log_softmax(0)).sum()).backward()
torch.testing.assert_close(z.grad, 2 * first)
extreme = torch.tensor([-1000.0, 0.0, 1000.0]).log_softmax(0)
assert torch.isfinite(extreme).all()
assert extreme.tolist() == [-2000.0, -1000.0, 0.0]
result["api_group"] = {
    "detach_graph_severed": upstream.grad is None,
    "detach_storage_shared": True, "clone_storage_independent": True,
    "requires_grad_leaf": z.is_leaf, "backward_accumulates_same_leaf": True,
    "stable_log_softmax_extreme": extreme.tolist(),
    "candidate_axis": 0, "loss_reduction": "sum over 3 candidates; one position",
}

# Named raw pointers only. No results summaries or author interpretation is read.
rp = HERE / "sources/original-distillation-result.json"
raw = json.loads(rp.read_text())
a = raw["results"]["tasks"]["attributes"]
selected = {
    "/revision": raw["revision"],
    "/device": raw["device"],
    "/seed": raw["seed"],
    "/torch_version": raw["torch_version"],
    "/python_version": raw["python_version"],
    "/code_sha256/scripts~1course_experiments~1compression.py": raw["code_sha256"]["scripts/course_experiments/compression.py"],
    "/code_sha256/tiny_perceptron~1data.py": raw["code_sha256"]["tiny_perceptron/data.py"],
    "/results/tasks/attributes/teacher_cache": a["teacher_cache"],
    "/results/tasks/attributes/teacher_provenance/sha256": a["teacher_provenance"]["sha256"],
    "/results/tasks/attributes/teacher_provenance/config/vocab_size": a["teacher_provenance"]["config"]["vocab_size"],
    "/results/tasks/attributes/hard_target_generation/file": a["hard_target_generation"]["file"],
    "/results/tasks/attributes/hard_target_generation/file_bytes": a["hard_target_generation"]["file_bytes"],
    "/results/tasks/attributes/hard_target_generation/sha256": a["hard_target_generation"]["sha256"],
    "/results/tasks/attributes/hard_target_generation/records": a["hard_target_generation"]["records"],
    "/results/tasks/attributes/data/student_training_records": a["data"]["student_training_records"],
}
manifest = {}
for i, artifact in enumerate(raw["artifacts"]):
    if artifact["path"] in ["sft-teacher-logits.pt", "sft-hard-targets.json"]:
        manifest[artifact["path"]] = artifact
        selected[f"/artifacts/{i}"] = artifact
assert len(manifest) == 2

lp = HERE / "sources/sft-teacher-logits.pt"
hp = HERE / "sources/sft-hard-targets.json"
for p in [rp, lp, hp, HERE / "code/compression-measurement-5af615e.py", HERE / "code/data.py"]:
    result["input_hashes"][str(p.relative_to(HERE))] = sha(p)
assert sha(HERE / "code/compression-measurement-5af615e.py") == selected["/code_sha256/scripts~1course_experiments~1compression.py"]
assert sha(HERE / "code/data.py") == selected["/code_sha256/tiny_perceptron~1data.py"]
for p in [lp, hp]:
    assert p.stat().st_size == manifest[p.name]["bytes"]
    assert sha(p) == manifest[p.name]["sha256"]
assert lp.stat().st_size == selected["/results/tasks/attributes/teacher_cache"]["file_bytes"] == 345101
cache = torch.load(lp, map_location="cpu", weights_only=True)
hard = json.loads(hp.read_text())
result["raw_schemas"] = {
    "result_top": {k: type(v).__name__ for k, v in raw.items()},
    "cache_top": {k: type(v).__name__ for k, v in cache.items()},
    "hard_top": {k: type(v).__name__ for k, v in hard.items()},
}
assert cache["teacher_sha256"] == selected["/results/tasks/attributes/teacher_provenance/sha256"]
assert len(cache["logits"]) == len(hard["records"]) == len(hard["audit"]) == 45
assert selected["/results/tasks/attributes/data/student_training_records"] == 45
gold_rows = []
hard_id_tokens = 0
for i, (record, audit, logits) in enumerate(zip(hard["records"], hard["audit"], cache["logits"])):
    assert record["_hard_ids"] == audit["teacher_ids"]
    assert len(record["_hard_ids"]) == audit["valid_target_tokens"]
    assert record["messages"][-2]["content"] == audit["question"]
    assert record["family"] == audit["family"]
    assert logits.ndim == 2 and logits.shape[1] == 264
    assert logits.dtype == torch.float32 and logits.device.type == "cpu" and torch.isfinite(logits).all()
    gold = [dict(message) for message in record["messages"]]
    gold[-1]["content"] = audit["gold_answer"]
    _, y = render_chat(gold)
    assert logits.shape[0] == (y != IGNORE).sum().item()
    gold_rows.append(logits.shape[0])
    hard_id_tokens += len(record["_hard_ids"])
result["empirical"] = {
    "selected_result_pointers": selected,
    "cache_read_pointers": ["/logits/*", "/teacher_sha256", "/alignment", "/temperature"],
    "hard_read_pointers": ["/records/*/messages", "/records/*/family", "/records/*/_hard_ids", "/audit/*/question", "/audit/*/family", "/audit/*/teacher_ids", "/audit/*/valid_target_tokens", "/audit/*/gold_answer"],
    "cache_metadata": {k: cache[k] for k in ["teacher_sha256", "alignment", "temperature"]},
    "records": 45, "gold_valid_positions": sum(gold_rows),
    "candidates_per_position": 264, "logit_scalar_count": sum(gold_rows) * 264,
    "gold_positions_per_record": gold_rows, "hard_generated_id_tokens": hard_id_tokens,
    "cache_bytes": lp.stat().st_size, "hard_file_bytes": hp.stat().st_size,
    "manifest_bytes_and_hashes_match": True,
    "cache_rows_match_gold_assistant_labels": True,
    "hard_ids_match_every_audit_row": True,
}

# Execute the original measurement-version cache method with a bounded synthetic
# teacher to check mask and prefix semantics without loading any model weights.
source_path = HERE / "code/compression-measurement-5af615e.py"
tree = ast.parse(source_path.read_text())
wanted = {"_chat", "_example", "_examples", "_cache_text"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
assert len(nodes) == 4
ns = dict(torch=torch, time=time, ByteTokenizer=ByteTokenizer, IGNORE=IGNORE,
          pad_batch=pad_batch, render_chat=render_chat, _sync=lambda device: None)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), ns)


class SyntheticTeacher:
    config = SimpleNamespace(max_length=128)

    def eval(self):
        return self

    def requires_grad_(self, enabled):
        assert enabled is False
        return self

    def __call__(self, x, valid):
        self.inputs = x.clone()
        # Explicit algebraic fixture, not a model-quality evaluation.
        return {"logits": x.cumsum(1).to(torch.float32)[..., None] + torch.arange(264)[None, None, :]}


records = [
    {"question": "same prompt", "answer": "ab"},
    {"question": "same prompt", "answer": "xy"},
]
teacher = SyntheticTeacher()
cached, _ = ns["_cache_text"](teacher, records, "cpu")
examples = ns["_examples"](records, 128)
x, y, valid = pad_batch(examples)
torch.testing.assert_close(teacher.inputs, x)
assert len(cached) == 2 and [list(t.shape) for t in cached] == [[3, 264], [3, 264]]
assert torch.equal(cached[0][0], cached[1][0])
assert not torch.equal(cached[0][1], cached[1][1])
result["cache_method_fixture"] = {
    "executed_original_functions": sorted(wanted), "records": 2,
    "gold_answer_labels_including_eos": [(yi != IGNORE).sum().item() for _, yi in examples],
    "shapes": [list(t.shape) for t in cached],
    "same_question_first_prediction_row_identical": True,
    "different_gold_first_token_changes_next_row": True,
    "no_free_generation_called": True,
}
result["all_assertions_passed"] = True
(HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({
    "all_assertions_passed": True,
    "numerics": result["numerics"],
    "counterexample_teacher_argmax": result["counterexample_teacher_argmax"],
    "raw_cache": {k: result["empirical"][k] for k in ["records", "gold_valid_positions", "candidates_per_position", "logit_scalar_count", "cache_bytes", "hard_file_bytes", "hard_generated_id_tokens"]},
    "fixture": result["cache_method_fixture"],
}, ensure_ascii=False, indent=2))
