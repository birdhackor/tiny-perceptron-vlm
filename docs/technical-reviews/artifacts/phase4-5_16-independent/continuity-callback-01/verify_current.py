"""Bounded real owner callback: current math fence and new PAD wording contract."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat, pad_batch
from tiny_perceptron.model import loss_sum, masked_loss

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def show(name, value):
    print(name, json.dumps(value, ensure_ascii=False, sort_keys=True))

spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
inputs = json.loads((BASE / "current-inputs.json").read_text())
for lesson in ["5.16", "5.2", "5.15", "7.16"]:
    source = inputs[lesson]["source"].partition("#")[0]
    raw, _, first = helper.original_section(ROOT / source, lesson)
    assert raw == (BASE / "current" / (lesson + ".md")).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == inputs[lesson]["sha256"]
show("current_inputs_exact", inputs)

body = (BASE / "current/5.16.md").read_bytes()
fences = helper.fences(body, inputs["5.16"]["first_line"])
assert len(fences) == 1 and fences[0]["closed"] and fences[0]["language"] == "python"
assert fences[0]["raw"] == (BASE / "current/fence-1.py").read_bytes() == (PRIOR / "original/fence-1.py").read_bytes()
assert not re.findall(rb'!\[[^\]]*\]\([^)]+\)|<(?:img|svg)\b', body)
namespace = {}
exec(compile(fences[0]["raw"], "current-5.16-fence", "exec"), namespace)
assert namespace["tokens"] == {"短答": 90, "長答": 200}
assert namespace["total"] == 290
assert namespace["ratios"] == {"短答": 90 / 290, "長答": 200 / 290}
show("current_fence", {"sha256": sha(BASE / "current/fence-1.py"), "counts": namespace["tokens"], "total": 290, "ratio": namespace["ratios"], "new_code_changes": False})

# Two tiny hard-coded chat fixtures verify the newly explained padding term.
examples = [render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": answer}]) for question, answer in [("Q", "x"), ("QQQQ", "abc")]]
x, y, valid = pad_batch(examples)
assert tuple(y.shape) == (2, 11)
assert int((~valid).sum()) == 5
assert torch.all(x[~valid] == ByteTokenizer.pad_id)
assert torch.all(y[~valid] == IGNORE)
assert (y != IGNORE).sum(dim=1).tolist() == [2, 4]
assert int((valid & (y == IGNORE)).sum()) == 11
logits = torch.zeros(2, 11, ByteTokenizer.vocab_size, dtype=torch.float64)
summed, n = loss_sum(logits, y)
mean = masked_loss(logits, y)
assert int(n) == 6 and math.isclose(float(mean), math.log(264), abs_tol=1e-12)
changed = logits.clone()
changed[y == IGNORE] = torch.linspace(-100.0, 100.0, 264, dtype=torch.float64)
assert torch.equal(masked_loss(changed, y), mean)
show("new_pad_wording_contract", {"input_shape": list(x.shape), "padded_cells": x.numel(), "padding_cells": 5, "padding_input_id": ByteTokenizer.pad_id, "padding_target": IGNORE, "nonpadding_ignored_question_role_positions": 11, "valid_answer_labels_including_EOS_per_record": [2, 4], "loss_denominator": int(n), "sum_NLL": float(summed), "mean_NLL": float(mean), "changing_all_ignored_scores_does_not_change_loss": True, "model_or_optimizer_created": False})

identities = []
for path in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "docs/course-experiments/results/sft_ablation.json"]:
    assert (ROOT / path).read_bytes() == (PRIOR / "inputs" / path).read_bytes()
    identities.append({"current_path": path, "original_snapshot": str((PRIOR / "inputs" / path).relative_to(ROOT)), "sha256": sha(ROOT / path), "unchanged": True})
original = json.loads((PRIOR / "inputs/docs/course-experiments/results/sft_ablation.json").read_bytes())
show("original_named_pointers", {"/revision": original["revision"], "/seed": original["seed"], "/results/runs/b-only/training/steps": original["results"]["runs"]["b-only"]["training"]["steps"], "/results/runs/b-only/training/effective_tokens": original["results"]["runs"]["b-only"]["training"]["effective_tokens"], "/results/runs/replay/training/steps": original["results"]["runs"]["replay"]["training"]["steps"], "/results/runs/replay/training/effective_tokens": original["results"]["runs"]["replay"]["training"]["effective_tokens"]})
assert original["results"]["runs"]["b-only"]["training"]["effective_tokens"] == 18453
assert original["results"]["runs"]["replay"]["training"]["effective_tokens"] == 36384
for name in ["attributes", "arithmetic"]:
    record_path = PRIOR / "inputs/original-training-records" / name / "train.jsonl"
    assert sha(record_path) == original["results"]["data"][name]["train"]["sha256"]
for source in json.loads((PRIOR / "sources/retrieval.json").read_text()):
    assert sha(PRIOR / "sources" / Path(source["path"]).name) == source["sha256"]
show("preserved_original_evidence", identities)
environment = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()), "scope": "Current fence plus two tiny chat fixtures for PAD/masking; original-code/data/result/source identity checks. No training, model, optimizer, weights, network, or new dataset preparation."}
(BASE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
show("environment", environment)
print("CURRENT_VERSION_CALLBACK_ASSERTIONS_PASSED")
