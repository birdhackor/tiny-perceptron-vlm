"""Independent, bounded CPU contract audit for chapter 7.9; no training or weights."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum, masked_loss

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()
report = {"scope": "CPU contract/numeric audits, hand-selected logits and existing JSON recount; no training or model loading",
          "environment": {"python": sys.version, "executable": sys.executable,
                          "torch": str(torch.__version__), "torch_git": str(torch.version.git_version),
                          "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads())}}

chat = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
x, y = render_chat(chat)
assert x.tolist() == [1, 3, 89, 2, 4, 73]
assert y.tolist() == [IGNORE, IGNORE, IGNORE, IGNORE, 73, 2]
assert tok.encode("A") == [65 + 8] and tok.encode("B") == [66 + 8]
assert tok.eos_id == 2 and tok.pad_id == 0
assert tok.encode(".") == [54] and tok.eos_id not in tok.encode("<eos>")
report["labels"] = {"x": x.tolist(), "y": y.tolist(), "prediction_pairs": list(zip(x.tolist(), y.tolist())),
                    "effective_targets": y[y != IGNORE].tolist(), "meaning": "assistant-role row predicts A; A row predicts EOS. Historical user EOS row is ignored."}

# Exercise and boundaries of the exact hand-written stopping rule.
def visible(ids, cap):
    result = []
    seen = []
    for token in ids[:cap]:
        seen.append(token)
        if token == tok.eos_id:
            break
        result.append(token)
    return {"candidate_ids": ids, "cap_new_ids": cap, "checked_ids": seen,
            "visible_ids": result, "text": tok.decode(result), "observed_eos": tok.eos_id in seen}

cases = [("original", [73, 2, 74], 2, "A", True),
         ("exercise_ABCD", tok.encode("ABCD"), 2, "AB", False),
         ("cap_before_EOS", [73, 2, 74], 1, "A", False),
         ("EOS_first", [2, 73], 2, "", True),
         ("no_EOS", [73, 74], 2, "AB", False),
         ("zero_budget", [73, 2], 0, "", False)]
report["hand_loop"] = {}
for name, ids, cap, text, eos in cases:
    value = visible(ids, cap)
    assert value["text"] == text and value["observed_eos"] == eos
    report["hand_loop"][name] = value

# Loss axis, valid denominator and EOS gradient; computes gradients, no update.
logits = torch.zeros((1, len(x), tok.vocab_size), dtype=torch.float64, requires_grad=True)
total, count = loss_sum(logits, y[None])
mean = masked_loss(logits, y[None])
mean.backward()
expected = math.log(264)
assert count.item() == 2
assert abs(mean.item() - expected) < 1e-12
assert abs(total.item() - 2 * expected) < 1e-12
assert torch.count_nonzero(logits.grad[0, :4]).item() == 0
assert logits.grad[0, -1, 2].item() < 0
assert abs(logits.grad[0, -1, 2].item() - (1 / 264 - 1) / 2) < 1e-12
report["loss_and_EOS_gradient"] = {"logits_shape_B_T_V": list(logits.shape), "dtype": str(logits.dtype),
    "effective_denominator": count.item(), "sum": total.item(), "mean": mean.item(),
    "expected_mean_ln264": expected, "absolute_tolerance": 1e-12,
    "ignored_rows_nonzero_gradients": torch.count_nonzero(logits.grad[0, :4]).item(),
    "EOS_target_gradient": logits.grad[0, -1, 2].item(), "optimizer_updates": 0}

longer = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "AB"}])
px, py, valid = pad_batch([(x, y), longer])
assert px[0, -1].item() == tok.pad_id
assert py[0, -1].item() == IGNORE and not valid[0, -1].item()
assert py[0, -2].item() == tok.eos_id and valid[0, -2].item()
report["padding"] = {"x": px.tolist(), "labels": py.tolist(), "valid": valid.tolist(),
                     "EOS_is_a_valid_target": True, "PAD_is_not_a_target": True}

class ScriptedScores(nn.Module):
    """Chosen output logits exercise the real generate function, not a trained LM."""
    def __init__(self, choices, max_length=128):
        super().__init__()
        self.choices = choices
        self.config = SimpleNamespace(max_length=max_length)
        self.calls = []

    def forward(self, ids, cache=None):
        index = len(self.calls)
        chosen = self.choices[index]
        self.calls.append({"input_shape": list(ids.shape), "input_ids": ids.tolist(),
                           "cache_supplied": cache is not None, "selected_id": chosen})
        scores = torch.full((*ids.shape, tok.vocab_size), -100.0)
        scores[:, -1, chosen] = 5.0
        return {"logits": scores, "cache": ("scripted-cache",)}

prompt = torch.tensor([[1, 3, 89, 2, 4]])
report["real_generate_contract"] = {}
for use_cache in (False, True):
    model = ScriptedScores([73, 2, 74])
    result = generate(model, prompt, max_new_tokens=3, use_cache=use_cache)
    new_ids = result[0, prompt.shape[1]:].tolist()
    assert new_ids == [73, 2] and model.training
    assert len(model.calls) == 2
    report["real_generate_contract"]["historical_EOS_cached_" + str(use_cache)] = {
        "prompt": prompt.tolist(), "new_ids_including_EOS": new_ids,
        "visible_text": tok.decode(new_ids), "calls": model.calls, "training_flag_restored": model.training}

for name, choices, budget, model_cap, expected_ids in [
    ("new_token_budget", tok.encode("ABCD"), 2, 128, [73, 74]),
    ("context_limit", [73, 74, 75], 3, 6, [73]),
    ("already_full_context", [73], 1, 5, []),
    ("no_new_token_budget", [73], 0, 128, []),
    ("PAD_is_not_EOS", [0, 73, 2], 3, 128, [0, 73, 2])]:
    model = ScriptedScores(choices, max_length=model_cap)
    result = generate(model, prompt, max_new_tokens=budget)
    new_ids = result[0, prompt.shape[1]:].tolist()
    assert new_ids == expected_ids
    report["real_generate_contract"][name] = {"budget": budget, "model_context_cap": model_cap,
        "new_ids": new_ids, "visible_text": tok.decode(new_ids), "observed_new_EOS": 2 in new_ids,
        "calls": model.calls}

circle_ids = tok.encode("圓")
assert len(circle_ids) == 3
assert tok.decode(circle_ids) == "圓" and tok.decode(circle_ids[:1]) == "\ufffd"
assert tok.decode(circle_ids[:2]) == "\ufffd"
report["UTF8_byte_cap"] = {"complete_character": "圓", "ids": circle_ids,
    "decoded_at_caps_1_2_3": [tok.decode(circle_ids[:n]) for n in (1, 2, 3)],
    "decoder_error_policy": "replace, not complete-character repair", "token_unit": "one UTF-8 byte or special ID"}

# A real untrained network checks the vocabulary score axis, not learning or quality.
torch.manual_seed(20261005)
small = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=16))
with torch.no_grad():
    scores = small(x[None])["logits"]
assert scores.shape == (1, 6, 264) and scores.isfinite().all()
report["real_untrained_forward"] = {"shape_B_T_V": list(scores.shape), "finite": True,
    "parameters_updated": False, "weights_saved": False, "quality_inference": "none"}

# Recount the existing fixed 900-step CUDA experiment, without loading its model.
raw_path = Path(__file__).parent / "sft-original-results.json"
document = json.loads(raw_path.read_text())
results = document["results"]
assert results["training"]["steps"] == 900
audit = {"input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
         "recorded_code_revision": document["revision"], "recorded_device": document["device"],
         "recorded_model_branch": results["checkpoint"], "recorded_training_steps": results["training"]["steps"],
         "retraining_performed": False, "splits": {}}
for split, expected_count in [("validation", 5), ("test", 10)]:
    data = results["after"][split]
    samples = data["samples"]
    assert len(samples) == data["records"] == data["examples"] == results["data"][split]["records"] == expected_count
    eos = matches = effective = generated_tokens = 0
    recounted_rows = []
    for row_index, row in enumerate(samples):
        ids = row["generated_ids"]
        ended = 2 in ids
        content = ids[:ids.index(2)] if ended else ids
        exact = content == tok.encode(row["expected"])
        decoded = tok.decode(content)
        assert decoded == row["generated"] and exact == row["exact"] and ended == row["eos"]
        assert ended and ids[-1] == 2 and ids.count(2) == 1
        eos += int(ended)
        matches += int(exact)
        effective += len(tok.encode(row["expected"])) + 1
        generated_tokens += len(ids)
        recounted_rows.append({"row": row_index, "question": row["messages"][0]["content"],
            "expected": row["expected"], "ids": ids, "decoded": decoded, "EOS": ended, "exact": exact})
    assert effective == data["effective_tokens"]
    assert eos / expected_count == data["eos_rate"]
    assert matches == data["matches"] and matches / expected_count == data["exact_match"]
    assert abs(data["nll_sum"] / effective - data["nll"]) < 1e-12
    audit["splits"][split] = {"records": expected_count, "EOS_count": eos, "EOS_rate": eos / expected_count,
        "exact_count": matches, "content_errors": expected_count - matches, "exact_rate": matches / expected_count,
        "effective_gold_targets_with_EOS": effective, "generated_ID_count_with_EOS": generated_tokens,
        "NLL_sum": data["nll_sum"], "recounted_NLL": data["nll_sum"] / effective,
        "tolerance": 1e-12, "rows": recounted_rows}
first = audit["splits"]["validation"]["rows"][0]
assert first["expected"] == "circle" and first["ids"] == [107, 113, 122, 109, 2]
assert first["decoded"] == "cire" and first["EOS"] and not first["exact"]
audit["original_run_code_hash_check"] = {}
for name, source in [("original-run-model.py", "tiny_perceptron/model.py"),
                     ("original-run-common.py", "scripts/course_experiments/common.py"),
                     ("original-run-text.py", "scripts/course_experiments/text.py")]:
    digest = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    assert digest == document["code_sha256"][source]
    audit["original_run_code_hash_check"][source] = digest
report["existing_result_recount"] = audit
report["input_provenance"] = {"section": "raw section.md extracted by original_section without newline normalization",
    "manual_candidates": "explicitly hand chosen for mechanics; not trained outputs",
    "scripted_scores": "one unambiguous high next-token logit; real generate called",
    "existing_results": "copied verbatim docs/course-experiments/results/sft.json; code snapshots from git show recorded revision",
    "UTF8_example": "Python UTF-8 encoding of 圓"}
report["hashes"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                     for p in ("tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py")}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
