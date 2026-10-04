"""Bounded current-runtime mechanism check; no model download or formal training."""
import hashlib, inspect, json, math, platform, sys
import torch
from tiny_perceptron.capstone import (
    TOK, CapstoneModel, build_dataset, frozen_reference, preference_loss,
    preference_pairs, prepare_batch,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.model import masked_loss

torch.set_num_threads(2)
torch.manual_seed(42)
splits, manifest = build_dataset()
pairs = preference_pairs(splits["train"])
pair = pairs[0]
policy = CapstoneModel()
reference = frozen_reference(policy)

def fingerprint(model):
    return hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()

before = fingerprint(reference)
policy_before = fingerprint(policy)
loss, details = preference_loss(policy, reference, [pair])
print("同一問題", pair["row"]["user"])
print("較喜歡", pair["chosen"], "較不喜歡", pair["rejected"])
print("相同起點的DPO誤差", round(loss.item(), 4), "參考可更新", any(p.requires_grad for p in reference.parameters()))
print("policy可更新", any(p.requires_grad for p in policy.parameters()))
scores, target_counts = {}, {}
for name in ("chosen", "rejected"):
    batch, labels = prepare_batch([pair["row"]], answers=[pair[name]])
    expected_ids = TOK.encode(pair[name]) + [TOK.eos_id]
    assert labels[labels != IGNORE].tolist() == expected_ids
    logits = policy(**batch)["logits"]
    log_probs = logits.log_softmax(-1)
    manual = sum(float(log_probs[0, i, int(labels[0, i])].detach()) for i in range(labels.shape[1]) if labels[0, i] != IGNORE)
    library = float(sequence_log_probability(logits, labels).detach())
    assert abs(manual - library) < 3e-5
    scores[name] = {"manual_sum": manual, "implementation": library,
                    "input_shape": list(batch["ids"].shape), "logits_shape": list(logits.shape),
                    "dtype": str(logits.dtype), "labels_include_eos": expected_ids[-1] == 2}
    target_counts[name] = len(expected_ids)
    assert int((labels[:,:-(len(expected_ids))] == IGNORE).sum()) == labels.shape[1] - len(expected_ids)
assert abs(loss.item() - math.log(2)) < 1e-6
assert details["policy_margin"] == details["reference_margin"]
assert before == policy_before
assert all(a.data_ptr() != b.data_ptr() for a,b in zip(policy.parameters(),reference.parameters(),strict=True))
assert not reference.training
assert not any(p.requires_grad for p in reference.parameters())
assert all(p.requires_grad for p in policy.parameters())
assert all(p.grad is None for p in reference.parameters())
assert {p["row"]["id"] for p in pairs} <= {r["id"] for r in splits["train"]}
assert not {p["row"]["family"] for p in pairs} & {r["family"] for split in ("validation","test") for r in splits[split]}
assert pair["row"]["user"] == "原題：3+6。計算器回報：9。請回答。"
assert pair["chosen"] == "DIRECT:9" and pair["rejected"] == "DIRECT:9，祝你愉快！"

# One update only, on random weights: prove copy/optimizer separation, not quality.
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0002)
assert not {id(p) for g in optimizer.param_groups for p in g["params"]} & {id(p) for p in reference.parameters()}
batch, labels = prepare_batch([p["row"] for p in pairs[:2]])
result = policy(**batch)
ce = masked_loss(result["logits"], labels)
dpo, _ = preference_loss(policy, reference, pairs[:2], beta=0.1)
objective = dpo + 0.2 * ce + 0.01 * result["auxiliary"]
optimizer.zero_grad(set_to_none=True)
objective.backward()
assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in policy.parameters())
assert all(p.grad is None for p in reference.parameters())
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
after = fingerprint(reference)
policy_after = fingerprint(policy)
assert before == after and policy_before != policy_after
try:
    reference.requires_grad_(True)
    preference_loss(policy, reference, [pair])
    raise AssertionError("unfrozen reference accepted")
except ValueError as error:
    error_message = str(error)
finally:
    reference.requires_grad_(False)

output = {
    "environment": {"python":sys.version.split()[0], "torch":torch.__version__,
        "torch_git":torch.version.git_version,"platform":platform.platform(),
        "device":"cpu","cuda_available":str(torch.cuda.is_available()),"threads":"2"},
    "split_counts":manifest["counts"],"data_version":manifest["version"],"seed":42,
    "preference_pair_count":len(pairs),"preference_task_counts":{task:sum(p["row"]["task"]==task for p in pairs) for task in sorted({p["row"]["task"] for p in pairs})},
    "pair":{key:(value if key != "row" else {k:value[k] for k in ("id","family","task","user","system","answer")}) for key,value in pair.items()},
    "initial_dpo_loss":float(loss.detach()),"analytic_log_2":math.log(2),"margins":details,
    "sequence_score_audit":scores,"valid_answer_positions":target_counts,
    "reference_eval":not reference.training,"reference_requires_grad_any":False,"policy_requires_grad_any":True,
    "own_single_update":{ "updates":1,"batch_rows":2,"preference_pairs":2,"beta":0.1,"ce_coefficient":0.2,"router_coefficient":0.01,
        "objective":float(objective.detach()),"reference_before":before,"reference_after":after,
        "policy_before":policy_before,"policy_after":policy_after,"unfrozen_reference_guard":error_message},
    "scope":"Random CPU mechanism probe and one optimizer update only; no official GPU training replication, model quality evaluation or downloaded weights.",
}
print(json.dumps(output,ensure_ascii=False,indent=2))
