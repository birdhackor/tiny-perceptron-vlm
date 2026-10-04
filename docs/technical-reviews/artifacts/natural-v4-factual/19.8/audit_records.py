"""Independently rescore original official records, not GPU replication."""
from pathlib import Path
from collections import Counter
import hashlib, inspect, json, random, re, sys, torch
from tiny_perceptron.capstone import build_dataset, preference_pairs

ROOT = Path(__file__).resolve().parents[5]
RAW = ROOT / "outputs/natural-v4/factual-research/19.8/official"

def load(path):
    return json.loads((RAW / path).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

data = load("docs/course-experiments/capstone-evidence/deployment/data.json")
splits, manifest = build_dataset(42)
assert data["splits"] == splits and data["manifest"] == manifest
expected_rows = {name:{r["id"]:r for r in rows} for name,rows in splits.items()}

def trace_value(trace):
    if trace is None:
        return None, False
    ids = trace["generated_ids"]
    decoded = bytes(i - 8 for i in ids if i >= 8).decode("utf-8",errors="replace")
    terminal = bool(ids) and ids[-1] == 2
    assert decoded == trace["raw"] and terminal == trace["eos"]
    if terminal:
        assert trace["stop_reason"] == "eos"
    return decoded, terminal

def rescore(path, split):
    doc = load(path)
    assert len(doc["records"]) == len(expected_rows[split])
    assert {r["id"] for r in doc["records"]} == set(expected_rows[split])
    totals, by_task, failures, style = Counter(), {}, [], []
    for record in doc["records"]:
        row = expected_rows[split][record["id"]]
        raw, eos = trace_value(record["action_trace"])
        final_raw, final_eos = trace_value(record["final_trace"])
        action_ok = eos and raw == row["answer"]
        answer = None
        if eos and raw.startswith("DIRECT:"):
            answer = raw[7:]
        elif eos and raw.startswith("ASK:"):
            answer = raw[4:]
        elif eos and (match := re.fullmatch(r"TOOL:calculator:([0-9]{1,3})\+([0-9]{1,3})",raw)):
            if row["available"]:
                assert record["runtime"] == {"status":"ok","result":str(int(match[1])+int(match[2]))}
                if final_eos and final_raw.startswith("DIRECT:"):
                    answer = final_raw[7:]
        expected_action = row["answer"]
        if expected_action.startswith("TOOL:"):
            request = re.fullmatch(r"TOOL:calculator:(\d+)\+(\d+)",expected_action)
            final_expected = str(int(request[1])+int(request[2]))
        else:
            final_expected = expected_action.split(":",1)[1]
        end_ok = bool(action_ok and answer == final_expected)
        assert record["expected_action"] == expected_action
        assert record["expected_final"] == final_expected
        assert answer == record["answer"]
        assert bool(action_ok) == record["action_correct"] and end_ok == record["end_to_end_correct"]
        task = by_task.setdefault(row["task"],Counter())
        for target in (totals, task):
            target["count"] += 1
            target["action_correct"] += int(action_ok)
            target["end_to_end_correct"] += int(end_ok)
        if not end_ok:
            failures.append({"id":row["id"],"task":row["task"],"user":row["user"],"expected":expected_action,"generated":raw,"final":final_raw})
        if row["task"] == "style":
            style.append({"id":row["id"],"user":row["user"],"generated":raw,"eos":eos,"end_to_end_correct":end_ok})
    assert dict(totals) == {k:doc[k] for k in totals}
    assert {k:dict(v) for k,v in by_task.items()} == doc["by_task"]
    return {"total":dict(totals),"by_task":{k:dict(v) for k,v in by_task.items()},"failures":failures,"style":style}

joint_path = "docs/course-experiments/capstone-evidence/joint/validation.json"
dpo_path = "docs/course-experiments/capstone-evidence/dpo/validation.json"
joint, dpo = rescore(joint_path,"validation"), rescore(dpo_path,"validation")
assert joint["total"]["end_to_end_correct"] == 75 and dpo["total"]["end_to_end_correct"] == 71
assert joint["by_task"]["style"]["end_to_end_correct"] == dpo["by_task"]["style"]["end_to_end_correct"] == 3
assert joint["by_task"]["joint"]["end_to_end_correct"] == 18 and dpo["by_task"]["joint"]["end_to_end_correct"] == 14

train = load("docs/course-experiments/capstone-evidence/dpo/train-report.json")
joint_run = load("docs/course-experiments/results/capstone_joint.json")
dpo_run = load("docs/course-experiments/results/capstone_preference.json")
assert train == dpo_run["results"]
assert train["parent_checkpoint_sha256"] == joint_run["results"]["inference_export"]["sha256"]
assert train["steps"] == train["requested_steps"] == 100 and train["schedule_completed"] and not train["test_evaluated"]
for run, score in [(joint_run,joint),(dpo_run,dpo)]:
    assert run["device"] == "cuda" and run["gpu"] == "NVIDIA L4" and run["seed"] == 42
    assert run["torch_version"] == "2.14.1+cu126"
    assert run["results"]["validation_summary"]["by_task"] == score["by_task"]
    assert run["results"]["data_manifest"] == manifest

# Reconstruct code/seed sample sequence; it is not a saved runtime sample log.
sampler = random.Random(3042)
tasks = {}
for row in splits["train"]:
    tasks.setdefault(row["task"],[]).append(row)
names = sorted(tasks)
pairs = preference_pairs(splits["train"])
effective = 0
first_pairs = last_pairs = None
for step in range(1,101):
    rows = [sampler.choice(tasks[sampler.choice(names)]) for _ in range(24)]
    effective += sum(len(r["answer"].encode("utf-8")) + 1 for r in rows)
    selected = sampler.choices(pairs,k=12)
    selected_ids = [p["row"]["id"] for p in selected]
    if step == 1: first_pairs = selected_ids
    if step == 100: last_pairs = selected_ids
assert effective == train["effective_tokens"] == 41403
assert first_pairs != last_pairs
history_checks = []
for record in train["history"]:
    recomputed = record["dpo_loss"] + 0.2*record["ce_before_update"] + 0.01*record["auxiliary_before_update"]
    discrepancy = abs(recomputed-record["loss_before_update"])
    assert discrepancy < 1e-7
    history_checks.append({"step":record["step"],"dpo_loss":record["dpo_loss"],"weighted_objective_discrepancy":discrepancy})

deployment = load("docs/course-experiments/results/capstone_deployment.json")["results"]
student = load("docs/course-experiments/results/capstone_student.json")["results"]
assert deployment["recommended_stage"] == "joint" and deployment["recipe_frozen_before_test"]
ptq = {}
for bits in (4,8):
    name = f"joint-int{bits}"
    record = deployment["public_stage_exports"][name]
    assert record["source_checkpoint_sha256"] == joint_run["results"]["inference_export"]["sha256"]
    ptq[name] = rescore(f"docs/course-experiments/capstone-evidence/deployment/test-joint-ptq{bits}.json","test")["total"]
    assert ptq[name]["end_to_end_correct"] == 78
assert student["teacher_checkpoint_sha256"] == dpo_run["results"]["inference_export"]["sha256"]
assert student["schedule_completed"] and student["recipe_frozen_before_test"]
for branch in student["branches"].values():
    assert branch["steps"] == 350 and branch["schedule_completed"]
    assert branch["teacher_checkpoint_sha256"] == student["teacher_checkpoint_sha256"]
    assert branch["parameters"]["config"]["experts"] == 0 and branch["parameters"]["config"]["width"] == 48

post = load("docs/course-experiments/results/posttraining.json")["results"]
assert post["effective_tokens"] == 0 and post["parameters"]["policy"] == 148
assert len(post["candidate_order"]) == 4 and post["ppo"]["sampled_actions"] == 7680

code_paths = ["tiny_perceptron/capstone.py","tiny_perceptron/alignment.py","tiny_perceptron/data.py","tiny_perceptron/model.py","tiny_perceptron/modern.py","scripts/course_experiments/capstone.py","scripts/course_experiments/capstone_student.py","scripts/course_experiments/capstone_deployment.py","tests/test_capstone.py"]
code_hashes = {p:sha(ROOT/p) for p in code_paths}
training_record_hash_match = {p:code_hashes[p] == h for p,h in train["code_sha256"].items()}
assert all(training_record_hash_match.values())
official_module = ROOT/"outputs/natural-v4/factual-research/19.8/torch-module.py"
installed_module = Path(inspect.getfile(torch.nn.Module))
module_bytes_same = official_module.read_bytes() == installed_module.read_bytes()
print(json.dumps({
    "environment":{"python":sys.version.split()[0],"torch":torch.__version__,"device":"cpu"},
    "method":"Fetch pinned originals; reconstruct data; decode all generated byte IDs; independently check terminal EOS, ordered tool arguments, calculator return, expected action/final payload; compare all stored flags and summaries.",
    "joint_validation":joint,"dpo_validation":dpo,
    "official_training":{"updates":train["steps"],"seed":42,"gpu":"NVIDIA L4","torch":"2.14.1+cu126","data_version":train["data_version"],"parent_joint_sha256":train["parent_checkpoint_sha256"],"dpo_export_sha256":train["inference_export"]["sha256"],"effective_replay_answer_positions":effective,
        "history_recomputed":history_checks,"first_last_pair_ids_reconstructed_different":first_pairs != last_pairs,"first_pair_ids_reconstructed":first_pairs,"last_pair_ids_reconstructed":last_pairs,
        "sample_log_limit":"IDs above reconstructed from source/seed; runtime batches not recorded; logged history losses are on different training draws, not fixed held-out preferences.",
        "reference_limit":"No full reference before/after runtime tensor fingerprints in inspected GPU reports; current freeze code and own CPU one-update are separate mechanism evidence."},
    "deployment":{"recommended_stage":"joint","recorded_basis":deployment["recommendation_basis"],"ptq_sources":{k:deployment["public_stage_exports"][k]["source_checkpoint_sha256"] for k in ptq},"independently_rescored_ptq":ptq},
    "student":{"teacher_stage":"dpo","teacher_sha256":student["teacher_checkpoint_sha256"],"branches_completed":list(student["branches"]),"updates_per_branch":350,"dense_width":48,"parameters":79920},
    "finite_ppo_scope":{"scope":post["scope"],"label_source":post["label_source"],"candidate_source":post["candidate_source"],"effective_tokens":0,"policy_parameters":148,"candidate_count":4},
    "current_code_sha256":code_hashes,"training_record_code_matches":training_record_hash_match,
    "official_torch_module_sha256":sha(official_module),"installed_torch_module_sha256":sha(installed_module),"official_module_matches_installed_bytes":module_bytes_same,
    "limits":"Official GPU records audited/recomputed only. No model weights loaded, new data downloaded, GPU training/benchmark or historical event replay. Frozen-before-test chronology supported by recorded metadata plus explicit source order, not independent observation of original run.",
},ensure_ascii=False,indent=2))
