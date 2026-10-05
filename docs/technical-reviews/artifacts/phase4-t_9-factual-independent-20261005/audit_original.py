import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source = ROOT / "docs/course-experiments/results/quantization.json"
dataset = ROOT / "outputs/technical-checks/fact_v2_17_14/dependencies/sft/dataset.json"
result = json.loads(source.read_bytes())
assert digest(dataset) == result["results"]["data"]["sha256"]
original = BASE / "original"
original.mkdir(exist_ok=True)
provenance = {}
for src, filename in [(source, "quantization.json"), (dataset, "sft-dataset.json")]:
    target = original / filename
    shutil.copyfile(src, target)
    assert digest(src) == digest(target)
    provenance[filename] = {"source": str(src.relative_to(ROOT)), "source_sha256": digest(src), "copy": str(target.relative_to(ROOT)), "copy_sha256": digest(target), "scope": "Complete original bytes preserved; only named measurement, sample and method/provenance pointers inspected; no notes, limitations or author correction values read"}

parts = json.loads(dataset.read_bytes())
tok = ByteTokenizer()
audit = {"source_sha256": digest(source), "dataset_sha256": digest(dataset), "original_revision": result["revision"], "source_device": result["device"], "source_torch": result["torch_version"], "source_python": result["python_version"], "source_scope": "Audit of existing original measurements, not re-training or new model scoring", "runs": {}}
for variant in ["fp32", "packed4", "packed8"]:
    run = result["results"]["runs"][variant]
    store = run["storage"]
    assert store["tensor_bytes"] == store["parameter_tensor_bytes"] + store["buffer_tensor_bytes"]
    assert store["buffer_tensor_bytes"] == sum(store["buffers"].values())
    audit["runs"][variant] = {"tensor_bytes": store["tensor_bytes"], "file_bytes": store["file_bytes"], "measurements": {}}
    for split in ["validation", "test"]:
        measurement = run[split]
        samples = measurement["generated_samples"]
        assert len(samples) == len(parts[split]) == measurement["examples"]
        exact_count = 0
        eos_count = 0
        for record, sample in zip(parts[split], samples):
            messages = record["messages"]
            assert sample["question"] == messages[-2]["content"]
            assert sample["expected"] == messages[-1]["content"]
            assert sample["family"] == record["family"]
            ids = sample["generated_ids"]
            raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            hit = raw == tok.encode(sample["expected"])
            assert hit == sample["exact"]
            assert sample["generated"] == tok.decode(raw)
            assert sample["ended_with_eos"] == (tok.eos_id in ids)
            exact_count += hit
            eos_count += tok.eos_id in ids
        assert exact_count == measurement["correct"]
        assert eos_count == measurement["eos_count"]
        assert measurement["max_new_tokens"] == 24
        audit["runs"][variant]["measurements"][split] = {"sample_count": len(samples), "source_example_count": measurement["examples"], "original_exact_count_recomputed": exact_count, "original_eos_count_recomputed": eos_count, "max_new_tokens": measurement["max_new_tokens"], "all_question_expected_family_and_id_contracts_match": True}
training = result["results"]["training"]
assert training["steps"] == training["optimizer_updates"] == training["loss_trace"][-1]["step"] == 120
assert training["initialization_sha256"] != training["final_sha256"]
audit["training_provenance"] = {k: training[k] for k in ["steps", "optimizer_updates", "batch_size", "training_examples", "effective_supervised_tokens", "initialization_sha256", "final_sha256"]}
old = ast.parse((BASE / "original-run-compression.py").read_bytes())
new = ast.parse((ROOT / "scripts/course_experiments/compression.py").read_bytes())
older = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
newer = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
audit["necessary_original_current_leaf_ast_equal"] = {}
for name in ["_teacher", "_steps", "_dataset", "_storage", "_packed", "_evaluate", "_fit_text", "run_quantization"]:
    same = ast.dump(older[name], include_attributes=False) == ast.dump(newer[name], include_attributes=False)
    assert same
    audit["necessary_original_current_leaf_ast_equal"][name] = same
assert digest(BASE / "original-run-compression.py") == result["code_sha256"]["scripts/course_experiments/compression.py"]
audit["original_compression_sha256_verified"] = digest(BASE / "original-run-compression.py")
audit["read_pointers"] = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/results/data/{source,sha256,counts,families,family_intersections}", "/results/teacher_provenance/{checkpoint,sha256,format_version,steps}", "/results/training/{steps,optimizer_updates,batch_size,training_examples,effective_supervised_tokens,initialization_sha256,final_sha256,loss_trace/*/step}", "/results/runs/{fp32,packed4,packed8}/storage/{checkpoint,file_bytes,tensor_bytes,parameter_tensor_bytes,buffer_tensor_bytes,buffers,optimizer_in_deployment_file}", "/results/runs/{fp32,packed4,packed8}/{validation,test}/{examples,correct,eos_count,max_new_tokens,generated_samples/*/{family,question,expected,generated,generated_ids,exact,ended_with_eos}}", "/code_sha256/{scripts/course_experiments/compression.py,tiny_perceptron/quantization.py,tiny_perceptron/training.py,tiny_perceptron/model.py}", "/artifacts/*/{path,bytes,sha256}", "sft-dataset.json /{train,validation,test}/*/{messages,family}"]
(BASE / "original-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
(BASE / "original-copy-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))
