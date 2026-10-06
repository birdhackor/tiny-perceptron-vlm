"""Same original T.3 reviewer: actual current delta/contract check, no training."""

import ast
import difflib
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / "docs/technical-reviews/artifacts/phase4-t_3-factual-independent-20261005-fresh"
TASK = "/root/phase4_factual_coordinator/factual_t_3"
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import BigramLM, ContextMLP

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


sp = importlib.util.spec_from_file_location("own_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(sp)
sp.loader.exec_module(facts)
body, _, first_line = facts.original_section(ROOT / "course/training.md", "T.3")
assert hashlib.sha256(body).hexdigest() == "3a68e19f115690dee0251804dc288d04d6828be2705559ddb6cdb03490852cfa"
assert (OUT / "current-section.md").read_bytes() == body
prior_body = (PRIOR / "section.md").read_bytes()
before = b"\xe8\xa6\x81\xe9\x87\x8d\xe5\x81\x9a\xe6\x9c\xac\xe7\xaf\x80\xe5\x9b\x9b\xe7\xa8\xae\xe5\x9b\xba\xe5\xae\x9a\xe6\xa8\xa1\xe5\x9e\x8b\xe6\xaf\x94\xe8\xbc\x83\xef\xbc\x8c\xe7\x94\xa8\xef\xbc\x9a"
after = "固定比較包含兩類模型、四個設定：接字表，以及一字、三字、五字視窗的 MLP。視窗取幾個位置的差別見[2.5](chapters/02.md#2.5)；要重做這組比較，用：".encode()
assert prior_body.count(before) == 1
assert prior_body.replace(before, after) == body
prior_fences = facts.fences(prior_body, first_line)
current_fences = facts.fences(body, first_line)
assert len(prior_fences) == len(current_fences) == 6
fence_rows = []
for index, (old, new) in enumerate(zip(prior_fences, current_fences, strict=True), 1):
    assert old["raw"] == new["raw"] and old["language"] == new["language"] and new["closed"]
    fence_rows.append({"number": index, "language": new["language"], "sha256": hashlib.sha256(new["raw"]).hexdigest(),
                       "unchanged": True})
assert b"![" not in body and b"<img" not in body and b"<svg" not in body

# Prior source/artifact fingerprints are checked, not prior adjudications adopted.
prior_report = json.loads((OUT / "prior-T.3.opaque.json").read_text())
assert prior_report["reviewer_task"] == TASK
assert digest(OUT / "prior-T.3.opaque.json") == "71fb7faf4a3205db9ae34d2bd9494aaf7862cff318615e08c664ba0513761d07"
reuse = []
for source in prior_report["sources"]:
    if source["kind"] == "repository_code":
        assert digest(ROOT / source["path"]) == source["sha256"], source["path"]
        reuse.append({"kind": "repository_source", "id": source["id"], "path": source["path"],
                      "sha256": source["sha256"], "same": True,
                      "support_scope": source["inspection_note"]})
for artifact in prior_report["artifacts"]:
    assert digest(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]
    reuse.append({"kind": "permanent_prior_artifact", "id": artifact["id"], "path": artifact["path"],
                  "sha256": artifact["sha256"], "same": True, "support_scope": artifact["description"]})

# Inspect the original calculation/constructor AST; skip returned author scope prose.
path = ROOT / "scripts/course_experiments/text.py"
text = path.read_text()
tree = ast.parse(text)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_simple_models")
loop = next(n for n in function.body if isinstance(n, ast.For))
configs = ast.literal_eval(loop.iter)
assert configs == (("bigram", 1), ("mlp1", 1), ("mlp3", 3), ("mlp5", 5))
constructor = next(n for n in loop.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in n.targets))
parts = split_documents(toy_documents(), 42)
vocabulary = {c: i + 2 for i, c in enumerate(sorted(set("".join(parts["train"]))))}
vocab_size = len(vocabulary) + 2
models = []
for name, context in configs:
    model = BigramLM(vocab_size) if name == "bigram" else ContextMLP(vocab_size, context, 16)
    models.append({"setting": name, "context": context, "class": type(model).__name__,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "scope": "Only original constructor instantiated on CPU; no backward/step/training/scoring/save."})
assert len(models) == 4
assert {n["class"] for n in models} == {"BigramLM", "ContextMLP"}
assert [n["context"] for n in models if n["class"] == "ContextMLP"] == [1, 3, 5]

# Raw JSON key/type shape precedes exact named context leaves; no score/notes read.
raw_path = ROOT / "docs/course-experiments/results/simple_models.json"
raw_result = json.loads(raw_path.read_text())
top_shape = {k: type(v).__name__ for k, v in raw_result.items()}
run_shape = {k: type(v).__name__ for k, v in raw_result["results"]["runs"].items()}
raw_pointers = {}
for name, context in configs:
    pointer = "/results/runs/" + name + "/context"
    value = raw_result["results"]["runs"][name]["context"]
    assert value == context
    raw_pointers[pointer] = value
assert set(run_shape) == {n["setting"] for n in models}

# The newly linked 2.5 explains how many recent character positions are selected.
context_body, _, context_line = facts.original_section(ROOT / "course/chapters/02.md", "2.5")
assert context_body == (OUT / "necessary-context-2_5.md").read_bytes()
context_fence = facts.fences(context_body, context_line)[0]
assert context_fence["language"] == "python"
code_path = OUT / "necessary-context-2_5-original-fence.py"
code_path.write_bytes(context_fence["raw"])
namespace = {"__name__": "__main__"}
exec(compile(context_fence["raw"], "course/chapters/02.md#2.5:first_original_fence", "exec"), namespace)
examples = namespace["examples"]
windows = {str(length): [(prefix[-length:], answer) for prefix, answer in examples] for length in [1, 3, 4, 5]}
assert windows["1"] == [("是", "圓"), ("是", "方")]
assert windows["3"] == [("物體是", "圓"), ("物體是", "方")]
assert windows["4"] == [("色物體是", "圓"), ("色物體是", "方")]
assert windows["5"] == [("紅色物體是", "圓"), ("藍色物體是", "方")]
contexts = [{"source": "course/chapters/02.md#2.5", "raw_section_sha256": hashlib.sha256(context_body).hexdigest(),
             "scope": "New link: definitions and recent-position slicing only. The already visible historical table and figures were not re-adjudicated as part of T.3."}]
for source, lesson, prior_name in [("course/chapters/05.md", "5.7", "5_7.md"), ("course/chapters/07.md", "7.9", "7_9.md")]:
    context_body, _, _ = facts.original_section(ROOT / source, lesson)
    assert context_body == (PRIOR / prior_name).read_bytes()
    contexts.append({"source": source + "#" + lesson, "raw_section_sha256": hashlib.sha256(context_body).hexdigest(),
                     "same_as_prior_own_context": True, "scope": "Unchanged linked save/resume or EOS/limit explanation; original support scope retained."})

environment = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
               "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
               "cuda_available": str(torch.cuda.is_available()), "device": "cpu"}
assert environment["torch"] == "2.14.1+cpu"
assert environment["torch_git_version"] == "5c4886908584029761b579af026dcfb627c84070"
result = {"kind": "current_original_reviewer_callback_inspection", "reviewer_task": TASK, "reviewed_on": "2026-10-06",
          "source": "course/training.md#T.3", "source_sha256": hashlib.sha256(body).hexdigest(),
          "prior_opaque_file": (OUT / "prior-T.3.opaque.json").relative_to(ROOT).as_posix(),
          "prior_opaque_sha256": digest(OUT / "prior-T.3.opaque.json"), "intro": None, "figure_sha256": {},
          "actual_read_scope": "Personally read all current T.3 and diff against own raw frozen section. No other section substituted for this adjudication.",
          "delta": "One paragraph: four fixed models becomes two model classes and four settings; newly links2.5. No command, numeric example, model API or figure changed.",
          "unchanged_fences": fence_rows, "exact_dependency_reuse": reuse, "necessary_context_versions": contexts,
          "new_paragraph_check": {"original_method_path": "scripts/course_experiments/text.py",
                                  "original_method_sha256": digest(path), "tuple_line": loop.iter.lineno,
                                  "tuple_expression": ast.get_source_segment(text, loop.iter),
                                  "constructor_line": constructor.lineno, "constructor_expression": ast.get_source_segment(text, constructor),
                                  "actual_cpu_constructor_results": models,
                                  "original_raw_result_sha256": digest(raw_path), "top_key_types_first": top_shape,
                                  "named_run_key_types": run_shape, "raw_context_pointers": raw_pointers,
                                  "new_link_window_position_examples": windows},
          "environment": environment, "result": "all_current_delta_assertions_passed",
          "limitations": "Original 200-step recipes, downloads, full experiment, published model generation/ability scores, uploads and GPU work not run. No new weights saved. Unchanged real prior evidence reused after exact hash/support check; no source refetch or unrelated CPU rerun."}
(OUT / "current-inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(OUT / "current-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print(json.dumps({"result": result["result"], "source_sha256": result["source_sha256"], "settings": models,
                  "dependency_reuse_count": len(reuse), "unchanged_fences": len(fence_rows),
                  "new_context_sha256": contexts[0]["raw_section_sha256"], "new_saved_weights": False}, ensure_ascii=False))
