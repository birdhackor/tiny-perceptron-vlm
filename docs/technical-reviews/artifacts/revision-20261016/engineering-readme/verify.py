"""Independent README audit: bounded CPU, local files and anonymous read-only HF metadata.

No Modal registration, GPU, training (--train), upload, dataset download or course gates.
Raw JSON is read only at explicitly selected measurement/provenance pointers.
"""
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
WORK = ROOT / "work/engineering-readme"
WORK.mkdir(parents=True, exist_ok=True)

import torch
import scripts.train as training
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads((ROOT / path).read_bytes())

def selected(path, pointers):
    data = read(path)
    values = {}
    for pointer in pointers:
        item = data
        for part in pointer.strip("/").split("/"):
            item = item[int(part)] if isinstance(item, list) else item[part]
        values[pointer] = item
    return {"path": path, "sha256": sha(ROOT / path), "pointers": values}

def run(args):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    result = {"argv": [sys.executable, *args], "returncode": p.returncode,
              "stdout": p.stdout, "stderr": p.stderr}
    assert p.returncode == 0, result
    return result

result = {"reviewer_task": "/root/tech_engineering_readme_20261016",
          "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/engineering-readme/verify.py",
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
                          "platform": platform.platform()},
          "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
assert not torch.cuda.is_available()
torch.set_num_threads(1)
readme = ROOT / "README.md"
result["source_sha256"] = sha(readme)
frozen = next(p for p in read("docs/course-revision-20261016/freeze-09.json")["pages"] if p["page_id"] == "readme")
assert frozen["source_sha256"] == result["source_sha256"]
result["freeze_match"] = {k: frozen[k] for k in ("page_id", "source", "selector", "source_sha256", "figures_sha256")}

chapter_files = sorted((ROOT / "course/chapters").glob("*.md"))
front = [ROOT / "course" / x for x in ("README.md", "first-steps.md", "training.md", "glossary.md")]
section_re = re.compile(r"^## ([A-Z\d]+\.\d+) ", re.M)
chapter_ids = [x for p in chapter_files for x in section_re.findall(p.read_text())]
front_ids = [x for p in front for x in section_re.findall(p.read_text())]
notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
assert len(chapter_ids) == len(set(chapter_ids)) == len(notebooks) == 312
assert len(front_ids) == 26 and len(chapter_ids) + len(front_ids) == 338
assert len([p for p in chapter_files if p.stem.isdecimal() and 1 <= int(p.stem) <= 20]) == 20
bootstrap_errors = []
for p in notebooks:
    data = json.loads(p.read_bytes())
    code = [c for c in data["cells"] if c["cell_type"] == "code"]
    if code and not code[0]["metadata"].get("course_setup"):
        bootstrap_errors.append(str(p.relative_to(ROOT)))
assert not bootstrap_errors
result["inventory"] = {"chapter_files": [str(p.relative_to(ROOT)) for p in chapter_files],
                       "lessons": len(chapter_ids), "notebooks": len(notebooks),
                       "front_sections": len(front_ids), "numbered_total": len(chapter_ids)+len(front_ids),
                       "bootstrap_errors": bootstrap_errors}

prep = run(["scripts/prepare_data.py", "--kind", "toy-text", "--output", str(WORK / "generated")])
data_path = WORK / "generated/toy-text/train.jsonl"
command = ["scripts/train.py", "--task", "text", "--data", str(data_path), "--device", "cpu",
           "--output", str(WORK / "must-not-exist.pt")]
dry = run(command)
report = json.loads(dry["stdout"].splitlines()[-1])
assert report["mode"] == "dry-run-no-weight-update" and len(report["history"]) == 1
assert math.isfinite(report["history"][0]["loss"])
assert math.isfinite(report["history"][0]["grad_norm"]) and report["history"][0]["grad_norm"] > 0
assert not (WORK / "must-not-exist.pt").exists()
captured = {}
original = training.TinyLM
def construct(config):
    model = original(config)
    captured["model"] = model
    captured["before"] = {k:v.detach().clone() for k,v in model.state_dict().items()}
    return model
with patch.object(sys, "argv", command), patch.object(training, "TinyLM", construct), \
     patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("unexpected update")) as step, \
     patch.object(training, "save_checkpoint", side_effect=AssertionError("unexpected save")) as save, \
     contextlib.redirect_stdout(io.StringIO()):
    training.main()
assert step.call_count == save.call_count == 0
assert all(torch.equal(v,captured["model"].state_dict()[k]) for k,v in captured["before"].items())
result["minimal_workflow"] = {"prepare": prep, "dry_run": dry, "mode": report["mode"],
                              "loss": report["history"][0]["loss"], "grad_norm": report["history"][0]["grad_norm"],
                              "optimizer_step_calls": step.call_count, "save_checkpoint_calls": save.call_count,
                              "all_parameters_identical": True, "checkpoint_created": False}

manifest = read("docs/selftrained/v2-manifest.json")
result["dataset_manifest"] = {"sha256": sha(ROOT / "docs/selftrained/v2-manifest.json"),
                              "pointers": ["/package", "/records/*/path", "/records/*/sha256", "/assets/*/path"],
                              "package": manifest["package"], "record_files": len(manifest["records"]),
                              "asset_files": len(manifest["assets"]),
                              "total_files": len(manifest["records"]) + len(manifest["assets"])}
assert result["dataset_manifest"]["total_files"] == 8962
assert manifest["package"]["bytes"] == 67862431
assert len({v["path"] for v in manifest["records"]+manifest["assets"]}) == 8962

result["v2_training"] = {}
for arch, folder, expected_steps, expected_params in (("moe","moe-native",4000,5447107), ("dense","dense-weighted",10000,2288067)):
    raw = selected(f"docs/selftrained/results/training-raw/{folder}/raw/train-receipt.json",
                   ["/steps", "/train_records", "/validation_records", "/test_used_for_selection", "/config",
                    "/total_parameters", "/completed_requested_steps", "/selected_checkpoint_available", "/inference_exported"])
    values = raw["pointers"]
    model = LimitedAssistant(SelftrainedConfig(**values["/config"]))
    count = sum(p.numel() for p in model.parameters())
    assert count == values["/total_parameters"] == expected_params
    assert values["/steps"] == expected_steps and values["/train_records"] == 28876 and values["/validation_records"] == 2435
    receipt = selected(f"docs/selftrained/results/training-raw/{folder}/receipt.json",
                       ["/run_id", "/revision", "/status", "/returncode", "/resource_spec"])
    assert receipt["pointers"]["/status"] == "completed" and receipt["pointers"]["/returncode"] == 0
    metrics = selected(f"docs/selftrained/results/public-raw/{arch}/test/metrics.json",
                       ["/count", "/expected_count", "/limited_smoke", "/evaluation_complete", "/interrupted",
                        "/per_task_final_reply/tool_call/tool_roundtrip", "/thresholds/tool_roundtrip",
                        "/per_task_final_reply/voice_qa/semantic", "/per_task_final_reply/ocr/exact"])
    m = metrics["pointers"]
    assert m["/count"] == m["/expected_count"] == 3734 and m["/evaluation_complete"] and not m["/limited_smoke"] and not m["/interrupted"]
    tool = m["/per_task_final_reply/tool_call/tool_roundtrip"]
    assert tool["denominator"] == 276 and tool["rate"] == tool["numerator"]/tool["denominator"]
    assert tool["rate"] < m["/thresholds/tool_roundtrip"]
    test_receipt = selected(f"docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json",
                            ["/status", "/expected_count", "/completed_count", "/outputs_sha256"])
    result["v2_training"][arch] = {"raw_train": raw, "execution": receipt, "constructed_parameters": count,
                                  "final_test": metrics, "evaluation_receipt": test_receipt,
                                  "acceptance_failed_independently": "tool_roundtrip is below 0.95"}

public = read("docs/course-experiments/public-models.json")
capstone = read("docs/course-experiments/capstone-public.json")
result["public_inventory"] = {"original_models": len(public["models"]),
                              "original_files": sum(len(m["files"]) for m in public["models"]),
                              "original_weight_files": sum(sum(f["output"].endswith((".pt",".pth")) for f in m["files"]) for m in public["models"]),
                              "old_capstone_models": len(capstone["models"]),
                              "old_capstone_weight_files": sum(sum(f["output"].endswith((".pt",".pth")) for f in m["files"]) for m in capstone["models"])}
assert result["public_inventory"]["original_models"] == 30
assert result["public_inventory"]["original_weight_files"] == 120
assert result["public_inventory"]["old_capstone_models"] == 11
result["list_commands"] = [run(["scripts/fetch_training_assets.py", "--list"]), run(["scripts/fetch_course_models.py", "--list"])]
result["runner_tests"] = run(["-m", "pytest", "-q", "tests/test_lesson_products_runner.py"])

revision = "979cdfacc588ad0536f1c64fff96f264571cf054"
url = f"https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/{revision}/selftrained/v2?recursive=true&expand=false"
with urllib.request.urlopen(url, timeout=45) as response:
    remote_raw = response.read()
tree = json.loads(remote_raw)
(WORK / "hf-tree.json").write_bytes(remote_raw)
files = [x for x in tree if x["type"] == "file"]
safe = [x for x in files if x["path"].endswith((".safetensors","model-config.json","tokenizer.json","inference-manifest.json"))]
assert len(safe) == 16
result["hf_public_receipt"] = {"url": url, "method": "anonymous HTTPS GET; no credential or upload", "accessed_on": "2026-10-16",
                               "response_sha256": hashlib.sha256(remote_raw).hexdigest(), "response_bytes": len(remote_raw),
                               "revision": revision, "paired_file_count": len(safe),
                               "files": [{k:x[k] for k in ("path","size")} for x in safe]}
result["remote_small_manifests"] = []
for prefix in ("moe-pretrain", "moe-sft", "moe-joint", "dense-joint"):
    u = f"https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/{revision}/selftrained/v2/{prefix}/inference-manifest.json"
    with urllib.request.urlopen(u, timeout=45) as response:
        raw = response.read()
    (WORK / f"{prefix}-inference-manifest.json").write_bytes(raw)
    d = json.loads(raw)
    print(json.dumps({"source":u,"top_level_keys_and_types":{k:type(v).__name__ for k,v in d.items()}}))
    # Explicit provenance fields only. No result commentary or author notes.
    result["remote_small_manifests"].append({"url":u,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),
        "pointers":{k:d[k] for k in ("selected_step","stage","selection","files","selected_checkpoint_sha256")}})
assert result["remote_small_manifests"][2]["pointers"]["selected_step"] == 1000
assert result["remote_small_manifests"][3]["pointers"]["selected_step"] == 1000

text = "貓看狗，狗看貓。"
chars = sorted(set(text))
ids = [chars.index(c) for c in text]
assert ids == [3,2,1,4,1,2,3,0] and "".join(chars[i] for i in ids) == text
result["figure_calculation"] = {"text":text,"chars":chars,"ids":ids,"restored":text}
(OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status":"passed","result_path":str((OUT/'results.json').relative_to(ROOT)),
                  "inventory":result["inventory"],"minimal_workflow":{k:report[k] for k in ('mode','device','history')},
                  "parameters":{a:v['constructed_parameters'] for a,v in result['v2_training'].items()},
                  "runner_tests":result['runner_tests']['stdout'],"hf_paired_files":len(safe)},ensure_ascii=False))
