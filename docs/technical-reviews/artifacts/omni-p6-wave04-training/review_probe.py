"""Reviewer-owned CPU contract probes; never run training, formal final, or teacher inference."""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).with_name("execution.json")
FROZEN = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18/sources/training.md"
env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
result = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": "/root/omni_p6_technical_training", "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform()}, "scope": "Only CLI help, one-batch dry runs, generated-data contracts, and untrained tiny checkpoint/quantization. No long recipe, formal training, final or teacher inference.", "commands": []}

def run(argv, cwd=ROOT):
    proc = subprocess.run([sys.executable, *argv], cwd=cwd, env=env, text=True, capture_output=True, timeout=45)
    row = {"argv": [sys.executable, *argv], "cwd": str(cwd), "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    result["commands"].append(row)
    if proc.returncode:
        raise RuntimeError(row)
    return proc.stdout

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

scripts = ["train.py", "train_simple.py", "prepare_data.py", "fetch_training_assets.py", "fetch_course_models.py", "evaluate.py", "infer.py", "pretrain_encoders.py", "infer_modal.py", "evaluate_modal.py", "quantize.py"]
for name in scripts:
    run(["scripts/" + name, "--help"])
for module in ["scripts.course_experiments.run", "scripts.omni_revision.public_download", "scripts.omni_revision.public_demo"]:
    run(["-m", module, "--help"])
run(["scripts/fetch_training_assets.py", "--list"])
run(["scripts/fetch_course_models.py", "--list"])
for task in ["text", "sft", "vision"]:
    run(["scripts/train.py", "--task", task, "--device", "cpu"])
for model in ["bigram", "mlp"]:
    run(["scripts/train_simple.py", "--model", model, "--context", "3", "--width", "16", "--seed", "42", "--device", "cpu"])

with tempfile.TemporaryDirectory(prefix="training-review-", dir="/workspace/work/reference") as folder:
    tmp = Path(folder)
    for name in ["outputs", "checkpoints"]:
        (tmp / name).mkdir()
    result["generated_data"] = {}
    for kind in ["toy-text", "attributes-sft", "style", "safety", "preference"]:
        run([str(ROOT / "scripts/prepare_data.py"), "--kind", kind, "--seed", "42", "--output", str(tmp / "data")])
        parts = {split: [json.loads(s) for s in (tmp / "data" / kind / (split + ".jsonl")).read_text().splitlines()] for split in ["train", "validation", "test"]}
        result["generated_data"][kind] = {"splits": {split: len(rows) for split, rows in parts.items()}, "family_overlap": {a + "/" + b: sorted(set(r["family"] for r in parts[a]) & set(r["family"] for r in parts[b])) for a,b in [("train","validation"),("train","test"),("validation","test")]}, "first_training_record": parts["train"][0]}
        if kind == "attributes-sft":
            result["generated_data"][kind]["blue_square_low_train"] = [r for r in parts["train"] if r["family"] == "blue:square:low"]
    body = FROZEN.read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", body, re.S)
    assert len(blocks) == 2
    (tmp / "outputs/save_start.py").write_text(blocks[0] + "\n")
    (tmp / "outputs/size_check.py").write_text(blocks[1] + "\n")
    run(["outputs/save_start.py"], cwd=tmp)
    # The page's checkpoint name is retained, but this is the untrained start.
    (tmp / "checkpoints/attributes.pt").write_bytes((tmp / "checkpoints/start.pt").read_bytes())
    for bits in [4, 8]:
        run([str(ROOT / "scripts/quantize.py"), "checkpoints/attributes.pt", "--bits", str(bits), "--output", "checkpoints/attributes-int" + str(bits) + ".pt"], cwd=tmp)
    run(["outputs/size_check.py"], cwd=tmp)
    import torch
    from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
    from tiny_perceptron.tokenization import generation_report
    result["environment"]["torch"] = str(torch.__version__)
    result["inline_code"] = {"start_python_sha256": hashlib.sha256(blocks[0].encode()).hexdigest(), "size_python_sha256": hashlib.sha256(blocks[1].encode()).hexdigest(), "untrained_start_optimizer": torch.load(tmp / "checkpoints/start.pt", map_location="cpu", weights_only=True)["optimizer"], "warning": "Untrained tiny checkpoint contract only; no inference, quality, speed, or formal outcome is claimed."}
    tok = ByteTokenizer()
    x,y = render_chat([{"role":"user","content":"q"},{"role":"assistant","content":"circle"}])
    result["mechanisms"] = {"A_ids": tok.encode("A"), "EOS": tok.eos_id, "Chinese_紅_token_count": len(tok.encode("紅")), "handwritten_generation": generation_report(tok, [73,2]), "circle_answer_targets": y[y != -100].tolist(), "document_splits_seed42": {k:len(v) for k,v in split_documents(toy_documents(),42).items()}, "arithmetic": {"2+3":2+3,"storage_saved":80000-60000,"remaining_matches_delta":3-4}}
result["frozen_sha256"] = sha(FROZEN)
DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(DEST.relative_to(ROOT)), "bytes": DEST.stat().st_size, "sha256": sha(DEST), "commands": len(result["commands"]), "environment":result["environment"]}, ensure_ascii=False))
