"""R.4 bounded CPU/read-only evidence check; no training or formal evaluation."""

import hashlib
import json
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch

from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.data import render
from tiny_perceptron.lesson_products.inference import Product, auto_solve
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    torch.set_num_threads(2)
    output = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/R.4-check.py",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "2"},
        "scope": "Reviewer personally ran metadata/hash checks, two finite B/C calls, Auto and tiny random-config forward passes. Existing final GPU tests/training were only inspected, never rerun.",
    }
    freeze = read("docs/course-revision-20261016/freeze-14.json")
    page = next(x for x in freeze["pages"] if x["page_id"] == "course")
    text = (ROOT / page["snapshot"]).read_text(encoding="utf-8")
    body = lesson_slices(text)["R.4"]
    assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
    output["frozen_source"] = {**page, "section_sha256": hashlib.sha256(body.encode()).hexdigest()}
    output["route"] = []
    for lesson in ("19.2", "19.6", "19.11", "19.12", "19.13", "19.17", "19.19", "19.20", "20.1"):
        p = next(x for x in freeze["pages"] if x["page_id"] == lesson)
        t = (ROOT / p["snapshot"]).read_text()
        assert sha(ROOT / p["snapshot"]) == p["source_sha256"]
        output["route"].append({"lesson": lesson, "snapshot": p["snapshot"], "sha256": p["source_sha256"], "heading": t.splitlines()[0]})
    final = read("docs/selftrained/results/v2-final-public-results.json")
    output["v2"] = {"hf_repo": final["hf_repo"], "hf_revision": final["hf_revision"], "scope": final["scope"], "data_scope": final["data_scope"], "architectures": {}}
    for arch, info in final["architectures"].items():
        prepath = f"docs/selftrained/results/training-raw/{arch}-pretrain/raw/train-receipt.json"
        journalpath = f"docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json"
        runnerpath = f"docs/selftrained/results/public-raw/{arch}/test/receipt.json"
        pre, journal, runner = read(prepath), read(journalpath), read(runnerpath)
        assert pre["origin"]["kind"] == "all-neural-weights-random" and pre["completed_requested_steps"]
        assert journal["status"] == "complete" and journal["completed_count"] == info["final_test"]["count"] == 3734
        assert runner["status"] == "completed" and runner["returncode"] == 0
        assert runner["job"]["public_export"]["revision"] == final["hf_revision"]
        tasks = info["final_test"]["per_task_final_reply"]
        failures = {"tool_roundtrip": tasks["tool_call"]["tool_roundtrip"], "voice_qa": tasks["voice_qa"]["semantic"], "ocr_exact": tasks["ocr"]["exact"]}
        assert failures["tool_roundtrip"]["rate"] < info["final_test"]["thresholds"]["tool_roundtrip"]
        assert info["capability_acceptance_passed"] is False
        output["v2"]["architectures"][arch] = {"origin": pre["origin"], "pretrain_steps": pre["steps"], "test_completed": journal["completed_count"], "checkpoint_sha256": journal["checkpoint_sha256"], "failures": failures, "thresholds": info["final_test"]["thresholds"], "capability_acceptance_passed": False, "sources": {p: sha(ROOT / p) for p in (prepath, journalpath, runnerpath)}}
    output["public_metadata"] = []
    # Anonymous GETs of small configurations and HEADs only: no weight downloads.
    for arch, info in final["architectures"].items():
        prefix = f"https://huggingface.co/{final['hf_repo']}/resolve/{final['hf_revision']}/selftrained/v2/{arch}-joint/"
        with urllib.request.urlopen(prefix + "model-config.json", timeout=30) as response:
            data = response.read()
        config = json.loads(data)
        assert config == info["capacity"]["config"]
        with urllib.request.urlopen(urllib.request.Request(prefix + "model.safetensors", method="HEAD"), timeout=30) as response:
            status = response.status
        assert status == 200
        output["public_metadata"].append({"architecture": arch, "config_url": prefix + "model-config.json", "config_sha256": hashlib.sha256(data).hexdigest(), "config": config, "weight_head_status": status})
    b = Product(ROOT / "outputs/lesson-products/products/b", "b")
    c = Product(ROOT / "outputs/lesson-products/products/c", "c")
    world = {"task": "boxes", "a": 2, "b": 5, "events": [("A", 1), ("swap", 0)], "query": "B"}
    state = render(world)
    judged = c.judge(state)
    assert set(judged["probabilities"]) == {"card", "address", "app", "boxes", "sum", "none"}
    assert abs(sum(judged["probabilities"].values()) - 1) < 1e-6
    output["products"] = {"state": state, "b": b.solve(state, 3), "c": judged, "auto": auto_solve(b, c, state), "weight_sha256": {p: sha(ROOT / f"outputs/lesson-products/products/{p}/model.pt") for p in ("b", "c")}}
    output["random_config_forward"] = {}
    with torch.inference_mode():
        for arch in ("moe", "dense"):
            torch.manual_seed(14)
            model = LimitedAssistant(SelftrainedConfig(vocab_size=32, architecture=arch, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32, max_length=64)).eval()
            out = model(torch.tensor([[1, 11, 12]], dtype=torch.long))
            image, _ = model.vision_encoder(torch.zeros(2, 1, 32, 32), torch.zeros(2, 2))
            audio, _ = model.audio_encoder(torch.zeros(20, 40))
            assert out["logits"].shape == (1, 3, 32) and image.shape == (2, 16) and audio.shape == (16, 16)
            output["random_config_forward"][arch] = {"logits_shape": list(out["logits"].shape), "image_tokens_shape": list(image.shape), "audio_tokens_shape": list(audio.shape), "ffn_class": type(model.lm.blocks[0].ffn).__name__, "trained": False}
    target = ROOT / "docs/technical-reviews/artifacts/revision-20261016/backfill/R.4-execution.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "artifact": str(target.relative_to(ROOT)), "sha256": sha(target), "environment": output["environment"], "b_answer": output["products"]["b"]["answer"], "c_selected": judged["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
