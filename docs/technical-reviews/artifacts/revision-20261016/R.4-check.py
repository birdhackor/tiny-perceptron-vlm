"""R.4 scoped deterministic checks; no training, weight updates, or quality evaluation."""

import hashlib
import json
import platform
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices  # noqa: E402
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    source = ROOT / "course/README.md"
    raw = source.read_bytes().decode("utf-8")
    body = lesson_slices(raw)["R.4"]
    freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-08.json").read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "course")
    snapshot = (ROOT / page["snapshot"]).read_bytes()
    assert snapshot == source.read_bytes()
    assert lesson_slices(snapshot.decode("utf-8"))["R.4"] == body
    results = {
        "reviewer_task": "/root/tech_20261016_r_4",
        "executed_at": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/R.4-check.py",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "threads": "1",
        },
        "scope": "Current implementation forward/generation and archived result arithmetic only; no training or optimizer step; no performance or quality rerun.",
        "section_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "frozen_section_matches": True,
        "links": [],
        "model_checks": {},
    }
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", body):
        path_text, _, fragment = target.partition("#")
        path = (source.parent / path_text).resolve()
        assert path.is_file(), target
        if fragment:
            assert fragment in lesson_slices(path.read_bytes().decode("utf-8")), target
        results["links"].append({"target": target, "exists": True, "numbered_anchor_exists": bool(fragment)})

    historical_path = ROOT / "docs/selftrained/results/v2-final-public-results.json"
    historical = json.loads(historical_path.read_bytes())
    for name, archived in historical["architectures"].items():
        torch.manual_seed(20261016)
        config = SelftrainedConfig(**archived["capacity"]["config"])
        model = LimitedAssistant(config).eval()
        assert model.description()["parameters"] == archived["capacity"]["parameters"]
        ffn_kind = type(model.lm.blocks[0].ffn).__name__
        assert ffn_kind == ("PaddingSafeMoE" if name == "moe" else "DenseFFN")
        with torch.no_grad():
            text_ids = torch.tensor([[1, 3, 11, 4, 12]], dtype=torch.long)
            text = model(text_ids)
            assert tuple(text["logits"].shape) == (1, 5, config.vocab_size)
            assert torch.isfinite(text["logits"]).all()
            payloads = [
                {"kind": "image", "values": torch.zeros(2, 1, 32, 32),
                 "coordinates": torch.tensor([[0.25, 0.5], [0.75, 0.5]])},
                {"kind": "ocr", "values": torch.zeros(1, 32, 128)},
                {"kind": "audio", "values": torch.zeros(64, 40)},
            ]
            ids = torch.tensor([[1, 3] + [7] * 2 + [8] * 32 + [9] * 16 + [4, 11]], dtype=torch.long)
            embedded, perception = model._embeddings(ids, None, [payloads])
            features, _ = model.vision_encoder(payloads[0]["values"], payloads[0]["coordinates"])
            assert torch.equal(embedded[0, 2:4], features)
            features, _ = model.ocr_encoder(payloads[1]["values"])
            assert torch.equal(embedded[0, 4:36], features)
            features, _ = model.audio_encoder(payloads[2]["values"])
            assert torch.equal(embedded[0, 36:52], features)
            output = model(ids, modalities=[payloads])
            assert tuple(output["logits"].shape) == (1, 54, config.vocab_size)
            assert [p["kind"] for p in output["perception"]] == ["image", "ocr", "audio"]
            assert torch.isfinite(output["logits"]).all()
            generated = model.generate(text_ids, max_new_tokens=2, eos_id=2)
            assert 1 <= generated.shape[1] <= 2
            assert not any(t in {0, 1, 3, 4, 5, 6, 7, 8, 9} for t in generated.flatten().tolist())
        results["model_checks"][name] = {
            "config": archived["capacity"]["config"],
            "parameters_match_historical_config": True,
            "ffn_kind": ffn_kind,
            "text_logits_shape": list(text["logits"].shape),
            "multimodal_logits_shape": list(output["logits"].shape),
            "perception_logit_shapes": {p["kind"]: list(p["logits"].shape) for p in perception},
            "encoder_vectors_equal_reserved_lm_input_slots": True,
            "generated_ids_random_untrained_no_quality_inference": generated.flatten().tolist(),
            "pretrained_weights_loaded": False,
            "gradient_or_optimizer_steps": 0,
        }

    results["historical_results"] = {"path": historical_path.relative_to(ROOT).as_posix(),
                                       "sha256": digest(historical_path), "architectures": {}}
    for name, archived in historical["architectures"].items():
        final = archived["final_test"]
        counts = {task: score["count"] for task, score in final["per_task_final_reply"].items()}
        assert sum(counts.values()) == final["count"] == 3734
        tool = final["per_task_final_reply"]["tool_call"]["tool_roundtrip"]
        assert tool["denominator"] == 276
        assert tool["rate"] == tool["numerator"] / tool["denominator"]
        assert tool["rate"] < final["thresholds"]["tool_roundtrip"]
        assert final["evaluation_complete"] and not archived["capability_acceptance_passed"]
        results["historical_results"]["architectures"][name] = {
            "training_revision": archived["training_revision"],
            "selected_source": archived["selected_source"],
            "final_run_id": final["run_id"], "final_revision": final["revision"],
            "total_test_rows": final["count"], "sum_per_task_rows": sum(counts.values()),
            "task_denominators": counts, "evaluation_complete": final["evaluation_complete"],
            "tool_roundtrip": tool, "original_tool_threshold": final["thresholds"]["tool_roundtrip"],
            "capability_acceptance_passed": archived["capability_acceptance_passed"],
            "historical_measurement_not_rerun": True,
        }
    results["historical_results"]["data_scope"] = historical["data_scope"]

    repo, revision = historical["hf_repo"], historical["hf_revision"]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}/selftrained/v2?recursive=true&expand=false"
    request = urllib.request.Request(url, headers={"User-Agent": "technical-review-R.4/1.0"})
    with urllib.request.urlopen(request, timeout=35) as response:
        remote_raw = response.read()
        http_status = response.status
    tree = json.loads(remote_raw)
    public_files = {}
    for name, archived in historical["architectures"].items():
        prefix = f"selftrained/v2/{name}-joint/"
        files = [x for x in tree if x["type"] == "file" and x["path"].startswith(prefix)]
        assert {Path(x["path"]).name for x in files} == {
            "model.safetensors", "model-config.json", "tokenizer.json", "inference-manifest.json"}
        weight = next(x for x in files if x["path"].endswith("/model.safetensors"))
        assert weight["lfs"]["oid"] == archived["safe_weights"]["sha256"]
        assert weight["size"] == archived["safe_weights"]["bytes"]
        public_files[name] = files
    results["public_release_metadata"] = {
        "url": url, "http_status": http_status, "anonymous_request": True,
        "response_sha256": hashlib.sha256(remote_raw).hexdigest(), "files": public_files,
        "scope": "Unauthenticated pinned repository tree lists inference payloads; weights were not downloaded or re-evaluated.",
    }
    license_text = (ROOT / "LICENSE").read_text()
    assert license_text.startswith("MIT License")
    assert 'license = "MIT"' in (ROOT / "pyproject.toml").read_text()
    notice = (ROOT / "docs/selftrained/V2-DATA-NOTICE.md").read_text()
    for text in ["MIT notice", "SIL Open Font License", "CC BY 4.0"]:
        assert text in notice
    results["license_declarations"] = {
        "project": "MIT", "v2_data_not_uniform_MIT": ["MIT", "SIL OFL", "CC BY 4.0"],
        "scope": "Checks the project's licensing declarations and separate source notices; no new legal opinion or comprehensive third-party compliance audit.",
    }
    observed_files = [
        "course/README.md", "course/chapters/19.md", "tiny_perceptron/model.py",
        "tiny_perceptron/selftrained/model.py", "scripts/selftrained/train.py",
        "docs/selftrained/results/v2-final-public-results.json", "docs/selftrained/v2-manifest.json",
        "docs/selftrained/v2-training-stage-index.json",
        "docs/selftrained/results/v2-final-training-source-selection.json",
        "docs/selftrained/V2-DATA-NOTICE.md", "LICENSE", "pyproject.toml",
    ]
    results["inspected_repository_hashes"] = {p: digest(ROOT / p) for p in observed_files}
    out = Path(__file__).with_name("R.4-execution.json")
    assert not out.exists(), "Preserve original execution evidence; use another filename for a rerun."
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(out.relative_to(ROOT)), "all_assertions_passed": True,
                      "models_checked": list(results["model_checks"]), "final_test_rows_per_model": 3734,
                      "public_metadata_http": http_status, "training_steps_executed": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
