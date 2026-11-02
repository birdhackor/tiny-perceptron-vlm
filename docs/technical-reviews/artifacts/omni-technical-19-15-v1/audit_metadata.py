"""Small receipt of source pins, historical training records, figures and a CPU CLI call."""

import hashlib
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = {"created_at": datetime.now(UTC).isoformat(), "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}}
    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
    freeze = json.loads(freeze_path.read_bytes())
    result["freeze"] = {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path)}
    pages = []
    for page_id in ("19.15", "14.12", "14.15", "4.5", "19.13", "19.18"):
        page = next(p for p in freeze["pages"] if p["page_id"] == page_id)
        snapshot = (ROOT / page["snapshot"]).read_bytes()
        live = (ROOT / page["source"]).read_text()
        section = re.search(r"^## " + re.escape(page_id) + r"\b.*?(?=^## |\Z)", live, re.M | re.S)[0].encode()
        assert snapshot == section
        assert hashlib.sha256(snapshot).hexdigest() == page["source_sha256"]
        pages.append({k: page[k] for k in ("page_id", "source", "selector", "snapshot", "source_sha256", "figures_sha256")})
    result["pinned_slices"] = pages

    records = {}
    for name in ("initial-b-execution.json", "mechanism-fixed3-runner-result.json", "mechanism-odd135-runner-result.json"):
        path = ROOT / "docs/lesson-products/results" / name
        full = json.loads(path.read_bytes())
        run = full.get("execution", full)
        training = run["training"]
        keys = ("product", "mode", "random_initialization", "resume_source", "steps_this_attempt", "last_step", "selected_step", "seed", "batch_size", "parameters", "depth_weights", "initial_weights_sha256", "selected_weights_sha256", "data_manifest_sha256", "resume_verification", "environment", "work")
        checks = {}
        for source in ("model.py", "training.py", "train.py", "data.py"):
            relative = "tiny_perceptron/lesson_products/" + source
            expected = run["source_hashes"][relative]
            blob = subprocess.run(["git", "show", run["code_revision"] + ":" + relative], cwd=ROOT, capture_output=True, check=True).stdout
            assert hashlib.sha256(blob).hexdigest() == expected
            checks[relative] = {"recorded_sha256": expected, "pinned_git_blob_matches": True, "current_matches": sha(ROOT / relative) == expected}
        records[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "code_revision": run["code_revision"], "status": run["status"], "gpu_name": run["gpu_name"], "training": {key: training[key] for key in keys}, "source_checks": checks, "private_checkpoint": run["private_checkpoint"], "scope": "Historical record inspection, not a new GPU execution or new verification of private remote availability."}
    first = records["mechanism-fixed3-runner-result.json"]["training"]
    second = records["mechanism-odd135-runner-result.json"]["training"]
    result["mechanism_pair_checks"] = {key: first[key] == second[key] for key in ("initial_weights_sha256", "seed", "batch_size", "last_step", "data_manifest_sha256")}
    assert all(result["mechanism_pair_checks"].values())
    result["historical_records"] = records
    settings_path = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
    start_path = ROOT / "docs/lesson-products/results/final-report-v1.started.json"
    final_path = ROOT / "docs/lesson-products/results/final-report-v1.json"
    settings, start, final = [json.loads(path.read_bytes()) for path in (settings_path, start_path, final_path)]
    assert settings["test_outcomes_observed"] is False
    assert settings["recorded_at"] < start["recorded_at"] < final["recorded_at"]
    assert start["settings_freeze_sha256"] == final["settings_freeze_sha256"] == sha(settings_path)
    result["pretest_protocol"] = {"protocol": settings["protocol"], "settings_at": settings["recorded_at"], "test_start_at": start["recorded_at"], "test_result_at": final["recorded_at"], "settings_sha256": sha(settings_path), "test_outcomes_observed_at_freeze": False, "mechanisms": settings["mechanisms"], "scope": "Predeclared thresholds and chronological records; not a claim of external timestamp attestation."}
    command = ["uv", "run", "--extra", "cpu", "python", "-m", "tiny_perceptron.lesson_products", "solve", "--b", "outputs/omni-revision/baseline/b", "--device", "cpu", "--loops", "5", "--state", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"]
    cli = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    answer = json.loads(cli.stdout)
    assert answer["answer"] == 2 and answer["loops"] == 5
    result["cpu_cli"] = {"argv": command, "exit_code": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr}
    figures = []
    for name in ("looped-mechanism-measured", "rewrite-04-block-flow"):
        source = ROOT / "course/figures" / (name + ".svg")
        rendered = Path("/workspace/work/omni-technical-19-15-v1") / (name + ".png")
        figures.append({"source": str(source.relative_to(ROOT)), "source_sha256": sha(source), "render_path_outside_git": str(rendered), "render_sha256": sha(rendered), "command": "inkscape " + str(source.relative_to(ROOT)) + " --export-type=png --export-filename=" + str(rendered), "viewed_with": "tools.view_image", "inspection": "Actually rendered and viewed. 14.15: log-scale CE and linear accuracy, six inference depths, blue fixed3 and orange sampled1/3/5, no monotonic guarantee. 4.5: two residual additions preserve three-position/eight-feature axes; attention label reads prior context, correctly contrasted by 19.15's unmasked whole-input attention."})
    result["prerequisite_figure_inspection"] = figures
    originals = []
    for path, url, locator in (
        ("/workspace/work/reference/omni-technical-t-8-v1/originals/looped-v2.pdf", "https://arxiv.org/pdf/2502.05171v2", "§3.1, Figure 2 and recurrence e=P(x), s_i=R(e,s_{i-1}), p=C(s_r); §3.2 architecture differences"),
        ("/workspace/work/reference/omni-technical-19-13-v1/universal-transformers-v3.pdf", "https://arxiv.org/pdf/1807.03819v3", "§2.1, equations (1)–(5), recurrence over depth rather than token positions"),
    ):
        originals.append({"url": url, "local_original_outside_git": path, "sha256": sha(path), "accessed_on": datetime.now(UTC).date().isoformat(), "actually_read": locator})
    result["original_papers"] = originals
    result["limits"] = "No formal training, GPU use, paid dispatch, large models, corpus dump or new release. The original section has no figure; prerequisite figure receipts do not expand its figure_sha256 map."
    (ARTIFACTS / "metadata-execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"historical_source_pins": "matched", "mechanism_pair": "matched", "pretest_order": "matched", "cpu_cli": answer, "artifact": str((ARTIFACTS / "metadata-execution.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
