"""Small receipts for the exact primary fields inspected by this reviewer."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).with_name("validation-primary-inspection.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = {"reviewer_task": "/root/backfill_tech_validation", "recorded_at": datetime.now(timezone.utc).isoformat(), "scope": "Primary result fields and immutable source receipts; not reproduction of GPU, old production tests, or human semantic grading.", "environment": {"python": platform.python_version(), "device": "cpu", "nbclient": version("nbclient"), "ipykernel": version("ipykernel")}}
    base = Path("/workspace/selftrained-v2/outputs/selftrained-v2/operations")
    old = {}
    for architecture, run, archived in [("moe", "moe-joint-native-balanced-4000-attempt-1", "moe-native"), ("dense", "dense-joint-weighted-10000-attempt-2", "dense-weighted")]:
        inference = next((base / run).rglob("raw/inference-manifest.json"))
        metrics = inference.with_name("metrics.jsonl")
        manifest = json.loads(inference.read_text())
        validations = [{k: row[k] for k in ("step", "validation_loss", "selected")} for row in map(json.loads, metrics.open()) if row.get("event") == "validation"]
        receipt = json.loads((ROOT / f"docs/selftrained/results/training-raw/{archived}/receipt.json").read_text())
        files = {row["path"]:row for row in receipt["files"]}
        assert sha(inference) == files["inference-manifest.json"]["sha256"]
        assert sha(metrics) == files["metrics.jsonl"]["sha256"]
        assert manifest["selected_step"] == 1000
        assert min(validations, key=lambda x:x["validation_loss"])["step"] == 1000
        old[architecture] = {"inference_manifest": str(inference), "inference_sha256": sha(inference), "metrics": str(metrics), "metrics_sha256": sha(metrics), "selected_step": manifest["selected_step"], "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"], "validation_rows": validations, "completed_steps": receipt["job"]["steps"], "receipt_sha256": sha(ROOT / f"docs/selftrained/results/training-raw/{archived}/receipt.json")}
    result["v2_selected_step_primary_fields"] = old
    prior = []
    numeric_seal = json.loads(Path(__file__).with_name("validation-numeric-observation-seal.json").read_text())
    assert sha(ROOT / numeric_seal["path"]) == numeric_seal["sha256"]
    for name in ("b-mechanisms", "c"):
        p = ROOT / f"docs/lesson-products/reviews/engineering-final-{name}-execution.json"
        x = json.loads(p.read_text())
        item = {k:x[k] for k in ("reviewer_task", "reviewer_context", "command", "environment", "scope")}
        item["path"] = str(p.relative_to(ROOT))
        item["sha256"] = sha(p)
        item["script_sha256"] = x.get("script_sha256", x.get("probe_script_sha256"))
        item["raw_reference"] = x.get("raw_reference", {"path": "outputs/lesson-products/final-report-v1.predictions.jsonl", "sha256": x["checked_documents"]["outputs/lesson-products/final-report-v1.predictions.jsonl"]}) if name == "b-mechanisms" else x["raw_reference"]
        prior.append(item)
    assert prior[0]["reviewer_task"] != prior[1]["reviewer_task"]
    assert all(x["raw_reference"]["sha256"] == "8b593421043b2d0ee8b81caea34612b468e8af7eafd5b84c552f8c25b0cf4896" for x in prior)
    result["separate_procedure_only_inspection_after_own_numeric_seal"] = {"numeric_seal": numeric_seal, "prior_execution_records": prior, "limitation": "Shows two distinct task records with executed commands and saved-score references. This is not an additional independent numeric conclusion. Neither -original.json judgment was read."}
    v4path = ROOT / "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json"
    v4 = json.loads(v4path.read_text())
    result["v4_primary_selection_counts"] = {"path": str(v4path.relative_to(ROOT)), "sha256": sha(v4path), "denominators": v4["denominators"], "selected_variant": v4["selected_variant"], "variants": {k:{key:val for key,val in value.items() if key in ("correct_counts", "all_generations_complete", "eligible", "gates")} for k,value in v4["variants"].items()}, "scope": "Original validation count/rubric artifacts inspected; no new manual grading or natural-model inference."}
    localbase = ROOT / "docs/selftrained/infrastructure/v2-local-stage/independent-review"
    local = []
    for n in ("pretrain.final.raw.log", "latest-exact-resume.final.raw.log", "reject-manifest-pin.final.raw.log", "reject-tampered-latest.final.raw.log"):
        p = localbase / n
        text = p.read_text()
        local.append({"path":str(p.relative_to(ROOT)), "sha256":sha(p), "supporting_excerpt":text[-450:] if n.startswith("reject") else [{k:row[k] for k in ("stage", "steps", "train_records", "validation_records", "completed_requested_steps")} for row in map(json.loads,text.splitlines()) if row.get("schema")]})
    result["local_stage_raw_logs"] = local
    ft = Path("/workspace/work/curriculum-20261016/validation-fulltexts")
    result["external_original_receipts"] = [
        {"url":"https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst", "version":"scikit-learn 1.7.2", "path":str(ft/"sklearn-1.7.2-cross-validation.rst"), "sha256":sha(ft/"sklearn-1.7.2-cross-validation.rst"), "locator":"Introduction lines 10–70, heldout group discussion", "inspection":"Training overlap can inflate scores; validation for settings, final test after selection. No guarantee that a finite test proves open-domain ability."},
        {"url":"https://proceedings.mlr.press/v70/guo17a/guo17a.pdf", "version":"PMLR 70, ICML 2017, pp.1321–1330", "path":str(ft/"guo17a.pdf"), "sha256":sha(ft/"guo17a.pdf"), "locator":"Section 2 equation 6; Section 4.2 equation 9 and adjacent temperature discussion", "inspection":"NLL equals negative log assigned gold probability; T>0 rescales logits and keeps class argmax; tuning T uses validation. No claim of guaranteed heldout improvement."},
        {"url":"https://arxiv.org/pdf/1506.03099", "version":"arXiv:1506.03099v3, 2015-09-23", "path":str(ft/"scheduled-sampling.pdf"), "sha256":sha(ft/"scheduled-sampling.pdf"), "locator":"Abstract, Section 1 and Section 2: Sequence Prediction", "inspection":"Reads gold previous token in likelihood training; inference feeds generated previous tokens and errors may accumulate. Used only for training/inference distinction; no scheduled-sampling benefit claim applied to this project."},
    ]
    result["own_notebook_execution"] = {"command":"uv run --frozen --extra cpu python scripts/check_notebooks.py --mode kernel --workers 1 --lesson 14.15 --output /workspace/work/curriculum-20261016/validation-notebooks", "result":json.loads(Path("/workspace/work/curriculum-20261016/validation-notebooks/validation-kernel.json").read_text()), "notebook_source_sha256":sha(ROOT/"notebooks/14/14.15.ipynb"), "executed_output_sha256":sha(Path("/workspace/work/curriculum-20261016/validation-notebooks/14/14.15.ipynb")), "scope":"One actual fresh CPU kernel; not a new whole-book run or a Colab/MPS/CUDA check."}
    result["own_readonly_command_outputs"] = [{"command":"uv run --frozen --extra cpu python scripts/build_course.py --check", "exit_code":0, "stdout":"312 個小節：同步檢查通過"}, {"command":"uv run --frozen --extra cpu python -m ipykernel install --help", "exit_code":0, "verified_options":["--sys-prefix", "--name NAME", "--display-name DISPLAY_NAME"], "scope":"CLI options inspected without registering or modifying a kernel."}]
    from tiny_perceptron.lesson_products.evaluation import frozen_guard
    frozen_guard(
        ROOT / "docs/lesson-products/results/settings-freeze-v1.json",
        ROOT / "outputs/lesson-products/data-v2",
        ROOT / "outputs/lesson-products/products/b",
        ROOT / "outputs/lesson-products/products/c",
        {"fixed3": ROOT / "outputs/lesson-products/mechanisms/fixed3", "odd": ROOT / "outputs/lesson-products/mechanisms/odd"},
    )
    result["readonly_frozen_guard"] = {"status": "passed", "scope": "Hash-only original frozen_guard executed; validates exact code/data/weights/settings identity, no model construction or forwards and no marker writes."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status":"all_assertions_passed", "output":str(OUT.relative_to(ROOT)), "prior_task_count":len(prior), "v2_selected_steps":{k:v["selected_step"] for k,v in old.items()}},ensure_ascii=False))


if __name__ == "__main__":
    main()
