"""先準備可審閱成品；三種審閱通過後測試實際載入並封裝。"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .experiment import write_json
from .qualification import active_contract, sha256, verify_training_receipts


def verify_candidate(root):
    verify_training_receipts(root, require_complete=True)
    qualifier = root / "qualification"
    result = json.loads((qualifier / "final-results.json").read_text())
    opening = json.loads((qualifier / "final-opening.json").read_text())
    checks = json.loads((qualifier / "product-checks.json").read_text())
    legacy = json.loads((qualifier / "legacy-regression.json").read_text())
    selection = json.loads((qualifier / "selection.json").read_text())
    current = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    if current != opening["implementation"]:
        raise ValueError("正式驗收後實作發生變更，需要明確處理而不能直接封裝")
    if (
        not result["primary_gates_pass"]
        or not all(result["gates"].values())
        or not checks["pass"]
        or not all(checks["gates"].values())
        or not legacy["pass"]
        or not all(legacy["gates"].values())
    ):
        raise ValueError("成品尚未通過全部驗收")
    for key, path in (
        ("final_opening_sha256", qualifier / "final-opening.json"),
        ("observations_sha256", qualifier / "final-observations.json"),
        ("examples_sha256", qualifier / "final-examples.json"),
        ("families_sha256", qualifier / "final-families.json"),
    ):
        if result[key] != sha256(path):
            raise ValueError("正式結果引用不符")
    if (
        opening["selection_sha256"] != sha256(qualifier / "selection.json")
        or opening["product_checks_sha256"] != sha256(qualifier / "product-checks.json")
        or opening["legacy_regression_sha256"] != sha256(qualifier / "legacy-regression.json")
        or opening["weights_sha256"] != sha256(root / "best.safetensors")
    ):
        raise ValueError("實際交付與開封時選定的權重、政策或服務檢查不符")
    if opening["previous_registry_sha256"] is not None and opening["previous_registry_sha256"] != sha256(
        qualifier / "registry-before-final.json"
    ):
        raise ValueError("排除歷次暴露世界的原始登記快照不符")
    contract, contract_path = active_contract(root.parent)
    if opening["contract_sha256"] != sha256(contract_path):
        raise ValueError("驗收規劃被變更")
    return result, opening, checks, selection, contract


def prepare(root, output):
    result, opening, checks, selection, contract = verify_candidate(root)
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "bundle"
    bundle.mkdir()
    shutil.copyfile(root / "best.safetensors", bundle / "weights.safetensors")
    shutil.copyfile(root / "tokenizer.json", bundle / "tokenizer.json")
    write_json(
        bundle / "policy.json",
        {"temperatures": selection["temperatures"], "threshold": selection["selected"]["threshold"]},
    )
    best = json.loads((root / "best.json").read_text())
    write_json(
        bundle / "bundle.json",
        {
            "schema": "c-product-bundle-v1",
            "status": "candidate_pending_independent_reviews",
            "config": best["config"],
            "files": {path.name: sha256(path) for path in sorted(bundle.iterdir())},
            "final_opening_sha256": result["final_opening_sha256"],
            "scope": contract["scope"],
        },
    )
    source = Path(__file__).resolve().parents[2]
    destination = output / "tiny_perceptron/decision_product"
    destination.mkdir(parents=True)
    shutil.copyfile(source / "tiny_perceptron/__init__.py", destination.parent / "__init__.py")
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, destination / path.name)
    examples = output / "examples"
    examples.mkdir()
    for path in (source / "examples/decision_product").glob("*.json"):
        shutil.copyfile(path, examples / path.name)
    shutil.copyfile(source / "examples/decision_product/USAGE.txt", output / "README.txt")
    shutil.copyfile(source / "LICENSE", output / "LICENSE")
    (output / "requirements.txt").write_text("torch>=2.11\nnumpy>=1.26\nsafetensors==0.8.0\n")
    evidence = output / "evidence"
    evidence.mkdir()
    for name in ("plan.json", "source-freeze.json", "completion.json", "best.json", "training.jsonl"):
        shutil.copyfile(root / name, evidence / name)
    for name in ("initial.safetensors", "lineage.json", "lineage-parent.json", "recipe-interpretation-v1.json"):
        if (root / name).exists():
            shutil.copyfile(root / name, evidence / name)
    for name in (
        "selection.json",
        "final-opening.json",
        "final-results.json",
        "product-checks.json",
        "final-observations.json",
        "final-examples.json",
        "final-families.json",
        "registry-before-final.json",
        "legacy-regression.json",
    ):
        shutil.copyfile(root / "qualification" / name, evidence / name)
    write_json(
        output / "qualification-summary.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "measured_candidate_pending_independent_reviews",
            "final": result,
            "manual_product_gates": checks["gates"],
            "manual_product_metrics": checks["metrics"],
            "latency_ms": checks["latency_ms"],
            "training": {
                "recipe": root.name,
                "best_step": best["step"],
                "weights_sha256": best["weights_sha256"],
                "lineage": json.loads((root / "lineage.json").read_text())
                if (root / "lineage.json").exists()
                else None,
            },
            "scope": contract["scope"],
            "limitations": contract["statistical_limits"],
        },
    )
    write_json(
        output / "candidate-origin.json",
        {
            "experiment": str(root.resolve()),
            "opening_sha256": sha256(root / "qualification/final-opening.json"),
            "implementation": opening["implementation"],
            "readme_sha256": sha256(output / "README.txt"),
            "candidate_bundle_sha256": sha256(bundle / "bundle.json"),
        },
    )
    print(json.dumps({"event": "candidate_prepared", "directory": str(output)}), flush=True)


def seal(root, output, review_index):
    verify_candidate(root)
    index = json.loads(review_index.read_text())
    origin = json.loads((output / "candidate-origin.json").read_text())
    if origin["opening_sha256"] != sha256(root / "qualification/final-opening.json"):
        raise ValueError("候選成品不是這次正式驗收的版本")
    runtime = output / "tiny_perceptron/decision_product"
    if {path.name: sha256(path) for path in runtime.glob("*.py")} != origin["implementation"]:
        raise ValueError("交付程式與正式驗收版本不符")
    manifest_path = output / "bundle/bundle.json"
    if sha256(manifest_path) != origin["candidate_bundle_sha256"]:
        raise ValueError("交付候選 manifest 已變更")
    manifest = json.loads(manifest_path.read_text())
    selection = json.loads((root / "qualification/selection.json").read_text())
    best = json.loads((root / "best.json").read_text())
    expected_files = {
        "weights.safetensors": sha256(root / "best.safetensors"),
        "tokenizer.json": sha256(root / "tokenizer.json"),
        "policy.json": sha256(output / "bundle/policy.json"),
    }
    policy = json.loads((output / "bundle/policy.json").read_text())
    if (
        manifest["files"] != expected_files
        or manifest["config"] != best["config"]
        or policy != {"temperatures": selection["temperatures"], "threshold": selection["selected"]["threshold"]}
    ):
        raise ValueError("交付的權重、tokenizer、模型設定或校準政策不符正式驗收")
    if any(sha256(output / "bundle" / name) != digest for name, digest in expected_files.items()):
        raise ValueError("交付 bundle 檔案發生變更")
    if (
        index["implementation"] != origin["implementation"]
        or index["readme_sha256"] != sha256(output / "README.txt")
        or index["candidate_bundle_sha256"] != origin["candidate_bundle_sha256"]
    ):
        raise ValueError("審閱來源與交付實作／使用說明不符")
    entries = index["reviews"]
    if (
        {entry["kind"] for entry in entries} != {"readability", "correctness", "coherence"}
        or len(entries) != 3
        or len({entry["actor"] for entry in entries}) != 3
    ):
        raise ValueError("必須有三位獨立審閱者的三種審閱")
    reviews = output / "evidence/reviews"
    reviews.mkdir(exist_ok=False)
    for entry in entries:
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ValueError("審閱原件須為index目錄內的相對路徑")
        path = review_index.parent / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("審閱原件指紋不符")
        report = json.loads(path.read_text())
        if (
            report["status"] not in {"pass", "pass_with_optional"}
            or report["reviewer"] != entry["actor"]
            or report["review_type"] != entry["kind"]
            or report.get("unresolved_necessary", 0) != 0
        ):
            raise ValueError("審閱未通過或存在必要問題")
        if (
            report["implementation"] != origin["implementation"]
            or report["readme_sha256"] != sha256(output / "README.txt")
            or report["candidate_bundle_sha256"] != origin["candidate_bundle_sha256"]
        ):
            raise ValueError("審閱原件沒有綁定這個交付版本")
        copy = reviews / entry["path"]
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
    shutil.copyfile(review_index, reviews / "index.json")
    manifest["status"] = "qualified"
    manifest["review_index_sha256"] = sha256(review_index)
    manifest["source_candidate_bundle_sha256"] = origin["candidate_bundle_sha256"]
    write_json(manifest_path, manifest)
    probes = []
    try:
        for name in ("record-ready.json", "record-unknown.json", "answer-absent.json"):
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tiny_perceptron.decision_product",
                    "--bundle",
                    "bundle",
                    "--request",
                    f"examples/{name}",
                    "--threads",
                    "1",
                ],
                cwd=output,
                env={**os.environ, "PYTHONPATH": str(output.resolve())},
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            reply = json.loads(process.stdout)
            choice = reply["results"]["choice"]
            if name == "record-ready.json":
                if (
                    choice["status"] != "accepted"
                    or choice["selected"] != "record"
                    or reply["results"]["info"]["model_choice"] != "yes"
                    or reply["results"]["rubric"]["model_choice"] != "2"
                ):
                    raise ValueError("完整範例交付後行為不符")
            elif choice["status"] != "abstained" or choice["model_choice"] != "none":
                raise ValueError("拒答範例交付後行為不符")
            if name == "record-unknown.json":
                info, rubric = reply["results"]["info"], reply["results"]["rubric"]
                if (
                    info["status"] != "prediction"
                    or info["model_choice"] != "no"
                    or rubric["status"] != "abstained"
                    or rubric["model_choice"] != "none"
                    or rubric["selected"] is not None
                ):
                    raise ValueError("缺欄位範例的資訊判斷或評分拒答不符")
            probes.append({"request": name, "response": reply, "process_exit_code": process.returncode})
    except Exception:
        manifest["status"] = "candidate_seal_failed"
        write_json(manifest_path, manifest)
        raise
    write_json(output / "evidence/delivery-probes.json", probes)
    summary_path = output / "qualification-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "qualified_in_declared_scope"
    summary["three_reviews"] = entries
    write_json(summary_path, summary)
    files = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    write_json(output / "MANIFEST.json", {"schema": "c-product-delivery-files-v1", "files": files})
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                handle.write(path, path.relative_to(output.parent))
    print(
        json.dumps(
            {
                "event": "qualified_product_packaged",
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "seal"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviews")
    args = parser.parse_args()
    if args.phase == "seal" and args.reviews is None:
        parser.error("seal 必須提供 --reviews")
    if args.phase == "prepare":
        prepare(Path(args.experiment), Path(args.output))
    else:
        seal(Path(args.experiment), Path(args.output), Path(args.reviews))


if __name__ == "__main__":
    main()
