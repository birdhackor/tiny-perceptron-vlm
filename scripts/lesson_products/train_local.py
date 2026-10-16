"""在自己的CPU/MPS/CUDA上重做固定配方；不連Modal或HF、不啟動遠端工作。"""

import argparse
import json
import os
from pathlib import Path

from scripts.lesson_products.modal_runner import DATA_MANIFEST, ROOT, sha256, validate_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/lesson-products-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--resume-from", type=Path, help="同一配方的既有best.pt/last.pt目錄；輸出仍另立新目錄")
    options = parser.parse_args()
    job = validate_job(json.loads(options.job.read_bytes()))
    if job["resume"] is not None:
        parser.error("遠端恢復描述請用Modal runner；本機恢復請以原配方加--resume-from")
    if options.output.exists():
        parser.error("輸出需另立新目錄，保留既有產物與失敗紀錄")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from tiny_perceptron.lesson_products.data import write_dataset
    from tiny_perceptron.lesson_products.train import train

    manifest = ROOT / DATA_MANIFEST
    if not (options.data / "manifest.json").is_file():
        if options.data.exists() and any(options.data.iterdir()):
            parser.error("既有非空資料目錄缺manifest；請另選目錄")
        config = json.loads(manifest.read_bytes())["config"]
        write_dataset(options.data, **{key: config[key] for key in ("per_length", "stress_per_length", "services")})
    if sha256(options.data / "manifest.json") != sha256(manifest):
        parser.error("資料manifest不同於固定v2配方；不覆寫或混用")
    if options.resume_from:
        import torch

        stored = torch.load(options.resume_from / "last.pt", weights_only=True, map_location="cpu")
        metadata = stored["metadata"]
        recipe = {"batch_size": job["batch_size"], "learning_rate": job["learning_rate"]}
        if (
            metadata.get("seed") != job["seed"]
            or metadata.get("training_recipe") != recipe
            or any(group["lr"] != job["learning_rate"] for group in stored["optimizer"]["param_groups"])
        ):
            parser.error("續訓checkpoint的seed／batch／learning rate與job不符；拒絕默默改用舊配方")
    receipt = train(
        options.data,
        options.output,
        job["product"],
        job["mode"],
        max_steps=job["max_steps"],
        max_seconds=job["train_seconds"],
        batch_size=job["batch_size"],
        learning_rate=job["learning_rate"],
        seed=job["seed"],
        depth_weights=job["depth_weights"],
        device=options.device,
        resume=options.resume_from / "last.pt" if options.resume_from else None,
        resume_best=options.resume_from / "best.pt" if options.resume_from else None,
    )
    print(
        json.dumps(
            {
                "product": receipt["product"],
                "step": receipt["last_step"],
                "output": str(options.output),
                "scope": "Local development training only; no final-test score or public release.",
            }
        )
    )


if __name__ == "__main__":
    main()
