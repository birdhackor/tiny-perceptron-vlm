"""固定 CPU 正式配方的單次執行器；不呼叫 Modal、不開 final。"""

import argparse
import json
import os
import platform
import signal
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402

from tiny_perceptron.omni_products import training_runtime as trainer  # noqa: E402
from tiny_perceptron.omni_products.inference import load_trained  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs/omni-products/v1"
AUDIT = ROOT / "docs/course-revision-20261029-omni/infrastructure/formal-core-audit-v1.json"


def utc():
    return datetime.now(UTC).isoformat()


def read_frozen(root, name, expected=None):
    path = root / name
    if path.is_symlink() or (expected is not None and trainer.file_sha(path) != expected):
        raise ValueError(f"固定檔案不同：{name}")
    return json.loads(path.read_bytes())


def checked_job(frozen, job_id):
    document = read_frozen(frozen, "formal-recipes.json")
    if document["schema"] != "omni-formal-recipes-cpu-v1" or document["final_opened"] is not False:
        raise ValueError("只接收final前固定的CPU正式配方")
    read_frozen(frozen, "data-definition.json", document["data_definition_sha256"])
    metrics = read_frozen(frozen, "metrics.json", document["metrics_sha256"])
    if metrics["data_definition_sha256"] != document["data_definition_sha256"]:
        raise ValueError("度量必須綁定相同資料")
    for name, digest in metrics["measurement_sources_sha256"].items():
        if (ROOT / name).is_symlink() or trainer.file_sha(ROOT / name) != digest:
            raise ValueError("實際度量／T／tau實作偏離先凍結的metrics來源")
    if document["implementation_sha256"] != trainer.source_binding():
        raise ValueError("正式訓練實作已漂移")
    for name, digest in document["execution_sources_sha256"].items():
        if trainer.file_sha(ROOT / name) != digest:
            raise ValueError(f"執行器已漂移：{name}")
    if document["runtime"]["torch"] != str(torch.__version__) or document["runtime"]["device"] != "cpu":
        raise ValueError("不可默默換torch或運算backend")
    audit = json.loads(AUDIT.read_bytes())
    required = {
        str((frozen / name).relative_to(ROOT)): trainer.file_sha(frozen / name)
        for name in ("data-definition.json", "metrics.json", "formal-recipes.json")
    }
    required.update(document["execution_sources_sha256"])
    required.update(metrics["measurement_sources_sha256"])
    required.update(
        {f"tiny_perceptron/omni_products/{name}": digest for name, digest in trainer.source_binding().items()}
    )
    if (
        audit.get("status") != "pass_for_reviewed_scope"
        or not audit.get("reviewer_task")
        or audit["reviewer_task"] == "/root"
        or any(audit.get("reviewed_files_sha256", {}).get(name) != digest for name, digest in required.items())
    ):
        raise ValueError("CPU正式資料／度量／實作／配方尚未完成獨立回核")
    selected = document["jobs"][job_id]
    config = selected["training"]
    if (
        selected["maximum_attempts"] != 1
        or selected["kind"] not in {"av", "choice", "codec", "talker"}
        or type(config["updates"]) is not int
        or not 1 <= config["updates"] <= 50000
        or type(config["batch_size"]) is not int
        or not 1 <= config["batch_size"] <= 1024
        or not 1 <= config["training_seconds"] <= 3600
        or not 1 <= config["validate_every"] <= config["updates"]
    ):
        raise ValueError("正式job超出已固定的更新／時間／批次界")
    if selected["kind"] == "choice" and (
        config["data_mode"] not in {"augmented", "compact"} or config["recompute"] is not False
    ):
        raise ValueError("C正式對照必須是資料對照")
    return selected, document


def verified_formal_training(output_root, frozen, job_id):
    """選擇檔的SHA之外，還須是同配方真正完成的正式工作紀錄。"""
    selected, document = checked_job(frozen, job_id)
    root = output_root / job_id
    origin = read_frozen(root, "job-origin.json")
    result = read_frozen(root, "job-result.json")
    training_path = root / "training/training-result.json"
    training = json.loads(training_path.read_bytes())
    expected = {
        "schema": "omni-formal-local-job-v1",
        "job": job_id,
        "kind": selected["kind"],
        "stage": "formal-training",
        "attempt": 1,
        "final_opened": False,
        "formal_recipe_sha256": trainer.file_sha(frozen / "formal-recipes.json"),
        "data_definition_sha256": document["data_definition_sha256"],
        "metrics_sha256": document["metrics_sha256"],
        "implementation_sha256": trainer.source_binding(),
    }
    if (
        any(origin.get(key) != value or result.get(key) != value for key, value in expected.items())
        or result.get("success") is not True
        or result.get("job_origin_sha256") != trainer.file_sha(root / "job-origin.json")
        or result.get("training_result_sha256") != trainer.file_sha(training_path)
        or training.get("complete") is not True
        or training.get("updates") != selected["training"]["updates"]
        or any(training["config"].get(key) != value for key, value in selected["training"].items())
        or training.get("implementation_sha256") != trainer.source_binding()
        or training.get("source_unchanged_during_training") is not True
    ):
        raise ValueError("必須是完整且成功的同一正式配方產物；development／partial codec不可產targets")
    return selected, document, result, training


def train(selected, output, codec_path=None):
    kind, config = selected["kind"], selected["training"]
    if kind == "av":
        return trainer.train_av(config, output, "cpu")
    if kind == "choice":
        return trainer.train_choice(config, output, "cpu")
    if kind == "codec":
        return trainer.train_codec(config, output, "cpu")
    if codec_path is None:
        raise ValueError("Talker需要先完成並凍結的正式codec")
    codec = load_trained(codec_path, "codec")
    return trainer.train_talker(config, output, "cpu", codec)


def execute(job_id, frozen, output_root):
    if not hasattr(signal, "SIGALRM") or not callable(getattr(signal, "alarm", None)):
        raise RuntimeError("正式執行器需要POSIX絕對deadline；此平台在寫origin／lock之前拒絕")
    selected, document = checked_job(frozen, job_id)
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".formal-running"
    lock.mkdir(exist_ok=False)
    output = output_root / job_id
    created_output = False
    alarm_installed = False
    started = time.monotonic()
    result = {
        "schema": "omni-formal-local-job-v1",
        "job": job_id,
        "kind": selected["kind"],
        "stage": "formal-training",
        "success": False,
        "attempt": 1,
        "started_at": utc(),
        "final_opened": False,
        "Modal_dispatched": False,
        "new_Modal_reservation": "0.00",
        "formal_recipe_sha256": trainer.file_sha(frozen / "formal-recipes.json"),
        "data_definition_sha256": document["data_definition_sha256"],
        "metrics_sha256": document["metrics_sha256"],
        "implementation_sha256": trainer.source_binding(),
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "python": platform.python_version()},
        "execution_host": {
            "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
    }
    try:
        output.mkdir(exist_ok=False)
        created_output = True
        trainer.json_file(output / "job-origin.json", result)
        codec_path = None
        if selected["kind"] == "talker":
            _, _, _, codec_record = verified_formal_training(output_root, frozen, "codec")
            codec_path = output_root / "codec" / "training"
            binding = read_frozen(frozen, "codec-selection.json")
            if (
                binding["formal_recipe_sha256"] != result["formal_recipe_sha256"]
                or trainer.file_sha(codec_path / "weights.safetensors") != binding["weights_sha256"]
                or trainer.file_sha(codec_path / "training-result.json") != binding["training_result_sha256"]
                or binding["final_opened"] is not False
                or codec_record["files"]["weights.safetensors"]["sha256"] != binding["weights_sha256"]
            ):
                raise ValueError("正式codec尚未依同配方凍結，不可產targets")
            result["codec_selection_sha256"] = trainer.file_sha(frozen / "codec-selection.json")
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("正式CPU工作絕對deadline")))
        signal.alarm(selected["training"]["training_seconds"] + 300)
        alarm_installed = True
        trained = train(selected, output / "training", codec_path)
        if not trained["complete"] or trained["updates"] != selected["training"]["updates"]:
            raise RuntimeError("未完成固定更新數；保留partial，不列完整正式產物")
        # 一般使用者真正會走的safetensors載入路徑也要通過。
        restored = load_trained(output / "training", selected["kind"])
        if trainer.parameter_sha(restored) == trained["initial_parameter_sha256"]:
            raise RuntimeError("輸出的best權重仍等同隨機初值")
        result.update(success=True, training_result_sha256=trainer.file_sha(output / "training/training-result.json"))
    except BaseException:
        result["exception"] = traceback.format_exc()
    finally:
        if alarm_installed:
            signal.alarm(0)
        result.update(finished_at=utc(), elapsed_seconds=time.monotonic() - started)
        if created_output and (output / "job-origin.json").is_file():
            result["job_origin_sha256"] = trainer.file_sha(output / "job-origin.json")
            trainer.json_file(output / "job-result.json", result)
        lock.rmdir()
    if not result["success"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    print(json.dumps({key: result[key] for key in ("job", "success", "elapsed_seconds", "training_result_sha256")}))
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("job")
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/omni-formal-core-v1")
    args = parser.parse_args()
    execute(args.job, args.frozen, args.output_root)


if __name__ == "__main__":
    main()
