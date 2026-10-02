"""匿名下載固定公開版本，實際在 CPU 載入／執行；範例輸出不是準確率。"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_course_models import MANIFEST, fetch_model, safe_relative  # noqa: E402

WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
TRAINING_ONLY = {
    "optimizer",
    "optimizer_state",
    "step",
    "training_state",
    "trainable_parameters",
    "teacher",
    "teacher_model",
    "reference",
    "reference_model",
    "ref_model",
    "rng_state",
    "torch_rng",
    "python_rng",
    "cuda_rng",
    "mps_rng",
    "sampler_rng",
    "dataset",
    "samples",
    "report",
}
FIELDS = {
    1: {"format_version", "config", "model", "tokenizer", "task"},
    "quantized-v1": {"format_version", "config", "model", "tokenizer", "bits", "task"},
    "multimodal-v1": {"format_version", "config", "model", "tokenizer", "modal_config", "task"},
    "simple-v1": {"format_version", "model", "vocabulary", "context", "width", "kind"},
    "lora-v1": {"format_version", "adapter", "config", "base_checkpoint", "base_sha256", "scaling"},
    "encoder-v1": {"format_version", "encoder", "classifier", "config", "classes"},
    "contrastive-v1": {"format_version", "model"},
    "finite-policy-v1": {"format_version", "model"},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained_file(folder, relative):
    path = folder / Path(safe_relative(relative))
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"缺少合法的公開 companion：{relative}")
    return path


def check_payload(saved):
    """未知格式或訓練狀態都拒絕；不以 torch.load 成功替代推論格式檢查。"""
    import torch

    if (
        not isinstance(saved, dict)
        or type(saved.get("format_version")) not in (str, int)
        or saved["format_version"] not in FIELDS
    ):
        raise ValueError("未知公開 checkpoint 格式")
    unexpected = set(saved) - FIELDS[saved["format_version"]] - {"metadata", "architecture"}
    if unexpected:
        raise ValueError(f"推論 checkpoint 含未知欄位：{sorted(unexpected)}")

    def visit(value, location="checkpoint"):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in TRAINING_ONLY or str(key).endswith("_rng"):
                    raise ValueError(f"公開檔含 training-only 欄位：{location}.{key}")
                visit(item, f"{location}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")
        elif isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"公開 tensor 含非有限值：{location}")

    visit(saved)
    return saved["format_version"]


def validate_folder(spec, folder):
    import torch

    files = {item["output"]: item for item in spec["files"]}
    if len(files) != len(spec["files"]):
        raise ValueError("公開下載清單有重複 output")
    for name, item in files.items():
        path = contained_file(folder, name)
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"下載後檔案 hash/bytes 不相符：{name}")
    exported = json.loads(contained_file(folder, "export-manifest.json").read_text())
    if exported.get("schema_version") != 1 or not exported.get("files"):
        raise ValueError("export-manifest 需要 schema_version=1 與非空 files")
    provenance = exported.get("provenance", {})
    if provenance.get("experiment_id") != spec["id"] or not re.fullmatch(
        r"[0-9a-f]{40}", provenance.get("revision", "")
    ):
        raise ValueError("export-manifest 缺少相符的 experiment/code revision")
    outputs = set()
    for item in exported["files"]:
        name = item["output"]
        if (
            name in outputs
            or name not in files
            or not item.get("license")
            or not re.fullmatch(r"[0-9a-f]{64}", item.get("source_sha256", ""))
        ):
            raise ValueError("export-manifest 路徑、授權或來源 SHA 無效")
        outputs.add(name)
        if item["sha256"] != files[name]["sha256"] or item["bytes"] != files[name]["bytes"]:
            raise ValueError("export-manifest 必須指向匯出後的實際 SHA/bytes")
    weights = []
    for name in files:
        if Path(name).suffix not in WEIGHT_SUFFIXES:
            continue
        if name not in outputs:
            raise ValueError(f"權重未列於 export-manifest：{name}")
        saved = torch.load(contained_file(folder, name), map_location="cpu", weights_only=True)
        version = check_payload(saved)
        tokenizer = saved.get("metadata", {}).get("tokenizer_file")
        if tokenizer and tokenizer not in files:
            raise ValueError("checkpoint 的 tokenizer 未列於公開下載清單")
        weights.append(
            {"output": name, "format_version": version, "sha256": files[name]["sha256"], "bytes": files[name]["bytes"]}
        )
    if not weights:
        raise ValueError("公開模型沒有可檢查的權重")
    return weights


def _finite_result(tensor):
    import torch

    if not torch.isfinite(tensor).all():
        raise ValueError("CPU forward 產生非有限結果")
    return {"shape": list(tensor.shape), "values": tensor.detach().flatten()[:16].tolist(), "finite": True}


def probe_checkpoint(checkpoint, folder):
    """真正重建並執行一次；固定合成輸入只檢查可運作，不做測驗計分。"""
    import torch
    from torch import nn

    from scripts.course_release import validate_base, validate_export
    from scripts.infer_simple import load_simple_checkpoint
    from tiny_perceptron.adapters import load_lora_adapter
    from tiny_perceptron.data import ByteTokenizer
    from tiny_perceptron.multimodal import AudioEncoder, VisionEncoder, scene, tone
    from tiny_perceptron.tokenization import load_tokenizer
    from tiny_perceptron.training import load_checkpoint

    checkpoint, folder = Path(checkpoint), Path(folder)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    version = check_payload(saved)
    validate_export(checkpoint)
    extra = {}
    with torch.no_grad():
        if version in (1, "quantized-v1", "multimodal-v1"):
            model, saved = load_checkpoint(checkpoint, "cpu")
            tokenizer = saved.get("metadata", {}).get("tokenizer_file")
            tokenizer_file = contained_file(folder, tokenizer) if tokenizer else None
            tok = load_tokenizer(tokenizer_file, saved["config"]["vocab_size"], saved)
            model.eval()
            if version == "multimodal-v1":
                task = saved.get("task") or saved.get("metadata", {}).get("task")
                if task not in ("vision", "audio", "joint"):
                    raise ValueError("多模態推論 checkpoint 缺少明確 task")
                image = scene(size=model.vision.image_size) if task != "audio" else None
                wave = tone() if task != "vision" else None
                markers = ([tok.image_id] if image is not None else []) + ([tok.audio_id] if wave is not None else [])
                ids = torch.tensor([tok.bos_id, tok.user_id, *markers, tok.assistant_id, tok.pad_id])
                result = model(ids, image=image, waveform=wave)["logits"]
                extra = {
                    "task": task,
                    "visual_tokens": 0 if image is None else (model.vision.image_size // model.vision.patch_size) ** 2,
                    "input_sources": {
                        "image": "synthetic scene" if image is not None else None,
                        "audio": "synthetic tone, 16kHz" if wave is not None else None,
                    },
                }
            else:
                result = model(torch.tensor([[tok.bos_id]]))["logits"]
                extra = {"tokenizer_vocab_size": tok.vocab_size}
        elif version == "simple-v1":
            model, saved, _ = load_simple_checkpoint(checkpoint, "cpu")
            model.eval()
            result = model(torch.zeros(1, saved["context"], dtype=torch.long))
            extra = {
                "kind": saved["kind"],
                "context": saved["context"],
                "vocabulary_size": len(saved["vocabulary"]) + 2,
            }
        elif version == "lora-v1":
            from huggingface_hub import hf_hub_download

            base = saved["base_checkpoint"]
            validate_base(base)
            base_file = Path(hf_hub_download(base["repo"], base["filename"], revision=base["revision"], token=False))
            if sha256(base_file) != base["sha256"]:
                raise ValueError("LoRA 的固定公開 base 檔案 SHA 不相符")
            model, original = load_checkpoint(base_file, "cpu")
            if check_payload(original) != 1:
                raise ValueError("LoRA 需要未量化 native TinyLM 公開 base")
            extra = {
                "base_checkpoint": base,
                "adapter": load_lora_adapter(model, checkpoint, base_state=original["model"]),
            }
            model.eval()
            result = model(torch.tensor([[ByteTokenizer.bos_id]]))["logits"]
        elif version == "encoder-v1":
            vision = saved["architecture"]["type"] == "VisionEncoder"
            model = VisionEncoder(**saved["config"]) if vision else AudioEncoder(**saved["config"])
            model.load_state_dict(saved["encoder"], strict=True)
            model.eval()
            result = model(scene(size=model.image_size)[None] if vision else tone()[None])
            extra = {
                "input_source": "synthetic scene" if vision else "synthetic tone, 16kHz",
                "features": _finite_result(result),
            }
            if "classifier" in saved:
                classifier = nn.Linear(saved["config"]["width"], len(saved["classes"]))
                classifier.load_state_dict(saved["classifier"], strict=True)
                result = classifier(result.mean(1))
                extra["classes"] = saved["classes"]
        elif version == "contrastive-v1":
            from scripts.course_experiments.modalities import _Contrastive

            model = _Contrastive()
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            result = model.scores(torch.stack([scene(), scene("blue", "circle")]), ["red square", "blue circle"])
            extra = {"input_source": "two synthetic scenes and literal descriptions"}
        elif version == "finite-policy-v1":
            from scripts.course_experiments.applications import _FinitePolicy

            model = _FinitePolicy()
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            features = torch.zeros(1, 18)
            features[0, [1, 8, 15]] = 1
            result = model(features)
            extra = {
                "input_source": "synthetic operands a=1,b=2,c=3",
                "chosen_action": saved["architecture"]["actions"][result.argmax(-1).item()],
            }
        else:
            raise ValueError("沒有此格式的 CPU probe")
    return {
        "format_version": version,
        "device": "cpu",
        "result": _finite_result(result),
        **extra,
        "scope": "load/CPU-forward smoke check only; synthetic inputs and sample output are not accuracy evidence",
    }


def inference_command(inference, folder, published):
    script = inference.get("script")
    allowed = {"scripts/infer.py", "scripts/infer_simple.py", "scripts/infer_modal.py"}
    if script not in allowed:
        raise ValueError(f"未知學生 inference CLI：{script}")
    checkpoint = inference.get("checkpoint")
    if checkpoint not in published:
        raise ValueError("inference checkpoint 未列於公開清單")
    command = [sys.executable, str(ROOT / script), str(contained_file(folder, checkpoint)), "--device", "cpu"]
    common = {"script", "checkpoint", "device", "prompt", "tokens"}
    supported = common | (
        {"tokenizer", "adapter", "chat", "temperature", "cache"}
        if script == "scripts/infer.py"
        else {"image", "audio", "resample_audio", "color", "shape", "frequency"}
        if script == "scripts/infer_modal.py"
        else set()
    )
    if set(inference) - supported:
        raise ValueError(f"不支援的 inference 設定：{sorted(set(inference) - supported)}")
    for field in ("prompt", "tokens", "temperature", "color", "shape", "frequency"):
        if field in inference:
            command += ["--" + field.replace("_", "-"), str(inference[field])]
    for field in ("tokenizer", "adapter", "image", "audio"):
        if field in inference:
            if inference[field] not in published:
                raise ValueError(f"inference companion 未公開：{field}")
            command += ["--" + field, str(contained_file(folder, inference[field]))]
    for field in ("chat", "cache", "resample_audio"):
        if inference.get(field):
            command += ["--" + field.replace("_", "-")]
    if script == "scripts/infer.py":
        command.append("--json")
    return command


def run_command(command):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    try:
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "returncode": None,
            "status": "timeout",
            "stdout": str(error.stdout or ""),
            "stderr": str(error.stderr or ""),
        }
    evidence = {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
    }
    try:
        evidence["output"] = json.loads(process.stdout)
        if not isinstance(evidence["output"], dict):
            raise ValueError("CLI JSON 必須是可核對欄位的 dict")
        evidence["status"] = "passed" if process.returncode == 0 else "failed"
    except (json.JSONDecodeError, ValueError):
        evidence["status"] = "failed"
        evidence["reason"] = "CLI 沒有輸出可核對的 JSON"
    return evidence


def save_evidence(directory, model_id, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("model id 必須是單一名稱")
    target = directory / f"{model_id}.json"
    if target.exists():
        target = directory / f"{model_id}.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return target


def check_model(spec, output, manifest_hash):
    import torch

    report = {
        "schema_version": 1,
        "model": spec["id"],
        "checked_at": datetime.now(UTC).isoformat(),
        "public_source": {"repo": spec["repo"], "revision": spec["revision"]},
        "manifest_sha256": manifest_hash,
        "anonymous_download": False,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "threads": 1,
        },
        "commands": [],
        "scope": "Student download/load/runtime verification; not a benchmark or model accuracy measurement.",
    }
    try:
        folder = Path(fetch_model(spec, output)).resolve()
        report["anonymous_download"] = True
        weights = validate_folder(spec, folder)
        report["verified_files"] = spec["files"]
        report["weights"] = weights
        report["source_hashes"] = {
            name: sha256(ROOT / name)
            for name in (
                "scripts/check_course_models.py",
                "scripts/course_release.py",
                "scripts/fetch_course_models.py",
                "tiny_perceptron/training.py",
                "tiny_perceptron/model.py",
                "tiny_perceptron/tokenization.py",
                "tiny_perceptron/multimodal.py",
                "tiny_perceptron/adapters.py",
                "tiny_perceptron/simple.py",
                "tiny_perceptron/quantization.py",
                "scripts/infer_simple.py",
                "scripts/course_experiments/modalities.py",
                "scripts/course_experiments/applications.py",
            )
        }
        report["code_checkout_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        for weight in weights:
            command = [
                sys.executable,
                str(ROOT / "scripts/check_course_models.py"),
                "--probe",
                str(contained_file(folder, weight["output"])),
                "--model-dir",
                str(folder),
            ]
            report["commands"].append(run_command(command))
        if spec.get("inference"):
            command = inference_command(spec["inference"], folder, {item["output"] for item in spec["files"]})
            report["commands"].append(run_command(command))
            report["source_hashes"][spec["inference"]["script"]] = sha256(ROOT / spec["inference"]["script"])
        report["status"] = "passed" if all(item["status"] == "passed" for item in report["commands"]) else "failed"
    except Exception as error:
        report.update(status="failed", exception_type=type(error).__name__, message=str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model", action="append", help="只檢查指定實驗；可重複指定")
    parser.add_argument("--all", action="store_true", help="明確選取此公開清單中的全部模型")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/course")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/course-experiments/student-checks")
    parser.add_argument("--probe", type=Path, help="內部 CPU 子命令：對單一 checkpoint 實際 forward")
    parser.add_argument("--model-dir", type=Path, help="probe 配對 tokenizer 的公開模型資料夾")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.probe:
        print(json.dumps(probe_checkpoint(args.probe, args.model_dir or args.probe.parent), ensure_ascii=False))
        return
    if not args.all and not args.model:
        parser.error("請明確指定 --model ID 或 --all；不預設下載全部模型")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("schema_version") != 1:
        raise ValueError("需要 schema_version=1 的公開模型清單")
    models = manifest["models"]
    if not models:
        raise ValueError("公開清單沒有可檢查的模型")
    if len({item["id"] for item in models}) != len(models):
        raise ValueError("公開模型 id 重複")
    requested = set(args.model or [])
    if requested - {item["id"] for item in models}:
        parser.error("指定模型尚未列於公開清單")
    selected = models if args.all else [item for item in models if item["id"] in requested]
    failed = False
    for spec in selected:
        report = check_model(spec, args.output, manifest_hash)
        path = save_evidence(args.evidence, spec["id"], report)
        print(json.dumps({"model": spec["id"], "status": report["status"], "evidence": str(path)}))
        failed |= report["status"] != "passed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
