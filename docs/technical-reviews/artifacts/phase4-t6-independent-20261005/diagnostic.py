"""Bounded CPU inspection of T.6 contracts; no capability evaluation or long training."""
import ast
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np
import PIL
from PIL import Image
import soundfile as sf
import torch
from scripts import pretrain_encoders as pretrain, train, infer_modal, evaluate_modal, prepare_ocr
from scripts.course_experiments import modalities, run
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder, scene, tone
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything

assert torch.version.cuda is None
torch.set_num_threads(1)
torch.set_default_device("cpu")
OUT = Path(__file__).resolve().parent
RESULT = {"environment": {"python": sys.version, "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
    "numpy": np.__version__, "pillow": PIL.__version__, "soundfile": sf.__version__, "device": "cpu", "threads": "1"},
    "scope": "1-step entrypoint diagnostics, 2-step resume comparison, and synthetic file/scoring checks; no recipe completion or model score claims",
    "invocations": []}


def invoke(module, arguments):
    argv = [str(module.__file__), *map(str, arguments)]
    buffer = io.StringIO()
    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(buffer):
        module.main()
    text = buffer.getvalue()
    RESULT["invocations"].append({"argv": argv, "stdout": text, "exit_code": 0})
    return json.loads(text.splitlines()[-1])


def rejected(function):
    try:
        function()
    except ValueError as error:
        return str(error)
    raise AssertionError("expected ValueError")


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def pretrain_check(modality, updating, temporary):
    models, starts, saves = [], [], []
    real_encoder = pretrain.VisionEncoder if modality == "vision" else pretrain.AudioEncoder
    def spy_encoder():
        model = real_encoder()
        models.append(model)
        starts.append(clone_state(model))
        return model
    def intercept_save(payload, path):
        saves.append({"path": str(path), "encoder": {k: v.detach().clone() for k, v in payload["encoder"].items()},
            "keys": list(payload)})
    args = ["--modality", modality, "--steps", "1", "--device", "cpu", "--output", temporary / (modality + ".pt")]
    if updating:
        args.append("--train")
    with patch.object(pretrain, "VisionEncoder" if modality == "vision" else "AudioEncoder", spy_encoder), patch.object(torch, "save", intercept_save):
        report = invoke(pretrain, args)
    changed = [k for k, v in models[0].state_dict().items() if not torch.equal(v, starts[0][k])]
    assert bool(changed) == updating
    assert bool(saves) == updating
    assert report["mode"] == ("train" if updating else "dry-run-no-weight-update")
    assert report["holdout_examples"] == 2 and len(report["loss"]) == 1
    assert report["train_examples"] == (16 if modality == "vision" else 4)
    return {"modality": modality, "updating": updating, "report": report, "changed_encoder_parameters": changed,
        "save_calls": [{"path": x["path"], "keys": x["keys"]} for x in saves], "saved_weights_retained": False}


def train_check(task, scope, base, encoder_paths, temporary):
    trace = {}
    real_optimizer = torch.optim.AdamW
    def optimizer(parameters, **kwargs):
        model = inspect.currentframe().f_back.f_locals["train_model"]
        trace["model"] = model
        trace["initial"] = clone_state(model)
        return real_optimizer(parameters, **kwargs)
    arguments = ["--task", task, "--checkpoint", base, "--freeze", scope, "--train", "--steps", "1", "--batch-size", "1", "--device", "cpu", "--output", temporary / f"{task}-{scope}.pt"]
    if task != "audio": arguments += ["--vision-encoder", encoder_paths["vision"]]
    if task != "vision": arguments += ["--audio-encoder", encoder_paths["audio"]]
    with patch.object(torch.optim, "AdamW", optimizer):
        report = invoke(train, arguments)
    model = trace["model"]
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, trace["initial"][k])]
    trainables = [k for k, p in model.named_parameters() if p.requires_grad]
    with_gradient = [k for k, p in model.named_parameters() if p.grad is not None]
    assert changed and set(changed) <= set(trainables)
    unused = [k for k in trainables if k.startswith(("audio.", "audio_projector."))] if task == "vision" else [k for k in trainables if k.startswith(("vision.", "image_projector."))] if task == "audio" else []
    assert not set(unused) & set(changed) and not set(unused) & set(with_gradient)
    if scope == "projector": assert all("projector" in k for k in trainables)
    if scope == "partial":
        assert all("projector" in k or k.startswith(("language.blocks.0.", "language.blocks.2.")) for k in trainables)
        assert any(k.startswith("language.blocks.0.") for k in changed)
        assert any(k.startswith("language.blocks.2.") for k in changed)
    if scope == "none": assert len(trainables) == len(list(model.parameters()))
    assert report["mode"] == "train" and len(report["history"]) == 1
    return {"task": task, "freeze": scope, "trainable_parameters": trainables, "parameters_with_gradient": with_gradient,
        "changed_parameters": changed, "unused_open_parameters": unused, "output": str(temporary / f"{task}-{scope}.pt")}


with tempfile.TemporaryDirectory(prefix="t6-factual-") as temporary_name:
    temporary = Path(temporary_name)
    tree = ast.parse((ROOT / "scripts/pretrain_encoders.py").read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "args.modality == 'vision'")
    generated = {}
    for modality, nodes in [("vision", branch.body), ("audio", branch.orelse)]:
        assignment = next(n for n in nodes if isinstance(n, ast.Assign))
        namespace = {"scene": scene, "tone": tone}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "original-pretrain-items", "exec"), namespace)
        items = namespace["items"]
        training = [(x, y) for x, y, flag in items if not flag]
        held = [(x, y) for x, y, flag in items if flag]
        assert len(held) == 2 and [y for x, y in held] == [0, 1]
        generated[modality] = {"original_assignment": ast.unparse(assignment), "train_examples": len(training), "holdout_examples": len(held), "holdout_labels": [y for x, y in held], "exact_tensor_overlap": sum(torch.equal(a, b) for a, y in training for b, z in held)}
    vision_rows = [(color, shape, offset, int(shape == "circle"), color == "blue" and offset == 1) for color in ("red", "green", "blue") for shape in ("square", "circle") for offset in (-1, 0, 1)]
    generated["vision"]["holdout_raw_attributes"] = [list(r[:4]) for r in vision_rows if r[-1]]
    generated["vision"]["training_offsets"] = sorted({r[2] for r in vision_rows if not r[-1]})
    generated["vision"]["training_at_holdout_offset"] = [list(r[:4]) for r in vision_rows if not r[-1] and r[2] == 1]
    generated["audio"]["holdout_raw_frequencies_hz"] = [180, 1000]
    generated["accuracy_arithmetic"] = {"one_of_two": 1 / 2, "two_of_two": 2 / 2}
    RESULT["pretrain_partition"] = generated
    RESULT["pretrain_runs"] = [pretrain_check(modality, updating, temporary) for modality in ("vision", "audio") for updating in (False, True)]

    seed_everything(42)
    base = temporary / "base.pt"
    save_checkpoint(base, TinyLM(ModelConfig(width=8, layers=3, heads=1, max_length=128)), metadata={"seed": 42})
    encoder_paths = {}
    for name, model in [("vision", VisionEncoder()), ("audio", AudioEncoder())]:
        path = temporary / f"external-{name}.pt"
        torch.save({"encoder": model.state_dict()}, path)
        encoder_paths[name] = path
    RESULT["train_freeze_runs"] = [train_check(task, scope, base, encoder_paths, temporary) for task in ("vision", "audio", "joint") for scope in ("projector", "partial", "none")]

    continuous, stopped, resumed = [temporary / f"{name}.pt" for name in ("continuous", "stopped", "resumed")]
    common = ["--task", "joint", "--freeze", "partial", "--train", "--steps", "2", "--batch-size", "1", "--device", "cpu"]
    invoke(train, common + ["--checkpoint", base, "--output", continuous])
    invoke(train, common + ["--checkpoint", base, "--stop-after", "1", "--output", stopped])
    invoke(train, common + ["--checkpoint", stopped, "--resume", "--output", resumed])
    a, b = [torch.load(p, weights_only=True) for p in (continuous, resumed)]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["step"] == b["step"] == 2 and torch.equal(a["torch_rng"], b["torch_rng"]) and a["python_rng"] == b["python_rng"]
    RESULT["resume"] = {"continuous_steps": a["step"], "stopped_steps": 1, "resumed_final_steps": b["step"], "all_weight_tensors_exactly_equal": True, "torch_python_rng_equal": True, "format_version": b["format_version"], "payload_keys": list(b),
        "external_encoder_rejected": rejected(lambda: invoke(train, common + ["--checkpoint", stopped, "--resume", "--vision-encoder", encoder_paths["vision"]])),
        "changed_schedule_rejected": rejected(lambda: invoke(train, common + ["--checkpoint", stopped, "--resume", "--steps", "3"])),
        "changed_task_rejected": rejected(lambda: invoke(train, common + ["--checkpoint", stopped, "--resume", "--task", "vision"])),
        "changed_freeze_rejected": rejected(lambda: invoke(train, common + ["--checkpoint", stopped, "--resume", "--freeze", "projector"]))}
    inference_only = torch.load(stopped, weights_only=True)
    inference_only["optimizer"] = None
    torch.save(inference_only, temporary / "inference-only.pt")
    RESULT["resume"]["missing_optimizer_rejected"] = rejected(lambda: invoke(train, common + ["--checkpoint", temporary / "inference-only.pt", "--resume"]))
    changed_data = temporary / "changed-data.jsonl"
    changed_data.write_text('{"image":"image.png","question":"read digits","answer":"12"}\n')
    RESULT["resume"]["changed_data_rejected"] = rejected(lambda: invoke(train, common + ["--checkpoint", stopped, "--resume", "--data", changed_data]))
    loaded, payload, task = infer_modal.load_modal_checkpoint(resumed, "cpu")
    x = torch.tensor([[8, 9, 10]])
    text = loaded.language(x)["logits"]
    loaded_text = TinyLM(ModelConfig(**payload["config"]))
    loaded_text.load_state_dict({k.removeprefix("language."): v for k, v in payload["model"].items() if k.startswith("language.")})
    assert torch.equal(text, loaded_text(x)["logits"])
    RESULT["language_extraction"] = {"class": type(loaded.language).__name__, "vocab_size": loaded.language.config.vocab_size, "same_text_input_logits_equal": True, "no_text_ability_claim": True}
    RESULT["inference_contract"] = []
    for name, args in [("vision", ["--color", "blue", "--shape", "circle", "--prompt", "shape?"]), ("audio", ["--frequency", "880", "--prompt", "pitch?"]), ("joint", ["--color", "red", "--shape", "square", "--frequency", "220", "--prompt", "joint?"])]:
        report = invoke(infer_modal, [temporary / f"{name}-projector.pt", *args, "--tokens", "16", "--device", "cpu"])
        assert "answer" in report and "eos" in report and "generation_status" in report
        RESULT["inference_contract"].append({"task": name, "prompt": report["prompt"], "input_sources": report["input_sources"], "answer_keys": [k for k in report if k in ("answer", "eos", "generated_ids", "generation_status", "valid_answer_tokens")], "expectations_are_task_targets_not_scores": True})

    tok = ByteTokenizer()
    RESULT["synthetic_answer_targets"] = []
    for name, choices, expected in [("vision", ["blue", "circle", True], "circle"), ("audio", ["blue", "circle", True], "high"), ("joint", ["red", "square", False], "square,low")]:
        target_model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=128)))
        captured = {}
        def capture_target(module, args, kwargs):
            captured["labels"] = args[1].detach().clone()
        handle = target_model.register_forward_pre_hook(capture_target, with_kwargs=True)
        with patch.object(train.random, "choice", side_effect=choices):
            loss = train.modal_loss(target_model, name, tok, "cpu")
        handle.remove()
        labels = captured["labels"]
        actual = tok.decode(labels[labels >= 0].tolist())
        assert actual == expected and torch.isfinite(loss)
        RESULT["synthetic_answer_targets"].append({"task": name, "color_shape_high": choices, "decoded_supervision": actual, "expected": expected, "last_label_is_eos": labels[-1].item() == tok.eos_id})

    Image.new("L", (32, 24), color=123).save(temporary / "image.png")
    ids, labels, image, wave = modal_example({"image": "image.png", "question": "read digits", "answer": "12"}, temporary, "vision")
    assert image.shape == (3, 16, 16) and wave is None
    tok = ByteTokenizer()
    assert tok.decode(labels[labels >= 0].tolist()) == "12"
    audio_results = []
    for name, values, rate in [("valid", np.zeros(1600, dtype="float32"), 16000), ("wrong_rate", np.zeros(800, dtype="float32"), 8000), ("stereo", np.zeros((1600, 2), dtype="float32"), 16000), ("empty", np.zeros(0, dtype="float32"), 16000)]:
        sf.write(temporary / f"{name}.wav", values, rate, subtype="FLOAT")
        record = {"audio": f"{name}.wav", "question": "pitch?", "answer": "low"}
        if name == "valid":
            _, _, _, waveform = modal_example(record, temporary, "audio")
            assert waveform.shape == (1600,)
            observation = {"accepted": True, "samples": len(waveform)}
        else: observation = {"accepted": False, "error": rejected(lambda: modal_example(record, temporary, "audio"))}
        audio_results.append({"case": name, "rate_hz": rate, **observation})
    RESULT["file_contract"] = {"image_shape_CHW": list(image.shape), "jsonl_relative_path_checked": True, "decoded_target": "12", "audio_cases": audio_results,
        "samples_per_second": 1600 / 0.1, "tone_220_count": len(tone(220)), "tone_880_count": len(tone(880)), "frequency_not_sample_rate": True}

    records = []
    for i, color in enumerate((64, 128, 192)):
        Image.new("RGB", (16, 16), (color, 0, 0)).save(temporary / f"score-{i}.png")
        records.append({"image": f"score-{i}.png", "question": "read digits", "answer": str(i)})
    data = temporary / "scoring.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    def deterministic_generator(model, prefix, image, wave, tokens):
        pixel = round(float(image[0, 0, 0]) * 255)
        answer = {64: "0", 128: "1", 192: "2", 0: "blank"}[pixel]
        return torch.cat((prefix, torch.tensor(tok.encode(answer) + [tok.eos_id])))
    scoring = {}
    for ablation in ("none", "blank", "shuffle"):
        output = temporary / f"eval-{ablation}.json"
        with patch.object(evaluate_modal, "generate_modal", deterministic_generator):
            invoke(evaluate_modal, [temporary / "vision-projector.pt", "--data", data, "--ablation", ablation, "--tokens", "16", "--device", "cpu", "--output", output])
        report = json.loads(output.read_text())
        assert report["examples"] == 3
        assert [r["target"] for r in report["samples"]] == ["0", "1", "2"]
        if ablation == "shuffle":
            assert all(s["row"] != s["donor_row"] and s["generated"] == str(s["donor_row"]) and not s["exact_match"] for s in report["samples"])
        scoring[ablation] = {k: report[k] for k in ("examples", "exact_match", "samples")}
    RESULT["ablation_scoring"] = {"controlled_generator": "Encodes the donor image identity; this inspects scoring, not model ability", "results": scoring}
    assert prepare_ocr.draw_digits("12").size == (16, 16)
    RESULT["ocr_renderer"] = {"input": "12", "size_WH": [16, 16], "too_many_digits_rejected": rejected(lambda: prepare_ocr.draw_digits("123"))}
    probe = MultiModalLM(TinyLM(ModelConfig(width=8, layers=3, heads=1, max_length=128)))
    modalities._freeze(probe, "partial")
    fixed = [k for k, p in probe.named_parameters() if p.requires_grad]
    assert all(k.startswith(("image_projector.", "language.blocks.2.")) for k in fixed)
    RESULT["fixed_vqa_partial"] = fixed
    RESULT["experiment_dependencies"] = {name: {k: run.experiment_spec(name)[k] for k in ("module", "function", "dependencies", "assets")} for name in ("sft", "encoders", "projector", "vqa")}

RESULT["temporary_weights_removed"] = not temporary.exists()
assert RESULT["temporary_weights_removed"]
RESULT["checked_code_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["scripts/pretrain_encoders.py", "scripts/train.py", "scripts/infer_modal.py", "scripts/evaluate_modal.py", "scripts/prepare_ocr.py", "scripts/course_experiments/modalities.py", "scripts/course_experiments/run.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/modal_data.py", "tiny_perceptron/training.py", "tiny_perceptron/tokenization.py", "docs/course-experiments/plan.json"]}
(OUT / "diagnostic-results.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"checks_completed": True, "pretrain_entrypoint_runs": len(RESULT["pretrain_runs"]), "modal_1_step_runs": len(RESULT["train_freeze_runs"]), "resume_weight_equality": RESULT["resume"]["all_weight_tensors_exactly_equal"], "training_offsets": RESULT["pretrain_partition"]["vision"]["training_offsets"], "temporary_weights_removed": RESULT["temporary_weights_removed"]}))
