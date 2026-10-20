"""保存人工分數算例、答案位置及既有實驗逐題紀錄的獨立核對。"""

import ast
import difflib
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.modalities import _audio_records, _vision_records, modal_inputs
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, patchify


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_bytes())


torch.set_num_threads(1)
torch.manual_seed(42)
tok = ByteTokenizer()
report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-13/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "platform": platform.platform(),
        "threads": str(torch.get_num_threads()),
        "seed": "42",
    },
    "checked_at": datetime.now(timezone.utc).isoformat(),
}

source = ROOT / "docs/course-review-20261019/freeze-01/sources/18.13.md"
body = (ROOT / "course/chapters/18.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", body))
index = next(i for i, h in enumerate(headers) if h[0].startswith(b"## 18.13 "))
current = body[headers[index].start() : headers[index + 1].start()]
assert current == source.read_bytes(), "目前課文與指定快照不同"
report["source_sha256"] = sha(current)

image = torch.zeros(1, 3, 16, 16)
report["patches"] = {str(p): list(patchify(image, p).shape) for p in (4, 8)}
report["toy"] = []
expected_kl = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
for student_prefix in (4, 6):
    teacher_prefix, answer_tokens = 16, 3
    t_rows = list(range(teacher_prefix - 1, teacher_prefix + answer_tokens - 1))
    s_rows = list(range(student_prefix - 1, student_prefix + answer_tokens - 1))
    teacher = torch.zeros(1, teacher_prefix + answer_tokens, 2)
    student = torch.zeros(1, student_prefix + answer_tokens, 2)
    teacher[:, t_rows] = torch.tensor([0.8, 0.2]).log()
    ta, sa = teacher[:, t_rows], student[:, s_rows]
    labels = torch.zeros(1, answer_tokens, dtype=torch.long)
    observed = distillation_kl(sa, ta, labels, temperature=1).item()
    assert abs(observed - expected_kl) < 1e-7, "KL 與獨立公式不同"
    report["toy"].append({
        "student_prefix": student_prefix,
        "teacher_rows": t_rows,
        "student_rows": s_rows,
        "teacher_full_shape": list(teacher.shape),
        "student_full_shape": list(student.shape),
        "teacher_selected_shape": list(ta.shape),
        "student_selected_shape": list(sa.shape),
        "teacher_probabilities": ta.softmax(-1).tolist(),
        "student_probabilities": sa.softmax(-1).tolist(),
        "valid_labels": int((labels != -100).sum()),
        "kl_float32": observed,
        "kl_independent_float64": expected_kl,
        "kl_rounded": round(observed, 4),
    })

experiment = read_json("docs/course-experiments/results/multimodal_distillation.json")
joint = read_json("docs/course-experiments/results/joint.json")
vqa = read_json("docs/course-experiments/results/vqa.json")
encoders = read_json("docs/course-experiments/results/encoders.json")
text_distillation = read_json("docs/course-experiments/results/distillation.json")
report["original_experiment_environment"] = {
    k: experiment[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu")
}

source_paths = [
    "tiny_perceptron/alignment.py",
    "tiny_perceptron/multimodal.py",
    "tiny_perceptron/data.py",
    "scripts/course_experiments/compression.py",
    "scripts/course_experiments/modalities.py",
    "docs/course-experiments/results/multimodal_distillation.json",
    "docs/course-experiments/results/encoders.json",
    "docs/course-experiments/results/joint.json",
    "docs/course-experiments/results/vqa.json",
    "docs/course-experiments/results/distillation.json",
    "course/figures/rewrite-18-modal-answer-rows.svg",
    "docs/course-review-20261019/figure-renders-04.json",
]
report["source_hashes"] = {p: sha((ROOT / p).read_bytes()) for p in source_paths}
report["experiment_code_matches"] = {
    p: experiment["code_sha256"][p] == report["source_hashes"][p]
    for p in source_paths[:5]
}
compression_path = "scripts/course_experiments/compression.py"
expected_code_sha = experiment["code_sha256"][compression_path]
history_command = ["git", "rev-list", "--all", "--", compression_path]
commits = subprocess.check_output(history_command, cwd=ROOT, text=True).splitlines()
for commit in commits:
    show_command = ["git", "show", f"{commit}:{compression_path}"]
    found = subprocess.run(show_command, cwd=ROOT, capture_output=True)
    if found.returncode == 0 and sha(found.stdout) == expected_code_sha:
        historical = found.stdout.decode()
        current_code = (ROOT / compression_path).read_text()
        old_functions = {n.name: n for n in ast.parse(historical).body if isinstance(n, ast.FunctionDef)}
        current_functions = {n.name: n for n in ast.parse(current_code).body if isinstance(n, ast.FunctionDef)}
        unchanged = {
            name: ast.dump(old_functions[name], include_attributes=False) == ast.dump(current_functions[name], include_attributes=False)
            for name in ("_teacher", "_dataset", "_save_float", "_cache_modal", "_fit_modal")
        }
        assert all(unchanged.values()), "相關訓練與對齊函式已改變"
        report["historical_compression"] = {
            "history_command_argv": history_command,
            "candidates": len(commits),
            "show_command_argv": show_command,
            "matching_commit": commit,
            "sha256": sha(found.stdout),
            "unchanged_functions": unchanged,
            "reviewed_diff": list(difflib.unified_diff(historical.splitlines(), current_code.splitlines(), fromfile="recorded_code", tofile="current_code")),
            "inspection_note": "差異只有教師 task 驗證及部署檔 metadata 的 task 欄；答案快取、學生訓練、對齊與評估呼叫相同。",
        }
        break
else:
    raise AssertionError("找不到實驗紀錄的原始程式 SHA")

datasets = {
    "vqa": {k: value["records"] for k, value in vqa["results"]["data"]["splits"].items()},
    "joint": {k: value["records"] for k, value in joint["results"]["data"]["splits"].items()},
}
assert datasets["vqa"] == _vision_records(("shape?", "color?")), "視覺資料與產生器不同"
assert encoders["results"]["audio"]["data"]["splits"]["train"]["records"] == _audio_records()["train"], "音訊預訓練資料不同"
report["tasks"] = {}
for name, task in experiment["results"]["tasks"].items():
    parts = datasets[name]
    families = {k: set(r["family"] for r in rows) for k, rows in parts.items()}
    overlap = {f"{a}/{b}": sorted(families[a] & families[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    dataset_sha = sha((json.dumps(parts, ensure_ascii=False, indent=2) + "\n").encode())
    assert dataset_sha == task["data"]["sha256"], "教師資料 SHA 不同"
    assert all(not values for values in overlap.values()), "資料家族有交集"
    provenance = task["teacher_provenance"]
    tr = {
        "teacher": {k: value for k, value in provenance.items() if k != "metadata"},
        "teacher_task": provenance["metadata"]["task"],
        "teacher_modal_config": provenance["metadata"]["report"]["modal_config"],
        "visual_tokens": task["visual_tokens"],
        "language_widths": task["language_widths"],
        "data": task["data"],
        "dataset_sha_recomputed": dataset_sha,
        "family_intersections_recomputed": overlap,
        "runs": {},
    }
    ce, kd = (task["runs"][k]["training"] for k in ("ce", "ce_kl"))
    same_fields = ("steps", "optimizer_updates", "batch_size", "training_examples", "effective_answer_tokens", "initialization_sha256", "batch_plan_sha256")
    tr["matched_training"] = {k: ce[k] == kd[k] for k in same_fields}
    assert all(tr["matched_training"].values()), "學生訓練條件不同"
    rng = random.Random(experiment["seed"])
    plan = [[rng.randrange(len(parts["train"])) for _ in range(ce["batch_size"])] for _ in range(ce["steps"])]
    assert sha(json.dumps(plan).encode()) == ce["batch_plan_sha256"], "抽樣計畫不同"
    effective = sum(len(tok.encode(parts["train"][i]["answer"])) + 1 for batch in plan for i in batch)
    assert effective == ce["effective_answer_tokens"], "有效答案 token 分母不同"
    tr["plan_recomputed"] = {"sha256": sha(json.dumps(plan).encode()), "effective_answer_tokens": effective}
    for method, run in task["runs"].items():
        rr = {"training": {k: value for k, value in run["training"].items() if k != "loss_trace"}, "evaluations": {}}
        for split in ("test", "blank_image_test", "blank_audio_test"):
            if split not in run:
                continue
            evaluation = run[split]
            samples = evaluation["samples"]
            assert len(samples) == len(parts["test"]) == 12, "測試樣本數不同"
            correct = 0
            for row, sample in zip(parts["test"], samples, strict=True):
                ids = sample["generated_ids"]
                raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                assert tok.decode(raw) == sample["generated"], "生成字串與 ID 不同"
                assert sample["target"] == row["answer"], "目標與資料不同"
                assert sample["family"] == row["family"], "測試家族不同"
                assert sample["question"] == row["question"], "問題不同"
                exact = raw == tok.encode(row["answer"])
                assert exact == sample["exact_match"], "逐題判分不同"
                correct += int(exact)
            assert correct == evaluation["correct"], "總答對題數不同"
            assert all(s["eos"] and not s["generation_error"] for s in samples), "生成未正常停止"
            rr["evaluations"][split] = {
                "examples": len(samples), "correct_recomputed": correct,
                "effective_tokens": evaluation["effective_tokens"],
                "ablation": evaluation["ablation"],
                "all_normal_eos": all(s["eos"] for s in samples),
            }
        if name == "joint":
            base, bi, ba = (run[k]["samples"] for k in ("test", "blank_image_test", "blank_audio_test"))
            rr["paired_samples"] = [{
                "family": a["family"], "question": a["question"], "target": a["target"],
                "generated": a["generated"], "generated_ids": a["generated_ids"],
                "blank_image_generated": b["generated"], "blank_image_generated_ids": b["generated_ids"],
                "blank_audio_generated": c["generated"], "blank_audio_generated_ids": c["generated_ids"],
            } for a, b, c in zip(base, bi, ba, strict=True)]
            rr["joint_recomputed"] = {
                "blank_image_identical_ids": sum(a["generated_ids"] == b["generated_ids"] for a, b in zip(base, bi, strict=True)),
                "generated_shapes": [s["generated"].split(",")[0] for s in base],
                "target_square_count": sum(s["target"].startswith("square,") for s in base),
                "pitch_correct": sum(s["generated"].split(",")[1] == s["target"].split(",")[1] for s in base),
                "blank_audio_pitches": [s["generated"].split(",")[1] for s in ba],
                "blank_audio_changed_ids": sum(a["generated_ids"] != b["generated_ids"] for a, b in zip(base, ba, strict=True)),
            }
        tr["runs"][method] = rr
    report["tasks"][name] = tr

report["frequencies"] = {
    "joint": {k: sorted(set(r["frequency"] for r in rows)) for k, rows in datasets["joint"].items()},
    "audio_encoder_train": sorted(set(r["frequency"] for r in _audio_records()["train"])),
}
report["text_teachers"] = {
    name: {k: task["teacher_provenance"][k] for k in ("experiment", "format_version", "sha256")}
    for name, task in text_distillation["results"]["tasks"].items()
}
def find_questions(value):
    if isinstance(value, dict):
        if isinstance(value.get("question"), str):
            yield value["question"]
        for child in value.values():
            yield from find_questions(child)
    elif isinstance(value, list):
        for child in value:
            yield from find_questions(child)


report["attribute_question_examples"] = list(dict.fromkeys(find_questions(text_distillation["results"]["tasks"]["attributes"])))[:3]

report["real_alignment"] = {}
ctx = SimpleNamespace(device=torch.device("cpu"))
for name, task in experiment["results"]["tasks"].items():
    training = task["runs"]["ce_kl"]["training"]
    rng = random.Random(experiment["seed"])
    record = datasets[name]["train"][rng.randrange(len(datasets[name]["train"]))]
    ids, labels, media_image, waveform, count = modal_inputs(record, ctx)
    teacher = MultiModalLM(TinyLM(ModelConfig(**task["teacher_provenance"]["config"])))
    student = MultiModalLM(TinyLM(ModelConfig(vocab_size=264, width=32, layers=1, heads=2, max_length=128)), vision_width=8, audio_width=8)
    student.vision = VisionEncoder(width=8, image_size=16, patch_size=8)
    selected = {}
    with torch.no_grad():
        for role, model in (("teacher", teacher), ("student", student)):
            output = model(ids, labels, image=media_image, waveform=waveform)
            valid = output["labels"][0] != -100
            selected[role] = {
                "prediction_rows": valid.nonzero().flatten().tolist(),
                "answer_ids": output["labels"][0, valid].tolist(),
                "full_logits_shape": list(output["logits"].shape),
                "vision_shape": list(model.vision(media_image[None]).shape),
            }
            assert selected[role]["prediction_rows"] == training["alignment"][role + "_prediction_rows"], "真實預測位置不同"
            assert selected[role]["answer_ids"] == training["alignment"]["answer_ids"], "真實答案 ID 不同"
    report["real_alignment"][name] = {"record": record, "answer_count_with_eos": count, **selected, "random_weights_only": True}

tree = ast.parse((ROOT / "scripts/course_experiments/compression.py").read_text())
functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
report["inspected_modal_training_calls"] = {
    name: sorted({ast.unparse(n.func) for n in ast.walk(functions[name]) if isinstance(n, ast.Call)})
    for name in ("_cache_modal", "_fit_modal", "run_multimodal_distillation")
}

report["original_sources"] = []
specifications = [
    ("causal", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py", "v4.57.1; ForCausalLMLoss", "shift_labels = labels[..., 1:].contiguous()"),
    ("kl", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/loss.py", "torch 2.14.1; KLDivLoss", "loss_pointwise = target * (target.log() - input)"),
    ("distillation", "https://arxiv.org/html/1503.02531v1", "arXiv:1503.02531v1; Distillation; S2.p2.1", "a soft target distribution for each case in the transfer set"),
    ("features", "https://arxiv.org/html/1412.6550v4", "arXiv:1412.6550v4; Hint-based training; S2.SS2.p2.1", "we add a regressor to the guided layer"),
]
for name, url, version, excerpt in specifications:
    command = ["curl", "-fsSL", url]
    completed = subprocess.run(command, check=True, capture_output=True, timeout=45)
    raw = completed.stdout
    plain = raw.decode() if url.endswith(".py") else re.sub(r"<[^>]+>", " ", raw.decode())
    plain = " ".join(plain.split())
    assert excerpt in plain, f"指定的原文摘錄不存在：{name}"
    report["original_sources"].append({
        "id": name, "url": url, "version": version,
        "command_argv": command, "exit_code": completed.returncode,
        "read_full_response_sha256": sha(raw), "read_bytes": len(raw),
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "excerpt": excerpt,
    })

report["result"] = "全部斷言通過；只重算既有紀錄與位置，沒有訓練或重跑已訓練權重的生成。"
print(json.dumps(report, ensure_ascii=False, indent=2))
