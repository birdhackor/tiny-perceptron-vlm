"""11.7 的獨立 CPU 核對；不訓練、不重現既有 CUDA 能力成績。"""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.course_experiments.common import split_records
from scripts.course_experiments.modalities import _freeze, _hash, _loss_fn, _vision_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
OUT = Path(__file__).with_name("execution.json")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(path, lesson):
    raw = (ROOT / path).read_text()
    match = re.search(r"^## " + re.escape(lesson) + r" .+$", raw, re.M)
    end = re.search(r"^## ", raw[match.end() :], re.M)
    return raw[match.start() : match.end() + end.start() if end else len(raw)].encode()


def run_snippet(code):
    output = io.StringIO()
    env = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "11.7 frozen embedded Python", "exec"), env)
    return {"stdout": output.getvalue(), "ratios": env["ratios"], "recorded": env["recorded"]}


torch.set_num_threads(1)
torch.manual_seed(7)
freeze = json.loads((ROOT / FREEZE).read_bytes())
pages = {page["page_id"]: page for page in freeze["pages"]}
pins = {}
for lesson in ["11.7", "11.5", "7.15", "7.16", "10.8", "19.21"]:
    page = pages[lesson]
    snap = (ROOT / page["snapshot"]).read_bytes()
    current = section(page["source"], lesson)
    assert digest(snap) == digest(current) == page["source_sha256"]
    pins[lesson] = {"snapshot": page["snapshot"], "source": page["source"], "sha256": digest(snap)}
assert pages["11.7"]["figures_sha256"] == {}
body = (ROOT / pages["11.7"]["snapshot"]).read_text()
assert not re.search(r"!\[[^\]]*\]\([^)]*\)", body)
code = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(code) == 1
original = run_snippet(code[0])
exercise = run_snippet(code[0].replace('750, "圖文有效目標": 250', '500, "圖文有效目標": 500'))
assert original["ratios"] == {"文字有效目標": 0.75, "圖文有效目標": 0.25}
assert exercise["ratios"] == {"文字有效目標": 0.5, "圖文有效目標": 0.5}
assert exercise["recorded"] == original["recorded"]
assert original["stdout"].splitlines()[1:] == ["訓練前 文字 0.8 圖文 0.2", "訓練後 文字 0.6 圖文 0.7"]

report_path = "docs/course-experiments/results/vqa.json"
report = json.loads((ROOT / report_path).read_bytes())
parts = split_records(generate_records("attributes-sft"), seed=42)
for name, rows in parts.items():
    assert len(rows) == report["results"]["text_data"][name]["count"]
    assert _hash(rows) == report["results"]["text_data"][name]["sha256"]
families = {name: {row["family"] for row in rows} for name, rows in parts.items()}
assert not families["train"] & (families["validation"] | families["test"])
assert not families["validation"] & families["test"]
assert any(row["messages"][0]["content"] == "color=blue;shape=circle;pitch=low;pitch?" and row["messages"][1]["content"] == "low" for row in parts["test"])
vision = _vision_records(("shape?", "color?"))
assert {name: _hash(rows) for name, rows in vision.items()} == {name: row["sha256"] for name, row in report["results"]["data"]["splits"].items()}
tok = ByteTokenizer()
ctx = SimpleNamespace(seed=42, device="cpu")
model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=2, max_length=128))).eval()
sampling = {}
for ratio, name in [(0, "all"), (0.5, "all_replay")]:
    counts = {"text": 0, "vision": 0}
    targets = {"text": 0, "vision": 0}
    per_step = []
    calculate = _loss_fn(model, vision["train"], ctx, replay=parts["train"], replay_ratio=ratio)
    for step in range(160):
        rng = random.Random(42 + step)
        n_step = 0
        for _ in range(4):
            if rng.random() < ratio:
                row = rng.choice(parts["train"])
                kind = "text"
                n = int((render_chat(row["messages"])[1] != -100).sum())
            else:
                row = rng.choice(vision["train"])
                kind = "vision"
                n = len(tok.encode(row["answer"])) + 1
            counts[kind] += 1
            targets[kind] += n
            n_step += n
        with torch.no_grad():
            loss, actual = calculate(step)
        assert torch.isfinite(loss) and actual == n_step
        per_step.append(actual)
    history = report["results"]["variants"][name]["training"]["history"]
    assert per_step == [row["effective_targets"] for row in history]
    sampling[name] = {"examples": counts, "effective_targets": targets, "total_targets": sum(targets.values()), "per_step_matches_published_history": True, "actual_helper_count_checked_at_all_160_steps": True}
assert sampling["all"]["total_targets"] == 3830
assert sampling["all_replay"]["total_targets"] == 4214
assert 4214 - 3830 == 384

scope = {}
for mode in ["projector", "partial", "all"]:
    _freeze(model, mode)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if mode == "projector":
        assert all(name.startswith("image_projector.") for name in names)
    elif mode == "partial":
        assert all(name.startswith(("image_projector.", "language.blocks.1.")) for name in names)
    else:
        assert all(value.requires_grad for value in model.parameters())
    scope[mode] = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "language_trainable": any(name.startswith("language.") for name in names)}

table = {}
expected = {"projector_only": (3830, 3, 5), "partial": (3830, 12, 2), "all": (3830, 9, 1), "all_replay": (4214, 9, 4)}
for name, (targets, image_correct, text_correct) in expected.items():
    variant = report["results"]["variants"][name]
    training = variant["training"]
    assert training["steps"] == len(training["history"]) == 160
    assert sum(row["effective_targets"] for row in training["history"]) == training["effective_targets"] == targets
    test = variant["test"]
    assert test["examples"] == len(test["samples"]) == 12
    assert sum(row["exact_match"] for row in test["samples"]) == test["correct"] == image_correct
    for sample in test["samples"]:
        ids = sample["generated_ids"]
        raw = ids[: ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        assert (raw == tok.encode(sample["target"])) == sample["exact_match"]
    for phase, correct in [("text_before", 5), ("text_after", text_correct)]:
        evaluation = variant[phase]
        assert evaluation["records"] == len(evaluation["samples"]) == 10
        assert sum(row["exact"] for row in evaluation["samples"]) == evaluation["matches"] == correct
        for i, sample in enumerate(evaluation["samples"]):
            assert sample["messages"] == parts["test"][i]["messages"][:-1]
            assert sample["expected"] == parts["test"][i]["messages"][-1]["content"]
            ids = sample["generated_ids"]
            raw = ids[: ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            assert (raw == tok.encode(sample["expected"])) == sample["exact"]
    table[name] = {"steps": 160, "effective_targets": targets, "image_correct": image_correct, "image_n": 12, "text_before_correct": 5, "text_after_correct": text_correct, "text_n": 10}
replay = report["results"]["variants"]["all_replay"]
flips = [{"question": b["messages"][0]["content"], "expected": b["expected"], "before": b["generated"], "after": a["generated"], "before_exact": b["exact"], "after_exact": a["exact"]} for b, a in zip(replay["text_before"]["samples"], replay["text_after"]["samples"], strict=True) if b["exact"] != a["exact"]]
assert len(flips) == 3 and sum(row["before_exact"] for row in flips) == 2

code_pins = []
for path in ["scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/prepare_data.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py"]:
    current = (ROOT / path).read_bytes()
    historical = subprocess.run(["git", "show", report["revision"] + ":" + path], cwd=ROOT, check=True, capture_output=True).stdout
    assert current == historical
    if path in report["code_sha256"]:
        assert digest(current) == report["code_sha256"][path]
    code_pins.append({"path": path, "sha256": digest(current), "equal_to_report_revision": True})

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-11-7-v1/verify.py",
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "platform": platform.platform(), "torch_threads": "1"},
    "source_freeze": {"path": FREEZE, "sha256": digest((ROOT / FREEZE).read_bytes())},
    "source_and_prerequisite_pins": pins,
    "figures": {"main_page_figures": {}, "render_required": False, "reason": "11.7 沒有圖片引用；19.21 僅作模組／工程狀態文字依據，沒有借用其圖作本節證據。"},
    "embedded_code": {"sha256": digest(code[0].encode()), "original": original, "exercise": exercise},
    "published_result": {"path": report_path, "sha256": digest((ROOT / report_path).read_bytes()), "revision": report["revision"], "original_environment": {key: report[key] for key in ["device", "gpu", "python_version", "torch_version", "seed"]}},
    "training_code_pins": code_pins,
    "split_counts": {name: len(rows) for name, rows in parts.items()},
    "text_family_overlap": False,
    "sampling": sampling,
    "freeze_scope_probe_width8_layers2": scope,
    "table_recomputed_from_saved_samples": table,
    "replay_flips": flips,
    "result": "所有斷言通過。完整當版內嵌程式／練習、既有紀錄的逐題 exact-match 與分母、160 步抽樣有效目標、當版 helper 計數及更新範圍一致。",
    "limits": "沒有執行 backward、optimizer.step、CUDA 或訓練，也沒有載入既有 CUDA checkpoint 重生答案；歷史能力數字的獨立核對是重算保存的生成 token／題目與指標。CPU 隨機小模型只供當版抽樣及計數通路核對，不支持任何品質、速度或 GPU 能力結論。",
}
if OUT.exists():
    raise FileExistsError(OUT)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"artifact": OUT.relative_to(ROOT).as_posix(), "sha256": digest(OUT.read_bytes()), "result": result["result"]}, ensure_ascii=False))
