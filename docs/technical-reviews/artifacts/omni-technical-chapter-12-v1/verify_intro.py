"""只核 chapter-12 導讀所需資料流；不訓練、不載權重、不構成逐節重審。"""

import ast
import contextlib
import copy
from datetime import UTC, datetime
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


chapter = (ROOT / "course/chapters/12.md").read_text()
freeze_path = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix12.json"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(row for row in freeze["pages"] if row["page_id"] == "chapter-12")
intro = (ROOT / page["snapshot"]).read_bytes()
assert digest(intro) == page["source_sha256"]
assert chapter[: chapter.index("\n## ") + 1].encode() == intro
assert page["figures_sha256"] == {}


def blocks(section):
    body = chapter.split("## " + section + " ", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"```python\n(.*?)\n```", body, flags=re.S)


def run_block(code, namespace=None):
    namespace = {} if namespace is None else namespace
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<current-course-inline>", "exec"), namespace)
    return namespace, output.getvalue().strip().splitlines()


class DropDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return node


def comparable(node):
    return ast.dump(DropDocstrings().visit(copy.deepcopy(node)), include_attributes=False)


torch.set_num_threads(1)
torch.manual_seed(0)
inline_definitions = [blocks("12.1")[1], blocks("12.6")[1], blocks("12.8")[1]]
module_tree = ast.parse((ROOT / "tiny_perceptron/multimodal.py").read_text())
module_nodes = {node.name: node for node in module_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
parity = {}
namespace = {"math": math, "torch": torch, "nn": nn, "F": F}
for code in inline_definitions:
    for node in ast.parse(code).body:
        parity[node.name] = comparable(node) == comparable(module_nodes[node.name])
    run_block(code, namespace)
assert all(parity.values())
wave = namespace["tone"]()[None]
spectrogram = namespace["log_mel"](wave)
encoder = namespace["AudioEncoder"](bands=16, width=8)
features = encoder(wave)
assert tuple(wave.shape) == (1, 1600)
assert tuple(spectrogram.shape) == (1, 16, 11)
assert tuple(features.shape) == (1, 11, 8)
assert torch.isfinite(spectrogram).all() and torch.isfinite(features).all()
assert sum(parameter.numel() for parameter in encoder.parameters()) > 0

baseline, baseline_lines = run_block(blocks("12.10")[0])
assert baseline["accuracy"] == 0.5
shape_namespace, shape_lines = run_block(blocks("12.15")[0])
av, av_lines = run_block(blocks("12.17")[0])
assert av["same_content"] and av["predictions"] == ["圓", "方"]
codec, codec_lines = run_block(blocks("12.18")[0])
assert codec["codes"].tolist() == [1, 2, 0]
assert not torch.equal(codec["waveform"], codec["reconstruction"])

# 只重算既有逐題紀錄的帳，不重新生成、不重跑既有 GPU 實驗。
audio_path = "docs/course-experiments/results/audio.json"
audio = json.loads((ROOT / audio_path).read_bytes())
results = audio["results"]
counts = {}
for mode in ("test", "blank", "shuffle"):
    samples = results[mode]["samples"]
    count = sum(row["generated"] == row["target"] for row in samples)
    assert count == results[mode]["correct"] and len(samples) == results[mode]["examples"]
    counts[mode] = {"n": len(samples), "correct_against_original_target": count}
donor_correct = 0
for index, row in enumerate(results["shuffle"]["samples"]):
    donor = (index + 7) % 14
    assert row["donor_row"] == donor
    donor_correct += row["generated"] == results["test"]["samples"][donor]["target"]
counts["shuffle"]["correct_against_donor_target"] = donor_correct
assert donor_correct == 11

# 真 ChatSession 的保存行為；假的 reply 只避免下載或載入任何模型。
from tiny_perceptron.selftrained.inference import ChatSession


class ProbeAssistant:
    def reply(self, messages, *, task, **kwargs):
        self.received = copy.deepcopy(messages)
        return {"messages": copy.deepcopy(messages) + [{"role": "assistant", "content": "保存測試"}]}


assistant = ProbeAssistant()
session = ChatSession(assistant)
session.ask("請用兩點回答")
audio_content = [{"type": "audio", "audio": "probe-no-file-opened.wav"}]
session.ask(audio_content, task="voice")
session.ask("改成一句")
assert [message["role"] for message in session.messages] == ["user", "assistant"] * 3
assert session.messages[0]["content"] == "請用兩點回答"
assert session.messages[2]["content"] == audio_content
assert session.messages[4]["content"] == "改成一句"

selections = {
    "course/chapters/12.md": [
        (9, 13), (31, 36), (252, 265), (319, 322), (325, 338), (344, 366),
        (390, 397), (402, 405), (411, 421), (464, 506), (643, 651), (670, 676),
        (688, 692), (696, 700), (702, 713), (717, 725), (728, 732),
        (738, 738), (742, 752), (763, 763), (777, 779), (783, 785),
        (787, 881), (883, 894), (899, 909), (913, 955),
    ],
    "course/chapters/19.md": [(1438, 1444), (1448, 1452), (1674, 1678), (1681, 1687), (1691, 1693), (1880, 1886)],
    "course/glossary.md": [(121, 129)],
    "tiny_perceptron/multimodal.py": [(43, 84), (108, 144)],
    "tiny_perceptron/selftrained/model.py": [(247, 284), (295, 306), (333, 355)],
    "tiny_perceptron/selftrained/dataset.py": [(125, 136)],
    "tiny_perceptron/selftrained/inference.py": [(280, 286), (313, 342)],
    "scripts/course_experiments/modalities.py": [(195, 216), (287, 334), (365, 405), (884, 914)],
}
pins = []
for path, ranges in selections.items():
    raw = (ROOT / path).read_bytes()
    lines = raw.splitlines(keepends=True)
    for start, end in ranges:
        selected = b"".join(lines[start - 1 : end])
        pins.append({"path": path, "start_line": start, "end_line_inclusive": end,
                     "sha256": digest(selected), "text": selected.decode()})
pins_record = {
    "created_at": datetime.now(UTC).isoformat(),
    "target": {"page_id": "chapter-12", "snapshot": page["snapshot"], "sha256": digest(intro),
               "freeze": freeze_path, "freeze_sha256": digest((ROOT / freeze_path).read_bytes()),
               "figure_sha256": {}, "has_code_fences": b"```" in intro},
    "selection_rule": "只保留本導讀之高層資料流、任務區分、銜接主張所需精確片段；不表示逐節審閱、未檢查前置圖。",
    "slices": pins,
}
save_new(OUT / "source-pins-v1.json", pins_record)
execution = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-12-v1/verify_intro.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "dtype": "float32 (12.17/12.18 explicitly float64)", "threads": "1"},
    "target_freeze_match": True,
    "inline_definition_ast_parity_ignoring_docstrings": parity,
    "inline_flow_probe": {"input": "current inline tone(440 Hz, 0.1 s, 16000 Hz), batch=1, seed=0",
                          "waveform_shape": list(wave.shape), "spectrum_shape": list(spectrogram.shape),
                          "encoder_shape": list(features.shape), "finite": True,
                          "learnable_parameters": sum(parameter.numel() for parameter in encoder.parameters())},
    "inline_12_10_baseline_stdout": baseline_lines,
    "inline_12_15_shape_only_stdout": shape_lines,
    "inline_12_17_analytic_time_rule_stdout": av_lines,
    "inline_12_18_fixed_codebook_stdout": codec_lines,
    "archived_audio_result_recount": {"path": audio_path, "sha256": digest((ROOT / audio_path).read_bytes()),
                                      "result_declares_device": audio["device"], "counts": counts},
    "shared_history_bookkeeping": {"actual_class": "tiny_perceptron.selftrained.inference.ChatSession",
                                   "reply": "ProbeAssistant, no model/file access", "message_count": len(session.messages),
                                   "typed_instructions_and_original_audio_content_retained": True},
    "scope": "本次只有 CPU forward、固定規則與既有 JSON 帳務重算；無 optimizer step、訓練、權重載入、真人錄音推論、ASR、GPU、模型品質或速度驗收。12.15 的隨機輸入不是已會理解需求；12.17 是解析規則；12.18 是有損查表重建。",
    "result": "all assertions passed",
}
save_new(OUT / "execution-v1.json", execution)
print(json.dumps({"result": execution["result"], "environment": execution["environment"],
                  "outputs": {name: digest((OUT / name).read_bytes()) for name in ("source-pins-v1.json", "execution-v1.json")}}, ensure_ascii=False))
