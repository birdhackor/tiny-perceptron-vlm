"""核對章首導覽、函式庫設定與共享狀態的短程式；不進行訓練。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.model import Block, ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


chapter_path = "course/chapters/14.md"
chapter = (ROOT / chapter_path).read_text()
intro_path = "docs/course-review-20261019/freeze-01/sources/chapter-14.md"
intro = (ROOT / intro_path).read_text()
matches = list(re.finditer(r"^## (14\.\d+) (.+)$", chapter, re.M))
sections = {}
for index, match in enumerate(matches):
    end = matches[index + 1].start() if index + 1 < len(matches) else len(chapter)
    sections[match[1]] = chapter[match.start():end]

paths = [
    intro_path,
    "docs/course-review-20261019/freeze-04.json",
    chapter_path,
    "course/chapters/04.md",
    "course/glossary.md",
    "tiny_perceptron/model.py",
    "tiny_perceptron/modern.py",
    "tiny_perceptron/attention.py",
]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-14/review_probe.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
    },
    "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "file_sha256": {path: digest(path) for path in paths},
    "frozen_intro_matches_current_preamble": intro == chapter[:matches[0].start()],
    "intro_has_figure": bool(re.search(r"!\[", intro)),
    "section_headings": [{"id": m[1], "title": m[2]} for m in matches],
    "navigation": [],
}
for label, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", intro):
    path, _, fragment = target.partition("#")
    resolved = (ROOT / chapter_path).parent / path
    content = resolved.read_text()
    result["navigation"].append({
        "label": label,
        "target": target,
        "exists": resolved.is_file(),
        "fragment_heading_exists": not fragment or bool(re.search(r"^## " + re.escape(fragment) + r"\s", content, re.M)),
    })

chapter4 = (ROOT / "course/chapters/04.md").read_text()
result["prerequisite_identifiers"] = {
    "block_in_linked_section": "class Block(nn.Module)" in chapter4.split("## 4.5 ", 1)[1].split("\n## ", 1)[0],
    "tinylm_in_linked_section": "class TinyLM(nn.Module)" in chapter4.split("## 4.6 ", 1)[1].split("\n## ", 1)[0],
}
result["library_scope_counterexample"] = {
    "intro_statement": next(p for p in intro.split("\n\n") if "每節先用" in p),
    "section_title": next(m[2] for m in matches if m[1] == "14.3"),
    "section_statement": next(p for p in sections["14.3"].split("\n\n") if "TinyLM也沒有" in p),
    "section_library_imports": re.findall(r"^from tiny_perceptron.*$", sections["14.3"], re.M),
    "model_config_fields": [f.name for f in fields(ModelConfig)],
}

torch.set_num_threads(1)
torch.manual_seed(42)
ids = torch.tensor([[1, 2, 3]])
result["library_variants"] = []
for name, overrides in [
    ("baseline", {}),
    ("rope", {"rotary": True}),
    ("rmsnorm", {"norm": "rms"}),
    ("relu2", {"activation": "relu2"}),
    ("swiglu", {"activation": "swiglu"}),
    ("tied", {"tied": True}),
]:
    model = TinyLM(ModelConfig(vocab_size=20, width=8, **overrides))
    with torch.no_grad():
        output = model(ids)["logits"]
    block = model.blocks[0]
    result["library_variants"].append({
        "name": name,
        "overrides": overrides,
        "logits_shape": list(output.shape),
        "finite_logits": bool(torch.isfinite(output).all()),
        "norm": type(block.norm1).__name__,
        "rotary": block.attention.rotary,
        "position_table": model.position is not None,
        "activation": block.ffn.activation,
        "gate_present": block.ffn.gate is not None,
        "input_output_same_parameter": model.output.weight is model.embedding.weight,
    })

shared = Block(ModelConfig(width=8))
state = torch.randn(1, 3, 8)
parameter_ids = [id(p) for p in shared.parameters()]
result["looped_block"] = []
looped_states = []
with torch.no_grad():
    for round_index in range(3):
        previous = state.clone()
        state, _, _ = shared(state)
        looped_states.append(state.clone())
        result["looped_block"].append({
            "round": round_index + 1,
            "same_parameters": parameter_ids == [id(p) for p in shared.parameters()],
            "state_shape": list(state.shape),
            "state_changed": not torch.equal(state, previous),
        })

direction = (looped_states[0][0, 0] - looped_states[1][0, 0]).double()
result["transformer_block_fixed_readout_counterexample"] = {
    "construction": "固定同一個共享 Block 與其第一、二輪狀態；令固定分類頭的第一類權重為兩狀態之差，第二類權重為零，目標為第一類。",
    "readout_weight": direction.tolist(),
    "logit_drop_from_first_to_second": direction.square().sum().item(),
    "scope": "人工設定固定讀出頭的存在性反例，無訓練；只檢查 Transformer 區塊共享本身不保證答案代價單調下降。",
    "rounds": [],
}
for round_index, state in enumerate(looped_states):
    logits = torch.stack((direction @ state[0, 0].double(), torch.tensor(0.0, dtype=torch.float64)))
    result["transformer_block_fixed_readout_counterexample"]["rounds"].append({
        "round": round_index + 1,
        "logits": logits.tolist(),
        "correct_class_probability": logits.softmax(-1)[0].item(),
        "cross_entropy": torch.nn.functional.cross_entropy(logits[None], torch.tensor([0])).item(),
    })

first, second = ["小貓", "追", "小狗"], ["小狗", "追", "小貓"]
result["word_order_derivation"] = {
    "same_word_multiset": Counter(first) == Counter(second),
    "ordinary_active_transitive_parse": [
        {"words": first, "subject": first[0], "predicate": first[1], "object": first[2]},
        {"words": second, "subject": second[0], "predicate": second[1], "object": second[2]},
    ],
    "interpretation": "依普通主動及物句讀法，前句由小貓追小狗，後句由小狗追小貓；沒有推論所有換序都改變語義。",
}
state = -1.0
result["nonmonotonic_shared_recurrence_counterexample"] = []
for round_index in range(4):
    state = -state
    probability = 1 / (1 + math.exp(-math.log(9) * state))
    result["nonmonotonic_shared_recurrence_counterexample"].append({
        "round": round_index + 1,
        "state": state,
        "correct_class_probability": probability,
        "cross_entropy": -math.log(probability),
    })
result["counterexample_scope"] = "手工共享純量遞迴，不是訓練後的 Transformer，也不是實測品質；僅反駁共享與反覆更新本身蘊含單調改善。"
print(json.dumps(result, ensure_ascii=False, indent=2))
