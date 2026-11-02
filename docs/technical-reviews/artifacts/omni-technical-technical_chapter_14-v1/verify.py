"""chapter-14 導讀的有界 CPU 核對；不訓練、不下載模型。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM  # noqa: E402
from tiny_perceptron.modern import RMSNorm  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(text, lesson):
    start = re.search(r"^## " + re.escape(lesson) + r" ", text, re.M).start()
    following = re.search(r"^## ", text[start + 1 :], re.M)
    end = start + 1 + following.start() if following else len(text)
    return text[start:end]


freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads(freeze_path.read_bytes())
pages = {page["page_id"]: page for page in freeze["pages"]}
observations = {
    "created_at": datetime.now(UTC).isoformat(),
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
    },
    "freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": digest(freeze_path)},
}
page = pages["chapter-14"]
chapter_text = (ROOT / page["source"]).read_text()
intro = chapter_text[: re.search(r"^## ", chapter_text, re.M).start()]
snapshot = ROOT / page["snapshot"]
assert intro.encode() == snapshot.read_bytes()
assert digest(snapshot) == page["source_sha256"]
assert page["figures_sha256"] == {}
assert not re.search(r"!\[|```", intro)
observations["introduction"] = {
    "sha256": digest(snapshot),
    "matches_current": True,
    "figures": {},
    "embedded_code_blocks": 0,
}
observations["prerequisites"] = []
for lesson in ["4.5", "4.6"]:
    prerequisite = pages[lesson]
    raw = (ROOT / prerequisite["snapshot"]).read_text()
    current = section((ROOT / prerequisite["source"]).read_text(), lesson)
    assert current == raw
    assert hashlib.sha256(raw.encode()).hexdigest() == prerequisite["source_sha256"]
    blocks = re.findall(r"```python\n(.*?)\n```", raw, re.S)
    namespace = {"__name__": "__chapter14_prerequisite_check__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, block in enumerate(blocks, 1):
            exec(compile(block, prerequisite["snapshot"] + f":python-block-{index}", "exec"), namespace)
    observations["prerequisites"].append({
        "lesson_id": lesson,
        "snapshot": prerequisite["snapshot"],
        "sha256": prerequisite["source_sha256"],
        "matches_current_section": True,
        "executed_exact_python_blocks": len(blocks),
        "stdout": output.getvalue(),
    })

torch.manual_seed(42)
observations["library_settings"] = []
for change in [{"rotary": True}, {"norm": "rms"}, {"activation": "swiglu"}, {"tied": True}]:
    config = ModelConfig(vocab_size=20, width=8, max_length=16, **change)
    model = TinyLM(config).eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]]))["logits"]
    assert tuple(logits.shape) == (1, 3, 20) and torch.isfinite(logits).all()
    if "rotary" in change:
        assert model.position is None and model.blocks[0].attention.rotary
    elif "norm" in change:
        assert isinstance(model.final_norm, RMSNorm) and isinstance(model.blocks[0].norm1, RMSNorm)
    elif "activation" in change:
        assert model.blocks[0].ffn.gate is not None
    else:
        assert model.output.weight is model.embedding.weight
    observations["library_settings"].append({"change": change, "logits_shape": list(logits.shape), "finite": True})

first, second = ["小貓", "追", "小狗"], ["小狗", "追", "小貓"]
assert Counter(first) == Counter(second) and first[0] != second[0]
observations["sentence_tokens"] = {"first": first, "second": second, "same_multiset": True}

# 相同共享運算 R(s)=-s，head logits=[s,0]、答案類別0。
# 第2輪到第3輪 loss 增加，是無一般單調保證的明確反例。
state, losses = 1.0, []
for iteration in range(1, 4):
    state = -state
    losses.append({"iteration": iteration, "state": state, "target_0_cross_entropy": math.log1p(math.exp(-state))})
assert losses[2]["target_0_cross_entropy"] > losses[1]["target_0_cross_entropy"]
observations["shared_update_counterexample"] = {"update": "R(s)=-s", "head": "logits=[s,0], target=0", "initial_state": 1.0, "observed": losses, "scope": "純量共享運算反例；不是教材Transformer或訓練品質測試"}
headings = re.findall(r"^## (14\.\d+) (.+)$", chapter_text, re.M)
observations["roadmap_headings_only"] = [{"lesson_id": key, "title": title} for key, title in headings]
assert headings[-1][0] == "14.19"
observations["limits"] = "只為章導讀核對明確前提、設定存在與導覽範圍；不替14.x小節作獨立簽核。無訓練、GPU、權重下載、速度或語言品質實測。"
print(json.dumps(observations, ensure_ascii=False, indent=2))
