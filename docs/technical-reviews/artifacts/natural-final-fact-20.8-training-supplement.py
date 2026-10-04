"""Pure CPU checks of architecture arithmetic, proxy limitations and immutable source reuse."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from tiny_perceptron import natural_assistant as core

train = json.loads((ROOT / "docs/natural-assistant/evidence/train/result.json").read_text())
config = json.loads((OUT / "natural-final-fact-20.8-base_model-config.json").read_text())["text_config"]
layers = sorted({int(re.search(r"layers\.(\d+)", name)[1]) for name in train["optimizer_parameter_names"]})
targets = sorted({re.search(r"self_attn\.(q_proj|v_proj)", name)[1] for name in train["optimizer_parameter_names"]})
assert layers == list(range(28)) and targets == ["q_proj", "v_proj"]
h, q, v, rank = config["hidden_size"], config["num_attention_heads"] * config["head_dim"], config["num_key_value_heads"] * config["head_dim"], 8
params = 28 * (rank * (h + q) + rank * (h + v))
assert params == 1605632 and 28 * 2 * 2 == 112
proxy = core.score_output({"answer": "dog", "references": {"kind": "facts", "fact_groups": [["dog"]]}}, "A dog. I also claim an unsupported event happened.")
manual = core.score_output({"answer": "anything", "references": {"kind": "manual"}}, "anything")
assert proxy["passed"] is True and manual["passed"] is None
partial_proxy = {"task": "example", "truncated": True, "ended_with_eos": False, "score": {"passed": True}}
summary = core.summarize([partial_proxy])
assert summary["example"]["passed"] == 1
# Actual production proxy deliberately counts score independently of EOS;
# guide correctly requires full answer/truncation review rather than using it as quality.
asr_cer = core.asr_cer_metrics("Ａ B。", "A B。")
assert asr_cer["raw_errors"] == 1 and asr_cer["errors"] == 0
result = {
    "scope": "Pure arithmetic and scalar text proxy functions; no model load or forward",
    "source_sha256": hashlib.sha256((ROOT / "tiny_perceptron/natural_assistant.py").read_bytes()).hexdigest(),
    "layers": layers,
    "targets": targets,
    "shapes_from_pinned_original_config": {"hidden": h, "q_output": q, "v_output": v, "rank": rank},
    "derivation": "28 * (8*(2048+2048) + 8*(2048+1024)) = 1605632; 28*2 projections*2 A/B tensors = 112",
    "trainable_parameters": params,
    "proxy_example_with_unsupported_addition": proxy,
    "manual_score": manual,
    "truncated_score_counted_by_proxy_summary": summary,
    "raw_vs_normalized_CER_example": asr_cer,
    "conclusion": "EOS and completed record counts are not semantic success; guide instructs whole-answer review, preserving unsupported additions/truncation and actual ASR errors",
}
(OUT / "natural-final-fact-20.8-training-supplement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"trainable_parameters": params, "layers": len(layers), "proxy_is_not_semantic_quality": True}))
