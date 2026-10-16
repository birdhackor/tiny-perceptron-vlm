"""G.4 術語核對：未訓練 CPU toy model，只檢查生成長度上限。"""
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM, generate

torch.set_num_threads(1)
model = TinyLM(ModelConfig(vocab_size=8, width=8, layers=1, heads=1, max_length=8))
with torch.no_grad():
    model.output.weight.zero_()  # greedy argmax=0，與預設 EOS=2 不同。
cases = []
for prompt_length, maximum, eos in [(3, 2, 2), (3, 0, 2), (7, 4, 2), (3, 4, 0)]:
    prompt = torch.ones((1, prompt_length), dtype=torch.long)
    output = generate(model, prompt, max_new_tokens=maximum, eos_id=eos)
    added = output.shape[1] - prompt_length
    assert added <= maximum
    expected = 1 if eos == 0 else min(maximum, 8 - prompt_length)
    assert added == expected
    assert torch.equal(output[:, :prompt_length], prompt)
    cases.append({"prompt_tokens": prompt_length, "max_new_tokens": maximum, "eos_id": eos,
                  "output_tokens": output.shape[1], "added_tokens": added, "expected_added": expected,
                  "prefix_preserved": True})
result = {"command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/G.4-max-new-tokens.py",
          "environment": {"python": platform.python_version(), "torch": torch.__version__,
                          "device": "cpu", "platform": platform.platform()},
          "configuration": model.description()["config"], "cases": cases,
          "result": "4 cases passed; max_new_tokens bounds only newly appended tokens; context limit and EOS may stop earlier",
          "scope": "No training, quality, speed, GPU, or multi-platform conclusion."}
path = ROOT / "docs/technical-reviews/artifacts/revision-20261016/G.4-max-new-tokens.json"
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
