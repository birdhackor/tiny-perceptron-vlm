"""Bounded masking probe; no model weights, forward/generate, or training."""
from pathlib import Path
import json
import sys
import math
import importlib.metadata
import torch
from tokenizers import Tokenizer
from jinja2 import Template
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss
from tiny_perceptron.natural_assistant import encode_training_row

print(json.dumps({"python": sys.version, "torch": torch.__version__, "device": "cpu",
                  "cuda_available": torch.cuda.is_available(),
                  "tokenizers": importlib.metadata.version("tokenizers")}, indent=2))
tok = ByteTokenizer()
for answer, expected in [("A", [73, 2]), ("AB", [73, 74, 2]), ("答", tok.encode("答") + [2])]:
    x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": answer}])
    active = y != -100
    first = active.nonzero()[0].item()
    assert y[active].tolist() == expected
    assert x[first].item() == tok.assistant_id and first == 4
    assert tok.encode("Q")[0] in x.tolist() and -100 not in x.tolist()
    print(json.dumps({"answer": answer, "X": x.tolist(), "Y": y.tolist(),
                      "input_positions": len(x), "active_targets": active.sum().item(),
                      "first": first, "target_ids": y[active].tolist()}, ensure_ascii=False))

# Uniform logits: ignored positions have exactly zero direct logits gradient.
x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
logits = torch.zeros(1, len(x), tok.vocab_size, dtype=torch.float64, requires_grad=True)
loss = masked_loss(logits, y[None, :])
loss.backward()
assert math.isclose(loss.item(), math.log(264), abs_tol=1e-12)
assert torch.count_nonzero(logits.grad[0, y == -100]).item() == 0
assert torch.count_nonzero(logits.grad[0, y != -100]).item() > 0
print(json.dumps({"uniform_loss": loss.item(), "expected_log264": math.log(264),
                  "ignored_logits_gradient_nonzero": 0, "scored_positions": 2}))

# Use original fixed template + native tokenizer for a real text-only encoding.
# This adapter is explicit: it does not replicate image preprocessing.
research = Path("outputs/natural-v4/factual-research/20.7")
template = Template(json.loads((research / "qwen-template.json").read_text())["chat_template"])
qtok = Tokenizer.from_file(str(research / "qwen-tokenizer.json"))
class TextProcessor:
    def __init__(self, bad_prefix=False, padded=False):
        self.bad_prefix, self.padded = bad_prefix, padded
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return rendered + "x" if self.bad_prefix and add_generation_prompt else rendered
    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        ids = qtok.encode(text[0], add_special_tokens=False).ids
        attention = [1] * len(ids)
        if self.padded:
            ids += [151643]; attention += [0]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attention])}

row = {"id": "own-text-control", "system": "S", "history": [
    {"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"}],
    "user": "Q", "answer": "AB"}
batch = encode_training_row(TextProcessor(), row, Path.cwd(), max_tokens=2048)
labels, ids = batch["labels"][0], batch["input_ids"][0]
first = (labels != -100).nonzero()[0].item()
assert labels[labels != -100].tolist() == qtok.encode("AB<|im_end|>\n", add_special_tokens=False).ids
assert qtok.decode(labels[labels != -100].tolist(), skip_special_tokens=False) == "AB<|im_end|>\n"
assert qtok.decode(ids[:first].tolist(), skip_special_tokens=False).endswith("<|im_start|>assistant\n")
print(json.dumps({"qwen_text_only_input_positions": len(ids), "first_unshifted_label": first,
                  "first_predictor_position_after_internal_shift": first-1,
                  "supervised_ids": labels[labels != -100].tolist(),
                  "decoded_full": qtok.decode(ids.tolist(), skip_special_tokens=False),
                  "history_assistant_ignored": True}, ensure_ascii=False))
for description, processor, maximum in [
    ("prefix_mismatch", TextProcessor(bad_prefix=True), 2048),
    ("over_length", TextProcessor(), len(ids)-1),
]:
    try:
        encode_training_row(processor, row, Path.cwd(), max_tokens=maximum)
    except ValueError as error:
        print(json.dumps({"guard": description, "error": str(error)}))
    else:
        raise AssertionError(description + " was not refused")

# Attention padding is ignored even when following a valid answer suffix.
class FullOnlyPadding(TextProcessor):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.padded = not add_generation_prompt
        return super().apply_chat_template(messages, tokenize, add_generation_prompt)
pb = encode_training_row(FullOnlyPadding(), row, Path.cwd())
assert pb["labels"][0, -1].item() == -100
assert int((pb["labels"][:, 1:] != -100).sum()) == 3
print(json.dumps({"padding_label": -100, "shifted_nonpadding_targets": 3, "guards_passed": True}))
