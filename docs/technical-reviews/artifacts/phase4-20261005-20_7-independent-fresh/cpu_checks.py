"""Independent, bounded CPU checks of lesson 20.7. No optimizer or downloaded model.

The template processor below is deliberately a controlled byte-token surrogate;
only the Jinja template is Qwen's original. Its counts are not Qwen counts.
"""
import ast
import json
from pathlib import Path
from typing import Optional

import jinja2
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss, loss_sum
from tiny_perceptron.natural_assistant import encode_training_row, messages_for, encode_messages

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(207)
assert torch.version.cuda is None and not torch.cuda.is_available()


def emit(name, **values):
    print(json.dumps(dict(check=name, **values), ensure_ascii=False))


tok = ByteTokenizer()


def pair(question="Q", answer="A"):
    return render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])


for question, answer, expected_first, expected_count in [("Q", "A", 4, 2), ("Q", "AB", 4, 3), ("QQ", "AB", 5, 3)]:
    x, y = pair(question, answer)
    active = y != -100
    positions = active.nonzero().flatten().tolist()
    assert positions[0] == expected_first and int(active.sum()) == expected_count
    assert x[positions[0]] == tok.assistant_id
    assert y[active].tolist() == tok.encode(answer) + [tok.eos_id]
    assert (x >= 0).all() and (x < tok.vocab_size).all()
    assert ((y[active] >= 0) & (y[active] < tok.vocab_size)).all()
    assert torch.equal(x[2:2 + len(tok.encode(question))], torch.tensor(tok.encode(question)))
    emit("byte-alignment", question=question, answer=answer, x=x.tolist(), y=y.tolist(), active_positions=positions, active_count=int(active.sum()))

x, y = pair()
logits = torch.zeros(1, len(x), tok.vocab_size, requires_grad=True)
total, count = loss_sum(logits, y[None])
mean = masked_loss(logits, y[None])
expected = torch.log(torch.tensor(float(tok.vocab_size)))
assert int(count) == 2 and torch.allclose(mean, expected, atol=1e-6, rtol=0)
mean.backward()
assert logits.grad[0, :4].eq(0).all() and logits.grad[0, 4:].abs().sum() > 0
changed = logits.detach().clone()
changed[0, :4] = torch.randn_like(changed[0, :4]) * 100
assert torch.equal(masked_loss(changed, y[None]), mean.detach())
emit("masked-loss", sum=float(total.detach()), denominator=int(count), mean=float(mean.detach()), expected_log_264=float(expected), ignored_logit_gradient_is_zero=True, ignored_logits_do_not_change_loss=True)

model = TinyLM(ModelConfig(width=16, heads=2, max_length=16)).eval()
original = model(x[None])["logits"]
x_answer, _ = pair(answer="B")
x_prompt, _ = pair(question="R")
answer_change = model(x_answer[None])["logits"]
prompt_change = model(x_prompt[None])["logits"]
assert torch.equal(original[:, :5], answer_change[:, :5])
assert not torch.equal(original[:, 4], prompt_change[:, 4])
embeddings = model.embedding(x[None]).detach().requires_grad_(True)
model(embeddings=embeddings)["logits"][0, 4, tok.encode("A")[0]].backward()
assert embeddings.grad[0, 2].abs().sum() > 0 and embeddings.grad[0, 5].eq(0).all()
emit("causal-context", logits_shape=list(original.shape), answer_change_cannot_affect_positions_0_through_4=True, prompt_Q_to_R_changes_first_answer_logits=True, question_embedding_gradient_nonzero=True, future_answer_embedding_gradient_zero=True, optimizer_steps=0)

cropped_x, cropped_y, valid = pad_batch([pair(answer="AB")], max_length=5)
assert int((cropped_y != -100).sum()) == 1 and cropped_y[0, 4] == tok.encode("A")[0]
assert tok.eos_id not in cropped_y[cropped_y != -100]
emit("toy-truncation-changes-target", max_length=5, retained_targets=cropped_y[cropped_y != -100].tolist(), effective_target_count=1, eos_lost=True)

# Execute just these two exact upstream functions; do not import Transformers
# or initialize a model. This is an independent test of their loss contract.
official_path = HERE / "sources/hf-loss-v4.57.6.py"
tree = ast.parse(official_path.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"fixed_cross_entropy", "ForCausalLMLoss"}]
assert len(nodes) == 2
namespace = {"torch": torch, "nn": nn, "Optional": Optional}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(official_path), "exec"), namespace)
official_loss = namespace["ForCausalLMLoss"]
full_ids = torch.cat((x, torch.tensor([tok.eos_id])))[None]
labels = full_ids.clone()
labels[:, :5] = -100
full_logits = torch.randn(1, 7, tok.vocab_size)
hf_loss = official_loss(full_logits, labels, tok.vocab_size)
same_position_loss = masked_loss(full_logits[:, :-1], y[None])
assert torch.allclose(hf_loss, same_position_loss, atol=1e-6, rtol=0)
assert int((labels[:, 1:] != -100).sum()) == 2
emit("one-shift-conventions", unshifted_labels=labels.tolist(), hf_loss=float(hf_loss), same_position_tiny_loss=float(same_position_loss), effective_supervised_count=2, tolerance_abs=1e-6)

template_data = json.loads((HERE / "sources/qwen3vl-chat-template-89644892.json").read_text())
template = jinja2.Environment().from_string(template_data["chat_template"])
SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<|vision_start|>": 3, "<|image_pad|>": 4, "<|vision_end|>": 5}


class ControlledProcessor:
    """Actual official template + explicit toy tokenizer/image placeholder expansion."""
    def __init__(self, *, bad_prefix=False, empty_suffix=False, pad_suffix=False):
        self.calls = []
        self.bad_prefix, self.empty_suffix, self.pad_suffix = bad_prefix, empty_suffix, pad_suffix

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        value = template.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=None, add_vision_id=False)
        if self.empty_suffix and not add_generation_prompt:
            value = template.render(messages=messages[:-1], add_generation_prompt=True, tools=None, add_vision_id=False)
        return value

    def __call__(self, *, text, return_tensors, images=None):
        assert return_tensors == "pt" and len(text) == 1
        value = text[0].replace("<|image_pad|>", "<|image_pad|>" * 3)
        ids = []
        while value:
            matched = next((token for token in SPECIAL if value.startswith(token)), None)
            if matched:
                ids.append(SPECIAL[matched]); value = value[len(matched):]
            else:
                c, value = value[0], value[1:]
                ids.extend(b + 8 for b in c.encode())
        self.calls.append({"text": text[0], "ids": ids.copy(), "image_count": len(images or [])})
        if self.bad_prefix and len(self.calls) == 2:
            ids[0] = 99
        mask = [1] * len(ids)
        if self.pad_suffix:
            ids.append(0); mask.append(0)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


asset_dir = HERE / "toy-assets"
asset_dir.mkdir(exist_ok=True)
Image.new("RGB", (2, 2), "red").save(asset_dir / "current.png")
Image.new("RGB", (2, 2), "blue").save(asset_dir / "history.png")
row = {"id": "independent-contract-only", "system": "S", "history": [
    {"role": "user", "content": [{"type": "image", "image": "history.png"}, {"type": "text", "text": "old Q"}]},
    {"role": "assistant", "content": "old A"}], "image": "current.png", "user": "Q", "answer": "AB"}
processor = ControlledProcessor()
batch = encode_training_row(processor, row, asset_dir, max_tokens=1000)
prefix = len(processor.calls[0]["ids"])
assert batch["input_ids"].ndim == 2 and batch["input_ids"].shape[0] == 1
assert batch["labels"][0, :prefix].eq(-100).all()
active_suffix = batch["labels"][0, prefix:].tolist()
assert active_suffix == [ord("A") + 8, ord("B") + 8, SPECIAL["<|im_end|>"], ord("\n") + 8]
assert all(call["image_count"] == 2 for call in processor.calls)
assert "old A" in processor.calls[1]["text"] and "old Q" in processor.calls[1]["text"]
assert batch["attention_mask"].eq(1).all()
length = int(batch["input_ids"].shape[-1])
encode_training_row(ControlledProcessor(), row, asset_dir, max_tokens=length)
emit("real-helper-controlled-processor", batch_shape=list(batch["input_ids"].shape), prefix_length=prefix, supervised_suffix_ids=active_suffix, supervised_count=len(active_suffix), history_and_both_images_preserved=True, prompt_input_attention_all_one=True, exact_length_budget_accepted=True, tokenizer_scope="toy byte tokenizer; these counts are not Qwen token counts", official_template_supervises_answer_plus_im_end_plus_newline=True)

for name, processor, budget, expected in [
    ("over-budget", ControlledProcessor(), length - 1, "refusing silent multimodal truncation"),
    ("wrong-prefix", ControlledProcessor(bad_prefix=True), 1000, "not an exact prefix"),
    ("no-supervised-suffix", ControlledProcessor(empty_suffix=True), 1000, "at least one supervised token"),
]:
    try:
        encode_training_row(processor, row, asset_dir, max_tokens=budget)
    except ValueError as exc:
        assert expected in str(exc)
        emit("real-helper-rejection", case=name, error=str(exc))
    else:
        raise AssertionError(name + " should have been rejected")

# Padding is checked with a controlled numeric input, independently of template
# prefix behavior, which is already tested above.
class PaddingProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "prompt" if add_generation_prompt else "full"
    def __call__(self, *, text, return_tensors, images=None):
        ids = [1, 3, 89, 2, 4] if text == ["prompt"] else [1, 3, 89, 2, 4, 73, 2, 0]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[1] * (len(ids) - (text == ["full"])) + ([0] if text == ["full"] else [])])}

padding = encode_training_row(PaddingProcessor(), {"id": "padding", "user": "Q", "answer": "A"}, asset_dir, max_tokens=8)
assert padding["labels"].tolist() == [[-100, -100, -100, -100, -100, 73, 2, -100]]
emit("real-helper-padding", labels=padding["labels"].tolist(), attention_mask=padding["attention_mask"].tolist(), supervised_count=int((padding["labels"][:, 1:] != -100).sum()))

emit("complete", device="cpu", optimizer_steps=0, model_or_training_data_downloads=0)
