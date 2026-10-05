"""Bounded CPU serialization/API probes. No training or checkpoint evaluation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
import torch
from torch import nn
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import generate
from scripts.evaluate import chat_prompt

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()
question = "1+1=?"
qbytes = list(question.encode("utf-8"))
prompt = [tok.bos_id, tok.user_id] + tok.encode(question) + [tok.eos_id, tok.assistant_id]
assert qbytes == [49, 43, 49, 61, 63]
assert tok.encode(question) == [b + 8 for b in qbytes] == [57, 51, 57, 69, 71]
assert prompt == [1, 3, 57, 51, 57, 69, 71, 2, 4] and len(prompt) == 9
assert prompt[-1] == prompt[len(prompt) - 1] == 4
assert tok.encode("2") == [58] and 58 not in prompt
x, y = render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": "2"}])
assert x.tolist() == prompt + [58]
assert len(x) == len(y) == 10 and y[8].item() == 58 and y[9].item() == 2
assert x[8].item() == 4 and (y[:8] == -100).all().item()

literal = "<assistant>"
literal_ids = tok.encode(literal)
assert tok.decode(literal_ids) == literal and all(i >= 8 for i in literal_ids)
assert tok.assistant_id not in literal_ids
literal_x, literal_y = render_chat([{"role": "user", "content": literal}, {"role": "assistant", "content": "2"}])
assert literal_x.tolist().count(tok.assistant_id) == 1
assert literal_x[1].item() == tok.user_id

history = [{"role": "user", "content": "0+1=?"}, {"role": "assistant", "content": "1"},
           {"role": "user", "content": question}]
multi_prompt = chat_prompt(history, tok)
multi_x, multi_y = render_chat(history + [{"role": "assistant", "content": "2"}])
assert multi_prompt == multi_x[:-1].tolist()
assert multi_prompt == [1, 3, 56, 51, 57, 69, 71, 2, 4, 57, 2, 3, 57, 51, 57, 69, 71, 2, 4]
assert len(multi_prompt) == 19 and multi_y[18].item() == 58
assert multi_x[8].item() == multi_x[18].item() == 4
assert multi_y[8].item() == 57 and multi_y[18].item() == 58
assert 58 not in multi_prompt

# A deliberately deterministic lookup table checks indices/shapes only, not semantics.
table = torch.arange(264 * 4, dtype=torch.float32).reshape(264, 4)
lookup = nn.Embedding.from_pretrained(table, freeze=True)
embedding_output = lookup(torch.tensor([prompt], dtype=torch.long))
assert list(embedding_output.shape) == [1, 9, 4]
assert embedding_output[0, -1].tolist() == [16.0, 17.0, 18.0, 19.0]

# A stub supplies synthetic logits so the real generate helper's chosen time axis is observable.
# The result below is not a model answering arithmetic; ID 58 is explicitly supplied by this stub.
class LastPositionStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=32)
        self.seen_shapes = []
    def forward(self, ids, cache=None):
        self.seen_shapes.append(list(ids.shape))
        logits = torch.zeros(ids.shape[0], ids.shape[1], tok.vocab_size)
        logits[:, :-1, 0] = 100
        logits[:, -1, 58] = 100
        self.logits_shape = list(logits.shape)
        return {"logits": logits, "cache": None}
stub = LastPositionStub()
output = generate(stub, torch.tensor([prompt]), max_new_tokens=1)
assert stub.seen_shapes == [[1, 9]] and stub.logits_shape == [1, 9, 264]
assert output.tolist() == [prompt + [58]] and stub.training is True

print(json.dumps({
    "device": "cpu", "python": sys.version, "torch": str(torch.__version__),
    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
    "question_utf8_bytes": qbytes, "question_byte_ids": tok.encode(question),
    "prompt": prompt, "prompt_token_count": len(prompt), "prompt_last_position": 8,
    "current_answer_content_id": 58, "answer_content_absent_from_prompt": 58 not in prompt,
    "training_x": x.tolist(), "training_y": y.tolist(),
    "literal_assistant_ids": literal_ids, "literal_is_role_metadata": False,
    "multi_turn_prompt": multi_prompt, "multi_turn_prompt_count": len(multi_prompt),
    "old_answer_prediction_position": 8, "current_answer_prediction_position": 18,
    "embedding_shape": list(embedding_output.shape), "last_embedding_lookup_row": 4,
    "synthetic_logits_shape": stub.logits_shape, "synthetic_generate_output": output.tolist(),
    "scope": "Exact prompt/render alignment, content-versus-role IDs, deterministic Embedding indexing, and generate time-axis control flow only. No model weights loaded, trained, or evaluated. No arithmetic capability conclusion."
}, ensure_ascii=False, indent=2))
