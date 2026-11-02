"""Independent G.1 checks: exact frozen prerequisite cells, then bounded CPU probes.

No fitting, model downloads, throughput measurements, or GPU work. G.1 itself
contains no fenced executable program; its linked program facts are checked on
the exact frozen teaching cells before comparison with library implementations.
"""

from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
from types import SimpleNamespace
import ast
import hashlib
import inspect
import io
import json
import platform
import re
import sys
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer, SPECIALS, IGNORE, render_chat
from tiny_perceptron.model import generate

ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"
CANONICAL = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


numbered = json.loads((ROOT / FREEZE).read_bytes())
canonical = json.loads((ROOT / CANONICAL).read_bytes())
assert digest((ROOT / CANONICAL).read_bytes()) == numbered["canonical_source_freeze"]["sha256"]
inputs = []


def section(identifier):
    freeze = numbered if identifier.startswith("W.") or identifier == "G.1" else canonical
    page = next(p for p in freeze["pages"] if p["page_id"] == identifier)
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    inputs.append({"id": identifier, "snapshot": page["snapshot"], "sha256": digest(raw)})
    return raw.decode("utf-8")


g1 = section("G.1")
assert "```" not in g1 and not re.search(r"!\[[^\]]*\]", g1)
assert next(p for p in numbered["pages"] if p["page_id"] == "G.1")["figures_sha256"] == {}
cell_results = {}


def run_cells(identifier):
    raw = section(identifier)
    blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
    ns = {"__name__": "g1_frozen_prerequisite"}
    output = io.StringIO()
    with redirect_stdout(output):
        for index, block in enumerate(blocks):
            exec(compile(block, f"{identifier}:frozen-cell-{index + 1}", "exec"), ns)
    cell_results[identifier] = {
        "cell_sha256s": [digest(block.encode()) for block in blocks],
        "stdout": output.getvalue(),
        "count": len(blocks),
    }
    return ns, blocks


torch.set_num_threads(1)
ns44, blocks44 = run_cells("4.4")
assert ns44["layer"].up.in_features == 4
assert ns44["layer"].up.out_features == 16
assert tuple(ns44["layer"].up(ns44["x"]).shape) == (1, 2, 16)
assert tuple(ns44["y"].shape) == (1, 2, 4)
assert ns44["same"] is True

ns66, blocks66 = run_cells("6.6")
assert ast.dump(ast.parse(blocks66[1]), include_attributes=False) == ast.dump(
    ast.parse(inspect.getsource(ByteTokenizer)), include_attributes=False
)
embedded_tok = ns66["ByteTokenizer"]()
tok = ByteTokenizer()
for text in ["貓看狗", "<assistant>", "<image>", "👍🏽"]:
    assert embedded_tok.encode(text) == tok.encode(text)
    assert embedded_tok.decode(embedded_tok.encode(text)) == text
    assert all(i >= 8 for i in embedded_tok.encode(text))
ns79, blocks79 = run_cells("7.9")
assert ns79["y"][-1].item() == tok.eos_id == 2
assert ns79["visible"] == tok.encode("A")
nsw3, _ = run_cells("W.3")
assert tuple(nsw3["scores"].shape) == (2, 3)
ns61, _ = run_cells("6.1")
ns162, _ = run_cells("16.2")
assert tuple(ns162["prefill"]["logits"].shape) == (1, 4, 264)
assert tuple(ns162["decode"]["logits"].shape) == (1, 1, 264)
nsb1, _ = run_cells("B.1")
parsed = json.loads(nsb1["text"])
assert isinstance(nsb1["text"], str) and isinstance(parsed, dict)
assert parsed == nsb1["request"]

vocab = {"狗": 0, "看": 1, "貓": 2}
ids = [vocab[c] for c in "貓看狗"]
assert ids == [2, 1, 0]
table = nn.Embedding(3, 2, dtype=torch.float64)
with torch.no_grad():
    table.weight.copy_(torch.tensor([[0., 0.], [1., 1.], [0.2, -0.1]], dtype=torch.float64))
looked_up = table(torch.tensor([2]))
assert looked_up.tolist() == [[0.2, -0.1]]
assert table.weight.requires_grad is True
shape_probe = table(torch.tensor([[2, 1, 0], [0, 1, 2]]))
assert tuple(shape_probe.shape) == (2, 3, 2)

linear = nn.Linear(2, 2, dtype=torch.float64)
no_bias = nn.Linear(2, 2, bias=False, dtype=torch.float64)
with torch.no_grad():
    linear.weight.copy_(torch.tensor([[2., 3.], [-1., 4.]], dtype=torch.float64))
    linear.bias.copy_(torch.tensor([5., -2.], dtype=torch.float64))
    no_bias.weight.copy_(linear.weight)
x = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=torch.float64)
y = linear(x)
assert torch.equal(y, x @ linear.weight.T + linear.bias)
assert torch.equal(no_bias(x), x @ linear.weight.T)
assert y.shape == x.shape and y[0, 0].tolist() == [13., 5.]
assert no_bias.bias is None
assert set(dict(linear.named_parameters())) == {"weight", "bias"}
assert set(linear.state_dict()) == {"weight", "bias"}
assert F.relu(torch.tensor([-1., 0., 2.])).tolist() == [0., 0., 2.]

chat_x, chat_y = render_chat([
    {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}
])
assert chat_x[0].item() == tok.bos_id
assert tok.bos_id not in chat_y.tolist()
assert [v for v in chat_y.tolist() if v != IGNORE] == [tok.encode("A")[0], tok.eos_id]


class FixedOutputs(nn.Module):
    """Fixture to isolate the course generation stop condition, without training."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=20)
        self.calls = 0

    def forward(self, sequence, cache=None):
        chosen = [tok.encode("A")[0], tok.eos_id, tok.encode("B")[0]][self.calls]
        self.calls += 1
        logits = torch.full((*sequence.shape, 264), -1000.)
        logits[:, -1, chosen] = 0.
        return {"logits": logits, "cache": None}


fixture = FixedOutputs()
generated = generate(fixture, torch.tensor([[tok.bos_id]]), max_new_tokens=5)
assert generated.tolist() == [[tok.bos_id, tok.encode("A")[0], tok.eos_id]]
assert fixture.calls == 2 and fixture.training is True
unicode_probes = {}
for text in ["貓", "看", "狗", "👍🏽"]:
    unicode_probes[text] = {
        "codepoints": [f"U+{ord(c):04X}" for c in text],
        "names": [unicodedata.name(c) for c in text],
        "utf8_bytes": list(text.encode("utf-8")),
        "byte_token_ids": tok.encode(text),
    }
assert len("👍🏽") == 2
assert [ord(c) for c in "👍🏽"] == [0x1F44D, 0x1F3FD]
assert all(len(c.encode("utf-8")) == len(tok.encode(c)) == 3 for c in "貓看狗")
try:
    "\ud800".encode("utf-8")
except UnicodeEncodeError:
    rejected_surrogate = True
else:
    raise AssertionError("UTF-8 must reject an isolated surrogate")

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": sys.version, "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version,
                    "unicode_database": unicodedata.unidata_version,
                    "platform": platform.platform(), "device": "cpu", "torch_threads": "1"},
    "input_sections": inputs,
    "frozen_cell_results": cell_results,
    "probes": {
        "toy_ids": ids, "toy_embedding_row": looked_up.tolist(),
        "BTD": list(shape_probe.shape), "byte_tokenizer_embedded_ast_matches_library": True,
        "special_ids": dict(zip(SPECIALS, range(8), strict=True)),
        "chat_input": chat_x.tolist(), "chat_targets": chat_y.tolist(),
        "single_sequence_generated": generated.tolist(), "generation_model_calls": fixture.calls,
        "generation_mode_restored": fixture.training,
        "linear_first_output": y[0, 0].tolist(), "linear_shape": list(y.shape),
        "linear_no_bias": no_bias.bias is None, "linear_parameters": list(linear.state_dict()),
        "relu": F.relu(torch.tensor([-1., 0., 2.])).tolist(),
        "unicode": unicode_probes, "isolated_surrogate_rejected": rejected_surrogate,
        "json_text_type": type(nsb1["text"]).__name__, "parsed_json_type": type(parsed).__name__,
    },
    "result": "All assertions passed; illustrative values and deterministic mechanisms only.",
    "limits": "No training, model-quality, performance, GPU, arbitrary-batch EOS or universal tokenizer round-trip claim is tested.",
}
output = Path(sys.argv[1])
with output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "output": str(output), "sha256": digest(output.read_bytes())}, ensure_ascii=False))
