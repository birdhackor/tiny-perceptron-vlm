"""Read-only, bounded CPU verification of the exact frozen 16.16 code."""

import contextlib
import hashlib
import io
import itertools
import json
import platform
import re
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix10.json"
PAGE = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == "16.16")
SOURCE = ROOT / PAGE["snapshot"]
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == PAGE["source_sha256"]
text = raw.decode()
embedded = re.findall(r"```python\n(.*?)```", text, re.S)
assert len(embedded) == 1
assert (HERE / "embedded.py").read_text() == embedded[0]
chapter = (ROOT / PAGE["source"]).read_text()
section = re.search(r"^## 16\.16 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
assert section == text
notebook = json.loads((ROOT / PAGE["notebook"]).read_bytes())
notebook_code = ["".join(c["source"]).strip() for c in notebook["cells"] if c["cell_type"] == "code"]
assert embedded[0].strip() in notebook_code
assert hashlib.sha256((ROOT / PAGE["notebook"]).read_bytes()).hexdigest() == PAGE["notebook_sha256"]
for path, digest in PAGE["figures_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    ns = runpy.run_path(str(HERE / "embedded.py"))
expected_stdout = re.findall(r"```text\n(.*?)```", text, re.S)[0]
assert stdout.getvalue().strip() == expected_stdout.strip()
Decoder, table, kernel, hop = (ns[k] for k in ("Decoder", "table", "kernel", "hop"))
ids = ns["ids"]

single = Decoder()
single_values = torch.cat([single.push(ids[i:i + 1]) for i in range(len(ids))])
assert torch.equal(single_values, ns["streamed"])
reset_before_second = torch.cat((Decoder().push(ids[:1]), Decoder().push(ids[1:])))
changed_indices = torch.nonzero(reset_before_second != ns["streamed"]).flatten().tolist()
assert changed_indices == [2, 3]

# A finite cross-check of every nonempty partition of every length-four code string.
# The general causal argument is the explicit recurrence, not this finite enumeration.
partition_checks = 0
for values in itertools.product(range(len(table)), repeat=4):
    inputs = torch.tensor(values)
    full = torch.nn.functional.conv_transpose1d(
        table[inputs][None, None], kernel[None, None], stride=hop
    ).flatten()[:hop * len(inputs)]
    for cuts in itertools.product((False, True), repeat=3):
        bounds = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [4]
        stream = Decoder()
        output = torch.cat([stream.push(inputs[a:b]) for a, b in zip(bounds, bounds[1:])])
        assert torch.equal(output, full)
        stream.finish()
        assert torch.equal(stream.tail, torch.zeros(2))
        partition_checks += 1

base = ns["streamed"]
future = ns["future_changed"]
assert torch.equal(base[:4], future[:4])
normalization = {}
for name, fn in (
    ("subtract_full_mean", lambda x: x - x.mean()),
    ("divide_full_std", lambda x: x / x.std(unbiased=False)),
):
    normalized_base, normalized_future = fn(base), fn(future)
    difference = (normalized_base[:4] - normalized_future[:4]).abs().max().item()
    assert difference > 0
    normalization[name] = {
        "base_first_four": normalized_base[:4].tolist(),
        "changed_future_first_four": normalized_future[:4].tolist(),
        "prefix_max_difference": difference,
    }

untrimmed = torch.nn.functional.conv_transpose1d(
    table[ids][None, None], kernel[None, None], stride=hop
).flatten()
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-16-16-v1/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
        "table_dtype": str(table.dtype),
        "ids_dtype": str(ids.dtype),
    },
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "embedded_sha256": hashlib.sha256((HERE / "embedded.py").read_bytes()).hexdigest(),
    "snapshot_equals_current_section": True,
    "notebook_contains_exact_frozen_code": True,
    "embedded_stdout": stdout.getvalue(),
    "embedded_stdout_matches_all_published_lines": True,
    "untrimmed_reference": untrimmed.tolist(),
    "untrimmed_length": untrimmed.numel(),
    "cropped_length": base.numel(),
    "finish_tail": ns["decoder"].tail.tolist(),
    "single_code_chunks": single_values.tolist(),
    "reset_before_second_code": reset_before_second.tolist(),
    "reset_before_second_changed_indices": changed_indices,
    "partition_comparisons": partition_checks,
    "partition_enumeration": "4^4 code strings x 2^3 nonempty contiguous partitions = 2048",
    "full_sequence_normalization_counterexamples": normalization,
    "result": "PASS: exact embedded outputs, state/crop/causal controls, exercise and finite partition checks",
    "limits": "No player, sampling rate, latency measurement, quality evaluation, model training or GPU execution. Finite toy checks are not a proof about arbitrary neural codecs.",
}
destination = HERE / "execution.json"
with destination.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
