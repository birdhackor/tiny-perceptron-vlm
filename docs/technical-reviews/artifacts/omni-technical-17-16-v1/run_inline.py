"""執行 17.16 凍結稿內嵌程式及明定的兩種練習；僅保存必要小結果。"""

import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/17.16.md"
EXPECTED_SHA = "544e4a565b7f6a50c623c2bd784a1d8ce9d69deb1038b09bab4cda26615f886a"
raw = SNAPSHOT.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
body = raw.decode("utf-8")
code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
expected_stdout = re.findall(r"```text\n(.*?)```", body, re.S)[0]


def run(name, source):
    scope = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(source, str(SNAPSHOT) + "::" + name, "exec"), scope)
    result = {
        "case": name,
        "code_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "stdout": stream.getvalue(),
        "target_shape": list(scope["target"].shape),
        "target_dtype": str(scope["target"].dtype),
        "valid_dtype": str(scope["valid"].dtype),
        "distance": scope["distance"].detach().tolist(),
        "ids_dtype": str(scope["ids"].dtype),
        "z_e": scope["z_e"].detach().tolist(),
        "z_q": scope["z_q"].detach().tolist(),
        "z_st": scope["z_st"].detach().tolist(),
        "losses": {k: v.item() for k, v in scope["losses"].items()},
        "encoder_gradient": scope["encoder"].weight.grad.tolist(),
        "updated_parameters": [p.detach().tolist() for p in scope["params"]],
        "optimizer": {
            key: scope["optimizer"].param_groups[0][key]
            for key in ("lr", "momentum", "weight_decay", "maximize")
        },
    }
    return result


baseline = run("unmodified_inline", code)
assert baseline["stdout"] == expected_stdout, (baseline["stdout"], expected_stdout)
assert baseline["target_shape"] == [1, 2]
assert baseline["target_dtype"] == "torch.float32"
assert baseline["ids_dtype"] == "torch.int64"
assert baseline["z_st"] == [[1.0, 1.0]]

assert code.count('"承諾": 0.25 *') == 1
zero_beta = run("exercise_beta_zero", code.replace('"承諾": 0.25 *', '"承諾": 0.0 *'))
torch.testing.assert_close(
    torch.tensor(zero_beta["encoder_gradient"]),
    torch.tensor([[0.16, 0.72], [0.02, 0.09]]),
    rtol=0,
    atol=1e-7,
)
assert zero_beta["updated_parameters"][1:] == baseline["updated_parameters"][1:]

padding_results = []
for values in ("[-17.5, 2.75]", "[1e20, -1e20]"):
    assert code.count("[999.0, -999.0]") == 1
    changed = run("exercise_padding_" + values, code.replace("[999.0, -999.0]", values))
    assert changed["stdout"] == baseline["stdout"]
    assert changed["updated_parameters"] == baseline["updated_parameters"]
    padding_results.append({"finite_padding": values, "stdout_exact_match": True, "parameters_exact_match": True})

result = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-17-16-v1/run_inline.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": "cpu",
        "num_threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
    },
    "snapshot": str(SNAPSHOT.relative_to(ROOT)),
    "snapshot_sha256": EXPECTED_SHA,
    "baseline": baseline,
    "zero_beta": zero_beta,
    "padding_checks": padding_results,
    "result": "PASS: unmodified textbook stdout matches byte for byte; beta=0 gradient and unchanged codebook/decoder verified; two finite padding replacements leave stdout and parameters unchanged.",
    "scope": "One valid two-feature frame; FP32 CPU; one SGD update in each case. No sound training, quality, GPU performance, held-out evaluation, or Talker implementation is demonstrated.",
}
output = Path(__file__).with_name("inline-results.json")
with output.open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "result": result["result"]}, ensure_ascii=False))
