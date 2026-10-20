"""核對填充位置的合法索引、查表順序與既存證據的完整性。"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM

BASE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_7-7/technical-7-7-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


original = json.loads(ORIGINAL.read_bytes())
assert sha(ORIGINAL.read_bytes()) == "cc6d73b23e5c8a22c87fb9d3ac784a59898433a78a0680c6a38bc7162fc83f79"
preserved = []
for artifact in original["artifacts"]:
    path = ROOT / artifact["path"]
    digest = sha(path.read_bytes())
    assert digest == artifact["sha256"]
    preserved.append({"path": artifact["path"], "sha256": digest})
prior_check = BASE / "self-check.json"
assert sha(prior_check.read_bytes()) == "d1bc300bc88d53fd29130cdfc125f665560ab3f0561ad9601a3823763c9547c3"
preserved.append({"path": relative(prior_check), "sha256": sha(prior_check.read_bytes())})
repository = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        path = ROOT / source["path"]
        digest = sha(path.read_bytes())
        assert digest == source["sha256"]
        repository.append({"path": source["path"], "sha256": digest})

prior_probe = subprocess.run(
    [sys.executable, str(BASE / "probe.py")],
    capture_output=True,
    check=False,
    cwd=ROOT,
)
assert prior_probe.returncode == 0
assert prior_probe.stderr == b""
assert sha(prior_probe.stdout) == sha((BASE / "probe-output.json").read_bytes())
prior_result = json.loads(prior_probe.stdout)

# 窮舉本例兩個 PAD 的合法整數位置組合。
torch.set_num_threads(1)
torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=10, width=8))
ids = torch.tensor([[0, 0, 1, 2, 3]])
valid = ids != 0
positions = torch.tensor([[0, 0, 0, 1, 2]])
limit = model.position.num_embeddings
pairs = torch.cartesian_prod(torch.arange(limit), torch.arange(limit))
max_difference = 0.0
verified_pairs = 0
with torch.no_grad():
    base = model(torch.tensor([[1, 2, 3]]))["logits"]
    for pair_chunk in pairs.split(128):
        count = pair_chunk.shape[0]
        trial = positions.repeat(count, 1)
        trial[:, :2] = pair_chunk
        value = model(
            ids.repeat(count, 1), valid=valid.repeat(count, 1), positions=trial
        )["logits"][:, 2:]
        expected = base.expand_as(value)
        difference = (value - expected).abs().max().item()
        max_difference = max(max_difference, difference)
        assert torch.allclose(value, expected, atol=1e-6, rtol=0.0)
        verified_pairs += count
assert verified_pairs == limit ** 2

# 先記錄位置表的輸入，再記錄是否走到注意力。
events = []


def before_position(module, arguments):
    events.append({"stage": "position_lookup", "positions": arguments[0].tolist()})


def before_attention(module, arguments):
    events.append({"stage": "attention", "valid": arguments[1].tolist()})


handles = [
    model.position.register_forward_pre_hook(before_position),
    model.blocks[0].attention.register_forward_pre_hook(before_attention),
]
with torch.no_grad():
    model(ids, valid=valid, positions=positions)
legal_trace = list(events)
assert [event["stage"] for event in legal_trace] == ["position_lookup", "attention"]
assert legal_trace[0]["positions"] == positions.tolist()
invalid_trials = []
for value in [-100, -1, limit]:
    events.clear()
    trial = positions.clone()
    trial[:, :2] = value
    try:
        with torch.no_grad():
            model(ids, valid=valid, positions=trial)
    except IndexError as error:
        invalid_trials.append({
            "pad_position": value,
            "exception": type(error).__name__,
            "message": str(error),
            "trace": list(events),
        })
    else:
        raise AssertionError("非法索引應在位置查表時失敗")
    assert [event["stage"] for event in events] == ["position_lookup"]
for handle in handles:
    handle.remove()

old_source = ROOT / "docs/course-review-20261019/freeze-01/sources/7.7.md"
new_source = ROOT / "docs/course-review-20261019/freeze-05/sources/7.7.md"
diff = subprocess.run(
    ["git", "diff", "--no-index", relative(old_source), relative(new_source)],
    cwd=ROOT, capture_output=True, text=True, check=False,
)
assert diff.returncode == 1
assert diff.stderr == ""
figure = ROOT / "course/figures/rewrite-07-07-padding-positions.svg"
png = ROOT / "outputs/course-review-20261019/figure-renders-04/16931629aba76361e505777950208c4af602515fb4e2b3e1314517b43fadf72b.png"
result = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": str(base.dtype),
        "platform": platform.platform(),
        "seed": "42",
        "threads": str(torch.get_num_threads()),
    },
    "configuration": model.description(),
    "source_comparison": {
        "command": diff.args,
        "exit_code": diff.returncode,
        "stdout": diff.stdout,
        "stderr": diff.stderr,
        "old_sha256": sha(old_source.read_bytes()),
        "new_sha256": sha(new_source.read_bytes()),
    },
    "original_report": {"path": relative(ORIGINAL), "sha256": sha(ORIGINAL.read_bytes())},
    "preserved_artifacts": preserved,
    "repository_sources": repository,
    "prior_probe_rerun": {
        "command_argv": prior_probe.args,
        "exit_code": prior_probe.returncode,
        "stdout_sha256": sha(prior_probe.stdout),
        "stdout_bytes": len(prior_probe.stdout),
        "reused_stdout_path": relative(BASE / "probe-output.json"),
        "stderr": "",
        "deduplication_note": "本次 stdout 與保留檔逐位元組相同，沿用原證據，不另存相同副本。",
        "differences": prior_result["differences"],
    },
    "legal_pad_positions": {
        "position_dtype": str(positions.dtype),
        "range_inclusive": [0, limit - 1],
        "cartesian_pair_count": verified_pairs,
        "chunk_size": 128,
        "real_positions": [0, 1, 2],
        "maximum_difference": max_difference,
        "atol": 1e-6,
        "rtol": 0.0,
        "allclose_for_every_pair": True,
    },
    "legal_forward_trace": legal_trace,
    "invalid_pad_positions": invalid_trials,
    "figure_integrity": {
        "svg": {"path": relative(figure), "sha256": sha(figure.read_bytes())},
        "png": {"path": relative(png), "sha256": sha(png.read_bytes())},
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
