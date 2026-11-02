"""重放凍結19.23內嵌程式；沒有訓練、optimizer、下載或GPU工作。"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from itertools import permutations
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("execution.json")
SOURCE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/19.23.md"
EXPECTED_SHA = "c1403c059eb098ba826ba7a2c02df12336127a07f62fe532797f91eeda27e96a"
raw = SOURCE.read_bytes()
assert sha256(raw).hexdigest() == EXPECTED_SHA
blocks = re.findall(r"^```python\n(.*?)^```", raw.decode(), re.M | re.S)
expected = re.findall(r"^```text\n(.*?)^```", raw.decode(), re.M | re.S)[0]
ns = {}
outputs = []
for number, code in enumerate(blocks):
    stream = StringIO()
    with redirect_stdout(stream):
        exec(compile(code, f"{SOURCE}:python-{number}", "exec"), ns)
    outputs.append(stream.getvalue())
assert outputs[0] == expected
assert outputs[1] == "等分數人工校準T 1.0\n"
torch, model = ns["torch"], ns["model"]
M, state, choice = ns["M"], ns["state"], ns["choice"]
utf8, validate = ns["utf8"], ns["validate"]
result = {"embedded_stdout": outputs, "embedded_blocks_sha256": [sha256(c.encode()).hexdigest() for c in blocks]}
with torch.no_grad():
    baseline = model.score(M, *choice)
    expanded_candidates = choice[2] + [("bill", "查帳單明細")]
    expanded = model.score(M, choice[0], choice[1], expanded_candidates)
    score_deltas = {cid: float(expanded["scores"][expanded["ids"].index(cid)] - baseline["scores"][i])
                    for i, cid in enumerate(baseline["ids"])}
    assert score_deltas["card"] == score_deltas["address"] == 0.0
    assert not torch.equal(baseline["p"][:2], expanded["p"][:2])
    renamed = model.score(M, choice[0], choice[1], [("x", choice[2][0][1]), ("y", choice[2][1][1])])
    assert torch.equal(baseline["scores"], renamed["scores"])
    permutation_error = 0.0
    for order in permutations(expanded_candidates):
        reordered = model.score(M, choice[0], choice[1], list(order))
        for cid in expanded["ids"]:
            for field in ["scores", "p"]:
                permutation_error = max(permutation_error, abs(float(expanded[field][expanded["ids"].index(cid)] - reordered[field][reordered["ids"].index(cid)])))
    assert permutation_error <= 1e-5
    ce_checks = []
    for gold in ["card", "none"]:
        losses = []
        for candidates in [choice[2], list(reversed(choice[2]))]:
            r = model.score(M, choice[0], choice[1], candidates)
            target = torch.tensor(r["ids"].index(gold), dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(r["scores"], target)
            manual = -r["scores"].log_softmax(-1)[target]
            assert torch.equal(loss, manual)
            losses.append(float(loss))
        assert abs(losses[0] - losses[1]) <= 1e-6
        ce_checks.append({"gold": gold, "original_and_reversed_ce": losses})
    accepted_counts = []
    for n in range(1, 8):
        r = model.score(M, "choice", "選", [(str(i), f"候選{i}") for i in range(n)])
        assert len(r["ids"]) == n + 1
        accepted_counts.append(len(r["ids"]))
    boundary_memory = model.encode_state("a" * 512)
    model.score(boundary_memory, "choice", "q" * 128, [("a", "d" * 128)])
    model.encode_state("漢" * 170)
    assert boundary_memory.shape == (513, 16)
    assert model.pool([ns["TYPE"]["choice"]] + utf8("q" * 128) + [ns["SEP"]] + utf8("d" * 128)).shape == (16,)
    assert model.byte.num_embeddings == 262 and model.byte.padding_idx == 256
    assert ns["PAD"] not in [ns["BOS"]] + utf8(state)
    assert baseline["attention"].shape == (2, len(utf8(state)) + 1)
    attention_error = float((baseline["attention"].sum(-1) - 1).abs().max())
    result["supplement"] = {
        "state_utf8_bytes": len(utf8(state)), "memory_shape": list(M.shape),
        "dtype": str(M.dtype), "attention_shape": list(baseline["attention"].shape),
        "attention_sum_max_error": attention_error,
        "candidate_addition_score_deltas": score_deltas,
        "original_probabilities": dict(zip(baseline["ids"], baseline["p"].tolist())),
        "expanded_probabilities": dict(zip(expanded["ids"], expanded["p"].tolist())),
        "renamed_ids_same_scores": True, "three_candidate_permutations_checked": 6,
        "permutation_scores_and_p_max_error": permutation_error,
        "ce_gold_remapping": ce_checks, "choice_output_counts_for_1_to_7": accepted_counts,
        "largest_state_memory_shape": list(boundary_memory.shape),
        "largest_branch_input_length": 1 + 128 + 1 + 128,
        "parameters_updated": 0,
    }

bad = [
    ("state_513_ASCII_bytes", lambda: model.encode_state("a" * 513)),
    ("state_171_Han_513_bytes", lambda: model.encode_state("漢" * 171)),
    ("question_129_bytes", lambda: model.score(M, "choice", "q" * 129, choice[2])),
    ("description_129_bytes", lambda: model.score(M, "choice", "q", [("a", "d" * 129)])),
    ("blank_state", lambda: model.encode_state(" \n")),
    ("surrogate_state", lambda: model.encode_state("x\ud800")),
    ("surrogate_id", lambda: validate("choice", "q", [("\ud800", "d")])),
    ("surrogate_description", lambda: validate("choice", "q", [("a", "\ud800")])),
    ("unknown_kind", lambda: validate("unknown", "q", choice[2])),
    ("empty_question", lambda: validate("choice", "", choice[2])),
    ("empty_candidates", lambda: validate("choice", "q", [])),
    ("eight_substantive_choices", lambda: validate("choice", "q", [(str(i), f"d{i}") for i in range(8)])),
    ("reserved_none", lambda: validate("choice", "q", [("none", "d")])),
    ("duplicate_id", lambda: validate("choice", "q", [("a", "d1"), ("a", "d2")])),
    ("duplicate_description", lambda: validate("choice", "q", [("a", "d"), ("b", "d")])),
    ("info_maybe", lambda: validate("info", "q", [("yes", "y"), ("maybe", "m")])),
    ("rubric_missing_2", lambda: validate("rubric", "q", [("0", "z"), ("1", "o")])),
    ("bad_pair_shape", lambda: validate("choice", "q", [("a", "d", "extra")])),
]
for t in [0.0, -1.0, float("nan"), float("inf")]:
    bad.append((f"invalid_temperature_{t}", lambda t=t: model.score(M, *choice, T=t)))
rejections = []
for label, call in bad:
    try:
        call()
    except ValueError as exc:
        rejections.append({"input": label, "exception": type(exc).__name__, "message": str(exc)})
    else:
        raise AssertionError(f"未拒絕{label}")
result["supplement"]["rejections"] = rejections
grid = [0.25 * 2 ** (k / 8) for k in range(33)]
assert len(grid) == 33 and grid[0] == 0.25 and grid[16] == 1.0 and grid[-1] == 4.0
result["supplement"]["temperature_grid"] = {"points": len(grid), "minimum": min(grid), "maximum": max(grid), "one_index": 16}
result["environment"] = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_git": torch.version.git_version,
                         "device": "cpu", "dtype": "float32", "threads": str(torch.get_num_threads()), "platform": platform.platform(), "seed": "7"}
result["created_at"] = datetime.now(timezone.utc).isoformat()
result["command"] = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-23-v1/replay.py"
result["source"] = str(SOURCE.relative_to(ROOT))
result["source_sha256"] = EXPECTED_SHA
result["result"] = "PASS: 兩段教材原樣執行，逐字輸出吻合；邊界／重排／ID隔離／新增候選／CE重映射補核均通過。沒有參數更新或能力評測。"
assert not OUT.exists(), "保留執行原件，不覆寫"
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(result["result"])
