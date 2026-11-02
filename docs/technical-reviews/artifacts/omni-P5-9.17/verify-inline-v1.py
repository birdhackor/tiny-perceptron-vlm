"""Independent CPU verification for frozen lesson 9.17; no model weights or training."""

import contextlib
import hashlib
import io
import itertools
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent / "inline-execution-v1.json"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
SHA = lambda data: hashlib.sha256(data).hexdigest()


def run_text(code, namespace):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-9.17-inline", "exec"), namespace)
    return output.getvalue()


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    started_at = datetime.now(timezone.utc).isoformat()
    freeze = json.loads(FREEZE.read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "9.17")
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert SHA(raw) == page["source_sha256"]
    current = (ROOT / page["source"]).read_text()
    current_body = re.search(r"^## 9\.17 .*?(?=^## |\Z)", current, re.M | re.S).group()
    assert current_body.encode() == raw
    code_blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    assert len(code_blocks) == 1
    code = code_blocks[0]
    nb_raw = (ROOT / page["notebook"]).read_bytes()
    assert SHA(nb_raw) == page["notebook_sha256"]
    nb = json.loads(nb_raw)
    nb_codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(nb_codes) == 2
    assert code.strip() == nb_codes[1].strip()
    inline_ns = {}
    inline_output = run_text(code, inline_ns)
    nb_ns = {}
    nb_setup_output = run_text(nb_codes[0], nb_ns)
    assert nb_ns["in_colab"] is False
    notebook_output = run_text(nb_codes[1], nb_ns)
    assert notebook_output == inline_output

    # Reference calculations use Python binary64, separately from torch's float32 example.
    def reference_softmax(xs):
        es = [math.exp(x - max(xs)) for x in xs]
        return [e / math.fsum(es) for e in es]

    r = math.sqrt(2)
    independent_logits = [r, 0.0, 1.0 - r]
    independent_reference = reference_softmax(independent_logits)
    causal_logits = [r, r / 2, 1.0 - r]
    causal_reference = reference_softmax(causal_logits)
    independent = []
    for name, question in inline_ns["questions"].items():
        runs = []
        for order in inline_ns["orders"]:
            probs, chosen = inline_ns["score"](order, question)
            assert chosen == ("card" if name == "現在要辦" else "address")
            refs = independent_reference if name == "現在要辦" else [independent_reference[1], independent_reference[0], independent_reference[2]]
            for cid, expected in zip(["card", "address", "none"], refs):
                assert abs(probs[cid] - expected) < 1e-7
            runs.append({"order": order, "probabilities": probs, "chosen": chosen})
        assert all(abs(runs[0]["probabilities"][cid] - runs[1]["probabilities"][cid]) < 1e-7 for cid in ["card", "address", "none"])
        independent.append({"question": name, "runs": runs})

    causal = []
    for order in inline_ns["orders"]:
        probs, chosen = inline_ns["score"](order, inline_ns["questions"]["現在要辦"], causal=True)
        assert chosen == "card"
        if order[0] == "card":
            refs = causal_reference
        else:
            refs = independent_reference
        for cid, expected in zip(["card", "address", "none"], refs):
            assert abs(probs[cid] - expected) < 1e-7
        causal.append({"order": order, "probabilities": probs, "chosen": chosen})

    ties = []
    for order in inline_ns["orders"]:
        probs, chosen = inline_ns["score"](order, torch.ones(2))
        assert probs["card"] == probs["address"]
        assert chosen == "address"
        ties.append({"order": order, "probabilities": probs, "chosen": chosen})

    # Exercise is applied only to the isolated artificial namespace, not to textbook bytes.
    inline_ns["features"]["card"] = torch.tensor([0.0, 1.0])
    exercise = []
    for name, question in inline_ns["questions"].items():
        for order in inline_ns["orders"]:
            probs, chosen = inline_ns["score"](order, question)
            assert probs["card"] == probs["address"]
            exercise.append({"question": name, "order": order, "probabilities": probs, "chosen": chosen})

    q = torch.tensor([2.0, 0.0]) @ torch.eye(2)
    keys = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])])
    real_scores = keys @ q / math.sqrt(q.numel())
    none_score = 1.0 - real_scores.max()
    shapes = {"identity": list(torch.eye(2).shape), "q": list(q.shape), "stacked_keys": list(keys.shape), "real_scores": list(real_scores.shape), "none_scalar": list(none_score.shape), "none_after_indexing": list(none_score[None].shape), "concatenated": list(torch.cat([real_scores, none_score[None]]).shape)}
    assert shapes == {"identity": [2, 2], "q": [2], "stacked_keys": [2, 2], "real_scores": [2], "none_scalar": [], "none_after_indexing": [1], "concatenated": [3]}
    assert not any(v.requires_grad for v in inline_ns["features"].values())
    assert not inline_ns["Wq"].requires_grad and not inline_ns["Wk"].requires_grad

    # A separate nonempty set illustrates that adding an option changes the softmax denominator.
    base = reference_softmax([r, 0.0, 1.0-r])
    extended = reference_softmax([r, 0.0, -1.0, 1.0-r])
    assert base[0] != extended[0]
    aggregations = []
    for values in itertools.permutations([0.0, 1.0, 3.0]):
        aggregations.append([sum(values) / len(values), max(values), len(values)])
    assert all(row == aggregations[0] for row in aggregations)

    result = {
        "reviewer_task": "/root/omni_p5_technical_9_17",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "CPU", "dtype": str(q.dtype), "threads": str(torch.get_num_threads())},
        "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": SHA(FREEZE.read_bytes())},
        "source": {"snapshot": page["snapshot"], "bytes": len(raw), "sha256": SHA(raw), "current_body_matches": True, "current_chapter_sha256": SHA((ROOT / page["source"]).read_bytes()), "figures": page["figures_sha256"], "optional_details_blocks": raw.decode().count("<details>")},
        "notebook": {"path": page["notebook"], "bytes": len(nb_raw), "sha256": SHA(nb_raw), "code_cells": len(nb_codes), "inline_block_sha256": SHA(code.encode()), "lesson_code_matches": True, "local_setup_executed": True, "setup_output": nb_setup_output, "all_local_code_output_matches": True},
        "stdout": inline_output,
        "independent_reference_logits": independent_logits,
        "independent_reference_probabilities": independent_reference,
        "causal_reference_logits_card_first": causal_logits,
        "causal_reference_probabilities_card_first": causal_reference,
        "max_absolute_tolerance": 1e-7,
        "independent": independent,
        "causal": causal,
        "ties": ties,
        "exercise_duplicate_features": exercise,
        "shapes": shapes,
        "set_extension": {"original_card_probability": base[0], "added_candidate_card_probability": extended[0]},
        "symmetric_aggregations_six_permutations": aggregations,
        "limitations": ["Artificial vectors only; no weights loaded, no optimizer/backward/training, no fixed-final-family inference.", "Local notebook setup branch executed; Colab network installation branch inspected but not executed.", "CPU float32 execution and Python binary64 reference do not establish all-device bitwise equivariance or trained semantic accuracy.", "No figures in assigned section; desktop/mobile rendered layout not examined."]
    }
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(DEST.relative_to(ROOT)), "sha256": SHA(DEST.read_bytes()), "bytes": DEST.stat().st_size, "all_assertions_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
