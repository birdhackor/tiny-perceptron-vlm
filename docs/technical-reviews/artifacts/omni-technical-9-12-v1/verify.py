"""9.12 technical review: run frozen embedded code and deterministic CPU probes."""

import contextlib
import datetime
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def embedded(page_id):
    page = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == page_id)
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    current = (ROOT / page["source"]).read_bytes()
    start = re.search(rb"^## " + re.escape(page_id.encode()) + rb"[^\n]*\n", current, re.M)
    tail = current[start.end():]
    end = re.search(rb"^## ", tail, re.M)
    section = current[start.start():start.end() + end.start()] if end else current[start.start():]
    assert section == raw
    blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    return page, raw, blocks


def run_block(code, name):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, name, "exec"), namespace)
    return namespace, output.getvalue()


def mapped_probabilities(order, scores):
    p = torch.tensor([scores[name] for name in order]).softmax(-1)
    return dict(zip(order, p.tolist()))


def main():
    p12, raw12, blocks12 = embedded("9.12")
    p11, raw11, _ = embedded("9.11")
    p23, raw23, blocks23 = embedded("19.23")
    notebook = ROOT / p12["notebook"]
    assert digest(notebook.read_bytes()) == p12["notebook_sha256"]
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "torch_git_version": torch.version.git_version, "device": "cpu",
           "default_dtype": str(torch.get_default_dtype()), "threads": str(torch.get_num_threads())}
    ns, output = run_block(blocks12[0], p12["snapshot"] + ":python[0]")
    exercise_ns, exercise = run_block(blocks12[0] + "\nprint(backward[0])\n", "9.12-exercise")
    scores, order = ns["scores"], ns["forward"]
    baseline = mapped_probabilities(order, scores)
    permutation_errors = [max(abs(mapped_probabilities(p, scores)[k] - baseline[k]) for k in order)
                          for p in itertools.permutations(order)]
    s = torch.tensor([scores[k] for k in order])
    reverse = list(reversed(order))
    reversed_s = torch.tensor([scores[k] for k in reverse])
    ce = lambda x, y: float(torch.nn.functional.cross_entropy(x, torch.tensor(y)))
    gold = "查卡片進度"
    label = {"gold_id": gold, "forward_position": order.index(gold),
             "reverse_position": reverse.index(gold), "stale_position_meaning": reverse[0],
             "forward_ce": ce(s, 0), "remapped_reverse_ce": ce(reversed_s, reverse.index(gold)),
             "stale_reverse_ce": ce(reversed_s, 0)}
    assert exercise.splitlines()[-1] == "以上皆不適合"
    assert abs(label["forward_ce"] - label["remapped_reverse_ce"]) <= 1e-7
    assert label["stale_reverse_ce"] > label["remapped_reverse_ce"]
    original = s.softmax(-1)
    added = torch.cat([s, torch.tensor([1.0])]).softmax(-1)
    removed = s[[0, 1]].softmax(-1)
    duplicate = torch.tensor([2., -1., 0., 2.]).softmax(-1)
    candidate_set = {"original_card": float(original[0]), "added_other_card": float(added[0]),
                     "removed_none_card": float(removed[0]), "two_equal_card_entries": [float(duplicate[0]), float(duplicate[-1])],
                     "missing_correct_option_softmax": torch.tensor([2., -1.]).softmax(-1).tolist()}
    assert candidate_set["added_other_card"] < candidate_set["original_card"] < candidate_set["removed_none_card"]
    # Original scores below are distinct even AFTER conversion to float32.
    close = {"a": 0.0, "b": 1e-8}
    precision = []
    for names in [list(close), list(reversed(close))]:
        logits = torch.tensor([close[k] for k in names])
        probs = logits.softmax(-1)
        precision.append({"ids": names, "logits": logits.tolist(), "p": probs.tolist(),
                          "selected_by_p": names[int(probs.argmax())],
                          "selected_by_score": names[int(logits.argmax())]})
        assert logits[0] != logits[1]
    assert precision[0]["selected_by_p"] != precision[1]["selected_by_p"]
    assert precision[0]["selected_by_score"] == precision[1]["selected_by_score"] == "b"
    ties = [{"ids": names, "naive": names[int(torch.tensor([.5, .5]).argmax())],
             "stable_id": min(names)} for names in [["card", "address"], ["address", "card"]]]
    # This dependency is executed from its actual frozen first code block; no helper substitutes for it.
    ns23, output23 = run_block(blocks23[0], p23["snapshot"] + ":python[0]")
    model, memory = ns23["model"], ns23["M"]
    choice = ns23["choice"]
    extended = choice[2] + [("bill", "查帳單明細")]
    with torch.no_grad():
        base = model.score(memory, *choice)
        ext = model.score(memory, choice[0], choice[1], extended)
        ref = dict(zip(ext["ids"], ext["p"].tolist()))
        max_error = 0.0
        for perm in itertools.permutations(extended):
            r = model.score(memory, choice[0], choice[1], list(perm))
            result = dict(zip(r["ids"], r["p"].tolist()))
            max_error = max(max_error, max(abs(ref[k] - result[k]) for k in ref))
        independent_error = float((base["scores"][:-1] - ext["scores"][:2]).abs().max())
        dependency = {"exact_embedded_stdout": output23, "three_substantive_candidate_permutations": 6,
                      "max_probability_reorder_error": max_error, "old_substantive_score_change_after_addition": independent_error,
                      "none_score_before": float(base["scores"][-1]), "none_score_after_addition": float(ext["scores"][-1]),
                      "weights": "random seed 7; no optimization or training", "dtype": str(memory.dtype)}
    assert max_error <= 1e-5 and independent_error <= 1e-5
    source_pins = [{"page_id": p["page_id"], "source": p["source"], "selector": p["selector"],
                    "snapshot": p["snapshot"], "source_sha256": digest(raw), "figures_sha256": p["figures_sha256"],
                    "current_exact_slice_matches": True}
                   for p, raw in [(p12, raw12), (p11, raw11), (p23, raw23)]]
    receipt = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-12-v1/verify.py",
               "environment": env, "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE.read_bytes())},
               "source_pins": source_pins, "embedded_9_12_code_sha256": digest(blocks12[0].encode()),
               "embedded_19_23_first_code_sha256": digest(blocks23[0].encode()),
               "notebook_sha256": digest(notebook.read_bytes()),
               "exact_embedded_stdout": output, "exercise_stdout": exercise,
               "actual_probabilities": original.tolist(), "all_6_permutations_max_error": max(permutation_errors),
               "sum_float32": float(original.sum()), "sum_tolerance": 1e-6,
               "label_remapping": label, "candidate_set": candidate_set, "exact_tie_probe": ties,
               "unique_logit_softmax_rounding_counterexample": precision, "dependency_19_23": dependency,
               "assertions": "all executed assertions passed; counterexample to unqualified p.argmax selection confirmed"}
    (HERE / "execution.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": receipt["assertions"], "embedded_stdout": output,
                      "exercise_last_line": exercise.splitlines()[-1], "permutation_max_error": max(permutation_errors),
                      "precision_counterexample": precision, "dependency_reorder_error": max_error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
