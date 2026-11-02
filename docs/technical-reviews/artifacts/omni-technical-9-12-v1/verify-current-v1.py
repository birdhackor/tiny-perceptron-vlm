"""Re-review frozen 9.12 fix13, preserving original review artifacts."""

import ast
import contextlib
import datetime
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def embedded(page_id):
    page = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == page_id)
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert sha(raw) == page["source_sha256"]
    current = (ROOT / page["source"]).read_bytes()
    start = re.search(rb"^## " + re.escape(page_id.encode()) + rb"[^\n]*\n", current, re.M)
    end = re.search(rb"^## ", current[start.end():], re.M)
    selected = current[start.start():start.end() + end.start()] if end else current[start.start():]
    assert selected == raw
    return page, raw, re.findall(r"```python\n(.*?)```", raw.decode(), re.S)


def execute(code, name, namespace=None):
    namespace = {"__name__": "__main__"} if namespace is None else namespace
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, name, "exec"), namespace)
    return namespace, output.getvalue()


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pages = {k: embedded(k) for k in ["9.12", "9.11", "19.23"]}
    p12, _, b12 = pages["9.12"]
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "torch_git_version": torch.version.git_version, "device": "cpu",
           "dtype": str(torch.get_default_dtype()), "threads_at_9_12": str(torch.get_num_threads())}
    ns, original_stdout = execute(b12[0], p12["snapshot"] + ":python[0]")
    _, exercise_stdout = execute(b12[0] + "\nprint(backward[0])\n", "9.12-current-exercise")
    assert exercise_stdout.splitlines()[-1] == "以上皆不適合"
    # Extract the actual current loop body. Probes execute those exact AST nodes,
    # including logits.max(), equality, min(tied), and the current print statement.
    tree = ast.parse(b12[0])
    loop = next(node for node in tree.body if isinstance(node, ast.For))
    body = ast.Module(body=loop.body, type_ignores=[])
    loop_text = ast.get_source_segment(b12[0], loop)
    cases = {
        "original_three": ns["scores"],
        "original_float32_counterexample": {"a": 0.0, "b": 1e-8},
        "two_maxima": {"查卡片進度": 1.0, "更改地址": 1.0, "以上皆不適合": 0.0},
        "all_maxima": {"查卡片進度": 0.0, "更改地址": 0.0, "以上皆不適合": 0.0},
        "input_rounds_to_score_tie": {"b": 1.0 + 1e-8, "a": 1.0},
    }
    permutation_results = {}
    for name, scores in cases.items():
        rows = []
        baseline = None
        max_error = 0.0
        for order in itertools.permutations(scores):
            order = list(order)
            tested, _ = execute(body, "9.12-current-exact-loop-body", {"torch": torch, "scores": scores, "order": order})
            row = {"ids": order, "logits": tested["logits"].tolist(), "p": tested["p"].tolist(),
                   "tied": tested["tied"], "chosen": tested["chosen"],
                   "old_p_argmax_choice": order[int(tested["p"].argmax())]}
            rows.append(row)
            mapped = dict(zip(order, row["p"]))
            baseline = mapped if baseline is None else baseline
            max_error = max(max_error, max(abs(mapped[k] - baseline[k]) for k in mapped))
        chosen = {row["chosen"] for row in rows}
        assert len(chosen) == 1
        for row in rows:
            assert row["chosen"] == min(row["tied"])
        permutation_results[name] = {"input_scores": scores, "permutation_count": len(rows),
                                     "selected_ids": sorted(chosen), "max_probability_reorder_error": max_error,
                                     "rows": rows if len(rows) == 2 else [],
                                     "tie_sets": sorted({tuple(sorted(row["tied"])) for row in rows})}
    counter = permutation_results["original_float32_counterexample"]
    assert counter["selected_ids"] == ["b"]
    assert {r["old_p_argmax_choice"] for r in counter["rows"]} == {"a", "b"}
    assert all(len(set(r["logits"])) == 2 and r["p"] == [.5, .5] for r in counter["rows"])
    assert permutation_results["two_maxima"]["selected_ids"] == ["更改地址"]
    assert permutation_results["all_maxima"]["selected_ids"] == ["以上皆不適合"]
    assert permutation_results["input_rounds_to_score_tie"]["selected_ids"] == ["a"]

    s = torch.tensor([2., -1., 0.])
    r = s.flip(0)
    loss = lambda x, y: float(torch.nn.functional.cross_entropy(x, torch.tensor(y)))
    labels = {"forward_gold_position": 0, "reverse_gold_position": 2,
              "forward_ce": loss(s, 0), "remapped_reverse_ce": loss(r, 2), "stale_reverse_ce": loss(r, 0)}
    assert labels["forward_ce"] == labels["remapped_reverse_ce"] < labels["stale_reverse_ce"]
    set_change = {"original_card": float(s.softmax(-1)[0]),
                  "add_logit_1_card": float(torch.cat([s, torch.tensor([1.])]).softmax(-1)[0]),
                  "remove_none_card": float(s[:2].softmax(-1)[0]),
                  "duplicate_logit_2_card": torch.tensor([2., -1., 0., 2.]).softmax(-1)[[0, 3]].tolist()}
    assert set_change["add_logit_1_card"] < set_change["original_card"] < set_change["remove_none_card"]
    _, prev_stdout = execute(pages["9.11"][2][0], "9.11-current-dependency")
    ns23, stdout23 = execute(pages["19.23"][2][0], "19.23-current-dependency")
    model, memory, choice = ns23["model"], ns23["M"], ns23["choice"]
    with torch.no_grad():
        base = model.score(memory, *choice)
        extended = choice[2] + [("bill", "查帳單明細")]
        extra = model.score(memory, choice[0], choice[1], extended)
        reference = dict(zip(extra["ids"], extra["p"].tolist()))
        max_error = 0.0
        for order in itertools.permutations(extended):
            result = model.score(memory, choice[0], choice[1], list(order))
            after = dict(zip(result["ids"], result["p"].tolist()))
            max_error = max(max_error, max(abs(after[k] - reference[k]) for k in after))
        independent_error = float((base["scores"][:-1] - extra["scores"][:2]).abs().max())
    assert max_error <= 1e-5 and independent_error <= 1e-5

    notebook = ROOT / p12["notebook"]
    assert sha(notebook.read_bytes()) == p12["notebook_sha256"]
    bundle = ROOT / "outputs/omni-notebooks-technical-fix13-current/09/9.12.ipynb"
    nb = json.loads(bundle.read_bytes())
    matching = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
                and "".join(c["source"]).rstrip("\n") == b12[0].rstrip("\n")]
    assert len(matching) == 1
    index, cell = matching[0]
    cell_stdout = "".join("".join(o["text"]) for o in cell.get("outputs", []) if o.get("output_type") == "stream")
    assert cell_stdout == original_stdout and cell["execution_count"] is not None
    cumulative = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix13-cumulative.json"
    pins = [{"page_id": k, "snapshot": p["snapshot"], "source": p["source"],
             "selector": p["selector"], "source_sha256": sha(raw), "figures_sha256": p["figures_sha256"],
             "current_exact_slice_matches": True} for k, (p, raw, _) in pages.items()]
    result = {"started_at": started, "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-12-v1/verify-current-v1.py",
              "environment": env, "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE.read_bytes())},
              "source_pins": pins, "embedded_code_sha256": sha(b12[0].encode()),
              "tested_exact_loop_sha256": sha(loop_text.encode()), "exact_embedded_stdout": original_stdout,
              "exercise_stdout": exercise_stdout, "permutation_probes": permutation_results,
              "total_probe_permutations": sum(p["permutation_count"] for p in permutation_results.values()),
              "actual_probabilities": s.softmax(-1).tolist(), "sum_float32": float(s.softmax(-1).sum()),
              "label_remapping": labels, "candidate_set": set_change,
              "dependency_9_11_stdout": prev_stdout,
              "dependency_19_23": {"exact_embedded_stdout": stdout23, "random_seed": 7, "threads": 1,
                                    "weights": "random, no training", "three_candidate_permutation_max_error": max_error,
                                    "old_score_change_after_addition": independent_error,
                                    "none_score_before": float(base["scores"][-1]), "none_score_after": float(extra["scores"][-1])},
              "notebook": {"path": p12["notebook"], "sha256": sha(notebook.read_bytes())},
              "kernel_bundle_inspection": {"path": str(bundle.relative_to(ROOT)), "sha256": sha(bundle.read_bytes()),
                                            "matching_cell": index, "execution_count": cell["execution_count"],
                                            "code_matches_frozen_block": True, "stdout_matches_own_execution": True,
                                            "cumulative_path": str(cumulative.relative_to(ROOT)), "cumulative_sha256": sha(cumulative.read_bytes()),
                                            "scope": "Inspected actual executed cell, not relying on collector pass or reader conclusion."},
              "result": "all assertions passed; original precision counterexample now consistently selects b from raw scores"}
    with (HERE / "execution-current-v1.json").open("x") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": result["result"], "permutations": result["total_probe_permutations"],
                      "counterexample_current_selected": counter["selected_ids"], "embedded_stdout": original_stdout,
                      "dependency_reorder_error": max_error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
