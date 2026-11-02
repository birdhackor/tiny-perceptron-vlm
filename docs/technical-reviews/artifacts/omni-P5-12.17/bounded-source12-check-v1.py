"""Same-owner source12 callback: identity checks and two scalar expressions only."""

import ast
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual"
ORIGINAL = REVIEW / "technical-12.17/technical-original-v1.json"
FREEZE = REVIEW / "source-freeze-v12.json"
REPAIR = REVIEW / "coherence-three-author-repairs-source12-v1.json"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ORIGINAL) == "2d8afbda1eecf0636c5be0ccb33913aea59565741d869b1660c0603b32cafa5a"
assert sha(FREEZE) == "1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
assert sha(REPAIR) == "76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"
original = json.loads(ORIGINAL.read_text())
freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "12.17")
old = (ROOT / next(s["path"] for s in original["sources"] if s["id"] == "s_inline")).read_bytes()
new = (ROOT / page["snapshot"]).read_bytes()
added = "本例的區塊能量取振幅平方平均，即[12.4](12.md#12.4)的平方和除以該塊樣本數；後面的0.01門檻與正規化都依這個尺度計算。"
assert new.decode() == old.decode().replace("\n```python\n", "\n" + added + "\n\n```python\n", 1)
assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
current = re.search(r"^## 12\.17 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0]
assert current.encode() == new
extract = lambda text: re.findall(r"```python\n(.*?)```", text, re.S)
old_code, new_code = extract(old.decode()), extract(new.decode())
assert old_code == new_code and len(new_code) == 1
assert ast.dump(ast.parse(old_code[0])) == ast.dump(ast.parse(new_code[0]))
nb = json.loads((ROOT / page["notebook"]).read_text())
assert sha(ROOT / page["notebook"]) == page["notebook_sha256"]
assert "".join(nb["cells"][5]["source"]) == new_code[0].rstrip("\n")
assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "markdown", "code", "code", "markdown", "code", "markdown"]
notebook_added = added.replace("(12.md#12.4)", "(https://birdhackor.github.io/tiny-perceptron-vlm/12.4.html)")
assert notebook_added in "".join(nb["cells"][4]["source"])
dependencies = []
for source in original["sources"]:
    if source["id"] in {"s_av", "s_data", "s_evaluation"}:
        digest = sha(ROOT / source["path"])
        assert digest == source["sha256"]
        dependencies.append({"path": source["path"], "sha256": digest})
for artifact in original["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"]
assert page["figures_sha256"] == original["figure_sha256"]
for path, digest in page["figures_sha256"].items():
    assert sha(ROOT / path) == digest

# Execute only the preserved energy/normalization assignments and threshold.
# No local_tokens call, STFT, complete inline block, kernel, model, or family data.
tree = ast.parse(new_code[0])
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "local_tokens")
assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id in {"energy", "normalized"} for t in n.targets)]
assert [n.targets[0].id for n in assignments] == ["energy", "normalized"]
answer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "answer")
active = next(n for n in ast.walk(answer) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "active" for t in n.targets))
predicate = active.value.generators[0].ifs[0]
energy_program = compile(ast.Module(body=assignments, type_ignores=[]), "preserved-energy-assignments", "exec")
threshold_program = compile(ast.Expression(predicate), "preserved-energy-threshold", "eval")
rows = []
for values, expected_energy, expected_active in [
    ([0.0, 0.0], 0.0, False),
    ([0.2, -0.2], 0.04, True),
    ([0.05, -0.05], 0.0025, False),
    ([0.2, -0.2, 0.2, -0.2], 0.04, True),
]:
    raw = torch.tensor(values, dtype=torch.float64)
    scope = {"raw": raw}
    exec(energy_program, scope)
    energy, normalized = scope["energy"], scope["normalized"]
    mean_from_sum = raw.square().sum() / raw.numel()
    active_value = bool(eval(threshold_program, {"token": {"content": energy[None]}}))
    assert torch.allclose(energy, mean_from_sum, atol=1e-15, rtol=1e-14)
    assert abs(energy.item() - expected_energy) < 1e-12
    assert active_value is expected_active
    assert torch.isfinite(normalized).all()
    if expected_energy:
        assert abs(normalized.square().mean().item() - 1.0) < 1e-12
    else:
        assert torch.equal(normalized, torch.zeros_like(normalized))
    rows.append({"raw": values, "n": len(values), "sum_squares": raw.square().sum().item(),
                 "energy": energy.item(), "sum_squares_div_n": mean_from_sum.item(),
                 "sqrt_energy": energy.sqrt().item(), "normalized": normalized.tolist(),
                 "normalized_mean_square": normalized.square().mean().item(),
                 "active_energy_gt_0_01": active_value})

prerequisite = next(p for p in freeze["pages"] if p["page_id"] == "12.4")
prerequisite_raw = (ROOT / prerequisite["snapshot"]).read_bytes()
assert hashlib.sha256(prerequisite_raw).hexdigest() == prerequisite["source_sha256"]
assert "一框所有振幅平方後相加" in prerequisite_raw.decode()
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": original["reviewer_task"],
    "review_mode": "same-owner bounded callback; not a new fresh/blind full review",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU"},
    "original": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": sha(ORIGINAL)},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)},
    "source": {"path": page["snapshot"], "sha256": page["source_sha256"]},
    "exact_added_paragraph": added,
    "identity": {"only_paragraph_added": True, "inline_code_bytes_equal": True, "inline_ast_equal": True,
                 "figures_unchanged": True, "relevant_implementation_unchanged": dependencies,
                 "all_original_artifact_hashes_unchanged": True},
    "current_notebook": {"path": page["notebook"], "sha256": sha(ROOT / page["notebook"]),
                         "added_paragraph_with_expected_notebook_link_rewrite": notebook_added,
                         "teaching_cell_equals_preserved_executed_inline_code": True,
                         "current_code_cell_signatures": [{"index": i, "type": c["cell_type"],
                            "source_sha256": hashlib.sha256("".join(c["source"]).encode()).hexdigest()}
                            for i,c in enumerate(nb["cells"]) if c["cell_type"] == "code"],
                         "limitation": "No old full Notebook byte copy in Git; compared teaching cell directly with preserved frozen/executed code, and inspected current cell types. Did not independently reassert historical setup-cell byte equality or rerun bootstrap."},
    "energy_expressions": [ast.unparse(n) for n in assignments],
    "active_predicate": ast.unparse(predicate),
    "artificial_scalar_rows": rows,
    "derivation": "E=sum(x_i^2)/n; n>0 is checked before slicing nonempty blocks. RMS=sqrt(E). normalized=x/max(RMS,1e-6). The comparison is original E>0.01, not normalized power. No samples outside raw enter the mean. Padding is later and cannot change n or the stored raw energy.",
    "prerequisite": {"page_id": "12.4", "path": prerequisite["snapshot"], "sha256": prerequisite["source_sha256"],
                     "quote": "這裡的**能量**是一框所有振幅平方後相加"},
    "original_claim_reuse": [{"claim_id": c["id"], "status": "still_verified", "basis":
                              "Exact only-paragraph delta; original teaching code, relevant implementation, SVG and evidence unchanged. Original execution/inspection retained, not rerun."}
                             for c in original["claims"]],
    "result": "all bounded assertions passed",
    "not_run": ["full inline block", "STFT", "full Notebook/kernel", "NN", "weights", "train", "fixed family", "Modal", "publish"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
