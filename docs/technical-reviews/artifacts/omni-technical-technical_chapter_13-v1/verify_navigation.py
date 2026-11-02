"""Bounded CPU check of chapter-13 introduction's navigation and named implementation entry points."""

import ast
import contextlib
from datetime import UTC, datetime
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import platform
import re

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
PAGES = {page["page_id"]: page for page in json.loads(FREEZE.read_bytes())["pages"]}
COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_chapter_13-v1/verify_navigation.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_page(page_id):
    page = PAGES[page_id]
    snapshot = (ROOT / page["snapshot"]).read_bytes()
    assert sha(snapshot) == page["source_sha256"], page_id
    current = (ROOT / page["source"]).read_text()
    starts = list(re.finditer(r"^## ", current, re.M))
    if page_id == "chapter-13":
        section = current[:starts[0].start()]
    else:
        selected = next(m for m in starts if re.match(r"^## " + re.escape(page_id) + r"(?:\s|\b)", current[m.start():]))
        end = next((m.start() for m in starts if m.start() > selected.start()), len(current))
        section = current[selected.start():end]
    assert section.encode() == snapshot, page_id
    return snapshot.decode()


def blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


def definition(text, name):
    return next(node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


checked_pages = {}
for page_id in ["chapter-13", "7.18", *[f"C.{i}" for i in range(1, 8)], *[f"13.{i}" for i in range(1, 18)]]:
    text = read_page(page_id)
    checked_pages[page_id] = {"snapshot": PAGES[page_id]["snapshot"], "sha256": PAGES[page_id]["source_sha256"], "heading": text.splitlines()[0], "current_matches_snapshot": True}

assert PAGES["chapter-13"]["figures_sha256"] == {}
assert not re.search(r"!\[[^\]]*\]\(", read_page("chapter-13"))
assert (ROOT / "course/glossary.md").is_file()

targets = [
    ("13.3", "alignment", "sequence_log_probability"),
    ("13.5", "alignment", "dpo_loss"),
    ("13.10", "posttraining", "preference_loss"),
    ("13.10", "posttraining", "FiniteRewardModel"),
    ("13.11", "posttraining", "bandit_advantage"),
    ("13.13", "posttraining", "ppo_clipped_objective"),
    ("13.14", "posttraining", "exact_kl"),
    ("13.15", "posttraining", "FiniteResponsePolicy"),
    ("13.15", "posttraining", "FiniteValueModel"),
]
entries = []
for page_id, module_name, symbol in targets:
    module = importlib.import_module("tiny_perceptron." + module_name)
    value = getattr(module, symbol)
    assert callable(value)
    implementation = inspect.getsource(value)
    embedded = next(code for code in blocks(read_page(page_id)) if re.search(r"^(?:def|class) " + symbol + r"\b", code, re.M))
    same = ast.dump(definition(implementation, symbol), include_attributes=False) == ast.dump(definition(embedded, symbol), include_attributes=False)
    assert same, symbol
    source_path = "tiny_perceptron/" + module_name + ".py"
    entries.append({"section": page_id, "symbol": symbol, "source": source_path, "source_sha256": sha((ROOT / source_path).read_bytes()), "importable": True, "embedded_definition_matches_implementation_ast": same})

# Execute the literal frozen 13.15 Python blocks once. This is a single-card,
# single-update diagnostic, not training a language model or evaluating quality.
torch.set_num_threads(1)
namespace = {}
output = io.StringIO()
integrated_blocks = blocks(read_page("13.15"))
with contextlib.redirect_stdout(output):
    for number, code in enumerate(integrated_blocks):
        exec(compile(code, f"{PAGES['13.15']['snapshot']}:block-{number}", "exec"), namespace)
policy = namespace["policy"]
assert not torch.equal(namespace["before"], next(policy.parameters()))
assert all(p.grad is None for p in namespace["reward_model"].parameters())
assert all(p.grad is None for p in namespace["reference"].parameters())
assert namespace["terms"]["ratio"].tolist() == [1.0]
assert namespace["action"].numel() == 1
assert namespace["new_logits"].shape == (1, 4)

print(json.dumps({
    "created_at": datetime.now(UTC).isoformat(),
    "command": COMMAND,
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1"},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE.read_bytes())},
    "checked_navigation_pages": checked_pages,
    "named_entry_points": entries,
    "glossary_target_exists": True,
    "chapter_figures": {},
    "bounded_13_15_execution": {"python_blocks": len(integrated_blocks), "block_sha256": [sha(code.encode()) for code in integrated_blocks], "updates": 1, "sampled_actions": 1, "stdout": output.getvalue(), "policy_changed": True, "reward_model_and_reference_have_no_gradients": True},
    "result": "PASS: frozen/current navigation matches; nine named inline definitions match their library AST; one literal 13.15 CPU cycle completed.",
    "scope": "Checks only chapter-13 introduction's navigation and named implementation statements. Does not independently certify 13.x section explanations, empirical quality, long training, or other devices."
}, ensure_ascii=False, indent=2))
