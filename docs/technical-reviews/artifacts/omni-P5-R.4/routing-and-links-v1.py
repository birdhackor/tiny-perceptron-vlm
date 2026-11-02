"""Execute exact pure Auto control functions with fixtures, without loading models."""

import ast
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


class Fixture:
    def __init__(self, choice="boxes", info="yes"):
        self.choice, self.info, self.calls = choice, info, []
        self.auto_policy = {"version": 1, "low_max_events": 0, "medium_max_events": 4, "sum_depth": 1}

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {"selected": self.choice if question == "choice" else self.info, "seconds": 0.0}

    def solve(self, state, loops):
        self.calls.append({"state": state, "loops": loops})
        return {"fixture": True, "loops": loops}


source_path = ROOT / "tiny_perceptron/lesson_products/inference.py"
text = source_path.read_text()
tree = ast.parse(text)
names = {"parse_state", "rule_depth", "auto_solve"}
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(definitions) == len(names)
scope = {"re": re, "time": time, "EFFORT": {"low": 1, "medium": 3, "high": 5}}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), scope)
state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
rows = []
for name, choice, info, effort, cap, expected, depth in [
    ("unsupported", "card", "yes", "auto", 5, "unsupported", 0),
    ("missing_information", "boxes", "no", "auto", 5, "clarification_needed", 0),
    ("auto_supported", "boxes", "yes", "auto", 5, "answered", 3),
    ("manual_high", "boxes", "yes", "high", 5, "answered", 5),
    ("hard_cap", "boxes", "yes", "medium", 2, "budget_limited", 2),
]:
    b, c = Fixture(), Fixture(choice, info)
    result = scope["auto_solve"](b, c, state, effort, cap)
    assert result["status"] == expected and result["b_loops"] == depth
    assert c.calls == ["choice", "info"]
    assert bool(b.calls) == bool(depth)
    rows.append({"case": name, "status": result["status"], "b_loops": result["b_loops"], "C_calls": c.calls, "B_calls": b.calls})

numbered_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v5.json"
numbered = json.loads(numbered_path.read_text())
section = next(page for page in numbered["pages"] if page["page_id"] == "R.4")
body = (ROOT / section["snapshot"]).read_bytes()
assert hashlib.sha256(body).hexdigest() == section["source_sha256"]
canonical_path = ROOT / numbered["canonical_source_freeze"]["path"]
assert hashlib.sha256(canonical_path.read_bytes()).hexdigest() == numbered["canonical_source_freeze"]["sha256"]
canonical = json.loads(canonical_path.read_text())
page = next(page for page in canonical["pages"] if page["page_id"] == "course")
canonical_body = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(canonical_body).hexdigest() == page["source_sha256"]
heading = b"## R.4 "
start = canonical_body.index(heading)
following = re.search(rb"^## ", canonical_body[start + len(heading):], re.M)
end = start + len(heading) + following.start() if following else len(canonical_body)
assert canonical_body[start:end] == body
live_body = (ROOT / "course/README.md").read_bytes()
assert live_body[live_body.index(heading):] == body
links = []
for url in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body.decode()):
    path, _, anchor = url.partition("#")
    target = (ROOT / "course" / path).resolve()
    assert target.is_relative_to(ROOT) and target.is_file(), url
    if anchor:
        assert re.search(r"^## " + re.escape(anchor) + r"\s", target.read_text(), re.M), url
    links.append({"link": url, "target": target.relative_to(ROOT).as_posix(), "anchor_checked": bool(anchor)})
assert not re.search(rb"!\[|```", body)
inspection = json.loads((Path(__file__).parent / "source-inspection-v1.json").read_text())
release = json.loads((ROOT / "docs/omni-products/v1/public-release.json").read_text())
live_assets = inspection["public_provider_metadata"]["assets"]
assert inspection["public_provider_metadata"]["receipt"]["status"] == 200
assert inspection["public_provider_metadata"]["release_tag"] == release["release_tag"]
assert len(live_assets) == len(release["components"]) == 5
for row in live_assets:
    selected = release["components"][row["component"]]
    assert (row["id"], row["name"], row["bytes"], row["sha256"]) == (selected["asset_id"], selected["name"], selected["bytes"], selected["sha256"])
license_text = (ROOT / "LICENSE").read_text()
asset_text = (ROOT / "assets/training/README.md").read_text()
assert license_text.startswith("MIT License") and "associated documentation files" in license_text
assert all(term in asset_text for term in ["CC-BY-NC-4.0", "SIL OFL", "不能統一當成此repo的MIT內容"])
print(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_r_4",
    "source": source_path.relative_to(ROOT).as_posix(),
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "extracted_exact_functions": sorted(names),
    "function_lines": {node.name: [node.lineno, node.end_lineno] for node in definitions},
    "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU; fixtures; no torch import, weights or neural calls"},
    "cases": rows,
    "numbered_freeze_sha256": hashlib.sha256(numbered_path.read_bytes()).hexdigest(),
    "canonical_freeze_sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
    "R4_bytes": len(body),
    "R4_sha256": hashlib.sha256(body).hexdigest(),
    "canonical_slice_matches": True,
    "live_section_matches": True,
    "links": links,
    "no_inline_code_or_figures": True,
    "public_provider_projection_assets_matching_fixed_manifest": len(live_assets),
    "public_provider_receipt_sha256": inspection["public_provider_metadata"]["receipt"]["sha256"],
    "license_boundary_checked": "Project MIT grant includes associated documentation; upstream data/fonts remain source-specific. Not a universal upstream license compliance audit.",
    "scope": "Exact routing-control functions exercised with artificial fixture judgments. This is not model inference or neural capability verification. R.4 links and freeze bytes checked locally; linked pages are prerequisites, not additional assigned sections.",
    "all_assertions_pass": True,
}, ensure_ascii=False, indent=2))
