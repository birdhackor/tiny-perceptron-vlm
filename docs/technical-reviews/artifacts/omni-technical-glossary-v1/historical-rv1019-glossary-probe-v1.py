"""Historical group6: exact identity receipts and bounded Unicode/PPO evidence.

No optimizer, neural training, GPU, model loading or historical report mutation.
Third-party full texts stay in the external reference directory.
"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.posttraining import ppo_clipped_objective

OWNER = "/root/omni_technical_coordinator/technical_glossary"
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
EXTERNAL = Path("/workspace/work/reference/technical-glossary-historical-v1")


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    return json.loads((ROOT / path).read_bytes())


def document(reference):
    assert sha(reference["path"]) == reference["sha256"]
    return read(reference["path"])


def node(value, pointer):
    for part in pointer.split("/")[1:] if pointer else []:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


group = read(REQUEST)["groups"][6]
assert group["page_id"] == "glossary" and group["old_role"] == "technical"
assert group["old_owner"] == "/root/rv1019_technical_glossary" and len(group["items"]) == 4
assert group["prior_generation"] == "accepted_20261027"
freeze = read(FREEZE)
pages = {p["page_id"]: p for p in freeze["pages"]}
glossary = pages["glossary"]
assert glossary["source_sha256"] == sha(glossary["snapshot"]) == sha("course/glossary.md") == "c71853d2368dc981d52c279eaedd7029e8f69d65fbf39a652606c83510f6f056"
assert glossary["figures_sha256"] == {}
body = (ROOT / glossary["snapshot"]).read_text()
quotes = {}
for key, prefix in [("unicode", "| 字元、碼點"), ("ppo", "| 近端策略最佳化")]:
    matches = [(i, line) for i, line in enumerate(body.splitlines(), 1) if line.startswith(prefix)]
    assert len(matches) == 1
    quotes[key] = {"line": matches[0][0], "source_quote": matches[0][1]}
identities = []
for index, item in enumerate(group["items"]):
    disposition = node(document(item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
    issue = node(document(item["original_report"]), item["json_pointer"])
    verification = document(item["historical_fix_verification"]["report"])
    assert disposition["issue_id"] == item["issue_id"]
    assert disposition["report"] == item["original_report"]
    assert disposition["json_pointer"] == item["json_pointer"]
    assert disposition["action"] == "fixed" and disposition["final_necessity"] == item["necessity"] == "necessary"
    assert item["historical_fix_verification"] in disposition["verification"]
    assert verification["reviewer_task"] == group["old_owner"] and verification["verdict"] == "pass"
    assert issue["necessary"] is True and disposition["affected_pages"] == item["affected_pages"] == ["glossary"]
    identities.append({"request_item_pointer": f"/groups/6/items/{index}", "request_item": item,
                       "original_disposition": disposition, "original_issue": issue})
unavailable = node(document(group["unavailability"]["report"]), group["unavailability"]["json_pointer"])
assert unavailable["reviewer_task"] == group["old_owner"] and unavailable["unavailable"] is True

retrievals = read(EXTERNAL / "retrieval-v1.json")
assert all(row["http_status"] == 200 and sha(row["path"]) == row["sha256"] for row in retrievals)
assert retrievals[0]["sha256"] == "b60b15fe9663f95a217a6b6bf9c1389984a1d8520e13bd0690d0f104dd93e97e"
assert retrievals[1]["sha256"] == "d6b51d1d2ae5c33b451b7ed994b48f1f4dc62b2272a5831e7fd418514a6bae89"
uax_raw = Path("/workspace/work/reference/technical-glossary-own/unicode-uax29-r47.html")
uax_text = EXTERNAL / "unicode-uax29-r47-inspected.txt"
rfc = Path("/workspace/work/reference/technical-glossary-own/utf8-rfc3629.txt")
ppo_raw = Path("/workspace/work/reference/technical-glossary-own/ppo-v2.pdf")
ppo_text = Path("/workspace/work/reference/technical-glossary-own/ppo-v2.txt")
assert sha(uax_raw) == "130d4e0d3f9da41bf9ad84bf580b3b92edc83886457cbc5a53b432761fd2c6ed"
assert sha(rfc) == "a2a3a39457d30420c812f87a38cc2b9194832f6982ef17e619de7d816879d6a6"
assert sha(ppo_raw) == "e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b"
uax = uax_text.read_text()
start = uax.index("A single Unicode code point")
end = uax.index("The notion of user-perceived", start)
unicode_quote = uax[start:end].strip()
gb9 = "Do not break before extending\ncharacters or ZWJ.\nGB9\n×\n(Extend | ZWJ)"
assert gb9 in uax
official = (EXTERNAL / "ppo-official-inspected.txt").read_text()
official_quotes = []
for start_term, end_term in [("PPO-Clip doesn’t have", "Here, we’ll focus"),
                             ("While this kind of clipping", "When you feel comfortable")]:
    start = official.index(start_term)
    end = official.index(end_term, start)
    official_quotes.append(official[start:end].strip())
gcb = (EXTERNAL / "GraphemeBreakProperty-17.0.0.txt").read_text()
modifier_line = next(line for line in gcb.splitlines() if line.startswith("1F3FB..1F3FF"))
assert "; Extend" in modifier_line
primary = {
    "created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": OWNER,
    "request": ref(REQUEST), "request_json_pointer": "/groups/6", "source_freeze": ref(FREEZE),
    "scope": "Only group6 old technical owner and its four immutable identities; original dicts copied exactly, not promoted into this round original issues.",
    "group": group, "preserved_identity_and_disposition_rows": identities,
    "actual_unavailability_record": unavailable, "current_quotes": quotes,
    "retrievals": retrievals,
    "original_source_receipts": [
        {"source_id": "historical-uax29", "url": "https://www.unicode.org/reports/tr29/tr29-47.html",
         "version": "Unicode17.0.0, rev47,2025-08-17", "raw": ref(uax_raw), "inspected_text": ref(uax_text),
         "locator": "§3 Grapheme Cluster Boundaries; §3.1.1 ruleGB9", "small_original_excerpts": [unicode_quote, gb9],
         "access_note": "Personally reread the existing original cache this callback; no new HTTP claimed for this pinned source."},
        {"source_id": "historical-gcb", "url": retrievals[1]["url"], "version": "GraphemeBreakProperty17.0.0,2025-06-30",
         "raw": ref(EXTERNAL / "GraphemeBreakProperty-17.0.0.txt"), "locator": "1F3FB..1F3FF;Extend", "small_original_excerpts": [modifier_line]},
        {"source_id": "historical-rfc3629", "url": "https://www.rfc-editor.org/rfc/rfc3629.txt", "version": "RFC3629,November2003",
         "raw": ref(rfc), "locator": "§3 code-point range table and surrogate exclusion",
         "small_original_excerpts": ["\n".join(rfc.read_text().splitlines()[179:183]), "\n".join(rfc.read_text().splitlines()[230:234])],
         "access_note": "Personally reread this existing pinned original cache; not a new retrieval."},
        {"source_id": "historical-ppo-paper", "url": "https://arxiv.org/pdf/1707.06347v2", "version": "arXiv1707.06347v2,2017-08-28",
         "raw": ref(ppo_raw), "inspected_text": ref(ppo_text), "locator": "§3 Eq7; motivation following equation",
         "small_original_excerpts": ["\n".join(ppo_text.read_text().splitlines()[118:129])],
         "access_note": "Personally reread primary Eq7; Figure1 was not visually re-reviewed or used as evidence."},
        {"source_id": "historical-ppo-official", "url": retrievals[0]["url"], "version": "Spinning Up latest bytes fetched2026-10-29, pinned rawSHA; Last-Modified2020-03-03",
         "raw": ref(EXTERNAL / "ppo-official.html"), "inspected_text": ref(EXTERNAL / "ppo-official-inspected.txt"),
         "locator": "Background; Key Equations; You Should Know", "small_original_excerpts": official_quotes},
    ],
    "necessary_current_repository_source": ref("tiny_perceptron/posttraining.py"),
    "necessary_current_frozen_sections": {pid: {"path": pages[pid]["snapshot"], "sha256": pages[pid]["source_sha256"]} for pid in ["6.1", "13.12", "13.13"]},
    "retention": "Original primary full texts remain outsideGit; only short excerpts and fingerprints here. Prior original/current/failed records are not changed.",
}
primary_path = OUT / "historical-rv1019-glossary-primary-v1.json"
with primary_path.open("x") as stream:
    json.dump(primary, stream, ensure_ascii=False, indent=2)
    stream.write("\n")

torch.set_num_threads(1)
unicode_examples = []
for text in ["貓", "👍🏽", "e\u0301", "🇹🇼", "👨‍👩‍👧‍👦"]:
    unicode_examples.append({"text": text, "codepoints": [f"U+{ord(c):04X}" for c in text], "python_length": len(text), "utf8_bytes": len(text.encode("utf-8"))})
assert unicode_examples[1]["codepoints"] == ["U+1F44D", "U+1F3FD"] and unicode_examples[1]["utf8_bytes"] == 8
boundary_checks = [{"codepoint": f"U+{cp:04X}", "utf8_bytes": len(chr(cp).encode("utf-8"))} for cp in [0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF]]
assert [x["utf8_bytes"] for x in boundary_checks] == [1,1,2,2,3,3,3,3,4,4]
rejected = []
for cp in [0xD800, 0xDFFF]:
    try:
        chr(cp).encode("utf-8")
    except UnicodeEncodeError:
        rejected.append(f"U+{cp:04X}")
assert rejected == ["U+D800", "U+DFFF"]
embedded = {}
for pid in ["6.1", "13.13"]:
    page = pages[pid]
    assert sha(page["snapshot"]) == page["source_sha256"]
    text = (ROOT / page["snapshot"]).read_text()
    code = re.findall(r"```python\n(.*?)\n```", text, re.S)[0]
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, page["snapshot"] + ":first-python-cell", "exec"), namespace)
    embedded[pid] = {"snapshot": {"path": page["snapshot"], "sha256": page["source_sha256"]}, "cell_index": 0,
                     "cell_sha256": hashlib.sha256(code.encode()).hexdigest(), "stdout": captured.getvalue(),
                     "execution_scope": "Exact frozen embedded cell; no model/optimizer construction or parameter updates."}
    if pid == "13.13":
        result = namespace["result"]
        embedded[pid]["input_ratio"] = namespace["ratio"].tolist()
        embedded[pid]["input_advantage"] = namespace["advantage"].tolist()
        embedded[pid]["surrogate"] = result["surrogate"].tolist()
        embedded[pid]["loss_gradient_wrt_ratio"] = namespace["ratio"].grad.tolist()
        assert torch.allclose(result["surrogate"], torch.tensor([.7,1.,1.2,-.8,-1.,-1.3]), atol=1e-6)
        assert torch.allclose(namespace["ratio"].grad, torch.tensor([-1.,-1.,0.,0.,1.,1.]), atol=1e-6)
old = torch.tensor([.5], dtype=torch.float64)
new = torch.tensor([.99], dtype=torch.float64)
outside = ppo_clipped_objective(new.log(), old.log(), torch.ones(1, dtype=torch.float64), clip_range=.2)
assert abs(outside["ratio"].item() - 1.98) < 1e-12
assert abs(outside["surrogate"].item() - 1.2) < 1e-12
assert new.item() == .99 and outside["ratio"].item() > 1.2
output = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": OWNER,
          "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-glossary-v1/historical-rv1019-glossary-probe-v1.py",
          "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu"},
          "source_freeze": ref(FREEZE), "source_sha256": sha("course/glossary.md"), "figure_sha256": {},
          "primary_evidence": ref(primary_path.relative_to(ROOT)), "unicode_examples": unicode_examples,
          "utf8_boundary_checks": boundary_checks, "utf8_surrogate_rejections": rejected,
          "thumb_cluster_derivation": "U+1F3FD is Extend in Unicode17 GCB. GB9 forbids a boundary beforeExtend, so standalone thumbs-up+modifier is one extended cluster with two scalar values; this is a rule derivation, not a grapheme library execution.",
          "embedded_cells_actually_executed": embedded,
          "ppo_not_probability_projection": {"old_chosen_probability": old.item(), "new_chosen_probability": new.item(), "returned_unclipped_ratio": outside["ratio"].item(), "objective_surrogate": outside["surrogate"].item(), "clip_fraction": outside["clip_fraction"].item(), "interpretation": "The objective function accepts the valid .99probability and returns actualratio1.98; its clipped objective is1.2, without projectingthepolicy probability."},
          "scope": "Only deterministic formula, encoding and gradient checks for the two historical fixes. Four separate oldissue identities retained. No optimizer, model, training run, GPU, quality estimate, reader/coherence replacement or stage seal."}
assert sha("course/glossary.md") == glossary["source_sha256"]
path = OUT / "historical-rv1019-glossary-probe-v1.json"
with path.open("x") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"primary": ref(primary_path.relative_to(ROOT)), "probe": ref(path.relative_to(ROOT)),
                  "four_historical_identities": len(identities), "glossary_lines": {k: v["line"] for k, v in quotes.items()},
                  "thumb": unicode_examples[1], "embedded_stdout": {p: x["stdout"] for p, x in embedded.items()},
                  "ppo_outside_ratio": outside["ratio"].item()}, ensure_ascii=False))
