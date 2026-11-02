"""Bounded v8 -> v10 canonical source / v8 -> v9 preview update proof for index."""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def main():
    old_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
    new_path = old_path.with_name("source-freeze-v10.json")
    old = json.loads(old_path.read_bytes())
    new = json.loads(new_path.read_bytes())
    op = {x["page_id"]: x for x in old["pages"]}
    np = {x["page_id"]: x for x in new["pages"]}
    previews = [ROOT / "outputs/omni-P5-preview-v8", ROOT / "outputs/omni-P5-preview-v9"]
    homes = [(p / "index.html").read_bytes() for p in previews]
    asset_refs = sorted(set(re.findall(r'(?:src|href)="([^"?#]+\.(?:js|css))"', homes[0].decode())))
    workers = re.findall(r'"search":"([^"?#]+\.js)"', homes[0].decode())
    asset_refs = sorted(set(asset_refs + workers))
    assets = []
    for relative in asset_refs:
        if "://" in relative:
            continue
        paths = [p / relative.removeprefix("./") for p in previews]
        assets.append({"relative": relative, "old": record(paths[0]), "new": record(paths[1]),
                       "bytes_equal": paths[0].read_bytes() == paths[1].read_bytes()})
    searches = [json.loads((p / "search.json").read_bytes()) for p in previews]
    search_maps = [{(x["location"], x["title"]): x for x in s["items"]} for s in searches]
    changed_search = [key for key in search_maps[0].keys() | search_maps[1].keys()
                      if search_maps[0].get(key) != search_maps[1].get(key)]
    previous = json.loads((HERE / "static-audit-v2.json").read_bytes())
    current = json.loads((HERE / "current-static-audit-v1.json").read_bytes())
    previous_rows = {x["id"]: x for x in previous["rows"]}
    changed_output_signatures = [x["id"] for x in current["rows"] if x["saved_output_signature_sha256"]
                                 != previous_rows[x["id"]]["saved_output_signature_sha256"]]
    frozen_index = [(ROOT / x["index"]["snapshot"]).read_bytes() for x in (op, np)]
    source_changes = [i for i in np if i in op and np[i]["source_sha256"] != op[i]["source_sha256"]]
    result = {
        "reviewer_task": "/root/omni_p5_technical_index", "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "device": "CPU; read-only SHA/source/output comparison"},
        "old_freeze": record(old_path), "new_freeze": record(new_path),
        "new_neutral_packet": record(old_path.with_name("technical-inputs-packet-v8.json")),
        "page_ids_identical": set(op) == set(np), "index_descriptor_identical": op["index"] == np["index"],
        "index_full_frozen_bytes_identical": frozen_index[0] == frozen_index[1],
        "index_source_sha256": sha(frozen_index[1]), "changed_source_page_ids": source_changes,
        "changed_notebook_page_ids": [i for i in np if i in op and np[i].get("notebook_sha256") != op[i].get("notebook_sha256")],
        "source_20_14_code_fences_identical": re.findall(r"```.*?```", (ROOT / op["20.14"]["snapshot"]).read_text(), re.S)
            == re.findall(r"```.*?```", (ROOT / np["20.14"]["snapshot"]).read_text(), re.S),
        "preview_home_bytes_identical": homes[0] == homes[1], "preview_home_records": [record(p / "index.html") for p in previews],
        "browser_assets": assets, "browser_assets_all_identical": all(x["bytes_equal"] for x in assets),
        "changed_search_entries": changed_search,
        "search_1_1_items_identical": [x for x in searches[0]["items"] if "1.1.html" in x["location"]]
            == [x for x in searches[1]["items"] if "1.1.html" in x["location"]],
        "all_331_v10_notebook_source_footer_output_checks_pass": current["failed_lesson_ids"] == [],
        "current_static_proof": record(HERE / "current-static-audit-v1.json"),
        "changed_saved_output_signature_ids": changed_output_signatures,
        "browser_reuse_reason": "Actual v8 browser observations retained: identical index HTML, browser JS/CSS/search-worker bytes, and unchanged1.1 search probe data. The new search index differs only in20.14 content; no new browser runtime or public deployment is claimed.",
        "read_again": "Read full identical index frozen body again after original; independently compared actual freeze10, packet8, preview9 and preserved bundle9. No peer conclusions were read.",
        "original_report_erratum": {
            "json_pointer": "/claims/6/scope", "original_transcription": "Chromium149.0.7827.0",
            "correct_observed_browser_version": "151.0.7922.34",
            "evidence": record(HERE / "browser-audit-v2.json"),
            "reason": "Reviewer transcription error in scope; browser artifact has the actual version. Corrected only in current report, original remains immutable.",
        },
        "limits": "No full20.14 technical rereview. Only mechanical source/version/code-fence/output equality relevant to homepage promises. No neural inference, training, retuning, workflow, Modal, commit or publication.",
    }
    assert result["index_full_frozen_bytes_identical"]
    assert result["preview_home_bytes_identical"]
    assert result["browser_assets_all_identical"]
    assert result["search_1_1_items_identical"]
    assert result["all_331_v10_notebook_source_footer_output_checks_pass"]
    assert not changed_output_signatures
    destination = HERE / "bounded-update-v1.json"
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "index_identical": result["index_full_frozen_bytes_identical"],
                      "changed_source_ids": source_changes, "all_checks_pass": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
