"""本人歷史 callback 的有界來源、數量與本機 MathJax 核驗；大檔留 Git 外。"""

import hashlib
import html
import importlib.util
import json
import platform
import re
import shutil
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORK = Path("/workspace/work/omni-technical-readme-v1/historical-callback")
OUT = Path(__file__).with_name("historical-callback-evidence-v1.json")
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
DOC_REV = "a39fc2d1f7ef82477636c1aeb9dd8d8fc3eb3fe4"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def node_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def ref(name):
    return {"path": name, "sha256": sha((ROOT / name).read_bytes())}


def load(reference):
    raw = (ROOT / reference["path"]).read_bytes()
    assert sha(raw) == reference["sha256"], reference["path"]
    return json.loads(raw)


def node(value, pointer):
    for part in pointer.split("/")[1:] if pointer else []:
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def headings(path, prefix):
    # Only numbered level-two headings outside fenced blocks count as lessons.
    ids, fenced = [], False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced:
            match = re.match(r"^##\s+(" + prefix + r"\.\d+)\s", line)
            if match:
                ids.append(match[1])
    return ids


def main():
    assert not OUT.exists(), "immutable output already exists"
    WORK.mkdir(exist_ok=True)
    freeze = json.loads((ROOT / FREEZE).read_bytes())
    frozen = next(p for p in freeze["pages"] if p["page_id"] == "readme")
    body = (ROOT / frozen["snapshot"]).read_text()
    assert sha(body.encode()) == frozen["source_sha256"] == sha((ROOT / "README.md").read_bytes())
    for name, expected in frozen["figures_sha256"].items():
        assert sha((ROOT / name).read_bytes()) == expected
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "python docs/technical-reviews/artifacts/omni-technical-readme-v1/historical_callback_probe_v1.py",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU / no model inference"},
        "source_freeze": ref(FREEZE),
        "current_readme": {"source_sha256": frozen["source_sha256"], "figure_sha256": frozen["figures_sha256"], "snapshot": frozen["snapshot"]},
        "canonical_dictionary_fingerprint": "UTF-8 json.dumps(sort_keys=True, ensure_ascii=False, separators=(',', ':'))",
        "groups": [],
    }
    requests = json.loads((ROOT / REQUEST).read_bytes())
    for gi in (1, 7):
        group = requests["groups"][gi]
        unavailable = node(load(group["unavailability"]["report"]), group["unavailability"]["json_pointer"])
        assert unavailable["reviewer_task"] == group["old_owner"] and unavailable["unavailable"] is True
        rows = []
        for item in group["items"]:
            original = load(item["original_report"])
            issue = node(original, item["json_pointer"])
            disposition = node(load(item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
            old_verification = item["historical_fix_verification"]
            verified = node(load(old_verification["report"]), old_verification["json_pointer"])
            assert original["reviewer_task"] == verified["reviewer_task"] == group["old_owner"]
            assert disposition["report"] == item["original_report"]
            assert disposition["json_pointer"] == item["json_pointer"]
            assert disposition["issue_id"] == item["issue_id"]
            assert disposition["action"] == "fixed" and disposition["final_necessity"] == item["necessity"] == "necessary"
            assert old_verification in disposition["verification"]
            assert verified["verdict"] == "pass"
            selected_claim = next(c for c in verified["claims"] if c["id"] == ("mathjax_network" if gi == 1 else "c03"))
            rows.append({
                **item,
                "original_issue": issue,
                "original_issue_sha256": node_sha(issue),
                "original_disposition_sha256": node_sha(disposition),
                "historical_fix_verification_dict_sha256": node_sha(old_verification),
                "historical_verified_member_sha256": node_sha(verified),
                "historical_selected_claim": selected_claim,
                "historical_selected_claim_sha256": node_sha(selected_claim),
                "original_disposition_reason": disposition["reason"],
            })
        result["groups"].append({"request_pointer": f"/groups/{gi}", "old_owner": group["old_owner"], "prior_generation": group["prior_generation"], "prior_active_pointer": group["prior_active_pointer"], "unavailability": group["unavailability"], "unavailability_record_sha256": node_sha(unavailable), "items": rows})

    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    lesson_ids = [r["id"] for r in index]
    indexed_paths = {r["notebook"] for r in index}
    actual_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "notebooks").rglob("*.ipynb")}
    chapter_sources = sorted({r["source"] for r in index})
    actual_ids = [i for p in chapter_sources for i in headings(ROOT / p, r"(?:\d+|[ABC])")]
    assert len(lesson_ids) == len(set(lesson_ids)) == len(actual_ids) == 331
    assert set(actual_ids) == set(lesson_ids) and indexed_paths == actual_paths
    guides = {path: headings(ROOT / path, prefix) for path, prefix in [("course/README.md", "R"), ("course/first-steps.md", "W"), ("course/training.md", "T"), ("course/glossary.md", "G")]}
    assert [len(v) for v in guides.values()] == [4, 7, 11, 4]
    chinese_counts = next(l for l in body.splitlines() if "共331個逐課小節" in l)
    english_counts = next(l for l in (ROOT / "README_en.md").read_text().splitlines() if "331 lesson sections" in l)
    assert "26個編號小節" in chinese_counts and "357個編號小節" in chinese_counts
    assert "26 numbered" in english_counts and "357 numbered" in english_counts
    result["counts"] = {"lesson_count": 331, "notebooks_count": len(actual_paths), "lesson_heading_count": len(actual_ids), "lesson_id_set_sha256": node_sha(sorted(lesson_ids)), "index_notebook_paths_exactly_equal": True, "index_heading_ids_exactly_equal": True, "source_count": len(chapter_sources), "chapter_and_extension_sources": chapter_sources, "guides": guides, "extra_numbered_count": 26, "total_numbered_count": 357, "chinese_quote": chinese_counts, "english_quote": english_counts, "scope": "Current exact set/count audit; not a new full-course semantic review or 331 new kernel runs."}
    paths = ["README.md", "README_en.md", "course/lesson-index.json", *guides, "scripts/build_course.py", "scripts/export_course.py", "scripts/stage_mathjax.py", "course/web/mathjax.js", "course/web/mathjax-assets.json", "zensical.toml"]
    result["current_sources"] = [ref(p) for p in paths]

    official = []
    for name in ("hosting", "typeset"):
        attempted = f"https://docs.mathjax.org/en/v3.2/web/{name}.html"
        try:
            with urllib.request.urlopen(attempted, timeout=30) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        url = f"https://raw.githubusercontent.com/mathjax/MathJax-docs/{DOC_REV}/web/{name}.rst"
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
            source_status = response.status
        target = WORK / (name + ".rst")
        target.write_bytes(raw)
        text = raw.decode()
        excerpt = text[text.index("We recommend using"):text.index(".. _obtain-mathjax:")] if name == "hosting" else text[text.index("MathJax makes it easy"):text.index("The first of these")]
        official.append({"title": name, "attempted_url": attempted, "attempted_status": status, "url": url, "revision": DOC_REV, "source_status": source_status, "accessed_at": datetime.now(UTC).isoformat(), "sha256": sha(raw), "bytes": len(raw), "external_path": str(target), "read_scope": "Hosting opening / acquiring and serving files; typeset-page opening and typeset/typesetPromise behavior", "excerpt": excerpt.strip()})
    result["official_originals"] = official

    spec = importlib.util.spec_from_file_location("readme_current_mathjax_stage", ROOT / "scripts/stage_mathjax.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    manifest = stage.validate_manifest(json.loads((ROOT / "course/web/mathjax-assets.json").read_bytes()))
    cache = Path("/workspace/work/omni-technical-readme-v1/sandbox/outputs/readme-preview/javascripts/mathjax")
    site = WORK / "site"
    assert not site.exists()
    stage.stage_assets(site / "javascripts/mathjax", (cache, manifest))
    for item in manifest["files"]:
        stage.read_asset(site / "javascripts/mathjax" / item["path"], item)
    shutil.copyfile(ROOT / "course/web/mathjax.js", site / "javascripts/mathjax.js")
    figure_name = next(iter(frozen["figures_sha256"]))
    shutil.copyfile(ROOT / figure_name, site / "character-ids.svg")
    formula_page = next(p for p in freeze["pages"] if p["page_id"] == "9.17")
    formula_body = (ROOT / formula_page["snapshot"]).read_text()
    formula = re.search(r"\\\[.*?\\\]", formula_body, re.S).group()
    sentence = "公式由網站隨附的 MathJax 在瀏覽器排版。"
    assert sentence in body
    fixture = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>本人有界 README callback 核驗</title><style>body{font:20px sans-serif;max-width:800px;margin:20px auto;padding:10px}img{max-width:100%}#math{border:1px solid #888;padding:30px}</style><script src="javascripts/mathjax.js"></script><script defer src="javascripts/mathjax/tex-chtml.js"></script><h1>本人有界核驗</h1><p id="counts">' + html.escape(chinese_counts) + '</p><p>' + sentence + '</p><div id="math" class="arithmatex">' + html.escape(formula) + '</div><img id="figure" src="character-ids.svg" alt="文字與識別編號的對照"></html>'
    (site / "index.html").write_text(fixture)
    result["staging"] = {"method": "actual current scripts/stage_mathjax.py stage_assets + read_asset; verified previous own preview cache used only as byte source", "asset_version": manifest["version"], "asset_revision": manifest["revision"], "files_verified_before_and_after": len(manifest["files"]), "bytes": sum(f["bytes"] for f in manifest["files"]), "all_exact_hash_and_size": True, "site_outside_git": str(site), "formula_source": {"page_id": "9.17", "snapshot": formula_page["snapshot"], "source_sha256": formula_page["source_sha256"]}, "formula_quote": formula, "fixture_sha256": sha(fixture.encode()), "scope": "Bounded local harness using current literal README text, current JS, exact 94-file assets and actual lesson formula. Not a complete new Zensical site build or a publication/Colab claim."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha(OUT.read_bytes()), "identities": [len(g["items"]) for g in result["groups"]], "counts": [331, 26, 357], "staged_assets": len(manifest["files"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
