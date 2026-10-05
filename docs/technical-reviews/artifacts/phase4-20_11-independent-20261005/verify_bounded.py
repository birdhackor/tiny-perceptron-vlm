"""Original fence plus finite reading-order/data-contract checks; no model execution."""
import ast
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy_raw(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    assert sha(source) == sha(target)
    return {"original": str(source.relative_to(ROOT)), "retained": str(target.relative_to(ROOT)), "sha256": sha(target)}

def main():
    fence = OUT / "inputs/fence-1.python"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(fence.read_bytes(), str(fence), "exec"), {})
    actual = buf.getvalue()
    expected = "字元種類與數量相同 True\n完整字串相同 False\n參考行序 ['牛奶', '麵包']\n預測行序 ['麵包', '牛奶']\n"
    assert actual == expected
    print("ORIGINAL_FENCE_STDOUT_BEGIN\n" + actual + "ORIGINAL_FENCE_STDOUT_END")
    ref, swapped, no_break, intraline = "牛奶\n麵包", "麵包\n牛奶", "牛奶麵包", "奶牛\n包麵"
    assert Counter(ref) == Counter(swapped) == Counter(intraline)
    assert Counter(ref) == Counter({"牛": 1, "奶": 1, "\n": 1, "麵": 1, "包": 1})
    assert ref != swapped and ref.splitlines() != swapped.splitlines()
    assert ref != no_break and "".join(ref.split()) == "".join(no_break.split())
    assert ref.splitlines() != intraline.splitlines()
    print("VARIANTS", json.dumps({"counter_characters": dict(Counter(ref)), "length": len(ref), "newlines": ref.count("\n"), "no_break_exact": ref == no_break, "no_break_whitespace_stripped": "".join(ref.split()) == "".join(no_break.split()), "intraline_counter_equal": Counter(ref) == Counter(intraline), "intraline_lines_equal": ref.splitlines() == intraline.splitlines()}, ensure_ascii=False))
    scorer = ROOT / "scripts/score_natural_v4_test.py"
    tree = ast.parse(scorer.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalized")
    ns = {"unicodedata": unicodedata}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(scorer), "exec"), ns)
    normal = ns["normalized"]
    assert normal(ref, True) == normal(no_break, True)
    assert normal(ref, False) != normal(no_break, False)
    assert normal(ref, False) != normal(swapped, False)
    print("ORIGINAL_NORMALIZED", json.dumps({"function_lines": [node.lineno, node.end_lineno], "single_ocr_strips_newline": normal(ref, True) == normal(no_break, True), "ordered_ocr_preserves_newline": normal(ref, False) != normal(no_break, False), "ordered_ocr_rejects_swap": normal(ref, False) != normal(swapped, False)}, ensure_ascii=False))
    labels = ROOT / "docs/natural-assistant/v4/data/ocr-order-labels.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    assert Counter(r["split"] for r in rows) == {"validation": 3, "test": 3}
    selected = []
    copies = [copy_raw(labels, OUT / "inputs/ocr-order-labels.jsonl"), copy_raw(scorer, OUT / "inputs/score_natural_v4_test.py"), copy_raw(ROOT / "scripts/build_natural_v4_assets.py", OUT / "inputs/build_natural_v4_assets.py")]
    for i, row in enumerate(rows):
        assert row["ordered_transcription"] is True and row["view"] == "whole"
        assert row["line_count"] == len(row["answer"].splitlines())
        assert "\n" in row["answer"] and row["reading_order"]
        assert len(row["target_xyxy"]) == 4
        if row["split"] != "test":
            continue
        original = ROOT / row["image"]
        assert sha(original) == row["image_sha256"] == row["source_image_sha256"]
        retained = OUT / "raw-images" / original.name
        copies.append(copy_raw(original, retained))
        with Image.open(retained) as im:
            box = row["target_xyxy"]
            assert 0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height
            crop = im.crop(box)
            crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST).save(OUT / "raw-images" / (original.stem + "-target-nearest3x.png"))
            size = list(im.size)
        selected.append({"jsonl_row_zero_based": i, "id": row["id"], "source_id": row["source_id"], "question": row["question"], "answer": row["answer"], "reading_order": row["reading_order"], "target_xyxy": box, "line_count": row["line_count"], "image_size": size, "shared_source_task_id": row["shared_source_task_id"], "image_sha256": row["image_sha256"]})
    (OUT / "inspected-gold-leaves.json").write_text(json.dumps({"rows_total": len(rows), "test_rows": len(selected), "selected": selected, "copy_sha_bindings": copies}, ensure_ascii=False, indent=2) + "\n")
    print("GOLD_CONTRACT", json.dumps({"rows_total": len(rows), "splits": dict(Counter(r["split"] for r in rows)), "test_rows": len(selected), "test_families": len({r["family"] for r in rows if r["split"] == "test"}), "test_line_counts": [r["line_count"] for r in rows if r["split"] == "test"], "all_line_count_matches": True, "all_whole_image": True, "raw_copy_hashes_match": True}, ensure_ascii=False))
    audit_file = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/selected-final-test-audit-summary.json"
    audit = json.loads(audit_file.read_text())
    manifest_file = ROOT / "docs/natural-assistant/v4/manifest.json"
    assert sha(manifest_file) == audit["artifact_binding"]["manifest_sha256"]
    assert sha(scorer) == audit["frozen_source_files"]["scripts/score_natural_v4_test.py"]["sha256"]
    manifest = json.loads(manifest_file.read_text())
    leaves, task_counts = [], Counter()
    gold = {"ocr-v4:" + r["id"]: r for r in rows if r["split"] == "test"}
    for i, row in enumerate(manifest["rows"]):
        if row["split"] != "test" or row["task"] not in {"ocr", "ocr_order"}:
            continue
        task_counts[row["task"]] += 1
        assert row["references"]["text"] == row["answer"]
        assert row["references"]["strip_whitespace"] is (row["task"] == "ocr")
        if row["task"] == "ocr_order":
            origin = gold[row["id"]]
            assert row["answer"] == origin["answer"] and row["user"] == origin["question"]
            assert row["image"] == origin["image_relative"]
            leaves.append({"pointer": f"/rows/{i}", **{k: row[k] for k in ["id", "task", "split", "family", "image", "user", "answer", "references"]}})
    assert task_counts == {"ocr": 10, "ocr_order": 3}
    binding_fields = ["manifest_sha256", "test_run_id", "execution_revision", "selected_variant", "model", "model_revision", "adapters", "scoring_sources"]
    bound = {"audit_full_sha256": sha(audit_file), "manifest_full_sha256": sha(manifest_file), "inspected_audit_pointers": ["/frozen_source_files/scripts~1score_natural_v4_test.py", "/frozen_source_files/docs~1natural-assistant~1v4~1manifest.json"] + ["/artifact_binding/" + k for k in binding_fields], "artifact_binding": {k: audit["artifact_binding"][k] for k in binding_fields}, "selected_manifest_leaves": leaves, "task_counts": dict(task_counts), "scope": "Original task/rubric/source binding only; model predictions and score conclusions were not read or recomputed."}
    (OUT / "inspected-bound-rows.json").write_text(json.dumps(bound, ensure_ascii=False, indent=2) + "\n")
    print("FROZEN_BINDING", json.dumps({"manifest_sha_match": True, "scorer_sha_match": True, "task_counts": dict(task_counts), "three_ordered_gold_questions_answers_images_match": True}, ensure_ascii=False))
    env = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "device": "cpu", "Pillow": pillow_version, "inkscape": subprocess.run(["inkscape", "--version"], capture_output=True, text=True, check=True).stdout.strip(), "model_execution": "none", "training": "none", "network_in_checks": "none"}
    (OUT / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n")
    print("ENVIRONMENT", json.dumps(env, ensure_ascii=False))

if __name__ == "__main__":
    main()
