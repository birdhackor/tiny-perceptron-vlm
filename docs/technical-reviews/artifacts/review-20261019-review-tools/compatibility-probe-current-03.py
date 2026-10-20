"""比較缺作者補丁欄位時的完整結果，並核對反例在改動前是否也被接受。"""
import argparse
import importlib.util
import json
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main(args):
    raw = json.loads(args.results.read_bytes())
    results = json.loads(raw["stdout"])
    current = load("current_gate", args.repo / "docs/review-tools/incremental_revision.py")
    previous = load("previous_gate", args.area / "incremental_before.py")
    comparisons, regressions = [], []
    for case in results["cases"]:
        original = Path(case["input"]["manifest_path"])
        root = original.parents[1]
        manifest = json.loads(original.read_bytes())
        if "author_patches" in manifest:
            del manifest["author_patches"]
            path = root / "round/without-author-patches.json"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        else:
            path = original
        for stage in ["reader", "technical", "all"]:
            before = previous.audit_manifest(root, path, stage)
            after = current.audit_manifest(root, path, stage)
            comparisons.append({"case": case["case"], "stage": stage, "same": before == after,
                                "previous": before, "current": after})
        if case["violation"]:
            before = previous.audit_manifest(root, original, "all")
            regressions.append({"case": case["case"], "previous": before, "current": case["gate"],
                                "newly_accepted": bool(before["failures"]) and not case["gate"]["failures"]})
    print(json.dumps({"comparison_count": len(comparisons), "all_equal": all(row["same"] for row in comparisons),
                      "without_field": comparisons, "newly_accepted_counterexamples": regressions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    main(parser.parse_args())
