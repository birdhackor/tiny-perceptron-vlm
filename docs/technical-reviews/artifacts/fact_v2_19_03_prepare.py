"""Capture the exact lesson read and original documentation for review 19.3."""

import hashlib
import json
import platform
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_03_"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    captured = datetime.now(timezone.utc).isoformat()
    readings = []
    targets = {
        "19.md": ("19.3", "19.6"),
        "05.md": ("5.10", "5.12"),
        "13.md": ("13.1",),
        "12.md": ("12.10",),
        "0B.md": ("B.7",),
    }
    for filename, wanted in targets.items():
        file = ROOT / "course/chapters" / filename
        bodies = dict(sections(file))
        for lesson in wanted:
            data = bodies[lesson].encode("utf-8")
            destination = OUT / f"{PREFIX}section_{lesson.replace('.', '_')}.md"
            destination.write_bytes(data)
            readings.append({
                "source": f"course/chapters/{filename}#{lesson}",
                "sha256": sha(data),
                "snapshot": destination.relative_to(ROOT).as_posix(),
                "captured_at": captured,
                "inspection": "Independently read complete section in terminal during this task; snapshot captured afterward using checker.sections without normalization.",
            })
    originals = []
    urls = {
        "sklearn_cross_validation_1_7_2.rst": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst",
        "sklearn_dummy_1_7_2.py": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/dummy.py",
        "cpython_collections_3_13_5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/collections.rst",
        "cpython_stdtypes_3_13_5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst",
        "pinned_data.json": "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/data.json",
        "pinned_test_capstone.py": "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1df335318bda03fd771807f66976953231d5a00b/tests/test_capstone.py",
    }
    for name, url in urls.items():
        entry = {"url": url, "attempted_at": datetime.now(timezone.utc).isoformat()}
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
                entry.update(status=response.status, final_url=response.url)
            destination = OUT / f"{PREFIX}{name}"
            destination.write_bytes(data)
            entry.update(path=destination.relative_to(ROOT).as_posix(), sha256=sha(data), bytes=len(data))
            if name == "pinned_data.json":
                entry["matches_local"] = data == (ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json").read_bytes()
            elif name == "pinned_test_capstone.py":
                entry["matches_local"] = data == (ROOT / "tests/test_capstone.py").read_bytes()
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        originals.append(entry)
    for suffix in ("txt", "pdf"):
        source = ROOT / f"outputs/technical-sources/12.12-fresh20261003/shortcut-v2.{suffix}"
        destination = OUT / f"{PREFIX}shortcut_v2.{suffix}"
        shutil.copyfile(source, destination)
        originals.append({
            "url": "https://arxiv.org/abs/2004.07780v2",
            "path": destination.relative_to(ROOT).as_posix(),
            "sha256": sha(destination.read_bytes()),
            "bytes": destination.stat().st_size,
            "provenance": source.relative_to(ROOT).as_posix(),
            "inspection": "Original paper copied from candidate original-source cache for independent reading, not review notes.",
        })
    result = {
        "reviewer_task": "/root/integration_technical_coordinator/fact_v2_19_03",
        "reviewer_context": "fresh",
        "guide_sha256": sha((ROOT / "docs/technical-review-guide.md").read_bytes()),
        "read_order": ["Complete technical-review-guide", "19.3 and 19.6", "5.10 and 5.12", "13.1 and 12.10", "B.7 complete reread after combined output truncation", "capstone dataset generator and relevant tests and stage runner"],
        "environment": {"python": platform.python_version(), "device": "cpu"},
        "section_reads": readings,
        "original_sources": originals,
        "scope": "No old technical or reader review contents or author history read. No child agents. No installs, training, model weights, or textbook edits.",
    }
    (OUT / f"{PREFIX}read_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sections":len(readings),"downloads":originals},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
