"""R.2 的有界 CPU 核驗；只抽取導覽所需切片，執行兩個前置與一個長度算例。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v2/sections/R.2.md"
EXPECTED = "4b21ed352bf2b87c7e10ff6e1528864443d22090f74ea76e5d7d78faa4318ff4"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(path, identifier):
    raw = path.read_text(encoding="utf-8")
    match = re.search(r"^## " + re.escape(identifier) + r" .+$", raw, re.M)
    assert match, (path, identifier)
    following = re.search(r"^## ", raw[match.end():], re.M)
    end = match.end() + following.start() if following else len(raw)
    body = raw[match.start():end]
    return body, raw[:match.start()].count("\n") + 1


def overview(path):
    raw = path.read_text(encoding="utf-8")
    first = re.search(r"^## ", raw, re.M)
    return raw[:first.start()] if first else raw


def main():
    import torch

    torch.set_num_threads(1)
    raw = FROZEN.read_bytes()
    assert digest(raw) == EXPECTED
    text = raw.decode("utf-8")
    actual, _ = section(ROOT / "course/README.md", "R.2")
    assert actual.encode("utf-8") == raw
    numbered_path = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v2.json"
    numbered = json.loads(numbered_path.read_bytes())
    canonical_ref = numbered["canonical_source_freeze"]
    canonical_path = ROOT / canonical_ref["path"]
    assert digest(canonical_path.read_bytes()) == canonical_ref["sha256"]
    canonical = json.loads(canonical_path.read_bytes())
    frozen_pages = {(p["source"], p["selector"]): p for p in canonical["pages"]}
    frozen_pages.update({(p["source"], p["selector"]): p for p in numbered["pages"]})
    selected = {}
    link_checks = []

    def inspect(path, identifier):
        key = path.relative_to(ROOT).as_posix() + ("#" + identifier if identifier else "#introduction")
        if key not in selected:
            body, line = section(path, identifier) if identifier else (overview(path), 1)
            selector = "section:" + identifier if identifier else "before-first-section"
            frozen = frozen_pages[(path.relative_to(ROOT).as_posix(), selector)]
            frozen_raw = (ROOT / frozen["snapshot"]).read_bytes()
            assert digest(frozen_raw) == frozen["source_sha256"], key
            assert frozen_raw == body.encode("utf-8"), key
            # Save only the precise lead prose and heading inventory used for routing.
            lead = "\n\n".join(body.split("\n\n")[:3])
            if not identifier:
                headings = re.findall(r"^## .+$", path.read_text(encoding="utf-8"), re.M)
            else:
                headings = [body.splitlines()[0]]
            selected[key] = {
                "origin": path.relative_to(ROOT).as_posix(),
                "selector": "section:" + identifier if identifier else "introduction",
                "canonical_snapshot": frozen["snapshot"],
                "snapshot_sha256": frozen["source_sha256"],
                "live_slice_matches_canonical_snapshot": True,
                "start_line": line,
                "section_sha256": digest(body.encode("utf-8")),
                "inspected_exact_quote": lead,
                "headings": headings,
            }
        return key

    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        relative, _, identifier = target.partition("#")
        path = (ROOT / "course" / relative).resolve() if relative else ROOT / "course/README.md"
        assert path.is_file(), target
        key = inspect(path, identifier)
        link_checks.append({"label": label, "target": target, "inspected_key": key, "exists": True})

    # Endpoints and intermediate sections explicitly promised by the displayed ranges.
    ranges = {
        "08": ("8", 14, 16),
        "07-mtp": ("7", 20, 22),
        "07-effort": ("7", 23, 25),
        "0C": ("C", 1, 5),
        "09": ("9", 11, 18),
        "14": ("14", 11, 18),
        "17": ("17", 16, 17),
        "19": ("19", 21, 24),
        "20": ("20", 14, 16),
    }
    for chapter, (prefix, first, last) in ranges.items():
        chapter = chapter.split("-")[0]
        for index in range(first, last + 1):
            inspect(ROOT / f"course/chapters/{chapter}.md", f"{prefix}.{index}")
    for chapter, identifier in [("01", "1.1"), ("01", "1.5"), ("02", "2.1"),
                                ("02", "2.2"), ("03", "3.2"), ("03", "3.4"),
                                ("04", "4.5"), ("07", "7.1"), ("07", "7.9")]:
        inspect(ROOT / f"course/chapters/{chapter}.md", identifier)
    for chapter in ["02", "03", "04", "13"]:
        inspect(ROOT / f"course/chapters/{chapter}.md", "")

    executions = []
    namespaces = {}
    for chapter, identifier in [("01", "1.1"), ("01", "1.5"), ("07", "7.19")]:
        path = ROOT / f"course/chapters/{chapter}.md"
        body, line = section(path, identifier)
        blocks = re.findall(r"```python\n(.*?)\n```", body, re.S)
        assert len(blocks) == 1, identifier
        code = blocks[0]
        namespace = {}
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, f"{path.relative_to(ROOT)}#{identifier}", "exec"), namespace)
        namespaces[identifier] = namespace
        executions.append({
            "origin": path.relative_to(ROOT).as_posix(), "selector": identifier,
            "start_line": line, "section_sha256": digest(body.encode("utf-8")),
            "code_sha256": digest(code.encode("utf-8")), "exact_code": code,
            "stdout": stream.getvalue(), "assertions_passed": True,
        })
    bigram = namespaces["1.5"]
    chars, probability = bigram["chars"], bigram["probability"]
    prefixes = ["貓看", "狗看"]
    rows = [probability[chars.index(prefix[-1])] for prefix in prefixes]
    assert torch.equal(rows[0], rows[1])
    length = namespaces["7.19"]
    assert (length["T"], length["P"], length["T"] - length["P"]) == (16, 12, 4)
    evidence = {"source_sha256": EXPECTED, "canonical_source_freeze": canonical_ref,
                "numbered_source_freeze": {"path": numbered_path.relative_to(ROOT).as_posix(),
                                           "sha256": digest(numbered_path.read_bytes())},
                "links": link_checks, "slices": selected,
                "scope": "導覽目的及必要前置切片；未核讀或執行其餘連結頁全文、GPU或品質實測。"}
    (OUT / "route-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "torch_threads": str(torch.get_num_threads())},
        "frozen_source_matches_dispatch": True, "canonical_R2_matches_frozen_bytes": True,
        "link_count": len(link_checks), "all_links_resolve": True,
        "inspection_slice_count": len(selected), "executed_embedded_blocks": executions,
        "additional_lookup_check": {"prefixes": prefixes, "selected_last_character": "看",
                                    "same_probability_row": True, "row_values": rows[0].tolist(),
                                    "scope": "固定相同查表模型；不是已訓練Transformer的實測"},
        "length_example": {"T": length["T"], "P": length["P"], "remaining": 4},
    }
    (OUT / "execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["environment", "link_count", "all_links_resolve",
                                                  "inspection_slice_count", "length_example"]}, ensure_ascii=False))
    for item in executions:
        print(item["selector"] + " stdout:\n" + item["stdout"])
    print("prefix lookup equality:", result["additional_lookup_check"])


if __name__ == "__main__":
    main()
