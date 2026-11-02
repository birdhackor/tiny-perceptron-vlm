"""Bounded CPU evidence for chapter-10 introduction; not a review of 10.x pages."""

import argparse
import ast
import contextlib
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
PACKET = "docs/course-revision-20261029-omni/reviews/technical-stage/dispatch-briefs-v10/technical_chapter_10.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    if identifier == "chapter-10":
        return raw[: raw.index("## ")]
    start = re.search(r"^## " + re.escape(identifier) + r" .+$", raw, re.M)
    stop = re.search(r"^## ", raw[start.end() :], re.M)
    end = start.end() + stop.start() if stop else len(raw)
    return raw[start.start() : end]


def definitions(raw):
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(raw).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    frozen = json.loads((ROOT / FREEZE).read_bytes())
    packet = json.loads((ROOT / PACKET).read_bytes())
    current = (ROOT / "course/chapters/10.md").read_text()
    ids = ["chapter-10", "10.1", "10.2", "10.5", "10.7", "10.8", "10.9", "10.10", "10.11", "10.12"]
    slices = {}
    receipts = []
    for identifier in ids:
        page = next(p for p in frozen["pages"] if p["page_id"] == identifier)
        raw = (ROOT / page["snapshot"]).read_bytes()
        selected = section(current, identifier).encode()
        assert sha(raw) == page["source_sha256"] == sha(selected), identifier
        slices[identifier] = raw.decode()
        receipts.append({"page_id": identifier, "snapshot": page["snapshot"], "sha256": sha(raw), "current_slice_matches": True})
    assert packet["page"]["source_sha256"] == sha(slices["chapter-10"].encode())
    assert packet["page"]["figures_sha256"] == {}
    assert not re.findall(r"!\[[^\]]*\]\([^)]+\)", slices["chapter-10"])

    library_raw = (ROOT / "tiny_perceptron/multimodal.py").read_text()
    library_defs = definitions(library_raw)
    equal_definitions = {}
    for identifier, names in {"10.1": ["scene"], "10.2": ["patchify", "unpatchify"], "10.5": ["VisionEncoder"], "10.7": ["expand_modalities"]}.items():
        blocks = re.findall(r"```python\n(.*?)```", slices[identifier], re.S)
        embedded_defs = {}
        for block in blocks:
            embedded_defs.update(definitions(block))
        for name in names:
            equal_definitions[name] = embedded_defs[name] == library_defs[name]
            assert equal_definitions[name], name

    outcomes = {}
    for identifier in ["10.1", "10.2", "10.5", "10.7", "10.8"]:
        blocks = re.findall(r"```python\n(.*?)```", slices[identifier], re.S)
        indexes = [0, 1] if identifier == "10.8" else [0]
        namespace = {"__name__": "__chapter10_intro_diagnostic__"}
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            for index in indexes:
                exec(compile(blocks[index], f"{identifier}:python-block-{index + 1}", "exec"), namespace)
        result = {"exact_embedded_blocks_executed": [i + 1 for i in indexes], "stdout": capture.getvalue()}
        if identifier == "10.1":
            image = namespace["image"]
            assert tuple(image.shape) == (3, 16, 16)
            assert image[:, 8, 8].tolist() == [1.0, 0.0, 0.0]
            assert image[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
            result.update({"image_shape": list(image.shape), "red_pixel_count": int(image[0].sum())})
        elif identifier == "10.2":
            assert torch.equal(namespace["image"], namespace["restored"])
            grid = torch.arange(16.0).reshape(1, 1, 4, 4)
            ordered = namespace["patchify"](grid, 2)
            assert ordered[0, 1].tolist() == [2.0, 3.0, 6.0, 7.0]
            result.update({"patches_shape": list(namespace["patches"].shape), "row_order_probe": ordered.tolist()})
        elif identifier == "10.5":
            grad = namespace["encoder"].projection.weight.grad
            assert tuple(namespace["features"].shape) == (2, 16, 8)
            assert grad.norm().item() > 0
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "step" for n in ast.walk(ast.parse(blocks[0])))
            result.update({"features_shape": list(namespace["features"].shape), "projection_gradient_nonzero": True, "optimizer_steps": 0})
        elif identifier == "10.7":
            assert tuple(namespace["x"].shape) == (1, 6, 8)
            assert namespace["y"].tolist() == [[-100, -100, -100, -100, 73, 2]]
            result.update({"expanded_shifted_shape": list(namespace["x"].shape), "labels": namespace["y"].tolist()})
        else:
            assert namespace["same_text"] and namespace["same_image"]
            loaded = namespace["loaded"]
            assert loaded.missing_keys == []
            assert all(name.startswith(("audio.", "audio_projector.")) for name in loaded.unexpected_keys)
            out = namespace["library"](namespace["sequence"], image=namespace["scene"]())
            assert tuple(out["logits"].shape) == (1, 19, 264)
            assert tuple(out["labels"].shape) == (1, 19)
            result.update({"same_text_logits": True, "simplified_and_library_image_logits_equal": True, "image_logits_shape": list(out["logits"].shape), "labels_shape": list(out["labels"].shape)})
        outcomes[identifier] = result

    headings = re.findall(r"^## (10\.\d+) (.+)$", current, re.M)
    assert len(headings) == 12
    assert [h[0] for h in headings[-3:]] == ["10.10", "10.11", "10.12"]
    assert headings[-1][1] == "同樣幾張圖，怎樣知道它們發生在什麼時候？"
    assert "最後三節改問圖文配對" in slices["chapter-10"]
    context_quotes = {
        "10.11": "檢索是從現成的候選中選一個，生成描述則要逐 token 寫出文字。",
        "10.12": "圖的順序相同，不能單靠「第幾張」回答「那個時刻看見什麼」。",
    }
    # Use exact brief current quotations, with no claim that the context pages were independently reviewed.
    for identifier, quote in context_quotes.items():
        assert quote in slices[identifier]
    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_chapter_10",
        "assigned_lesson_id": "chapter-10",
        "scope": "Introduction only. Linked examples executed only to check named referents; no independent verdict for 10.x pages or their figures/experiments.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "torch_threads": "1", "platform": platform.platform()},
        "git_head_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "freeze": {"path": FREEZE, "sha256": sha((ROOT / FREEZE).read_bytes())},
        "packet": {"path": PACKET, "sha256": sha((ROOT / PACKET).read_bytes())},
        "source_receipts": receipts,
        "implementation_sha256": {path: sha((ROOT / path).read_bytes()) for path in ["tiny_perceptron/multimodal.py", "tiny_perceptron/model.py"]},
        "embedded_definitions_ast_equal_to_current_library": equal_definitions,
        "bounded_cpu_results": outcomes,
        "chapter_headings": [{"id": identifier, "title": title} for identifier, title in headings],
        "last_three_section_ids": [h[0] for h in headings[-3:]],
        "outline_claim_contradicted": True,
        "context_quotes": context_quotes,
        "glossary_target_exists": (ROOT / "course/glossary.md").is_file(),
        "training_performed": False,
    }
    destination = ROOT / args.output
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": args.output, "sha256": sha(destination.read_bytes()), "embedded_example_runs": list(outcomes), "last_three": output["last_three_section_ids"], "result": "referents verified; outline last-three claim contradicted"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
