"""Independent chapter-10 introduction checks; no neural forward or training."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch

from tiny_perceptron.data import IGNORE
from tiny_perceptron.multimodal import expand_modalities, patchify, scene, unpatchify


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
freeze = json.loads(FREEZE.read_bytes())
pages = {page["page_id"]: page for page in freeze["pages"]}
observations = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen(page_id):
    page = pages[page_id]
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    return raw.decode("utf-8")


def blocks(page_id):
    return re.findall(r"```python\n(.*?)\n```", frozen(page_id), re.S)


def definition(block_list, name):
    for block in block_list:
        for node in ast.parse(block).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
                return node
    raise AssertionError(name)


body = frozen("chapter-10")
chapter_raw = (ROOT / "course/chapters/10.md").read_bytes()
intro_end = re.search(rb"^## ", chapter_raw, re.M).start()
assert chapter_raw[:intro_end] == body.encode("utf-8")
observations["assigned_source"] = {
    "page_id": "chapter-10",
    "selector": "before-first-section",
    "bytes": len(body.encode("utf-8")),
    "sha256": digest(body.encode("utf-8")),
    "canonical_freeze_sha256": digest(FREEZE.read_bytes()),
    "figure_references": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body),
    "code_fences": body.count("```"),
    "current_introduction_equals_frozen": True,
}

destination_ids = ["10.1", "10.2", "10.5", "10.7", "10.8", "10.9", "10.10", "10.11", "10.12"]
destinations = []
chapter_text = chapter_raw.decode("utf-8")
for page_id in destination_ids:
    match = re.search(r"^## " + re.escape(page_id) + r" .+$", chapter_text, re.M)
    assert match is not None
    later = re.search(r"^## ", chapter_text[match.end():], re.M)
    end = match.end() + later.start() if later else len(chapter_text)
    assert chapter_text[match.start():end] == frozen(page_id)
    destinations.append({
        "page_id": page_id,
        "heading": match.group(),
        "snapshot": pages[page_id]["snapshot"],
        "sha256": pages[page_id]["source_sha256"],
        "current_slice_equals_frozen": True,
    })
glossary_text = frozen("glossary")
assert glossary_text == (ROOT / "course/glossary.md").read_text()
assert all(word in glossary_text for word in ["圖片入口", "patch embedding", "時間戳", "tensor"])
observations["navigation"] = {
    "authored_links": re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", body),
    "destinations": destinations,
    "glossary_snapshot_sha256": pages["glossary"]["source_sha256"],
    "glossary_exists_and_has_named_terms": True,
}

module_path = ROOT / "tiny_perceptron/multimodal.py"
module_tree = ast.parse(module_path.read_text())
module_nodes = {node.name: node for node in module_tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
inline_map = {"scene": "10.1", "patchify": "10.2", "VisionEncoder": "10.5", "expand_modalities": "10.7"}
inline_results = []
for name, page_id in inline_map.items():
    inline = definition(blocks(page_id), name)
    original = module_nodes[name]
    assert ast.dump(inline, include_attributes=False) == ast.dump(original, include_attributes=False)
    inline_results.append({"name": name, "page_id": page_id, "library_lines": [original.lineno, original.end_lineno], "inline_ast_matches_library": True})

vision = ast.unparse(module_nodes["VisionEncoder"])
assert "self.projection(patchify(images, self.patch_size)) + self.position" in vision
library_wrapper = ast.unparse(module_nodes["MultiModalLM"])
assert all(s in library_wrapper for s in ["self.vision = VisionEncoder", "self.image_projector", "expand_modalities", "self.language(embeddings=x)"])
inline_wrapper = ast.unparse(definition(blocks("10.8"), "MultiModalLM"))
assert all(s in inline_wrapper for s in ["self.vision = VisionEncoder", "self.image_projector", "expand_modalities", "self.language(embeddings=x)"])
assert "self.audio" not in inline_wrapper and "只保留圖片的精簡版" in frozen("10.8")
observations["implementation_inspection"] = {
    "path": "tiny_perceptron/multimodal.py",
    "sha256": digest(module_path.read_bytes()),
    "exact_inline_definitions": inline_results,
    "vision_encoder_has_patch_projection_and_position_addition": True,
    "multimodal_lm_image_branch_reaches_language_embeddings": True,
    "section_10_8_is_explicit_image_only_simplification": True,
    "named_definition_line_counts": {name: module_nodes[name].end_lineno - module_nodes[name].lineno + 1 for name in ["scene", "patchify", "VisionEncoder", "expand_modalities", "MultiModalLM"]},
    "neural_encoder_or_language_forward_executed": False,
}

image = scene("red", "square")
assert tuple(image.shape) == (3, 16, 16)
assert image[:, 8, 8].tolist() == [1.0, 0.0, 0.0]
assert image[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
assert int((image[0] == 1).sum()) == 81 and bool((image[1:] == 0).all())
patches = patchify(image[None], 4)
assert tuple(patches.shape) == (1, 16, 48)
assert torch.equal(unpatchify(patches, 3, 16, 16, 4), image[None])
grid_patches = patchify(torch.arange(16.0).reshape(1, 1, 4, 4), 2)
expected_grid = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
assert grid_patches[0].tolist() == expected_grid
observations["synthetic_scene_and_patches"] = {
    "image_shape": list(image.shape),
    "center_rgb": image[:, 8, 8].tolist(),
    "corner_rgb": image[:, 0, 0].tolist(),
    "red_pixel_count": int((image[0] == 1).sum()),
    "patch_shape": list(patches.shape),
    "round_trip_exact": True,
    "distinct_grid_patch_rows": grid_patches[0].tolist(),
    "kind": "artificial_CPU_preprocessing_demonstration",
}


class FixedLookup:
    """Deterministic table lookup, with no nn.Module or neural forward."""

    def __init__(self):
        self.weight = torch.arange(264 * 8, dtype=torch.float32).reshape(264, 8)

    def __call__(self, ids):
        return self.weight[ids]


ids = torch.tensor([1, 5, 4, 73, 2])
labels = torch.tensor([IGNORE, IGNORE, IGNORE, 73, 2])
features = torch.arange(24, dtype=torch.float32).reshape(3, 8)
x, y = expand_modalities(ids, labels, FixedLookup(), {5: features}, {5})
assert tuple(x.shape) == (1, 6, 8)
assert y.tolist() == [[-100, -100, -100, -100, 73, 2]]
assert torch.equal(x[0, 1:4], features)
observations["placeholder_expansion"] = {
    "ids": ids.tolist(),
    "labels_before_shift": labels.tolist(),
    "hand_set_image_feature_shape": list(features.shape),
    "expanded_input_shape": list(x.shape),
    "shifted_labels": y.tolist(),
    "image_feature_rows_preserved": True,
    "embedding_implementation": "FixedLookup: deterministic tensor indexing; no learned weights",
    "kind": "artificial_CPU_helper_demonstration",
}

retrieval_stdout = io.StringIO()
retrieval_namespace = {}
with contextlib.redirect_stdout(retrieval_stdout):
    exec(compile(blocks("10.9")[0], "frozen-10.9-first-code-block", "exec"), retrieval_namespace)
similarity = retrieval_namespace["similarity"]
assert similarity.tolist() == [[1.0, 0.0], [0.0, 1.0]]
assert similarity.argmax(-1).tolist() == [0, 1]
observations["candidate_matching"] = {
    "input_vectors_are_hand_set": True,
    "similarity": similarity.tolist(),
    "selected_candidate_indices": similarity.argmax(-1).tolist(),
    "stdout": retrieval_stdout.getvalue(),
    "new_text_tokens_generated": False,
}

time_stdout = io.StringIO()
time_namespace = {}
with contextlib.redirect_stdout(time_stdout):
    exec(compile(blocks("10.12")[0], "frozen-10.12-code-block", "exec"), time_namespace)
at = time_namespace["at"]
frames = time_namespace["frames"]
duration = time_namespace["duration_ms"]
index_a = at(frames, time_namespace["times_a"], duration, 1500.0)[0]
index_b = at(frames, time_namespace["times_b"], duration, 1500.0)[0]
assert (index_a, index_b) == (1, 3)
observations["frame_timestamps"] = {
    "same_frames_exact": True,
    "timestamps_a_ms": time_namespace["times_a"].tolist(),
    "timestamps_b_ms": time_namespace["times_b"].tolist(),
    "query_ms": 1500,
    "selected_frame_a": index_a,
    "selected_frame_b": index_b,
    "stdout": time_stdout.getvalue(),
    "kind": "artificial_sample_and_hold_CPU_data_lookup",
    "model_recognition_executed": False,
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_href = None
        self.current_text = ""
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current_href = dict(attrs).get("href")
            self.current_text = ""

    def handle_data(self, data):
        if self.current_href is not None:
            self.current_text += data

    def handle_endtag(self, tag):
        if tag == "a" and self.current_href is not None:
            self.links.append((self.current_text.strip(), self.current_href))
            self.current_href = None


preview = ROOT / "outputs/omni-P5-preview-v8"
preview_file = preview / "chapter-10.html"
html = preview_file.read_text()
parser = LinkParser()
parser.feed(html)
expected_links = {"10.1": "10.1.html", "10.2": "10.2.html", "10.5": "10.5.html", "10.7": "10.7.html", "10.8": "10.8.html", "10.9–10.11": "10.9.html", "10.12": "10.12.html", "名詞頁": "glossary.html"}
rendered_links = {label: href for label, href in parser.links if label in expected_links}
assert rendered_links == expected_links
assert all((preview / target).is_file() for target in expected_links.values())
assert "從一張黑底紅方塊開始" in html
observations["observed_preview_links"] = {
    "acquisition_path": "outputs/omni-P5-preview-v8/chapter-10.html",
    "html_sha256_at_inspection": digest(preview_file.read_bytes()),
    "extracted_links": rendered_links,
    "all_targets_existed_at_inspection": True,
    "layout_or_browser_visual_validation": False,
    "portability_note": "Extracted observations are preserved here; the ignored preview is not registered as a report artifact or authority source.",
}

result = {
    "reviewer_task": "/root/omni_p5_technical_chapter_10",
    "created_at": datetime.now(UTC).isoformat(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU"},
    "scope": "chapter-10 introduction only; linked sections inspected as navigation and implementation-location evidence",
    "forbidden_actions_performed": [],
    "all_assertions_passed": True,
    "observations": observations,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
