"""Bounded, offline CPU verification of R.1 entry and generation contracts."""

import ast
import contextlib
import copy
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import nbformat
import torch
from scripts import build_course

spec = importlib.util.spec_from_file_location("section_facts_r1", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(name, value):
    (PROOF / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "nbformat": nbformat.__version__,
    "device": "CPU",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "network": "No network operations in this verification; Hugging Face offline flags set",
    "working_directory": str(Path.cwd()),
}
write_json("environment.json", environment)

section, whole, first_line = facts.original_section(ROOT / "course/README.md", "R.1")
assert section == (PROOF / "section.md").read_bytes()
assert whole == (PROOF / "frozen-input/course/README.md").read_bytes()
assert not facts.fences(section, first_line)
assert not re.findall(rb"!\[[^\]]*\]\([^)]*\)|<img\b|<svg\b", section)

entry_source = ROOT / "course/chapters/01.md"
entry_section, entry_whole, entry_line = facts.original_section(entry_source, "1.1")
(PROOF / "entry-1.1.md").write_bytes(entry_section)
entry_fences = facts.fences(entry_section, entry_line)
assert len(entry_fences) == 1 and entry_fences[0]["language"] == "python"
original_code = entry_fences[0]["raw"]
(PROOF / "original-1.1-fence.py").write_bytes(original_code)


def run_code(code, label):
    namespace, captured = {}, io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, label, "exec"), namespace)
    assert namespace["restored"] == namespace["text"]
    for char in set(namespace["text"]):
        assigned = [identifier for value, identifier in zip(namespace["text"], namespace["ids"], strict=True) if value == char]
        assert len(set(assigned)) == 1
    return namespace, captured.getvalue()


original_namespace, original_stdout = run_code(original_code, "course/chapters/01.md#1.1 original fence")
assert original_namespace["text"] == "貓看狗，狗看貓。"
assert original_namespace["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
assert original_namespace["ids"][0] == original_namespace["ids"][6] == 3
(PROOF / "original-1.1-stdout.txt").write_text(original_stdout, encoding="utf-8")

reversed_code = original_code.replace(b"chars = sorted(set(text))", b"chars = sorted(set(text), reverse=True)")
assert reversed_code != original_code
(PROOF / "variant-reversed.py").write_bytes(reversed_code)
reverse_namespace, reverse_stdout = run_code(reversed_code, "R.1 bounded changed ID order")
assert reverse_namespace["ids"] != original_namespace["ids"]
assert reverse_namespace["restored"] == original_namespace["restored"]
(PROOF / "variant-reversed-stdout.txt").write_text(reverse_stdout, encoding="utf-8")

changed_code = original_code.replace("貓看狗，狗看貓。".encode(), "鳥看狗，狗看鳥。".encode())
(PROOF / "variant-changed-material.py").write_bytes(changed_code)
changed_namespace, changed_stdout = run_code(changed_code, "R.1 bounded changed source material")
assert changed_namespace["restored"] == "鳥看狗，狗看鳥。"
(PROOF / "variant-changed-material-stdout.txt").write_text(changed_stdout, encoding="utf-8")

index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
entry_matches = [(position, item) for position, item in enumerate(index) if item["id"] == "1.1"]
assert len(entry_matches) == 1
entry_position, entry = entry_matches[0]
assert entry["source"] == "course/chapters/01.md"
assert entry["notebook"] == "notebooks/01/1.1.ipynb"
contract = json.loads((ROOT / "course/lesson-contract.json").read_bytes())
assert contract["schema_version"] == 1 and "1.1" in contract["lesson_ids"]
assert not any(item["id"].startswith(("R.", "W.")) for item in index)
assert len({item["id"] for item in index}) == len(index)
assert {item["id"] for item in index} == set(contract["lesson_ids"])

entry_text = entry_section.decode("utf-8")
heading, body = entry_text.split("\n", 1)
regenerated = build_course.notebook("1.1", entry["title"], body, entry_source)
serialized = json.dumps(regenerated, ensure_ascii=False, indent=1) + "\n"
original_notebook_bytes = (ROOT / entry["notebook"]).read_bytes()
assert serialized.encode("utf-8") == original_notebook_bytes
original_notebook = json.loads(original_notebook_bytes)
nbformat.validate(nbformat.reads(serialized, as_version=4))
assert original_notebook["metadata"]["lesson_id"] == "1.1"
assert original_notebook["metadata"]["kernelspec"]["name"] == "tiny-perceptron"
assert original_notebook["metadata"]["kernelspec"]["display_name"] == "Tiny Perceptron"
matching_cells = [i for i, cell in enumerate(original_notebook["cells"]) if cell["cell_type"] == "code" and not cell["metadata"].get("course_setup") and not cell["metadata"].get("course_figure")]
assert len(matching_cells) == 1
code_cell_index = matching_cells[0]
assert "".join(original_notebook["cells"][code_cell_index]["source"]).strip() == original_code.decode().strip()

# Load only the pure rendering contracts; do not run a whole-site build or reading-time machinery.
export_tree = ast.parse((ROOT / "scripts/export_course.py").read_text(encoding="utf-8"))
functions = {node.name: node for node in export_tree.body if isinstance(node, ast.FunctionDef)}
selected_names = ("diagram_html", "joined", "output_html", "reading_markdown", "code_markdown")
export_namespace = {
    "ROOT": ROOT,
    "REPOSITORY": "birdhackor/tiny-perceptron-vlm",
    "COURSE_URL": build_course.COURSE_URL,
    "lesson_anchor_aliases": build_course.lesson_anchor_aliases,
    "html": __import__("html"),
    "re": re,
}
exec(compile(ast.Module(body=[functions[name] for name in selected_names], type_ignores=[]), "scripts/export_course.py selected pure contracts", "exec"), export_namespace)
targets = {
    (ROOT / "course/README.md").resolve(): "course.md",
    (ROOT / "course/first-steps.md").resolve(): "first-steps.md",
    entry_source.resolve(): "chapter-01.md",
}
lesson_targets = {entry_source.resolve(): {"1.1": "1.1.md"}}


def reading(content, source):
    return export_namespace["reading_markdown"](content, source, targets, lesson_targets, "main")


rendered_r1 = reading(section.decode("utf-8"), ROOT / "course/README.md")
(PROOF / "rendered-R.1.md").write_text(rendered_r1, encoding="utf-8")
link_results = []
for label, destination in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", section.decode("utf-8")):
    relative, _, fragment = destination.partition("#")
    target = (ROOT / "course" / relative).resolve()
    assert target.is_file()
    assert re.search(r"^## " + re.escape(fragment) + r" ", target.read_text(encoding="utf-8"), re.M)
    expected_target = "1.1.md" if fragment == "1.1" else "first-steps.md#" + fragment
    assert f"]({expected_target})" in rendered_r1
    link_results.append({"label": label, "source_target": destination, "generated_target": expected_target, "exists": True})
assert len(link_results) == 4

for identifier, required in (("W.1", ("Colab", "jupyter lab notebooks", "01/1.1.ipynb")), ("W.2", ("清單（list）", "字典（dictionary）")), ("W.3", (".shape", "每個軸"))):
    warmup, _, _ = facts.original_section(ROOT / "course/first-steps.md", identifier)
    assert all(value in warmup.decode("utf-8") for value in required)

# Exercise the actual exporter lesson loop for one lesson. Only the original
# character-recording fence receives our actual stdout; this is no model result.
observed_notebook = copy.deepcopy(original_notebook)
observed_notebook["cells"][code_cell_index]["execution_count"] = 1
observed_notebook["cells"][code_cell_index]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": original_stdout.splitlines(keepends=True)}]
lesson_loop = next(node for node in functions["build"].body if isinstance(node, ast.For) and ast.unparse(node.target) == "(item, notebook)")
rendered_dir = PROOF / "bounded-render"
rendered_dir.mkdir(exist_ok=True)
export_namespace.update(index=[entry], notebooks=[observed_notebook], docs=rendered_dir, reading=reading, revision="main")
exec(compile(ast.Module(body=[lesson_loop], type_ignores=[]), "scripts/export_course.py actual single-lesson loop", "exec"), export_namespace)
rendered_page = (rendered_dir / "1.1.md").read_text(encoding="utf-8")
colab_url = "https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/main/notebooks/01/1.1.ipynb"
assert colab_url in rendered_page
assert '<a class="md-button" href="notebooks/01/1.1.ipynb" download>' in rendered_page
assert "在 Colab 動手做" in rendered_page
assert "實際執行結果 · CPU" in rendered_page
assert "貓看狗，狗看貓。" in rendered_page
assert original_code.decode().strip() in rendered_page
assert "輸出依序是" in rendered_page

measurements = {
    "source_sha256": digest(section),
    "intro_sha256": digest((PROOF / "intro.md").read_bytes()),
    "navigation": link_results,
    "original_roundtrip": {"text": original_namespace["text"], "chars": original_namespace["chars"], "ids": original_namespace["ids"], "restored": original_namespace["restored"], "same_cat_id": original_namespace["ids"][0] == original_namespace["ids"][6]},
    "reverse_order": {"ids": reverse_namespace["ids"], "restored": reverse_namespace["restored"]},
    "changed_material": {"restored": changed_namespace["restored"]},
    "notebook_contract": {"index_pointer": f"/{entry_position}", "notebook": entry["notebook"], "matches_regeneration_byte_for_byte": True, "valid_nbformat": True, "markdown_cell_indices": [i for i,c in enumerate(original_notebook["cells"]) if c["cell_type"] == "markdown"], "code_cell_pointer": f"/cells/{code_cell_index}/source", "original_outputs_pointer": f"/cells/{code_cell_index}/outputs", "original_outputs_count": len(original_notebook["cells"][code_cell_index]["outputs"]), "kernel_name": original_notebook["metadata"]["kernelspec"]["name"], "guides_in_notebook_index": False},
    "export_contract": {"executed_source_ast_lines": [lesson_loop.lineno, lesson_loop.end_lineno], "colab_url": colab_url, "download_target": "notebooks/01/1.1.ipynb", "same_page_prose_code_actual_roundtrip_stdout": True, "bounded_execution_scope": "Single lesson page loop with actual original character-ID fence stdout; bootstrap and figure cells not executed; no whole-site build, model, training, data download, remote Colab UI, or deployed-site validation"},
    "inspected_json_pointers": {"course/lesson-index.json": [f"/{entry_position}/id", f"/{entry_position}/title", f"/{entry_position}/source", f"/{entry_position}/notebook", "/*/id (inventory identity only)"], "course/lesson-contract.json": ["/schema_version", "/lesson_ids (identity only)"], "notebooks/01/1.1.ipynb": ["/nbformat", "/nbformat_minor", "/metadata/lesson_id", "/metadata/kernelspec", "/cells/*/cell_type", "/cells/*/metadata/course_setup", "/cells/*/metadata/course_figure", "/cells/*/source (current lesson and generator contract)", f"/cells/{code_cell_index}/outputs (empty original outputs)"]},
    "scope": "R.1 entry and file/render generation contracts only. No training or semantic-understanding measurement. R.1 itself has no code fence or figure; guides remain reading pages rather than claiming their own executable Notebook.",
}
write_json("measurements.json", measurements)
print(json.dumps(measurements, ensure_ascii=False, indent=2))
print("R.1 bounded contract and original-fence checks passed.")
