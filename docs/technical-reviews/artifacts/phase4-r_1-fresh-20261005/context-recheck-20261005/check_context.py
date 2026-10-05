"""Narrow byte/context dependency recheck by the original R.1 reviewer."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
INITIAL = HERE.parent
REVIEWER = "/root/phase4_factual_coordinator/factual_r_1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    result = {"intro": raw[:headings[0].start()]}
    lines = {"intro": 1}
    for index, heading in enumerate(headings):
        key = heading[0].decode().split()[1]
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        result[key] = raw[heading.start():end]
        lines[key] = raw[:heading.start()].count(b"\n") + 1
    return result, lines


current_raw = (ROOT / "course/README.md").read_bytes()
assert current_raw == (HERE / "frozen-current-README.md").read_bytes()
original_raw = (INITIAL / "frozen-input/course/README.md").read_bytes()
current, first_lines = sections(current_raw)
original, _ = sections(original_raw)
assert current["R.1"] == original["R.1"] == (INITIAL / "section.md").read_bytes()
assert current["intro"] == original["intro"] == (INITIAL / "intro.md").read_bytes()
changed = [key for key in original if original[key] != current[key]]
assert changed == ["R.2", "R.4"]

# Record only the actual changed paragraphs and their immediate necessary context.
context_read = []
for key, markers in (
    ("R.2", ("跳讀遇到陌生概念時",)),
    ("R.4", ("主線整合專題規劃", "MoE、PPO、DPO、量化、蒸餾")),
):
    for marker in markers:
        for offset, line in enumerate(current[key].decode().splitlines()):
            if line.startswith(marker):
                item = {"section": key, "source_line": first_lines[key] + offset, "current_paragraph": line}
                context_read.append(item)
                print(json.dumps(item, ensure_ascii=False))

# Verify the original R.1 claims' actual dependencies without rerunning code.
entry_current, _ = sections((ROOT / "course/chapters/01.md").read_bytes())
assert entry_current["1.1"] == (INITIAL / "entry-1.1.md").read_bytes()
warmup_current, _ = sections((ROOT / "course/first-steps.md").read_bytes())
warmup_original, _ = sections((INITIAL / "frozen-input/course/first-steps.md").read_bytes())
warmup_checks = {key: warmup_current[key] == warmup_original[key] for key in ("W.1", "W.2", "W.3")}
assert all(warmup_checks.values())
notebook_path = "notebooks/01/1.1.ipynb"
assert (ROOT / notebook_path).read_bytes() == (INITIAL / "frozen-input" / notebook_path).read_bytes()

ast_checks = {}
for filename, names in (
    ("scripts/build_course.py", ("cell", "notebook_reading_links", "notebook")),
    ("scripts/export_course.py", ("diagram_html", "joined", "output_html", "reading_markdown", "code_markdown")),
):
    current_tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    old_tree = ast.parse((INITIAL / "frozen-input" / filename).read_text(encoding="utf-8"))
    current_functions = {node.name: node for node in current_tree.body if isinstance(node, ast.FunctionDef)}
    old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    for name in names:
        equal = ast.dump(current_functions[name], include_attributes=False) == ast.dump(old_functions[name], include_attributes=False)
        ast_checks[filename + ":" + name] = equal
        assert equal
    def assignments(tree):
        result = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = node
        return result
    current_assignments, old_assignments = assignments(current_tree), assignments(old_tree)
    constant_names = ("ROOT", "COURSE_URL", "READING_PAGES", "BOOTSTRAP") if filename.endswith("build_course.py") else ("ROOT", "REPOSITORY", "COURSE_URL")
    for name in constant_names:
        equal = ast.dump(current_assignments[name], include_attributes=False) == ast.dump(old_assignments[name], include_attributes=False)
        ast_checks[filename + ":constant:" + name] = equal
        assert equal
    if filename.endswith("export_course.py"):
        def lesson_loop(function):
            return next(node for node in function.body if isinstance(node, ast.For) and ast.unparse(node.target) == "(item, notebook)")
        loop_equal = ast.dump(lesson_loop(current_functions["build"]), include_attributes=False) == ast.dump(lesson_loop(old_functions["build"]), include_attributes=False)
        ast_checks[filename + ":build:single-lesson-loop"] = loop_equal
        assert loop_equal

index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
old_index = json.loads((INITIAL / "frozen-input/course/lesson-index.json").read_bytes())
entry = next(item for item in index if item["id"] == "1.1")
assert entry == next(item for item in old_index if item["id"] == "1.1")
link_targets = []
for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", current["R.1"].decode()):
    relative, _, fragment = target.partition("#")
    path = ROOT / "course" / relative
    assert path.is_file()
    assert fragment in sections(path.read_bytes())[0]
    link_targets.append(target)
assert len(link_targets) == 4

environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "device": "CPU text/byte/AST comparisons only; no tensor execution",
    "torch_installed_distribution": importlib.metadata.version("torch"),
    "nbformat_installed_distribution": importlib.metadata.version("nbformat"),
    "network": "No network requests; no external source reacquisition",
    "working_directory": str(Path.cwd()),
}
(HERE / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
measurements = {
    "reviewer_task": REVIEWER,
    "source_sha256": digest(current["R.1"]),
    "intro_sha256": digest(current["intro"]),
    "same_r1_original_bytes": True,
    "same_intro_original_bytes": True,
    "original_initial_pass_archive_sha256": digest((HERE / "initial-pass-R.1.json").read_bytes()),
    "original_frozen_whole_sha256": digest(original_raw),
    "current_frozen_recheck_whole_sha256": digest(current_raw),
    "changed_section_ids": changed,
    "actual_context_paragraphs_read": context_read,
    "dependency_checks": {"first_lesson_raw_section_unchanged": True, "warmup_raw_sections_unchanged": warmup_checks, "first_notebook_bytes_unchanged": True, "first_lesson_inventory_entry_unchanged": True, "original_r1_link_targets": link_targets, "selected_generation_ast_unchanged": ast_checks},
    "scope": "Original reviewer rechecks unchanged R.1/intro plus changed R.2/R.4 navigation paragraphs; original source/proof remains preserved. No reexecution of original fence, model, training, whole-site, reading-time, publication or remote service.",
}
(HERE / "measurements.json").write_text(json.dumps(measurements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(measurements, ensure_ascii=False, indent=2))
print("Original reviewer R.1 narrow context/dependency check passed; process exit 0.")
