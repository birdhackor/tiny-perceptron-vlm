"""Read-only recheck of W.7 after natural-route navigation and environment ignores."""

import ast
import hashlib
import json
import platform
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_finish_w_7_route_cache"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def nav_paths(items):
    for entry in items:
        if isinstance(entry, str):
            yield entry
        else:
            for value in entry.values():
                yield from nav_paths(value) if isinstance(value, list) else [value]


def command(argv, stdin=None):
    process = subprocess.run(argv, cwd=ROOT, input=stdin, text=True, capture_output=True, check=False)
    return {
        "command": argv,
        "stdin": stdin,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def main():
    report = json.loads(PREFIX.with_name(PREFIX.name + "_previous_report_bytes.json").read_text(encoding="utf-8"))
    recorded = {item["id"]: item for item in report["sources"]}
    old_config = PREFIX.with_name(PREFIX.name + "_config_old.txt").read_bytes()
    new_config = PREFIX.with_name(PREFIX.name + "_config_new.txt").read_bytes()
    old_ignore = PREFIX.with_name(PREFIX.name + "_ignore_old.txt").read_bytes()
    new_ignore = PREFIX.with_name(PREFIX.name + "_ignore_new.txt").read_bytes()
    assert sha(old_config) == recorded["s_config"]["sha256"]
    assert sha(old_ignore) == recorded["s_ignore"]["sha256"]
    assert sha(new_config) == "e1699f60e409d3f7e8cbea7ccf35d8934cb4db06c4818243aaebbb26e281f6d0"
    assert sha(new_ignore) == "019419ee40c740f1e027fcf9e1d6c6c953f0d3345e0813ec4947c4f2a35beeae"
    assert (ROOT / "zensical.toml").read_bytes() == new_config
    assert (ROOT / ".gitignore").read_bytes() == new_ignore
    old_settings = tomllib.loads(old_config.decode("utf-8"))["project"]
    settings = tomllib.loads(new_config.decode("utf-8"))["project"]
    assert settings["docs_dir"] == old_settings["docs_dir"] == "outputs/zensical/docs"
    assert settings["site_dir"] == old_settings["site_dir"] == "outputs/site"
    old_nav = list(nav_paths(old_settings["nav"]))
    new_nav = list(nav_paths(settings["nav"]))
    added = sorted(set(new_nav) - set(old_nav))
    removed = sorted(set(old_nav) - set(new_nav))
    assert not removed
    assert len(new_nav) == len(set(new_nav))
    index_path = ROOT / "course/lesson-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    export_path = ROOT / "scripts/export_course.py"
    tree = ast.parse(export_path.read_text(encoding="utf-8"))
    documents = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DOCUMENTS" for target in node.targets)
    )
    expected = {"index.md"} | {key + ".md" for key in documents}
    expected |= {item["id"] + ".md" for item in index}
    expected |= {"chapter-" + Path(item["source"]).stem + ".md" for item in index}
    assert set(new_nav) == expected
    selected = [item for item in index if item["id"] + ".md" in added]
    for item in selected:
        assert (ROOT / item["source"]).is_file()
        assert (ROOT / item["notebook"]).is_file()
    non_nav_differences = {
        key: {"old": old_settings.get(key), "new": settings.get(key)}
        for key in set(old_settings) | set(settings)
        if key != "nav" and old_settings.get(key) != settings.get(key)
    }
    assert not non_nav_differences
    ignore_added = sorted(set(new_ignore.decode().splitlines()) - set(old_ignore.decode().splitlines()))
    ignore_removed = sorted(set(old_ignore.decode().splitlines()) - set(new_ignore.decode().splitlines()))
    assert ignore_added == [".venv-natural/"]
    assert not ignore_removed
    probes = [
        "data/natural-photos/reviewer.jpg",
        "data/chinese-ocr/reviewer.png",
        "data/human-speech/reviewer.wav",
        "checkpoints/natural-adapter/reviewer.json",
        "outputs/natural-assistant/reviewer.json",
        ".cache/huggingface/reviewer.json",
        ".venv-natural/bin/python",
    ]
    ignored = command(["git", "check-ignore", "-v", "--stdin"], "\n".join(probes) + "\n")
    assert ignored["exit_code"] == 0
    assert len(ignored["stdout"].splitlines()) == len(probes)
    help_result = command([str(ROOT / ".venv/bin/python"), "-B", "scripts/export_course.py", "--help"])
    assert help_result["exit_code"] == 0
    natural_help = command([str(ROOT / ".venv/bin/python"), "-B", "scripts/natural_assistant.py", "--help"])
    assert natural_help["exit_code"] == 0
    assert "--data-root" in natural_help["stdout"]
    assert "--cache-dir" in natural_help["stdout"]
    print(
        json.dumps(
            {
                "environment": {"python": platform.python_version(), "device": "cpu; no model loaded"},
                "old_source_sha256": {"zensical.toml": sha(old_config), ".gitignore": sha(old_ignore)},
                "new_source_sha256": {"zensical.toml": sha(new_config), ".gitignore": sha(new_ignore)},
                "old_report_sha256": sha(PREFIX.with_name(PREFIX.name + "_previous_report_bytes.json").read_bytes()),
                "docs_dir": settings["docs_dir"],
                "site_dir": settings["site_dir"],
                "navigation_added": added,
                "navigation_removed": removed,
                "non_nav_project_differences": non_nav_differences,
                "navigation_count": len(new_nav),
                "all_navigation_matches_source_generation_targets": True,
                "added_lesson_metadata": selected,
                "ignore_lines_added": ignore_added,
                "ignore_lines_removed": ignore_removed,
                "git_check_ignore": ignored,
                "export_help": help_result,
                "natural_help": natural_help,
                "related_source_sha256": {
                    "scripts/export_course.py": sha(export_path.read_bytes()),
                    "course/lesson-index.json": sha(index_path.read_bytes()),
                    "scripts/natural_assistant.py": sha((ROOT / "scripts/natural_assistant.py").read_bytes()),
                    "tiny_perceptron/natural_assistant.py": sha(
                        (ROOT / "tiny_perceptron/natural_assistant.py").read_bytes()
                    ),
                },
                "scope": "Recheck W.7 directory purposes only: navigation is metadata, not model-quality evidence. CLI cache/output locations are caller-selected; only paths under the listed ignored roots are shown ignored. No site build, dataset read, model download, GPU command, training, test output or Git mutation.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
