"""Rebuild and publish only root-reviewed native Git LFS data objects.

The initial workflow push selects inspect. A publish push must change the one
explicit control descriptor and bind the frozen package recipe SHA-256.
No Modal, HF token, GPU or model weights are involved.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "docs/selftrained/data-publish.json"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\n" in value or "\r" in value:
        raise ValueError("Need a plain relative path without traversal or line breaks")
    return path


def committed_json(root, path, revision):
    path = relative(path)
    if path.parts[:2] != ("docs", "selftrained") or path.suffix != ".json":
        raise ValueError("Control and recipe must be committed JSON under docs/selftrained")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Need actual full dispatch Git SHA")
    raw = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"], cwd=root, capture_output=True, check=True
    ).stdout
    if (Path(root) / path).read_bytes() != raw:
        raise ValueError("File differs from selected committed Git revision")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_recipe(recipe):
    if recipe.get("schema_version") != 1 or recipe.get("frozen") is not True:
        raise ValueError("Publication requires an already frozen root-reviewed package recipe")
    if not re.fullmatch(r"3\.13\.[0-9]+", recipe.get("python_version", "")):
        raise ValueError("Recipe needs an exact Python 3.13 patch version")
    dependencies = recipe.get("dependencies")
    if not isinstance(dependencies, dict) or len(dependencies) > 20:
        raise ValueError("Recipe requires a bounded exact dependency-version map")
    for name, version in dependencies.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or not re.fullmatch(r"[0-9][A-Za-z0-9_.+-]*", version):
            raise ValueError("Dependency pins must be package names and versions, never URLs or pip flags")
    archive = recipe.get("archive", {})
    path = relative(archive.get("path", ""))
    if path.parts[:2] != ("assets", "training") or not re.fullmatch(r"selftrained-[A-Za-z0-9_.-]+\.tar\.gz", path.name):
        raise ValueError("Publication is limited to a selftrained tar.gz under assets/training")
    if not re.fullmatch(r"[a-f0-9]{64}", archive.get("sha256", "")):
        raise ValueError("Archive must already have frozen SHA-256")
    for field, maximum in (("bytes", 512 * 1024 * 1024), ("unpacked_bytes", 1024 * 1024 * 1024)):
        if type(archive.get(field)) is not int or not 0 < archive[field] <= maximum:
            raise ValueError("Archive size exceeds the bounded data publication")
    if not recipe.get("source_files") or not recipe.get("producers"):
        raise ValueError("Recipe needs actual pinned files and reconstruction producers")
    return recipe


def select_operation(root, event, inputs, revision):
    operation, recipe_path, expected_sha = "inspect", "", ""
    if event.get("event_name") == "workflow_dispatch":
        operation = inputs.get("operation", "inspect")
        if operation == "publish":
            recipe_path = inputs.get("recipe", "")
    elif event.get("event_name") == "push":
        before = event.get("before", "")
        if re.fullmatch(r"[a-f0-9]{40}", before) and before != "0" * 40:
            changed = subprocess.run(
                ["git", "diff", "--name-only", before, revision, "--", CONTROL],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            if CONTROL in changed:
                control, _ = committed_json(root, CONTROL, revision)
                if control.get("operation") == "publish":
                    if control.get("approved") is not True:
                        raise ValueError("Publish descriptor needs explicit root-reviewed approval")
                    operation, recipe_path, expected_sha = (
                        "publish",
                        control.get("recipe", ""),
                        control.get("recipe_sha256", ""),
                    )
                    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
                        raise ValueError("Publish descriptor must bind exact frozen recipe SHA")
    if operation not in ("inspect", "publish"):
        raise ValueError("Only inspect or explicit publish is supported")
    selection = {
        "operation": operation,
        "source_git_sha": revision,
        "gpu_started": False,
        "modal_used": False,
        "publication_started": False,
    }
    if operation == "publish":
        recipe, recipe_sha = committed_json(root, recipe_path, revision)
        validate_recipe(recipe)
        if expected_sha and expected_sha != recipe_sha:
            raise ValueError("Publish descriptor differs from exact committed recipe bytes")
        selection.update(recipe=recipe_path, recipe_sha256=recipe_sha, python_version=recipe["python_version"])
    return selection


def verify_archive(root, recipe, revision, output_dir):
    validate_recipe(recipe)
    archive = recipe["archive"]
    path = relative(archive["path"])
    actual = Path(output_dir) / path.name
    if (
        not actual.is_file()
        or actual.is_symlink()
        or actual.stat().st_size != archive["bytes"]
        or digest(actual) != archive["sha256"]
    ):
        raise ValueError("Reconstructed archive differs from frozen recipe bytes/hash")
    expected = f"version https://git-lfs.github.com/spec/v1\noid sha256:{archive['sha256']}\nsize {archive['bytes']}\n".encode()
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", revision, "--", path.as_posix()],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    pointer_present = bool(listing)
    if pointer_present:
        committed = subprocess.run(
            ["git", "show", f"{revision}:{path.as_posix()}"], cwd=root, capture_output=True, check=True
        ).stdout
        if committed != expected:
            raise ValueError("Selected Git commit does not contain the exact reviewed LFS pointer")
    return actual, path, expected, pointer_present


def publish(root, recipe_path, revision, output_dir):
    recipe, recipe_sha = committed_json(root, recipe_path, revision)
    actual, path, expected, pointer_present = verify_archive(root, recipe, revision, output_dir)
    subprocess.run(["git", "lfs", "install", "--local"], cwd=root, check=True)
    target = Path(root) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(actual, target)
    subprocess.run(["git", "add", "--", path.as_posix()], cwd=root, check=True)
    staged = subprocess.run(["git", "show", f":{path.as_posix()}"], cwd=root, capture_output=True, check=True).stdout
    if staged != expected:
        raise ValueError("Native Git LFS did not stage the frozen object pointer")
    changed = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    if any(name != path.as_posix() for name in changed):
        raise ValueError("Refusing publication with an unrelated staged change")
    subprocess.run(["git", "lfs", "push", "--object-id", "origin", recipe["archive"]["sha256"]], cwd=root, check=True)
    return {
        "source_git_sha": revision,
        "recipe_sha256": recipe_sha,
        "archive": recipe["archive"],
        "lfs_upload_completed": True,
        "pointer_committed": pointer_present,
        "pointer_not_committed_yet": not pointer_present,
        "gpu_started": False,
        "modal_used": False,
        "hf_used": False,
        "scope": "Actual native Git LFS object push succeeded; existing pointer verified when present; missing pointer requires root's later commit/push; no Git commit or model publication",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("select", "verify", "publish"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--recipe", default="docs/selftrained/package-recipe.json")
    parser.add_argument("--output-dir", default="outputs/selftrained/data-publish")
    args = parser.parse_args()
    output = ROOT / relative(args.output_dir)
    if args.operation == "select":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        event["event_name"] = os.environ["GITHUB_EVENT_NAME"]
        selection = select_operation(
            ROOT,
            event,
            {
                "operation": os.environ.get("SELFTRAINED_DATA_OPERATION", "inspect"),
                "recipe": os.environ.get("SELFTRAINED_DATA_RECIPE", args.recipe),
            },
            args.revision,
        )
        write_json(output / "selection.json", selection)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            for field in ("operation", "recipe", "python_version"):
                stream.write(f"{field}={selection.get(field, '')}\n")
        print(json.dumps(selection))
    elif args.operation == "verify":
        recipe, recipe_sha = committed_json(ROOT, args.recipe, args.revision)
        _, _, _, pointer_present = verify_archive(ROOT, recipe, args.revision, output)
        write_json(
            output / "verified.json",
            {
                "source_git_sha": args.revision,
                "recipe_sha256": recipe_sha,
                "archive": recipe["archive"],
                "lfs_uploaded": False,
                "pointer_committed": pointer_present,
                "pointer_not_committed_yet": not pointer_present,
            },
        )
    else:
        receipt = publish(ROOT, args.recipe, args.revision, output)
        write_json(output / "upload-receipt.json", receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
