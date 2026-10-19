"""檢查還原清單以外的讀取邊界；所有測試材料都放在指定的隔離區。"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return {"path": path, "sha256": sha(target.read_bytes())}


def load(path):
    spec = importlib.util.spec_from_file_location("boundary_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    area = args.area.resolve()
    repo = area / "repo"
    outside = area / "outside"
    repo.mkdir(parents=True)
    outside.mkdir()
    module = load(args.tool.resolve())
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    template = area / "empty-template"
    template.mkdir()
    subprocess.run(["git", "init", "-q", "--template=" + str(template), str(repo)], env=env, check=True)
    freeze = {key: {} for key in ("source_files", "criteria", "figures", "notebooks")}
    save(repo, module.FREEZE, freeze)
    save(repo, module.OVERLAY, {"files": []})
    save(repo, "scripts/sample.py", b"protected sample\n")
    validator = b'''from pathlib import Path
import hashlib

def audit_manifest(root, manifest, stage):
    raw = (Path(root) / "scripts/sample.py").read_bytes()
    return {"failures": [], "observed_sha256": hashlib.sha256(raw).hexdigest()}
'''
    save(repo, "docs/review-tools/composite_review.py", validator)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], env=env, check=True)
    opens = []

    def audit(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0]))
            if path.is_absolute() and path.is_relative_to(repo):
                resolved = path.resolve()
                if resolved.is_relative_to(outside):
                    opens.append({"requested": str(path), "resolved": str(resolved)})

    sys.addaudithook(audit)
    results = []
    for label, name in [("freeze_document", module.FREEZE), ("overlay_document", module.OVERLAY)]:
        target = repo / name
        raw = target.read_bytes()
        external = outside / (label + ".json")
        external.write_bytes(raw)
        target.unlink()
        target.symlink_to(external)
        opens.clear()
        restored = module.checked_restores(repo, [])
        results.append({"case": label, "return_count": len(restored), "external_reads": list(opens)})
        target.unlink()
        target.write_bytes(raw)
    external_scripts = outside / "scripts"
    (repo / "scripts").rename(external_scripts)
    (repo / "scripts").symlink_to(external_scripts, target_is_directory=True)
    links = []
    original_link = module.os.link

    def record_link(source, destination, *args, **kwargs):
        source = Path(source)
        if source.resolve().is_relative_to(outside):
            links.append({"source": str(source), "resolved": str(source.resolve()), "destination": str(destination)})
        return original_link(source, destination, *args, **kwargs)

    module.os.link = record_link
    try:
        replayed = module.replay(repo)
    finally:
        module.os.link = original_link
    results.append({"case": "tracked_source_parent", "result": replayed, "outside_sources_hardlinked": links,
                    "external_file_sha256": sha((external_scripts / "sample.py").read_bytes())})
    print(json.dumps({"command": sys.argv, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    main(parser.parse_args())
