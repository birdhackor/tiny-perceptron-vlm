"""Retrieve anonymous pinned evidence and save small original-source snapshots."""

import ast
import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_10_"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch_one(path):
    revision = "1df335318bda03fd771807f66976953231d5a00b"
    url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{revision}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "independent-technical-review"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
        destination = OUT / f"{PREFIX}pinned_{Path(path).name}"
        destination.write_bytes(raw)
        return {
            "url": url,
            "local": path,
            "snapshot": str(destination.relative_to(ROOT)),
            "sha256": sha(raw),
            "matches_local": raw == (ROOT / path).read_bytes(),
            "method": "anonymous HTTPS GET",
        }
    except Exception as error:
        raw = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        destination = OUT / f"{PREFIX}pinned_{Path(path).name}"
        destination.write_bytes(raw)
        return {
            "url": url,
            "local": path,
            "snapshot": str(destination.relative_to(ROOT)),
            "sha256": sha(raw),
            "matches_local": raw == (ROOT / path).read_bytes(),
            "method": "local immutable Git commit blob",
            "https_error": f"{type(error).__name__}: {error}",
        }


def main():
    paths = [
        "docs/course-experiments/results/capstone_deployment.json",
        "docs/course-experiments/results/capstone_student.json",
        *[
            f"docs/course-experiments/capstone-evidence/deployment/{n}.json"
            for n in ["test-joint", "test-joint-ptq4", "test-joint-ptq8"]
        ],
        *[
            f"docs/course-experiments/capstone-evidence/student/{n}.json"
            for n in ["test-ce", "test-kd", "test-kd-ptq4"]
        ],
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        evidence = list(pool.map(fetch_one, paths))
    for row in evidence:
        assert row["matches_local"]
    authorities = []
    for filename, start, end in [("3.5-fresh/distilling-v1", 1, 179), ("gptq-2022", 1, 149)]:
        pdf = ROOT / f"outputs/technical-sources/{filename}.pdf"
        raw = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"])
        lines = raw.decode().splitlines(keepends=True)
        name = filename.split("/")[-1]
        destination = OUT / f"{PREFIX}{name}_original_excerpt.txt"
        destination.write_text("".join(lines[start - 1 : end]))
        authorities.append(
            {
                "paper": filename,
                "pdf_sha256": sha(pdf.read_bytes()),
                "fresh_pdftotext_sha256": sha(raw),
                "excerpt_lines": [start, end],
                "snapshot": str(destination.relative_to(ROOT)),
                "excerpt_sha256": sha(destination.read_bytes()),
            }
        )
    commit = "5c4886908584029761b579af026dcfb627c84070"
    selections = {
        "torch/nn/modules/linear.py": ["Linear"],
        "torch/autograd/grad_mode.py": ["no_grad"],
        "torch/random.py": ["manual_seed"],
        "torch/nn/functional.py": ["kl_div"],
    }
    for path, symbols in selections.items():
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{path}"
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
        source = raw.decode()
        lines = source.splitlines(keepends=True)
        selected = []
        locators = []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in symbols:
                selected.append(
                    f"# Original {path} lines {node.lineno}-{node.end_lineno}\n"
                    + "".join(lines[node.lineno - 1 : node.end_lineno])
                )
                locators.append([node.name, node.lineno, node.end_lineno])
        assert len(selected) == len(symbols)
        destination = OUT / f"{PREFIX}torch_{Path(path).name}.txt"
        destination.write_text("\n\n".join(selected))
        authorities.append(
            {
                "url": url,
                "version": "2.14.1 / " + commit,
                "original_sha256": sha(raw),
                "locators": locators,
                "snapshot": str(destination.relative_to(ROOT)),
                "excerpt_sha256": sha(destination.read_bytes()),
                "method": "anonymous official-source HTTPS GET",
            }
        )
    public = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    names = [
        "joint",
        "joint-int4",
        "joint-int8",
        "dpo",
        "dpo-int4",
        "dpo-int8",
        "student-ce",
        "student-kd",
        "student-kd-int4",
    ]
    public_paths = {}
    public_receipts = []
    for name in names:
        model = next(m for m in public["models"] if m["id"] == name)
        file = next(f for f in model["files"] if f["output"] == "model.pt")
        path = hf_hub_download(
            public["repo"],
            filename=file["path"],
            revision=public["revision"],
            token=False,
            local_dir=ROOT / "outputs/technical-sources/fact_v2_19_10_public",
        )
        raw = Path(path).read_bytes()
        assert len(raw) == file["bytes"] and sha(raw) == file["sha256"]
        public_paths[name] = path
        public_receipts.append(
            {
                "id": name,
                "repo": public["repo"],
                "revision": public["revision"],
                "filename": file["path"],
                "token": False,
                "bytes": len(raw),
                "sha256": sha(raw),
            }
        )
    (OUT / f"{PREFIX}source_receipts.json").write_text(
        json.dumps(
            {"pinned_evidence": evidence, "authorities": authorities, "public_hf": public_receipts},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (ROOT / "outputs/technical-sources/fact_v2_19_10_public/paths.json").write_text(json.dumps(public_paths) + "\n")
    print(
        "Pinned evidence",
        len(evidence),
        "; official sources",
        len(authorities),
        "; anonymous HF weights",
        len(public_receipts),
    )


if __name__ == "__main__":
    main()
