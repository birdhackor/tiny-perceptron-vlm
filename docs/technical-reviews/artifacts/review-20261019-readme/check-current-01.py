# 核對凍結來源差異、沿用證據及本次 CPU 短流程，輸出只在自己的目錄。
import contextlib
import hashlib
import html
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_readme/technical-readme-original-02.json"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(command):
    result = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=90)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

def main():
    import torch
    original = json.loads(ORIGINAL.read_bytes())
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
              "original_report_sha256": sha(ORIGINAL), "sources": [], "artifacts": [], "commands": [], "network": []}
    for group in ("sources", "artifacts"):
        for entry in original[group]:
            if "path" in entry:
                current = sha(ROOT / entry["path"])
                result[group].append({"id": entry["id"], "path": entry["path"], "expected": entry["sha256"], "observed": current, "unchanged": current == entry["sha256"]})
    assert all(r["unchanged"] for k in ("sources", "artifacts") for r in result[k])
    for name in ("README.md", "docs/course-review-20261019/freeze-02/sources/readme.md", "docs/course-review-20261019/freeze-05/sources/readme.md", "docs/course-review-20261019/freeze-05.json", "course/figures/rewrite-01-character-ids.svg"):
        result.setdefault("hashes", {})[name] = sha(ROOT / name)
    old = "docs/course-review-20261019/freeze-02/sources/readme.md"
    new = "docs/course-review-20261019/freeze-05/sources/readme.md"
    result["source_diff"] = run(["git", "diff", "--no-index", old, new])
    assert result["source_diff"]["returncode"] == 1
    before = (ROOT / old).read_text()
    after = (ROOT / new).read_text()
    result["single_replacement_exact"] = before.replace("公式由 MathJax 連網排版。", "公式由網站隨附的 MathJax 在瀏覽器排版。") == after
    assert result["single_replacement_exact"]
    headings = {p.relative_to(ROOT).as_posix(): re.findall(r"^## ([\dABC]+\.\d+) (.+)$", p.read_text(), re.M) for p in sorted((ROOT / "course/chapters").glob("*.md"))}
    extra = {name: re.findall(r"^## (\S+\.\d+) (.+)$", (ROOT/name).read_text(), re.M) for name in ("course/README.md", "course/first-steps.md", "course/training.md", "course/glossary.md")}
    result["catalogue"] = {"chapters": headings, "auxiliary": extra, "chapter_sections": sum(map(len, headings.values())), "auxiliary_sections": sum(map(len, extra.values())), "notebooks": len(list((ROOT/"notebooks").rglob("*.ipynb")))}
    uv = ["uv", "run", "--no-sync", "python"]
    result["commands"].append(run(uv + ["scripts/build_course.py", "--check"]))
    result["commands"].append(run(uv + ["scripts/fetch_training_assets.py", "--list"]))
    result["commands"].append(run(uv + ["scripts/fetch_course_models.py", "--list"]))
    with tempfile.TemporaryDirectory(prefix="scratch-current-", dir=HERE) as scratch:
        scratch = Path(scratch)
        result["commands"].append(run(uv + ["scripts/prepare_data.py", "--kind", "toy-text", "--output", str(scratch/"data")]))
        data = scratch / "data/toy-text/train.jsonl"
        target = scratch / "unexpected.pt"
        command = uv + ["scripts/train.py", "--task", "text", "--data", str(data), "--device", "cpu", "--output", str(target)]
        result["commands"].append(run(command))
        result["inputs"] = {p.name: p.read_text() for p in sorted(data.parent.iterdir())}
        from scripts import train
        optimizers = []
        actual_adamw = torch.optim.AdamW
        class CheckedAdamW(actual_adamw):
            def __init__(self, params, *args, **kwargs):
                params = list(params)
                self.checked = params
                self.before = [p.detach().clone() for p in params]
                optimizers.append(self)
                super().__init__(params, *args, **kwargs)
            def step(self, *args, **kwargs):
                raise AssertionError("dry-run 不應更新權重")
        stream = io.StringIO()
        argv = command[4:]
        with patch.object(torch.optim, "AdamW", CheckedAdamW), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stream):
            train.main()
        result["instrumented_dry_run"] = {"argv": argv, "stdout": stream.getvalue(), "parameters_unchanged": all(torch.equal(p,b) for o in optimizers for p,b in zip(o.checked,o.before)), "checkpoint_exists": target.exists()}
        assert result["instrumented_dry_run"]["parameters_unchanged"] and not target.exists()
    for ident, url, phrase in (
        ("mathjax_typeset", "https://docs.mathjax.org/en/v3.2/web/typeset.html", "MathJax makes it easy to typeset all the math in a web page"),
        ("mathjax_hosting", "https://docs.mathjax.org/en/v3.2/web/hosting.html", "if you wish to use MathJax off-line."),
        ("public_page", "https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html", ""),
    ):
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
        process = subprocess.run(command, capture_output=True, timeout=45)
        raw = process.stdout
        text = raw.decode(errors="replace")
        plain = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", text)))
        item = {"id":ident,"url":url,"command":command,"returncode":process.returncode,"stderr":process.stderr.decode(errors="replace"),"raw_sha256":hashlib.sha256(raw).hexdigest(),"bytes_read":len(raw),"accessed_on":"2026-10-20","locator":phrase,"found":phrase in plain,"excerpt":phrase}
        if ident == "public_page":
            item["scripts"] = re.findall(r'<script[^>]+src="([^"]+)"', text)
            assert "./javascripts/mathjax/tex-chtml.js" in item["scripts"]
        assert process.returncode == 0 and item["found"]
        result["network"].append(item)
    manifest = json.loads((ROOT / "course/web/mathjax-assets.json").read_text())
    from scripts.stage_mathjax import validate_manifest
    validate_manifest(manifest)
    result["mathjax"] = {"version":manifest["version"],"revision":manifest["revision"],"files":len(manifest["files"]),"main":next(f for f in manifest["files"] if f["path"] == "tex-chtml.js"),"font":next(f for f in manifest["files"] if f["path"] == "output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff")}
    for name in ("course/web/mathjax.js", "course/web/mathjax-assets.json"):
        result["hashes"][name] = sha(ROOT/name)
    result["figure"] = {"viewed_png":"outputs/course-review-20261019/figure-renders-04/61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad.png","visual_inspection":"已用 view_image 目視；上排先讀、下排後讀，各字與ID、雙向箭頭、同字同色及底部標註一致。"}
    assert all(p["returncode"] == 0 for p in result["commands"])
    target = HERE / "check-current-01.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target),"source_files_unchanged":len(result["sources"]),"artifacts_unchanged":len(result["artifacts"]),"counts":{k:result["catalogue"][k] for k in ("chapter_sections","auxiliary_sections","notebooks")},"dry_run":result["instrumented_dry_run"],"mathjax":result["mathjax"]},ensure_ascii=False))

if __name__ == "__main__":
    main()
