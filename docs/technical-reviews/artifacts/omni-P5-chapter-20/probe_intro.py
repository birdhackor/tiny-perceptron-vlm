"""Reviewer-owned checks for the chapter-20 introduction only.

No trained checkpoint is loaded. The only neural operation executed is the
chapter's artificial 4-by-3 LoRA teaching example. Saved validation rows are
read as JSON and their termination fields are recomputed without inference.
"""

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}


def main():
    import torch

    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json"
    freeze = json.loads(freeze_path.read_text())
    page = next(page for page in freeze["pages"] if page["page_id"] == "chapter-20")
    snapshot = ROOT / page["snapshot"]
    chapter = ROOT / page["source"]
    raw = snapshot.read_bytes()
    current = chapter.read_bytes().split(b"\n## ", 1)[0] + b"\n"
    assert digest(snapshot) == page["source_sha256"]
    assert raw == current
    assert page["figures_sha256"] == {}
    body = raw.decode()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_chapter_20",
        "scope": "canonical chapter-20 introduction, before first ##; other code blocks inspected/executed only to verify its blanket demonstration claim",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "platform": platform.platform()},
        "canonical_freeze": ref(freeze_path),
        "assigned_source": {**ref(snapshot), "current_selector_equal": True, "paragraphs": len(body.strip().split("\n\n")) - 1, "figure_count": 0},
    }

    links = []
    for label, url in re.findall(r"\[([^]]+)\]\(([^)]+)\)", body):
        parsed = urlsplit(url)
        assert not parsed.scheme
        target = (chapter.parent / unquote(parsed.path)).resolve()
        assert target.is_file()
        anchor = unquote(parsed.fragment)
        heading = None
        if anchor:
            match = re.search(r"^## " + re.escape(anchor) + r"\s.*$", target.read_text(), re.M)
            assert match, (target, anchor)
            heading = match.group()
        links.append({"label": label, "href": url, "target": str(target.relative_to(ROOT)), "anchor": anchor, "heading": heading, "exists": True})
    preview = ROOT / "outputs/omni-P5-preview-v9/chapter-20.html"
    html = preview.read_text()
    article = html[html.index("<article"):html.index("</article>")]
    preview_links = []
    for label in [link["label"] for link in links]:
        match = re.search(r'<a href="([^"]+)">' + re.escape(label) + r"</a>", article)
        assert match, label
        url = match.group(1)
        path = (preview.parent / urlsplit(url).path).resolve()
        assert path.is_file(), path
        preview_links.append({"label": label, "href": url, "target_exists": True})
    result["navigation"] = {"markdown_links": links, "preview_html": ref(preview), "preview_links": preview_links, "scope": "parsed HTML link targets; no browser layout or Colab execution claimed"}

    code_rows = []
    child = '''import contextlib,io,json,sys,time
code = json.loads(sys.stdin.read())
buf = io.StringIO()
started = time.perf_counter()
with contextlib.redirect_stdout(buf):
    exec(compile(code, "<chapter-20-artificial-example>", "exec"), {})
elapsed = time.perf_counter() - started
print(json.dumps({"stdout":buf.getvalue(),"execution_seconds":elapsed,"mature_modules_loaded":[m for m in sys.modules if m == "transformers" or m.startswith("transformers.") or m == "whisper" or m.startswith("whisper.")]}))
'''
    code_count = 0
    for section in re.split(r"(?=^## )", chapter.read_text(), flags=re.M):
        heading = section.splitlines()[0]
        for code in re.findall(r"^```python\n(.*?)^```", section, re.M | re.S):
            code_count += 1
            tree = ast.parse(code)
            imports = [node.module if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            assert "from_pretrained" not in code and "load_model" not in code and "torch.load" not in code
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT))
            started = time.perf_counter()
            proc = subprocess.run([sys.executable, "-c", child], input=json.dumps(code), text=True, capture_output=True, cwd=ROOT, env=env, timeout=30, check=True)
            wall = time.perf_counter() - started
            row = json.loads(proc.stdout)
            assert not row["mature_modules_loaded"]
            code_rows.append({"heading": heading, "source_sha256": hashlib.sha256(code.encode()).hexdigest(), "imports": imports, "process_wall_seconds": wall, "stderr": proc.stderr, **row})
    assert code_count == 14
    result["artificial_examples"] = {"chapter_source": ref(chapter), "count": code_count, "rows": code_rows, "maximum_process_wall_seconds": max(row["process_wall_seconds"] for row in code_rows), "scope": "one fresh process per exact fenced Python block, including import time; all on CPU, no installation/download/launch Bash commands executed"}

    training = ROOT / "outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review/training.json"
    tx = json.loads(training.read_text())
    keys = ["model", "model_revision", "asr_model", "asr_revision", "completed_steps", "requested_steps", "total_parameters", "trainable_parameters", "lora_rank", "lora_targets", "status", "optimizer_only_lora", "frozen_parameter_samples_unchanged", "frozen_check_scope", "changed_adapter_tensor_count"]
    names = tx["optimizer_parameter_names"]
    assert names == tx["trainable_parameter_names"] and names and all("lora_" in name for name in names)
    assert tx["completed_steps"] == 2077 and tx["changed_adapter_tensor_count"] == 112
    result["historical_lora_training"] = {"original": ref(training), "selected_fields": {key: tx[key] for key in keys}, "optimizer_names_count": len(names), "all_optimizer_parameters_are_lora": True, "archives": [{"completed_steps": item["completed_steps"], "adapter_model": next(file for file in item["files"] if file["path"] == "adapter_model.safetensors")} for item in tx["archived_checkpoints"]], "scope": "read saved training metadata; no fresh training; frozen base samples are not an all-values bitwise audit"}

    rawdir = ROOT / "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw"
    variants = []
    for variant in ["base", "adapter-step-001039", "adapter-step-002077"]:
        path = rawdir / ("generations-" + variant + ".json")
        rows = json.loads(path.read_text())
        assert len(rows) == 132
        failures = []
        for row in rows:
            tokens = row["generated_token_ids"]
            complete = bool(tokens and len(tokens) <= 384 and tokens[-1] in row["eos_token_ids"] and row["generated_tokens"] == len(tokens) and row["ended_with_eos"] is True and row["stop_reason"] == "eos" and row["truncated"] is False and row["completion_unknown"] is False)
            if not complete:
                failures.append({"id": row["id"], "task": row["task"], "generated_tokens": len(tokens), "last_token": tokens[-1] if tokens else None, "eos_token_ids": row["eos_token_ids"], "ended_with_eos": row["ended_with_eos"], "stop_reason": row["stop_reason"], "truncated": row["truncated"]})
        variants.append({"variant": variant, "source": ref(path), "denominator": len(rows), "complete_count": len(rows) - len(failures), "incomplete_count": len(failures), "all_eos_gate": not failures, "incomplete_rows": failures})
    assert [item["incomplete_count"] for item in variants] == [6, 3, 3]
    selection = ROOT / "docs/natural-assistant/v4/selection.json"
    release = ROOT / "docs/natural-assistant/v4/public-release.json"
    selected = json.loads(selection.read_text())
    public = json.loads(release.read_text())
    assert selected["selected_variant"] == public["selected_variant"] == "base"
    assert public["adapter_parameters"] == 0
    result["candidate_rejection"] = {"variants": variants, "selection": ref(selection), "release": ref(release), "selected_variant": selected["selected_variant"], "public_adapter_parameters": public["adapter_parameters"], "scope": "objective all-EOS gate is a necessary condition and both adapters fail it; no model inference and no new subjective grading or selection"}

    random_rows = []
    for name in ["moe-pretrain", "moe-native"]:
        path = ROOT / f"docs/selftrained/results/training-raw/{name}/raw/train-receipt.json"
        record = json.loads(path.read_text())
        assert record["origin"]["kind"] == "all-neural-weights-random"
        random_rows.append({"source": ref(path), "stage": record["stage"], "origin_kind": record["origin"]["kind"], "seed": record["origin"]["seed"], "stage_history_count": len(record["stage_history"])})
    settings = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
    setting = json.loads(settings.read_text())
    for product in ["b", "c"]:
        path = ROOT / f"docs/lesson-products/results/initial-{product}-execution.json"
        record = json.loads(path.read_text())
        train = record["training"]
        assert train["random_initialization"] is True and train["weights_changed"] is True
        assert record["client_artifacts"]["model.pt"]["sha256"] == setting["products"][product]["model.pt"]
        assert record["client_artifacts"]["training-receipt.json"]["sha256"] == setting["products"][product]["training-receipt.json"]
        random_rows.append({"source": ref(path), "product": product, "random_initialization": train["random_initialization"], "seed": train["seed"], "initial_weights_sha256": train["initial_weights_sha256"], "selected_weights_sha256": train["selected_weights_sha256"], "weights_changed": train["weights_changed"], "matches_pretest_settings_model_and_receipt": True})
    result["three_product_origin"] = {"rows": random_rows, "pretest_settings": ref(settings), "scope": "saved run metadata and constructor/trainer source inspection; no checkpoints loaded and no final-family forward executed"}
    path = OUT / "probe-results-v1.json"
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path), "examples": code_count, "maximum_process_wall_seconds": result["artificial_examples"]["maximum_process_wall_seconds"], "candidate_incomplete_counts": [item["incomplete_count"] for item in variants]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
