"""Replay R.1's local CPU and saved public-entry checks without installing tools."""

import hashlib
import io
import json
import platform
import re
import sys
import tomllib
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
import torch  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from jupyter_client.kernelspec import KernelSpecManager  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from scripts.build_course import notebook as generate_notebook  # noqa: E402
from scripts.check_technical_reviews import sections  # noqa: E402
from scripts.export_course import checked_notebook, joined, output_html, reading_markdown  # noqa: E402

ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_r_1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def saved(name, suffix="data"):
    return ARTIFACTS / f"{PREFIX}-{name}.{suffix}"


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return "".join(text.replace("`", "").split())


def exercise(code):
    namespace = {}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(code, "<reviewed-example>", "exec"), namespace)
    return namespace, output.getvalue()


def nav_paths(items):
    for item in items:
        if isinstance(item, str):
            yield item
        else:
            for value in item.values():
                if isinstance(value, list):
                    yield from nav_paths(value)
                else:
                    yield value


def main():
    os_environment = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "device": "CPU; no accelerator requested",
        "platform": platform.platform(),
        "nbclient": version("nbclient"),
        "nbformat": version("nbformat"),
        "beautifulsoup4": version("beautifulsoup4"),
    }
    receipt_records = []
    for label in ["initial", "actions", "originals"]:
        receipt = json.loads(saved(f"fetch-{label}", "json").read_text(encoding="utf-8"))
        for record in receipt["records"]:
            if "error" in record:
                # The first attempt to a guessed versioned docs route was 404.
                receipt_records.append(record)
                continue
            data = (ROOT / record["path"]).read_bytes()
            assert digest(data) == record["sha256"]
            assert record["status"] == 200
            receipt_records.append(record)

    guide_path = ROOT / "course/README.md"
    raw_guide = guide_path.read_text(encoding="utf-8")
    intro = raw_guide[: re.search(r"^## ", raw_guide, re.M).start()]
    assert intro == saved("intro", "md").read_text(encoding="utf-8")
    r1 = dict(sections(guide_path))["R.1"]
    assert r1 == saved("source-R.1", "md").read_text(encoding="utf-8")
    assert next(iter(sections(guide_path)))[0] == "R.1"

    source_path = ROOT / "course/chapters/01.md"
    section = dict(sections(source_path))["1.1"]
    assert section == saved("source-1.1", "md").read_text(encoding="utf-8")
    title_line, body = section.split("\n", 1)
    generated = generate_notebook("1.1", title_line.removeprefix("## 1.1 "), body, source_path)
    serialized = (json.dumps(generated, ensure_ascii=False, indent=1) + "\n").encode()
    local_notebook_bytes = (ROOT / "notebooks/01/1.1.ipynb").read_bytes()
    assert serialized == local_notebook_bytes
    assert saved("site-download-notebook").read_bytes() == local_notebook_bytes
    assert saved("colab-pinned-notebook").read_bytes() == local_notebook_bytes
    original = json.loads(local_notebook_bytes)
    nbformat.validate(nbformat.from_dict(original))

    site = BeautifulSoup(saved("site-1.1", "html").read_text(encoding="utf-8"), "html.parser")
    article = site.find("article")
    site_paragraphs = {plain(p.get_text()) for p in article.find_all("p")}
    prose_paragraphs = []
    for cell in original["cells"][1:]:
        if cell["cell_type"] == "markdown":
            for paragraph in joined(cell["source"]).strip().split("\n\n"):
                normalized = plain(paragraph)
                assert normalized in site_paragraphs, paragraph
                prose_paragraphs.append(normalized)
    example_cell = next(cell for cell in original["cells"] if cell["cell_type"] == "code" and not cell["metadata"])
    example_code = joined(example_cell["source"])
    published_code = next(c.get_text() for c in article.find_all("code") if "\n" in c.get_text())
    assert published_code.rstrip() == example_code.rstrip()

    kernel = KernelSpecManager().get_kernel_spec("tiny-perceptron")
    assert Path(kernel.argv[0]).resolve() == Path(sys.executable).resolve()
    executed = NotebookClient(
        nbformat.reads(local_notebook_bytes.decode("utf-8"), as_version=4),
        timeout=90,
        kernel_name="tiny-perceptron",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    executed_path = saved("executed-1.1", "ipynb")
    nbformat.write(executed, executed_path)
    checked = checked_notebook(original, executed_path)
    code_results = [
        c
        for c in checked["cells"]
        if c["cell_type"] == "code" and not c["metadata"].get("course_setup") and "course_figure" not in c["metadata"]
    ]
    actual_stdout = "".join(joined(o["text"]) for o in code_results[0]["outputs"] if o["output_type"] == "stream")
    expected_stdout = "['。', '狗', '看', '貓', '，']\n[3, 2, 1, 4, 1, 2, 3, 0]\n貓看狗，狗看貓。\n"
    assert actual_stdout == expected_stdout == article.select_one(".output pre").get_text()
    assert "實際執行結果 · CPU" in output_html(code_results[0]["outputs"])
    visible_copy = saved("visible-rendered-1.1", "md").read_text(encoding="utf-8")
    assert re.search(r"```python\n(.*?)```", visible_copy, re.S)[1].rstrip() == example_code.rstrip()
    assert re.search(r"```text\n(.*?)```", visible_copy, re.S)[1].rstrip() == expected_stdout.rstrip()
    namespace, short_stdout = exercise(example_code)
    assert short_stdout == expected_stdout
    assert namespace["ids"][0] == namespace["ids"][6] == 3
    reversed_namespace, reversed_stdout = exercise(
        example_code.replace("sorted(set(text))", "sorted(set(text), reverse=True)")
    )
    assert reversed_namespace["restored"] == namespace["restored"]
    assert reversed_namespace["ids"] != namespace["ids"]

    w2 = dict(sections(ROOT / "course/first-steps.md"))["W.2"]
    assert w2 == saved("source-W.2", "md").read_text(encoding="utf-8")
    w2_code = re.search(r"```python\n(.*?)```", w2, re.S)[1]
    _, w2_stdout = exercise(w2_code)
    assert w2_stdout == "貓 2\n貓在睡覺\n看到 貓\n看到 狗\n"
    w1 = dict(sections(ROOT / "course/first-steps.md"))["W.1"]
    assert w1 == saved("source-W.1", "md").read_text(encoding="utf-8")
    _, changed_stdout = exercise(example_code.replace("貓看狗，狗看貓。", "鳥看狗，狗看鳥。"))
    assert changed_stdout.splitlines()[-1] == "鳥看狗，狗看鳥。"

    guide = BeautifulSoup(saved("site-guide", "html").read_text(encoding="utf-8"), "html.parser")
    heading = guide.find("article").find(id="R.1")
    published_guide_paragraphs = []
    for sibling in heading.next_siblings:
        if getattr(sibling, "name", None) == "h2":
            break
        if getattr(sibling, "name", None) == "p":
            published_guide_paragraphs.append(plain(sibling.get_text()))
    assert [plain(p) for p in r1.split("\n\n")[1:] if p.strip()] == published_guide_paragraphs
    warmup = BeautifulSoup(saved("site-warmup", "html").read_text(encoding="utf-8"), "html.parser")
    assert all(warmup.find(id=anchor) is not None for anchor in ["W.1", "W.2"])
    for page in [site, guide, warmup]:
        assert any(
            urljoin("https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html", a.get("href", ""))
            == "https://birdhackor.github.io/tiny-perceptron-vlm/first-steps.html"
            for a in page.select("nav a")
        )
    settings = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
    assert "first-steps.md" in list(nav_paths(settings["nav"]))
    targets = {(ROOT / "course/first-steps.md").resolve(): "first-steps.md"}
    lesson_targets = {source_path.resolve(): {"1.1": "1.1.md"}}
    transformed = reading_markdown(r1, guide_path, targets, lesson_targets, "main")
    assert "(1.1.md)" in transformed
    assert "(first-steps.md#W.1)" in transformed and "(first-steps.md#W.2)" in transformed

    record = {
        "reviewer_task": "/root/fact_finish_r_1",
        "executed_at": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_r_1-replay.py",
        "environment": os_environment,
        "source_sha256": digest(r1.encode()),
        "intro_sha256": digest(intro.encode()),
        "notebook_sha256": digest(local_notebook_bytes),
        "public_notebook_bytes_equal_local": True,
        "colab_pinned_notebook_bytes_equal_local": True,
        "prose_paragraphs_equal": len(prose_paragraphs),
        "code_equal": True,
        "executed_notebook_cells": [c["execution_count"] for c in checked["cells"] if c["cell_type"] == "code"],
        "example_stdout": actual_stdout,
        "visible_rendered_receipt": {
            "origin": "outputs/reading-time/rendered/1.1.md",
            "persistent_copy": saved("visible-rendered-1.1", "md").relative_to(ROOT).as_posix(),
            "sha256": digest(visible_copy.encode()),
            "inspection": "Preexisting root-rendered CPU text personally read this round; code and output equal the current source and this reviewer's fresh CPU execution.",
        },
        "same_character_ids": [namespace["ids"][0], namespace["ids"][6]],
        "restored": namespace["restored"],
        "reverse_exercise_stdout": reversed_stdout,
        "w2_example_stdout": w2_stdout,
        "w1_changed_text_stdout": changed_stdout,
        "public_R1_paragraphs_equal": len(published_guide_paragraphs),
        "warmup_anchors_and_shared_nav": True,
        "website_download_and_colab_actions": [a.get("href") for a in article.select(".actions a")],
        "public_fetch_records": receipt_records,
        "cloud_kernel_executed": False,
        "limitations": "Local Linux CPU notebook run and saved public responses; no live Colab kernel, Windows install, GPU, or global training claim was tested.",
        "result": "All assertions passed; public CPU output, current source, downloadable notebook and pinned Colab notebook agree.",
    }
    saved("audit", "json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
