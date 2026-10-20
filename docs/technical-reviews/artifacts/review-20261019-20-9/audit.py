"""保存課文實跑、練習變體與原始來源的最小查核收據。"""

import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from html import unescape
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = HERE / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def curl(url):
    command = ["curl", "-fsSL", "--max-time", "30", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, command


def plain(raw):
    text = raw.decode("utf-8")
    text = re.sub(r"<script\b[^>]*>.*?</script>", "", text, flags=re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", "", text, flags=re.S)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", " ", text))).strip()


def run_card(code):
    stream = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "answer-cards", "exec"), namespace)
    return stream.getvalue(), namespace


def main():
    environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu"}
    command = [
        ".venv/bin/python", "docs/review-tools/section_facts.py",
        "course/chapters/20.md#20.9", "--execute",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, check=True,
    )
    summary = json.loads(completed.stdout)
    run_path = Path(summary["artifact_directory"]).resolve()
    assert run_path.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    files = {}
    for name in ["section.md", "fence-1.py", "extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
        raw = (run_path / name).read_bytes()
        files[name] = {"sha256": sha(raw), "text": raw.decode("utf-8")}
    save("section-run.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 " + " ".join(command),
        "command_stdout": completed.stdout,
        "command_stderr": completed.stderr,
        "returncode": completed.returncode,
        "environment": environment,
        "copied_from": str(run_path.relative_to(ROOT)),
        "files": files,
    })

    source = ROOT / "docs/course-review-20261019/freeze-01/sources/20.9.md"
    raw = source.read_bytes()
    assert sha(raw) == "4f83878f621e54710280e76a93917302c92949d6d551f9951fae93f300149210"
    assert files["section.md"]["text"] == raw.decode("utf-8")
    code = re.search(r"```python\n(.*?)```", raw.decode("utf-8"), re.S).group(1)
    assert code == files["fence-1.py"]["text"]
    original_stdout, namespace = run_card(code)
    expected = "回答卡 0 物件吻合 True 活動符合本題 True\n回答卡 1 物件吻合 True 活動符合本題 False\n"
    assert original_stdout == expected == files["stdout.txt"]["text"]
    tree = ast.parse(code)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cards" for t in node.targets))
    assignment.value.elts[1].values[0] = ast.Set(elts=[ast.Constant(value="人")])
    ast.fix_missing_locations(tree)
    changed_code = ast.unparse(tree)
    changed_stdout, changed_namespace = run_card(changed_code)
    assert changed_namespace["cards"][1]["objects"] == {"人"}
    assert changed_stdout == "回答卡 0 物件吻合 True 活動符合本題 True\n回答卡 1 物件吻合 False 活動符合本題 False\n"
    called_names = sorted({node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)})
    assert called_names == ["enumerate", "print"]
    assert {"腳踏車", "人"} == {"人", "腳踏車"}
    assert {"人"} != {"人", "腳踏車"}
    assert [i for i, _ in enumerate(namespace["cards"])] == [0, 1]

    image_path = ROOT / "course/figures/natural-v4-training-cat.svg"
    embedded = re.search(r'href="data:image/jpeg;base64,([^"]+)"', image_path.read_text()).group(1)
    image_raw = base64.b64decode(embedded)
    image_url = "https://storage.googleapis.com/docci/thumbnails/train_01827.jpg?generation=1713028329088595"
    upstream_image, image_command = curl(image_url)
    assert image_raw == upstream_image
    metadata_path = ROOT / "docs/natural-assistant/v4/data/vision-sources.json"
    metadata = json.loads(metadata_path.read_text())
    row = next(row for row in metadata["rows"] if row["example_id"] == "train_01827")
    assert row["split"] == "train" and row["sha256"] == sha(image_raw)

    receipt_path = ROOT / "docs/course-review-20261019/figure-renders-04.json"
    receipt = json.loads(receipt_path.read_text())
    matches = []

    def visit(value):
        if isinstance(value, dict):
            if any(isinstance(item, str) and "natural_photo_evidence.svg" in item for item in value.values()):
                matches.append(value)
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(receipt)
    png_path = ROOT / "outputs/course-review-20261019/figure-renders-04/a944f5ace973d74a037caff4ebabe5d9bb15f65e9cad3fad89eefc44c19f528a.png"
    inputs = [
        source, ROOT / "course/chapters/20.md", ROOT / "course/figures/natural_photo_evidence.svg",
        image_path, metadata_path, receipt_path, ROOT / "docs/course-review-20261019/freeze-04.json",
        ROOT / "docs/review-tools/section_facts.py",
    ]
    save("audit-output.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
        "environment": environment,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in inputs},
        "original_stdout": original_stdout,
        "exercise_code": changed_code,
        "exercise_stdout": changed_stdout,
        "called_builtin_names": called_names,
        "set_equality_order_independent": True,
        "missing_bicycle_is_unequal": True,
        "enumerate_indices": [i for i, _ in enumerate(namespace["cards"])],
        "official_photo": {
            "command": image_command, "url": image_url, "bytes": len(image_raw),
            "sha256": sha(image_raw), "embedded_equals_upstream": True,
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "local_split": row["split"], "example_id": row["example_id"],
        },
        "render_receipt_matches": matches,
        "viewed_png_sha256": sha(png_path.read_bytes()),
        "assertions": "全部通過；只驗證這個範例與指定圖片，不是模型能力實測。",
    })

    minor = ".".join(platform.python_version_tuple()[:2])
    specs = [
        {
            "id": "python_sets", "url": f"https://docs.python.org/{minor}/library/stdtypes.html",
            "locator": "Set Types — set, frozenset；比較操作的 == 定義",
            "needles": ["unordered collection", "same elements"],
            "quote": "Test whether the set and other have the same elements.",
        },
        {
            "id": "python_enumerate", "url": f"https://docs.python.org/{minor}/library/functions.html",
            "locator": "enumerate(iterable, start=0)",
            "needles": ["enumerate", "defaults to 0"],
            "quote": "from start which defaults to 0",
        },
        {
            "id": "gqa", "url": "https://cs.stanford.edu/people/dorarad/gqa/about.html",
            "locator": "Question Answering on Image Scene Graphs；場景圖、問題語意與不同題型診斷",
            "needles": ["objects, attributes and relations", "different question types"],
            "quote": "Each image is associated with a scene graph of the image's objects, attributes and relations",
        },
        {
            "id": "validation", "url": "https://scikit-learn.org/1.7/modules/cross_validation.html",
            "locator": "Cross-validation: evaluating estimator performance；Cross-validation iterators for grouped data / Group K-fold",
            "needles": ["evaluation metrics no longer report on generalization performance", "not represented in both testing and training sets"],
            "quote": ["evaluation metrics no longer report on generalization performance.", "the same group is not represented in both testing and training sets."],
        },
        {
            "id": "docci", "url": "https://google.github.io/docci/",
            "locator": "Abstract；Dataset",
            "needles": ["spatial relations, counting", "related or similar"],
            "quote": "crafted to clearly distinguish each image from those that are related or similar.",
        },
    ]
    snapshots = []
    for spec in specs:
        downloaded, curl_command = curl(spec["url"])
        text = plain(downloaded)
        for needle in spec["needles"]:
            assert needle in text, (spec["id"], needle)
            position = text.index(needle)
            print(json.dumps({"source": spec["id"], "read_context": text[max(0, position - 180):position + 850]}, ensure_ascii=False))
        title = re.search(r"<title[^>]*>(.*?)</title>", downloaded.decode("utf-8"), re.S)
        snapshots.append({
            "id": spec["id"], "url": spec["url"], "command": curl_command,
            "download_sha256": sha(downloaded), "download_bytes": len(downloaded),
            "title": unescape(title.group(1)).strip() if title else spec["id"],
            "locator": spec["locator"], "short_excerpt": spec["quote"],
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "full_text_saved": False,
        })
    save("primary-sources.json", {"sources": snapshots})
    print(json.dumps({"result": "全部查核斷言通過", "helper_result": summary, "artifacts": ["section-run.json", "audit-output.json", "primary-sources.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
