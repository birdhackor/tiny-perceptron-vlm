"""Current-source recheck after the completed corrective reader round."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]


def section(path, identifier):
    raw = path.read_bytes().decode("utf-8")
    start = re.search(r"^## " + re.escape(identifier) + r"(?:\s|$)", raw, re.M)
    end = re.search(r"^## ", raw[start.end():], re.M)
    return raw[start.start():start.end() + end.start() if end else len(raw)].encode()


current = section(ROOT / "course/chapters/13.md", "13.16")
assert current == (OUT / "source-current-reread.md").read_bytes()
assert hashlib.sha256(current).hexdigest() == "99b78665cf25940b79388a79c9881fd1f7455a65c75d90079aa06fba4821a1ae"
manifest = {"primary": {"path": "course/chapters/13.md#13.16", "sha256": hashlib.sha256(current).hexdigest(), "bytes": len(current)}, "prerequisites": {}, "figures": {}}
for lesson in ["13.10", "13.15", "13.5", "7.13", "13.13", "13.14", "13.11", "13.12"]:
    path = ROOT / ("course/chapters/07.md" if lesson == "7.13" else "course/chapters/13.md")
    body = section(path, lesson)
    assert body == (OUT / f"prerequisite-{lesson}-original.md").read_bytes()
    manifest["prerequisites"][str(path.relative_to(ROOT)) + "#" + lesson] = hashlib.sha256(body).hexdigest()
for name in ["multimodal_dpo_direction", "ppo_clip", "ppo_roles"]:
    path = ROOT / "course/figures" / (name + ".svg")
    manifest["figures"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
assert manifest["figures"] == json.loads((OUT / "source-manifest.json").read_text())["figures"]

code = re.findall(r"```python\n(.*?)```", current.decode(), re.S)[0]
assert hashlib.sha256(code.encode()).hexdigest() == json.loads((OUT / "cpu-results.json").read_text())["primary_snippet"]["sha256"]
ns, buf = {}, io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, "13.16-current-snippet", "exec"), ns)
indices = list(range(len(ns["answers"])))
key = lambda index: ns["scores"][index]
key_scores = [key(i) for i in indices]
assert indices == [0,1] and key_scores == [1,14] and max(indices,key=key) == 1
exercise = [int(answer == str(1+2)) for answer in ns["answers"]]
assert exercise == [1,0]
assert ns["answers"][max(indices,key=lambda i: exercise[i])] == "3"
other = {}
for lesson in ["13.11", "13.12"]:
    code = re.findall(r"```python\n(.*?)```", (OUT / f"prerequisite-{lesson}-original.md").read_text(), re.S)[0]
    sbuf, sn = io.StringIO(), {}
    with contextlib.redirect_stdout(sbuf):
        exec(compile(code, f"{lesson}-prerequisite-example", "exec"), sn)
    other[lesson] = sbuf.getvalue()
result = {"environment": {"python": platform.python_version(), "device": "cpu"},
          "source": manifest, "current_primary_stdout": buf.getvalue(),
          "range_indices": indices, "lambda_key_scores": key_scores, "max_chosen_index": ns["chosen"],
          "exercise_scores": exercise, "additional_prerequisite_stdout": other,
          "read_scope": "Personally reread all 4107 bytes of current 13.16 after coordinator confirmed corrective reader closure; reread covers all prose, blank lines, code, table and caveats. Eight explicit/necessary prerequisite sections were read in full. Only syntax prose changed from the preserved first read; identical primary code was executed again. Current SVG bytes equal the three rendered and personally viewed originals. Reader judgment is not used as evidence of factual truth."}
(OUT / "reread-current-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
