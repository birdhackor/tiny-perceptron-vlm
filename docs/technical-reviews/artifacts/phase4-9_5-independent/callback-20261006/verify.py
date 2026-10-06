"""Same factual owner's narrow callback; no prior report parsed, no model run."""
import ast
import difflib
import hashlib
import json
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
old = (BASE / "execution/section.md").read_bytes()
new = (OUT / "section.md").read_bytes()
raw = (ROOT / "course/chapters/09.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
i = next(i for i, h in enumerate(headings) if h[0].startswith(b"## 9.5 "))
current = raw[headings[i].start():headings[i + 1].start()]
assert current == new
changes = [
    ("里", "裡"), ("我们", "我們"), ("帮", "幫"), ("询", "詢"),
    ("这", "這"), ("比较", "比較"), ("天气", "天氣"), ("并", "並"),
    ("標注", "標註"), ("刚", "剛"), ("于", "於"),
]
normalized = old.decode("utf-8")
for before, after in changes:
    normalized = normalized.replace(before, after)
assert normalized.encode("utf-8") == new, "Unexpected non-glyph delta"
fences = lambda x: re.findall(rb"(?ms)^```python\n(.*?)^```\s*$", x)
old_fences, new_fences = fences(old), fences(new)
assert len(old_fences) == len(new_fences) == 1 and old_fences == new_fences
assert new_fences[0] == (BASE / "execution/fence-1.py").read_bytes()
(OUT / "current-fence-1.py").write_bytes(new_fences[0])
(OUT / "section.diff").write_text("".join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True), fromfile="own-frozen-20261005-9.5", tofile="current-20261006-9.5")))
receipt = json.loads((BASE / "execution/execution.json").read_bytes())
proofs = []
for item in receipt["artifacts"]:
    path = BASE / "execution" / item["path"]
    assert sha(path.read_bytes()) == item["sha256"]
    proofs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": item["sha256"], "matched_original_receipt": True})
originals = []
for item in json.loads((BASE / "sources/fetch-receipts.json").read_bytes()):
    assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"]
    originals.append({"path": item["path"], "sha256": item["sha256"], "url": item["url"], "accessed_on": item["accessed_on"], "matched_original_fetch_receipt": True})
tree = ast.parse((ROOT / "scripts/build_course.py").read_bytes())
node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BOOTSTRAP" for t in n.targets))
bootstrap = ast.literal_eval(node.value).encode()
assert bootstrap == (BASE / "execution/bootstrap.py").read_bytes()
variant_receipt = json.loads((BASE / "execution/variants-execution.json").read_bytes())
assert sha((BASE / "variants.py").read_bytes()) == variant_receipt["program_sha256"]
variants = json.loads((BASE / "execution/variants-result.json").read_bytes())
assert variants["original_fence_sha256"] == sha(new_fences[0])
assert variants["original_stdout"] == (BASE / "execution/stdout.txt").read_text()
assert variants["original_labels"] == [[True, False], [True, True], [True, False]]
assert variants["row_count"] == 3
assert variants["text_only_change_stdout"].splitlines()[1].endswith("邊界 True 替代 True")
assert variants["exercise_change_stdout"].splitlines()[1].endswith("邊界 True 替代 False")
assert (BASE / "execution/variants-stdout.txt").read_bytes() == (BASE / "execution/variants-result.json").read_bytes()
assert not re.findall(rb"!\[[^\]]*\]\([^)]+\)", new)
assert not re.search(rb"<(?:img|svg)\b", new)
facts = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_9_5",
    "current_source_sha256": sha(new), "frozen_source_sha256": sha(old),
    "current_first_line": raw[:headings[i].start()].count(b"\n") + 1,
    "source_scope": "current raw UTF-8 9.5; full chapter hash is not claimed",
    "delta": "Only the explicitly listed traditional-character replacements; no added, deleted or changed substantive claim.",
    "actual_glyph_replacements": changes,
    "fence_bytes_unchanged": True, "current_fence_sha256": sha(new_fences[0]),
    "bootstrap_bytes_unchanged": True, "bootstrap_sha256": sha(bootstrap),
    "figure_sha256": {}, "original_proof_checks": proofs,
    "original_source_checks": originals,
    "original_execution_receipt_sha256": sha((BASE / "execution/execution.json").read_bytes()),
    "variant_program_sha256": variant_receipt["program_sha256"],
    "variant_result_sha256": sha((BASE / "execution/variants-result.json").read_bytes()),
    "variant_output_checked_without_rerun": True,
    "python": platform.python_version(),
    "reuse_limits": "2026-10-05 original CPU executions and personally inspected official sources are reused after exact byte/hash checks. Callback does not rerun the fence or semantic judgments, refetch originals, run a model, or claim current policy/version beyond the dated original normative document.",
    "prior_report": "Preserved opaque; not parsed or read for callback answers.",
}
(OUT / "comparison.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(facts, ensure_ascii=False, indent=2))
