"""5.11 bounded CPU verification: execute frozen lesson, inspect fixed data; no training."""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import _deduplicate_text
from tiny_perceptron.data import ByteTokenizer, fingerprint as library_fingerprint

OUT = Path(__file__).with_name("execution.json")
if OUT.exists():
    raise FileExistsError(OUT)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def captured(blocks):
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, block in enumerate(blocks):
            exec(compile(block, f"frozen-5.11-block-{index + 1}", "exec"), namespace)
    return namespace, output.getvalue()


freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "5.11")
raw = (ROOT / page["snapshot"]).read_bytes()
assert digest(raw) == page["source_sha256"]
chapter = (ROOT / page["source"]).read_bytes()
start = chapter.index(b"## 5.11 ")
end = chapter.index(b"\n## ", start + 1) + 1
assert chapter[start:end] == raw
assert digest((ROOT / page["notebook"]).read_bytes()) == page["notebook_sha256"]
assert page["figures_sha256"] == {} and ".svg" not in raw.decode()
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(blocks) == 3
ns, stdout = captured(blocks)
assert ns["normalized"] == ["貓 看狗", "貓 看狗", "狗 看貓"]
assert len(ns["hashes"]) == 3 and len(set(ns["hashes"])) == 2
assert ns["hashes"][0] == ns["hashes"][1]
assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in ns["hashes"])
inline_def = ast.parse(blocks[1]).body[0]
library_def = ast.parse(inspect.getsource(library_fingerprint)).body[0]
assert ast.dump(inline_def) == ast.dump(library_def)
assert [library_fingerprint(t) for t in ns["docs"]] == ns["hashes"]
whitespace_probe = " \t貓\n\n看狗 \t"
assert whitespace_probe.split() == ["貓", "看狗"]
assert " ".join(whitespace_probe.split()) == "貓 看狗"
assert ns["docs"] == ["貓  看狗", "貓 看狗", "狗 看貓"]
try:
    hashlib.sha256("貓 看狗")
except TypeError as error:
    string_hash_error = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("sha256 unexpectedly accepted str")
assert "貓 看狗".encode() == "貓 看狗".encode("utf-8")

practice = [blocks[0].replace('"貓 看狗"', '"鳥 看狗"'), *blocks[1:]]
practice_ns, practice_stdout = {}, io.StringIO()
with contextlib.redirect_stdout(practice_stdout):
    exec(compile(practice[0], "practice-1", "exec"), practice_ns)
    exec(compile(practice[1], "practice-2", "exec"), practice_ns)
    try:
        exec(compile(practice[2], "practice-3", "exec"), practice_ns)
    except AssertionError:
        stale_assert_failed = True
    else:
        stale_assert_failed = False
assert stale_assert_failed and len(set(practice_ns["hashes"])) == 3
assert practice_ns["hashes"][0] != practice_ns["hashes"][1]
exec('assert hashes[0] != hashes[1]', practice_ns)

prefixes, collision = {}, None
for i in range(17):
    text = f"distinct-document-{i}"
    full = library_fingerprint(text)
    if full[0] in prefixes and collision is None:
        prior_text, prior_full = prefixes[full[0]]
        assert prior_text != text and prior_full != full
        collision = {"first_text": prior_text, "second_text": text,
                     "first_full_sha256": prior_full, "second_full_sha256": full,
                     "shared_one_hex_prefix": full[0]}
    prefixes[full[0]] = (text, full)
assert collision is not None

toy_rows = [{"id": "a", "text": "貓  看狗\n"},
            {"id": "b", "text": "貓 看狗"},
            {"id": "c", "text": "狗 看貓"}]
toy_dedup = _deduplicate_text(toy_rows)
assert len(toy_dedup) == 2 and toy_dedup[0]["text"] == toy_rows[0]["text"]
tok = ByteTokenizer()
x, y = text_examples([toy_dedup[0]], mode="text", max_length=128)[0]
assert x.tolist() == [tok.bos_id] + tok.encode(toy_rows[0]["text"])
assert y.tolist() == tok.encode(toy_rows[0]["text"]) + [tok.eos_id]

result_path = ROOT / "docs/course-experiments/results/real_text.json"
record = json.loads(result_path.read_text())
common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip())
data_checks = {}
for asset in record["assets"]:
    identifier = asset["id"]
    basename = {"tinystories": "tinystories-train-512.jsonl",
                "chinese-poetry": "chinese-classical-train-365.jsonl"}[identifier]
    archive = ROOT / asset["archive"]
    if archive.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        sha = asset["archive_sha256"]
        archive = common / "lfs" / "objects" / sha[:2] / sha[2:4] / sha
    assert digest(archive.read_bytes()) == asset["archive_sha256"]
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("/" + basename)]
        assert len(members) == 1
        payload = tf.extractfile(members[0]).read()
    file_metadata = next(f for f in asset["files"] if f["path"].endswith("/" + basename))
    assert digest(payload) == file_metadata["sha256"]
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    deduped = _deduplicate_text(rows)
    reported = record["results"]["runs"][identifier]
    assert len(rows) == reported["source_records"]
    assert len(deduped) == reported["deduplicated_records"]
    assert len(rows) == len(deduped)
    assert [r["text"] for r in rows] == [r["text"] for r in deduped]
    assert all(r["family"] == library_fingerprint(r["text"]) for r in deduped)
    parts = split_records(deduped, seed=record["seed"])
    splits = {}
    all_families = []
    for split, records in parts.items():
        encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
        metadata = reported["data"][split]
        assert digest(encoded) == metadata["sha256"]
        assert len(records) == metadata["records"]
        families = {r["family"] for r in records}
        all_families.append(families)
        splits[split] = {"records": len(records), "sha256": digest(encoded)}
    assert all(not (a & b) for i, a in enumerate(all_families) for b in all_families[i + 1:])
    data_checks[identifier] = {"archive_path_used": str(archive),
        "archive_sha256": asset["archive_sha256"], "input_jsonl_sha256": digest(payload),
        "source_records": len(rows), "deduplicated_records": len(deduped),
        "original_texts_preserved": True, "all_family_labels_are_full_normalized_sha256": True,
        "seed": record["seed"], "split_digests_match_record": True, "splits": splits,
        "recorded_split_warning": reported["split_warning"]}

code_checks = {}
for path in ["tiny_perceptron/data.py", "scripts/course_experiments/text.py",
             "scripts/course_experiments/common.py"]:
    observed = digest((ROOT / path).read_bytes())
    assert observed == record["code_sha256"][path]
    code_checks[path] = observed

output = {"created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_5_11",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_5_11-v1/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "platform": platform.platform()},
    "source_snapshot": {"path": page["snapshot"], "sha256": digest(raw),
                        "current_section_equal": True, "notebook_digest_matches_freeze": True,
                        "figure_references": []},
    "inline_program": {"blocks_executed": len(blocks), "stdout": stdout,
        "hashes": ns["hashes"], "inline_definition_ast_equals_library": True,
        "whitespace_probe_tokens": whitespace_probe.split(), "str_sha256_error": string_hash_error},
    "exercise": {"stdout": practice_stdout.getvalue(), "unique_contents": 3,
                 "old_assertion_failed": stale_assert_failed, "new_assertion_passed": True},
    "short_prefix_collision": collision,
    "sampling_derivation": {"assumption": "uniform draw over stored records",
         "with_two_copies_among_three": str(Fraction(2, 3)),
         "after_dedup_one_among_two": str(Fraction(1, 2))},
    "toy_dedup_preserves_first_original_text": True,
    "tokenization_receives_original_whitespace": True,
    "existing_experiment_result": {"path": str(result_path.relative_to(ROOT)),
       "sha256": digest(result_path.read_bytes()), "recorded_revision": record["revision"],
       "replayed_training": False, "checked_code_sha256": code_checks},
    "data": data_checks,
    "result": "PASS: frozen inline code/exercise and fixed-data preprocessing verified; no model training performed."}
OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
