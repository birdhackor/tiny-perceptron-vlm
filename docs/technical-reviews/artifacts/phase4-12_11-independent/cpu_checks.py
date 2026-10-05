"""Bounded original-fence and input-information checks; no training/inference."""
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

original = BASE / "original/fence-1.py"
command = [sys.executable, "-I", str(original)]
result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False, cwd=BASE)
(BASE / "original-fence.stdout.txt").write_text(result.stdout)
(BASE / "original-fence.stderr.txt").write_text(result.stderr)
expected = "轉寫相同 True\n音高相同 False\n只看轉寫 你好 實際標籤 low\n只看轉寫 你好 實際標籤 high\n"
assert result.returncode == 0 and result.stdout == expected
print("original fence exact stdout verified")

rows = [{"transcript": "你好", "pitch": "low"}, {"transcript": "你好", "pitch": "high"}]
truth = [r["pitch"] for r in rows]
constant_checks = []
for response in ("low", "high", "unknown"):
    predictions = [response for _ in rows]
    correct = sum(a == b for a, b in zip(predictions, truth))
    assert correct <= 1
    constant_checks.append({"response": response, "predictions": predictions, "correct": correct, "denominator": 2})
copy_answers = [r["transcript"] for r in rows]
assert copy_answers == ["你好", "你好"]
augmented_predictions = [r["pitch"] for r in rows]
assert augmented_predictions == truth
# A transcript may retain spoken-word order, but it does not encode unspoken pitch order.
ordered = [{"transcript": "你好", "pitch_sequence": ["low", "high"]},
           {"transcript": "你好", "pitch_sequence": ["high", "low"]}]
assert ordered[0]["transcript"] == ordered[1]["transcript"]
assert [r["pitch_sequence"][0] == "high" for r in ordered] == [False, True]
variation = {"constant_text_only_rules": constant_checks, "copy_task": copy_answers,
             "additional_pitch_metadata": augmented_predictions, "unspoken_pitch_order_rows": ordered,
             "scope": "Handwritten data consistency and deterministic input collision only; no ASR or trained-model result."}
(BASE / "variation-results.json").write_text(json.dumps(variation, ensure_ascii=False, indent=2) + "\n")
print("bounded variants: deterministic identical input, copying, extra metadata, and pitch order verified")

source = ROOT / "docs/course-experiments/results/audio.json"
raw = json.loads(source.read_bytes())
# These are the only result values inspected. Do not print notes, scope, or corrections.
pointers = ["/schema_version", "/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version",
            "/results/training/steps", "/results/test/examples", "/results/test/correct", "/results/test/exact_match",
            "/results/data/split_policy", "/results/test/samples"]
for split in ("train", "validation", "test"):
    pointers.extend([f"/results/data/splits/{split}/count", f"/results/data/splits/{split}/records",
                     f"/results/data/splits/{split}/sha256"])
selected = {}
for pointer in pointers:
    value = raw
    for part in pointer.strip("/").split("/"):
        value = value[part]
    selected[pointer] = value
samples = selected["/results/test/samples"]
records = selected["/results/data/splits/test/records"]
assert len(samples) == len(records) == selected["/results/test/examples"] == 14
assert sum(s["exact_match"] for s in samples) == selected["/results/test/correct"] == 11
assert abs(11 / 14 - selected["/results/test/exact_match"]) <= 1e-15
assert all(r["modality"] == "audio" and r["question"] == "pitch?" for r in records)
assert all(r["answer"] == ("high" if r["frequency"] > 300 else "low") for r in records)
assert [s["target"] for s in samples] == [r["answer"] for r in records]
frequencies = {}
for split in ("train", "validation", "test"):
    items = selected[f"/results/data/splits/{split}/records"]
    assert len(items) == selected[f"/results/data/splits/{split}/count"]
    frequencies[split] = sorted({r["frequency"] for r in items})
assert not (set(frequencies["train"]) & set(frequencies["test"]))
selection = {"source_sha256": digest(source), "pointers": pointers, "values": selected,
             "independent_checks": {"raw_test_samples": len(samples), "exact_matches": 11,
                                    "frequency_families": frequencies,
                                    "speech_transcription_examples": 0},
             "support_limit": "Existing synthetic tone-classification data only; no evidence of ASR, prosody recognition, or trained natural-speech capability."}
(BASE / "audio-raw-inspection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
print("existing raw JSON: 14 synthetic-tone samples, 11 exact matches; no ASR samples; no retraining")

class Visible(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}: self.skip += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style"}: self.skip -= 1
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)
page = BASE / "rendered-12.11.html"
parser = Visible(); parser.feed(page.read_text())
visible = "".join(parser.parts)
normalized = "".join(visible.split())
body = (BASE / "original/section.md").read_text()
paragraphs = [p for p in body.split("\n\n") if p.startswith(("兩段聲音", "這是手寫", "自動語音", "直接聲音", "練習將問題"))]
for paragraph in paragraphs:
    assert "".join(paragraph.split()) in normalized
assert "".join(original.read_text().split()) in normalized
page_check = {"page_sha256": digest(page), "section_sha256": digest(BASE / "original/section.md"),
              "paragraphs_verified": paragraphs, "fence_matched_current_source": True}
(BASE / "render-source-match.json").write_text(json.dumps(page_check, ensure_ascii=False, indent=2) + "\n")
print("rendered page: all five substantive paragraphs and original fence match frozen current source")

execution = {"command": " ".join(command), "exit_code": result.returncode,
             "stdout_sha256": digest(BASE / "original-fence.stdout.txt"), "original_fence_sha256": digest(original),
             "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU",
                             "training": "none", "model_inference": "none"}}
(BASE / "cpu-environment.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n")
