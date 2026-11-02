"""Bind the full v6 lesson to unchanged evidence; probe corrected decoder scopes.

Only Python doubles and the small preserved-row projection are executed.
"""

import difflib
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.omni_revision.core_evaluation import generated_sound
from tiny_perceptron.omni_products.data import Event
from tiny_perceptron.omni_products.inference import OmniAssistant

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.22"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v6.json"
assert sha(freeze_path) == "57f17dde51dac7f9b190e0b02dac68ad8f4f8214f368af1c8c8f8f2aa6ba18df"
page = next(p for p in read(freeze_path)["pages"] if p["page_id"] == "19.22")
body = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(body).hexdigest() == page["source_sha256"] == "60d4c1315f1db96e773999ed22cd1d76e83c8d51638f9abd1b113d174df8ea07"
actual = re.search(r"(?ms)^## 19\.22 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text()).group()
assert actual.encode() == body
original = read(REVIEW / "technical-original-v1.json")
old_body = (ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.22.md").read_bytes()
assert hashlib.sha256(old_body).hexdigest() == original["source_sha256"]
codes = re.findall(r"```python\n(.*?)\n```", body.decode(), re.S)
old_codes = re.findall(r"```python\n(.*?)\n```", old_body.decode(), re.S)
assert len(codes) == 2 and codes == old_codes
source_matches = {s["path"]: sha(ROOT / s["path"]) == s["sha256"] for s in original["sources"] if s["kind"] == "repository_code"}
artifact_matches = {a["path"]: sha(ROOT / a["path"]) == a["sha256"] for a in original["artifacts"]}
assert all(source_matches.values()) and all(artifact_matches.values())
assert page["figures_sha256"] == original["figure_sha256"]
for name, digest in page["figures_sha256"].items():
    assert sha(ROOT / name) == digest
figure_receipt = read(HERE / "figure-inspection.json")
assert sha(Path(figure_receipt["render"]["outside_git_path"])) == figure_receipt["render"]["sha256"]
assert sha(ROOT / page["notebook"]) == page["notebook_sha256"]
notebook = read(ROOT / page["notebook"])
notebook_codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert all(any(code.strip() == cell.strip() for cell in notebook_codes) for code in codes)
v7_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
v7_page = next(p for p in read(v7_path)["pages"] if p["page_id"] == "19.22")
assert v7_page == page
bundle_path = ROOT / "outputs/omni-P5-current-kernel-bundle-v7-receipt.json"
assert sha(bundle_path) == "9133865489a89010e3786d3840dc0d9348a9a165010ce9d08a247bcd10d5141b"
bundle = read(bundle_path)
bundle_row = next(r for r in bundle["rows"] if r["id"] == "19.22")
assert bundle_row["source"] == page["notebook"]
assert bundle_row["source_sha256"] == page["notebook_sha256"]


class FailedTalker:
    def generate(self, program):
        return {"codes": torch.tensor([0]), "eos": False, "capped": True}


class RecordingCodec:
    def __init__(self):
        self.decode_calls = 0
        self.chunk_calls = 0

    def decode_codes(self, codes):
        self.decode_calls += 1
        return torch.zeros(80)

    def chunks(self, codes):
        self.chunk_calls += 1
        yield torch.zeros(80)


class FakeAV:
    config = None

    def parameters(self):
        return iter([torch.zeros(())])

    def __call__(self, ignored_batch):
        return torch.tensor([[1.0, 0.0, 0.0]])


class FakeItem:
    def validate(self):
        return self


program = (Event(220, 100),)
codec = RecordingCodec()
evaluation = generated_sound(codec, FailedTalker(), program)
assert codec.decode_calls == 1
assert not evaluation["legal"] and not evaluation["program_success"]
video = OmniAssistant.__new__(OmniAssistant)
video.codec, video.talker, video.av = RecordingCodec(), FailedTalker(), FakeAV()
with patch("tiny_perceptron.omni_products.inference.collate_av", return_value=None):
    video_result = video.video(FakeItem(), with_sound=True)
assert video.codec.decode_calls == 1
assert not video_result["sound"]["eos"] and video_result["sound"]["capped"]
stream = OmniAssistant.__new__(OmniAssistant)
stream.codec, stream.talker = RecordingCodec(), FailedTalker()
try:
    list(stream.sound_chunks(program))
except ValueError as exc:
    stream_error = str(exc)
else:
    raise AssertionError("Expected failed output rejected by streaming entry")
assert stream.codec.decode_calls == stream.codec.chunk_calls == 0

replay_process = subprocess.run([sys.executable, str(HERE / "recompute_measurements.py")], cwd=ROOT, capture_output=True, text=True, check=True)
replay = json.loads(replay_process.stdout)
assert replay["passed"]
corrected_line = next(line for line in body.decode().splitlines() if "可播放分塊入口" in line)
result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_19_22",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU; no neural model", "torch_threads_for_projection": "1"},
    "source_freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_path)},
    "page": page,
    "complete_body_reread": "Reviewer read the entire v6 body including inline code, optional full calculation, results table, limitations, exercise and final timing transition; no other lesson reviewed.",
    "diff": "".join(difflib.unified_diff(old_body.decode().splitlines(True), body.decode().splitlines(True), fromfile="original19.22", tofile="v6 19.22")),
    "corrected_line": corrected_line,
    "current_chapter_slice_matches_freeze": True,
    "both_python_blocks_exact_unchanged": True,
    "code_block_sha256": [hashlib.sha256(code.encode()).hexdigest() for code in codes],
    "all_registered_current_implementation_sources_unchanged": source_matches,
    "all_original_portable_evidence_artifacts_unchanged": artifact_matches,
    "notebook_current_sha256": sha(ROOT / page["notebook"]),
    "both_python_blocks_match_current_notebook": True,
    "v7_19_22_page_entry_identical_to_v6": True,
    "figure_reread": {"same_source_sha256": page["figures_sha256"], "reused_render_receipt": {"path": (HERE / "figure-inspection.json").relative_to(ROOT).as_posix(), "sha256": sha(HERE / "figure-inspection.json")}, "actual_render_viewed_again": True, "scope": "Existing exact-source 400x780 rendering inspected again. Gold arrow enters analyzer only; 220/rest/440 formal program and 8kHz decoder match current text. No new screenshot duplication or full-page/mobile acceptance."},
    "corrected_three_entry_behavior_probe": {
        "fixture": {"codes": [0], "eos": False, "capped": True},
        "generated_sound": {"decoder_calls": codec.decode_calls, "legal": evaluation["legal"], "program_success": evaluation["program_success"]},
        "video": {"decoder_calls": video.codec.decode_calls, "eos": video_result["sound"]["eos"], "capped": video_result["sound"]["capped"], "samples": len(video_result["sound"]["waveform"])},
        "sound_chunks": {"decoder_calls": stream.codec.decode_calls, "chunk_calls": stream.codec.chunk_calls, "error": stream_error},
        "scope": "All doubles; no actual AV, codec or Talker neural model, saved weights or final families run.",
    },
    "small_projection_replay": {"command": "python docs/technical-reviews/artifacts/omni-P5-19.22/recompute_measurements.py", "mode": replay["mode"], "projection": replay["projection"], "passed": replay["passed"], "mean_snr_db": replay["mean_snr_db_from_200_preserved_family_values"], "gold_correct_legal": replay["gold_program_correct_and_legal"], "chain_counts": replay["chain_all_six_counts"], "timing_original_rows_only": replay["timing_10_programs"]},
    "author_execution_receipt_provenance_only": {"path": bundle_path.relative_to(ROOT).as_posix(), "sha256": sha(bundle_path), "relevant_row": bundle_row, "scope": "Inspected receipt metadata only; not an independent current kernel execution claim and not a required future-clone artifact."},
    "passed": True,
    "limits": "No formal NN, original float32 reconstruction, training, model selection, latency benchmark, Modal, workflow or publication executed. Prior primary-package inspection retained unchanged by SHA; current independent work is complete text/code reread, byte binding, artificial three-entry probe and small projection recomputation.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
