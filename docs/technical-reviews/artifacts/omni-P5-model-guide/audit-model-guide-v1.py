"""Independent bounded audit: generated inputs, saved rows/receipts and file-only fixtures.

Never loads weights, calls trained neural forward, retrains, changes T/tau, or publishes.
"""
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import wave
import zipfile

import torch

from scripts.omni_revision import public_demo as demo, public_download as download
from tiny_perceptron.omni_products.data import Event, av_pair, split_a, validate_program
from tiny_perceptron.omni_products.inference import CJudge, SOUND_MAP, verify_information_binding
from tiny_perceptron.omni_products.reliability import calibrated

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("audit-model-guide-v1.json")
CORE = Path("/workspace/work/omni-core-completed-40988311100/verified")
FINAL = Path("/workspace/work/omni-final-completed-41004229262/verified")
RAW_ARCHIVE = FINAL.parent / "final-results.zip"
PORTABLE = ROOT / "docs/course-revision-20261029-omni/infrastructure-review/public-live-author-portable-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": path.stat().st_size, "sha256": sha(path)}

def read(path):
    return json.loads(Path(path).read_bytes())

started = datetime.now(UTC).isoformat()
body_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/auxiliary-sources-v2/model-guide.md"
body = body_path.read_text()
assert body_path.read_bytes() == (ROOT / "docs/omni-products/README.md").read_bytes()
assert sha(body_path) == "8167259a711d506ee3d39daa1c7fc6c6c70bdbfb047e7152dd616230a4b52662"
blocks = re.findall(r"```json\n(.*?)\n```", body, re.S)
program_json, choice_json = map(json.loads, blocks)
program = tuple(Event(**row) for row in program_json)
assert validate_program(program) == SOUND_MAP["square"]
assert choice_json == demo.choice_example()
choice = demo.public_question(choice_json)
information = demo.public_question(choice_json["information_question"])
assert verify_information_binding(choice, information)

# Source/fragment targets are checked, without pretending to review their full lessons.
links = []
for target in re.findall(r"\]\(([^)]+)\)", body):
    if target.startswith("https://"):
        links.append({"target": target, "scope": "public provider metadata acquired separately"})
        continue
    name, _, anchor = target.partition("#")
    path = (ROOT / "docs/omni-products" / name).resolve()
    assert path.is_file(), target
    anchor_found = not anchor or bool(re.search(r"^## " + re.escape(anchor) + r"\s", path.read_text(), re.M))
    assert anchor_found, target
    links.append({"target": target, "resolved": str(path.relative_to(ROOT)), "anchor_found": anchor_found})

# Pure finite-data generator: no model object or neural forward.
family = split_a()["train"][0]
pair = av_pair(family)
first, second = [item for item, _ in pair]
assert family == 30 and first.duration_ms == 3750 and len(first.frames) == 15
assert torch.equal(first.frames, second.frames) and torch.equal(first.waveform, second.waveform)
assert torch.equal(first.frame_ms, second.frame_ms)
active = (first.waveform.abs() > 1e-4).nonzero().flatten()
pulse_start, pulse_end = active[0].item() / 16, (active[-1].item() + 1) / 16
assert (pulse_start, pulse_end) == (45, 125)
bits = [(family >> (7 - min(7, i * 8 // 15))) & 1 for i in range(15)]
runs = []
for i, bit in enumerate(bits):
    label = "square" if bit else "circle"
    if runs and runs[-1]["shape"] == label:
        runs[-1]["end_ms"] = (i + 1) * 250
    else:
        runs.append({"start_ms": i * 250, "end_ms": (i + 1) * 250, "shape": label})
assert runs == [{"start_ms": 0, "end_ms": 1500, "shape": "circle"},
                {"start_ms": 1500, "end_ms": 3500, "shape": "square"},
                {"start_ms": 3500, "end_ms": 3750, "shape": "circle"}]
av_inputs = []
for index, (item, gold) in enumerate(pair):
    interval = [item.audio_start_ms + pulse_start, item.audio_start_ms + pulse_end]
    assert (item.audio_start_ms, interval) == [(1290, [1335, 1415]), (2790, [2835, 2915])][index]
    av_inputs.append({"which": index, "recording_start_ms": item.audio_start_ms,
                      "pulse_interval_ms": interval, "gold_from_generator": ["circle", "square"][gold]})

# Validate each finite contract dimension and a synthetic policy; no trained model.
invalid_programs = [(Event(123, 200),), (Event(220, 50),), (Event(220, 200, 100),),
                    (Event(220, 200), Event(220, 200)), tuple(Event(220, 100, 100) for _ in range(5)),
                    tuple(Event(f, 300, 200 if i < 3 else 0) for i, f in enumerate((220, 440, 220, 440)))]
invalid_programs[-1] = (Event(220, 300, 200), Event(440, 300, 200), Event(220, 300, 200), Event(440, 300))
# Last is a legal 1800ms input near the finite contract's duration bound.
near_limit = invalid_programs.pop()
assert validate_program(near_limit)
rejected = 0
for bad in invalid_programs:
    try:
        validate_program(bad)
    except ValueError:
        rejected += 1
assert rejected == len(invalid_programs)
for value in [dict(choice_json, gold="record"), dict(choice_json, candidates=[{"id":"none","description":"none"}])]:
    try:
        demo.public_question(value)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden public input accepted")

class SyntheticPredictions:
    training = False
    answer = "record"
    logits = [2.0, 0.0]
    def parameters(self):
        yield torch.zeros(())
    def predict(self, question, temperature):
        ids = ["yes", "no"] if question.kind == "info" else [self.answer, "other"]
        answer = "yes" if question.kind == "info" else self.answer
        return {"answer": answer, "ids": ids, "logits": self.logits}

synthetic = SyntheticPredictions()
judge = CJudge(synthetic, temperature=1.0, threshold=0.6)
assert judge.judge(choice)["decision"] == "handoff"
assert judge.judge(choice, information)["decision"] == "accept"
synthetic.answer = "none"
assert judge.judge(choice, information)["decision"] == "handoff"
synthetic.answer, synthetic.logits = "record", [0.0, 0.0]
assert judge.judge(choice, information)["decision"] == "handoff"
try:
    verify_information_binding(choice, replace(information, state="different"))
except ValueError:
    binding_rejected = True
else:
    raise AssertionError("different information state accepted")

# File-only downloader fixture: .pt is opaque data; no deserialization occurs.
with tempfile.TemporaryDirectory(prefix="model-guide-audit-") as temporary:
    tmp = Path(temporary)
    prefix = "core/toy/"
    payload = {prefix+"training/weights.safetensors": b"toy-opaque-weights",
               prefix+"training/training-result.json": b'{"config": {}}',
               prefix+"job-result.json": b'{"toy": true}',
               prefix+"training/latest-continuation.pt": b"not-pickle-and-never-deserialized"}
    manifest = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in payload.items()}
    archive = tmp / "toy.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, raw in payload.items(): z.writestr(name, raw)
        z.writestr("package-file-manifest.json", json.dumps(manifest))
    asset = {"job": "toy", "sha256": sha(archive), "bytes": archive.stat().st_size}
    expected = {name: entry for name, entry in manifest.items() if not name.endswith(".pt")}
    ordinary = download.install(archive, asset, tmp / "ordinary", expected_files=expected)
    full = download.install(archive, asset, tmp / "full", True, expected_files=expected)
    assert len(ordinary) == 3 and len(full) == 4
    assert (tmp / "full" / (prefix+"training/latest-continuation.pt")).read_bytes() == payload[prefix+"training/latest-continuation.pt"]
    conflict = tmp / "ordinary" / (prefix+"job-result.json")
    conflict.write_bytes(b"preserved-different-user-content")
    try:
        download.install(archive, asset, tmp / "ordinary", expected_files=expected)
    except ValueError:
        assert conflict.read_bytes() == b"preserved-different-user-content"
    else:
        raise AssertionError("conflicting existing file overwritten")
    wav = tmp / "fixture.wav"
    demo.save_wave(wav, torch.zeros(4000))
    with wave.open(str(wav)) as stream:
        assert (stream.getframerate(), stream.getnframes()) == (8000, 4000)
    try: demo.save_wave(wav, torch.zeros(4000))
    except FileExistsError: pass
    else: raise AssertionError("existing WAV overwritten")

# Original public CPU receipt: replay hashes and inspect the original stdout, never rerun it.
index = read(ROOT / "docs/course-revision-20261029-omni/evidence/P5-public-assets-live-author-v1.json")
locations = {row["original_path"]: ROOT / row["path"] for row in index["script_and_receipt_locations"]}
producer_receipt = PORTABLE / "a3facfa119e1f40f82ad9f784aeffe320427368fa47780b3c79277e0b5a424ae.json"
receipt = read(producer_receipt)
original_download_receipt = PORTABLE / "141176200219ef14a91e2840d6b2932387613c0e3a4380d2df30bb90dbfb33c6.json"
download_record = read(original_download_receipt)
published = download.descriptor(ROOT / "docs/omni-products/v1/public-release.json")
assert download_record["descriptor_sha256"] == sha(ROOT / "docs/omni-products/v1/public-release.json")
assert set(download_record["assets"]) == {"av", "choice", "codec", "talker"}
assert download_record["anonymous"] and not download_record["full_continuation_requested"]
for kind, asset in download_record["assets"].items():
    assert asset["archive_sha256"] == published["components"][kind]["sha256"]
    for name, expected in download.selected_files(published["_model_selection"], kind).items():
        assert all(asset["files"][name][key] == value for key, value in expected.items())
    assert not any(name.endswith(".pt") for name in asset["files"])
observations = []
for case in receipt["checks"]:
    raw_path = ROOT / case["stdout"]["path"]
    portable = locations[str(raw_path)]
    assert sha(portable) == case["stdout"]["sha256"]
    assert case["exit_code"] == 0
    output = read(portable)
    if case["case"] == "av-early": assert output["answer"] == "circle" and output["sound"]["samples"] == 4000
    if case["case"] == "av-late": assert output["answer"] == "square" and output["sound"]["samples"] == 4000
    if case["case"].startswith("av-missing"):
        assert output["answer"] == "unknown" and "sound" not in output and "waveform" not in output
    if case["case"] == "sound": assert (output["samples"], output["sample_rate"]) == (4000, 8000)
    if case["case"] == "choice":
        assert (output["prediction"]["answer"],output["information"]["answer"],output["accepted"],output["decision"]) == ("record","yes",True,"accept")
    observations.append({"case": case["case"], "original_command": case["command"],
                         "original_started_at": case["started_at"], "original_finished_at": case["finished_at"],
                         "original_exit_code": case["exit_code"], "stdout_reference": ref(portable), "stdout": output})

# Archive hashes certify which raw rows were inspected. No formal inference replay.
assert sha(RAW_ARCHIVE) == "50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3"
with zipfile.ZipFile(RAW_ARCHIVE) as archive:
    manifest = json.loads(archive.read("package-file-manifest.json"))
    assert set(archive.namelist()) == set(manifest) | {"package-file-manifest.json"}
    for name, binding in manifest.items():
        raw = archive.read(name)
        assert len(raw) == binding["bytes"] and hashlib.sha256(raw).hexdigest() == binding["sha256"]
    for part in ("av","sound","choice","integration"):
        name = f"measurements/{part}/results.json"
        assert archive.read(name) == (FINAL / name).read_bytes()

selection = read(ROOT / "docs/omni-products/v1/model-selection.json")
metrics = read(ROOT / "docs/omni-products/v1/metrics.json")
core_inputs = read(FINAL / "core-inputs.json")
assert core_inputs["core_selection_sha256"] == sha(ROOT / "docs/omni-products/v1/model-selection.json")
for name, digest in core_inputs["files"].items():
    assert sha(FINAL / "frozen" / name) == digest == sha(ROOT / "docs/omni-products/v1" / name)
av = read(FINAL / "measurements/av/results.json")
av_counts = {}
for name, model in av["models"].items():
    rows = model["records"]
    assert len(rows) == len({row["family"] for row in rows}) == 200
    paired = sum(row["answers"] == row["gold"] for row in rows)
    swap = sum(row["waveform_swap_answers"] == row["waveform_swap_gold"] for row in rows)
    swap_obs = sum(a == g for row in rows for a, g in zip(row["waveform_swap_answers"], row["waveform_swap_gold"]))
    assert paired == model["primary"]["correct"] and swap == model["same_metadata_waveform_swap"]["paired"]["correct"]
    av_counts[name] = {"paired": paired, "families": 200, "waveform_swap_paired": swap,
                       "waveform_swap_observations": swap_obs,
                       "remove_image": sum(row["remove_image_answer"] == 2 for row in rows),
                       "remove_audio": sum(row["remove_audio_answer"] == 2 for row in rows)}
assert av_counts[selection["selected"]["av"]] == {"paired":200,"families":200,"waveform_swap_paired":148,"waveform_swap_observations":348,"remove_image":200,"remove_audio":200}
sound = read(FINAL / "measurements/sound/results.json")
gold_rows = sound["codec_and_gold_program"]["rows"]
chain_rows = sound["actual_A_chain"]["rows"]
sound_counts = {"gold_families":len(gold_rows), "gold_correct":sum(row["generation"]["program_success"] for row in gold_rows),
                "gold_legal":sum(row["generation"]["legal"] for row in gold_rows),
                "active_reference_mean_snr_db":sum(row["active_reference_snr_db"] for row in gold_rows)/len(gold_rows),
                "chain_families":len(chain_rows), "chain_correct":sum(row["chain_correct"] for row in chain_rows),
                "chain_legal":sum(row["legal"] for row in chain_rows)}
controls = sound["codec_and_gold_program"]["chunk_controls"]
sound_counts["chunk_control_programs"] = len(controls)
sound_counts["all_chunk_equivalence_and_future_prefix_checks"] = all(
    row["chunks_bit_exact"] and row["future_code_prefix_bit_exact"] for row in controls)
assert len(controls) == 10 and sound_counts["all_chunk_equivalence_and_future_prefix_checks"]
assert sound_counts["gold_families"] == sound_counts["gold_correct"] == sound_counts["gold_legal"] == 200
assert sound_counts["chain_families"] == sound_counts["chain_correct"] == sound_counts["chain_legal"] == 200
assert sound_counts["active_reference_mean_snr_db"] >= metrics["sound"]["active_reference_snr_minimum_db"]
c = read(FINAL / "measurements/choice/results.json")["primary"]
rows = c["rows"]
c_counts = {kind:sum(row["tasks"][kind]["answer"] == row["tasks"][kind]["gold"] for row in rows) for kind in ("choice","info","rubric")}
accepted = errors = 0
for row in rows:
    pred, info = row["tasks"]["choice"], row["choice_info"]
    pmax = max(calibrated(pred["logits"], selection["temperature"]))
    take = pmax >= selection["threshold"] and pred["answer"] != "none" and info["answer"] == "yes"
    accepted += take
    errors += take and (pred["answer"] != pred["gold"] or not row["world_supported"] or not row["choice_information_sufficient"])
assert (c_counts, accepted, errors) == ({"choice":39,"info":55,"rubric":35},53,19)
assert (accepted, errors) == (c["policy"]["accepted"], c["policy"]["errors"])
c_counts.update(families=len(rows),accepted=accepted,errors=errors,risk=errors/accepted,
                T=selection["temperature"],tau=selection["threshold"],
                choice_pass=c_counts["choice"] >= metrics["C"]["choice_minimum_correct"],
                risk_pass=errors/accepted <= metrics["C"]["maximum_empirical_accepted_risk"])
training = []
for kind, job in selection["selected"].items():
    path = CORE / "core" / job / "training/training-result.json"
    value = read(path)
    assert sha(path) == selection["model_files"][kind]["training_result_sha256"]
    assert value["complete"] and value["optimizer_updates_observed"] and not value["final_opened"]
    assert value["initial_parameter_sha256"] != value["latest_parameter_sha256"]
    for name, digest in value["implementation_sha256"].items():
        assert sha(ROOT / "tiny_perceptron/omni_products" / name) == digest
    training.append({"component":kind,"job":job,"source":ref(path),"updates":value["updates"],
                     "initial_parameter_sha256":value["initial_parameter_sha256"],
                     "latest_parameter_sha256":value["latest_parameter_sha256"],
                     "runtime":value["runtime"],"source_unchanged_during_training":value["source_unchanged_during_training"]})
selection_receipts = []
for name, binding in selection["selection_evidence"].items():
    path = CORE / "selection-evidence" / name
    assert sha(path) == binding["sha256"] and path.stat().st_size == binding["bytes"]
    selection_receipts.append(ref(path))
assert read(CORE / "selection-evidence/temperature.json")["temperature"] == selection["temperature"]
assert read(CORE / "selection-evidence/policy.json")["threshold"] == selection["threshold"]

commands = []
for command in [["uv","sync","--extra","cpu","--locked","--dry-run"],
                ["uv","run","--extra","cpu","--no-sync","python","-m","scripts.omni_revision.public_download","--help"],
                ["uv","run","--extra","cpu","--no-sync","python","-m","scripts.omni_revision.public_demo","--help"]]:
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45)
    commands.append({"command":command,"exit_code":run.returncode,"stdout":run.stdout,"stderr":run.stderr})
    assert run.returncode == 0

result = {"started_at":started,"created_at":datetime.now(UTC).isoformat(),
          "reviewer_task":"/root/omni_p5_technical_model_guide",
          "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"device":"CPU",
                         "platform":platform.platform()},
          "constraints":{"trained_weight_loads":0,"formal_or_teacher_NN_forwards":0,"training_runs":0,
                         "retuning":False,"Modal_or_workflow_dispatch":False,"publication":False},
          "body":ref(body_path),"inline_json":{"sound_program":program_json,"choice_matches_default":True},
          "local_links":links,"generated_input":{"family":family,"recording_ms":len(first.waveform)/16,
              "pulse_ms":[pulse_start,pulse_end],"frequency_hz":440,"frames":len(first.frames),
              "frame_step_ms":250,"duration_ms":3750,"shape_runs":runs,"pair":av_inputs},
          "contract_and_fixture_execution":{"invalid_sound_inputs_rejected":rejected,"gold_rejected":True,
              "none_user_candidate_rejected":True,"same_state_binding_enforced":binding_rejected,
              "synthetic_policy_cases":["missing-info handoff","eligible accept","none handoff","low-probability handoff"],
              "toy_download_default_files":len(ordinary),"toy_download_full_files":len(full),
              "opaque_pt_not_deserialized":True,"different_existing_file_preserved":True,"WAV_overwrite_rejected":True},
          "original_public_execution":{"producer_receipt":ref(producer_receipt),"cases":observations,
              "original_download_receipt":ref(original_download_receipt),
              "original_download_record":download_record,
              "original_environment":"Producer states an already installed CPU environment, uv --no-sync; this reviewer did not rerun inference.",
              "demonstration_cases":6,"new_inference_cases":0},
          "original_final":{"archive":ref(RAW_ARCHIVE),"manifest_members_verified":len(manifest),
              "raw_result_sources":[ref(FINAL/f"measurements/{p}/results.json") for p in ("av","sound","choice","integration")],
              "av_row_recounts":av_counts,"sound_row_recounts":sound_counts,"choice_row_recounts":c_counts},
          "original_training":{"selected_receipts":training,"selection_evidence":selection_receipts,
              "inspection":"Read train_av/train_choice/train_codec/train_talker constructors and formal_local dispatch: fresh seeded constructors, self-authored generator labels and own codec codes, no Qwen/Kev weights/output API. Original receipts bind exact inspected source SHA. No weights/checkpoint objects were loaded."},
          "commands_executed":commands,
          "not_verified":["New neural inference or reproduction of measured capability", "Fresh package installation (uv sync was dry-run)",
              "Native Windows/MPS/CUDA or Colab execution", "Exact cross-hardware continuation",
              "Desktop/mobile visual layout (this page has no figures)", "Arbitrary external-world Choice reliability"]}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"output":str(OUT.relative_to(ROOT)),"CPU_demo_originals_inspected":6,
                  "final_archive_members":len(manifest),"T":selection["temperature"],"tau":selection["threshold"],
                  "AV_selected":av_counts[selection["selected"]["av"]],"C":c_counts,
                  "model_loads":0,"NN_forwards":0},ensure_ascii=False))
