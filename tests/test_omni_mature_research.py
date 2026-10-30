"""成熟研究的費用／輸入／單次派工／精確停止反例，全部離線fixture。"""

import asyncio
import copy
import hashlib
import io
import json
import sys
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts.omni_revision import research_core_inputs as core_inputs
from scripts.omni_revision import research_execution as runner
from scripts.omni_revision import research_measurements as checks
from scripts.omni_revision import research_model_files as model_files
from scripts.omni_revision import research_recovery as recovery
from scripts.omni_revision import research_worker as worker
from scripts.omni_revision import staged_execution as staged
from scripts.omni_revision.experiment_execution import storage_quote
from scripts.omni_revision.finance import digest
from scripts.omni_revision.kev_research import public_record, state_view
from scripts.omni_revision.qwen_inputs import make_inputs


def core_fixture():
    return {"core_selection_sha256": "fixture-selection", "core_final_opening_sha256": "fixture-opening", "files": {}}


def rates_fixture():
    return {
        "gpu_hour_cost_l4": "0.8",
        "gpu_hour_cost_l40s": "1.95",
        "cpu_hour_cost_sandbox": "0.1419",
        "mem_gib_hour_cost_sandbox": "0.024",
        "egress_gib_cost": "0.04",
    }


def storage_fixture():
    return storage_quote(b"<p>Price per GB/month Shared storage (artifacts and GitHub Packages) $0.25 USD</p>")


@pytest.mark.parametrize("name,total,hourly", [("kev", "4.30", "3.6696"), ("qwen25", "3.10", "2.5196")])
def test_full_fixed_shape_cost_bound(name, total, hourly):
    job = runner.job_config(name, core_fixture())
    proof = runner.cost_proof(rates_fixture(), job, storage_fixture())
    assert proof["reserved_usd"] == total and proof["hourly_resource_bound_usd"] == hourly
    assert job["cpu"] == 4 and job["memory_mib"] == 49152
    assert job["sandbox_seconds"] == 2100 and job["worker_seconds"] == 1800
    assert proof["resource_wall_bound_seconds"] == 2700
    assert not proof["image_build_permitted"] and not proof["new_persistent_storage"]
    assert not proof["automatic_retry_or_gpu_upgrade"]


@pytest.mark.parametrize("key", list(rates_fixture()))
def test_official_rate_increase_rejected_before_create(key):
    rates = rates_fixture()
    rates[key] = str(float(rates[key]) + 0.01)
    job = "kev" if key.endswith("l40s") else "qwen25"
    with pytest.raises(ValueError):
        runner.cost_proof(rates, runner.job_config(job, core_fixture()), storage_fixture())


def test_paid_research_keeps_other_research_and_final_bounds():
    state = {
        "reserved_total_usd": "6.50",
        "attempts": [
            {
                "status": "completed",
                "purpose": json.dumps({"job": "cpu-core-formal"}),
                "receipt": {"success": True, "all_training_processes_exited": True},
            },
            {
                "status": "reserved",
                "purpose": json.dumps({"job": "mature-kev"}),
                "guard": {"retained_official_increment_usd": "0", "exact_legacy_carry_usd": "4.73"},
            },
        ],
    }
    proof = runner.protect_remaining(state, "kev")
    assert proof["future_bounds_usd"] == "5.10"
    assert proof["guarded_total_usd"] == "27.93"
    state["attempts"][0]["status"] = "failed-stopped"
    with pytest.raises(ValueError):
        runner.protect_remaining(state, "kev")
    state["attempts"][0]["status"] = "completed"
    state["reserved_total_usd"] = "30"
    with pytest.raises(ValueError):
        runner.protect_remaining(state, "kev")


def test_requirement_locks_do_not_replace_torch_cuda(tmp_path, monkeypatch):
    assert worker.requirements("kev").exists() and worker.requirements("qwen25").exists()
    monkeypatch.chdir(tmp_path)
    path = Path("scripts/omni_revision/research_requirements/kev.txt")
    path.parent.mkdir(parents=True)
    for text in (
        "torch==2.8.0",
        "triton==3.4.0",
        "nvidia-cublas-cu12==12.8.0",
        "numpy>=2",
        "--index-url https://fixture.invalid",
        "numpy==2.5.3\nnumpy==2.5.3",
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            worker.requirements("kev")


def test_short_bootstrap_checks_archive_identity_and_paths(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape", b"unsafe")
    payload = buffer.getvalue()
    code = staged.staged_bootstrap(payload)
    assert len(code.encode()) < 4096 and "research_worker" in code
    code = (
        code.replace("/tmp/omni-input.zip", (tmp_path / "input.zip").as_posix())
        .replace("/tmp/omni-input.sha256", (tmp_path / "input.sha256").as_posix())
        .replace("/tmp/omni-job", (tmp_path / "root").as_posix())
    )
    (tmp_path / "input.zip").write_bytes(payload)
    (tmp_path / "input.sha256").write_text(hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="unsafe input"):
        exec(compile(code, "private-bootstrap-fixture", "exec"), {})
    assert not (tmp_path / "escape").exists()
    monkeypatch.setattr(staged, "MAX_INPUT_BYTES", 2)
    with pytest.raises(ValueError):
        staged.staged_bootstrap(b"123")


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "worker_failure",
        "corrupt_result",
        "wrong_selection",
        "lost_empty",
        "lost_found",
        "stop_failure",
        "list_failure",
        "input_failure",
    ],
)
def test_single_create_staged_transfer_and_exact_stop(tmp_path, mode):
    job = {**runner.job_config("kev", core_fixture()), "config_sha256": "fixture-config"}
    names = {
        "app_name": "omni-mature-kev-123-1",
        "tags": {"omni-round": "omni-20261029", "omni-attempt": "mature-kev-123-1"},
    }
    calls, stopped = [], set()
    exported = b"explicit-fixture-result"
    result = {
        "job": "kev",
        "success": mode != "worker_failure",
        "config_sha256": "fixture-config",
        "core_selection_sha256": "other" if mode == "wrong_selection" else "fixture-selection",
        "core_final_opening_sha256": "fixture-opening",
        "final_opened": mode != "worker_failure",
        "bytes": len(exported),
        "sha256": hashlib.sha256(exported).hexdigest(),
    }

    class App:
        def __init__(self, name):
            self.app_id = "ap-owned"

        @asynccontextmanager
        async def run(self, **kwargs):
            yield self

    class Sandbox:
        def __init__(self):
            self.filesystem = self

        async def _get_task_id(self, **kwargs):
            return "ta-owned"

        async def write_bytes(self, data, remote_path):
            assert isinstance(data, bytes) and isinstance(remote_path, str)
            calls.append(("write", remote_path))
            if mode == "input_failure":
                raise ConnectionError("fixture")

        async def read_bytes(self, path):
            assert not stopped
            calls.append(("read", path))
            if path.endswith(".json"):
                return json.dumps(result).encode()
            return exported + (b"wrong" if mode == "corrupt_result" else b"")

        @property
        def stdout(self):
            async def parts():
                yield "OMNI_JOB_READY\n"

            return parts()

    sandbox = Sandbox()

    async def create(request, **kwargs):
        calls.append(("create", request))
        assert kwargs == {"retry": None, "timeout": 480}
        assert request.definition.timeout_secs == 2100
        assert len(request.definition.entrypoint_args[-1].encode()) < 4096
        if mode.startswith("lost_"):
            raise TimeoutError("fixture create response lost")
        return SimpleNamespace(sandbox_id="sb-owned", metadata=None)

    async def listing(request, **kwargs):
        if mode == "list_failure":
            raise ConnectionError("fixture discovery failure")
        rows = (
            []
            if stopped or mode == "lost_empty" or request.before_timestamp
            else [SimpleNamespace(id="sb-owned", created_at=1)]
        )
        return SimpleNamespace(sandboxes=rows)

    async def terminate(request, **kwargs):
        calls.append(("terminate", request.sandbox_id))
        if mode == "stop_failure":
            raise RuntimeError("fixture cannot stop")
        stopped.add(request.sandbox_id)

    async def image(*args, **kwargs):
        return SimpleNamespace(image_id=runner.IMAGE_ID)

    async def wait(*args, **kwargs):
        return SimpleNamespace(result=SimpleNamespace(status=1))

    async def close():
        calls.append(("close", None))

    api = SimpleNamespace(
        **{
            name: SimpleNamespace
            for name in (
                "SandboxTag",
                "ImageFromIdRequest",
                "Sandbox",
                "SandboxCreateRequest",
                "SandboxListRequest",
                "SandboxTerminateRequest",
                "SandboxWaitRequest",
            )
        }
    )
    client = SimpleNamespace(
        _stub=SimpleNamespace(
            ImageFromId=image, SandboxCreate=create, SandboxList=listing, SandboxTerminate=terminate, SandboxWait=wait
        ),
        _close=close,
    )
    hydrated = SimpleNamespace(_new_hydrated=lambda *args: sandbox)
    receipt = asyncio.run(
        staged.run_staged_job(
            client,
            api,
            App,
            hydrated,
            names,
            tmp_path,
            [],
            {"gpu": "L40S"},
            {"network": "fixture"},
            job,
            b"fixed input fixture",
        )
    )
    assert len([call for call in calls if call[0] == "create"]) == 1
    assert receipt["success"] is (mode == "normal")
    assert receipt["cleanup_checks"]["client_closed"]
    if mode not in ("lost_empty", "stop_failure", "list_failure"):
        assert receipt["cleanup_checks"]["owned_sandboxes_stopped"]
    else:
        assert not receipt["cleanup_checks"]["owned_sandboxes_stopped"]
    if mode == "normal":
        assert receipt["result_exported_before_stop"] and receipt["research_final_opened"]
        assert [call[0] for call in calls].index("read") < [call[0] for call in calls].index("terminate")
        assert (tmp_path / "result.zip").read_bytes() == exported


def test_worker_bundle_contains_real_old_A_dependencies():
    files = set(runner.worker_files())
    for name in ("model", "modern", "multimodal", "data", "attention", "__init__"):
        assert f"tiny_perceptron/{name}.py" in files
    assert all((runner.ROOT / name).is_file() for name in files)


def test_kev_public_view_preserves_unknown_and_does_not_send_gold():
    assert "permission=unknown" in state_view(
        "裝置A：任務record；許可未知；就緒是。 裝置B：任務play；許可否；就緒否。 裝置C：任務save；許可是；就緒未知。",
        "english",
    )
    question = {
        "state": "A: task=record; permission=unknown; ready=yes. B: task=play; permission=no; ready=no. C: task=save; permission=yes; ready=unknown.",
        "kind": "choice",
        "text": "Choose A.",
        "candidates": [{"id": "record", "description": "Record"}],
        "gold": "none",
    }
    public, ids = public_record({"questions": {"choice": question}}, "english", ("choice",))
    assert public["questions"][0]["label"] is None
    assert ids == [["record", "none"]] and "gold" not in json.dumps(public)


def test_qwen_preprocessing_contract_without_loading_neural_weights():
    calls = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            return "explicit prompt fixture"

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            return {"input_ids": torch.ones(1, 231, dtype=torch.long), "video_grid_thw": torch.tensor([[4, 8, 8]])}

    make_inputs(
        Processor(), text="What color?", audio=torch.zeros(64000), frames=[SimpleNamespace(size=(112, 112))] * 8
    )
    assert isinstance(calls["messages"][0]["content"], list)
    video = calls["kwargs"]["videos_kwargs"]
    assert video["size"] == {"shortest_edge": 12544, "longest_edge": 12544}
    assert video["fps"] == 2.0 and video["use_audio_in_video"]


def result_zip(path, files):
    manifest = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr("file-manifest.json", json.dumps(manifest))


def test_partial_results_are_verified_but_never_upstream_weights(tmp_path):
    summary = {
        "schema": "omni-mature-worker-summary-v1",
        "job": "kev",
        "core_selection_sha256": "fixture-selection",
        "success": False,
        "final_opened": False,
        "own_training_permitted": False,
    }
    path = tmp_path / "result.zip"
    files = {"worker-summary.json": json.dumps(summary).encode(), "install.log": b"explicit fixture failure"}
    result_zip(path, files)
    result = runner.verify_results(path, runner.job_config("kev", core_fixture()))
    assert result["all_entries_verified"] and not result["worker_success"]
    result_zip(path, {**files, "model.safetensors": b"forbidden fixture"})
    with pytest.raises(ValueError, match="不允許"):
        runner.verify_results(path, runner.job_config("kev", core_fixture()))


def test_bounded_artifact_storage(tmp_path, monkeypatch):
    (tmp_path / "control.json").write_bytes(b"123")
    monkeypatch.setattr(runner, "MAX_CONTROL_BYTES", 2)
    with pytest.raises(ValueError):
        runner.bound(tmp_path)


def test_incomplete_core_selection_rejected_before_download(tmp_path, monkeypatch):
    frozen = tmp_path / "docs/frozen"
    frozen.mkdir(parents=True)
    for name in (
        "data-definition.json",
        "metrics.json",
        "formal-recipes.json",
        "codec-selection.json",
        "model-selection.json",
        "final-opening.json",
    ):
        (frozen / name).write_text("{}")
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    monkeypatch.setattr(core_inputs, "ROOT", tmp_path)
    monkeypatch.setattr(core_inputs, "FROZEN", frozen)
    monkeypatch.setattr(core_inputs.core_final, "AUDIT", audit)
    monkeypatch.setattr(
        core_inputs.subprocess,
        "check_output",
        lambda args, **kwargs: (tmp_path / args[2].split(":", 1)[1]).read_bytes(),
    )
    with pytest.raises(ValueError, match="缺完整核心selection"):
        core_inputs.check_metadata("explicit-private-fixture")


def test_core_archive_rejects_forged_entry_hash(tmp_path):
    path = tmp_path / "core.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("core/codec/training/weights.safetensors", b"explicit fixture")
        archive.writestr(
            "package-file-manifest.json",
            json.dumps({"core/codec/training/weights.safetensors": {"bytes": 16, "sha256": "wrong"}}),
        )
    with pytest.raises(ValueError, match="SHA"):
        core_inputs.unpack_verified(path, tmp_path / "destination")


@pytest.fixture
def qwen_failed_fixture():
    # 完全人工失敗觀察；只證完整分母，不存在神經模型或研究成績。
    cases = {
        "audio_text_pairs": [{"id": f"speech-pair-{i:02}", "gold": "no"} for i in range(24)],
        "av_shift_pairs": [
            {"id": f"rgb-time-pair-{i:02}", "pair": [{"gold": "red"}, {"gold": "blue"}]} for i in range(24)
        ],
        "speech_output_prompts": [{"id": f"short-speech-{i}", "gold_word": "no"} for i in range(8)],
    }
    failure = {"error_type": "ExplicitFixtureFailure", "error": "no neural call happened", "correct": False}
    raw = [
        {"panel": "audio-text", "id": case["id"], "modality": modality, "gold": case["gold"], **failure}
        for case in cases["audio_text_pairs"]
        for modality in ("audio", "text")
    ]
    raw.extend(
        {"panel": "AV-shift", "id": case["id"], "which": i, "gold": item["gold"], **failure}
        for case in cases["av_shift_pairs"]
        for i, item in enumerate(case["pair"])
    )
    speech = [
        {
            "id": case["id"],
            "gold_word": case["gold_word"],
            "generated": False,
            "error_type": "ExplicitFixtureFailure",
            "error": "no neural call happened",
            "ASR_proxy_attempted": False,
            "ASR_proxy_correct": None,
            "ASR_proxy_skip_reason": "no generated audio",
        }
        for case in cases["speech_output_prompts"]
    ]
    result = {
        "model": "Qwen/Qwen2.5-Omni-3B",
        "revision": "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        "audio_text_pairs": [
            {"id": case["id"], "gold": case["gold"], "audio": False, "text": False, "both_correct": False}
            for case in cases["audio_text_pairs"]
        ],
        "AV_shift_pairs": [
            {"id": case["id"], "observations": [False, False], "both_correct": False}
            for case in cases["av_shift_pairs"]
        ],
        "audio_text_counts": {"families": 24, "both_correct": 0},
        "AV_counts": {"families": 24, "both_correct": 0},
        "speech_outputs": speech,
        "speech_count": 8,
        "speech_counts": {
            "prompts": 8,
            "generated": 0,
            "legal": 0,
            "ASR_proxy_attempted": 0,
            "ASR_proxy_correct_out_of_all8": 0,
        },
        "ASR_proxy": {"model": "openai/whisper-small", "revision": "973afd24965f72e36ca33b3055d56a652f456b4d"},
        "teacher_outputs_used_for_own_training": False,
        "dtype": "BF16 requested neural load; official Token2Wav converts to FP32 when decoding speech; no quantization",
        "parameter_numel_by_dtype_after_load": {
            name: {"torch.bfloat16": 10} for name in ("thinker", "talker", "token2wav")
        },
        "parameter_numel_by_dtype_after_speech": {
            name: {"torch.bfloat16": 10} for name in ("thinker", "talker", "token2wav")
        },
    }

    def jsonl(rows):
        return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()

    files = {
        "qwen-raw-predictions.jsonl": jsonl(raw),
        "qwen-speech-raw.jsonl": jsonl(speech),
        "qwen-asr-raw.jsonl": jsonl(speech),
    }
    for model, revision, filename in (
        ("Qwen/Qwen2.5-Omni-3B", "f75b40e3da2003cdd6e1829b1f420ca70797c34e", "qwen-frozen-input-files.json"),
        ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "asr-frozen-input-files.json"),
    ):
        names = ["config.json", "model.safetensors"] + (["spk_dict.pt"] if model.startswith("Qwen/") else [])
        note = {
            "schema": "omni-fixed-public-model-input-files-v1",
            "model": model,
            "revision": revision,
            "local_files_only_for_neural_load": True,
            "implicit_tokens_permitted": False,
            "ordinary_git_weight_bytes": 0,
            "files": {name: {"bytes": 1, "sha256": "1" * 64, "source_lfs_sha256": None} for name in names},
        }
        files[filename] = json.dumps(note).encode()
    return cases, result, files


def write_qwen_fixture(tmp_path, fixture):
    cases, result, files = fixture
    raw = json.dumps(cases).encode()
    with zipfile.ZipFile(tmp_path / "input.zip", "w") as archive:
        archive.writestr("cases/cases.json", raw)
    output = tmp_path / "observations.zip"
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    job = {"input_files": {"cases/cases.json": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}
    return output, result, job


def test_all104_failed_Qwen_attempts_remain_complete(tmp_path, qwen_failed_fixture):
    path, result, job = write_qwen_fixture(tmp_path, qwen_failed_fixture)
    with zipfile.ZipFile(path) as archive:
        checks.check_qwen(archive, result, job, tmp_path / "input.zip")
    assert result["speech_counts"]["ASR_proxy_correct_out_of_all8"] == 0


@pytest.mark.parametrize(
    "missing", ["one_raw_row", "empty_raw_rows", "speech_raw", "proxy_row", "pair_count", "wrong_gold", "false_correct"]
)
def test_missing_Qwen_observations_cannot_be_complete(tmp_path, qwen_failed_fixture, missing):
    cases, result, files = copy.deepcopy(qwen_failed_fixture)
    if missing == "one_raw_row":
        files["qwen-raw-predictions.jsonl"] = b"\n".join(files["qwen-raw-predictions.jsonl"].splitlines()[:-1])
    elif missing == "empty_raw_rows":
        files["qwen-raw-predictions.jsonl"] = ("{}\n" * 96).encode()
    elif missing == "speech_raw":
        files["qwen-speech-raw.jsonl"] = ("{}\n" * 8).encode()
    elif missing == "proxy_row":
        files["qwen-asr-raw.jsonl"] = b"\n".join(files["qwen-asr-raw.jsonl"].splitlines()[:-1])
    elif missing == "pair_count":
        result["audio_text_counts"]["both_correct"] = 1
    elif missing == "wrong_gold":
        rows = [json.loads(line) for line in files["qwen-raw-predictions.jsonl"].decode().splitlines()]
        rows[0]["gold"] = "wrong"
        files["qwen-raw-predictions.jsonl"] = ("\n".join(map(json.dumps, rows)) + "\n").encode()
    else:
        rows = [json.loads(line) for line in files["qwen-raw-predictions.jsonl"].decode().splitlines()]
        rows[0]["correct"] = True
        files["qwen-raw-predictions.jsonl"] = ("\n".join(map(json.dumps, rows)) + "\n").encode()
    path, result, job = write_qwen_fixture(tmp_path, (cases, result, files))
    with zipfile.ZipFile(path) as archive, pytest.raises((ValueError, KeyError)):
        checks.check_qwen(archive, result, job, tmp_path / "input.zip")


def test_empty_Kev_final_view_rejected_even_with_valid_row_counts(tmp_path):
    files = {
        "kev-calibration.json": json.dumps([{}] * 80).encode(),
        "kev-policy-predictions.json": json.dumps([{}] * 60).encode(),
        "kev-final-english.json": b"{}",
        "kev-final-chinese.json": b"{}",
    }
    path = tmp_path / "kev.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    with zipfile.ZipFile(path) as archive, pytest.raises((ValueError, KeyError)):
        checks.check_kev(archive, {}, core_fixture())


@pytest.fixture
def kev_complete_fixture():
    from scripts.omni_revision.core_evaluation import evaluate_c
    from scripts.omni_revision.kev_research import ADAPTER, BASE, BASE_REVISION
    from tiny_perceptron.omni_products.reliability import select_policy, select_temperature

    data = json.loads((runner.FROZEN / "data-definition.json").read_bytes())

    def predicted(question):
        ids = [row["id"] for row in question["candidates"]] + (["none"] if question["kind"] == "choice" else [])
        return {
            "ids": ids,
            "logits": [8.0 if x == question["gold"] else 0.0 for x in ids],
            "answer": question["gold"],
            "gold": question["gold"],
        }

    def family(row, language):
        return {
            "family": row["family"],
            "source": row["source"] + "/" + language + "-state-view",
            "panel": row["panel"],
            "tasks": {kind: predicted(row["questions"][kind]) for kind in ("choice", "info", "rubric")},
            "choice_info": predicted(row["questions"]["choice_info"]),
            "world_supported": row["world_supported"],
            "choice_information_sufficient": row["choice_information_sufficient"],
            "context_tokens": 100,
            "state_tokens": 30,
            "forward_seconds": 0.0,
        }

    calibration = [family(row, "english") for row in data["C"]["panels"]["calibration"]]
    policy = [family(row, "english") for row in data["C"]["panels"]["policy"]]
    temperature = select_temperature(calibration)
    selection = {
        "temperature": temperature,
        "policy": select_policy(policy, temperature["temperature"]),
        "upstream_temperature": 2.406050072164233,
        "head_temperature_during_raw_logits": 1.0,
        "core_selection_sha256": "fixture-selection",
        "final_opened": False,
    }
    files = {
        "kev-calibration.json": calibration,
        "kev-policy-predictions.json": policy,
        "kev-reliability-selection.json": selection,
    }
    raw = [
        {"split": split, "language": "english", "prediction": row}
        for split, rows in (("calibration", calibration), ("policy", policy))
        for row in rows
    ]
    for language in ("english", "chinese"):
        rows = [family(row, language) for row in data["C"]["panels"]["final"]]
        files[f"kev-final-{language}.json"] = {
            "raw": evaluate_c(rows, 1.0, None),
            "upstream_temperature": evaluate_c(rows, selection["upstream_temperature"], None),
            "new_temperature_policy": evaluate_c(rows, temperature["temperature"], selection["policy"]["threshold"]),
        }
        raw.extend({"split": "final", "language": language, "prediction": row} for row in rows)
    cache = {
        "development_families": 8,
        "atol": 1e-5,
        "all_observed_within_tolerance": True,
        "rows": [
            {
                "family": row["family"],
                "prefix_before": "1" * 64,
                "prefix_after": "1" * 64,
                "uncached_cached_max_abs": 0.0,
                "Q1_Q2_Q1_probability_max_abs": 0.0,
            }
            for row in data["C"]["panels"]["dev"][:8]
        ],
    }
    result = {
        "adapter": ADAPTER,
        "base": BASE,
        "base_revision": BASE_REVISION,
        "families_per_language": 60,
        "language_views_share_same60_families": True,
        "own_training_used_Kev_outputs": False,
        "option_isolation": False,
        "selection": selection,
        "cache": cache,
    }
    encoded = {name: json.dumps(value).encode() for name, value in files.items()}
    encoded["kev-raw-predictions.jsonl"] = ("\n".join(map(json.dumps, raw)) + "\n").encode()
    return result, encoded


@pytest.mark.parametrize("mutation", [None, "empty_view", "missing_type", "wrong_selected_T", "missing_raw_family"])
def test_complete_Kev_views_need_every_actual_observation(tmp_path, kev_complete_fixture, mutation):
    result, files = copy.deepcopy(kev_complete_fixture)
    if mutation == "empty_view":
        files["kev-final-english.json"] = b"{}"
    elif mutation == "missing_type":
        value = json.loads(files["kev-final-chinese.json"])
        value["raw"]["types"].pop("rubric")
        files["kev-final-chinese.json"] = json.dumps(value).encode()
    elif mutation == "wrong_selected_T":
        value = json.loads(files["kev-reliability-selection.json"])
        value["temperature"]["temperature"] = 4.0
        files["kev-reliability-selection.json"] = json.dumps(value).encode()
    elif mutation == "missing_raw_family":
        files["kev-raw-predictions.jsonl"] = b"\n".join(files["kev-raw-predictions.jsonl"].splitlines()[:-1])
    path = tmp_path / "observations.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    with zipfile.ZipFile(path) as archive:
        if mutation is None:
            checks.check_kev(archive, result, core_fixture())
        else:
            with pytest.raises((ValueError, KeyError)):
                checks.check_kev(archive, result, core_fixture())


@pytest.mark.parametrize("wrong_revision", [False, True])
def test_fixed_snapshot_downloads_exact_speaker_and_weights(tmp_path, monkeypatch, wrong_revision):
    revision = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    snapshot = tmp_path / revision
    snapshot.mkdir()
    files = {
        "config.json": b"{}",
        "spk_dict.pt": b"explicit speaker fixture, not a tensor",
        "model.safetensors": b"explicit weight fixture, not neural weights",
    }
    siblings = []
    for name, raw in files.items():
        path = snapshot / name
        path.write_bytes(raw)
        sha256, git_sha = model_files.file_digests(path)
        siblings.append(
            SimpleNamespace(
                rfilename=name,
                size=len(raw),
                blob_id=git_sha,
                lfs=SimpleNamespace(sha256=sha256) if name.endswith(".safetensors") else None,
            )
        )
    calls = []

    class API:
        def __init__(self, **kwargs):
            assert kwargs == {"token": False}

        def model_info(self, model, **kwargs):
            assert model == "Qwen/Qwen2.5-Omni-3B" and kwargs == {"revision": revision, "files_metadata": True}
            return SimpleNamespace(sha="wrong" if wrong_revision else revision, siblings=siblings)

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    hub = ModuleType("huggingface_hub")
    hub.HfApi, hub.snapshot_download = API, download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    if wrong_revision:
        with pytest.raises(ValueError, match="commit"):
            model_files.frozen_snapshot("Qwen/Qwen2.5-Omni-3B", revision)
        assert not calls
    else:
        root, note = model_files.frozen_snapshot("Qwen/Qwen2.5-Omni-3B", revision)
        assert root == snapshot and set(note["files"]) == set(files)
        assert calls == [
            {
                "repo_id": "Qwen/Qwen2.5-Omni-3B",
                "revision": revision,
                "token": False,
                "allow_patterns": sorted(files),
                "max_workers": 4,
            }
        ]
        assert note["files"]["spk_dict.pt"]["sha256"] == hashlib.sha256(files["spk_dict.pt"]).hexdigest()


def test_precision_records_each_component_and_official_float_decoder():
    components = SimpleNamespace(
        **{name: torch.nn.Linear(2, 2).to(torch.bfloat16) for name in ("thinker", "talker", "token2wav")}
    )
    before = model_files.component_parameter_dtypes(components)
    components.token2wav.float()
    after = model_files.component_parameter_dtypes(components)
    result = {
        "parameter_numel_by_dtype_after_load": before,
        "parameter_numel_by_dtype_after_speech": after,
        "speech_outputs": [{"generated": True}],
    }
    checks.parameter_precision(result)
    assert before["thinker"] == before["token2wav"] == {"torch.bfloat16": 6}
    assert after["token2wav"] == {"torch.float32": 6}
    result["parameter_numel_by_dtype_after_speech"] = before
    with pytest.raises(ValueError, match="FP32"):
        checks.parameter_precision(result)


@pytest.fixture
def preparation_recovery_fixture(tmp_path, monkeypatch):
    """只用自製狀態測守門；真舊ZIP／ledger由獨立runtime review另核。"""
    identity = copy.deepcopy(recovery.IDENTITY)
    receipt = {
        "phase": "uploading_fixed_input_payload",
        "error_type": "TypeError",
        "app_id": identity["app_id"],
        "sandbox_id": identity["sandbox_id"],
        "observed_owned_sandbox_ids": [identity["sandbox_id"]],
        "stopped_sandboxes": {identity["sandbox_id"]: 3},
        "remaining_owned_sandboxes": [],
        "stop_errors": {},
        "compute_stopped": True,
        "resources_verified": True,
        "cleanup_checks": {"official_app_stopped": True, "owned_sandboxes_stopped": True},
    }
    row = {
        "attempt_id": identity["attempt"],
        "status": "failed-stopped",
        "revision": identity["source_revision"],
        "reserved_usd": "4.30",
        "purpose": json.dumps({"job": "mature-kev"}),
        "receipt": receipt,
    }
    identity.update(row_sha256=digest(row), receipt_sha256=digest(receipt))
    monkeypatch.setattr(recovery, "IDENTITY", identity)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    original = {"created_at": "2026-10-30T04:59:00+00:00", "reviewer_task": "/root/independent_fixture"}
    path = tmp_path / recovery.ORIGINAL_NAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(original))
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(recovery, "ORIGINAL_SHA", original_sha)
    permit = {
        "schema": "omni-mature-preparation-recovery-permit-v1",
        "status": "pass_for_exact_preparation_recovery",
        "reviewer_task": original["reviewer_task"],
        "created_at": "2026-10-30T04:59:01+00:00",
        "identity": identity,
        "original_runtime_review": {"path": recovery.ORIGINAL_NAME, "sha256": original_sha},
        "maximum_new_attempts": 1,
        "new_bound_usd": "4.30",
        "source_path_proof": {
            "sdk_version": "1.6.0",
            "first_write_failed_before_any_fs_rpc": True,
            "sha_marker_not_sent_by_failed_path": True,
            "provider_historical_NN_telemetry_directly_observed": False,
        },
    }
    permit_path = tmp_path / recovery.PERMIT_NAME
    permit_path.write_text(json.dumps(permit))
    body = (
        "Owned unpublished limited mature-model research; "
        + identity["attempt"]
        + "; source "
        + identity["source_revision"]
    )
    release = {
        "id": identity["release_id"],
        "draft": True,
        "tag_name": identity["release_tag"],
        "body": body,
        "target_commitish": identity["source_revision"],
        "assets": [
            {
                "id": identity["asset_id"],
                "name": identity["asset_name"],
                "size": identity["asset_bytes"],
                "digest": "sha256:" + identity["asset_sha256"],
            }
        ],
    }
    gets = []

    def get(endpoint):
        assert endpoint == f"releases/{identity['release_id']}"
        gets.append(endpoint)
        return copy.deepcopy(release)

    monkeypatch.setattr(recovery, "api", get)
    return SimpleNamespace(
        state={"attempts": [row]}, row=row, permit=permit, permit_path=permit_path, release=release, gets=gets
    )


def test_general_once_guard_and_exact_preparation_permit(preparation_recovery_fixture):
    f = preparation_recovery_fixture
    assert recovery.eligibility({"attempts": []}, "kev") is None
    assert recovery.eligibility(f.state, "qwen25") is None
    lineage = recovery.eligibility(f.state, "kev")
    assert lineage == recovery.lineage(hashlib.sha256(f.permit_path.read_bytes()).hexdigest())
    assert len(f.gets) == 1 and f.row["reserved_usd"] == "4.30"
    f.state["attempts"].append({"purpose": json.dumps({"job": "mature-kev", **lineage})})
    with pytest.raises(ValueError):
        recovery.eligibility(f.state, "kev")
    assert len(f.gets) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "completed"),
        ("revision", "foreign"),
        ("reserved_usd", "0"),
        ("attempt_id", "foreign"),
    ],
)
def test_recovery_does_not_rewrite_original_attempt(preparation_recovery_fixture, field, value):
    f = preparation_recovery_fixture
    f.row[field] = value
    with pytest.raises(ValueError):
        recovery.eligibility(f.state, "kev")
    assert f.gets == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "worker_stdout_until_export_ready"),
        ("error_type", "TimeoutError"),
        ("input_payload_uploaded", True),
        ("compute_stopped", False),
        ("resources_verified", False),
        ("remaining_owned_sandboxes", ["sb-unknown"]),
        ("stop_errors", {"sb-owned": "unknown"}),
    ],
)
def test_recovery_rejects_changed_or_ambiguous_predecessor(preparation_recovery_fixture, field, value):
    f = preparation_recovery_fixture
    f.row["receipt"][field] = value
    with pytest.raises(ValueError):
        recovery.eligibility(f.state, "kev")
    assert f.gets == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "author_proposal_not_authorized"),
        ("reviewer_task", "/root"),
        ("maximum_new_attempts", True),
        ("maximum_new_attempts", 2),
        ("new_bound_usd", "0"),
        ("created_at", "2026-10-30T04:58:59+00:00"),
    ],
)
def test_recovery_requires_independent_specific_once_permit(preparation_recovery_fixture, field, value):
    f = preparation_recovery_fixture
    f.permit[field] = value
    f.permit_path.write_text(json.dumps(f.permit))
    with pytest.raises(ValueError):
        recovery.eligibility(f.state, "kev")
    assert f.gets == []


@pytest.mark.parametrize("mutation", ["draft", "target", "extra_asset", "digest"])
def test_recovery_rechecks_original_owned_provider_asset(preparation_recovery_fixture, mutation):
    f = preparation_recovery_fixture
    if mutation == "draft":
        f.release["draft"] = False
    elif mutation == "target":
        f.release["target_commitish"] = "foreign"
    elif mutation == "extra_asset":
        f.release["assets"].append({"name": "result.zip"})
    else:
        f.release["assets"][0]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        recovery.eligibility(f.state, "kev")


def test_permit_cannot_be_consumed_under_another_job(preparation_recovery_fixture):
    f = preparation_recovery_fixture
    proof = recovery.eligibility(f.state, "kev")
    f.state["attempts"].append({"purpose": json.dumps({"job": "mature-qwen25", **proof})})
    with pytest.raises(ValueError, match="已使用"):
        recovery.eligibility(f.state, "kev")


@pytest.fixture
def original_input_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    monkeypatch.setattr(recovery, "FROZEN", frozen)
    worker_name = "scripts/omni_revision/research_worker.py"
    files = {worker_name: b"# first-party offline fixture\n"}
    for name in ("data-definition.json", "model-selection.json", "final-opening.json"):
        files[name] = b"{}\n"
        (frozen / name).write_bytes(files[name])
    worker = tmp_path / worker_name
    worker.parent.mkdir(parents=True)
    worker.write_bytes(files[worker_name])
    expected = {"job": "kev", "core_selection_sha256": "fixture-selected"}
    job = {
        **expected,
        "input_files": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()
        },
    }
    raw_job = (json.dumps(job, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr("job.json", raw_job)
    payload = buffer.getvalue()
    identity = {
        **recovery.IDENTITY,
        "asset_bytes": len(payload),
        "asset_sha256": hashlib.sha256(payload).hexdigest(),
        "job_config_sha256": hashlib.sha256(raw_job).hexdigest(),
    }
    monkeypatch.setattr(recovery, "IDENTITY", identity)

    def download(arguments):
        assert arguments[:3] == ["gh", "release", "download"]
        (Path(arguments[-1]) / identity["asset_name"]).write_bytes(payload)

    monkeypatch.setattr(recovery, "command", download)
    output = tmp_path / "new-output"
    output.mkdir()
    return SimpleNamespace(
        output=output,
        payload=payload,
        raw_job=raw_job,
        expected=expected,
        worker_name=worker_name,
        worker=worker,
        frozen=frozen,
    )


def test_recovery_reuses_exact_input_job_and_worker_bytes(original_input_fixture):
    f = original_input_fixture
    job, payload = recovery.reuse_input(f.expected, f.output, [f.worker_name], recovery.lineage("fixture-permit"))
    assert payload == f.payload == (f.output / "input.zip").read_bytes()
    assert (f.output / "job.json").read_bytes() == f.raw_job
    assert job["config_sha256"] == hashlib.sha256(f.raw_job).hexdigest()
    assert json.loads((f.output / "input-preparation.json").read_bytes())["reused_original_owned_input_asset"]
    with pytest.raises(FileExistsError):
        recovery.reuse_input(f.expected, f.output, [f.worker_name], recovery.lineage("fixture-permit"))


@pytest.mark.parametrize("mutation", ["worker", "frozen", "science", "worker_set"])
def test_preparation_recovery_cannot_change_scientific_inputs(original_input_fixture, mutation):
    f = original_input_fixture
    workers = [f.worker_name]
    if mutation == "worker":
        f.worker.write_bytes(b"changed")
    elif mutation == "frozen":
        (f.frozen / "data-definition.json").write_bytes(b"changed")
    elif mutation == "science":
        f.expected["core_selection_sha256"] = "changed"
    else:
        workers = []
    with pytest.raises(ValueError):
        recovery.reuse_input(f.expected, f.output, workers, recovery.lineage("fixture-permit"))
    assert not (f.output / "input.zip").exists()


def receipt_binding_fixture():
    attempt = "mature-kev-123-1"
    sources = {"revision": "a" * 40, "files": {"first-party.py": "b" * 64}, "audit_sha256": "c" * 64}
    purpose = {
        "job": "mature-kev",
        "sources_sha256": digest(sources),
        "controller_audit_sha256": sources["audit_sha256"],
        "ownership": {
            "app_name": "omni-" + attempt,
            "tags": {"omni-round": runner.POLICY["round_id"], "omni-attempt": attempt},
        },
        "input_sha256": "d" * 64,
        "job_config_sha256": "e" * 64,
        "core_selection_sha256": "f" * 64,
        "core_final_opening_sha256": "0" * 64,
    }
    reserved = {
        "attempt_id": attempt,
        "revision": sources["revision"],
        "status": "reserved",
        "purpose": json.dumps(purpose),
    }
    tag, body = runner.release_identity(attempt, sources["revision"])
    receipt = {
        "schema": "omni-mature-controller-receipt-v2",
        "attempt": attempt,
        "job": "kev",
        "source_revision": sources["revision"],
        "controller_audit_sha256": sources["audit_sha256"],
        "release_tag": tag,
        "release_body": body,
        "input_payload_sha256": purpose["input_sha256"],
        "config_sha256": purpose["job_config_sha256"],
        "core_selection_sha256": purpose["core_selection_sha256"],
        "core_final_opening_sha256": purpose["core_final_opening_sha256"],
        "preparation_recovery": None,
    }
    return attempt, sources, reserved, receipt


def test_mature_receipt_is_bound_to_this_attempt_and_sources():
    attempt, sources, reserved, receipt = receipt_binding_fixture()
    runner.check_receipt_identity("kev", attempt, sources, reserved, receipt)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "attempt",
        "job",
        "source_revision",
        "controller_audit_sha256",
        "release_tag",
        "release_body",
        "input_payload_sha256",
        "config_sha256",
        "core_selection_sha256",
        "core_final_opening_sha256",
        "preparation_recovery",
    ],
)
def test_foreign_mature_receipt_identity_rejected(field):
    attempt, sources, reserved, receipt = receipt_binding_fixture()
    receipt[field] = "foreign"
    with pytest.raises(ValueError):
        runner.check_receipt_identity("kev", attempt, sources, reserved, receipt)


def test_foreign_success_cannot_reach_provider_or_ledger_writer(tmp_path, monkeypatch):
    attempt, sources, reserved, _ = receipt_binding_fixture()
    (tmp_path / "reservation.json").write_text(json.dumps({"attempt": reserved}))
    (tmp_path / "research-receipt.json").write_text(
        json.dumps(
            {
                "schema": "omni-cpu-core-execution-v1",
                "attempt": "cpu-core-foreign",
                "success": True,
                "compute_stopped": True,
                "resources_verified": True,
            }
        )
    )
    monkeypatch.setattr(runner, "identity", lambda *args: None)
    monkeypatch.setattr(runner, "checked_sources", lambda: sources)

    def forbidden(*args, **kwargs):
        raise AssertionError("foreign receipt必須先拒絕，不可進provider／writer")

    monkeypatch.setattr(runner, "api", forbidden)
    monkeypatch.setattr(runner, "GitHubLedger", forbidden)
    monkeypatch.setattr(runner, "bound", forbidden)
    with pytest.raises(ValueError):
        runner.finish_run("kev", attempt, tmp_path)


def test_repair_receipt_requires_complete_original_lineage(preparation_recovery_fixture, monkeypatch):
    f = preparation_recovery_fixture
    monkeypatch.setattr(runner, "ROOT", recovery.ROOT)
    proof = recovery.eligibility(f.state, "kev")
    attempt, sources, reserved, receipt = receipt_binding_fixture()
    original = json.loads(reserved["purpose"])
    reserved["purpose"] = json.dumps({**original, **proof})
    receipt["preparation_recovery"] = copy.deepcopy(proof)
    runner.check_receipt_identity("kev", attempt, sources, reserved, receipt)
    receipt["preparation_recovery"]["original_input_payload_sha256"] = "changed"
    with pytest.raises(ValueError):
        runner.check_receipt_identity("kev", attempt, sources, reserved, receipt)
    del proof["original_row_sha256"]
    reserved["purpose"] = json.dumps({**original, **proof})
    with pytest.raises(ValueError):
        runner.check_receipt_identity("kev", attempt, sources, reserved, receipt)


@pytest.mark.parametrize("right_target,expected", [(True, "completed"), (False, "unresolved")])
def test_finish_requires_the_current_owned_provider_target(tmp_path, monkeypatch, right_target, expected):
    attempt, sources, reserved, receipt = receipt_binding_fixture()
    assets = [
        {"id": index, "name": name, "bytes": 1, "sha256": str(index) * 64, "roundtrip_verified": True}
        for index, name in enumerate(["input.zip", "result.zip"], 1)
    ]
    receipt.update(
        success=True,
        compute_stopped=True,
        resources_verified=True,
        cleanup_checks={"fixture_stop": True},
        release_id=1,
        assets=assets,
        result_exported_before_stop=True,
        research_final_opened=True,
    )
    (tmp_path / "reservation.json").write_text(json.dumps({"attempt": reserved}))
    (tmp_path / "research-receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(runner, "identity", lambda *args: None)
    monkeypatch.setattr(runner, "checked_sources", lambda: sources)
    monkeypatch.setattr(runner, "bound", lambda *args: {"total_bytes": 1, "control_bytes": 1, "retention_days": 7})
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("OMNI_ARTIFACT_OUTCOME", "success")

    def get(endpoint):
        if endpoint.startswith("actions/"):
            return {
                "total_count": 1,
                "artifacts": [
                    {
                        "id": 1,
                        "name": "omni-" + attempt,
                        "size_in_bytes": 1,
                        "created_at": "2000-01-01T00:00:00Z",
                        "expires_at": "2000-01-08T00:00:00Z",
                    }
                ],
            }
        assert endpoint == "releases/1"
        return {
            "draft": True,
            "tag_name": receipt["release_tag"],
            "body": receipt["release_body"],
            "target_commitish": sources["revision"] if right_target else "foreign",
            "assets": [{"id": x["id"], "name": x["name"], "size": x["bytes"]} for x in assets],
        }

    writes = []

    class Store:
        def read(self):
            return {"attempts": [copy.deepcopy(reserved)]}, "fixture-blob"

        def write(self, state, blob):
            assert blob == "fixture-blob"
            writes.append(state)
            return {"fixture": "no real ledger write"}

    monkeypatch.setattr(runner, "api", get)
    monkeypatch.setattr(runner, "GitHubLedger", Store)
    runner.finish_run("kev", attempt, tmp_path)
    assert len(writes) == 1 and writes[0]["attempts"][0]["status"] == expected
