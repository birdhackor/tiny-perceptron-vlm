"""Reviewer-owned, bounded CPU checks for canonical 12.13; no model inference."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(path, section_id):
    raw = (ROOT / path).read_text()
    start = re.search(r"^## " + re.escape(section_id) + r" ", raw, re.M).start()
    match = re.search(r"^## ", raw[start + 1 :], re.M)
    return raw[start : start + 1 + match.start()] if match else raw[start:]


def main():
    torch.set_num_threads(1)
    freeze = json.loads((ROOT / FREEZE).read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "12.13")
    body = (ROOT / page["snapshot"]).read_text()
    assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
    assert section(page["source"], "12.13") == body
    code_blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(code_blocks) == 1
    nb = json.loads((ROOT / page["notebook"]).read_bytes())
    assert sha(page["notebook"]) == page["notebook_sha256"]
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 3
    assert code_cells[2] == code_blocks[0].rstrip("\n")
    env = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        # Local bootstrap, figure display, and exact frozen inline example.
        # google.colab is absent here, so bootstrap performs no clone or install.
        exec(compile(code_cells[0], "12.13-notebook-local-setup", "exec"), env)
        assert not env["in_colab"]
        exec(compile(code_cells[1], "12.13-notebook-figure-display", "exec"), env)
        exec(compile(code_blocks[0], "12.13-frozen-inline", "exec"), env)
    expected_lines = ["時間框與頻帶 21 16", "先後序列相同 False", "平均摘要近似相同 True"]
    assert output.getvalue().strip().splitlines()[-3:] == expected_lines
    seq, reverse = env["sequence"], env["reordered"]
    mean_a, mean_b = seq.mean(0), reverse.mean(0)
    difference = (mean_a - mean_b).abs()
    bounds = 1e-8 + 1e-5 * mean_b.abs()
    assert bool((difference <= bounds).all())
    assert torch.equal(torch.sort(seq, dim=0).values, torch.sort(reverse, dim=0).values)
    assert torch.equal(seq[0], reverse[-1])
    assert torch.equal(seq[-1], reverse[0])
    assert bool(torch.isfinite(seq).all())
    assert bool((env["bank"] >= 0).all())

    from tiny_perceptron.multimodal import log_mel, mel_filter_bank

    bank_ref = mel_filter_bank(16000, 400, 16)
    seq_ref = log_mel(env["wave"], 16000, 400, 160, 16).T
    assert torch.equal(env["bank"], bank_ref)
    assert torch.equal(seq, seq_ref)

    # Inspect product A's implementation without instantiating/loading/running a NN.
    model_path = "tiny_perceptron/selftrained/model.py"
    tree = ast.parse((ROOT / model_path).read_text())
    audio = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AudioEncoder")
    calls = [n for n in ast.walk(audio) if isinstance(n, ast.Call)]
    pool = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "adaptive_avg_pool1d"]
    embeddings = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "Embedding"]
    assert len(pool) == len(embeddings) == 1
    assert ast.literal_eval(pool[0].args[1]) == 16
    assert ast.literal_eval(embeddings[0].args[0]) == 16
    # Pure pooling operation, deliberately outside the product's NN forward.
    artificial_sequence = torch.arange(64.0).reshape(1, 1, 64)
    pooled = F.adaptive_avg_pool1d(artificial_sequence, 16)
    expected_pool = torch.arange(16.0) * 4 + 1.5
    assert torch.equal(pooled[0, 0], expected_pool)

    # The target named by 12.13 literally cuts raw chunks before center=False STFT.
    target = section("course/chapters/12.md", "12.17")
    assert "raw = waveform[begin : begin + chunk_samples]" in target
    assert "center=False" in target
    assert "support_start = start_ms + begin / rate * 1000" in target
    assert "support_end = start_ms + (begin + len(raw)) / rate * 1000" in target

    # All variants below are transparent artificial values, not speech/model results.
    symbolic = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    symbolic_reverse = symbolic.flip(0)
    assert torch.equal(symbolic.mean(0), symbolic_reverse.mean(0))
    order_answers = ["低先", "高先"]
    assert order_answers[0] != order_answers[1]
    transcript_reference = "請更新地址謝謝"
    transcript_with_error = "請更新地址謝"
    classify = lambda x: "地址更新" if "更新地址" in x else "其他"
    assert transcript_reference != transcript_with_error
    assert classify(transcript_reference) == classify(transcript_with_error)

    result = {
        "reviewer_task": "/root/omni_p5_technical_12_13",
        "created_at": datetime.now(UTC).isoformat(),
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-12.13/audit_cpu_v1.py",
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "CPU", "platform": platform.platform(), "threads": str(torch.get_num_threads())},
        "freeze_sha256": sha(FREEZE),
        "source_sha256": page["source_sha256"],
        "notebook_sha256": sha(page["notebook"]),
        "inline_code_sha256": hashlib.sha256(code_blocks[0].encode()).hexdigest(),
        "current_section_matches_frozen": True,
        "notebook_inline_code_exact_match": True,
        "notebook_local_setup_figure_and_inline_executed": True,
        "stdout": output.getvalue(),
        "configuration": {"sample_rate": env["sample_rate"], "n_fft": env["n_fft"], "hop": env["hop"], "bands": env["bands"], "frequency_hz": [440, 880], "amplitude": 0.5, "clamp_min": 1e-8, "center": True, "pad_mode": "reflect", "dtype": str(env["wave"].dtype)},
        "shapes": {key: list(env[key].shape) for key in ["time", "wave", "mels", "hz", "frequency", "bank", "spectrum", "sequence", "reordered"]},
        "duration_seconds": {"each_tone": len(env["time"]) / env["sample_rate"], "concatenated": len(env["wave"]) / env["sample_rate"]},
        "mean_comparison": {"exact_equal": torch.equal(mean_a, mean_b), "allclose": torch.allclose(mean_a, mean_b), "max_abs_difference": float(difference.max()), "max_ratio_to_default_allclose_bound": float((difference / bounds).max()), "default_rtol": 1e-5, "default_atol": 1e-8, "finite": True},
        "per_band_multiset_unchanged": True,
        "flip_moves_endpoints_without_recomputing_stft": True,
        "inline_bank_equals_repository_bank": True,
        "inline_sequence_equals_repository_log_mel": True,
        "product_a_ast_inspection": {"path": model_path, "sha256": sha(model_path), "audio_encoder_lines": [audio.lineno, audio.end_lineno], "adaptive_average_output_intervals": 16, "position_embedding_slots": 16, "no_neural_forward_or_weights": True},
        "pure_pooling_artificial_output": pooled[0, 0].tolist(),
        "literal_12_17_crossreference_checks": True,
        "derivation_counterexamples": {"mean_of_both_orders": symbolic.mean(0).tolist(), "order_answers": order_answers, "transcripts_different": True, "answer_for_both_transcripts": classify(transcript_reference), "ordinal_positions_same_at_10ms_and_20ms_spacing": [0, 1]},
        "result": "All assertions passed; no training, formal final-family neural inference, network model fetch, calibration or policy changes.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
