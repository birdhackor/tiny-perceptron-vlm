"""獨立 technical 16.16：只執行人工 CPU 範例，不載入模型或量測時延。"""

import contextlib
import hashlib
import io
import itertools
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    freeze = json.loads(FREEZE.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "16.16")
    snapshot = ROOT / page["snapshot"]
    body = snapshot.read_text()
    current_chapter = (ROOT / page["source"]).read_text()
    current = re.search(r"^## 16\.16 .*?(?=^## |\Z)", current_chapter, re.M | re.S)[0]
    assert current.encode() == snapshot.read_bytes()
    assert digest(snapshot) == page["source_sha256"]
    for path, expected in page["figures_sha256"].items():
        assert digest(ROOT / path) == expected
    assert digest(ROOT / page["notebook"]) == page["notebook_sha256"]

    code_blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(code_blocks) == 1
    code = code_blocks[0].rstrip("\n")
    notebook = json.loads((ROOT / page["notebook"]).read_bytes())
    nb_code = "".join(notebook["cells"][5]["source"]).rstrip("\n")
    assert nb_code == code
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, f"{page['snapshot']}::python-block-0", "exec"), namespace)
    expected_output = re.search(r"```text\n(.*?)```", body, re.S)[1]
    assert output.getvalue() == expected_output

    Decoder = namespace["Decoder"]
    table = namespace["table"]
    kernel = namespace["kernel"]
    F = namespace["F"]
    ids = namespace["ids"]
    baseline = namespace["streamed"]

    # 由官方運算獨立核對 overlap-add：一般 N 碼長度為 2N+2，本例裁為 2N。
    raw_reference = F.conv_transpose1d(table[ids][None, None], kernel[None, None], stride=2).flatten()
    assert raw_reference.numel() == 10
    assert raw_reference[8:].tolist() == [0.125, 0.0625]
    assert namespace["decoder"].tail.tolist() == [0.0, 0.0]

    # 練習：四次各送一碼；只在第二碼之前重建一次 decoder。
    per_code = Decoder()
    one_code_chunks = [per_code.push(ids[i : i + 1]) for i in range(4)]
    assert torch.equal(torch.cat(one_code_chunks), baseline)
    reset = Decoder()
    reset_chunks = [reset.push(ids[:1])]
    reset = Decoder()
    reset_chunks += [reset.push(ids[i : i + 1]) for i in range(1, 4)]
    reset_output = torch.cat(reset_chunks)
    reset_mismatches = (baseline != reset_output).nonzero().flatten().tolist()
    assert reset_mismatches == [2, 3]
    assert (baseline - reset_output)[2:4].tolist() == [0.125, 0.0625]

    # 對有限教學碼表的全部四碼序列及八種非空分塊核對，仍不是訓練／效果評測。
    partition_count = 0
    prefix_count = 0
    maximum_error = 0.0
    for codes in itertools.product(range(4), repeat=4):
        sequence = torch.tensor(codes)
        reference = F.conv_transpose1d(table[sequence][None, None], kernel[None, None], stride=2).flatten()[:8]
        for cuts in itertools.product((False, True), repeat=3):
            bounds = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [4]
            d = Decoder()
            streamed = torch.cat([d.push(sequence[a:b]) for a, b in zip(bounds, bounds[1:])])
            error = (reference - streamed).abs().max().item()
            maximum_error = max(error, maximum_error)
            assert torch.equal(reference, streamed)
            partition_count += 1
        emitted = Decoder().push(sequence[:2])
        assert torch.equal(emitted, reference[:4])
        prefix_count += 1

    # 全段統計量可洩漏未來：原 prefix 相同，改未來後全段去平均的 prefix 不同。
    changed = namespace["future_changed"]
    assert torch.equal(baseline[:4], changed[:4])
    centered = baseline - baseline.mean()
    changed_centered = changed - changed.mean()
    normalization_prefix_delta = (centered[:4] - changed_centered[:4]).abs().max().item()
    assert normalization_prefix_delta > 0

    # 人工時間／取樣率算例只核單位，不量測實際串流。
    hypothetical_seconds = {"samples": 4, "sample_rate": 24000, "duration_seconds": 4 / 24000}
    hypothetical_rtf = {"compute_seconds": 0.2, "audio_seconds": 1.0, "course_rtf": 0.2, "encodec_rtf": 5.0}
    assert hypothetical_rtf["course_rtf"] * hypothetical_rtf["encodec_rtf"] == 1.0

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_16_16",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "CPU", "threads": str(torch.get_num_threads())},
        "source_binding": {"freeze": str(FREEZE.relative_to(ROOT)), "freeze_sha256": digest(FREEZE), "snapshot": page["snapshot"], "source_sha256": digest(snapshot), "current_section_matches": True, "figure_sha256": page["figures_sha256"], "notebook": page["notebook"], "notebook_sha256": digest(ROOT / page["notebook"]), "inline_python_sha256": hashlib.sha256(code.encode()).hexdigest(), "inline_equals_notebook_cell5": True},
        "exact_inline_stdout": output.getvalue(),
        "exact_inline_output_matches_frozen_text": True,
        "inline_assertions_passed": True,
        "raw_full_length": raw_reference.numel(),
        "raw_full_last_two_discarded": raw_reference[8:].tolist(),
        "finish_tail": namespace["decoder"].tail.tolist(),
        "one_code_chunks": [v.tolist() for v in one_code_chunks],
        "one_code_equals_two_code_chunks": True,
        "reset_before_second_code_output": reset_output.tolist(),
        "reset_before_second_code_mismatch_indices": reset_mismatches,
        "reset_before_second_code_lost_tail": (baseline - reset_output)[2:4].tolist(),
        "exhaustive_4code_8partitions": {"sequences": 256, "partition_checks": partition_count, "prefix_checks": prefix_count, "maximum_absolute_error": maximum_error},
        "whole_waveform_mean_normalization_example": {"original_mean": baseline.mean().item(), "changed_future_mean": changed.mean().item(), "normalized_old_prefix_max_delta": normalization_prefix_delta},
        "hypothetical_duration_units_only": hypothetical_seconds,
        "hypothetical_rtf_reciprocal_units_only": hypothetical_rtf,
        "scope": "人工 CPU tensor 機制與固定碼表運算；未載入或下載任何權重，未訓練，未重跑 final family inference，未接播放器，未量測 TTFA／正式時延。",
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
