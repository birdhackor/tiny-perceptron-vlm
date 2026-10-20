"""核對單音算例、時間軸、批次補零與 Python 捨入行為。"""

import ast
import hashlib
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.utils.rnn import pad_sequence

from tiny_perceptron.multimodal import tone


def main():
    torch.set_default_device("cpu")
    wave = tone(440.0)
    longer = tone(440.0, seconds=0.2)
    expected = torch.tensor(
        [0.5 * math.sin(2 * math.pi * 440 * n / 16000) for n in range(1600)],
        dtype=torch.float64,
    )
    observed_error = (wave.double() - expected).abs().max().item()
    assert tuple(wave.shape) == (1600,)
    assert tuple(longer.shape) == (3200,)
    assert wave.dtype == torch.float32
    assert [round(v, 3) for v in wave[:3].tolist()] == [0.0, 0.086, 0.169]
    assert torch.equal(longer[:1600], wave)
    assert observed_error < 2e-5
    assert [round(wave.min().item(), 2), round(wave.max().item(), 2)] == [-0.5, 0.5]

    times = torch.arange(1600) / 16000
    assert times[0].item() == 0.0
    assert math.isclose(times[-1].item(), 1599 / 16000, abs_tol=1e-8)
    phase = [0.0, 0.25, 0.5, 0.75, 1.0]
    quarter_values = [0.5 * math.sin(2 * math.pi * p) for p in phase]
    assert all(abs(a - b) < 1e-15 for a, b in zip(quarter_values, [0.0, 0.5, 0.0, -0.5, 0.0]))

    batch = torch.stack([wave, wave])
    short = wave[:800]
    padded = pad_sequence([wave, short], batch_first=True)
    lengths = torch.tensor([len(wave), len(short)])
    valid = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    assert tuple(batch.shape) == (2, 1600)
    assert tuple(padded.shape) == (2, 1600)
    assert torch.equal(padded[1, :800], short)
    assert torch.count_nonzero(padded[1, 800:]).item() == 0
    assert valid.sum(dim=1).tolist() == [1600, 800]

    doubled = 2 * wave
    higher = tone(880.0)
    def positive_crossings(values):
        return int(((values[:-1] <= 0) & (values[1:] > 0)).sum().item())
    assert positive_crossings(wave) == 44
    assert positive_crossings(doubled) == 44
    assert positive_crossings(higher) == 88
    assert wave.max().item() == higher.max().item()
    assert not torch.equal(wave, higher)

    implementation = ROOT / "tiny_perceptron/multimodal.py"
    tree = ast.parse(implementation.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tone")
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/12.1.md").read_text()
    implementation_text = ast.get_source_segment(implementation.read_text(), function)
    assert implementation_text in frozen
    assert function.end_lineno - function.lineno + 1 == 3

    figure = ROOT / "course/figures/multimodal_wave_cycle.svg"
    svg = ET.fromstring(figure.read_bytes())
    circles = svg.findall(".//{http://www.w3.org/2000/svg}circle")
    positions = [(float(c.attrib["cx"]), float(c.attrib["cy"])) for c in circles]
    mapped = [((x - 80) / 460, (210 - y) / 170) for x, y in positions]
    assert mapped == list(zip(phase, [0.0, 0.5, 0.0, -0.5, 0.0]))

    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-1/verify_tone.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": str(wave.device), "platform": platform.platform()},
        "tone": {"shape": list(wave.shape), "dtype": str(wave.dtype), "first_three": wave[:3].tolist(), "first_three_rounded": [round(v, 3) for v in wave[:3].tolist()], "min": wave.min().item(), "max": wave.max().item(), "max_absolute_error_vs_double_formula": observed_error, "formula_tolerance": 2e-5},
        "time_axis": {"sample_count": len(times), "first_three_seconds": times[:3].tolist(), "last_seconds": times[-1].item(), "samples_per_cycle": 16000 / 440, "period_seconds": 1 / 440, "cycles_in_0_1_seconds": 440 * 0.1},
        "longer_tone": {"shape": list(longer.shape), "prefix_equal": torch.equal(longer[:1600], wave), "cycles_in_0_2_seconds": 440 * 0.2},
        "amplitude_and_frequency": {"positive_crossings_440": positive_crossings(wave), "positive_crossings_double_amplitude": positive_crossings(doubled), "positive_crossings_880": positive_crossings(higher), "peak_440": wave.max().item(), "peak_880": higher.max().item(), "waveforms_differ": not torch.equal(wave, higher)},
        "batch_and_padding": {"stack_shape": list(batch.shape), "padded_shape": list(padded.shape), "lengths": lengths.tolist(), "valid_counts": valid.sum(dim=1).tolist(), "padded_tail_nonzero_count": torch.count_nonzero(padded[1, 800:]).item()},
        "rounding": {"exactly_representable_input": 0.0625, "round_to_three_places": round(0.0625, 3), "decimal_half_up_result": "0.063", "reason": "0.0625 可用二進位精確表示；兩個相鄰的千分位等距時，Python 選尾數為偶數的 0.062。"},
        "figure_quarter_points": mapped,
        "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "implementation_excerpt_matches": True,
        "all_assertions_passed": True,
    }
    destination = Path(__file__).with_name("tone-verification.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
