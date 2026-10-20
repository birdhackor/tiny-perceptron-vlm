"""核對凍結課文差異、捨入敘述與既有證據的有效性。"""

import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import tone
from torch.nn.utils.rnn import pad_sequence


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_report = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_12-1/technical-12-1-original-01.json"
    assert sha(original_report) == "15ccee9d4a09dbb6bc8c3bbccb70b26f688fb79b63bbfd35adae1cae0767e79d"
    original = json.loads(original_report.read_bytes())
    evidence_audit = []
    for item in original["artifacts"]:
        observed = sha(ROOT / item["path"])
        assert observed == item["sha256"]
        evidence_audit.append({"path": item["path"], "sha256": observed, "matches": True})
    code_audit = []
    for item in original["sources"]:
        if item["kind"] == "repository_code":
            observed = sha(ROOT / item["path"])
            assert observed == item["sha256"]
            code_audit.append({"path": item["path"], "sha256": observed, "matches": True})

    old_path = ROOT / "docs/course-review-20261019/freeze-01/sources/12.1.md"
    new_path = ROOT / "docs/course-review-20261019/freeze-05/sources/12.1.md"
    old, current = old_path.read_bytes(), new_path.read_bytes()
    old_phrase = "`round(v, 3)` 四捨五入到小數第三位"
    new_phrase = "`round(v, 3)` 取到小數第三位"
    assert old.count(old_phrase.encode()) == 1
    assert old.replace(old_phrase.encode(), new_phrase.encode(), 1) == current
    assert sha(new_path) == "24e08171c7eef01641d47abe459d50372fd67c4aa63982016bfea5b8681b0a0c"
    diff_command = ["git", "diff", "--no-index", str(old_path.relative_to(ROOT)), str(new_path.relative_to(ROOT))]
    diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, check=False)
    assert diff.returncode == 1 and not diff.stderr

    url = "https://docs.python.org/3.13/library/functions.html#round"
    curl_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "40", url]
    fetched = subprocess.run(curl_command, capture_output=True, check=False)
    assert fetched.returncode == 0
    parser = Text()
    parser.feed(fetched.stdout.decode())
    text = " ".join(" ".join(parser.parts).split())
    start = text.index("round ( number")
    end = text.index("For a general Python object", start)
    inspected = text[start:end]
    assert "Return number rounded to ndigits precision after the decimal point" in inspected
    assert "rounding is done toward the even choice" in inspected
    print(inspected)
    source_receipt = {
        "url": url, "version": "Python 3.13.16 documentation",
        "accessed_at": datetime.now(UTC).isoformat(), "command_argv": curl_command,
        "exit_code": fetched.returncode, "stderr": fetched.stderr.decode(),
        "response_sha256": hashlib.sha256(fetched.stdout).hexdigest(),
        "response_bytes": len(fetched.stdout), "locator": "Built-in Functions: round(number, ndigits=None)",
        "excerpts": ["Return number rounded to ndigits precision after the decimal point.", "rounding is done toward the even choice"],
    }

    torch.set_default_device("cpu")
    wave, longer = tone(440.0), tone(440.0, seconds=0.2)
    ref = torch.tensor([0.5 * math.sin(2 * math.pi * 440 * n / 16000) for n in range(1600)], dtype=torch.float64)
    error = (wave.double() - ref).abs().max().item()
    first = [round(v, 3) for v in wave[:3].tolist()]
    assert first == [0.0, 0.086, 0.169]
    assert wave.shape == (1600,) and longer.shape == (3200,)
    assert torch.equal(wave, longer[:1600])
    assert error < 2e-5
    assert [round(wave.min().item(), 2), round(wave.max().item(), 2)] == [-0.5, 0.5]
    assert round(0.0625, 3) == 0.062
    times = torch.arange(1600) / 16000
    assert math.isclose(times[-1].item(), 1599 / 16000, abs_tol=1e-8)
    batch = torch.stack([wave, wave])
    padded = pad_sequence([wave, wave[:800]], batch_first=True)
    lengths = torch.tensor([1600, 800])
    valid = torch.arange(1600)[None, :] < lengths[:, None]
    assert batch.shape == padded.shape == (2, 1600)
    assert not padded[1, 800:].any()
    assert valid.sum(dim=1).tolist() == [1600, 800]
    cross = lambda x: int(((x[:-1] <= 0) & (x[1:] > 0)).sum().item())
    higher = tone(880.0)
    assert [cross(wave), cross(2 * wave), cross(higher)] == [44, 44, 88]
    assert wave.max().item() == higher.max().item()
    implementation = ROOT / "tiny_perceptron/multimodal.py"
    tree = ast.parse(implementation.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tone")
    excerpt = ast.get_source_segment(implementation.read_text(), function)
    assert excerpt in current.decode() and function.end_lineno - function.lineno + 1 == 3
    figure = ROOT / "course/figures/multimodal_wave_cycle.svg"
    svg = ET.fromstring(figure.read_bytes())
    circles = svg.findall(".//{http://www.w3.org/2000/svg}circle")
    mapped = [((float(c.attrib["cx"]) - 80) / 460, (210 - float(c.attrib["cy"])) / 170) for c in circles]
    assert mapped == [(0, 0), (.25, .5), (.5, 0), (.75, -.5), (1, 0)]
    png = ROOT / "outputs/course-review-20261019/figure-renders-04/cbeb5cbe04f4018114e3d9004ee9202e6429928b077d614dd5427bc6bce644f4.png"
    assert sha(png) == "2e8550597a0abe6d8c5f623edbdcdd3cecd9a5262da1c287c1b642c4bcce7cce"
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-1/recheck-current-01.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": str(wave.device), "platform": platform.platform()},
        "original_report": {"path": str(original_report.relative_to(ROOT)), "sha256": sha(original_report)},
        "old_artifacts": evidence_audit, "repository_sources": code_audit,
        "source_change": {"before": old_phrase, "after": new_phrase, "only_change": True, "source_sha256": sha(new_path), "diff_command_argv": diff_command, "diff_exit_code": diff.returncode, "diff_stdout": diff.stdout.decode(), "diff_stderr": diff.stderr.decode()},
        "round_source": source_receipt,
        "tone": {"shape": list(wave.shape), "dtype": str(wave.dtype), "first_three": wave[:3].tolist(), "first_three_rounded": first, "min": wave.min().item(), "max": wave.max().item(), "max_absolute_error_vs_double_formula": error, "tolerance": 2e-5},
        "duration": {"longer_shape": list(longer.shape), "prefix_equal": torch.equal(wave, longer[:1600]), "frequency_hz": 440, "samples_per_cycle": 16000 / 440},
        "time_axis": {"first_three_seconds": times[:3].tolist(), "last_seconds": times[-1].item()},
        "batch": {"shape": list(batch.shape), "padded_shape": list(padded.shape), "valid_counts": valid.sum(dim=1).tolist(), "padded_tail_nonzero": int(torch.count_nonzero(padded[1, 800:]).item())},
        "crossings": {"original": cross(wave), "double_amplitude": cross(2 * wave), "double_frequency": cross(higher)},
        "rounding": {"input": 0.0625, "digits": 3, "observed": round(0.0625, 3), "reason": "取到小數第三位是精度說明，未指定遇五進位；正好一半時取偶數仍相容。"},
        "figure": {"svg_sha256": sha(figure), "mapped_quarter_points": mapped, "viewed_png": str(png.relative_to(ROOT)), "png_sha256": sha(png)},
        "all_assertions_passed": True,
    }
    destination = Path(__file__).with_name("recheck-current-01.json")
    with destination.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
