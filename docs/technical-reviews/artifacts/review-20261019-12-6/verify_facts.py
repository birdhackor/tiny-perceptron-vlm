"""核對凍結課文的數值、實作與原始來源；只保存精簡證據。"""

import ast
import hashlib
import html
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.multimodal import AudioEncoder, log_mel, mel_filter_bank, tone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ARTIFACT / name).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def plain(raw):
    text = raw.decode("utf-8")
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def source_receipt(item):
    identifier, url, version, locator, excerpts, note = item
    command = ["curl", "-fsSL", "--max-time", "25", url]
    fetched = subprocess.run(command, capture_output=True, check=True)
    raw = fetched.stdout
    text = plain(raw)
    if identifier == "pitch":
        record = json.loads(raw)["message"]
        text = plain(record["abstract"].encode())
    if identifier == "audio":
        text = raw.decode()
    for excerpt in excerpts:
        if excerpt not in text:
            raise ValueError(f"來源摘錄不存在：{identifier}")
    return {
        "id": identifier,
        "url": url,
        "version": version,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "command_argv": command,
        "exit_code": fetched.returncode,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "locator": locator,
        "excerpts": excerpts,
        "inspection_note": note,
        "retention": "原始回應僅在記憶體讀取；只保存支持主張的小段摘錄及取得收據。",
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    run = ROOT / "outputs/reviewer-tools/runs/20261020T062308942752Z"
    section_execution = {
        "invocation": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/12.md#12.6" --execute',
        "execution": json.loads((run / "execution.json").read_bytes()),
        "environment": json.loads((run / "environment.json").read_bytes()),
        "extraction": json.loads((run / "extraction.json").read_bytes()),
        "stdout": (run / "stdout.txt").read_text(),
        "stderr": (run / "stderr.txt").read_text(),
        "original_file_sha256": {name: sha(run / name) for name in ["execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt"]},
    }
    assert section_execution["execution"]["exit_code"] == 0
    save("section-execution.json", section_execution)

    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/12.6.md"
    snapshot = frozen.read_text()
    fences = re.findall(r"```python\n(.*?)\n```", snapshot, flags=re.S)
    frozen_function = ast.parse(fences[1]).body[0]
    actual_function = ast.parse(inspect.getsource(mel_filter_bank)).body[0]
    function_equal = ast.dump(frozen_function) == ast.dump(actual_function)
    assert function_equal
    freeze = ROOT / "docs/course-review-20261019/freeze-04.json"
    svg = ROOT / "course/figures/multimodal_overlapping_bands.svg"
    receipt_path = ROOT / "docs/course-review-20261019/figure-renders-04.json"
    figure_record = next(r for r in json.loads(receipt_path.read_bytes())["records"] if r["page_id"] == "12.6")
    assert sha(ROOT / figure_record["png"]["path"]) == figure_record["png"]["sha256"]
    assert sha(svg) == figure_record["frozen_svg"]["sha256"]

    details = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-6/verify_facts.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "platform": platform.platform(), "device": "cpu", "dtype": str(torch.get_default_dtype()), "seed": "0", "threads": "1"},
        "hashes": {str(path.relative_to(ROOT)): sha(path) for path in [frozen, freeze, svg, ROOT / "tiny_perceptron/multimodal.py", ROOT / "scripts/pretrain_encoders.py", receipt_path]},
        "frozen_function_ast_equals_repository": function_equal,
        "figure_receipt": figure_record,
        "toy": {"power": [4, 2, 1], "weights": [[1, .5, 0], [0, .5, 1]], "output": (torch.tensor([[1, .5, 0], [0, .5, 1]]) @ torch.tensor([4., 2., 1.])).tolist()},
        "hz_to_mel": {str(f): 2595 * math.log10(1 + f / 700) for f in [0, 700, 2100, 8000]},
        "banks": {},
    }
    assert details["toy"]["output"] == [5., 2.]
    for bands in [3, 16, 32]:
        bank = mel_filter_bank(bands=bands)
        max_mel = 2595 * math.log10(1 + 8000 / 700)
        points = [700 * (10 ** ((max_mel * i / (bands + 1)) / 2595) - 1) for i in range(bands + 2)]
        reference = torch.tensor([
            [max(0., min((f - points[i]) / (points[i + 1] - points[i]), (points[i + 2] - f) / (points[i + 2] - points[i + 1]))) for f in range(0, 8001, 40)]
            for i in range(bands)
        ], dtype=torch.float64)
        power = torch.stack([torch.ones(201), 2 * torch.ones(201), 3 * torch.ones(201)], dim=1)
        output = bank @ power
        maximum_error = float((reference - bank.double()).abs().max())
        counts = (bank > 0).sum(1).tolist()
        assert bank.shape == (bands, 201)
        assert output.shape == (bands, 3)
        assert bool((bank >= 0).all()) and bool(torch.isfinite(bank).all())
        assert float(output.min()) >= 0
        assert maximum_error < 1e-5
        assert torch.allclose(output[:, 1], output[:, 0] * 2, atol=1e-5)
        assert torch.allclose(output[:, 2], output[:, 0] * 3, atol=1e-5)
        details["banks"][str(bands)] = {
            "bank_shape": list(bank.shape), "output_shape": list(output.shape), "dtype": str(bank.dtype), "device": str(bank.device),
            "positive_bin_counts": counts, "nonnegative": bool((bank >= 0).all()), "finite": bool(torch.isfinite(bank).all()),
            "weight_min": float(bank.min()), "weight_max": float(bank.max()), "mel_points": torch.linspace(0, max_mel, bands + 2).tolist(),
            "hz_points_float64_reference": points, "hz_points_float32": (700 * (10 ** (torch.linspace(0, max_mel, bands + 2) / 2595) - 1)).tolist(),
            "float64_loop_reference_max_error": maximum_error, "reference_tolerance": 1e-5,
            "frame_scaled_input_preserved": True, "output_first_row": output[0].tolist(), "requires_grad": bank.requires_grad,
        }
    assert details["banks"]["16"]["positive_bin_counts"] == [6, 7, 8, 10, 11, 12, 14, 17, 20, 23, 27, 31, 35, 42, 49, 55]
    assert details["banks"]["32"]["positive_bin_counts"][0] == 2
    assert details["banks"]["32"]["positive_bin_counts"][-1] == 30
    frequencies = torch.linspace(0, 8000, 201)
    hz = 700 * (10 ** (torch.linspace(0, 2595 * math.log10(1 + 8000 / 700), 18) / 2595) - 1)
    left = (frequencies[None] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
    right = (hz[2:, None] - frequencies[None]) / (hz[2:] - hz[1:-1])[:, None]
    details["shapes"] = {"frequencies": list(frequencies.shape), "frequencies_newaxis": list(frequencies[None].shape), "left_knots": list(hz[:-2, None].shape), "center_knots": list(hz[1:-1, None].shape), "right_knots": list(hz[2:, None].shape), "left_slopes": list(left.shape), "right_slopes": list(right.shape), "hz_step": float(frequencies[1] - frequencies[0]), "hz_first_last": [float(frequencies[0]), float(frequencies[-1])]}
    errors = []
    for kwargs in [{"bands": 0}, {"bands": -1}, {"sample_rate": 0}, {"sample_rate": -1}]:
        try:
            mel_filter_bank(**kwargs)
        except ValueError as error:
            errors.append({"kwargs": kwargs, "type": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError("無效的帶數或取樣率未被拒絕")
    details["validation"] = errors
    bank = mel_filter_bank()
    x = torch.ones(201)
    y = x.clone()
    y[0] += 1
    assert not torch.equal(x, y)
    assert torch.equal(bank @ x, bank @ y)
    details["nonunique"] = {"rank_float64": int(torch.linalg.matrix_rank(bank.double())), "input_dimension": bank.shape[1], "output_dimension": bank.shape[0], "dc_column_nonzero_count": int((bank[:, 0] != 0).sum()), "two_nonnegative_inputs_differ_only_in_dc": True, "outputs_exactly_equal": True}
    waveform = tone()
    power = torch.stft(waveform, 400, 160, window=torch.hann_window(400), return_complex=True).abs().square()
    explicit = (mel_filter_bank(device=waveform.device).to(power.dtype) @ power).clamp(min=1e-8).log()
    actual = log_mel(waveform)
    assert torch.equal(actual, explicit)
    encoder = AudioEncoder()
    encoder(waveform.unsqueeze(0)).square().mean().backward()
    gradients = {name: {"requires_grad": parameter.requires_grad, "gradient_present": parameter.grad is not None, "gradient_finite": bool(torch.isfinite(parameter.grad).all())} for name, parameter in encoder.named_parameters()}
    assert all(item["requires_grad"] and item["gradient_present"] and item["gradient_finite"] for item in gradients.values())
    details["preprocessing"] = {"waveform_device": str(waveform.device), "power_device": str(power.device), "bank_device": str(bank.device), "power_dtype": str(power.dtype), "power_shape": list(power.shape), "log_mel_shape": list(actual.shape), "explicit_log_mel_exactly_equal": True, "filter_requires_grad": bank.requires_grad, "encoder_parameters": gradients, "optimizer_updates": 0, "scope": "只驗證固定配方、裝置相同與梯度連通；不測訓練品質。"}
    with torch.device("meta"):
        default_bank = mel_filter_bank()
    assert default_bank.device.type == "meta"
    assert mel_filter_bank().device.type == "cpu"
    details["device_contract"] = {"unspecified_device_with_meta_context": str(default_bank.device), "unspecified_device_without_context": str(mel_filter_bank().device), "gpu_executed": False}
    save("verification-results.json", details)

    specs = [
        ("mel", "https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix", "TensorFlow v2.16.1；頁面最後更新 2024-04-26", "linear_to_mel_weight_matrix 的 HTK 定義及矩陣相乘說明", ["This function follows the Hidden Markov Model Toolkit (HTK) convention"], "直接讀官方 API 正文，核對 2595、log10、700、線性頻率格、三角濾波器及逐框矩陣乘法。TensorFlow 矩陣方向與課文互為轉置；不把兩個實作說成逐位元組相同。"),
        ("audio", "https://raw.githubusercontent.com/pytorch/audio/v2.8.0/src/torchaudio/functional/functional.py", "torchaudio v2.8.0", "_hz_to_mel、_mel_to_hz、_create_triangular_filterbank、melscale_fbanks", ["return 2595.0 * math.log10(1.0 + (freq / 700.0))", "m_pts = torch.linspace(m_min, m_max, n_mels + 2)", "fb = torch.max(zero, torch.min(down_slopes, up_slopes))"], "直接讀官方原始碼，核對 HTK 正反換算、等距 mel 節點、Hz 軸線性斜坡及相鄰重疊。未安裝或執行 torchaudio。"),
        ("pitch", "https://api.crossref.org/works/10.1121/1.1915893", "Stevens、Volkmann、Newman，JASA 8(3)，185–190，1937；ASA 登錄的原摘要", "message.abstract：主觀音高尺度及原始受試條件", ["A subjective scale for the measurement of pitch was constructed from determinations of the half-value of pitches at various frequencies."], "透過 Crossref 讀 ASA 登錄的原摘要，非第三方摘要。摘要記載五位觀察者、十個頻率、60 dB 的主觀分割實驗；只支持 mel 的感知動機，沒有核對論文全文，也不以此證明任何音色或音量下都精確等距。"),
        ("linspace", "https://docs.pytorch.org/docs/2.14/generated/torch.linspace.html", "PyTorch 2.14 文件；本機 2.14.1", "torch.linspace：定義、steps、device", ["Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end , inclusive."], "直接讀 API 說明，核對包含端點、steps 與預設 tensor type／device 的規則。"),
        ("broadcast", "https://docs.pytorch.org/docs/2.14/notes/broadcasting.html", "PyTorch 2.14 文件；本機 2.14.1", "General semantics：尾軸對齊與結果形狀", ["the dimension sizes must either be equal, one of them is 1, or one of them does not exist."], "直接讀一般廣播規則與結果形狀規則；(1,201) 與 (16,1) 逐軸最大值得 (16,201)。"),
        ("matmul", "https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html", "PyTorch 2.14 文件；本機 2.14.1", "torch.matmul：兩個二維 tensor 的行為", ["If both arguments are 2-dimensional, the matrix-matrix product is returned."], "直接讀二維矩陣乘法契約；課文的 (16,201)@(201,3) 不會對三個時間框取平均。"),
        ("independence", "https://textbooks.math.gatech.edu/ila/linear-independence.html", "Interactive Linear Algebra，2026-10-20 線上版；回應 SHA 另記", "Linear Independence：以 pivot 解釋 wide matrix 的段落", ["A wide matrix (a matrix with more columns than rows) has linearly dependent columns."], "直接讀 Georgia Tech 官方教材的寬矩陣推導：每列最多一個 pivot，欄數較多就有自由變數。用來支持 16×201 線性映射不能單射；另以 DC 欄為零構造非負功率的反例。"),
    ]
    with ThreadPoolExecutor(max_workers=7) as pool:
        receipts = list(pool.map(source_receipt, specs))
    save("source-receipts.json", {"created_at": datetime.now(UTC).isoformat(), "sources": receipts})
    print(json.dumps({"status": "通過", "source_sha256": sha(frozen), "bank_counts": {key: value["positive_bin_counts"] for key, value in details["banks"].items()}, "original_sources_checked": len(receipts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
