"""Independent hand-specified edit/normalization checks; no OCR model is run."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.natural_concepts import edit_distance, text_error_report

ARTIFACT = Path(__file__).resolve().parents[1]
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
fence = (ARTIFACT / "inputs/fence-1.py").read_bytes()
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(fence, "course/chapters/20.md#20.10:285", "exec"), {"__name__": "__main__"})
assert output.getvalue() == "完整相同 False\n最少編輯次數 1\n參考字數 5\nCER 0.2\n"

# Expected edit counts are hand-specified, rather than computed by a second DP.
cases = [
    ("substitute", "今天去臺北", "今天去台北", 1, 5, 0.2, False),
    ("delete", "今天去臺北", "今天去臺", 1, 5, 0.2, False),
    ("substitute_then_delete_North", "今天去臺北", "今天去台", 2, 5, 0.4, False),
    ("insert", "今天去臺北", "今天去臺北！", 1, 5, 0.2, False),
    ("unchanged", "今天去臺北", "今天去臺北", 0, 5, 0.0, True),
    ("whitespace", "A B", "AB", 1, 3, 1 / 3, False),
    ("punctuation", "A!", "A", 1, 2, 0.5, False),
    ("empty_both", "", "", 0, 0, None, True),
    ("empty_reference_false_text", "", "今天", 2, 0, None, False),
    ("missing_all", "今天", "", 2, 2, 1.0, False),
    ("more_insertions_than_reference", "a", "aaaa", 3, 1, 3.0, False),
    ("code_points_not_graphemes", "é", "e\u0301", 2, 1, 2.0, False),
    ("fullwidth_raw", "Ａ", "A", 1, 1, 1.0, False),
]
checked = []
for name, ref, pred, edits, length, cer, exact in cases:
    report = text_error_report(ref, pred)
    assert report == {
        "reference": ref, "prediction": pred, "exact": exact,
        "edits": edits, "reference_characters": length, "cer": cer,
    }, (name, report)
    assert edit_distance(ref, pred) == edits
    checked.append({"name": name, "expected_edits": edits, "expected_length": length,
                    "expected_cer": cer, "observed": report})

norm = {
    "fullwidth_A": unicodedata.normalize("NFKC", "Ａ"),
    "traditional_Tai": unicodedata.normalize("NFKC", "臺"),
    "variant_Tai": unicodedata.normalize("NFKC", "台"),
    "decomposed_e_acute": unicodedata.normalize("NFKC", "e\u0301"),
}
assert norm == {"fullwidth_A": "A", "traditional_Tai": "臺", "variant_Tai": "台", "decomposed_e_acute": "é"}
assert len("今天去臺北") == 5 and len("今天去臺北".encode("utf-8")) == 15
assert text_error_report(unicodedata.normalize("NFKC", "Ａ"), "A")["exact"] is True
assert text_error_report(unicodedata.normalize("NFKC", "臺"), "台")["exact"] is False

result = {
    "original_fence_stdout": output.getvalue(),
    "original_fence_sha256": hashlib.sha256(fence).hexdigest(),
    "cases": checked, "normalization": norm,
    "reference_code_points": 5, "reference_utf8_bytes": 15,
    "environment": {
        "python": sys.version, "python_executable": sys.executable,
        "torch": str(torch.__version__), "torch_cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()), "device": "cpu",
        "unicode_database": unicodedata.unidata_version,
    },
    "scope": "Literal original fence and 13 hand-specified scoring cases; no OCR model, model/data download, training, or model accuracy measurement.",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
