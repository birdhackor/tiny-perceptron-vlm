"""5.8 技術核對：只執行手寫算例、CPU交叉熵與歷史JSON核算；不載入或訓練模型。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import loss_sum


ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
WORK = Path("/workspace/work/textbook-repair-20261017/technical-5-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(code):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "frozen-5.8-code", "exec"), {})
    return out.getvalue()


def main():
    torch.set_num_threads(1)
    source = ROOT / "course/chapters/05.md"
    raw = source.read_text()
    body = re.search(r"^## 5\.8 .*?(?=^## )", raw, re.M | re.S)[0]
    pack = Path("/workspace/work/textbook-repair-20261017/input-freeze02/pages/5.8.md")
    freeze = json.loads((ROOT / "docs/course-repair-20261017/freeze-02.json").read_text())
    entry = next(x for x in freeze["pages"] if x["page_id"] == "5.8")
    digest = hashlib.sha256(body.encode()).hexdigest()
    assert digest == entry["source_sha256"] == sha(pack)
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    original = capture(code)
    exercise = capture(code.replace("teacher_forced = [1, 2, 0, 4]", "teacher_forced = [1, 2, 3, 4]"))
    assert original == "正確前文下 逐字正確率 0.75 整串相同 False\n自由生成 逐字正確率 0.5 整串相同 False\n"
    assert exercise == "正確前文下 逐字正確率 1.0 整串相同 True\n自由生成 逐字正確率 0.5 整串相同 False\n"
    unequal = []
    for prediction in ([1, 2, 3], [1, 2, 3, 4, 5]):
        try:
            sum(a == b for a, b in zip([1, 2, 3, 4], prediction, strict=True))
        except ValueError as error:
            unequal.append({"prediction": prediction, "exception": str(error)})
        else:
            raise AssertionError("strict zip must detect length mismatch")
    sequence_checks = {
        "same": [1, 2, 3, 4] == [1, 2, 3, 4],
        "wrong_order": [1, 2, 3, 4] == [1, 2, 4, 3],
        "shorter": [1, 2, 3, 4] == [1, 2, 3],
        "longer": [1, 2, 3, 4] == [1, 2, 3, 4, 0],
        "sum_bool": sum([True, True, False, True]),
    }
    assert sequence_checks == {"same": True, "wrong_order": False, "shorter": False, "longer": False, "sum_bool": 3}
    probs = torch.tensor([[0.6, 0.4], [0.9, 0.1]], dtype=torch.float64)
    logits = probs.log()
    ce = F.cross_entropy(logits, torch.tensor([0, 0]), reduction="none")
    expected = torch.tensor([-math.log(0.6), -math.log(0.9)], dtype=torch.float64)
    assert torch.allclose(ce, expected, atol=1e-15, rtol=0)
    assert logits.argmax(-1).tolist() == [0, 0]
    masked_logits = torch.cat((logits, torch.zeros((1, 2), dtype=torch.float64))).unsqueeze(0)
    summed, count = loss_sum(masked_logits, torch.tensor([[0, 0, -100]]))
    assert int(count) == 2 and abs(float(summed) - float(ce.sum())) < 1e-15

    toy_path = ROOT / "docs/course-experiments/results/text_foundation.json"
    real_path = ROOT / "docs/course-experiments/results/real_text.json"
    toy_doc, real_doc = (json.loads(p.read_text()) for p in (toy_path, real_path))
    toy, real = toy_doc["results"], real_doc["results"]["runs"]["tinystories"]
    parts = split_records(generate_records("toy-text"), seed=42)
    split_info = {}
    for split, rows in parts.items():
        split_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        split_hash = hashlib.sha256(split_bytes).hexdigest()
        assert split_hash == toy["data"][split]["sha256"]
        split_info[split] = {"records": len(rows), "sha256": split_hash}
    validation_text = parts["validation"][0]["text"]
    sample = toy["after"]["validation"]["samples"][0]
    assert validation_text == sample["prompt"] + "side=left."
    assert sample["generated"] == "side=right."
    alternative_rows = [r["text"] for rows in parts.values() for r in rows if r["text"].startswith(sample["prompt"])]
    assert set(alternative_rows) == {sample["prompt"] + "side=left.", sample["prompt"] + "side=right."}
    label_count = sum(y.numel() for _, y in text_examples(parts["validation"], max_length=128))
    assert label_count == 35
    tok = ByteTokenizer()
    assert len(tok.encode("as a big was a she was a ber the")) == 32
    assert tok.decode(tok.encode("as a big was a she was a ber the")) == "as a big was a she was a ber the"
    assert tok.eos_id == 2
    arithmetic = []
    for name, result in (("text_foundation", toy), ("real_text/tinystories", real)):
        for phase in ("before", "after"):
            for split, v in result[phase].items():
                recomputed = v["nll_sum"] / v["effective_tokens"]
                assert abs(recomputed - v["nll"]) < 1e-12
                assert v["effective_tokens"] == v["raw_utf8_bytes"] + v["records"]
                assert len(v["samples"]) == min(8, v["records"])
                for s in v["samples"]:
                    assert tok.decode(s["generated_ids"]) == s["generated"]
                    assert len(s["generated_ids"]) <= 32
                arithmetic.append({"experiment": name, "phase": phase, "split": split,
                                   "nll_sum": v["nll_sum"], "effective_tokens": v["effective_tokens"],
                                   "nll": v["nll"], "five_decimals": f"{v['nll']:.5f}",
                                   "records": v["records"], "examples": v["examples"],
                                   "raw_utf8_bytes": v["raw_utf8_bytes"], "sample_count": len(v["samples"])})
    assert real["data"]["train"]["records"] == 409
    assert real["data"]["validation"]["records"] == 51
    assert real["after"]["validation"]["effective_tokens"] == 42453
    quote_locations = []
    for split, values in real["after"].items():
        for index, s in enumerate(values["samples"]):
            if s["prompt"] == "Once upon a time there w" and s["generated"] == "as a big was a she was a ber the":
                quote_locations.append(f"results.runs.tinystories.after.{split}.samples[{index}]")
    assert quote_locations
    figure_refs = re.findall(r"!\[[^\]]*\]\([^)]+\)|<img\b[^>]*>", body)
    assert figure_refs == []
    output = {
        "reviewer_task": "/root/repair_tech_5_8", "lesson_id": "5.8",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "actual CPU arithmetic/API checks and inspection of existing historical JSON; no model initialized, loaded, trained, or generated",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torch_git_version": torch.version.git_version, "device": "cpu", "torch_threads": "1"},
        "source_sha256": digest, "freeze02_and_input_pack_match": True,
        "original_code_stdout": original, "exercise_stdout": exercise,
        "strict_zip_errors": unequal, "sequence_checks": sequence_checks,
        "cross_entropy_check": {"dtype": "float64", "probabilities": probs.tolist(),
                                "argmax": logits.argmax(-1).tolist(), "nll": ce.tolist(),
                                "expected_negative_log": expected.tolist(), "absolute_tolerance": 1e-15,
                                "masked_nll_sum": float(summed), "effective_targets": int(count)},
        "toy_data_reconstruction": {"mode": "deterministic data only; not an experiment rerun", "seed": 42,
                                    "splits": split_info, "validation_text": validation_text,
                                    "validation_label_count": label_count,
                                    "same_prompt_legal_completions": alternative_rows,
                                    "historical_sample": sample},
        "historical_json_arithmetic": arithmetic,
        "historical_configuration": {
            "text_foundation": {k: toy_doc[k] for k in ("revision", "seed", "device", "gpu", "torch_version", "python_version", "step_scale", "evidence_status")},
            "tinystories": {k: real_doc[k] for k in ("revision", "seed", "device", "gpu", "torch_version", "python_version", "step_scale", "evidence_status")},
            "text_foundation_training": {k: toy["training"][k] for k in ("steps", "effective_tokens", "records", "parameters")},
            "tinystories_training": {k: real["training"][k] for k in ("steps", "effective_tokens", "records", "parameters")},
            "tinystories_data": real["data"], "tinystories_scope": real["scope"],
            "tinystories_split_warning": real["split_warning"],
            "evaluation_code_settings": "text_examples: byte IDs, BOS input, EOS target, nonoverlapping target chunks <=128 with context reset; _nll: sum CE / effective targets; generation first <=24 content IDs, max_new_tokens=32, default temperature=0.0 (argmax), EOS=2"
        },
        "historical_story_samples_after": {split: v["samples"] for split, v in real["after"].items()},
        "quote_locations": quote_locations,
        "raw_data_limit": "assets/training/tinystories-v1.tar.gz is a 131-byte Git LFS pointer in this checkout; TinyStories source rows/checkpoints were not downloaded. 42453 is verified as the original saved counter plus NLL arithmetic, not recomputed from the original 51 text files or model logits.",
        "figure_check": {"referenced_images": figure_refs, "missing_necessary_figure": False,
                         "reason": "same four positions shown directly in lists and prose; context distinction stated explicitly; no visual geometry or hidden object correspondence requires a diagram", "webpage_layout_verified": False},
        "input_file_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (source, toy_path, real_path,
            ROOT / "scripts/course_experiments/text.py", ROOT / "scripts/course_experiments/common.py",
            ROOT / "scripts/prepare_data.py", ROOT / "tiny_perceptron/data.py", ROOT / "tiny_perceptron/model.py")},
    }
    (ART / "verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": "all assertions passed", "output": str((ART / "verification.json").relative_to(ROOT)),
                      "historical_models_rerun": False, "source_sha256": digest,
                      "original_stdout": original, "exercise_stdout": exercise,
                      "nll_06_09": ce.tolist(), "historical_story_quote_locations": quote_locations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
