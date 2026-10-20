"""複核現行來源、既有證據及變更句子的執行結果。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
ARTIFACT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-16/technical-chapter-16-original-01.json"
FREEZE = ROOT / "docs/course-review-20261019/freeze-05.json"
COMMAND = "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-16/verify-current-01.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def python_block(section, start):
    import re

    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    return next(block for block in blocks if start in block)


def execute(code, expect_assertion=False):
    namespace = {}
    output = io.StringIO()
    caught = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(code, "<教材程式>", "exec"), namespace)
        except AssertionError as error:
            if not expect_assertion:
                raise
            caught = type(error).__name__
    assert bool(caught) == expect_assertion, "練習的斷言結果與預期不符"
    return namespace, {"stdout": output.getvalue(), "expected_assertion_failure": expect_assertion, "exception": caught}


def verify():
    report = json.loads(ORIGINAL.read_bytes())
    assert sha(ORIGINAL) == "1503224b1d51bd450970407421ae58fb420d08c87f70957114ca12d2f17886f6"
    integrity = []
    for artifact in report["artifacts"]:
        current = sha(ROOT / artifact["path"])
        assert current == artifact["sha256"], "既有證據已改變"
        integrity.append({"path": artifact["path"], "recorded_sha256": artifact["sha256"], "current_sha256": current})
    unchanged_self_check = ARTIFACT / "self-check.txt"
    assert sha(unchanged_self_check) == "427dcaf0443fe2582f6d683c6d181a2736bf433e0e9e611062f197c2ac1d834c"
    sources = []
    for source in report["sources"]:
        if source["kind"] == "repository_code":
            current = sha(ROOT / source["path"])
            changed = current != source["sha256"]
            assert not changed or source["path"] == "course/chapters/16.md", "有額外來源變更，需另行核對"
            sources.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current, "changed": changed})
    endpoint_hashes = {}
    for revision in ("5ef56bc756f3", "3af7f13ef959"):
        raw = subprocess.check_output(["git", "show", revision + ":course/chapters/16.md"])
        endpoint_hashes[revision] = hashlib.sha256(raw).hexdigest()
    assert endpoint_hashes["5ef56bc756f3"] == "36df4c1312242eb13942e4b653a6762da66d37b275b7cbf2c0b2929efdf4f2c7"
    assert endpoint_hashes["3af7f13ef959"] == sha(ROOT / "course/chapters/16.md") == "5c53a5678eaecab7ec75a683807da1bf4aafc511fde465b334b8e134cda391e3"
    diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/16.md"]
    diff = subprocess.check_output(diff_command, text=True)
    base_path = ARTIFACT / "verify.py"
    spec = importlib.util.spec_from_file_location("original_probe", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.PRIMARY = []
    checks = base.verify()
    checks.pop("primary_sources")
    checks.pop("command")
    old_freeze = checks["checks"]["freeze"].pop("freeze_sha256")
    checks["checks"]["freeze"]["inherited_probe_freeze_sha256"] = old_freeze
    checks["checks"]["freeze"]["freeze_path"] = FREEZE.relative_to(ROOT).as_posix()
    checks["checks"]["freeze"]["freeze_sha256"] = sha(FREEZE)
    assert sha(FREEZE) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
    frozen_page = next(page for page in json.loads(FREEZE.read_bytes())["pages"] if page["page_id"] == "chapter-16")
    assert frozen_page["source_sha256"] == report["source_sha256"]
    assert frozen_page["figures_sha256"] == {}
    cache_text = base.section("course/chapters/16.md", "16.3")
    cache_code = python_block(cache_text, "model = TinyLM(ModelConfig(width=8))")
    _, plain = execute(cache_code)
    _, new_token = execute(cache_code.replace("[[1, 2, 3, 4]]", "[[1, 2, 3, 5]]"))
    stale_code = cache_code.replace('full = model(ids)["logits"][:, -1]', 'full = model(torch.tensor([[9, 2, 3, 4]]))["logits"][:, -1]')
    namespace, stale = execute(stale_code, expect_assertion=True)
    model = namespace["model"]
    assert model.config.layers == 1
    with torch.no_grad():
        old = model(torch.tensor([[1, 2, 3]]))["cache"]
        new = model(torch.tensor([[9, 2, 3]]))["cache"]
    differences = {}
    for label, old_tensor, new_tensor in zip(("K", "V"), old[0], new[0], strict=True):
        errors = (old_tensor - new_tensor).abs().amax(dim=(0, 1, 3)).tolist()
        assert errors[0] > 0 and errors[1:] == [0.0, 0.0], "單層 K/V 的位置差異與預期不符"
        differences[label] = errors
    cache_result = {
        "section_title": cache_text.splitlines()[0], "code_sha256": hashlib.sha256(cache_code.encode()).hexdigest(),
        "configuration": {"seed": 0, "width": 8, "layers": model.config.layers, "dtype": "torch.float32", "device": "cpu"},
        "unchanged_prefix": plain, "changed_last_token": new_token, "stale_prefix": stale,
        "stale_logit_max_abs_error": namespace["error"],
        "prefixes": [[1, 2, 3], [9, 2, 3]], "per_position_max_abs_difference": differences,
        "position_axis": 2,
    }
    precision_text = base.section("course/chapters/16.md", "16.7")
    precision_code = python_block(precision_text, 'torch.autocast("cpu", dtype=torch.bfloat16)')
    namespace, precision_output = execute(precision_code)
    full, low = namespace["full"], namespace["low"]
    error = (full - low.float()).abs().max().item()
    rounded = round(error, 4)
    assert rounded == 0.0075, "最大差異的小數輸出與預期不符"
    assert namespace["a"].dtype == namespace["b"].dtype == torch.float32
    assert low.dtype == torch.bfloat16
    assert low.float().tolist() == [[3.5, -1.0], [7.5, -1.0]]
    precision_result = {
        "section_title": precision_text.splitlines()[0], "code_sha256": hashlib.sha256(precision_code.encode()).hexdigest(),
        "execution": precision_output, "input_a": namespace["a"].tolist(), "input_b": namespace["b"].tolist(),
        "fp32_output": full.tolist(), "bf16_output_as_fp32": low.float().tolist(),
        "raw_max_abs_error": error, "round_ndigits": 4, "displayed_error": rounded,
        "round_tie_examples": {"round(0.5)": round(0.5), "round(1.5)": round(1.5)},
        "scope": "本機 CPU 程式結果；取到四位小數指 round 的 ndigits，不表示固定寬度格式，也不定義所有半值的進位方向。",
    }
    return {
        "command": COMMAND, "environment": checks["environment"],
        "original_report": {"path": ORIGINAL.relative_to(ROOT).as_posix(), "sha256": sha(ORIGINAL), "issues": report["issues"]},
        "original_artifact_integrity": integrity, "original_self_check_sha256": sha(unchanged_self_check),
        "repository_source_integrity": sources,
        "diff": {"command": " ".join(diff_command), "endpoint_sha256": endpoint_hashes, "stdout": diff},
        "reused_primary_source_receipts": {"path": "docs/technical-reviews/artifacts/review-20261019-chapter-16/verification.json", "sha256": sha(ARTIFACT / "verification.json"), "reason": "凍結導言未變、相同原審閱者複查；沿用已核對的固定證據，不重抓網頁。"},
        "base_probe": {"path": base_path.relative_to(ROOT).as_posix(), "sha256": sha(base_path), "entrypoint": "verify()", "network_fetches": "未執行；只在記憶體把 PRIMARY 設為空清單，原始檔不變。"},
        "current_probe": checks,
        "changed_sentence_checks": {"prefix_cache": cache_result, "decimal_output": precision_result},
        "result": "既有證據完整、導言未變；全部原有實作核驗重跑通過，兩處變更的三項補充主張有直接執行證據。",
    }


if __name__ == "__main__":
    result = verify()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    with (ARTIFACT / "verification-current-01.json").open("x") as stream:
        stream.write(output)
    sys.stdout.write(output)
