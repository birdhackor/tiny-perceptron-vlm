"""核對凍結修訂、沿用證據與 UTF-8／UTF-16 邊界，僅新增執行收據。"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
ARTIFACTS = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_6-1/technical-6-1-original-01.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.parent != ARTIFACTS:
        raise ValueError("執行收據必須放在指定 artifact 目錄")
    if target.exists():
        raise FileExistsError(target)

    original = json.loads(ORIGINAL.read_bytes())
    if digest(ORIGINAL) != "c4f86339e2d03f2cfc421b51b3386a10eeef921874c4f6446d5c5b28968cdfba":
        raise ValueError("原始報告雜湊不符")
    previous_checks = []
    for item in original["sources"] + original["artifacts"]:
        if "path" in item and "sha256" in item:
            current = digest(ROOT / item["path"])
            previous_checks.append({"id": item["id"], "path": item["path"], "expected_sha256": item["sha256"],
                                    "current_sha256": current, "unchanged": current == item["sha256"]})
            if current != item["sha256"]:
                raise ValueError(f"沿用證據已變動：{item['path']}")
    old_section = ROOT / "docs/course-review-20261019/freeze-01/sources/6.1.md"
    new_section = ROOT / "docs/course-review-20261019/freeze-05/sources/6.1.md"
    freeze = ROOT / "docs/course-review-20261019/freeze-05.json"
    assert digest(new_section) == "16ab07b1aa49eb915dc96f11123485919188400ee7e138719a5474f59d22c5f3"
    assert digest(freeze) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
    diff_command = ["git", "diff", "--no-index", str(old_section.relative_to(ROOT)), str(new_section.relative_to(ROOT))]
    diff_run = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
    assert diff_run.returncode == 1 and not diff_run.stderr

    module_spec = importlib.util.spec_from_file_location("review_audit", ARTIFACTS / "audit.py")
    audit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(audit)
    section_run = audit.bundle(args.run_directory)
    extraction = json.loads(section_run["copied_files"]["extraction.json"]["content"])
    execution = json.loads(section_run["copied_files"]["execution.json"]["content"])
    environment = json.loads(section_run["copied_files"]["environment.json"]["content"])
    assert extraction["source_sha256"] == digest(new_section)
    assert extraction["figure_sha256"] == {} and extraction["svg_references"] == []
    assert execution["exit_code"] == 0
    assert section_run["copied_files"]["stderr.txt"]["content"] == ""
    expected_stdout = "小小貓 Unicode碼點 3 UTF-8 bytes 9\ncat Unicode碼點 3 UTF-8 bytes 3\n🙂 Unicode碼點 1 UTF-8 bytes 4\n"
    assert section_run["copied_files"]["stdout.txt"]["content"] == expected_stdout

    url = "https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/"
    curl_command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    response = subprocess.run(curl_command, capture_output=True, check=True)
    text = unescape(re.sub(r"<[^>]+>", " ", response.stdout.decode("utf-8")))
    text = re.sub(r"\s+", " ", text)
    locators = ["D71 High-surrogate code point:", "D73 Low-surrogate code point:",
                "D75 Surrogate pair:", "D76 Unicode scalar value:", "D91 UTF-16 encoding form:",
                "D92 UTF-8 encoding form:", "Table 3-6", "Table 3-7"]
    assert all(locator in text for locator in locators)
    excerpts = ["High-surrogate and low-surrogate code points are designated only for that use.",
                "Surrogate pairs are used only in UTF-16."]
    assert all(excerpt in text for excerpt in excerpts)

    successful = Counter()
    rejected = 0
    for value in range(0x110000):
        try:
            encoded = chr(value).encode("utf-8")
        except UnicodeEncodeError:
            assert 0xD800 <= value <= 0xDFFF
            rejected += 1
        else:
            assert not 0xD800 <= value <= 0xDFFF
            assert encoded.decode("utf-8") == chr(value)
            assert 1 <= len(encoded) <= 4
            successful[len(encoded)] += 1
    assert successful == {1: 128, 2: 1920, 3: 61440, 4: 1048576}
    assert rejected == 2048
    surrogate_probes = []
    for label, sample in [("high", chr(0xD800)), ("low", chr(0xDFFF)),
                          ("two_surrogate_codepoints", chr(0xD83D) + chr(0xDE42))]:
        try:
            sample.encode("utf-8")
        except UnicodeEncodeError as error:
            surrogate_probes.append({"label": label, "codepoints": [f"U+{ord(c):04X}" for c in sample],
                                     "exception": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError("surrogate 碼點不應直接編成合法 UTF-8")
    utf16_raw = bytes.fromhex("3d d8 42 de")
    scalar = utf16_raw.decode("utf-16-le")
    assert scalar == "🙂" and len(scalar) == 1

    old_facts = json.loads((ARTIFACTS / "facts.json").read_bytes())
    result_input = old_facts["recorded_result"]
    result_hash = digest(ROOT / result_input["path"])
    assert result_hash == result_input["sha256"]
    archive_input = result_input["validation_reconstruction"]
    archive_pointer_hash = digest(ROOT / archive_input["archive"])
    assert archive_pointer_hash == archive_input["retrieval"]["local_sha256"]
    from scripts.course_experiments.run import experiment_spec
    specification = experiment_spec("real_text")
    assert specification == old_facts["recipe_interface"]["selected_specification"]
    training = ROOT / "course/training.md"
    training_command = ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda"
    assert training_command in training.read_text(encoding="utf-8")
    help_command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]
    help_run = subprocess.run(help_command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                              capture_output=True, text=True, check=True)

    record = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-1/recheck-current-01.py --run-directory "
                   + str(args.run_directory) + " --output " + str(args.output),
        "environment": {"python": sys.version, "torch": str(environment["torch"]), "device": "cpu", "platform": sys.platform},
        "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(ORIGINAL)},
        "previous_evidence_hash_checks": previous_checks,
        "freeze": {"path": str(freeze.relative_to(ROOT)), "sha256": digest(freeze)},
        "section": {"path": str(new_section.relative_to(ROOT)), "sha256": digest(new_section)},
        "diff": {"command_argv": diff_command, "exit_code": diff_run.returncode, "stdout": diff_run.stdout,
                 "stderr": diff_run.stderr, "interpretation": "只有 UTF-8 定義新增 surrogate 範圍及 UTF-16 用途的括號說明；其他文字未變。"},
        "section_execution": section_run,
        "unicode_original": {"url": url, "version": "Unicode 17.0.0", "checked_original": True,
                             "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": curl_command,
                             "exit_code": response.returncode, "response_bytes": len(response.stdout),
                             "original_response_sha256": hashlib.sha256(response.stdout).hexdigest(),
                             "locators_read": locators, "excerpts": excerpts,
                             "inspection_note": "surrogate 碼點的 high 範圍為 D800..DBFF、low 範圍為 DC00..DFFF；其用途限定於 UTF-16。UTF-8 對每個 scalar value 使用 1–4 bytes，scalar value 排除整個 surrogate 範圍。UTF-16 中有效 surrogate pair 代表另一個補充平面 scalar；不是把兩個 surrogate 碼點直接編成 UTF-8。"},
        "exhaustive_utf8_probe": {"codespace_start": "U+0000", "codespace_end": "U+10FFFF", "tested_codepoints": 0x110000,
                                  "accepted_scalar_values": sum(successful.values()), "length_counts": dict(successful),
                                  "rejected_surrogate_codepoints": rejected, "rejected_range": "U+D800..U+DFFF",
                                  "all_accepted_roundtrip": True, "all_accepted_lengths_in_1_to_4": True,
                                  "scope": "Python 預設 strict UTF-8；枚舉全部 Unicode codespace 值，不使用 surrogatepass。"},
        "surrogate_probes": surrogate_probes,
        "utf16_pair_conversion": {"utf16_le_hex": utf16_raw.hex(" "), "decoded_text": scalar,
                                  "decoded_codepoint": f"U+{ord(scalar):04X}", "decoded_codepoint_count": len(scalar),
                                  "utf8_hex": scalar.encode("utf-8").hex(" "), "utf8_bytes": len(scalar.encode("utf-8"))},
        "unchanged_experiment_inputs": {"result_path": result_input["path"], "result_sha256": result_hash,
                                        "archive_pointer_path": archive_input["archive"], "archive_pointer_sha256": archive_pointer_hash,
                                        "reuse_scope": "沿用先前已核對的實驗樣本、資料包下載與 validation 重建；原始物件 SHA-256、紀錄與程式碼未變。"},
        "recipe": {"specification": specification, "unchanged": True,
                   "training_path": str(training.relative_to(ROOT)), "training_sha256": digest(training),
                   "command_present": training_command, "help_command_argv": help_command,
                   "help_exit_code": help_run.returncode, "help_stdout": help_run.stdout, "help_stderr": help_run.stderr},
        "result": "新版排除條件與 Unicode 定義一致；所有 scalar value 編碼長度為 1–4 bytes，全部 surrogate 碼點皆被拒絕。原始小節程式成功，沿用證據及其現行輸入雜湊全部相符。",
    }
    with target.open("x", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": digest(target),
                      "result": record["result"], "scalar_values": sum(successful.values()), "surrogates_rejected": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
