"""Execute only the two artificial list examples assigned for review of 20.14."""

import contextlib
import datetime
import hashlib
import io
import json
import pathlib
import platform
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
FREEZE_SHA = "8df31b044f819c10ba0db5db9ae65f36bd39134c017a646e902b7376e78cbcaa"
SOURCE_SHA = "415e7f60a4a479fb305658e895360c6211af09d7e5c50ea239de85276c37682c"
OUTPUT = ROOT / "docs/technical-reviews/artifacts/omni-P5-20.14/inline-cpu-execution-v1.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run_case(name, code, expected_stdout, expected_separate, expected_shared, expected_prefix):
    namespace = {"__name__": "__inline_review__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, f"<20.14:{name}>", "exec"), namespace)
    actual = stdout.getvalue()
    assert actual == expected_stdout, (name, actual, expected_stdout)
    assert namespace["separate"] == expected_separate
    assert namespace["contaminated"] == expected_shared
    assert namespace["prefix"] == expected_prefix
    assert namespace["shared"]["kv"] is namespace["prefix"]["kv"]
    assert namespace["shared"]["recurrent"] is namespace["prefix"]["recurrent"]
    return {
        "name": name,
        "code_sha256": digest(code.encode("utf-8")),
        "stdout": actual,
        "expected_stdout": expected_stdout,
        "separate": namespace["separate"],
        "contaminated": namespace["contaminated"],
        "final_prefix": namespace["prefix"],
        "inner_lists_are_shared_in_shallow_copy": True,
        "all_assertions_passed": True,
    }


def main():
    started = datetime.datetime.now(datetime.UTC).isoformat()
    assert not OUTPUT.exists(), f"immutable output already exists: {OUTPUT}"
    freeze_bytes = (ROOT / FREEZE).read_bytes()
    assert digest(freeze_bytes) == FREEZE_SHA
    page = next(p for p in json.loads(freeze_bytes)["pages"] if p["page_id"] == "20.14")
    source_bytes = (ROOT / page["snapshot"]).read_bytes()
    assert digest(source_bytes) == SOURCE_SHA == page["source_sha256"]
    blocks = re.findall(r"^```python\n(.*?)^```", source_bytes.decode("utf-8"), re.M | re.S)
    assert len(blocks) == 1
    code = blocks[0]
    marker = '("資訊足夠", 5)'
    assert code.count(marker) == 1
    variant = code.replace(marker, '("資訊足夠", 1)', 1)
    cases = [
        run_case(
            "unmodified_frozen_inline",
            code,
            "獨立分支 [(3, 12), (3, 15)]\n原背景 2 10\n共用可變清單 [(3, 12), (4, 17)]\n被改過的背景 4 17\n",
            [(3, 12), (3, 15)],
            [(3, 12), (4, 17)],
            {"kv": [7, 8, 2, 5], "recurrent": [17]},
        ),
        run_case(
            "exercise_second_increment_1_fresh_namespace",
            variant,
            "獨立分支 [(3, 12), (3, 11)]\n原背景 2 10\n共用可變清單 [(3, 12), (4, 13)]\n被改過的背景 4 13\n",
            [(3, 12), (3, 11)],
            [(3, 12), (4, 13)],
            {"kv": [7, 8, 2, 1], "recurrent": [13]},
        ),
    ]
    result = {
        "schema_version": 1,
        "reviewer_task": "/root/omni_p5_technical_20_14",
        "started_at": started,
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_freeze": {"path": FREEZE, "sha256": FREEZE_SHA},
        "source": {"path": page["snapshot"], "sha256": SOURCE_SHA},
        "runner": {
            "path": pathlib.Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": digest(pathlib.Path(__file__).read_bytes()),
        },
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-20.14/verify_inline_cpu_v1.py",
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "device": "CPU; standard-library list/dictionary arithmetic only",
            "neural_libraries_imported": "none",
        },
        "cases": cases,
        "scope": "Only the frozen artificial inline list/dictionary example and its explicitly assigned increment-1 exercise. Each case starts in a fresh namespace. No model or weights are loaded; no neural forward, training, retuning, Modal, workflow, or publication.",
        "result": "Both cases matched all four printed lines exactly and passed list-state/alias assertions.",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": OUTPUT.relative_to(ROOT).as_posix(), "bytes": OUTPUT.stat().st_size, "sha256": digest(OUTPUT.read_bytes()), "completed_at": result["completed_at"], "result": result["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
