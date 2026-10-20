"""核對凍結小節的算例、邊界條件與完整內容去重流程；不訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import _deduplicate_text


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/5.12.md"
    chapter = (ROOT / "course/chapters/05.md").read_bytes()
    matched = re.search(rb"(?ms)^## 5\.12 .*?(?=^## |\Z)", chapter)
    assert matched is not None
    assert matched[0] == frozen.read_bytes()
    code = re.search(r"```python\n(.*?)```", matched[0].decode(), re.S)[1]
    namespace = {}
    original_stdout = io.StringIO()
    with contextlib.redirect_stdout(original_stdout):
        exec(compile(code, str(frozen), "exec"), namespace)
    grams = namespace["grams"]
    left, right = "紅色圓形在左邊", "紅色圓形在右邊"
    assert len(left) == len(right) == 7
    assert sum(x != y for x, y in zip(left, right, strict=True)) == 1
    assert sha(left.encode()) != sha(right.encode())
    gram_results = {}
    for n, expected_common, expected_union in ((2, 4, 8), (3, 3, 7)):
        sequences = [[text[i : i + n] for i in range(len(text) - n + 1)] for text in (left, right)]
        sets = [grams(text, n=n) for text in (left, right)]
        assert sets == [set(sequence) for sequence in sequences]
        assert all(len(fragment) == n for sequence in sequences for fragment in sequence)
        common, union = sets[0] & sets[1], sets[0] | sets[1]
        assert len(common) == expected_common and len(union) == expected_union
        score = len(common) / len(union)
        sorted_common = sorted(common)
        assert len(common) / len(union) == score and set(sorted_common) == common
        gram_results[str(n)] = {
            "starts": list(range(len(left) - n + 1)),
            "sequences": sequences,
            "common": sorted_common,
            "union": sorted(union),
            "common_count": len(common),
            "union_count": len(union),
            "score": score,
            "rounded_three_decimals": round(score, 3),
        }
    assert gram_results["2"]["score"] == 0.5
    assert gram_results["3"]["score"] == 3 / 7
    repeated = grams("哈哈哈哈")
    assert repeated == {"哈哈"}
    empty_cases = []
    for a, b in (("", ""), ("甲", "乙")):
        x, y = grams(a), grams(b)
        assert not x and not y
        try:
            len(x & y) / len(x | y)
        except ZeroDivisionError as error:
            empty_cases.append({"texts": [a, b], "denominator": len(x | y), "exception": type(error).__name__})
        else:
            raise AssertionError("空聯集應造成除以零")
    assert len(grams("") | grams(left)) > 0
    synonym_pair = ["我很開心", "吾甚喜悅"]
    synonym_sets = [grams(text) for text in synonym_pair]
    synonym_score = len(synonym_sets[0] & synonym_sets[1]) / len(synonym_sets[0] | synonym_sets[1])
    assert synonym_score == 0
    opposite_pair = ["我喜歡貓", "我喜歡狗"]
    opposite_sets = [grams(text) for text in opposite_pair]
    opposite_score = len(opposite_sets[0] & opposite_sets[1]) / len(opposite_sets[0] | opposite_sets[1])
    assert opposite_score == 0.5
    shared_family = [{"family": "方向題", "text": text, "answer": answer} for text, answer in ((left, "左"), (right, "右"))]
    grouped_input = shared_family + [{"family": f"獨立題-{i}", "text": f"題目-{i}", "answer": str(i)} for i in range(8)]
    grouped = split_records(grouped_input, seed=42)
    family_owners = {split for split, rows in grouped.items() for row in rows if row["family"] == "方向題"}
    assert len(family_owners) == 1
    answers = [row["answer"] for rows in grouped.values() for row in rows if row["family"] == "方向題"]
    assert Counter(answers) == Counter(["左", "右"])
    dedup_input = [{"text": text} for text in (left, left, right, "alpha  beta", "alpha\nbeta", "獨立材料甲", "獨立材料乙")]
    deduped = _deduplicate_text(dedup_input)
    assert len(deduped) == 5
    assert {left, right} <= {row["text"] for row in deduped}
    assert len({row["family"] for row in deduped if row["text"] in (left, right)}) == 2
    normalized = [" ".join(row["text"].split()) for row in deduped]
    assert len(set(normalized)) == len(normalized)
    dedup_splits = split_records(deduped, seed=42)
    split_keys = {split: {" ".join(row["text"].split()) for row in rows} for split, rows in dedup_splits.items()}
    overlap = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap[f"{a}/{b}"] = sorted(split_keys[a] & split_keys[b])
        assert not overlap[f"{a}/{b}"]
    record_path = ROOT / "docs/course-experiments/results/real_text.json"
    record = json.loads(record_path.read_bytes())
    code_bindings = {}
    for path in ("scripts/course_experiments/text.py", "scripts/course_experiments/common.py"):
        actual = sha((ROOT / path).read_bytes())
        expected = record["code_sha256"][path]
        assert actual == expected
        code_bindings[path] = {"recorded_sha256": expected, "current_sha256": actual}
    runs = {}
    for name, run in record["results"]["runs"].items():
        assert "near-duplicate clustering was not performed" in run["split_warning"]
        data = {split: run["data"][split] for split in ("train", "validation", "test")}
        assert sum(part["records"] for part in data.values()) == run["deduplicated_records"]
        runs[name] = {"split_warning": run["split_warning"], "source_records": run["source_records"], "deduplicated_records": run["deduplicated_records"], "data": data}
    text_source = (ROOT / "scripts/course_experiments/text.py").read_text()
    text_tree = ast.parse(text_source)
    experiment = next(node for node in text_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_real_text")
    calls = [ast.unparse(node.func) for node in ast.walk(experiment) if isinstance(node, ast.Call)]
    assert "_deduplicate_text" in calls and "split_records" in calls
    cli_command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "real_text"]
    cli = subprocess.run(cli_command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert cli.stdout.splitlines() == ["assets/training/tinystories-v1.tar.gz", "assets/training/chinese-poetry-v1.tar.gz"]
    training_text = (ROOT / "course/training.md").read_text()
    recipe = ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda"
    assert recipe in training_text
    archives = {}
    for name in cli.stdout.splitlines():
        raw = (ROOT / name).read_bytes()
        archives[name] = {"bytes": len(raw), "working_tree_sha256": sha(raw), "is_lfs_pointer": raw.startswith(b"version https://git-lfs.github.com/spec/v1\n")}
    payload = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-12/verification.py",
        "result": "全部斷言通過；未執行訓練。",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "source_sha256": sha(frozen.read_bytes()),
        "current_source_matches_freeze": True,
        "original_stdout": original_stdout.getvalue(),
        "fingerprints": {left: sha(left.encode()), right: sha(right.encode())},
        "grams": gram_results,
        "repeated_fragments": sorted(repeated),
        "empty_cases": empty_cases,
        "synonym_example": {"texts": synonym_pair, "score": synonym_score, "interpretation": "語義由審閱者判讀；程式只核對片段分數。"},
        "different_answer_example": {"texts": opposite_pair, "score": opposite_score},
        "family_grouping": {"owner": sorted(family_owners), "preserved_answers": answers},
        "deduplication": {"input_rows": len(dedup_input), "output_rows": len(deduped), "near_variants_retained": True, "normalized_key_overlap_between_splits": overlap},
        "recorded_experiment": {"path": record_path.relative_to(ROOT).as_posix(), "sha256": sha(record_path.read_bytes()), "revision": record["revision"], "code_bindings": code_bindings, "runs": runs},
        "recipe": {"text": recipe, "asset_query_command": cli_command, "exit_code": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr},
        "local_archives": archives,
        "limitations": ["本機資料包是 Git LFS 指標，未逐筆重讀固定語料或重新核對當年 split 檔案內容。", "未重跑 GPU 訓練、品質或速度量測；正文未引用這些數值。", "完整內容不跨組的結論由相同程式雜湊、去重與切分的結構及合成測資支持；不等於完成語義家族稽核。"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
