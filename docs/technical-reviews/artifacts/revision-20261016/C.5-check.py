"""C.5 independent fixed-report audit; no generation, training or GPU benchmark."""

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

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
from tiny_perceptron.data import ByteTokenizer

source = ROOT / "course/chapters/0C.md"
body = lesson_slices(source.read_text())["C.5"]
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(code, "C.5 frozen example", "exec"), {})
expected = "候選 1 生成 80 驗證 5 總token 85\n候選 4 生成 80 驗證 20 總token 100\n"
assert output.getvalue() == expected

path = ROOT / "docs/course-experiments/results/reasoning.json"
report = json.loads(path.read_text())
tokenizer = ByteTokenizer()
summary = []
questions = None
for mode, comparison in report["results"]["comparison"].items():
    for budget in comparison["budgets"]:
        k = budget["candidate_count"]
        rows = budget["samples"]
        assert len(rows) == 24
        current_questions = [row["question"] for row in rows]
        questions = current_questions if questions is None else questions
        assert current_questions == questions
        totals = {"tokens": 0, "seconds": 0.0, "coverage": 0, "majority": 0, "selection_failed": 0}
        for row in rows:
            assert row["truth"] == row["a"] + row["b"] + row["c"]
            candidates = row["candidates"]
            assert len(candidates) == k
            answers, final_correct = [], []
            token_count = 0
            for candidate in candidates:
                ids = candidate["generated_ids"]
                assert candidate["generated_tokens"] == len(ids)
                eos = bool(ids and ids[-1] == tokenizer.eos_id)
                assert candidate["eos"] == eos
                decoded = tokenizer.decode(ids[:-1] if eos else ids)
                assert decoded == candidate["generated"]
                text = decoded.strip()
                clean = not candidate["invalid_special_tokens"]
                matched = re.fullmatch(r"-?[0-9]+", text) if mode == "direct" else re.search(r";answer=(-?[0-9]+)$", text)
                answer = (int(text) if mode == "direct" else int(matched[1])) if clean and matched else None
                assert answer == candidate["final_answer"]
                correct = answer == row["truth"]
                assert correct == candidate["final_correct"]
                final_correct.append(correct)
                if answer is not None:
                    answers.append(answer)
                token_count += len(ids)
            covered = any(final_correct)
            majority = Counter(answers).most_common(1)[0][0] if answers else None
            majority_correct = majority == row["truth"]
            assert covered == row["oracle_coverage"]
            assert majority == row["majority_answer"]
            assert majority_correct == row["majority_correct"]
            assert (covered and not majority_correct) == row["selection_failed_despite_coverage"]
            generation = row["generation"]
            assert generation["candidate_count"] == k
            assert generation["temperature"] == 0.7
            assert generation["max_new_tokens"] == (8 if mode == "direct" else 48)
            assert generation["generated_tokens"] == token_count
            totals["tokens"] += token_count
            totals["seconds"] += generation["seconds"]
            totals["coverage"] += covered
            totals["majority"] += majority_correct
            totals["selection_failed"] += covered and not majority_correct
        assert budget["generated_tokens"] == totals["tokens"]
        assert abs(budget["generation_seconds"] - totals["seconds"]) < 1e-12
        for field, key in [("oracle_coverage", "coverage"), ("majority_accuracy", "majority")]:
            assert budget[field]["numerator"] == totals[key]
            assert budget[field]["denominator"] == 24
        summary.append({"mode": mode, "k": k, "questions": len(rows), "candidate_samples": len(rows)*k, **totals})

historical_path = "scripts/course_experiments/applications.py"
revision = report["revision"]
historical = subprocess.check_output(["git", "show", revision + ":" + historical_path], cwd=ROOT)
text = historical.decode()
assert text.index("prompt = _prompt_ids") < text.index("started = time.perf_counter()")
assert text.index("seconds = time.perf_counter() - started") < text.index('"generated": tokenizer.decode(raw)')
assert 'current = torch.tensor([prompt] * count, device=ctx.device)' in text
assert 'for candidate_count in (1, 2, 4, 8):' in text
assert 'temperature=0.7,' in text
assert 'tokens=8 if mode == "direct" else 48,' in text
result = {
    "purpose": "Actual execution of C.5 toy budget example and offline arithmetic inspection of fixed historical report; no rerun of historical GPU experiment.",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU report inspection only", "platform": platform.platform()},
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/C.5-check.py",
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "toy_stdout": output.getvalue(),
    "fixed_report": {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "revision": revision},
    "historical_environment": {key: report[key] for key in ["python_version", "torch_version", "device", "gpu", "seed"]},
    "historical_source": {"path": historical_path, "revision": revision, "sha256": hashlib.sha256(historical).hexdigest()},
    "split": report["results"]["split"],
    "training": {mode: {key: comparison["training"][key] for key in ["steps", "records", "effective_tokens"]} for mode, comparison in report["results"]["comparison"].items()},
    "recomputed": summary,
    "steps_k8_over_k1_token_ratio": 4132/512,
    "timing_scope": "Historical _sample: prompt formatting and batch tensor initialization precede timer; synchronized generation, sampling and bookkeeping are timed; tokenizer.decode, vote and arithmetic verifier follow timer. One batch per question; one pass per k; no documented dedicated warmup or repeated timing trials.",
    "limitations": report["results"]["limits"],
    "result": "All assertions passed; 8 budget cells and 720 retained candidates checked; C.5 quoted token and rounded second values agree with historical report."
}
print(json.dumps(result, ensure_ascii=False, indent=2))
