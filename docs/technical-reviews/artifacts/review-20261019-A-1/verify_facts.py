"""核對提示算例、資料切分與保存的生成紀錄；不重新訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256, split_records
from tiny_perceptron.data import ByteTokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    run = ROOT / "outputs/reviewer-tools/runs/20261020T070519705427Z"
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/A.1.md"
    raw = frozen.read_bytes()
    assert sha(raw) == "8d9222e2c763e97a00e4eb8a60850a395c41d16d3a281e5c297a1edcb280c859"
    assert (run / "section.md").read_bytes() == raw
    code = raw.decode().split("```python\n", 1)[1].split("```", 1)[0]

    def execute(source):
        capture = io.StringIO()
        state = {}
        with contextlib.redirect_stdout(capture):
            exec(compile(source, "prompt_example", "exec"), state)
        return state, capture.getvalue()

    base, original_output = execute(code)
    altered, reversed_output = execute(code.replace('["1→2", "2→4"]', '["2→4", "1→2"]'))
    assert base["zero"] == altered["zero"]
    assert base["changed"] == altered["changed"]
    assert original_output.splitlines()[-1] == reversed_output.splitlines()[-1]
    assert altered["few"] == "示例：2→4；1→2。\n" + base["question"]
    assert all(p.endswith(base["question"]) for p in [base["zero"], base["few"], base["changed"]])
    assert [p.endswith(base["question"]) for p in [base["zero"], base["few"], base["changed"]]] == [True]*3
    assert "；".join(base["examples"]) == "1→2；2→4"
    assert base["few"].splitlines() == ["示例：1→2；2→4。", base["question"]]
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.parse(code)))

    proof = {
        "doubling": {str(x): 2*x for x in [1,2,3]},
        "tripling": {str(x): 3*x for x in [1,2,3]},
        "squaring_without_examples": {"3": 3**2},
        "alternative_matching_double_examples": {str(x): 2*x+(x-1)*(x-2) for x in [1,2,3]},
        "offsets": {"plus_2": (3+2)%10, "plus_5": (3+5)%10},
    }
    assert proof["doubling"]["3"] == 6
    assert proof["tripling"]["3"] == 9
    assert proof["alternative_matching_double_examples"] == {"1":2,"2":4,"3":8}
    encoding = {
        "ascii_count": len([chr(n) for n in range(128)]),
        "ascii_utf8_single_byte": all(chr(n).encode("utf-8") == bytes([n]) for n in range(128)),
        "ascii_arrow": {"characters":len("->"),"bytes":list("->".encode("utf-8"))},
        "unicode_arrow": {"characters":len("→"),"code_point":hex(ord("→")),"bytes":list("→".encode("utf-8"))},
    }
    assert encoding["ascii_utf8_single_byte"]

    rag_path = ROOT / "docs/course-experiments/results/rag.json"
    rag = json.loads(rag_path.read_bytes())
    pinned = json.loads((directory / "experiment-code-extracts.json").read_bytes())
    historic = subprocess.run(["git", "show", pinned["revision"]+":"+pinned["path"]], capture_output=True, check=True).stdout
    assert sha(historic) == rag["code_sha256"][pinned["path"]] == pinned["file_sha256"]
    historic_tree = ast.parse(historic)
    namespace = {}
    for name in ["_conversation", "_icl_question", "_icl_records"]:
        node = next(n for n in historic_tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "historical_dataset_functions", "exec"), namespace)
    splits = split_records(namespace["_icl_records"](), rag["seed"])
    split_summary = {}
    for name, records in splits.items():
        summary = {"records":len(records), "families":len({r["family"] for r in records}), "sha256":records_sha256(records)}
        assert summary == rag["results"]["icl_split"][name]
        split_summary[name] = {**summary,"values":sorted({r["value"] for r in records}),"family_names":sorted({r["family"] for r in records})}
        for row in records:
            answer = row["messages"][-1]["content"]
            expected = "UNKNOWN" if row["offset"] is None else str((row["value"]+row["offset"])%10)
            assert answer == expected
    assert split_summary["test"]["values"] == [3]
    assert not ({r["family"] for r in splits["train"]} & {r["family"] for r in splits["test"]})
    assert not ({r["family"] for r in splits["train"]} & {r["family"] for r in splits["validation"]})
    assert not ({r["family"] for r in splits["validation"]} & {r["family"] for r in splits["test"]})

    tokenizer = ByteTokenizer()
    samples = rag["results"]["icl_samples_same_model_weights"]
    assert len(samples) == 4
    verified_samples = []
    for row, mode, offset, reverse, answer, expected_ids in zip(
        samples,
        ["zero_shot","few_shot","changed_examples","reversed_examples"],
        [None,2,5,2], [False,False,False,True],
        ["UNKNOWN","9","9","8"],
        [[93,86,83,86,87,95,86,2],[65,2],[65,2],[64,2]], strict=True,
    ):
        g = row["generation"]
        sample = g["samples"][0]
        question = namespace["_icl_question"](3, offset, reverse)
        expected = "UNKNOWN" if offset is None else str((3+offset)%10)
        input_ids = [tokenizer.bos_id,tokenizer.user_id]+tokenizer.encode(question)+[tokenizer.eos_id,tokenizer.assistant_id]
        assert row["mode"] == mode and row["family"] == "map:3"
        assert row["expected"] == expected
        assert g["messages"] == [{"role":"user","content":question}]
        assert g["input_ids"] == input_ids and g["input_tokens"] == len(input_ids)
        assert sample["generated_ids"] == expected_ids
        assert sample["generated"] == answer == tokenizer.decode(expected_ids[:-1])
        assert expected_ids[-1] == tokenizer.eos_id and sample["eos"]
        assert sample["invalid_special_tokens"] == []
        assert g["temperature"] == 0.0 and g["candidate_count"] == 1 and g["max_new_tokens"] == 16
        assert row["correct"] == (answer == expected)
        verified_samples.append(row)
    assert sum(not r["correct"] for r in samples[1:]) == 3
    assert samples[1]["generation"]["samples"][0]["generated"] != samples[3]["generation"]["samples"][0]["generated"]

    source_checks = {}
    for name in ["scripts/course_experiments/common.py","tiny_perceptron/data.py","tiny_perceptron/model.py"]:
        actual = sha((ROOT/name).read_bytes())
        assert actual == rag["code_sha256"][name]
        source_checks[name] = actual
    receipt = {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/0A.md#A.1" --execute',
        "execution": json.loads((run/"execution.json").read_bytes()),
        "environment": json.loads((run/"environment.json").read_bytes()),
        "stdout": (run/"stdout.txt").read_text(),
        "stderr": (run/"stderr.txt").read_text(),
    }
    assert receipt["execution"]["exit_code"] == 0 and receipt["stdout"] == original_output
    with (directory/"section-execution.json").open("x",encoding="utf-8") as output:
        json.dump(receipt,output,ensure_ascii=False,indent=2);output.write("\n")
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-1/verify_facts.py",
        "environment": {"python":platform.python_version(),"torch":str(torch.__version__),"device":"cpu","platform":platform.platform()},
        "result": "所有斷言通過；核對既有生成紀錄，沒有重新訓練或重跑原模型推論。",
        "frozen_source_sha256":sha(raw),
        "prompt_example":{"original_stdout":original_output,"reversed_stdout":reversed_output,"suffix_checks":[True,True,True],"no_imports":True},
        "arithmetic":proof,
        "encoding":encoding,
        "rag_path":str(rag_path.relative_to(ROOT)),
        "rag_sha256":sha(rag_path.read_bytes()),
        "recorded_experiment_environment": {k:rag[k] for k in ["device","seed","torch_version","python_version","gpu"]},
        "training":rag["results"]["training"],
        "model":rag["results"]["model"],
        "split_summary":split_summary,
        "verified_samples":verified_samples,
        "verified_current_source_sha256":source_checks,
        "historical_source":{"revision":pinned["revision"],"path":pinned["path"],"sha256":pinned["file_sha256"]},
        "scope":"確認保存紀錄與原始 ID、資料生成器及同一模型的生成迴圈一致；不證明任何未測數字或其他模型的能力。",
    }
    with (directory/"verification.json").open("x",encoding="utf-8") as output:
        json.dump(result,output,ensure_ascii=False,indent=2);output.write("\n")
    print(json.dumps({"result":result["result"],"splits":split_summary,"generated":[r["generation"]["samples"][0]["generated"] for r in samples],"arithmetic":proof,"encoding":encoding},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
