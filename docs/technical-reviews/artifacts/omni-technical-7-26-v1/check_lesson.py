"""Execute the frozen 7.26 code and its stated exercise; inspect the linked SFT path.

This is a deterministic CPU data check, not a model training or quality experiment.
Run from the repository root with uv run --extra cpu python.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import re
import sys

import torch

from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(code, name):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, name, "exec"), namespace)
    return namespace, output.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path.cwd()
    freeze_path = root / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
    freeze = json.loads(freeze_path.read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "7.26")
    snapshot = root / page["snapshot"]
    markdown = snapshot.read_text()
    assert digest(snapshot) == page["source_sha256"]
    live = (root / page["source"]).read_text()
    live_section = re.search(r"^## 7\.26[^\n]*\n.*?(?=^## |\Z)", live, re.M | re.S).group(0)
    assert hashlib.sha256(live_section.encode()).hexdigest() == page["source_sha256"]
    assert page["figures_sha256"] == {}
    code_blocks = re.findall(r"```python\n(.*?)```", markdown, re.S)
    assert len(code_blocks) == 1
    code = code_blocks[0]
    expected_first_six = re.search(r"```text\n(.*?)```", markdown, re.S).group(1).splitlines()
    namespace, stdout = execute(code, str(snapshot))
    assert stdout.splitlines()[:6] == expected_first_six
    rows = namespace["rows"]
    expected_answers = ["查卡片進度", "以上皆不適合", "查卡片進度", "查卡片進度", "詢問申請編號", "以上皆不適合"]
    expected_reasons = ["有合適候選", "列表缺正解", "有合適候選", "有合適候選", "有合適候選", "不支援"]
    assert [r["answer"] for _, r in rows] == expected_answers
    assert [r["audit_reason"] for _, r in rows] == expected_reasons
    expected_prompt = (
        "規則：查卡片進度要申請編號；未提供就先詢問。查地址變更規則不需編號。其餘任務不支援。\n"
        "材料：任務=查卡片進度；申請編號=K17\n"
        "問題：依規則，下一步選哪個候選？只輸出描述；沒有適合動作就輸出以上皆不適合。\n"
        "候選：查地址變更規則／詢問申請編號／以上皆不適合"
    )
    assert rows[1][1]["prompt"] == expected_prompt
    original_lines = rows[0][1]["prompt"].splitlines()
    removed_lines = rows[1][1]["prompt"].splitlines()
    assert original_lines[:3] == removed_lines[:3]
    original_options = original_lines[3].removeprefix("候選：").split("／")
    removed_options = removed_lines[3].removeprefix("候選：").split("／")
    assert [o for o in original_options if o != "查卡片進度"] == removed_options
    assert "以上皆不適合" in original_options and "以上皆不適合" in removed_options
    assert rows[0][1]["answer"] != rows[1][1]["answer"]

    notebook_path = root / page["notebook"]
    assert digest(notebook_path) == page["notebook_sha256"]
    notebook = json.loads(notebook_path.read_text())
    notebook_code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(notebook_code) == 2
    assert notebook_code[1].strip() == code.strip()
    notebook_namespace = {}
    notebook_output = io.StringIO()
    with contextlib.redirect_stdout(notebook_output):
        for i, cell in enumerate(notebook_code):
            exec(compile(cell, f"{notebook_path}:code-cell-{i}", "exec"), notebook_namespace)
    assert not notebook_namespace["in_colab"]
    assert notebook_output.getvalue() == stdout

    exercise_old = '("缺編號", make({"任務": "查卡片進度"}, full))'
    exercise_new = '("缺編號", make({"任務": "查卡片進度"}, ["card", "address", "none"]))'
    assert code.count(exercise_old) == 1
    exercise_namespace, _ = execute(code.replace(exercise_old, exercise_new), "7.26-stated-exercise")
    exercise_row = exercise_namespace["rows"][4][1]
    assert exercise_row["answer"] == "以上皆不適合"
    assert exercise_row["audit_reason"] == "列表缺正解"

    make = namespace["make"]
    order_checks = 0
    for order in itertools.permutations(namespace["full"]):
        assert make(namespace["state"], list(order))["answer"] == "查卡片進度"
        order_checks += 1
    address = make({"任務": "查地址變更規則"}, namespace["full"])
    assert address["answer"] == "查地址變更規則"
    assert len(set(namespace["meaning"].values())) == len(namespace["meaning"])

    tok = ByteTokenizer()
    sft_checks = []
    for name, row in rows:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        assert all(set(m) == {"role", "content"} for m in messages)
        assert "audit_reason" not in json.dumps(messages, ensure_ascii=False)
        x, y = render_chat(messages)
        wanted = tok.encode(row["answer"]) + [tok.eos_id]
        assert y[y != -100].tolist() == wanted
        first = (y != -100).nonzero()[0].item()
        assert x[first].item() == tok.assistant_id
        assert y[first].item() == tok.encode(row["answer"])[0]
        assert (y[:first] == -100).all().item()
        assert y[-1].item() == tok.eos_id
        assert x.dtype == torch.long and y.dtype == torch.long
        sft_checks.append({"case": name, "x_length": len(x), "first_answer_prediction_position": first,
                           "target_count": len(wanted), "targets_equal_answer_bytes_and_eos": True})

    first_row = rows[0][1]
    x, y = render_chat([{"role": "user", "content": first_row["prompt"]},
                        {"role": "assistant", "content": first_row["answer"]}])
    logits = torch.zeros((1, len(x), tok.vocab_size), dtype=torch.float64, requires_grad=True)
    loss = masked_loss(logits, y[None])
    loss.backward()
    assert abs(loss.item() - math.log(tok.vocab_size)) < 1e-12
    ignored_gradient_max = logits.grad[0, y == -100].abs().max().item()
    supervised_gradient_max = logits.grad[0, y != -100].abs().max().item()
    assert ignored_gradient_max == 0.0 and supervised_gradient_max > 0.0

    result = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-7-26-v1/check_lesson.py --output docs/technical-reviews/artifacts/omni-technical-7-26-v1/execution.json",
        "environment": {"python": sys.version, "torch": torch.__version__, "torch_git_version": torch.version.git_version,
                        "platform": platform.platform(), "device": "cpu", "threads": str(torch.get_num_threads())},
        "hashes": {"freeze": digest(freeze_path), "snapshot": digest(snapshot), "notebook": digest(notebook_path),
                   "data_py": digest(root / "tiny_perceptron/data.py"), "model_py": digest(root / "tiny_perceptron/model.py"),
                   "lesson_embedded_code": hashlib.sha256(code.encode()).hexdigest()},
        "embedded_stdout": stdout,
        "all_six_displayed_lines_match": True,
        "notebook_local_cells_match_embedded_stdout": True,
        "only_candidates_differ_in_original_and_removed_prompts": True,
        "six_rows": [{"case": name, "answer": row["answer"], "audit_reason": row["audit_reason"]} for name, row in rows],
        "stated_exercise": {"answer": exercise_row["answer"], "audit_reason": exercise_row["audit_reason"]},
        "address_without_id": {"answer": address["answer"], "audit_reason": address["audit_reason"]},
        "permutations_checked": order_checks,
        "sft_checks": sft_checks,
        "uniform_logits_loss_check": {"dtype": str(logits.dtype), "expected": math.log(tok.vocab_size), "observed": loss.item(), "atol": 1e-12,
                                       "ignored_logit_gradient_max": ignored_gradient_max,
                                       "supervised_logit_gradient_max": supervised_gradient_max},
        "reviewer_harness_diagnostic": "The first harness run used float32 uniform logits and an extra reviewer-chosen 1e-6 tolerance. It failed only that supplemental assertion: observed 5.5759501457214355 versus log(264)=5.575949103146316, absolute error 1.042575119569733e-6. A separate CPU diagnostic also found float64 error 8.881784197001252e-16. The harness now uses float64 and 1e-12 for this added check. No course code or claimed lesson output was changed.",
        "scope": "Exact frozen code, six examples, the stated exercise, 24 permutations of the shown full list, and the linked local render_chat/masked_loss data path only. No model forward, optimizer, training, GPU, or learned-quality measurement. Inputs use known unique IDs and retain none; key presence defines whether an application ID was provided."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": "all assertions passed", "artifact": str(args.output), "sha256": digest(args.output),
                      "cases": len(rows), "exercise": exercise_row["answer"], "permutations": order_checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
