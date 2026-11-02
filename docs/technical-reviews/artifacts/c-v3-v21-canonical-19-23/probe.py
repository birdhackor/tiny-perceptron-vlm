"""19.23 fresh review: source delta, small CPU examples, retained raw rubric rows."""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASK = "/root/c_v3_technical_coord/canonical_19_23"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    started = datetime.now(UTC).isoformat()
    torch.set_num_threads(1)
    freeze = ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v21.json"
    old_freeze = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json"
    new_entry = next(p for p in json.loads(freeze.read_text())["pages"] if p["page_id"] == "19.23")
    old_entry = next(p for p in json.loads(old_freeze.read_text())["pages"] if p["page_id"] == "19.23")
    new = (ROOT / new_entry["snapshot"]).read_bytes()
    old = (ROOT / old_entry["snapshot"]).read_bytes()
    body = new.decode()
    chapter = (ROOT / "course/chapters/19.md").read_text()
    current = re.search(r"^## 19\.23 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
    assert digest(new) == new_entry["source_sha256"] == digest(current.encode())
    assert digest(old) == old_entry["source_sha256"]
    additions = [
        "\n本節保留C-v2的共享計算設計與當時未達標的實測。後續合格的裝置成品是[19.25](19.md#19.25)的C-v3；本節的byte介面、權重與採用規則不是C-v3的替換配套。\n",
        "\n程式裡，`q`是問題側的特徵，`u`把問題與當前候選描述一起編成分支特徵，`H`再把各分支讀取M後的結果疊成一列一個候選的表。各實質候選的分數逐列計算；none才讀整張表的彙整。\n",
        "本批正式rubric題的permission與ready都已知，0／1就是已知no／yes對應的數字；生成器拒絕用unknown建立rubric標籤。C-v2這個三分類介面未替缺欄另設none，因此不能由合法文字輸入推論它會正確拒答；[19.25](19.md#19.25)的C-v3才把rubric缺欄的none納入服務與驗收。\n\n",
    ]
    restored = body
    for addition in additions:
        assert restored.count(addition) == 1
        restored = restored.replace(addition, "", 1)
    restored = restored.replace(
        "因此**本節C-v2只發布為研究候選，當時保留舊版C預設**；目前裝置決策改用[19.25](19.md#19.25)另行驗收的C-v3。",
        "因此**新版C只發布為研究候選，舊版C保留預設**。",
        1,
    )
    assert restored.encode() == old

    namespace = {"__name__": "__review_fences__"}
    fence_stdout = []
    blocks = re.findall(r"^```python\n(.*?)^```", body, re.M | re.S)
    assert len(blocks) == 3
    for index, code in enumerate(blocks):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, f"19.23:python-fence-{index + 1}", "exec"), namespace)
        fence_stdout.append(stdout.getvalue())
    model, memory, choice = namespace["model"], namespace["M"], namespace["choice"]
    with torch.inference_mode():
        result = model.score(memory, *choice)
        expanded = model.score(memory, choice[0], choice[1], choice[2] + [("bill", "查帳單明細")])
        renamed = model.score(memory, choice[0], choice[1], [("changed-" + cid, desc) for cid, desc in choice[2]])
        q = model.pool([namespace["TYPE"][choice[0]]] + namespace["utf8"](choice[1], 128))
        u = model.pool([namespace["TYPE"][choice[0]]] + namespace["utf8"](choice[1], 128) + [namespace["SEP"]] + namespace["utf8"](choice[2][0][1], 128))
        score_error = float((result["scores"][:2] - expanded["scores"][:2]).abs().max())
        rename_error = float((result["scores"] - renamed["scores"]).abs().max())
        assert score_error == 0 and rename_error == 0
        softmax = torch.tensor([0.0, 1e-8], dtype=torch.float32).softmax(-1).tolist()
        assert softmax == [0.5, 0.5]

    from tiny_perceptron.omni_products.choice import Candidate, ChoiceModel, Question
    from tiny_perceptron.omni_products.questions import rubric_question, state_text

    rubric_rows, rejected = [], 0
    for task in range(3):
        for permission in (False, True):
            for ready in (False, True):
                fact = {"task": task, "permission": permission, "ready": ready}
                question = rubric_question(state_text([fact] * 3), fact)
                expected = (int(permission) + int(ready)) if task == 0 else (2 if permission else int(ready)) if task == 1 else (2 if ready else int(permission))
                assert question.gold == str(expected)
                rubric_rows.append({**fact, "expected": str(expected), "observed": question.gold})
        for permission, ready in ((None, False), (True, None), (None, None)):
            fact = {"task": task, "permission": permission, "ready": ready}
            try:
                rubric_question(state_text([fact] * 3), fact)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("C-v2 rubric accepted an unknown field")
    assert rejected == 9
    formal = ChoiceModel().eval().requires_grad_(False)
    assert sum(p.numel() for p in formal.parameters()) == 279170
    unknown = Question("A permission=unknown ready=yes", "rubric", "Grade A", tuple(Candidate(str(i), f"grade {i}") for i in range(3)))
    with torch.inference_mode():
        unknown_prediction = formal.predict(unknown)
    assert unknown_prediction["ids"] == ["0", "1", "2"]

    descriptor = json.loads((ROOT / "docs/decision-product/v3/public-release.json").read_text())
    archive_path = Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
    archive_bytes = archive_path.read_bytes()
    assert len(archive_bytes) == descriptor["bytes"] and digest(archive_bytes) == descriptor["sha256"]
    acquisition = json.loads((ROOT / "docs/decision-product/v3/metadata/acquisition.json").read_text())
    members = {}
    with zipfile.ZipFile(archive_path) as archive:
        for item in acquisition["files"]:
            data = archive.read(item["archive_member"])
            assert data == (ROOT / item["path"]).read_bytes()
            assert len(data) == item["bytes"] and digest(data) == item["sha256"]
            members[item["archive_member"]] = {"bytes": len(data), "sha256": digest(data)}
        raw = archive.read("tiny-perceptron-c-v3/evidence/final-observations.json")
        observations = json.loads(raw)
        members["tiny-perceptron-c-v3/evidence/final-observations.json"] = {"bytes": len(raw), "sha256": digest(raw)}
        policy = json.loads(archive.read("tiny-perceptron-c-v3/bundle/policy.json"))
    rubric = [row for row in observations if row["kind"] == "rubric"]
    missing = [row for row in rubric if row["panel"] == "missing"]
    assert len(rubric) == 1500 and len(missing) == 362
    assert all(row["gold_id"] == "none" and row["ids"][max(range(len(row["logits"])), key=lambda i: row["logits"][i])] == "none" for row in missing)
    assert policy["threshold"] == 0.99
    assert policy["temperatures"]["rubric"] == 0.42044820762685725
    implementation_matches = {}
    for name in ("service.py", "model.py", "record_model.py", "indexed_model.py", "runtime.py", "cli.py", "data.py"):
        path = "tiny_perceptron/decision_product/" + name
        actual = digest((ROOT / path).read_bytes())
        assert actual == descriptor["implementation"][path]
        implementation_matches[path] = actual
    shared_path = ROOT / "docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/execution.json"
    shared = json.loads(shared_path.read_text())
    assert digest(shared_path.read_bytes()) == "375deeebc7d2107171fd38ed80ffbe127823b7865640ca3118a8fa479c6b1959"
    unknown_example = next(row for row in shared["new_cpu_examples"] if row["request"] == "record-unknown.json")
    assert unknown_example["returncode"] == 0
    assert unknown_example["response"]["results"]["rubric"]["model_choice"] == "none"
    assert unknown_example["response"]["results"]["rubric"]["status"] == "abstained"

    test = subprocess.run([sys.executable, "-m", "pytest", "tests/test_course_product_entry.py", "-q", "-p", "no:cacheprovider"], cwd=ROOT, text=True, capture_output=True)
    assert test.returncode == 0, test.stdout + test.stderr
    output = {
        "schema": "c-v3-v21-canonical-19-23-execution-v1",
        "reviewer_task": TASK,
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "CPU", "threads": "1"},
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-23/probe.py",
        "source_identity": {"freeze_sha256": digest(freeze.read_bytes()), "old_freeze_sha256": digest(old_freeze.read_bytes()), "source_sha256": digest(new), "old_source_sha256": digest(old), "current_matches_snapshot": True, "undo_exactly_three_additions_and_one_replacement_matches_source18": True, "python_fence_count": len(blocks), "old_python_fences_byte_identical": re.findall(r"^```python\n(.*?)^```", old.decode(), re.M | re.S) == blocks},
        "fence_stdout": fence_stdout,
        "toy_features": {"q_shape": list(q.shape), "u_shape": list(u.shape), "H_shape": list(result["hidden"].shape), "adding_candidate_original_raw_score_max_error": score_error, "renaming_ids_raw_score_max_error": rename_error, "float32_softmax_0_1e_minus8": softmax},
        "c_v2_rubric": {"known_truth_table": rubric_rows, "unknown_generator_rejections": rejected, "unknown_text_prediction_ids": unknown_prediction["ids"], "parameter_count": sum(p.numel() for p in formal.parameters())},
        "c_v3_archive": {"path": str(archive_path), "bytes": len(archive_bytes), "sha256": digest(archive_bytes), "metadata_byte_identity_count": len(acquisition["files"]), "members_read": members, "implementation_matches": implementation_matches, "public_transport_retested": False},
        "retained_c_v3_rubric": {"evidence_mode": "saved observations only; no new final inference or opening", "rubric_observations": len(rubric), "missing_observations": len(missing), "missing_gold_none": len(missing), "missing_predicted_none": len(missing), "all_final_family_count": len({row["family"] for row in observations}), "temperatures": policy["temperatures"], "choice_threshold": policy["threshold"]},
        "shared_raw_cpu_example": {"path": str(shared_path.relative_to(ROOT)), "sha256": digest(shared_path.read_bytes()), "request": unknown_example["request"], "rubric_response": unknown_example["response"]["results"]["rubric"], "evidence_mode": "read existing raw command/result/environment; not my new inference"},
        "entry_tests": {"command": [sys.executable, "-m", "pytest", "tests/test_course_product_entry.py", "-q", "-p", "no:cacheprovider"], "returncode": test.returncode, "stdout": test.stdout, "stderr": test.stderr},
        "limits": ["Only the three source-identical local reference fences and deterministic generator checks were newly executed.", "C-v3 missing-rubric measurements and public example are retained original observations; no training, final opening, final inference or full product re-acceptance.", "C-v2 training provenance, stored final logits and Kev original-source evidence remain to be inspected as immutable retained evidence after original receipt.", "No network download, package installation, weights extraction, publication or device action."],
    }
    (HERE / "execution.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"execution_path": str((HERE / "execution.json").relative_to(ROOT)), "sha256": digest((HERE / "execution.json").read_bytes()), "fences": len(blocks), "rubric_known": len(rubric_rows), "rubric_unknown_rejected": rejected, "c_v3_missing_rubric_retained": len(missing), "entry_tests": test.stdout.strip()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
