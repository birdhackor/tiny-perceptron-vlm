"""Independently replay 19.3 and inspect every frozen record without training."""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from tiny_perceptron.capstone import build_dataset, modality_tensors, preference_pairs, prompt_ids

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_03_"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (OUT / f"{PREFIX}{name}").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def json_digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def file_info(path):
    file = ROOT / path
    return {"path": path, "sha256": sha(file.read_bytes()), "bytes": file.stat().st_size}


def inspect_rows(splits, manifest):
    by_split = {}
    for split, rows in splits.items():
        trace = []
        groups = defaultdict(list)
        for row in rows:
            assert set(row) == {"id", "family", "task", "user", "system", "available", "answer", "image", "audio"}
            assert row["id"] == json_digest({key: value for key, value in row.items() if key != "id"})[:20]
            assert row["system"] == f"計算器={'開' if row['available'] else '關'}；風格=短。"
            prompt = prompt_ids(row)
            fingerprint = hashlib.sha256(json.dumps(prompt).encode())
            features = {}
            for kind, feature in zip(("image", "audio"), modality_tensors(row), strict=True):
                if feature is not None:
                    raw = feature.numpy().tobytes()
                    fingerprint.update(raw)
                    assert feature.dtype == torch.float32 and torch.isfinite(feature).all()
                    features[kind] = {
                        "shape": list(feature.shape), "dtype": str(feature.dtype),
                        "bytes": len(raw), "sha256": sha(raw),
                        "min": float(feature.min()), "max": float(feature.max()),
                    }
            assert len(prompt) < 192
            groups[row["family"]].append(row)
            trace.append({
                **row, "prompt_ids": prompt, "features": features,
                "actual_input_sha256": fingerprint.hexdigest(),
                "all_fields_equal_frozen_record": True,
            })
        for family, members in groups.items():
            category, *parts = family.split(":")
            if category == "numbers":
                assert len(members) == 6
                left, right = map(int, parts)
                assert left <= right
                assert Counter(row["task"] for row in members) == {"calculator": 2, "unavailable": 2, "tool_return": 1, "concept": 1}
                for row in members:
                    if row["task"] == "calculator":
                        assert row["available"] and row["answer"] in {f"TOOL:calculator:{left}+{right}", f"TOOL:calculator:{right}+{left}"}
                    elif row["task"] == "unavailable":
                        assert not row["available"] and row["answer"] == "ASK:計算器未開"
                    elif row["task"] == "tool_return":
                        assert row["answer"] == f"DIRECT:{left + right}"
                    else:
                        assert row["answer"] == "DIRECT:把兩個數合起來"
            elif category == "modalities":
                color, shape = parts
                assert len(members) == 36
                assert Counter(row["task"] for row in members) == {"image_color": 9, "image_shape": 9, "joint": 18}
                for row in members:
                    image = row["image"]
                    assert image["color"] == color and image["shape"] == shape
                    assert image["offset"] in {-1, 0, 1} and image["variant"] in {0, 1, 2}
                    if row["task"] == "image_color":
                        assert row["answer"] == f"DIRECT:{color}" and row["audio"] is None
                    elif row["task"] == "image_shape":
                        assert row["answer"] == f"DIRECT:{shape}" and row["audio"] is None
                    else:
                        assert row["answer"] == f"DIRECT:{color},{row['audio']['pitch']}"
            elif category == "audio":
                pitch, number = parts
                assert len(members) == 3
                assert {row["audio"]["variation"] for row in members} == {0, 1, 2}
                for row in members:
                    assert row["audio"]["delta"] == -60 + int(number) * 9
                    assert row["audio"]["pitch"] == pitch
                    assert row["answer"] == f"DIRECT:{pitch}" and row["user"] == "聲音是高還是低？"
            else:
                assert category == "context" and len(members) == 4
                variant = int(parts[0])
                for row in members:
                    expected = {
                        "rag": f"DIRECT:{('書櫃', '桌子', '抽屜')[variant % 3]}",
                        "missing": "ASK:請提供數量", "safety": "DIRECT:不能提供他人密碼", "style": f"DIRECT:{variant}",
                    }
                    assert row["answer"] == expected[row["task"]]
        task_labels = defaultdict(Counter)
        for row in rows:
            task_labels[row["task"]][row["answer"].split(":", 1)[1]] += 1
        baselines = {}
        for task, labels in task_labels.items():
            label = min(labels, key=lambda value: (-labels[value], value))
            baselines[task] = {"count":sum(labels.values()),"correct":labels[label],"accuracy":labels[label]/sum(labels.values()),"majority_label":label}
        assert manifest["task_label_counts"][split] == task_labels
        assert manifest["task_majority_baselines"][split] == baselines
        by_split[split] = {
            "count": len(rows), "distinct_families": len(groups),
            "family_counts_by_category": Counter(family.split(":")[0] for family in groups),
            "tasks": Counter(row["task"] for row in rows),
            "sha256": json_digest(rows),
            "audio_labels": Counter(row["audio"]["pitch"] for row in rows if row["task"] == "audio"),
            "joint_audio_labels": Counter(row["audio"]["pitch"] for row in rows if row["task"] == "joint"),
            "baselines": baselines,
            "actual_input_count": len({row["actual_input_sha256"] for row in trace}),
            "records": trace,
        }
        assert by_split[split]["sha256"] == manifest["sha256"][split]
    overlaps = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        families_l = {row["family"] for row in splits[left]}
        families_r = {row["family"] for row in splits[right]}
        inputs_l = {row["actual_input_sha256"] for row in by_split[left]["records"]}
        inputs_r = {row["actual_input_sha256"] for row in by_split[right]["records"]}
        item = {"left":left,"right":right,"family_intersection":sorted(families_l & families_r),"actual_input_intersection":sorted(inputs_l & inputs_r)}
        assert not item["family_intersection"] and not item["actual_input_intersection"]
        overlaps.append(item)
    return by_split, overlaps


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    source_body = dict(sections(ROOT / "course/chapters/19.md"))["19.3"]
    assert source_body.encode() == (OUT / f"{PREFIX}section_19_3.md").read_bytes()
    snippet = re.findall(r"```python\n(.*?)```", source_body, re.S)[0]
    snippets = []
    for seed in (42, 43):
        code = snippet if seed == 42 else snippet.replace("seed=42", "seed=43")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, f"19.3-seed-{seed}", "exec"), {})
        (OUT / f"{PREFIX}snippet_seed_{seed}.py").write_text(code)
        (OUT / f"{PREFIX}snippet_seed_{seed}.txt").write_text(stdout.getvalue())
        snippets.append({"seed":seed,"stdout":stdout.getvalue()})
    frozen_path = "docs/course-experiments/capstone-evidence/deployment/data.json"
    frozen = json.loads((ROOT / frozen_path).read_text())
    splits, manifest = build_dataset(seed=42)
    assert frozen["manifest"] == manifest
    assert frozen["splits"] == splits
    assert manifest["counts"] == {"train":552,"validation":84,"test":90}
    assert sum(map(len, splits.values())) == 726
    assert splits["train"][0]["user"] == "3+6等於多少？"
    traces, overlaps = inspect_rows(splits, manifest)
    heldout = [row for row in splits["test"] if row["family"] == "numbers:1:2"]
    assert len(heldout) == 6
    assert not any(row["family"] == "numbers:1:2" for name in ("train","validation") for row in splits[name])
    for split, count in (("train",8),("validation",1),("test",1)):
        audio = [family for family in manifest["families"][split]["audio"]]
        assert Counter(family.split(":")[1] for family in audio) == {"low":count,"high":count}
    assert manifest["families"]["validation"]["modalities"] == ["modalities:blue:circle"]
    assert manifest["families"]["test"]["modalities"] == ["modalities:green:square"]
    train_images = [row["image"] for row in splits["train"] if row["image"] is not None]
    assert {image["color"] for image in train_images} == {"red","green","blue"}
    assert {image["shape"] for image in train_images} == {"circle","square"}
    pairs = preference_pairs(splits["train"])
    assert all(pair["row"] in splits["train"] for pair in pairs)
    assert len(pairs) == 92
    stages = []
    for stage in ("pretrain","sft","joint","dpo"):
        base = f"docs/course-experiments/capstone-evidence/{stage}"
        report_path = base + "/train-report.json"
        report = json.loads((ROOT / report_path).read_text())
        assert report["data_manifest"] == manifest
        assert json.loads((ROOT / (base + "/data-manifest.json")).read_text()) == manifest
        assert report["seed"] == 42 and report["schedule_completed"] and not report["test_evaluated"]
        assert report["code_sha256"]["tiny_perceptron/capstone.py"] == file_info("tiny_perceptron/capstone.py")["sha256"]
        assert report["code_sha256"]["scripts/course_experiments/capstone.py"] == file_info("scripts/course_experiments/capstone.py")["sha256"]
        stages.append({"stage":stage,**file_info(report_path),"manifest_equal":True,"data_sha256":report["data_manifest"]["sha256"],"seed":report["seed"],"device":report["device"],"steps":report["steps"],"requested_steps":report["requested_steps"],"effective_tokens":report["effective_tokens"],"test_evaluated":report["test_evaluated"],"recorded_training_seconds":report["seconds"],"time_scope":"Historical CUDA training record, not replayed or claimed as CPU performance in 19.3"})
    additional = []
    for path in ("docs/course-experiments/capstone-evidence/deployment/deployment-report.json","docs/course-experiments/capstone-evidence/student/student-report.json"):
        report = json.loads((ROOT / path).read_text())
        assert report["data_manifest"] == manifest
        additional.append({**file_info(path),"manifest_equal":True,"data_sha256":report["data_manifest"]["sha256"],"recipe_frozen_before_test":report["recipe_frozen_before_test"]})
    selection_path = "docs/course-experiments/capstone-selection.json"
    selection = json.loads((ROOT / selection_path).read_text())
    assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"]
    for stage in ("joint","dpo"):
        candidate = selection["candidates"][stage]
        val = json.loads((ROOT / f"docs/course-experiments/capstone-evidence/{stage}/validation.json").read_text())
        assert candidate["validation_count"] == val["count"] == 84
        assert candidate["validation_end_to_end_correct"] == val["end_to_end_correct"]
        assert candidate["evidence_sha256"] == file_info(candidate["evidence"])["sha256"]
    exercise_splits, exercise_manifest = build_dataset(seed=43)
    exercise_traces, exercise_overlaps = inspect_rows(exercise_splits, exercise_manifest)
    changed = {name:sorted({row["family"] for row in splits[name]} ^ {row["family"] for row in exercise_splits[name]}) for name in splits}
    assert any(changed.values()) and manifest["sha256"] != exercise_manifest["sha256"]
    result = {
        "reviewer_task":"/root/integration_technical_coordinator/fact_v2_19_03",
        "executed_at":datetime.now(timezone.utc).isoformat(),
        "command":"PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_03_audit.py",
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"torch_git":torch.version.git_version,"device":"cpu","threads":str(torch.get_num_threads()),"platform":platform.platform()},
        "source_sha256":sha(source_body.encode()),
        "source_files":[file_info(p) for p in ("tiny_perceptron/capstone.py","tiny_perceptron/multimodal.py","scripts/course_experiments/capstone.py","scripts/course_experiments/capstone_student.py","scripts/course_experiments/capstone_deployment.py","tests/test_capstone.py",frozen_path,selection_path)],
        "snippet_executions":snippets,
        "frozen_record_equality":{"train":552,"validation":84,"test":90,"total":726,"manifest_equal":True,"all_original_fields_equal":True},
        "manifest":manifest,"seed42":traces,"overlaps_seed42":overlaps,"reserved_1_2":heldout,
        "preference_train_pairs":len(pairs),"formal_stages":stages,"comparison_manifests":additional,"selection":selection,
        "seed43":{"manifest":exercise_manifest,"changed_family_memberships":changed,"overlaps":exercise_overlaps,"summary":{name:{key:value for key,value in detail.items() if key!='records'} for name,detail in exercise_traces.items()}},
        "elapsed_seconds":time.perf_counter()-started,
        "timing_scope":"One CPU audit of fixed generators, all original row fields, tokenizer and real synthetic media tensors; no model loading, optimization, generation or speed comparison.",
        "result":"All assertions passed, 726 original records equal frozen manifest and splits; three family and real-input intersections zero for seeds42 and43.",
    }
    write("audit.json",result)
    print(json.dumps({"result":result["result"],"sha256":manifest["sha256"],"counts":manifest["counts"],"preference_train_pairs":len(pairs),"seed43_changed_families":{k:len(v) for k,v in changed.items()},"elapsed_seconds":result["elapsed_seconds"]},ensure_ascii=False,indent=2))
    test_command = [str(ROOT / ".venv/bin/python"),"-m","pytest","tests/test_capstone.py","-q","-k","family_split_and_actual or audio_family_labels or one_plus_two"]
    completed = subprocess.run(test_command, cwd=ROOT, text=True, capture_output=True, check=False)
    test_environment = {key:value for key,value in result["environment"].items() if key != "threads"}
    write("pytest.json",{"command":".venv/bin/python -m pytest tests/test_capstone.py -q -k 'family_split_and_actual or audio_family_labels or one_plus_two'","environment":test_environment,"returncode":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr,"source_files":result["source_files"],"result":"Selected dataset-integrity tests passed" if completed.returncode==0 else "Selected tests failed"})
    print(completed.stdout)
    assert completed.returncode == 0


if __name__ == "__main__":
    main()
