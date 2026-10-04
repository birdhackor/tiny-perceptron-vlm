"""Bounded CPU evidence for independent technical review of lesson 18.1."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments import compression
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import ByteTokenizer, CharTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import load_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
TMP = Path("/tmp/fact_v2_18_01")
PREFIX = "fact_v2_18_01_"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = OUT / (PREFIX + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


def tensor_receipts(state):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
                   "finite": bool(torch.isfinite(value).all())}
            for name, value in state.items()}


def verify_samples(samples, tokenizer):
    hits = 0
    for sample in samples:
        ids = sample["generated_ids"]
        raw = ids[:ids.index(tokenizer.eos_id)] if tokenizer.eos_id in ids else ids
        exact = raw == tokenizer.encode(sample["expected"])
        assert exact == sample["exact"]
        assert tokenizer.decode(raw) == sample["generated"]
        assert (tokenizer.eos_id in ids) == sample["ended_with_eos"]
        hits += exact
    return hits


def main():
    torch.set_num_threads(1)
    env = {"python": platform.python_version(), "torch": str(torch.__version__),
           "torch_git": torch.version.git_version, "device": "cpu",
           "platform": platform.platform(), "threads": str(torch.get_num_threads())}
    lesson = dict(sections(ROOT / "course/chapters/18.md"))["18.1"]
    (OUT / (PREFIX + "section.md")).write_text(lesson)
    prerequisites = []
    for source, ids in [("course/chapters/04.md", ["4.6"]), ("course/chapters/07.md", ["7.11"]),
                        ("course/training.md", ["T.4", "T.5", "T.8"]),
                        ("course/chapters/18.md", ["18.6"])]:
        bodies = dict(sections(ROOT / source))
        for sid in ids:
            prerequisites.append({"source": source + "#" + sid,
                                  "sha256": hashlib.sha256(bodies[sid].encode()).hexdigest(),
                                  "body": bodies[sid]})
    save("prerequisites.json", prerequisites)
    code = re.findall(r"```python\n(.*?)```", lesson, re.S)[0]
    (OUT / (PREFIX + "lesson_program.txt")).write_text(code)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "18.1-original", "exec"), {})
    exercise = code.replace('print("回答", answer, "分布", prob.tolist())', 'print("回答", answer)')
    with contextlib.redirect_stdout(output):
        exec(compile(exercise, "18.1-exercise", "exec"), {})
    assert output.getvalue().splitlines()[-2:] == ["回答 4", "回答 4"]
    (OUT / (PREFIX + "snippet_stdout.txt")).write_text(output.getvalue())
    probs = [[0.8, 0.15, 0.05], [0.4, 0.35, 0.25]]
    numeric = [{"argmax": max(range(3), key=p.__getitem__), "sum": sum(p),
                "first_second_gap": p[0] - p[1]} for p in probs]
    student = torch.log(torch.tensor([[[0.4, 0.35, 0.25], [0.1, 0.2, 0.7]]], requires_grad=True))
    student.retain_grad()
    teacher = torch.log(torch.tensor([[[0.8, 0.15, 0.05], [0.9, 0.09, 0.01]]], requires_grad=True))
    teacher.retain_grad()
    labels = torch.tensor([[0, -100]])
    kl = distillation_kl(student, teacher, labels, temperature=2)
    p = [math.sqrt(x) for x in probs[0]]
    q = [math.sqrt(x) for x in probs[1]]
    p = [x / sum(p) for x in p]
    q = [x / sum(q) for x in q]
    manual = 4 * sum(a * math.log(a / b) for a, b in zip(p, q, strict=True))
    assert math.isclose(float(kl.detach()), manual, abs_tol=1e-6)
    kl.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[:, 1]).item() == 0
    kl_probe = {"expected_hand_formula": "4 sum_i pT_i log(pT_i/qT_i), pT=normalize(sqrt(p)), qT=normalize(sqrt(q))",
                "manual": manual, "executed": float(kl.detach()), "valid_positions": 1,
                "ignored_position_gradient": student.grad[:, 1].tolist(), "teacher_gradient": None}
    # A string can be encoded without sharing a teacher vocabulary.
    tokenizers = {"byte": ByteTokenizer().encode("4"), "char": CharTokenizer("345").encode("4")}
    assert tokenizers["byte"] != tokenizers["char"]
    original = types.ModuleType("scripts.course_experiments.fact_v2_18_01_original")
    original.__package__ = "scripts.course_experiments"
    source = OUT / (PREFIX + "run_compression.txt")
    exec(compile(source.read_text(), str(source), "exec"), original.__dict__)
    record = {"question": "2+2=?", "answer": "4", "family": "2+2"}
    smoke = {}
    for version, module in (("historical", original), ("current", compression)):
        seed_everything(42)
        teacher_model = TinyLM(ModelConfig(width=8, layers=1, max_length=32))
        teacher_model.eval().requires_grad_(False)
        cache, _ = module._cache_text(teacher_model, [record], "cpu")
        seed_everything(42)
        initial = TinyLM(ModelConfig(width=8, layers=1, max_length=32))
        for method in ("ce", "ce_kl"):
            seed_everything(42)
            model = TinyLM(initial.config)
            model.load_state_dict(initial.state_dict())
            ctx = SimpleNamespace(seed=42, device="cpu", output=TMP)
            _, _, training = module._fit_text(ctx, model, [record],
                                               version + "-" + method, steps=1,
                                               teacher_cache=cache if method == "ce_kl" else None)
            assert training["effective_supervised_tokens"] == 32  # 16 draws × (one byte + EOS)
            assert training["weights_changed"]
            smoke[version + "_" + method] = training
    report_path = ROOT / "docs/course-experiments/results/distillation.json"
    report = json.loads(report_path.read_text())
    raw_path = ROOT / "outputs/course-control/37060809775/result.json"
    raw = json.loads(raw_path.read_text())
    assert raw["results"] == report["results"]
    tasks = report["results"]["tasks"]
    manifest = json.loads((OUT / (PREFIX + "export-manifest.json")).read_text())
    official_export = {entry["output"]: entry for entry in manifest["files"]}
    teacher_receipts, evaluated, data_receipts = {}, {}, {}
    tok = ByteTokenizer()
    for identifier, key in (("sft", "attributes"), ("style", "style_transfer"), ("moe", "moe_to_dense")):
        checkpoint = TMP / (identifier + "-teacher.pt")
        model, payload = load_checkpoint(checkpoint, "cpu")
        assert sha(checkpoint) == official_export[checkpoint.name]["sha256"]
        assert payload["config"] == tasks[key]["teacher_provenance"]["config"]
        original_artifact = next(a for a in report["artifacts"] if a["path"] == checkpoint.name)
        assert official_export[checkpoint.name]["source_sha256"] == original_artifact["sha256"]
        tensors = tensor_receipts(payload["model"])
        assert all(v["finite"] for v in tensors.values())
        teacher_receipts[identifier] = {"downloaded_export_sha256": sha(checkpoint),
                                        "export_source_sha256": official_export[checkpoint.name]["source_sha256"],
                                        "historical_teacher_provenance": tasks[key]["teacher_provenance"],
                                        "export_metadata": payload.get("metadata"),
                                        "export_has_training_step": "step" in payload,
                                        "config": payload["config"], "tensor_receipts": tensors,
                                        "parameters": sum(v.numel() for v in payload["model"].values()),
                                        "scope": "Actually loaded public inference export; historical source checkpoint file itself not loaded in this audit."}
        for split in ("validation", "test"):
            historical = tasks[key]["teacher_" + split]
            hits = verify_samples(historical["generated_samples"], tok)
            assert hits == historical["correct"]
            assert len(historical["generated_samples"]) == historical["examples"]
        if identifier == "moe":
            logits = model(torch.tensor([[1, 107, 113]]))["logits"]
            assert logits.shape == (1, 3, 264) and torch.isfinite(logits).all()
            teacher_receipts[identifier]["cpu_forward_shape"] = list(logits.shape)
            continue
        path = ROOT / "outputs/text-behavior-interface-check" / identifier / "dataset.json"
        assert sha(path) == tasks[key]["data"]["sha256"]
        parts = json.loads(path.read_text())
        save(identifier + "_dataset.json", parts)
        families = {split: {row["family"] for row in rows} for split, rows in parts.items()}
        assert not (families["train"] & families["test"] or families["train"] & families["validation"]
                    or families["test"] & families["validation"])
        data_receipts[identifier] = {"sha256": sha(path), "counts": {s: len(v) for s, v in parts.items()},
                                      "families": {s: len(v) for s, v in families.items()},
                                      "family_intersections": 0}
        tokens = 24 if identifier == "sft" else 96
        evaluated[identifier] = {}
        for split in ("validation", "test"):
            result = compression._evaluate(model, parts[split], "cpu", tokens)
            historical = tasks[key]["teacher_" + split]
            assert result["examples"] == historical["examples"]
            assert result["supervised_tokens"] == historical["supervised_tokens"]
            assert result["correct"] == historical["correct"]
            assert [x["generated_ids"] for x in result["generated_samples"]] == [
                x["generated_ids"] for x in historical["generated_samples"]]
            evaluated[identifier][split] = result
        if identifier == "style":
            scored = compression._style_scores(evaluated[identifier]["test"], parts["test"])
            assert scored["content_correct"] == tasks[key]["teacher_style"]["content_correct"] == 3
            evaluated[identifier]["style"] = scored
        else:
            cache_path = ROOT / "outputs/private-review-artifacts/original-distillation/sft-teacher-logits.pt"
            recorded_cache = torch.load(cache_path, weights_only=True, map_location="cpu")
            expected_cache_hash = next(a["sha256"] for a in report["artifacts"]
                                       if a["path"] == "sft-teacher-logits.pt")
            assert sha(cache_path) == expected_cache_hash
            assert recorded_cache["teacher_sha256"] == tasks[key]["teacher_provenance"]["sha256"]
            recomputed, _ = compression._cache_text(model, parts["train"], "cpu")
            diffs = [float((a - b).abs().max()) for a, b in zip(recomputed, recorded_cache["logits"], strict=True)]
            assert max(diffs) < 1e-4
            teacher_receipts[identifier]["original_cache"] = {"sha256": sha(cache_path), "records": len(diffs),
                                                              "max_abs_difference_by_record": diffs,
                                                              "bound": 1e-4, "teacher_sha256": recorded_cache["teacher_sha256"]}
    figure = ROOT / "course/figures/architecture_vocab_alignment.svg"
    tree = ET.parse(figure)
    labels_svg = [x.text for x in tree.iter() if x.tag.endswith("text")]
    assert "ID0：貓 = 0.8" in labels_svg and "ID1：貓 = 0.8" in labels_svg
    snapshot = {"environment": env, "source_sha256": hashlib.sha256(lesson.encode()).hexdigest(),
                "guide_sha256": sha(ROOT / "docs/technical-review-guide.md"),
                "snippet_stdout": output.getvalue(), "numeric": numeric, "kl_probe": kl_probe,
                "different_tokenizers_example": tokenizers, "one_step_training": smoke,
                "published_report_sha256": sha(report_path), "raw_report_sha256": sha(raw_path),
                "historical_and_published_results_equal": True,
                "historical_environment": {key: report[key] for key in
                                            ("revision", "device", "seed", "torch_version", "python_version", "gpu",
                                             "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
                "teacher_receipts": teacher_receipts, "data_receipts": data_receipts,
                "cpu_evaluation": evaluated,
                "historical_teacher_evaluation": {key: {s: value["teacher_" + s] for s in ("validation", "test")}
                                                   for key, value in tasks.items()},
                "figure": {"sha256": sha(figure), "labels": labels_svg,
                           "renderer": subprocess.check_output(["inkscape", "--version"], text=True).strip(),
                           "render_sha256": sha(OUT / (PREFIX + "vocab_alignment.png")),
                           "inspection": "Reviewer viewed rendered PNG: green cat and orange dog connect matching identities; cross at center; labels/probabilities/order match 18.6. No figure is embedded in 18.1."}}
    save("audit.json", snapshot)
    print(json.dumps({"result": "all bounded checks passed", "environment": env,
                      "snippet": output.getvalue(), "manual_kl": manual,
                      "sft_test": evaluated["sft"]["test"]["correct"],
                      "style_content": evaluated["style"]["style"]["content_correct"],
                      "cache_max_abs": max(teacher_receipts["sft"]["original_cache"]["max_abs_difference_by_record"])
                      }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
