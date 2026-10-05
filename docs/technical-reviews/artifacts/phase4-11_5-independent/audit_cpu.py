"""Independent bounded audit: no fitting, model loading, generation or tensor save."""
import ast
import contextlib
import hashlib
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene
from scripts.course_experiments.modalities import _freeze

BASE = Path(__file__).parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
report = {"environment": {"python": sys.version, "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "scope": "Only constructor/flag inspection, raw JSON arithmetic, one short backward; no optimizer.step, saved tensors, checkpoints or training."}
code = (BASE / "fence-1.py").read_text()
report["short_fence_variations"] = []
for layers in (1, 2):
    program = code if layers == 1 else code.replace("ModelConfig(width=8)", "ModelConfig(width=8, layers=2)")
    ns, stdout = {}, io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(program, f"11.5-original-fence-layers-{layers}", "exec"), ns)
    model = ns["model"]
    counts, names = {}, {}
    for mode in ("all", "partial", "projector"):
        model.requires_grad_(mode == "all")
        model.image_projector.requires_grad_(True)
        if mode == "partial":
            model.language.blocks[-1].requires_grad_(True)
        counts[mode] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        names[mode] = [name for name, p in model.named_parameters() if p.requires_grad]
    assert counts == {"projector": 136, "partial": 976, "all": 8296 + (layers - 1) * 840}
    partial_blocks = {name.split(".")[2] for name in names["partial"] if name.startswith("language.blocks.")}
    assert partial_blocks == {str(layers - 1)}
    report["short_fence_variations"].append({"layers": layers, "forward_order_stdout": stdout.getvalue(),
        "reverse_order_counts": counts, "trainable_names": names})

# Extract the actual CLI statements from the current AST, without invoking main.
cli_path = ROOT / "scripts/train.py"
tree = ast.parse(cli_path.read_text())
cli_freeze = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
    and ast.unparse(n.test) == "args.freeze != 'none'")
cli_program = ast.Module(body=[cli_freeze], type_ignores=[])
ast.fix_missing_locations(cli_program)
report["cli_ast_contract"] = {"path": "scripts/train.py", "line": cli_freeze.lineno,
    "code": ast.unparse(cli_program), "cases": []}
for layers in (1, 2, 3):
    m = MultiModalLM(TinyLM(ModelConfig(width=8, layers=layers)))
    exec(compile(cli_program, "scripts/train.py:selected-freeze-AST", "exec"),
        {"args": SimpleNamespace(freeze="partial"), "train_model": m})
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    blocks = sorted({n.split(".")[2] for n in names if n.startswith("language.blocks.")})
    assert blocks == sorted({"0", str(layers - 1)})
    assert all(not p.requires_grad for p in m.vision.parameters())
    assert all(not p.requires_grad for p in m.audio.parameters())
    assert all(p.requires_grad for p in m.image_projector.parameters())
    assert all(p.requires_grad for p in m.audio_projector.parameters())
    report["cli_ast_contract"]["cases"].append({"layers": layers, "block_indices": blocks,
        "trainable_count": sum(p.numel() for p in m.parameters() if p.requires_grad), "trainable_names": names})

# A fresh random model, only to check the unconnected audio branch; no update.
torch.manual_seed(13)
m = MultiModalLM(TinyLM(ModelConfig(width=8)))
tok = ByteTokenizer()
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
answer = tok.encode("red") + [tok.eos_id]
out = m(torch.tensor(prefix + answer), torch.tensor([-100] * len(prefix) + answer), image=scene())
masked_loss(out["logits"], out["labels"]).backward()
audio_grads = {n: p.grad is None for n, p in m.named_parameters() if n.startswith(("audio.", "audio_projector."))}
assert all(audio_grads.values())
assert m.image_projector.weight.grad is not None
report["unused_branch_backward"] = {"valid_answer_positions": int((out["labels"] != -100).sum()),
    "audio_grad_is_none": audio_grads, "image_projector_has_grad": True, "optimizer_step": False}

# Only explicit measurement/config/provenance pointers from the historical JSON.
raw_path = ROOT / "docs/course-experiments/results/vqa.json"
raw = raw_path.read_bytes()
original = json.loads(raw)
pointers = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/step_scale"]
selected = {p: original[p[1:]] for p in pointers}
for path in ("tiny_perceptron/model.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py", "scripts/course_experiments/modalities.py"):
    pointer = "/code_sha256/" + path.replace("~", "~0").replace("/", "~1")
    value = original["code_sha256"][path]
    assert value == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    selected[pointer] = value
    pointers.append(pointer)
splits = {}
for side in ("train", "validation", "test"):
    value = original["results"]["data"]["splits"][side]
    for key in ("count", "sha256", "records"):
        pointer = f"/results/data/splits/{side}/{key}"
        selected[pointer] = value[key]
        pointers.append(pointer)
    records = value["records"]
    assert len(records) == value["count"]
    assert hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == value["sha256"]
    splits[side] = records
families = {side: {row["family"] for row in rows} for side, rows in splits.items()}
assert not families["train"] & families["validation"]
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]
assert {r["offset"] for r in splits["validation"]} == {1}
assert {r["offset"] for r in splits["test"]} == {2}
report["raw_file"] = {"path": raw_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
    "bytes": len(raw), "inspection_policy": "Exact listed pointers only. Did not inspect scope, notes, review, or scope corrections."}
report["empirical_recomputation"] = []
expected = {"projector_only": (1088, 3, 3), "partial": (50816, 10, 12), "all": (145664, 12, 9)}
for name, (param_count, v_correct, t_correct) in expected.items():
    variant = original["results"]["variants"][name]
    training = variant["training"]
    for key in ("parameters", "trainable_parameters", "config", "modal_config", "steps", "effective_tokens", "effective_targets", "weights_changed", "nonzero_gradient_seen"):
        pointer = f"/results/variants/{name}/training/{key}"
        selected[pointer] = training[key]
        pointers.append(pointer)
    history_targets = []
    for i, row in enumerate(training["history"]):
        for key in ("step", "effective_targets"):
            pointer = f"/results/variants/{name}/training/history/{i}/{key}"
            selected[pointer] = row[key]
            pointers.append(pointer)
        history_targets.append(row["effective_targets"])
    assert len(history_targets) == training["steps"] == 160
    assert sum(history_targets) == training["effective_tokens"] == training["effective_targets"] == 3830
    reconstructed_targets = []
    for step in range(160):
        rng = random.Random(original["seed"] + step)
        batch_targets = 0
        for _ in range(4):
            # run_vqa supplies the nonempty replay list even at ratio zero.
            # The guard consumes one rng.random() draw before the row choice.
            rng.random()
            batch_targets += len(tok.encode(rng.choice(splits["train"])["answer"])) + 1
        reconstructed_targets.append(batch_targets)
    assert reconstructed_targets == history_targets
    modal_cfg = training["modal_config"]
    fresh = MultiModalLM(TinyLM(ModelConfig(**training["config"])),
        vision_width=modal_cfg["vision_width"], audio_width=modal_cfg["audio_width"])
    _freeze(fresh, "projector" if name == "projector_only" else name)
    assert sum(p.numel() for p in fresh.parameters() if p.requires_grad) == param_count == training["trainable_parameters"]
    assert sum(p.numel() for p in fresh.parameters()) == training["parameters"] == 145664
    row_result = {"variant": name, "parameter_count": param_count, "effective_answer_targets": sum(history_targets),
        "steps": len(history_targets), "reconstructed_sampling_targets_match": True, "evaluation": {}}
    for side, expected_correct in (("validation", v_correct), ("test", t_correct)):
        evaluation = variant[side]
        for key in ("examples", "correct", "exact_match", "effective_tokens", "samples", "skipped"):
            pointer = f"/results/variants/{name}/{side}/{key}"
            selected[pointer] = evaluation[key]
            pointers.append(pointer)
        samples = evaluation["samples"]
        assert len(samples) == evaluation["examples"] == len(splits[side]) == 12
        computed = 0
        eos_count = 0
        for record, sample in zip(splits[side], samples, strict=True):
            assert sample["family"] == record["family"] and sample["question"] == record["question"]
            assert sample["target"] == record["answer"]
            ids = sample["generated_ids"]
            trimmed = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact = trimmed == tok.encode(record["answer"])
            assert exact == sample["exact_match"]
            assert tok.decode(trimmed) == sample["generated"]
            computed += exact
            eos_count += tok.eos_id in ids
        effective = sum(len(tok.encode(row["answer"])) + 1 for row in splits[side])
        assert effective == evaluation["effective_tokens"]
        assert computed == evaluation["correct"] == expected_correct
        assert evaluation["exact_match"] == computed / 12
        assert not evaluation["skipped"]
        row_result["evaluation"][side] = {"correct": computed, "denominator": len(samples),
            "effective_answer_targets": effective, "eos_count": eos_count, "score_criterion": "exact byte token answer excluding EOS; EOS reported separately"}
    report["empirical_recomputation"].append(row_result)
report["split_summary"] = {side: {"questions": len(rows), "image_families": len(families[side]),
    "offsets": sorted({r["offset"] for r in rows})} for side, rows in splits.items()}
report["inspected_json_pointers"] = pointers
(BASE / "raw-selected-pointers.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
(BASE / "audit-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"environment": report["environment"], "short_counts": [x["forward_order_stdout"] for x in report["short_fence_variations"]],
    "cli_blocks": [{"layers": x["layers"], "blocks": x["block_indices"], "count": x["trainable_count"]} for x in report["cli_ast_contract"]["cases"]],
    "raw_sha256": report["raw_file"]["sha256"], "split_summary": report["split_summary"],
    "table": report["empirical_recomputation"], "pointer_count": len(pointers), "result": "all assertions passed"}, ensure_ascii=False, indent=2))
