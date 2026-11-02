"""T.4 fresh review: short CPU checks; no optimizer update or research run."""

import contextlib
import datetime
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import evaluate as evaluation
from scripts import train as training_cli
from scripts.course_experiments import run as experiment_cli
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, load_jsonl, render_chat, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum
from tiny_perceptron.tokenization import load_tokenizer
from tiny_perceptron.training import load_checkpoint, seed_everything


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v8/sources/T.4.md"
torch.set_num_threads(1)
tok = ByteTokenizer()
report = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "reviewer_task": "/root/omni_p6_technical_t4",
    "scope": "Artificial CPU mechanisms, exact saved inline baseline, CLI parsing/help, and preserved JSON arithmetic only. No optimizer.step(), formal training, teacher forward, GPU job, download of weights, or publication.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "interpreter": sys.executable, "platform": platform.platform(), "threads": "1", "PYTHONPATH": os.environ.get("PYTHONPATH", "unset")},
    "source_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd):
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
    result = {"command": shlex.join(str(x) for x in command), "cwd": str(cwd), "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    assert process.returncode == 0, result
    return result


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


raw = FROZEN.read_text()
inline = re.findall(r"```python\n(.*?)```", raw, re.S)
assert len(inline) == 1 and (HERE / "save_start.py").read_text() == inline[0]
report["inline_python"] = {"path": str((HERE / "save_start.py").relative_to(ROOT)), "sha256": digest(HERE / "save_start.py"), "exact_frozen_block": True}

with tempfile.TemporaryDirectory(prefix="t4-cpu-") as temporary:
    work = Path(temporary)
    (work / "outputs").mkdir()
    (work / "checkpoints").mkdir()
    (work / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    (work / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    (work / "outputs/save_start.py").write_text(inline[0])
    commands = []
    for kind in ("toy-text", "attributes-sft"):
        command = [".venv/bin/python", "scripts/prepare_data.py", "--kind", kind, "--seed", "42"]
        first = run(command, work)
        paths = sorted((work / "data/generated" / kind).glob("*"))
        before = {p.name: digest(p) for p in paths}
        run(command, work)
        after = {p.name: digest(p) for p in paths}
        assert before == after
        manifests = json.loads((work / "data/generated" / kind / "manifest.json").read_text())
        rows = {split: load_jsonl(work / "data/generated" / kind / f"{split}.jsonl") for split in ("train", "validation", "test")}
        families = {split: sorted({r["family"] for r in records}) for split, records in rows.items()}
        assert not set(families["train"]) & set(families["validation"])
        assert not set(families["train"]) & set(families["test"])
        assert not set(families["validation"]) & set(families["test"])
        assert [len(rows[x]) for x in rows] == ([9, 1, 2] if kind == "toy-text" else [45, 5, 10])
        if kind == "attributes-sft":
            assert all(sum(row["family"] == f for row in rows[s]) == 5 for s in rows for f in families[s])
        report[kind] = {"execution": first, "manifest": manifests, "families": families, "identical_repeat_hashes": before}
        (HERE / f"{kind}-validation.jsonl").write_bytes((work / "data/generated" / kind / "validation.jsonl").read_bytes())
        commands.append(first["command"])

    first_start = run([".venv/bin/python", "outputs/save_start.py"], work)
    payload = torch.load(work / "checkpoints/start.pt", map_location="cpu", weights_only=True)
    assert payload["optimizer"] is None and payload["step"] == 0
    assert {k: payload["config"][k] for k in ("width", "layers", "heads", "max_length")} == {"width": 32, "layers": 1, "heads": 1, "max_length": 128}
    start_hash = state_digest(payload["model"])
    second_start = run([".venv/bin/python", "outputs/save_start.py"], work)
    second_payload = torch.load(work / "checkpoints/start.pt", map_location="cpu", weights_only=True)
    assert state_digest(second_payload["model"]) == start_hash
    report["inline_python"].update(execution=first_start, identical_repeat_state_sha256=start_hash, optimizer=payload["optimizer"], step=payload["step"], config=payload["config"], parameters=sum(v.numel() for v in payload["model"].values()), checkpoint_sha256=digest(work / "checkpoints/start.pt"), checkpoint_retention="Temporary small baseline deleted after inspection; exact source and persistent receipt regenerate it.")

    evaluations = []
    for kind, mode, tokens, output in (("toy-text", "text", "32", "text-before.json"), ("attributes-sft", "sft", "24", "attributes-before.json")):
        command = [".venv/bin/python", "scripts/evaluate.py", "checkpoints/start.pt", "--data", f"data/generated/{kind}/validation.jsonl", "--mode", mode, "--tokens", tokens, "--device", "cpu", "--limit", "all", "--output", f"outputs/{output}"]
        execution = run(command, work)
        result = json.loads((work / "outputs" / output).read_text())
        (HERE / output).write_bytes((work / "outputs" / output).read_bytes())
        rows = load_jsonl(work / "data/generated" / kind / "validation.jsonl")
        expected_count = sum(len(tok.encode(r["text"])) + 1 for r in rows) if mode == "text" else sum(int((render_chat(r["messages"])[1] != -100).sum()) for r in rows)
        assert result["effective_tokens"] == expected_count
        assert result["records_read"] == result["records_selected"] == len(rows)
        assert result["generation_evaluated_records"] == len(rows) and result["skipped"] == []
        assert result["exact_match"] is None if mode == "text" else result["metric_denominators"]["exact_match"] == 5
        assert [x["row"] for x in result["samples"]] == list(range(len(rows)))
        assert all(len(s["generated_ids"]) <= int(tokens) for s in result["samples"])
        evaluations.append({"execution": execution, "output": output, "independent_expected_effective_tokens": expected_count, "observed": {k: result[k] for k in ("mean_token_nll", "effective_tokens", "records_read", "records_selected", "metric_denominators", "skipped")}})
    report["actual_untrained_before_evaluations"] = evaluations

    class StopBeforeForward(Exception):
        pass

    optimizer_events = []

    class GuardOptimizer:
        def __init__(self, parameters, lr):
            self.parameters = list(parameters)
            self.param_groups = [{"lr": lr}]
            optimizer_events.append("new_optimizer")

        def zero_grad(self, **kwargs):
            optimizer_events.append("stop_before_forward_or_update")
            raise StopBeforeForward()

        def load_state_dict(self, state):
            optimizer_events.append("loaded_parent_optimizer")

    # Artificial parent tests only control flow. This is not a trained text.pt.
    parent = dict(payload)
    parent.update(step=200, optimizer={"artificial_marker": "must_be_ignored_for_new_stage"})
    torch.save(parent, work / "checkpoints/artificial-text.pt")
    args = ["scripts/train.py", "--task", "sft", "--data", str(work / "data/generated/attributes-sft/train.jsonl"), "--checkpoint", str(work / "checkpoints/artificial-text.pt"), "--seed", "42", "--device", "cpu", "--train", "--steps", "500", "--output", str(work / "checkpoints/never-written.pt")]
    with patch.object(sys, "argv", args), patch.object(torch.optim, "AdamW", GuardOptimizer):
        try:
            training_cli.main()
        except StopBeforeForward:
            pass
        else:
            raise AssertionError("guard was not reached")
    assert optimizer_events == ["new_optimizer", "stop_before_forward_or_update"]
    assert not (work / "checkpoints/never-written.pt").exists()
    resume_args = args.copy()
    resume_args[resume_args.index(str(work / "checkpoints/artificial-text.pt"))] = str(work / "checkpoints/start.pt")
    resume_args.append("--resume")
    with patch.object(sys, "argv", resume_args), patch.object(torch.optim, "AdamW", GuardOptimizer):
        try:
            training_cli.main()
        except ValueError as error:
            assert "optimizer" in str(error)
            resume_error = str(error)
        else:
            raise AssertionError("resume must reject baseline without optimizer")
    report["new_stage_control_flow"] = {"artificial_parent_step": 200, "events_without_resume": optimizer_events[:2], "baseline_resume_error": resume_error, "scope": "Guard optimizer stops before forward, backward, optimizer updates or output checkpoint. Does not execute 200/500-step recipes."}

example = next(r for r in generate_records("attributes-sft") if r["messages"][0]["content"] == "color=red;shape=circle;pitch=low;shape?")
x, y = render_chat(example["messages"])
expected = tok.encode("circle") + [tok.eos_id]
actual = y[y != -100].tolist()
assert actual == expected and len(actual) == 7
uniform_logits = torch.zeros(1, len(y), tok.vocab_size)
nll_sum, count = loss_sum(uniform_logits, y[None])
expected_nll = torch.log(torch.tensor(264.0)).item()
assert count.item() == 7 and abs((nll_sum / count).item() - expected_nll) < 1e-6
report["sft_mask"] = {"record": example, "input_ids": x.tolist(), "labels": y.tolist(), "effective_target_ids": actual, "effective_targets": int(count.item()), "uniform_mean_nll": (nll_sum / count).item(), "expected_log_264": expected_nll, "tolerance": 1e-6, "dtype": str(x.dtype)}
report["utf8"] = [{"text": s, "codepoints": len(s), "bytes": list(s.encode()), "token_ids": tok.encode(s)} for s in ("中", "貓", "A", "🙂")]

cases = {
    "correct_eos": tok.encode("circle") + [tok.eos_id],
    "correct_no_eos": tok.encode("circle"),
    "leading_space": tok.encode(" circle") + [tok.eos_id],
    "illegal_role": [tok.user_id] + tok.encode("circle") + [tok.eos_id],
}
case_results = {k: evaluation.answer_sample(tok, ids, "circle") for k, ids in cases.items()}
assert case_results["correct_eos"]["completed_exact_match"] is True
assert case_results["correct_no_eos"]["exact_match"] is True and case_results["correct_no_eos"]["completed_exact_match"] is False
assert case_results["leading_space"]["exact_match"] is False
assert case_results["illegal_role"]["exact_match"] is False and case_results["illegal_role"]["invalid_special_tokens"]
report["answer_comparisons"] = case_results

class ArtificialLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=8)

    def forward(self, ids, cache=None):
        logits = torch.zeros(ids.shape[0], ids.shape[1], tok.vocab_size)
        logits[:, -1, tok.encode("A")[0] if ids.shape[1] == 1 else tok.eos_id] = 10.0
        return {"logits": logits, "cache": None}

with patch.object(torch, "multinomial", side_effect=AssertionError("default generation sampled")):
    greedy = generate(ArtificialLogits(), torch.tensor([[tok.bos_id]]), max_new_tokens=5)
    budget = generate(ArtificialLogits(), torch.tensor([[tok.bos_id]]), max_new_tokens=1)
assert greedy.tolist() == [[tok.bos_id, tok.encode("A")[0], tok.eos_id]]
assert budget.tolist() == [[tok.bos_id, tok.encode("A")[0]]]
report["greedy_mechanism"] = {"artificial_logits_argmax_sequence": greedy.tolist(), "one_new_token_budget": budget.tolist(), "sampling_called": False, "scope": "Deterministic fabricated logits test; no trained-model capability inference."}

try:
    load_tokenizer(None, 512)
except ValueError as error:
    assert "--tokenizer" in str(error)
    report["bpe_missing_tokenizer_rejection"] = str(error)
else:
    raise AssertionError("non-byte vocabulary accepted without tokenizer")
assert isinstance(load_tokenizer(None, 264), ByteTokenizer)

seed_everything(42)
uniform_model = TinyLM(ModelConfig())
with torch.no_grad():
    for parameter in uniform_model.parameters():
        parameter.zero_()
records = [example, {"messages": [{"role": "user", "content": "x" * 128}, {"role": "assistant", "content": "circle"}]}]
skip_report = evaluation.evaluate(uniform_model, records, "sft", max_new_tokens=2)
assert skip_report["records_selected"] == 2 and skip_report["generation_evaluated_records"] == 1
assert skip_report["effective_tokens"] == 7 and skip_report["samples"][0]["row"] == 0
assert skip_report["skipped"][0]["row"] == 1 and skip_report["metric_denominators"]["exact_match"] == 1
assert abs(skip_report["mean_token_nll"] - expected_nll) < 1e-6
report["skip_and_denominator_mechanism"] = skip_report

helps = []
for command in ([str(ROOT / ".venv/bin/python"), "scripts/prepare_data.py", "--help"], [str(ROOT / ".venv/bin/python"), "scripts/train.py", "--help"], [str(ROOT / ".venv/bin/python"), "scripts/evaluate.py", "--help"], [str(ROOT / ".venv/bin/python"), "scripts/fetch_training_assets.py", "--help"], [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"], [str(ROOT / ".venv/bin/python"), "scripts/infer.py", "--help"]):
    helps.append(run(command, ROOT))
assert "temperature" not in helps[2]["stdout"] and "--tokenizer" in helps[5]["stdout"]
report["cli_help"] = helps

command_coverage = []
for block in re.findall(r"```bash\n(.*?)```", raw, re.S):
    for line in block.strip().splitlines():
        parts = shlex.split(line)
        entry = {"original_command": line}
        if len(parts) > 1 and parts[1] == "scripts/train.py":
            args = training_cli.parser().parse_args(parts[2:])
            entry.update(method="actual parser; control-flow guard separately checked", parsed={"task": args.task, "steps": args.steps, "checkpoint": str(args.checkpoint), "output": str(args.output), "train": args.train, "resume": args.resume})
        elif len(parts) > 1 and parts[1] == "scripts/evaluate.py":
            class ParsedBeforeLoad(Exception):
                pass
            def stop_at_load(path, device):
                entry.update(parsed_checkpoint=str(path), parsed_device=str(device))
                raise ParsedBeforeLoad()
            with patch.object(sys, "argv", parts[1:]), patch.object(evaluation, "load_checkpoint", stop_at_load):
                try:
                    evaluation.main()
                except ParsedBeforeLoad:
                    pass
                else:
                    raise AssertionError("parse guard not reached")
            entry.update(method="actual main argparse; stopped at checkpoint load; before commands also executed above")
        elif "--experiment" in parts:
            identifier = parts[parts.index("--experiment") + 1]
            spec = experiment_cli.experiment_spec(identifier)
            entry.update(method="entrypoint help, actual experiment_spec/list_assets, source inspection; no experiment executed", experiment=identifier, function=spec["function"], dependencies=spec["dependencies"], assets=experiment_cli.list_assets(identifier))
        elif "--kind" in parts:
            entry.update(method="exact command executed twice in isolated working directory above")
        elif "--asset" in parts:
            manifest = json.loads((ROOT / "assets/training/manifest.json").read_text())
            selected = [parts[i + 1] for i, item in enumerate(parts) if item == "--asset"]
            assert all(any(a["id"] == item for a in manifest["assets"]) for item in selected)
            entry.update(method="entrypoint help and asset manifest/source inspected; no network fetch", assets=selected)
        command_coverage.append(entry)
report["all_fenced_commands"] = command_coverage
assert len(command_coverage) == 17
if not torch.cuda.is_available():
    try:
        experiment_cli.execute("text_foundation", "cuda", HERE / "never-created", HERE, ROOT / "assets/training")
    except RuntimeError as error:
        assert "must not silently fall back" in str(error)
        report["explicit_cuda_rejection"] = str(error)
    else:
        raise AssertionError("CUDA fallback occurred")
assert not (HERE / "never-created").exists()

recorded = []
expected_rounded = {"text_foundation": "16.0", "real_text": "23.9", "tokenizer": "16.2", "sft": "31.7", "sft_ablation": "23.5"}
for identifier, expected_seconds in expected_rounded.items():
    path = ROOT / f"docs/course-experiments/results/{identifier}.json"
    result = json.loads(path.read_text())
    assert result["gpu"] == "NVIDIA L4" and result["device"] == "cuda" and result["step_scale"] == 1
    assert f"{result['elapsed_seconds']:.1f}" == expected_seconds
    branches = []
    def walk(value, locator="results"):
        if isinstance(value, dict):
            if "steps" in value and "checkpoint" in value:
                branches.append({"locator": locator, **{k: value[k] for k in ("steps", "seconds", "checkpoint", "records", "effective_tokens", "parameters") if k in value}})
            for k, v in value.items():
                if k not in ("samples", "history", "artifacts"):
                    walk(v, locator + "." + k)
        elif isinstance(value, list):
            for index, v in enumerate(value):
                walk(v, locator + f"[{index}]")
    walk(result["results"])
    recorded.append({"input_path": str(path.relative_to(ROOT)), "sha256": digest(path), "revision": result["revision"], "recorded_environment": {k: result[k] for k in ("python_version", "torch_version", "device", "gpu", "seed")}, "modal": result["modal"], "elapsed_seconds": result["elapsed_seconds"], "rounded_seconds": f"{result['elapsed_seconds']:.1f}", "timing_scope": result["timing_scope"], "step_scale": result["step_scale"], "branches": branches, "data": result["results"].get("data"), "scope": "Recalculation of preserved original JSON; no rerun or independent timing measurement."})
report["recorded_experiment_recalculation"] = recorded

report["torch_original_source_inspection"] = []
for name, function in (("manual_seed", torch.manual_seed), ("cross_entropy", F.cross_entropy)):
    path = Path(inspect.getsourcefile(function))
    lines, first = inspect.getsourcelines(function)
    needles = ("Sets the seed", "torch.cuda.manual_seed_all", "default_generator.manual_seed") if name == "manual_seed" else ("ignore_index (int)", "reduction (str)", "cross_entropy_loss(", "ignore_index,", "_Reduction.get_enum")
    excerpts = [{"line": first + index, "text": line.rstrip()} for index, line in enumerate(lines) if any(needle in line for needle in needles)]
    report["torch_original_source_inspection"].append({"api": name, "path": str(path), "sha256": digest(path), "first_line": first, "small_source_excerpts": excerpts, "inspection_note": "Original installed package function read; only directly relevant lines retained, not the full third-party source."})
report["completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()
(HERE / "probe-output.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "all assertions passed", "fenced_commands": len(command_coverage), "inline_exact": True, "recorded_experiments": len(recorded), "output": str((HERE / 'probe-output.json').relative_to(ROOT)), "completed_at": report["completed_at"]}, ensure_ascii=False))
