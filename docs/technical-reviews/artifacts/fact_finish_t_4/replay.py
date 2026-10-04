"""T.4 independent CPU verification: records, denominators, published models; no full training."""

import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import tarfile
from pathlib import Path

import torch

from scripts.course_experiments.common import _nll, records_sha256, split_records, text_examples
from scripts.course_experiments.text import (
    _deduplicate_text,
    _digest,
    _evaluate_tokenizer,
    _utf8_prefix,
    arithmetic_records,
)
from scripts.evaluate import answer_sample, evaluate
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.tokenization import load_tokenizer
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
os_records = {}
results = {}
torch.set_num_threads(2)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected, tolerance=2e-5):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def check_parts(name, parts, manifest):
    info = {}
    families = []
    for split, rows in parts.items():
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        expected = manifest[split]
        assert len(rows) == expected["records"]
        assert digest_bytes(encoded) == expected["sha256"], (name, split)
        members = {str(row["family"]) for row in rows}
        assert len(members) == expected["families"]
        families.append(members)
        info[split] = {"records": len(rows), "families": len(members), "sha256": digest_bytes(encoded)}
    assert not any(a & b for i, a in enumerate(families) for b in families[i + 1 :])
    os_records[name] = parts
    results.setdefault("data", {})[name] = info


def read_asset(identifier, basename):
    manifest = json.loads((ROOT / "assets/training/manifest.json").read_text())
    entry = next(a for a in manifest["assets"] if a["id"] == identifier)
    archive = ROOT / entry["archive"]
    assert archive.stat().st_size == entry["archive_bytes"]
    assert digest_bytes(archive.read_bytes()) == entry["archive_sha256"]
    with tarfile.open(archive) as tf:
        contents = {}
        for f in entry["files"]:
            body = tf.extractfile(f["path"]).read()
            assert len(body) == f["bytes"] and digest_bytes(body) == f["sha256"]
            contents[f["path"]] = body
        raw = contents[next(p for p in contents if p.endswith(basename))]
        rows = [json.loads(line) for line in raw.decode().splitlines() if line]
        if identifier == "tinystories":
            stories = contents["text-initial/tinystories-train-prefix-complete.txt"].decode().split("<|endoftext|>")
            stories = [story.strip() for story in stories if story.strip()]
            assert [r["text"] for r in rows] == stories and len(stories) == 512
        elif identifier == "chinese-poetry":
            source = json.loads(contents["text-initial/tang300-source.json"])
            texts = ["\n".join([row["title"], row["author"], *row["paragraphs"]]) for row in source]
            assert len(source) == 366 and len(set(texts)) == 365
            assert [r["text"] for r in rows] == list(dict.fromkeys(texts))
        for p, body in contents.items():
            if p.endswith(("source-README.md", "chinese-poetry-LICENSE", "chinese-poetry-README.md")):
                (OUT / (identifier + "-" + Path(p).name)).write_bytes(body)
    results.setdefault("assets", {})[identifier] = {
        "archive": entry["archive"],
        "archive_sha256": entry["archive_sha256"],
        "verified_members": len(contents),
        "records": len(rows),
        "data_sha256": digest_bytes(raw),
    }
    return rows


def verify_schedule(rows, mode, training, max_length=128, batch_size=16):
    assert records_sha256(rows) == training["records_sha256"]
    examples = text_examples(rows, mode=mode, max_length=max_length)
    sampler = random.Random(42)
    count = sum(
        int((y != -100).sum()) for _ in range(training["steps"]) for x, y in sampler.choices(examples, k=batch_size)
    )
    assert count == training["effective_tokens"], (count, training["effective_tokens"])
    return {
        "steps": training["steps"],
        "batch_size": batch_size,
        "effective_targets_replayed_without_training": count,
        "corpus_effective_targets": sum(int((y != -100).sum()) for x, y in examples),
        "windows": len(examples),
        "records": len(rows),
        "records_sha256": records_sha256(rows),
    }


def inspect_model(experiment, name):
    directory = ROOT / "checkpoints/course" / experiment
    export = json.loads((directory / "export-manifest.json").read_text())
    binding = next(f for f in export["files"] if f["output"] == name)
    path = directory / name
    assert digest_bytes(path.read_bytes()) == binding["sha256"]
    assert path.stat().st_size == binding["bytes"]
    original = reports[experiment]
    expected = next(f for f in original["artifacts"] if f["path"] == name)
    assert binding["source_sha256"] == expected["sha256"]
    model, payload = load_checkpoint(path, "cpu")
    tensors = []
    for key, tensor in payload["model"].items():
        assert torch.isfinite(tensor).all()
        assert torch.equal(tensor, model.state_dict()[key])
        tensors.append(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": digest_bytes(tensor.contiguous().numpy().tobytes()),
            }
        )
    results.setdefault("models", {})[experiment + "/" + name] = {
        "export_sha256": binding["sha256"],
        "original_training_checkpoint_sha256": binding["source_sha256"],
        "binding": "Downloaded export hash and manifest source hash equal corresponding original report artifact hash.",
        "config": payload["config"],
        "step": payload.get("step"),
        "metadata": payload.get("metadata"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "tensors": tensors,
    }
    for n in ("export-manifest.json", "download-manifest.json"):
        (OUT / f"{experiment}-{n}").write_bytes((directory / n).read_bytes())
    return model, payload


def nll_compare(model, rows, mode, original):
    observed = _nll(model, text_examples(rows, mode, model.config.max_length))
    assert observed["effective_tokens"] == original["effective_tokens"]
    close(observed["nll"], original["nll"])
    return observed


reports = {
    name: json.loads((ROOT / f"docs/course-experiments/results/{name}.json").read_text())
    for name in ("text_foundation", "real_text", "tokenizer", "sft", "sft_ablation")
}
for name, report in reports.items():
    assert report["device"] == "cuda" and report["gpu"] == "NVIDIA L4"
    assert report["seed"] == 42 and report["step_scale"] == 1 and report["evidence_status"] == "complete_run"

    def verify_aggregates(value, location):
        if isinstance(value, dict):
            if "nll_sum" in value:
                close(value["nll_sum"] / value["effective_tokens"], value["nll"], 1e-12)
            if "bpb_including_eos_boundary_targets" in value:
                total = value.get("nll_sum", value.get("mean_token_nll", 0) * value["effective_tokens"])
                close(
                    total / (value["raw_utf8_bytes"] * math.log(2)), value["bpb_including_eos_boundary_targets"], 1e-12
                )
            if "matches" in value:
                assert len(value["samples"]) == value["records"]
                exact = ended = effective = 0
                for sample in value["samples"]:
                    ids = sample["generated_ids"]
                    raw = ids[: ids.index(2)] if 2 in ids else ids
                    correct = raw == ByteTokenizer().encode(sample["expected"])
                    assert correct == sample["exact"]
                    assert (2 in ids) == sample["eos"]
                    exact += correct
                    ended += 2 in ids
                    effective += len(ByteTokenizer().encode(sample["expected"])) + 1
                assert exact == value["matches"] and effective == value["effective_tokens"]
                close(exact / value["records"], value["exact_match"], 1e-12)
                close(ended / value["records"], value["eos_rate"], 1e-12)
                results.setdefault("complete_sft_generation_audits", {})[location] = {
                    "records": value["records"],
                    "effective_targets": effective,
                    "matches": exact,
                    "ended": ended,
                    "all_record_ids_recomputed": True,
                }
            for k, v in value.items():
                verify_aggregates(v, location + "." + k)
        elif isinstance(value, list):
            for index, v in enumerate(value):
                verify_aggregates(v, location + f"[{index}]")

    verify_aggregates(report["results"], name)

short = split_records(generate_records("toy-text"), seed=42)
attributes = split_records(generate_records("attributes-sft"), seed=42)
arithmetic = split_records(arithmetic_records(), seed=42)
check_parts("toy-text", short, reports["text_foundation"]["results"]["data"])
check_parts("attributes", attributes, reports["sft"]["results"]["data"])
check_parts("arithmetic", arithmetic, reports["sft_ablation"]["results"]["data"]["arithmetic"])
raw_stories = read_asset("tinystories", "tinystories-train-512.jsonl")
raw_poetry = read_asset("chinese-poetry", "chinese-classical-train-365.jsonl")
raw_chat = read_asset("ultrachat-sft", "train-first-100.jsonl")
for identifier, raw in [("tinystories", raw_stories), ("chinese-poetry", raw_poetry)]:
    parts = split_records(_deduplicate_text(raw), seed=42)
    original = reports["real_text"]["results"]["runs"][identifier]
    check_parts(identifier, parts, original["data"])
    model, _ = inspect_model("real_text", identifier + ".pt")
    train = verify_schedule(parts["train"], "text", original["training"])
    scores = {"train": _nll(model, text_examples(parts["train"], max_length=128))}
    close(scores["train"]["nll"], original["training"]["final_loss"])
    for split in ("validation", "test"):
        scores[split] = nll_compare(model, parts[split], "text", original["after"][split])
        raw_bytes = sum(len(row["text"].encode()) for row in parts[split])
        assert scores[split]["effective_tokens"] == raw_bytes + len(parts[split])
        scores[split]["bpb"] = scores[split]["nll_sum"] / (raw_bytes * math.log(2))
    results[identifier] = {"training_schedule": train, "cpu_final_scores": scores}

foundation = reports["text_foundation"]["results"]
start, _ = inspect_model("text_foundation", "start.pt")
trained, _ = inspect_model("text_foundation", "model.pt")
results["text_foundation"] = {"schedule": verify_schedule(short["train"], "text", foundation["training"]), "cpu": {}}
for name, model in [("before", start), ("after", trained)]:
    for split in ("train", "validation", "test"):
        observed = _nll(model, text_examples(short[split]))
        expected = (
            foundation["training"]["initial_loss" if name == "before" else "final_loss"]
            if split == "train"
            else foundation[name][split]["nll"]
        )
        close(observed["nll"], expected)
        results["text_foundation"]["cpu"][name + "." + split] = observed
prefix = torch.tensor([[1] + ByteTokenizer().encode("color=blue;shape=circle;")])
new_ids = generate(trained, prefix, 32)[0, prefix.shape[1] :].tolist()
assert ByteTokenizer().decode(new_ids) == "side=right."
results["text_foundation"]["generation"] = {
    "prompt": "color=blue;shape=circle;",
    "ids": new_ids,
    "text": ByteTokenizer().decode(new_ids),
}

text_pretrain = [
    {"text": row["messages"][0]["content"] + row["messages"][1]["content"], "family": row["family"]}
    for row in attributes["train"]
]
sft = reports["sft"]["results"]
results["sft"] = {
    "direct": verify_schedule(attributes["train"], "sft", sft["training"]),
    "pretrain": verify_schedule(text_pretrain, "text", sft["pretrain_then_sft"]["pretraining"]),
    "continuation": verify_schedule(attributes["train"], "sft", sft["pretrain_then_sft"]["sft"]),
    "cpu": {},
}
for filename, key in [("model.pt", "after"), ("pretrain.pt", "before_sft"), ("pretrain-sft.pt", "after_sft")]:
    model, _ = inspect_model("sft", filename)
    original = sft[key] if key == "after" else sft["pretrain_then_sft"][key]
    for split in ("validation", "test"):
        observed = evaluate(model, attributes[split], mode="sft", max_new_tokens=32)
        close(observed["mean_token_nll"], original[split]["nll"])
        assert observed["effective_tokens"] == original[split]["effective_tokens"]
        assert observed["exact_match"] == original[split]["exact_match"]
        assert [s["generated_ids"] for s in observed["samples"]] == [
            s["generated_ids"] for s in original[split]["samples"]
        ]
        results["sft"]["cpu"][filename + "." + split] = observed

chat = []
for index, row in enumerate(raw_chat):
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    assistant = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    chat.append(
        {
            "family": row.get("prompt_id", _digest(user)),
            "source_row": index,
            "source_record_sha256": _digest(row),
            "scope": "first-turn UTF-8-safe excerpt",
            "messages": [
                {"role": "user", "content": _utf8_prefix(user, 120)},
                {"role": "assistant", "content": _utf8_prefix(assistant, 120)},
            ],
        }
    )
chat_parts = split_records(chat, seed=42)
check_parts("ultrachat-excerpts", chat_parts, sft["ultrachat_pilot"]["data"])
results["ultrachat"] = verify_schedule(
    chat_parts["train"], "sft", sft["ultrachat_pilot"]["training"], max_length=256, batch_size=4
)

noisy = copy.deepcopy(attributes["train"])
corruptions = reports["sft_ablation"]["results"]["corruptions"]
for corruption in corruptions:
    row = noisy[corruption["row"]]
    assert row["family"] == corruption["family"] and row["messages"][-1]["content"] == corruption["correct"]
    row["messages"][-1]["content"] = corruption["wrong"]
assert len(corruptions) == 4
ablation = reports["sft_ablation"]["results"]["runs"]
results["sft_ablation"] = {}
for name, rows in [
    ("b-only", arithmetic["train"]),
    ("replay", attributes["train"] + arithmetic["train"]),
    ("clean", attributes["train"]),
    ("noisy", noisy),
]:
    results["sft_ablation"][name] = verify_schedule(rows, "sft", ablation[name]["training"])
assert (
    results["sft_ablation"]["clean"]["effective_targets_replayed_without_training"]
    == results["sft_ablation"]["noisy"]["effective_targets_replayed_without_training"]
)

complete = split_records(_deduplicate_text(raw_stories[:96] + raw_poetry[:96]), seed=42)
parts = {
    split: [
        {**row, "text": _utf8_prefix(row["text"], 256), "complete_text_sha256": digest_bytes(row["text"].encode())}
        for row in rows
    ]
    for split, rows in complete.items()
}
token = reports["tokenizer"]["results"]
check_parts("tokenizer", parts, token["data"])
sampler = random.Random(42)
exposure = [sampler.choices(range(len(parts["train"])), k=8) for _ in range(400)]
assert _digest(exposure) == token["raw_document_schedule_sha256"]
exposed_bytes = sum(len(parts["train"][i]["text"].encode()) for batch in exposure for i in batch)
assert exposed_bytes == 664759
results["tokenizer"] = {"exposed_raw_bytes": exposed_bytes, "schedule_sha256": _digest(exposure), "cpu": {}}
for name in ("byte256", "bpe512"):
    model, payload = inspect_model("tokenizer", name + ".pt")
    tok_path = ROOT / "checkpoints/course/tokenizer" / token["runs"][name]["tokenizer_file"]
    tok = load_tokenizer(tok_path, model.config.vocab_size, payload)
    (OUT / tok_path.name).write_bytes(tok_path.read_bytes())
    for sample in token["roundtrip"]:
        if name == "bpe512":
            assert tok.encode(sample["text"]) == sample["ids"]
        assert tok.decode(tok.encode(sample["text"])) == sample["text"]
        assert all(value >= 8 for value in tok.encode(sample["text"]))
    for split in ("validation", "test"):
        observed = _evaluate_tokenizer(model, parts[split], tok)
        original = token["runs"][name]["after"][split]
        close(observed["mean_token_nll"], original["mean_token_nll"])
        assert observed["effective_tokens"] == original["effective_tokens"]
        assert observed["raw_utf8_bytes"] == original["raw_utf8_bytes"]
        assert observed["samples"] == original["samples"]
        results["tokenizer"]["cpu"][name + "." + split] = observed

seed_everything(42)
minimal = TinyLM(ModelConfig(width=32))
row = {"messages": [{"role": "user", "content": "shape?"}, {"role": "assistant", "content": "circle"}]}
mini = evaluate(minimal, [row], mode="sft", max_new_tokens=2)
assert mini["mean_token_nll"] == 5.97386714390346 and mini["effective_tokens"] == 7
assert mini["samples"][0]["generated_ids"] == [143, 30]
results["minimal_cpu"] = mini
results["mask"] = {
    "input": render_chat(row["messages"])[0].tolist(),
    "labels": render_chat(row["messages"])[1].tolist(),
}
assert answer_sample(ByteTokenizer(), ByteTokenizer().encode("circle"), "circle")["exact_match"]
assert not answer_sample(ByteTokenizer(), ByteTokenizer().encode("circle"), "circle")["completed_exact_match"]
assert not answer_sample(ByteTokenizer(), ByteTokenizer().encode("circle") + [4, 2], "circle")["exact_match"]
assert not answer_sample(ByteTokenizer(), ByteTokenizer().encode(" circle") + [2], "circle")["exact_match"]
results["answer_contract_checks"] = (
    "Content-only exact, EOS-complete exact, illegal control, leading whitespace checked."
)

seed_everything(42)
small = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
start_path = OUT / "local-start.pt"
save_checkpoint(start_path, small, step=0, metadata={"seed": 42, "note": "untrained baseline"})
results["saved_start"] = {"step": 0, "parameters": sum(p.numel() for p in small.parameters()), "no_optimizer": True}
for mode, rows in [("text", short["validation"]), ("sft", attributes["validation"])]:
    ev = evaluate(small, rows, mode=mode, max_new_tokens=32 if mode == "text" else 24)
    assert not ev["skipped"] and ev["effective_tokens"] == (35 if mode == "text" else 36)
    results["cpu_validation_" + mode] = ev
cli = []
for task in ("text", "sft"):
    proc = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "scripts/train.py",
            "--task",
            task,
            "--checkpoint",
            str(start_path),
            "--device",
            "cpu",
            "--output",
            str(OUT / "should-not-exist.pt"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and not (OUT / "should-not-exist.pt").exists()
    cli.append({"command": proc.args, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
proc = subprocess.run(
    [
        str(ROOT / ".venv/bin/python"),
        "scripts/train.py",
        "--task",
        "text",
        "--train",
        "--steps",
        "0",
        "--device",
        "cpu",
    ],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
assert proc.returncode != 0
cli.append({"command": proc.args, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
results["cli"] = cli
start_path.unlink()
results["environment"] = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "device": "cpu",
    "threads": 2,
}
results["scope"] = (
    "No full training and no GPU timing rerun. Published result records, source/archive hashes, full denominators, sampler exposure and existing export weights evaluated on CPU."
)
(OUT / "execution.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": "pass",
            "data": results["data"],
            "record_audits": len(results["complete_sft_generation_audits"]),
            "models": len(results["models"]),
            "environment": results["environment"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
