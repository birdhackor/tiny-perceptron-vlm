"""核對表格、固定資料的切詞與模型參數；不執行模型訓練。"""

import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scripts.course_experiments.common import split_records
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import _BPE, _deduplicate_text, _utf8_prefix
from tiny_perceptron.data import SPECIALS, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = ROOT / "assets/training/manifest.json"
manifest = json.loads(manifest_path.read_bytes())
asset_receipts = []
raw = []
for identifier, basename in (
    ("tinystories", "tinystories-train-512.jsonl"),
    ("chinese-poetry", "chinese-classical-train-365.jsonl"),
):
    entry = next(row for row in manifest["assets"] if row["id"] == identifier)
    archive = ROOT / entry["archive"]
    archive_bytes = archive.read_bytes()
    archive_url = None
    local_sha256 = sha(archive)
    if archive_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
        archive_url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/" + entry["archive"]
        archive_bytes = subprocess.run(["curl", "-fsSL", "--max-time", "30", archive_url], capture_output=True, check=True).stdout
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    assert archive_digest == entry["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        members = [member for member in bundle.getmembers() if Path(member.name).name == basename]
        assert len(members) == 1
        content = bundle.extractfile(members[0]).read()
    digest = hashlib.sha256(content).hexdigest()
    declared = next(row for row in entry["files"] if Path(row["path"]).name == basename)
    assert digest == declared["sha256"]
    raw.extend([json.loads(line) for line in content.decode().splitlines()][:96])
    asset_receipts.append({"archive": entry["archive"], "local_sha256": local_sha256, "archive_url": archive_url, "archive_sha256": archive_digest, "member": members[0].name, "member_sha256": digest})

parts = {
    split: [{**row, "text": _utf8_prefix(row["text"], 256), "complete_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()} for row in rows]
    for split, rows in split_records(_deduplicate_text(raw), seed=42).items()
}
result_path = ROOT / "docs/course-experiments/results/tokenizer.json"
recorded = json.loads(result_path.read_bytes())
recipe = experiment_spec("tokenizer")
assert recipe["module"] == "text" and recipe["function"] == "run_tokenizer"
split_checks = {}
for split, rows in parts.items():
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == recorded["results"]["data"][split]["sha256"]
    split_checks[split] = {"records": len(rows), "sha256": digest}
family_sets = [set(row["family"] for row in rows) for rows in parts.values()]
assert all(not left.intersection(right) for index, left in enumerate(family_sets) for right in family_sets[index + 1:])

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.train_from_iterator(
    [row["text"] for row in parts["train"]],
    trainers.BpeTrainer(vocab_size=512, special_tokens=list(SPECIALS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False),
)
bpe = _BPE(tokenizer)
runs = {}
for name, tok in (("byte256", ByteTokenizer()), ("bpe512", bpe)):
    lengths = [len(tok.encode(row["text"])) for row in parts["validation"]]
    expected = recorded["results"]["runs"][name]
    assert lengths == expected["validation_text_token_lengths"]
    assert all(tok.decode(tok.encode(row["text"])) == row["text"] for rows in parts.values() for row in rows)
    model = TinyLM(ModelConfig(vocab_size=tok.vocab_size, width=32, layers=1, max_length=384))
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == expected["parameters"]
    assert model.output.bias is None
    assert model.output.weight is not model.embedding.weight
    logits = model(torch.tensor([[tok.bos_id, 8]]))["logits"]
    runs[name] = {
        "vocab_size": tok.vocab_size, "validation_text_token_lengths": lengths,
        "sum": sum(lengths), "mean": sum(lengths) / len(lengths),
        "parameters": count, "embedding_parameters": model.embedding.weight.numel(),
        "output_parameters": model.output.weight.numel(),
        "remaining_parameters": count - model.embedding.weight.numel() - model.output.weight.numel(),
        "logits_shape": list(logits.shape), "all_splits_roundtrip": True,
        "recorded_training_steps": expected["training"]["steps"],
    }
assert runs["bpe512"]["parameters"] - runs["byte256"]["parameters"] == (512 - 264) * 32 * 2 == 15872
assert runs["byte256"]["sum"] == 3894 and runs["bpe512"]["sum"] == 2112

table = lambda width, vocabs: [{"vocab": vocab, "embedding_parameters": vocab * width, "fp32_bytes": vocab * width * 4, "two_tables_parameters": 2 * vocab * width} for vocab in vocabs]
base = table(32, [300, 1000, 3000])
wide = table(64, [300, 1000, 3000])
for before, after in zip(base, wide, strict=True):
    assert all(after[key] == 2 * before[key] for key in before if key != "vocab")
changed_vocab = table(32, [600, 1000, 3000])
assert all(changed_vocab[0][key] == 2 * base[0][key] for key in base[0] if key != "vocab")
assert changed_vocab[1:] == base[1:]
assert torch.empty((), dtype=torch.float32).element_size() == 4
tied = TinyLM(ModelConfig(vocab_size=512, width=32, layers=1, max_length=384, tied=True))
assert tied.output.weight is tied.embedding.weight
assert sum(p.numel() for p in tied.parameters()) == 57696 - 512 * 32

# 只以英文訓練的一個具體反例，核對字表變大不保證中文切得短。
english_counts = {}
for size in (264, 300):
    english = Tokenizer(models.BPE())
    english.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    english.decoder = decoders.ByteLevel()
    english.train_from_iterator(["the little cat sees the little dog " * 20], trainers.BpeTrainer(vocab_size=size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    english_counts[str(size)] = {"actual_vocab": english.get_vocab_size(), "english_tokens": len(english.encode("the little cat sees the little dog").ids), "chinese_tokens": len(english.encode("小小貓看狗").ids)}
assert english_counts["264"]["chinese_tokens"] == english_counts["300"]["chinese_tokens"] == 15
assert english_counts["300"]["english_tokens"] < english_counts["264"]["english_tokens"]

receipt_path = Path(__file__).with_name("section-facts-command.txt")
tool_directory = Path(json.loads(receipt_path.read_text())["artifact_directory"])
copied = {name: (tool_directory / name).read_text() for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt")}
Path(__file__).with_name("section-execution.json").write_text(json.dumps({"copied_from": str(tool_directory), "original_files": {name: {"sha256": hashlib.sha256(text.encode()).hexdigest(), "content": text} for name, text in copied.items()}}, ensure_ascii=False, indent=2) + "\n")

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-4/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "tokenizers": tokenizers.__version__, "device": "cpu", "platform": platform.platform()},
    "inputs": {str(path.relative_to(ROOT)): sha(path) for path in (manifest_path, result_path, ROOT / "tiny_perceptron/model.py", ROOT / "tiny_perceptron/data.py", ROOT / "scripts/course_experiments/text.py", ROOT / "scripts/course_experiments/common.py")},
    "assets": asset_receipts, "splits": split_checks, "family_overlap": False,
    "table_width32": base, "table_width64": wide, "table_vocab600": changed_vocab,
    "fp32_element_bytes": 4, "runs": runs, "parameter_difference": 15872,
    "tied_parameters": sum(p.numel() for p in tied.parameters()),
    "english_only_tokenizers": english_counts,
    "resolved_recipe": recipe,
    "result": "全部斷言通過；未執行模型訓練，未量測訓練時間或記憶體。",
}, ensure_ascii=False, indent=2))
