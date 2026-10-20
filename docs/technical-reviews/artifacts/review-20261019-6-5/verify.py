"""離線算例、實驗紀錄與資料分母核對；第三方全文只在記憶體中讀取。"""

import hashlib
import io
import json
import math
import platform
import random
import subprocess
import sys
import tarfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path.cwd()
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import _BPE, _deduplicate_text, _digest, _evaluate_tokenizer, _utf8_prefix
from tiny_perceptron.data import SPECIALS, ByteTokenizer, pad_batch, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    with (ART / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


network = []


def fetch(url):
    command = ["curl", "--max-time", "30", "-fLsS", url]
    result = subprocess.run(command, capture_output=True, check=False)
    network.append(
        {
            "command_argv": command,
            "exit_code": result.returncode,
            "stderr": result.stderr.decode(),
            "bytes": len(result.stdout),
            "sha256": sha(result.stdout),
        }
    )
    if result.returncode:
        return None
    return result.stdout


def html_text(raw):
    parser = Text()
    parser.feed(raw.decode())
    return " ".join(" ".join(parser.parts).split())


def fragment(text, anchor, before=0, after=450):
    index = text.index(anchor)
    return text[max(0, index - before) : index + len(anchor) + after]


torch.set_num_threads(1)
environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "tokenizers": str(tokenizers.__version__),
    "device": "cpu",
    "platform": platform.platform(),
}
record_path = ROOT / "docs/course-experiments/results/tokenizer.json"
record = json.loads(record_path.read_bytes())
results = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-5/verify.py", "environment": environment}
paths = [
    "docs/course-review-20261019/freeze-01/sources/6.5.md",
    "docs/course-review-20261019/freeze-04.json",
    "course/figures/rewrite-06-05-shared-denominator.svg",
    "course/figures/tokenizer_common_scale.svg",
    "docs/course-experiments/results/tokenizer.json",
    "scripts/course_experiments/text.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/tokenization.py",
    "course/training.md",
]
results["input_sha256"] = {path: sha((ROOT / path).read_bytes()) for path in paths}
results["historical_code_matches"] = {
    path: sha((ROOT / path).read_bytes()) == record["code_sha256"][path]
    for path in paths
    if path in record["code_sha256"]
}
assert all(results["historical_code_matches"][path] for path in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"])

results["opening_example"] = {"a": 10 * 0.6, "b": 5 * 1.0}
results["conditional_chain"] = {
    "product": 0.5 * 0.25,
    "individual_nll": [-math.log(0.5), -math.log(0.25)],
    "nll_sum": -math.log(0.5) - math.log(0.25),
    "negative_log_product": -math.log(0.125),
    "ln2": math.log(2),
    "half_probability_bits": -math.log(0.5) / math.log(2),
}
results["utf8"] = [{"text": s, "codepoints": [f"U+{ord(c):04X}" for c in s], "bytes": list(s.encode()), "count": len(s.encode())} for s in ["貓", "🙂", "貓🙂", "貓🙂貓🙂"]]
results["example_bpb"] = {
    "original": [n / (7 * math.log(2)) for n in [5, 4]],
    "double_nll": [n / (7 * math.log(2)) for n in [10, 8]],
    "double_bytes": [n / (14 * math.log(2)) for n in [5, 4]],
}
results["rounding"] = {"total_nll": 2.77947 * 4213, "bpb": 2.77947 * 4213 / (4193 * math.log(2)), "bpb_rounded": round(2.77947 * 4213 / (4193 * math.log(2)), 5)}
results["record_metrics"] = {}
for name, run in record["results"]["runs"].items():
    entries = {}
    for stage in ["before", "after"]:
        for split in ["validation", "test"]:
            metric = run[stage][split]
            total = metric["mean_token_nll"] * metric["effective_tokens"]
            bpb = total / (metric["raw_utf8_bytes"] * math.log(2))
            assert math.isclose(bpb, metric["bpb_including_eos_boundary_targets"], abs_tol=2e-15)
            entries[f"{stage}.{split}"] = {k: v for k, v in metric.items() if k != "samples"}
            entries[f"{stage}.{split}"].update(recomputed_total_nll=total, recomputed_bpb=bpb)
    config = ModelConfig(width=32, layers=1, max_length=384, vocab_size=run["vocab_size_including_8_specials"])
    model = TinyLM(config)
    assert sum(p.numel() for p in model.parameters()) == run["parameters"]
    entries["model"] = {"config": vars(config), "parameters": run["parameters"], "input_embedding_parameters": model.embedding.weight.numel()}
    entries["training"] = {"steps": run["training"]["steps"], "raw_utf8_bytes_exposed": run["training_raw_utf8_bytes_exposed"]}
    results["record_metrics"][name] = entries

logits = torch.tensor([[[0.5, 0.5], [0.75, 0.25], [0.5, 0.5]]], dtype=torch.float64).log()
labels = torch.tensor([[0, 1, -100]])
total, count = loss_sum(logits, labels)
mean = masked_loss(logits, labels)
assert int(count) == 2 and math.isclose(float(total), -math.log(0.125), abs_tol=1e-14)
assert math.isclose(float(mean), float(total) / 2, abs_tol=1e-14)
tok = ByteTokenizer()
examples = [shifted([tok.bos_id] + tok.encode(s) + [tok.eos_id]) for s in ["貓🙂", "a"]]
x, y, valid = pad_batch(examples)
results["loss_and_boundaries"] = {"nll_sum": float(total), "effective_count": int(count), "masked_loss": float(mean), "input_ids": x.tolist(), "target_ids": y.tolist(), "attention_valid": valid.tolist(), "effective_targets": int((y != -100).sum()), "specials": list(SPECIALS), "vocab_size": tok.vocab_size}


class UniformLM(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = ModelConfig(vocab_size=size, max_length=384)

    def forward(self, ids, cache=None):
        return {"logits": torch.zeros((*ids.shape, self.config.vocab_size)) + self.anchor}


measured = _evaluate_tokenizer(UniformLM(tok.vocab_size), [{"text": "貓🙂"}, {"text": "a"}], tok)
results["uniform_evaluator"] = {k: v for k, v in measured.items() if k != "samples"}
assert measured["effective_tokens"] == 10 and measured["raw_utf8_bytes"] == 8
assert math.isclose(measured["mean_token_nll"], math.log(tok.vocab_size), abs_tol=1e-6)

asset_rows = []
results["asset_receipts"] = []
for name, member in [("tinystories", "tinystories-train-512.jsonl"), ("chinese-poetry", "chinese-classical-train-365.jsonl")]:
    entry = next(a for a in record["assets"] if a["id"] == name)
    url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/{entry['archive']}"
    raw = fetch(url)
    assert raw is not None and sha(raw) == entry["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        payload = archive.extractfile(f"text-initial/{member}").read()
    expected = next(f for f in entry["files"] if f["path"] == f"text-initial/{member}")
    assert sha(payload) == expected["sha256"]
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    asset_rows += rows[:96]
    results["asset_receipts"].append({"url": url, "archive_sha256": sha(raw), "member": member, "member_sha256": sha(payload), "rows_used": 96})
complete = split_records(_deduplicate_text(asset_rows), seed=42)
parts = {split: [{**row, "text": _utf8_prefix(row["text"], 256), "complete_text_sha256": sha(row["text"].encode())} for row in rows] for split, rows in complete.items()}
results["reconstructed_splits"] = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert sha(raw) == record["results"]["data"][split]["sha256"]
    results["reconstructed_splits"][split] = {"records": len(rows), "sha256": sha(raw), "raw_utf8_bytes": sum(len(r["text"].encode()) for r in rows), "max_raw_utf8_bytes": max(len(r["text"].encode()) for r in rows)}
sampler = random.Random(42)
schedule = [sampler.choices(range(len(parts["train"])), k=8) for _ in range(400)]
exposed = sum(len(parts["train"][i]["text"].encode()) for batch in schedule for i in batch)
assert _digest(schedule) == record["results"]["raw_document_schedule_sha256"]
assert exposed == 664759
results["schedule"] = {"seed": 42, "steps": 400, "documents_per_step": 8, "raw_document_schedule_sha256": _digest(schedule), "raw_utf8_bytes_exposed_each_model": exposed}

bpe_tokenizer = Tokenizer(models.BPE())
bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe_tokenizer.decoder = decoders.ByteLevel()
bpe_tokenizer.train_from_iterator([r["text"] for r in parts["train"]], trainers.BpeTrainer(vocab_size=512, special_tokens=list(SPECIALS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
bpe = _BPE(bpe_tokenizer)
results["retokenized"] = {}
for name, tokenizer in [("byte256", tok), ("bpe512", bpe)]:
    splits = {}
    for split, rows in parts.items():
        lengths = [len(tokenizer.encode(r["text"])) for r in rows]
        effective = sum(lengths) + len(rows)
        splits[split] = {"text_tokens": sum(lengths), "effective_targets": effective, "max_input_positions": max(lengths) + 1, "roundtrip_all": all(tokenizer.decode(tokenizer.encode(r["text"])) == r["text"] for r in rows), "training_examples": len(text_examples(rows, mode="text", max_length=384, tokenizer=tokenizer))}
        if split in ["test", "validation"]:
            splits[split]["matches_recorded_count"] = effective == record["results"]["runs"][name]["after"][split]["effective_tokens"]
            if split == "validation":
                splits[split]["matches_recorded_lengths"] = lengths == record["results"]["runs"][name]["validation_text_token_lengths"]
    results["retokenized"][name] = {"vocab_size": tokenizer.vocab_size, "splits": splits}
results["retokenization_scope"] = "BPE 依相同輸入與配方重新建立；僅核對長度與可還原性，不宣稱 ID／merges 與當年序列化檔逐位元組相同。"

sample_lengths = []
for byte, merged in zip(record["results"]["runs"]["byte256"]["after"]["test"]["samples"], record["results"]["runs"]["bpe512"]["after"]["test"]["samples"], strict=True):
    assert byte["row"] == merged["row"] and byte["prompt"] == merged["prompt"]
    sample_lengths.append({"row": byte["row"], "byte_chars": len(byte["generated"]), "bpe_chars": len(merged["generated"]), "byte_reencoded_bytes": len(byte["generated"].encode()), "bpe_reencoded_bytes": len(merged["generated"].encode())})
results["generation_lengths"] = {"test_records": 20, "max_new_tokens_from_evaluator_source": 32, "bpe_longer_in_characters": sum(r["bpe_chars"] > r["byte_chars"] for r in sample_lengths), "records": sample_lengths, "scope": "解碼後展示文字長度；含 replacement character 時，重新編碼 byte 數不等於實際生成的 byte token 數。"}
vocabulary = bpe_tokenizer.get_vocab()
piece = next(p for p in sorted(vocabulary) if p.isascii() and p.isalpha() and len(p) > 1 and all(c in vocabulary for c in p))
merged_ids = [vocabulary[piece]]
unmerged_ids = [vocabulary[c] for c in piece]
assert bpe.decode(merged_ids) == bpe.decode(unmerged_ids) == piece
results["multiple_token_sequences"] = {"text": piece, "merged_ids": merged_ids, "unmerged_ids": unmerged_ids, "decode_same": True, "scope": "相同配方建立的 BPE 有可解碼成同一原文的不同序列；未測量已訓練模型的非預設切分機率。"}
results["normalized_toy_counterexample"] = {
    "vocabulary": ["a", "b", "ab", "EOS"],
    "conditionals": {"BOS": {"a": 0.4, "b": 0.2, "ab": 0.3, "EOS": 0.1}, "a": {"b": 0.5, "EOS": 0.5}, "b": {"EOS": 1.0}, "ab": {"b": 0.5, "EOS": 0.5}, "a,b": {"EOS": 1.0}, "ab,b": {"EOS": 1.0}},
    "canonical_sequence": ["ab", "EOS"],
    "canonical_sequence_probability": 0.3 * 0.5,
    "alternative_sequence": ["a", "b", "EOS"],
    "alternative_sequence_probability": 0.4 * 0.5,
    "decoded_string_probability": 0.3 * 0.5 + 0.4 * 0.5,
    "canonical_nll": -math.log(0.3 * 0.5),
    "marginal_string_nll": -math.log(0.3 * 0.5 + 0.4 * 0.5),
}

results["figure_checks"] = {}
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
for path in ["course/figures/rewrite-06-05-shared-denominator.svg", "course/figures/tokenizer_common_scale.svg"]:
    row = next(r for r in receipt["records"] if r["page_id"] == "6.5" and r["figure"] == path)
    png = ROOT / row["png"]["path"]
    assert sha(png.read_bytes()) == row["png"]["sha256"]
    results["figure_checks"][path] = {"svg_sha256": sha((ROOT / path).read_bytes()), "viewed_png": row["png"], "svg_matches_frozen": sha((ROOT / path).read_bytes()) == row["frozen_svg"]["sha256"]}
results["bar_scale"] = {"widths": [333.5364, 478.4892, 483.4872, 412.0812], "values": [2.77947, 3.98741, 4.02906, 3.43401], "pixels_per_unit": [w / v for w, v in zip([333.5364, 478.4892, 483.4872, 412.0812], [2.77947, 3.98741, 4.02906, 3.43401], strict=True)]}

snapshots = []
url = "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/perplexity.md"
raw = fetch(url)
text = raw.decode()
snapshots.append({"url": url, "version": "transformers v4.57.1", "document_sha256": sha(raw), "locator": "Perplexity of fixed-length models；Calculating PPL with fixed-length models；target_ids context mask", "excerpt": "Perplexity is defined as the exponentiated average negative log-likelihood of a sequence.", "inspection_note": "已讀平均負條件對數機率公式、tokenization 影響、固定 context 與滑動視窗說明及 -100 遮罩範例；不採用該範例的模型內部 shift 計數為本專案契約。"})
url = "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.9.1/lm_eval/api/metrics.py"
raw = fetch(url)
text = raw.decode()
snapshots.append({"url": url, "version": "lm-evaluation-harness v0.4.9.1", "document_sha256": sha(raw), "locator": "bits_per_byte；weighted_mean", "excerpt": fragment(text, "def bits_per_byte(items):", after=53) + "\n" + fragment(text, "def weighted_mean(items):", after=55), "inspection_note": "BPB 以總 log-likelihood／總 byte 權重，再除以 ln(2)；越低越好。"})
url = "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/library/math.rst"
raw = fetch(url)
text = raw.decode()
snapshots.append({"url": url, "version": "CPython v3.13.16", "document_sha256": sha(raw), "locator": "math.log(x[, base])", "excerpt": fragment(text, ".. function:: log(x[, base])", after=110), "inspection_note": "單參數 log 使用自然對數；與本機 Python 版本相同。"})
url = "https://www.rfc-editor.org/rfc/rfc3629.txt"
raw = fetch(url)
text = raw.decode()
snapshots.append({"url": url, "version": "RFC 3629, November 2003", "document_sha256": sha(raw), "locator": "UTF-8 definition 的碼點／octet 對照表", "excerpt": fragment(text, "0000 0800-0000 FFFF", after=124), "inspection_note": "U+8C93 使用三個 octet；U+1F642 使用四個 octet；不涉及 Unicode 字素群數。"})
url = "https://aclanthology.org/2023.acl-short.1/"
raw = fetch(url)
text = html_text(raw)
snapshots.append({"url": url, "version": "ACL 2023 出版版，DOI 10.18653/v1/2023.acl-short.1", "document_sha256": sha(raw), "locator": "Abstract；出版版 PDF 的 Introduction 與 Preliminaries", "excerpt": fragment(text, "To truly compute the probability of a string one should marginalize over all tokenizations", after=0), "inspection_note": "curl 讀出版方摘要與 PDF 原檔；另以 web PDF 文字擷取讀出版版前兩頁，包括 P(T,S) 的連乘與 P(S) 的多切分加總。本文不以該論文的效益大小外推本課模型。"})
url = "https://aclanthology.org/2023.acl-short.1.pdf"
raw = fetch(url)
snapshots.append({"url": url, "version": "ACL 2023 出版版，12 頁", "document_sha256": sha(raw), "locator": "Introduction；Preliminaries", "excerpt": "P(T,S) = 1[T→S] ∏j P(tj | t<j)；P(S) = Σ[T:T→S] P(T,S)", "inspection_note": "上述為已讀公式的純文字轉寫；全文不保存到 repo。"})
url = "https://www.deeplearningbook.org/contents/prob.html"
raw = fetch(url)
text = html_text(raw)
snapshots.append({"url": url, "version": "Deep Learning, Goodfellow/Bengio/Courville, 2016，作者網站 HTML，讀取日 2026-10-20", "document_sha256": sha(raw), "locator": "Information Theory 中 self-information 與 nats", "excerpt": fragment(text, "nats", before=160, after=120), "inspection_note": "原文確認 I(x)=-log P(x)；自然對數的單位稱 nats，底數二的單位稱 bits。"})
write("source-excerpts.json", {"accessed_on": "2026-10-20", "sources": snapshots, "network_receipts": network})
results["network_receipts"] = network[:2]
results["verification_scope"] = "核對當次既有實驗紀錄、資料切分指紋、分母、配方及算例；未重跑訓練，未下載或重評私有權重。"
write("verification.json", results)
print(json.dumps({"result": "核對完成", "environment": environment, "artifacts": {n: sha((ART / n).read_bytes()) for n in ["verification.json", "source-excerpts.json"]}}, ensure_ascii=False))
