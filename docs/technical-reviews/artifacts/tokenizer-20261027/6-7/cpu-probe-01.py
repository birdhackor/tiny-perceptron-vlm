"""Independent, CPU-only verification of canonical 6.7 and its direct prerequisites."""

import codecs
import contextlib
import hashlib
import importlib.util
import io
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import tokenizers
import torch

from tiny_perceptron.data import SPECIALS, ByteTokenizer

assert not torch.cuda.is_available()
torch.set_default_device("cpu")
torch.set_num_threads(1)

spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_fences(lesson):
    body, whole, first_line = facts.original_section(ROOT / "course/chapters/06.md", lesson)
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    code_facts = []
    started = time.perf_counter()
    with contextlib.redirect_stdout(output):
        for number, fence in enumerate(facts.fences(body, first_line), 1):
            if fence["language"] != "python":
                continue
            assert fence["closed"]
            code_facts.append({"number": number, "line": fence["code_line"], "sha256": digest(fence["raw"])})
            exec(compile(fence["raw"], f"course/chapters/06.md#{lesson}:fence-{number}", "exec"), namespace)
    return namespace, {
        "source": f"course/chapters/06.md#{lesson}",
        "source_sha256": digest(body),
        "source_file_sha256": digest(whole),
        "fences": code_facts,
        "stdout": output.getvalue(),
        "elapsed_seconds": time.perf_counter() - started,
    }


ns, exact_67 = run_fences("6.7")
raw = ns["raw"]
assert list(raw) == [232, 178, 147, 240, 159, 153, 130]
assert ns["shown"] == ["", "", "貓", "", "", "", "🙂"]

strict_error = None
try:
    raw[:1].decode("utf-8")
except UnicodeDecodeError as error:
    strict_error = {"type": type(error).__name__, "start": error.start, "end": error.end, "reason": error.reason}
assert strict_error is not None

variants = {str(length): raw[:length].decode("utf-8", errors="replace") for length in (1, 3, 4)}
assert variants == {"1": "�", "3": "貓", "4": "貓�"}
individually = "".join(bytes([byte]).decode("utf-8", errors="replace") for byte in raw)
assert individually == "�" * 7
damaged_bytes = raw[:1].decode("utf-8", errors="replace").encode("utf-8") + raw[1:]
assert damaged_bytes != raw
assert damaged_bytes.decode("utf-8", errors="replace") != "貓🙂"

decoder_factory = codecs.getincrementaldecoder("utf-8")
assert inspect.isclass(decoder_factory)
decoder = decoder_factory()
trace = []
for byte in raw:
    shown = decoder.decode(bytes([byte]))
    pending, state = decoder.getstate()
    trace.append({"received": byte, "shown": shown, "pending": list(pending), "state": state})
assert decoder.decode(b"", final=True) == ""

partial_decoder = codecs.getincrementaldecoder("utf-8")()
assert partial_decoder.decode(raw[:2]) == ""
assert partial_decoder.getstate()[0] == raw[:2]
assert partial_decoder.decode(raw[2:3]) == "貓"

unfinished = decoder_factory()
assert unfinished.decode(raw[:2]) == ""
unfinished_error = None
try:
    unfinished.decode(b"", final=True)
except UnicodeDecodeError as error:
    unfinished_error = type(error).__name__ + ": " + error.reason
assert unfinished_error == "UnicodeDecodeError: unexpected end of data"

tok = ByteTokenizer()
ids = tok.encode("貓🙂")
assert ids == [byte + len(SPECIALS) for byte in raw]
assert tok.decode(ids) == "貓🙂"
assert tok.decode(ids[:1]) == "�"

bpe_ns, exact_63 = run_fences("6.3")
bpe = bpe_ns["tok"]
assert bpe.get_vocab_size() == 277
bpe_cases = []
for text in ("貓看狗。", "貓看狗", "小", "小鳥🙂new"):
    encoded = bpe.encode(text)
    restored = bpe.decode(encoded.ids)
    assert restored == text
    bpe_cases.append({"text": text, "bytes": list(text.encode("utf-8")), "ids": encoded.ids,
                      "tokens": encoded.tokens, "token_count": len(encoded.ids), "restored": restored})
assert bpe_cases[0]["token_count"] == 2
assert bpe_cases[1]["token_count"] == 1
assert bpe_cases[2]["token_count"] == 3
assert bpe_cases[3]["token_count"] == 13

record_path = ROOT / "docs/course-experiments/results/real_text.json"
record = json.loads(record_path.read_text(encoding="utf-8"))
sample = record["results"]["runs"]["chinese-poetry"]["after"]["validation"]["samples"][7]
sample_ids = sample["generated_ids"]
sample_raw = bytes(value - len(SPECIALS) for value in sample_ids)
assert len(sample_ids) == 32
assert tok.decode(sample_ids) == sample["generated"] == "孟沙，天夜天天天天春�"
assert sample_raw[:-2].decode("utf-8") == "孟沙，天夜天天天天春"
sample_decoder = codecs.getincrementaldecoder("utf-8")()
assert sample_decoder.decode(sample_raw) == "孟沙，天夜天天天天春"
assert sample_decoder.getstate()[0] == sample_raw[-2:]
assert list(sample_raw[-2:]) == [229, 164]
assert sample["prompt"] == "渡漢江\n李頻\n嶺外"
assert len(sample["prompt"].encode("utf-8")) <= 24

source_files = ["course/chapters/06.md", "course/chapters/01.md", "tiny_perceptron/data.py",
                "tiny_perceptron/model.py", "scripts/course_experiments/text.py",
                "scripts/course_experiments/common.py", "docs/course-experiments/results/real_text.json"]
source_versions = {name: digest((ROOT / name).read_bytes()) for name in source_files}
provenance = {name: {"current_sha256": source_versions[name], "recorded_sha256": record["code_sha256"].get(name),
                     "same": source_versions[name] == record["code_sha256"].get(name)}
              for name in source_files if name.endswith(".py")}
assert all(item["same"] for item in provenance.values())

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/6-7/cpu-probe-01.py",
    "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "device": "cpu", "torch": str(torch.__version__), "tokenizers": tokenizers.__version__,
                    "cuda_available": str(torch.cuda.is_available()), "cuda_build": str(torch.version.cuda),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")},
    "exact_6_7": exact_67,
    "strict_first_byte_error": strict_error,
    "slice_replace_outputs": variants,
    "independent_byte_replace": individually,
    "stored_replacement": {"bytes": list(damaged_bytes), "decoded": damaged_bytes.decode("utf-8", errors="replace"),
                           "equals_original": damaged_bytes == raw},
    "incremental_trace": trace,
    "incremental_decoder_class": decoder_factory.__module__ + "." + decoder_factory.__qualname__,
    "unfinished_stream_finalization": unfinished_error,
    "byte_tokenizer": {"ids": ids, "restored": tok.decode(ids), "first_id_alone": tok.decode(ids[:1])},
    "exact_6_3": exact_63,
    "bpe_cases": bpe_cases,
    "recorded_sample": {"path": "docs/course-experiments/results/real_text.json", "sample_index": 7,
                        "recorded_device": record["device"], "recorded_seed": record["seed"],
                        "recorded_revision": record["revision"], "prompt": sample["prompt"],
                        "prompt_bytes": len(sample["prompt"].encode("utf-8")), "generated": sample["generated"],
                        "generated_ids": sample_ids, "raw_bytes": list(sample_raw),
                        "generated_byte_count": len(sample_raw), "pending_tail_bytes": list(sample_decoder.getstate()[0]),
                        "retrained": False, "code_provenance_matches_current": provenance},
    "source_file_sha256": source_versions,
    "limits": "Executed exact 6.7 and tiny two-sentence 6.3 tokenizer rule learning only; no LM training, model inference, GPU, dataset download, publishing, or quality benchmark. Read existing generated IDs and decoded them on CPU; did not recreate the historical GPU run.",
    "result": "All exact-code assertions and targeted behavioral checks passed.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
