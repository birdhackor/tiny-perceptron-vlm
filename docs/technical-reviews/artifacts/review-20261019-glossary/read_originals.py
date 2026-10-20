"""以 curl 讀原始來源；全文只在記憶體，保存取得收據與短摘錄。"""

import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).parent
SOURCES = {
    "unicode": ("https://www.unicode.org/reports/tr29/tr29-47.html", ["user-perceived", "multiple", "grapheme cluster"]),
    "utf8": ("https://www.rfc-editor.org/rfc/rfc3629.txt", ["one to four octets", "Surrogates", "3.  UTF-8 definition"]),
    "json": ("https://www.rfc-editor.org/rfc/rfc8259.txt", ["serialized value", "object is an unordered", "string is a sequence"]),
    "tokenizer": ("https://huggingface.co/docs/tokenizers/pipeline", ["Pre-tokenization", "BPE", "Decoding"]),
    "linear": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", ["y=", "bias", "Input:"]),
    "embedding": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html", ["lookup table", "weight", "Input:"]),
    "ce": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", ["ignore_index", "reduction", "unnormalized logits"]),
    "softmax": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softmax.html", ["Softmax", "sum to 1", "exp"]),
    "norm": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html", ["variance", "eps", "elementwise_affine"]),
    "rms": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html", ["RMS", "eps"]),
    "adam": ("https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html", ["first and second moments", "beta", "state"]),
    "autograd": ("https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html", ["chain rule", "accumulated", "backward"]),
    "checkpoint": ("https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html", ["optimizer_state_dict", "resume training", "random"]),
    "mlp": ("https://www.deeplearningbook.org/contents/mlp.html", ["feedforward", "gradient", "activation"]),
    "attention": ("https://arxiv.org/html/1706.03762v7", ["Scaled Dot-Product", "Position-wise Feed-Forward", "self-attention"]),
    "vit": ("https://arxiv.org/html/2010.11929v2", ["patches", "linear projection"]),
    "dpo": ("https://arxiv.org/html/2305.18290v3", ["reference", "binary cross", "preference"]),
    "lora": ("https://arxiv.org/html/2106.09685v2", ["rank", "frozen", "trainable"]),
    "mtp": ("https://arxiv.org/html/2404.19737v1", ["shared", "output", "future tokens"]),
    "ppo": ("https://spinningup.openai.com/en/latest/algorithms/ppo.html", ["not a hard constraint", "min", "Advantage"]),
    "rlhf": ("https://arxiv.org/html/2203.02155v1", ["reward model", "PPO", "human"]),
    "calibration": ("https://proceedings.mlr.press/v70/guo17a.html", ["temperature scaling", "probability", "calibration"]),
    "moe": ("https://arxiv.org/html/2101.03961v3", ["router", "feed-forward", "auxiliary"]),
    "rope": ("https://arxiv.org/html/2104.09864v5", ["rotation", "relative", "query"]),
    "interpolation": ("https://arxiv.org/html/2306.15595v2", ["linearly", "position", "interpolation"]),
    "loop": ("https://arxiv.org/html/1807.03819v3", ["recurrent", "shared", "depth"]),
    "flash": ("https://arxiv.org/html/2205.14135v2", ["exact", "tiling", "HBM"]),
    "mistral": ("https://arxiv.org/html/2310.06825v1", ["sliding", "layers", "window"]),
    "distill": ("https://arxiv.org/html/1503.02531v1", ["soft targets", "temperature", "teacher"]),
    "speculative": ("https://proceedings.mlr.press/v202/leviathan23a.html", ["distribution", "speculative", "sampling"]),
    "passk": ("https://arxiv.org/html/2107.03374v2", ["pass@", "unbiased", "samples"]),
    "rag": ("https://arxiv.org/html/2005.11401v4", ["non-parametric", "retriever", "generator"]),
    "quant": ("https://docs.pytorch.org/docs/2.14/quantization.html", ["scale", "calibration", "activation"]),
    "audio": ("https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html", ["power", "mel", "sr"]),
    "jev": ("https://docs.typesafe.ai/introduction", ["Choice", "Noul", "Score"]),
    "jev_confidence": ("https://docs.typesafe.ai/confidence", ["Choice", "Noul", "confidence"]),
    "qwen": ("https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/README.md", ["AutoProcessor", "apply_chat_template", "image"]),
    "whisper": ("https://raw.githubusercontent.com/openai/whisper/main/README.md", ["speech recognition", "transcribe", "multilingual"]),
    "cuda": ("https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html", ["parallel computing", "GPU"]),
    "split": ("https://scikit-learn.org/stable/modules/cross_validation.html", ["test set", "groups", "validation"]),
}


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch(name):
    url, needles = SOURCES[name]
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True)
    record = {"id": name, "url": url, "command": " ".join(command), "accessed_at": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode}
    if result.returncode:
        record["error"] = result.stderr.decode(errors="replace")
        return record, record["error"]
    raw = result.stdout
    record["retrieved_sha256"] = hashlib.sha256(raw).hexdigest()
    record["bytes"] = len(raw)
    decoded = raw.decode(errors="replace")
    if "<html" in decoded[:1000].lower() or "<!doctype" in decoded[:1000].lower():
        parser = Plain()
        parser.feed(decoded)
        decoded = " ".join(parser.parts)
    plain = re.sub(r"\s+", " ", decoded)
    excerpts = []
    for needle in needles:
        found = re.search(re.escape(needle), plain, re.I)
        if found:
            excerpts.append({"needle": needle, "text": plain[max(0, found.start()-120):found.end()+850]})
    record["located_terms"] = [x["needle"] for x in excerpts]
    record["short_quote"] = " ".join((excerpts[0]["text"] if excerpts else plain[:300]).split()[:24])
    return record, excerpts or plain[:400]


if __name__ == "__main__":
    names = sys.argv[1:]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for name, (record, excerpts) in zip(names, pool.map(fetch, names)):
            records.append(record)
            print(json.dumps({"id": name, "reading": excerpts}, ensure_ascii=False))
    target = BASE / ("originals-" + "-".join(names) + ".json")
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")
