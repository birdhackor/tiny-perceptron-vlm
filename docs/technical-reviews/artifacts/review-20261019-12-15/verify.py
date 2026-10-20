"""保存原文執行紀錄、聲音入口核對結果與原始來源的小段摘錄。"""

import ast
import hashlib
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import log_mel, tone
from tiny_perceptron.selftrained.model import AudioEncoder, LimitedAssistant, SelftrainedConfig


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1
        if tag in ("p", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def fetch(url, version, selections, html=False):
    argv = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "40", url]
    completed = subprocess.run(argv, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode("utf-8")
    if html:
        parser = PlainText()
        parser.feed(text)
        text = "\n".join(" ".join(p.split()) for p in "".join(parser.parts).splitlines() if p.strip())
    snippets = []
    for locator, start, end in selections:
        begin = text.index(start)
        finish = text.index(end, begin) + len(end)
        snippets.append({"locator": locator, "text": text[begin:finish]})
    return {
        "url": url,
        "version": version,
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "command_argv": argv,
        "exit_code": completed.returncode,
        "original_bytes": len(raw),
        "original_sha256": sha(raw),
        "excerpts": snippets,
    }, text


def local_file(path, symbols=()):
    raw = (ROOT / path).read_bytes()
    tree = ast.parse(raw)
    excerpts = {}
    for symbol in symbols:
        node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol)
        excerpts[symbol] = ast.get_source_segment(raw.decode(), node)
    return {"path": path, "sha256": sha(raw), "excerpts": excerpts}


torch.set_num_threads(1)
torch.manual_seed(0)
run_dir = ROOT / sys.argv[1]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-15/verify.py " + sys.argv[1],
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu", "threads": str(torch.get_num_threads())},
    "section_execution": {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/12.md#12.15" --execute',
        "copied_files": {name: {"sha256": sha((run_dir / name).read_bytes()), "text": (run_dir / name).read_text()} for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "section.md", "fence-1.py")},
    },
    "local_files": [
        local_file("tiny_perceptron/selftrained/model.py", ("AudioEncoder", "_embeddings", "generate")),
        local_file("tiny_perceptron/multimodal.py", ("log_mel", "mel_filter_bank")),
        local_file("tiny_perceptron/selftrained/dataset.py", ("_modality",)),
        local_file("scripts/selftrained/prepare_voice.py"),
        local_file("scripts/selftrained/train.py", ("set_trainable",)),
    ],
}
source = (ROOT / "course/chapters/12.md").read_bytes()
body = re.search(rb"(?ms)^## 12\.15 .*?(?=^## |\Z)", source)[0]
assert sha(body) == "9e98f7298525611b54484b6f1adca6488b851b93cda10df8be776f003ff6038f"
result["source_sha256"] = sha(body)
encoder = AudioEncoder(64)
probes = []
for frames in (1, 2, 63, 64, 65, 173, 600, 1000):
    mel = torch.randn(frames, 40)
    with torch.no_grad():
        temporal = encoder.temporal(mel.T[None])
        tokens, logits = encoder(mel)
    shortened = temporal.shape[-1]
    assert shortened == math.ceil(frames / 4)
    assert tuple(tokens.shape) == (16, 64)
    assert tuple(logits.shape) == (3,)
    intervals = [(i * shortened // 16, ((i + 1) * shortened + 15) // 16) for i in range(16)]
    basis = torch.eye(shortened)[None]
    observed_weights = F.adaptive_avg_pool1d(basis, 16)[0].T
    expected_weights = torch.zeros_like(observed_weights)
    for i, (start, end) in enumerate(intervals):
        expected_weights[i, start:end] = 1 / (end - start)
    torch.testing.assert_close(observed_weights, expected_weights, rtol=0, atol=0)
    probes.append({"input_shape": list(mel.shape), "shortened": shortened, "tokens_shape": list(tokens.shape), "logits_shape": list(logits.shape), "intervals_half_open": intervals, "interval_sizes": [b - a for a, b in intervals], "adjacent_overlap": [max(0, intervals[i][1] - intervals[i + 1][0]) for i in range(15)], "pool_basis_max_error": float((observed_weights - expected_weights).abs().max())})
result["audio_probes"] = probes
invalid = []
for bands in (16, 39, 41):
    try:
        encoder(torch.randn(173, bands))
    except ValueError as error:
        invalid.append({"bands": bands, "error_type": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("頻帶數驗證沒有拒絕不相符的輸入")
result["invalid_bands"] = invalid
result["module_configuration"] = str(encoder)
torch.manual_seed(23)
first = AudioEncoder(64)
torch.manual_seed(23)
second = AudioEncoder(64)
torch.manual_seed(24)
third = AudioEncoder(64)
result["initialization"] = {"same_seed_equal": all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())), "different_seed_any_difference": any(not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters()))}
wide = AudioEncoder(256)
with torch.no_grad():
    result["wide_output_shape"] = list(wide(torch.randn(173, 40))[0].shape)
    assistant = LimitedAssistant(SelftrainedConfig(width=64, layers=1))
    input_ids = torch.tensor([[1, 4] + [9] * 16 + [2, 5]])
    payload = [[{"kind": "audio", "values": torch.randn(173, 40)}]]
    embeddings, perception = assistant._embeddings(input_ids, None, payload)
    audio_tokens, _ = assistant.audio_encoder(payload[0][0]["values"])
    assert torch.equal(embeddings[0, 2:18], audio_tokens)
    generated = assistant.generate(input_ids, modalities=payload, max_new_tokens=1)
    result["embedding_and_generation"] = {"audio_slots": 16, "embeddings_shape": list(embeddings.shape), "audio_slots_equal_encoder_output": True, "perception_logits_shape": list(perception[0]["logits"].shape), "generated_shape_without_labels": list(generated.shape), "uses_gold_answer": False, "trained": False}
waveform = tone(440)
mel = log_mel(waveform, bands=40)
power = torch.stft(waveform, 400, 160, window=torch.hann_window(400), return_complex=True).abs().square()
from tiny_perceptron.multimodal import mel_filter_bank
independent = (mel_filter_bank(bands=40) @ power).clamp(min=1e-8).log()
torch.testing.assert_close(mel, independent, rtol=0, atol=0)
result["log_mel"] = {"shape": list(mel.shape), "finite": bool(mel.isfinite().all()), "max_error_against_explicit_formula": float((mel-independent).abs().max()), "has_trainable_parameters": False}
voice_tree = ast.parse((ROOT / "scripts/selftrained/prepare_voice.py").read_text())
result["voice_metadata"] = {n.targets[0].id: ast.literal_eval(n.value) for n in voice_tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("SELECTED", "ANSWERS")}
chapter = (ROOT / "course/chapters/19.md").read_text()
result["course_configuration_excerpt"] = next(x for x in chapter.splitlines() if x.startswith('    config = SelftrainedConfig(vocab_size=550'))
result["staged_tone_excerpt"] = next(x for x in source.decode().splitlines() if x.startswith("真正更新參數的既有實驗"))
result["diagnosis_counterexample"] = {
    "input_semantics": "需求以 x=-1 或 x=1 表示；文字入口為 [x,0]，語音入口與接頭為 [0,x]，兩者都無損保留需求。",
    "core": "共享文字核心使用線性分數 w dot z，再取正負判類；w=[1,-1]。",
    "rows": [{"intent": x, "text_prediction": int(torch.sign(torch.tensor([1., -1.]) @ torch.tensor([float(x), 0.]))), "audio_prediction": int(torch.sign(torch.tensor([1., -1.]) @ torch.tensor([0., float(x)])))} for x in (-1, 1)],
    "repair_core_only": [{"intent": x, "audio_prediction": int(torch.sign(torch.tensor([1., 1.]) @ torch.tensor([0., float(x)])))} for x in (-1, 1)],
    "scope": "這是反駁唯一故障歸因的存在性反例，不是本專案模型的實測品質；也可改接頭解決，正說明只有兩個答案不能唯一歸因。",
}
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_text())
figure = next(x for x in receipt["records"] if x["page_id"] == "12.15")
assert sha((ROOT / figure["png"]["path"]).read_bytes()) == figure["png"]["sha256"]
result["figure_receipt"] = figure
result["external_sources"] = {}
card, card_text = fetch("https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md", "40ce77cb32a384e4d50a568e1ec39ac804019d33", [("YAML license", "license:\n", "- cc-by-4.0"), ("Dataset Description", "MINDS-14 is training", "14 diverse language varieties."), ("Data Fields", "### Data Fields", "- **lang_id** (int): Id of language")])
zh = card_text[card_text.rindex("- config_name: zh-CN"):card_text.index("\n---", card_text.rindex("- config_name: zh-CN"))]
card["zh_schema_field_names"] = re.findall(r"^  - name: (\w+)", zh.split("  splits:")[0], re.M)
card["zh_intent_names"] = re.findall(r"^          '\d+': (\w+)$", zh.split("  - name: lang_id")[0], re.M)
card["zh_num_examples"] = int(re.search(r"num_examples: (\d+)", zh)[1])
card["zh_count_excerpt"] = zh[zh.index("  splits:"):zh.index("  download_size:")]
assert card["zh_num_examples"] == 502
result["external_sources"]["minds_card"] = card
paper, _ = fetch("https://arxiv.org/html/2104.08524v1", "arXiv:2104.08524v1, 17 Apr 2021", [("Dataset Collection / Spoken Data Collection", "With both protocols, human subjects", "associated with the intent class."), ("Dataset Collection / online recording", "The second, simpler study design", "via this approach."), ("Methodology / Pretrained Sentence Encoders", "We keep all pretrained sentence encoders fixed", "intent detection training."), ("Experimental Setup / Speech Transcription", "Speech Transcription. For all recordings", "written transcriptions (i.e., ASR hypotheses).")], html=True)
result["external_sources"]["minds_paper"] = paper
paper, _ = fetch("https://arxiv.org/html/1904.03670v1", "arXiv:1904.03670v1, 7 Apr 2019", [("Introduction / conventional SLU", "The conventional SLU pipeline", "speaker’s intent [2, 3, 4]."), ("Introduction / end-to-end SLU", "In end-to-end SLU,", "a text transcript (Fig. 1)."), ("Dataset splits", "The utterances are randomly divided", "more than one split."), ("Model and Pre-training Strategy", "The model proposed in this paper, shown in Fig. 2,", "supervised SLU task."), ("Generalizing to new wordings", "However, there are many situations", "the correct meaning.")], html=True)
result["external_sources"]["slu_paper"] = paper
commit = torch.version.git_version
pool, _ = fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/aten/src/ATen/native/AdaptivePooling.h", f"PyTorch {torch.__version__}, commit {commit}", [("start_index/end_index", "inline int64_t start_index", "return 1 + ((a + 1) * c - 1) / b;\n}")])
result["external_sources"]["adaptive_source"] = pool
pool, _ = fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/pooling.py", f"PyTorch {torch.__version__}, commit {commit}", [("AdaptiveAvgPool1d", "class AdaptiveAvgPool1d", "return F.adaptive_avg_pool1d(input, self.output_size)")])
result["external_sources"]["adaptive_docs"] = pool
groups, _ = fetch("https://scikit-learn.org/1.7/modules/cross_validation.html", "scikit-learn 1.7 official documentation", [("Cross-validation iterators for grouped data", "The i.i.d. assumption is broken", "GroupKFold" )], html=True)
result["external_sources"]["grouped_data"] = groups
result["result"] = "全部斷言通過；時間池化與唯一故障歸因的反例已保存。模型未受訓，不把隨機輸出當成語音能力證據。"
output = Path(__file__).with_name("verification.json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "environment": result["environment"], "result": result["result"]}, ensure_ascii=False))
