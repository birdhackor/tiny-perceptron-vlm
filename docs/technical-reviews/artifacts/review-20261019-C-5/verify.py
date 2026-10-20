"""獨立核對 token 預算、原始候選、計時邊界與原始來源摘錄。"""
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
import importlib.util
spec = importlib.util.spec_from_file_location("scripts.course_experiments._review_recorded_app", Path(__file__).parent / "applications-recorded.py")
app = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = app
spec.loader.exec_module(app)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(value, message):
    if not value:
        raise AssertionError(message)

torch.set_num_threads(1)
torch.manual_seed(31)
tok = ByteTokenizer()
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-5/verify.py",
    "checked_at": datetime.now(UTC).isoformat(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1"},
    "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "file_sha256": {},
}
paths = ["course/chapters/0C.md", "scripts/course_experiments/applications.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "docs/course-experiments/results/reasoning.json"]
for name in paths:
    result["file_sha256"][name] = digest(ROOT / name)
result["tokenizer"] = {s: tok.encode(s) + [tok.eos_id] for s in ("pass", "fail")}
check(all(len(ids) == 5 for ids in result["tokenizer"].values()), "ASCII 回答加 EOS 的 token 數不符")
bundle = json.loads((Path(__file__).parent / "section-execution.json").read_text())
check(bundle["extraction.json"]["source_sha256"] == "ca26ec5dff92b17cfa5e7e2b430ab90a2cb5be40a239bf76303b7683d59d8575", "執行原文與凍結內容不符")
code = bundle["fence-1.py"]
namespace = {}
with contextlib.redirect_stdout(io.StringIO()) as stream:
    exec(compile(code.replace("verify_tokens_each = 5", "verify_tokens_each = 0"), "exercise", "exec"), namespace)
result["exercise_stdout"] = stream.getvalue()
check(result["exercise_stdout"] == "候選 1 生成 80 驗證 0 總token 80\n候選 4 生成 80 驗證 0 總token 80\n", "零驗證 token 的預算不符")
result["budget_derivation"] = {"one_answer": "1*80 + 1*5 = 85", "four_answers": "4*20 + 4*5 = 100", "zero_verifier": "1*80 = 4*20 = 80", "steps_ratio": 4132 / 512}
x = json.loads((ROOT / "docs/course-experiments/results/reasoning.json").read_bytes())
result["recorded_environment"] = {key: x[key] for key in ("revision", "device", "gpu", "python_version", "torch_version", "seed", "modal")}
result["recorded_code_matches"] = {name: x["code_sha256"].get(name) == result["file_sha256"][name] for name in paths if name in x["code_sha256"]}
result["code_provenance"] = json.loads((Path(__file__).parent / "code-provenance.json").read_text())
check(digest(Path(__file__).parent / "applications-recorded.py") == x["code_sha256"]["scripts/course_experiments/applications.py"], "封存程式與實驗 SHA 不符")
check(all(value for name, value in result["recorded_code_matches"].items() if name != "scripts/course_experiments/applications.py"), "模型與 tokenizer 的實驗 SHA 不符")
result["split"] = x["results"]["split"]
result["model_config"] = x["results"]["model_config"]
result["budgets"] = []
for mode in ("direct", "steps"):
    questions = None
    for budget in x["results"]["comparison"][mode]["budgets"]:
        k = budget["candidate_count"]
        samples = budget["samples"]
        check(len(samples) == 24, "測試題數不符")
        current_questions = [row["question"] for row in samples]
        if questions is None:
            questions = current_questions
        check(current_questions == questions and len(set(current_questions)) == 24, "題目集合不一致")
        token_total, seconds_total, coverage, majority_correct, eos_count = 0, 0.0, 0, 0, 0
        caps, temperatures, durations = set(), set(), []
        for row in samples:
            truth = row["a"] + row["b"] + row["c"]
            check(truth == row["truth"], "已知答案不符")
            candidates = row["candidates"]
            g = row["generation"]
            check(len(candidates) == k == g["candidate_count"], "候選分母不符")
            caps.add(g["max_new_tokens"])
            temperatures.add(g["temperature"])
            per_row_tokens, final_answers = 0, []
            for c in candidates:
                ids = c["generated_ids"]
                check(len(ids) == c["generated_tokens"] <= g["max_new_tokens"], "原始 ID 計數不符")
                eos = bool(ids and ids[-1] == tok.eos_id)
                check(eos == c["eos"], "EOS 標記不符")
                raw = ids[:-1] if eos else ids
                check(tok.decode(raw) == c["generated"], "解碼文字與原始 ID 不符")
                check([v for v in raw if v < 8] == c["invalid_special_tokens"], "特殊 ID 不符")
                text = c["generated"].strip()
                m = re.fullmatch(r"-?[0-9]+", text) if mode == "direct" else re.search(r";answer=(-?[0-9]+)$", text)
                answer = int(m[0] if mode == "direct" else m[1]) if m and not c["invalid_special_tokens"] else None
                check(answer == c["final_answer"] and c["final_correct"] == (answer == truth), "最後答案核對不符")
                for key, value in app._verify_reasoning(c, row, mode).items():
                    check(c[key] == value, "驗證器輸出與儲存結果不符")
                if answer is not None:
                    final_answers.append(answer)
                per_row_tokens += len(ids)
                eos_count += eos
            majority = Counter(final_answers).most_common(1)[0][0] if final_answers else None
            covered = any(c["final_answer"] == truth for c in candidates)
            check(row["majority_answer"] == majority and row["majority_correct"] == (majority == truth), "多數決不符")
            check(row["oracle_coverage"] == covered, "集合命中不符")
            check(g["generated_tokens"] == per_row_tokens, "逐題 token 加總不符")
            check(g["input_ids"] == app._prompt_ids(g["messages"], tok), "輸入 ID 不符")
            token_total += per_row_tokens
            seconds_total += g["seconds"]
            durations.append(g["seconds"])
            coverage += covered
            majority_correct += majority == truth
        check(token_total == budget["generated_tokens"], "全體 token 加總不符")
        check(abs(seconds_total - budget["generation_seconds"]) < 1e-12, "逐題時間加總不符")
        check(coverage == budget["oracle_coverage"]["numerator"], "命中題數不符")
        check(majority_correct == budget["majority_accuracy"]["numerator"], "最終答對題數不符")
        check(budget["oracle_coverage"]["denominator"] == budget["majority_accuracy"]["denominator"] == 24, "比例分母不符")
        check(caps == ({8} if mode == "direct" else {48}), "每份生成上限不符")
        check(temperatures == {0.7}, "抽樣溫度不符")
        result["budgets"].append({"mode": mode, "k": k, "questions": 24, "candidates": 24*k, "generated_tokens": token_total, "eos_count": eos_count, "generation_seconds": seconds_total, "rounded_seconds": f"{seconds_total:.3f}", "oracle_coverage": coverage, "majority_correct": majority_correct, "max_new_tokens": next(iter(caps)), "temperature": 0.7, "stored_time_observations_per_question": 1})
result["sample_instrumentation"] = []
original_encode, original_decode = ByteTokenizer.encode, ByteTokenizer.decode
original_sync, original_clock = app._sync, app.time.perf_counter
for count in (1, 8):
    events = []
    def encoded(self, text):
        events.append({"event": "encode"})
        return original_encode(self, text)
    def decoded(self, ids):
        events.append({"event": "decode"})
        return original_decode(self, ids)
    def sync(ctx):
        events.append({"event": "sync"})
        return original_sync(ctx)
    def clock():
        events.append({"event": "clock"})
        return original_clock()
    ByteTokenizer.encode, ByteTokenizer.decode = encoded, decoded
    app._sync, app.time.perf_counter = sync, clock
    model = TinyLM(ModelConfig(width=8, heads=1, max_length=128))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    def forward_event(module, args, kwargs):
        events.append({"event": "forward", "shape": list(args[0].shape), "cache_reused": kwargs.get("cache") is not None})
    handle = model.register_forward_pre_hook(forward_event, with_kwargs=True)
    try:
        generated = app._sample(model, [{"role": "user", "content": "2+2=?"}], type("Context", (), {"device": "cpu"})(), count=count, tokens=3)
    finally:
        handle.remove()
        ByteTokenizer.encode, ByteTokenizer.decode = original_encode, original_decode
        app._sync, app.time.perf_counter = original_sync, original_clock
    names = [event["event"] for event in events]
    clocks = [i for i, event in enumerate(names) if event == "clock"]
    check(len(clocks) == 2 and names.index("encode") < clocks[0] < names.index("forward") < clocks[1] < names.index("decode"), "計時與編解碼邊界不符")
    check(all(event["shape"][0] == count for event in events if event["event"] == "forward"), "batch 維度不符")
    check(all(torch.equal(before[key], value) for key, value in model.state_dict().items()), "生成改動了模型參數")
    check(model.training, "模型模式未還原")
    result["sample_instrumentation"].append({"candidate_count": count, "events": events, "parameters_unchanged": True, "training_mode_restored": True, "measured_seconds_cpu": generated["seconds"], "scope": "僅核對本機行為與計時邊界，不作速度比較"})

class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skipped = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skipped += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skipped -= 1
    def handle_data(self, text):
        if not self.skipped:
            self.parts.append(text)

requests = [
    ("snell", "https://arxiv.org/html/2408.03314v1", "arXiv:2408.03314v1", "Introduction; How to Scale Test-Time Computation Optimally; Search Methods Against a PRM; Refining the Proposal Distribution", "by asking the base model to revise its original responses"),
    ("pope", "https://arxiv.org/html/2211.05102v1", "arXiv:2211.05102v1", "Introduction; Background; KV cache and batch/latency discussion", "generative inference proceeds one token at a time"),
    ("hf_cache", "https://huggingface.co/docs/transformers/v5.15.0/kv_cache", "Transformers v5.15.0", "Cache strategies, introductory explanation", "A KV cache stores these calculations so they can be reused without recomputing them."),
    ("nvidia_latency", "https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/", "NVIDIA Technical Blog, retrieved HTML SHA-256 recorded", "End-to-end request latency", "the end-to-end request latency is the time difference between when the request is sent and the final token is received"),
]
result["original_source_receipts"] = []
for identifier, url, version, locator, excerpt in requests:
    raw = subprocess.check_output(["curl", "-fsSL", "--max-time", "30", url])
    parser = Plain()
    parser.feed(raw.decode("utf-8"))
    text = " ".join(" ".join(parser.parts).split())
    check(excerpt in text, "原始來源摘錄未找到")
    result["original_source_receipts"].append({"id": identifier, "url": url, "version": version, "accessed_on": datetime.now(UTC).date().isoformat(), "retrieved_sha256": hashlib.sha256(raw).hexdigest(), "locator": locator, "excerpt": excerpt, "full_text_saved": False, "command": "curl -fsSL --max-time 30 " + url})
result["status"] = "passed"
print(json.dumps(result, ensure_ascii=False, indent=2))
