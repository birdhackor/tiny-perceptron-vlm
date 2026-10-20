"""核對凍結教材、短例子與保存的模型選擇證據。"""

import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def save(name, value):
    path = ARTIFACT / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def originals():
    records = []
    definitions = [
        (
            "qwen",
            "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/README.md",
            "89644892e4d85e24eaac8bacfd4f463576704203",
            "This is the weight repository for Qwen3-VL-2B-Instruct.",
            "pipeline_tag: image-text-to-text",
            "Model Architecture Updates 後的權重說明，以及 Using Transformers to Chat 的圖文輸入與 generate 範例",
        ),
        (
            "whisper",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md",
            "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
            "Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation",
            'print(result["text"])',
            "Whisper 開頭的任務及訓練說明、turbo 來源，以及 Usage 的音訊輸入與文字輸出",
        ),
        (
            "lora",
            "https://arxiv.org/abs/2106.09685v2",
            "arXiv:2106.09685v2",
            "which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture",
            "",
            "Abstract：凍結預訓練權重、加入可訓練的低秩分解矩陣；未引用該摘要的效益倍數",
        ),
    ]
    for name, url, version, excerpt, extra, locator in definitions:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
        result = subprocess.run(command, capture_output=True, check=True)
        body = result.stdout.decode()
        assert excerpt in body
        assert not extra or extra in body
        records.append(
            {
                "id": name,
                "url": url,
                "version": version,
                "accessed_on": datetime.now().astimezone().date().isoformat(),
                "command": command,
                "exit_code": result.returncode,
                "raw_bytes": len(result.stdout),
                "raw_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "locator": locator,
                "excerpt": excerpt,
                "additional_excerpt": extra,
                "checked_original": True,
                "inspection_note": "原文以 curl 取得並閱讀；只保存支持主張的小段摘錄，完整下載內容留在程序記憶體。",
            }
        )
    save("original-sources.json", {"sources": records})


def snippets(freeze):
    runs = []
    for page in freeze["pages"]:
        if not page["page_id"].startswith("20."):
            continue
        assert digest(page["snapshot"]) == page["source_sha256"]
        for code in re.findall(r"```python\n(.*?)```", (ROOT / page["snapshot"]).read_text(), re.S):
            ast.parse(code)
            wrapper = '''
import contextlib, io, json, socket, sys, time
def reject_network(*args, **kwargs):
    raise RuntimeError("短例子不需要網路")
socket.create_connection = reject_network
socket.socket.connect = reject_network
import torch
torch.set_num_threads(2)
torch.set_default_device("cpu")
namespace = {}
output = io.StringIO()
started = time.perf_counter()
with contextlib.redirect_stdout(output):
    exec(CODE, namespace)
elapsed = time.perf_counter() - started
devices = set()
for value in namespace.values():
    if isinstance(value, torch.Tensor):
        devices.add(str(value.device))
    if isinstance(value, torch.nn.Module):
        devices.update(str(p.device) for p in value.parameters())
print(json.dumps({"stdout": output.getvalue(), "snippet_seconds": elapsed, "tensor_devices": sorted(devices), "external_model_modules": [x for x in sys.modules if x.split(".")[0] in {"transformers", "whisper", "peft"}]}, ensure_ascii=False))
'''.replace("CODE", repr(code))
            env = dict(os.environ)
            env.update(
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(ROOT),
                CUDA_VISIBLE_DEVICES="",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            started = time.perf_counter()
            process = subprocess.run(
                [sys.executable, "-c", wrapper], cwd=ROOT, env=env, capture_output=True, text=True, timeout=45
            )
            elapsed = time.perf_counter() - started
            assert process.returncode == 0, process.stderr
            result = json.loads(process.stdout)
            assert not result["external_model_modules"]
            assert set(result["tensor_devices"]) <= {"cpu"}
            runs.append(
                dict(
                    page_id=page["page_id"],
                    snapshot=page["snapshot"],
                    source_sha256=page["source_sha256"],
                    code=code,
                    code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    subprocess_seconds=elapsed,
                    exit_code=process.returncode,
                    stderr=process.stderr,
                    **result,
                )
            )
    return runs


def repository():
    import torch
    from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer
    from tiny_perceptron.lesson_products.training import state_digest
    from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

    torch.set_num_threads(2)
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    freeze = read_json(freeze_path)
    header = next(p for p in freeze["pages"] if p["page_id"] == "chapter-20")
    assert digest(header["snapshot"]) == header["source_sha256"]
    assert header["figures_sha256"] == {}
    tokenizer = Tokenizer()
    initialization = {}
    for product, seed, cls in [("b", 7421, LoopedSolver), ("c", 9041, CandidateScorer)]:
        receipt_path = f"docs/lesson-products/results/initial-{product}-execution.json"
        receipt = read_json(receipt_path)
        torch.manual_seed(seed)
        config = Config(len(tokenizer), width=128 if product == "b" else 64, ff=256 if product == "b" else 128)
        model = cls(config)
        actual = state_digest(model.state_dict())
        expected = receipt["training"]["initial_weights_sha256"]
        torch.manual_seed(seed)
        repeated = state_digest(cls(config).state_dict())
        assert actual == repeated
        assert receipt["training"]["random_initialization"] is True
        assert receipt["training"]["resume_source"] is None
        initialization[product] = {
            "receipt": receipt_path,
            "seed": seed,
            "expected_initial_state_sha256": expected,
            "observed_initial_state_sha256": actual,
            "repeated_initial_state_sha256": repeated,
            "matches_historical_initial_state": actual == expected,
            "historical_environment": receipt["training"]["environment"],
            "scope": "核對當前建構器的隨機初值與同 seed 重現，以及歷史執行的 random_initialization/resume_source；當前環境與歷史初始權重的逐位元一致性另記結果。",
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    moe_states = []
    for seed in [20261006, 20261007, 20261006]:
        torch.manual_seed(seed)
        model = LimitedAssistant(SelftrainedConfig(vocab_size=550, width=256, ffn_hidden=512, top_k=2))
        moe_states.append(state_digest(model.state_dict()))
    assert moe_states[0] == moe_states[2] and moe_states[0] != moe_states[1]
    initialization["moe"] = {
        "manifest_initialization": read_json("docs/selftrained/v2-manifest.json")["initialization"],
        "seed_sequence": [20261006, 20261007, 20261006],
        "observed_initial_state_sha256": moe_states,
        "scope": "建構器的隨機初值與可重現性；歷史 MoE 訓練來源另依 manifest、訓練入口與自有 checkpoint 契約核對。",
    }
    base = "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/"
    scores = read_json(base + "scored/scores.json")
    protocol_path = "docs/natural-assistant/v4/validation-protocol-lower-lr.json"
    protocol = read_json(protocol_path)
    spec = importlib.util.spec_from_file_location("validation_scoring", ROOT / "scripts/score_natural_v4_validation.py")
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    results = {}
    base_counts = scores["variants"]["base"]["correct_counts"]
    for variant, record in scores["variants"].items():
        generations_path = base + f"raw/generations-{variant}.json"
        generations = read_json(generations_path)
        assert len(generations) == protocol["validation_denominators"]["lm_generations_per_candidate"]
        incomplete = [scoring.case_id(row["id"], row["task"]) for row in generations if not scoring.eos_complete(row, protocol["generation"]["max_new_tokens"])]
        assert sorted(incomplete) == sorted(record["incomplete_case_ids"])
        counts = record["correct_counts"]
        numerator = sum(counts[group] * weight for group, weight in scoring.WEIGHTS.items())
        assert numerator == record["primary_numerator"]
        gates = {}
        if variant != "base":
            for group in scoring.WEIGHTS:
                rule = protocol["adapter_nonregression_gates_vs_base"][group + "_correct_count_minimum"]
                minimum = max(0, base_counts[group] - int(rule == "base minus 1"))
                gates[group] = {"minimum": minimum, "actual": counts[group], "passed": counts[group] >= minimum}
            assert gates == record["gates"]
        results[variant] = {
            "correct_counts": counts,
            "numerator": numerator,
            "denominator": scoring.DENOMINATOR,
            "generation_count": len(generations),
            "incomplete_count": len(incomplete),
            "gates": gates,
            "eligible": not incomplete and all(g["passed"] for g in gates.values()),
        }
    selection = "base"
    for variant in protocol["candidate_order"][1:]:
        if results[variant]["eligible"] and results[variant]["numerator"] > results[selection]["numerator"]:
            selection = variant
    selected = read_json("docs/natural-assistant/v4/selection.json")
    release = read_json("docs/natural-assistant/v4/public-release.json")
    assert selection == scores["selected_variant"] == selected["selected_variant"] == release["selected_variant"] == "base"
    assert release["adapter_parameters"] == 0
    assert digest(protocol_path) == selected["validation_protocol_sha256"]
    assert digest(base + "scored/scores.json") == selected["validation_scoring_sha256"]
    training = read_json("docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json")
    archives = training["independent_audit"]["archives"]
    assert [a["step"] for a in archives] == protocol["training"]["checkpoint_updates"]
    paths = [
        freeze_path, header["snapshot"],
        "docs/selftrained/v2-manifest.json", "scripts/selftrained/train.py",
        "tiny_perceptron/selftrained/model.py", "tiny_perceptron/model.py",
        "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/train.py",
        "docs/lesson-products/results/initial-b-execution.json",
        "docs/lesson-products/results/initial-c-execution.json",
        "tiny_perceptron/natural_assistant.py", "tiny_perceptron/alignment.py",
        "tiny_perceptron/data.py", "tiny_perceptron/natural_concepts.py",
        "tiny_perceptron/multimodal.py", "scripts/score_natural_v4_validation.py",
        "docs/natural-assistant/v4/selection.json", "docs/natural-assistant/v4/public-release.json",
        "docs/natural-assistant/v4/TRAINING.md", protocol_path,
        "docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json",
        base + "scored/scores.json",
    ] + [base + f"raw/generations-{variant}.json" for variant in scores["variants"]]
    runs = snippets(freeze)
    save(
        "verification.json",
        {
            "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-20/verify.py repository",
            "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "threads": "2"},
            "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_hashes": {p: digest(p) for p in paths},
            "initialization": initialization,
            "training_archives": archives,
            "selection_recalculation": results,
            "selected_variant": selection,
            "release": {key: release[key] for key in ["selected_variant", "base_model", "asr_model", "adapter_parameters"]},
            "python_examples": runs,
            "timing_scope": "每段一個新 Python 子程序；subprocess_seconds 包含 torch 載入，snippet_seconds 只含原程式格；固定 CPU、兩個執行緒，未安裝套件或下載模型。",
            "limitations": "未重跑完整 GPU 訓練，未逐位元組比較全部 Qwen 底座；選版重算採既有逐題語義評分，未重新評讀全部照片；未測任意硬體的時間上限。",
        },
    )


if __name__ == "__main__":
    {"originals": originals, "repository": repository}[sys.argv[1]]()
