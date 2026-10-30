"""固定Kev-4B的有限決策／cache研究；不是本課C的公平架構對照。"""

import hashlib
import json
import re
import time
from pathlib import Path

import torch

from scripts.omni_revision.core_evaluation import evaluate_c
from scripts.omni_revision.worker_outputs import json_file
from tiny_perceptron.omni_products.reliability import select_policy, select_temperature

ADAPTER = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
BASE = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def state_view(state, language):
    patterns = (
        r"([ABC]): task=(record|play|save); permission=(yes|no|unknown); ready=(yes|no|unknown)\.",
        r"Device ([ABC]) requests (record|play|save)\. Its permission is (yes|no|unknown); its readiness is (yes|no|unknown)\.",
    )
    rows = []
    for pattern in patterns:
        rows = re.findall(pattern, state)
        if rows:
            break
    if not rows:
        rows = [
            (name, task, permission, ready)
            for name, task, ready, permission in re.findall(
                r"Status for ([ABC]): requested action (record|play|save), readiness (yes|no|unknown), permission (yes|no|unknown)\.",
                state,
            )
        ]
    if not rows:
        convert = {"是": "yes", "否": "no", "未知": "unknown"}
        rows = [
            (name, task, convert[permission], convert[ready])
            for name, task, permission, ready in re.findall(
                r"裝置([ABC])：任務(record|play|save)；許可(是|否|未知)；就緒(是|否|未知)。", state
            )
        ]
    if [row[0] for row in rows] != ["A", "B", "C"]:
        raise ValueError("僅轉寫可見三裝置欄位，不讀private family/gold補資料")
    if language == "english":
        if "裝置" not in state:
            return state
        return " ".join(
            f"{name}: task={task}; permission={permission}; ready={ready}." for name, task, permission, ready in rows
        )
    if language != "chinese":
        raise ValueError("未知state語言")
    convert = {"yes": "是", "no": "否", "unknown": "未知"}
    return " ".join(
        f"裝置{name}：任務{task}；許可{convert[permission]}；就緒{convert[ready]}。"
        for name, task, permission, ready in rows
    )


def public_record(row, language, kinds=("choice", "info", "rubric", "choice_info")):
    questions = []
    ids = []
    for kind in kinds:
        question = row["questions"][kind]
        names = [item["id"] for item in question["candidates"]]
        options = [item["description"] for item in question["candidates"]]
        if question["kind"] == "choice":
            names.append("none")
            options.append("None of these actions can be selected from the supplied facts and required-field contract.")
        questions.append({"instr": question["text"], "options": options, "label": None})
        ids.append(names)
    return {"state": state_view(row["questions"][kinds[0]]["state"], language), "questions": questions}, ids


def encode(model, tokenizer, record):
    value = model.encode(tokenizer, record, max_state=384, max_branch=1024, strict=True)
    if value["state_truncated"] or len(value["ids"]) > 2048:
        raise ValueError("固定Kev研究context超界，不能截斷")
    return value


@torch.inference_mode()
def predict_family(model, tokenizer, row, language):
    kinds = ("choice", "info", "rubric", "choice_info")
    public, ids = public_record(row, language, kinds)
    encoded = encode(model, tokenizer, public)
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model.forward(encoded)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    predictions = {}
    for kind, names, value in zip(kinds, ids, logits):
        values = value.detach().float().cpu().tolist()
        if len(values) != len(names) or not torch.isfinite(value).all():
            raise ValueError("Kev logits非有限或候選數不同")
        best = min(range(len(names)), key=lambda i: (-values[i], names[i]))
        predictions[kind] = {"ids": names, "logits": values, "answer": names[best]}
    return {
        "family": row["family"],
        "source": row["source"] + "/" + language + "-state-view",
        "panel": row["panel"],
        "tasks": {
            kind: {**predictions[kind], "gold": row["questions"][kind]["gold"]} for kind in ("choice", "info", "rubric")
        },
        "choice_info": predictions["choice_info"],
        "world_supported": row["world_supported"],
        "choice_information_sufficient": row["choice_information_sufficient"],
        "context_tokens": len(encoded["ids"]),
        "state_tokens": encoded["state_tokens"],
        "forward_seconds": seconds,
    }


def prefix_fingerprint(prefix):
    digest = hashlib.sha256()

    def add(value):
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            digest.update(str((x.shape, x.dtype)).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                add(value[key])
        else:
            digest.update(repr(value).encode())

    add(prefix[0])
    add(prefix[2])
    for layer in prefix[1].layers:
        for key in (
            "keys",
            "values",
            "conv_states",
            "recurrent_states",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
            "has_previous_state",
            "conv_kernel_size",
        ):
            digest.update(key.encode())
            add(getattr(layer, key, None))
    return digest.hexdigest()


@torch.inference_mode()
def cache_diagnostics(model, tokenizer, rows):
    results = []
    for row in rows:
        record, _ = public_record(row, "english", ("choice",))
        other, _ = public_record(row, "english", ("info",))
        q1, q2 = encode(model, tokenizer, record), encode(model, tokenizer, other)
        uncached = torch.softmax(model.forward(q1)[0].float(), -1).cpu()
        first, prefix = model.probs_and_prefix(q1)
        before = prefix_fingerprint(prefix)
        model.probs_with_prefix(q2, prefix)
        repeated = model.probs_with_prefix(q1, prefix)
        results.append(
            {
                "family": row["family"],
                "prefix_before": before,
                "prefix_after": prefix_fingerprint(prefix),
                "uncached_cached_max_abs": float((uncached - first[0]).abs().max()),
                "Q1_Q2_Q1_probability_max_abs": float((first[0] - repeated[0]).abs().max()),
            }
        )
    return {
        "development_families": len(rows),
        "atol": 1e-5,
        "rows": results,
        "all_observed_within_tolerance": all(
            row["prefix_before"] == row["prefix_after"]
            and row["uncached_cached_max_abs"] <= 1e-5
            and row["Q1_Q2_Q1_probability_max_abs"] <= 1e-5
            for row in results
        ),
    }


def main():
    from kev.checkpoint import Checkpoint, LoadOptions

    job = json.loads(Path("job.json").read_bytes())
    data = json.loads(Path("data-definition.json").read_bytes())
    output = Path("result")
    output.mkdir(exist_ok=True)
    checkpoint = Checkpoint(ADAPTER)
    if (
        checkpoint.meta.base != BASE
        or checkpoint.meta.base_revision != BASE_REVISION
        or checkpoint.meta.option_isolation is not False
        or checkpoint.meta.weights_dtype != "fp32"
        or checkpoint.meta.lora != 16
        or checkpoint.meta.head_dim != 256
        or hashlib.sha256(checkpoint.file("head.pt").read_bytes()).hexdigest()
        != "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c"
        or hashlib.sha256(checkpoint.file("adapter_config.json").read_bytes()).hexdigest()
        != "8a05dfd6c5e7f61a62e6db5d8de8093a8dbbd105aa7ee7b5e7b73f6249f45617"
    ):
        raise ValueError("下載的Kev metadata與事前pin不符")
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer, model = checkpoint.load(
        "cuda",
        LoadOptions(dtype=torch.float32, temperature=1.0, merge=True, backend="torch", cuda_graphs=False, fused=False),
    )
    if model.head.temperature != 1.0:
        raise ValueError("raw logits不可先除checkpoint溫度")

    def collect(rows, split, language):
        predictions = []
        for row in rows:
            value = predict_family(model, tokenizer, row, language)
            with (output / "kev-raw-predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"split": split, "language": language, "prediction": value}, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                )
            predictions.append(value)
        return predictions

    calibration = collect(data["C"]["panels"]["calibration"], "calibration", "english")
    json_file(output / "kev-calibration.json", calibration)
    selected_t = select_temperature(calibration)
    policy = collect(data["C"]["panels"]["policy"], "policy", "english")
    json_file(output / "kev-policy-predictions.json", policy)
    selected_policy = select_policy(policy, selected_t["temperature"])
    selection = {
        "temperature": selected_t,
        "policy": selected_policy,
        "upstream_temperature": checkpoint.meta.temperature,
        "head_temperature_during_raw_logits": 1.0,
        "core_selection_sha256": job["core_selection_sha256"],
        "final_opened": False,
    }
    json_file(output / "kev-reliability-selection.json", selection)
    controls = cache_diagnostics(model, tokenizer, data["C"]["panels"]["dev"][:8])
    final = {}
    for language in ("english", "chinese"):
        rows = collect(data["C"]["panels"]["final"], "final", language)
        final[language] = {
            "raw": evaluate_c(rows, 1.0, None),
            "upstream_temperature": evaluate_c(rows, checkpoint.meta.temperature, None),
            "new_temperature_policy": evaluate_c(rows, selected_t["temperature"], selected_policy["threshold"]),
        }
        json_file(output / f"kev-final-{language}.json", final[language])
    value = {
        "schema": "omni-kev-limited-research-v1",
        "adapter": ADAPTER,
        "base": BASE,
        "base_revision": BASE_REVISION,
        "dtype": "float32 loaded directly as reference; no bf16-upcast reference",
        "option_isolation": False,
        "cache": controls,
        "selection": selection,
        "families_per_language": 60,
        "language_views_share_same60_families": True,
        "language_scope": "Only state rendering translated from public visible fields; questions remain English. These are project-authored rule-worlds, not independent corpora/pretraining-unseen data.",
        "own_C_fair_architecture_comparison": False,
        "own_training_used_Kev_outputs": False,
        "final_opened": True,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "completed": True,
    }
    json_file(output / "research-result.json", value)


if __name__ == "__main__":
    main()
