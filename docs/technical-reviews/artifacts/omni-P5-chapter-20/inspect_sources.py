"""Save small independent source receipts for the introduction's claims."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path("/workspace/work/reference/omni-P5-chapter-20")


def record(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    receipts = json.loads((CACHE / "acquisition-receipt.json").read_text())
    excerpts = [
        {"name": "qwen3-vl-2b-card.md", "locator": "front matter pipeline_tag; title; Quickstart / Using Transformers to Chat", "quote": 'pipeline_tag: image-text-to-text', "inspection": "Read the official pinned model card's title and Quickstart. Its image-and-text user message is processed by Qwen3VLForConditionalGeneration; generate() output is decoded into output_text. This supports modality and pretrained-model identity, not a guarantee of correctness on arbitrary photos."},
        {"name": "whisper-turbo-card.md", "locator": "Whisper introduction; Usage; speech transcription versus translation", "quote": "Whisper large-v3-turbo is a finetuned version of a pruned [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3).", "inspection": "Read model introduction, the automatic-speech-recognition pipeline example returning result['text'], and lines 208–218 distinguishing speech transcription and translation. This confirms upstream trained ASR, not a chat-answer model or a Chinese quality guarantee."},
        {"name": "lora-v2.txt", "locator": "Section 4.1, Equation (3); Section 2 pre-trained parameter starting point", "quote": "During training, W0 is frozen and does not receive gradient updates, while A and B contain trainable", "inspection": "Read abstract, introduction, Section 2 and Section 4.1. Eq. (3) adds BAx to W0x. A is random and B zero initially; the update is alpha/r scaled. The introduction's 'base model' is a role for adaptation; it does not rename this Instruct checkpoint an upstream Base edition."},
    ]
    for item in excerpts:
        assert item["quote"] in (CACHE / item["name"]).read_text()
    local = [
        ("tiny_perceptron/natural_assistant.py", "load_core lines 263–304; train lines 527–554", "All pretrained parameters receive requires_grad_(False); LoRA is attached to language q/v modules. The optimizer receives only requires_grad parameters and rejects any name without lora_. This supports the training update policy. The historical base audit was sampled, not exhaustive."),
        ("tiny_perceptron/selftrained/model.py", "SelftrainedLanguageModel constructor lines 133–151; VisionEncoder, OCREncoder, AudioEncoder constructors lines 187–263; LimitedAssistant lines 295–303", "The neural components are constructed from own PyTorch modules; these constructors do not load third-party pretrained weights."),
        ("scripts/selftrained/train.py", "lines 731–766", "New runs construct LimitedAssistant. Continued stages accept only this pipeline's checkpoint schema, preserve provenance, and load the own preceding stage."),
        ("tiny_perceptron/lesson_products/model.py", "Block, LoopedSolver and CandidateScorer constructors lines 104–172", "The two product models construct their own embeddings, blocks and heads directly; no upstream pretrained encoder is used."),
        ("tiny_perceptron/lesson_products/train.py", "lines 70–124", "The new-run branch sets the seed and constructs LoopedSolver or CandidateScorer, recording initial state SHA; resume checks the original metadata. Saved execution receipts explicitly record random_initialization."),
        ("scripts/score_natural_v4_validation.py", "eos_complete lines 111–127; score_blind lines 398–447", "All-EOS completion is required for eligibility. Either candidate's three incomplete records already reject it, independent of subjective semantic grades. The intro's adoption claim does not require asserting general model inferiority."),
        ("tiny_perceptron/alignment.py", "LoRALinear lines 11–26", "The artificial 4-by-3 teaching layer freezes its base and adds alpha/r-scaled A/B correction; execution changes B while base stays equal."),
        ("tiny_perceptron/natural_concepts.py", "imports lines 6–9; edit_distance and text_error_report lines 47–67", "The only helper used by 20.10/20.12 computes edit distance on strings. It does not load Whisper or Qwen."),
        ("docs/natural-assistant/v4/TRAINING.md", "sections 3, 4, 6 and 7", "The intro link leads to a separate operational guide with pinned upstream model versions, LoRA training, validation, then frozen test. None of those training or evaluation commands were executed here."),
        ("docs/natural-assistant/v4/selection.json", "selected_variant; base_model", "The product selection metadata names base at the pinned Qwen revision."),
        ("docs/natural-assistant/v4/public-release.json", "selected_variant; base_model; asr_model; adapter_parameters", "The delivery manifest selects base with adapter_parameters=0, the same Qwen revision and pinned Whisper turbo. This checks declared delivery configuration, not live provider download bytes."),
        ("docs/lesson-products/results/settings-freeze-v1.json", "products.b; products.c", "B/C model and training-receipt hashes match their original execution client artifact entries."),
    ]
    local_receipts = [{**record(ROOT / path), "locator": locator, "inspection": note} for path, locator, note in local]
    mappings = []
    for chapter, lesson, purpose in [("09", "9.17", "candidate meaning scoring"), ("09", "9.18", "frozen calibration, policy and final roles"), ("12", "12.17", "shared audio/video time"), ("12", "12.18", "sound codes and reconstruction"), ("14", "14.19", "time/height/width positional axes"), ("16", "16.15", "material reuse and independent branches"), ("16", "16.16", "incremental waveform output"), ("17", "17.17", "residual vector quantization")]:
        path = ROOT / f"course/chapters/{chapter}.md"
        section = re.search(r"^## " + re.escape(lesson) + r"\s.*?(?=^## |\Z)", path.read_text(), re.M | re.S).group()
        mappings.append({"lesson": lesson, "source": record(path), "heading": section.splitlines()[0], "first_paragraph": section.split("\n\n")[1], "mapped_role": purpose, "scope": "read opening explanation to verify existing mainline location; no independent technical verdict for this other lesson"})
    training = ROOT / "outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review/training.json"
    tx = json.loads(training.read_text())
    initial, final = tx["initial_adapter_tensors"], tx["final_adapter_tensors"]
    assert set(initial) == set(final)
    changed = sum(initial[name]["sha256_values"] != final[name]["sha256_values"] for name in initial)
    assert changed == 112 == tx["changed_adapter_tensor_count"]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_chapter_20",
        "external_acquisition_receipts": receipts,
        "small_original_excerpts_and_actual_inspection": excerpts,
        "local_source_inspection": local_receipts,
        "earlier_mainline_location_check": mappings,
        "historical_adapter_fingerprint_recomputation": {"original": record(training), "tensor_name_sets_equal": True, "initial_and_final_tensor_count": len(initial), "recomputed_changed_tensor_count": changed, "scope": "compares saved per-tensor SHA strings only; does not reload trained tensors or claim a full base-weight bitwise audit"},
        "demonstration_scope": "All 14 exact Python fences were read, then executed in probe_intro.py. Their tasks are list equality, parameter-storage multiplication, family-set intersection, toy LoRA update, response-target masking, arithmetic validation gates, object/activity equality, edit distance, string line order, ASR-error arithmetic, file hashes, mutable branch copies, nominal MoE storage, and synthetic row scoring. Each isolates a teaching relation and none evaluates mature-model quality.",
        "unverified": ["browser desktop/mobile layout", "a new Qwen/Whisper/Kev/Omni inference", "live published asset retrieval", "new training or exact all-base-weight bitwise comparison", "general latency across hardware/platforms"],
    }
    out = OUT / "inspection-notes-v1.json"
    with out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(out.relative_to(ROOT)), **record(out), "changed_tensor_count": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
