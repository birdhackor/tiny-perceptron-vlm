"""固定3B的24語音/文字、24影音配對與8短語音輸出；沒有自家訓練。"""

import gc
import hashlib
import json
import string
import time
import wave
from pathlib import Path

import soundfile as sf
import torch
from PIL import Image

from scripts.omni_revision.qwen_inputs import MODEL, REVISION, input_receipt, make_inputs
from scripts.omni_revision.research_model_files import component_parameter_dtypes, frozen_snapshot
from scripts.omni_revision.worker_outputs import json_file

ASR = "openai/whisper-small"
ASR_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def normalized(value):
    text = " ".join(value.strip().lower().strip(string.punctuation + " ").split())
    return NUMBER_WORDS[int(text)] if text.isdigit() and 0 <= int(text) < len(NUMBER_WORDS) else text


def audio_file(path, samples):
    values = samples.detach().float().cpu().reshape(-1)
    legal = bool(0 < len(values) <= 192000 and torch.isfinite(values).all())
    result = {
        "sample_rate": 24000,
        "samples": len(values),
        "legal": legal,
        "float32_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }
    if len(values) and torch.isfinite(values).all():
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (values.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(pcm)
        result.update(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            encoding="PCM16 export; no claim of human listening or naturalness",
        )
    return result


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def infer(model, processor, *, text=None, audio=None, frames=None, speech=False, seed=20261029):
    inputs = make_inputs(processor, text=text, audio=audio, frames=frames)
    receipt = input_receipt(inputs)
    count = inputs["input_ids"].shape[-1]
    inputs = inputs.to("cuda").to(model.dtype)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(
        **inputs,
        return_audio=speech,
        use_audio_in_video=frames is not None,
        thinker_max_new_tokens=16 if speech else 32,
        talker_max_new_tokens=192,
        thinker_do_sample=False,
        talker_do_sample=False,
        speaker="Chelsie",
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if speech:
        tokens, waveform = generated
    else:
        tokens, waveform = generated, None
    continuation = tokens[0, count:]
    text = processor.tokenizer.decode(continuation, skip_special_tokens=True)
    return {
        "text": text,
        "generated_ids": continuation.cpu().tolist(),
        "new_tokens": len(continuation),
        "whole_completion_seconds": elapsed,
        "input": receipt,
        "seed": seed,
        "generation": {
            "return_audio": speech,
            "use_audio_in_video": frames is not None,
            "thinker_max_new_tokens": 16 if speech else 32,
            "talker_max_new_tokens": 192,
            "thinker_do_sample": False,
            "talker_do_sample": False,
            "speaker": "Chelsie",
        },
    }, waveform


def main():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    root = Path("cases")
    cases = json.loads((root / "cases.json").read_bytes())
    output = Path("result")
    output.mkdir(exist_ok=True)
    model_root, model_files = frozen_snapshot(MODEL, REVISION)
    json_file(output / "qwen-frozen-input-files.json", model_files)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(model_root), local_files_only=True, token=False, use_fast=False
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_root),
        local_files_only=True,
        token=False,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    if not model.has_talker:
        raise ValueError("固定完整3B checkpoint缺Talker，不能冒稱做語音輸出")
    parameter_dtypes_after_load = component_parameter_dtypes(model)
    rows = []
    pairs = []
    for index, case in enumerate(cases["audio_text_pairs"]):
        wave, sr = sf.read(root / case["audio"], dtype="float32")
        if sr != 16000:
            raise ValueError("研究輸入須16kHz")
        pair = {"id": case["id"], "gold": case["gold"]}
        for modality, kwargs in (("audio", {"audio": wave}), ("text", {"text": case["transcript"]})):
            row = {
                "panel": "audio-text",
                "id": case["id"],
                "modality": modality,
                "gold": case["gold"],
                "correct": False,
            }
            try:
                prediction, _ = infer(model, processor, **kwargs, seed=20261029 + index)
                row.update(prediction=prediction, correct=normalized(prediction["text"]) == case["gold"])
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error)[:1000])
            append(output / "qwen-raw-predictions.jsonl", row)
            rows.append(row)
            pair[modality] = row["correct"]
        pair["both_correct"] = pair["audio"] and pair["text"]
        pairs.append(pair)
    av = []
    for index, case in enumerate(cases["av_shift_pairs"]):
        frames = [Image.open(root / item["path"]).convert("RGB") for item in case["frames"]]
        pair = {"id": case["id"], "observations": []}
        for which, item in enumerate(case["pair"]):
            wave, sr = sf.read(root / item["audio"], dtype="float32")
            if sr != 16000:
                raise ValueError("影音研究輸入須16kHz")
            row = {"panel": "AV-shift", "id": case["id"], "which": which, "gold": item["gold"], "correct": False}
            try:
                prediction, _ = infer(
                    model, processor, text=case["question"], audio=wave, frames=frames, seed=20262029 + index
                )
                row.update(prediction=prediction, correct=normalized(prediction["text"]) == item["gold"])
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error)[:1000])
            append(output / "qwen-raw-predictions.jsonl", row)
            rows.append(row)
            pair["observations"].append(row["correct"])
        pair["both_correct"] = all(pair["observations"])
        av.append(pair)
    spoken = []
    for index, case in enumerate(cases["speech_output_prompts"]):
        row = {"id": case["id"], "gold_word": case["gold_word"], "generated": False}
        try:
            prediction, waveform = infer(model, processor, text=case["text"], speech=True, seed=20263029 + index)
            row.update(
                prediction=prediction, audio=audio_file(output / "speech" / f"{index}.wav", waveform), generated=True
            )
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error)[:1000])
        append(output / "qwen-speech-raw.jsonl", row)
        spoken.append(row)
    parameter_dtypes_after_speech = component_parameter_dtypes(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from scipy.signal import resample_poly
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    asr_root, asr_files = frozen_snapshot(ASR, ASR_REVISION)
    json_file(output / "asr-frozen-input-files.json", asr_files)
    asr_processor = WhisperProcessor.from_pretrained(str(asr_root), local_files_only=True, token=False)
    asr = (
        WhisperForConditionalGeneration.from_pretrained(
            str(asr_root), local_files_only=True, token=False, torch_dtype=torch.float16
        )
        .to("cuda")
        .eval()
    )
    for row in spoken:
        row["ASR_proxy_attempted"] = False
        if not row["generated"] or not row["audio"]["legal"]:
            row["ASR_proxy_correct"] = None
            row["ASR_proxy_skip_reason"] = "generation failed or output violated fixed waveform bounds"
            append(output / "qwen-asr-raw.jsonl", row)
            continue
        try:
            row["ASR_proxy_attempted"] = True
            values, sr = sf.read(row["audio"]["path"], dtype="float32")
            if sr != 24000:
                raise ValueError("語音輸出24kHz契約不同")
            features = asr_processor(
                resample_poly(values, 2, 3), sampling_rate=16000, return_tensors="pt"
            ).input_features.to("cuda", torch.float16)
            with torch.inference_mode():
                tokens = asr.generate(features, language="en", task="transcribe", max_new_tokens=32, do_sample=False)
            text = asr_processor.batch_decode(tokens, skip_special_tokens=True)[0]
            row.update(ASR_proxy_text=text, ASR_proxy_correct=normalized(text) == row["gold_word"])
        except Exception as error:
            row.update(
                ASR_proxy_correct=False, ASR_proxy_error_type=type(error).__name__, ASR_proxy_error=str(error)[:500]
            )
        append(output / "qwen-asr-raw.jsonl", row)
    value = {
        "schema": "omni-qwen25-limited-research-v1",
        "model": MODEL,
        "revision": REVISION,
        "dtype": "BF16 requested neural load; official Token2Wav converts to FP32 when decoding speech; no quantization",
        "parameter_numel_by_dtype_after_load": parameter_dtypes_after_load,
        "parameter_numel_by_dtype_after_speech": parameter_dtypes_after_speech,
        "precision_scope": "Observed parameter dtypes per component; not a claim that every operator/activation uses the same precision. Peak CUDA includes the actual speech path.",
        "audio_text_pairs": pairs,
        "AV_shift_pairs": av,
        "speech_outputs": spoken,
        "audio_text_counts": {"families": len(pairs), "both_correct": sum(row["both_correct"] for row in pairs)},
        "AV_counts": {"families": len(av), "both_correct": sum(row["both_correct"] for row in av)},
        "speech_count": len(spoken),
        "speech_counts": {
            "prompts": len(spoken),
            "generated": sum(row["generated"] for row in spoken),
            "legal": sum(row.get("audio", {}).get("legal", False) for row in spoken),
            "ASR_proxy_attempted": sum(row["ASR_proxy_attempted"] for row in spoken),
            "ASR_proxy_correct_out_of_all8": sum(row.get("ASR_proxy_correct") is True for row in spoken),
        },
        "ASR_proxy": {
            "model": ASR,
            "revision": ASR_REVISION,
            "scope": "word-content proxy only, not naturalness/human listening/MOS",
        },
        "latency_scope": "whole generate completion including speech output; API does not yield playable chunks, no streaming TTFA claim",
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "teacher_outputs_used_for_own_training": False,
        "final_opened": True,
        "completed": True,
    }
    json_file(output / "research-result.json", value)


if __name__ == "__main__":
    main()
