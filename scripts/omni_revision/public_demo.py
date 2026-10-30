"""公開微型候選的CPU展示；保留舊成品入口，測試成績以密封報表為準。"""

import argparse
import json
import wave
from dataclasses import asdict, replace
from pathlib import Path

import torch

from scripts.omni_revision.public_download import ROOT, descriptor, verify_installed
from tiny_perceptron.omni_products.choice import Candidate, Question
from tiny_perceptron.omni_products.data import Event, av_pair, split_a
from tiny_perceptron.omni_products.inference import OmniAssistant, load_trained


def prediction(value):
    return {key: value[key] for key in ("answer", "ids", "logits", "probabilities", "p_max")}


def assistant(root, manifest):
    published = descriptor(manifest)
    selected = published["_model_selection"]
    for kind in ("av", "choice", "codec", "talker"):
        verify_installed(root, selected, kind)
    models = {
        kind: load_trained(Path(root) / "core" / selected["selected"][kind] / "training", kind)
        for kind in ("av", "choice", "codec", "talker")
    }
    return OmniAssistant(
        models["av"],
        models["choice"],
        models["codec"],
        models["talker"],
        temperature=selected["temperature"],
        threshold=selected["threshold"],
    ), published


def save_wave(path, samples):
    values = samples.detach().float().cpu().reshape(-1)
    if not 0 < len(values) <= 16000 or not torch.isfinite(values).all():
        raise ValueError("候選沒有產生合法的有限waveform")
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("請用新的WAV路徑，以保留前一次輸出")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (values.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    with path.open("xb") as stream, wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(pcm)
    return {"path": str(path), "sample_rate": 8000, "samples": len(values), "encoding": "PCM16"}


def public_question(value):
    if "gold" in value:
        raise ValueError("公開展示只收使用者材料，不接受gold欄位")
    return Question(
        value["state"], value["kind"], value["text"], tuple(Candidate(**row) for row in value["candidates"])
    ).validate()


def choice_example():
    state = "A: task=record; permission=yes; ready=yes. B: task=play; permission=no; ready=yes. C: task=save; permission=yes; ready=no."
    return {
        "state": state,
        "kind": "choice",
        "text": "Choose A's action. Require permission and ready known; no permission: stop; not ready: wait; else requested task.",
        "candidates": [
            {"id": name, "description": "Device A: " + name} for name in ("record", "play", "save", "wait", "stop")
        ],
        "information_question": {
            "state": state,
            "kind": "info",
            "text": "Are permission and ready both known for A's action?",
            "candidates": [
                {"id": "yes", "description": "Required information is sufficient"},
                {"id": "no", "description": "Required information is missing"},
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode", choices=("av", "choice", "sound"))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/omni-products/v1/public-release.json")
    parser.add_argument("--request", type=Path, help="choice使用公開typed question；sound使用Event物件的JSON清單")
    parser.add_argument("--family", type=int, help="AV自製八段形狀模式；省略時展示第一個train模式")
    parser.add_argument("--which", type=int, choices=(0, 1), default=0)
    parser.add_argument("--missing", choices=("none", "image", "audio", "both"), default="none")
    parser.add_argument("--output-wav", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    product, published = assistant(args.model_root, args.manifest)
    report = {
        "release_tag": published["release_tag"],
        "scope": "one explicit candidate demonstration, not final evaluation accuracy",
    }
    if args.mode == "choice":
        public = json.loads(args.request.read_bytes()) if args.request else choice_example()
        question = public_question(public)
        information = public_question(public["information_question"]) if "information_question" in public else None
        value = product.c.judge(question, information)
        report.update(
            prediction=prediction(value["prediction"]), accepted=value["accepted"], decision=value["decision"]
        )
        if "information" in value:
            report["information"] = prediction(value["information"])
        if "policy_supported" in value:
            report["policy_supported"] = value["policy_supported"]
    elif args.mode == "av":
        family = split_a()["train"][0] if args.family is None else args.family
        item = av_pair(family)[args.which][0]
        item = replace(
            item,
            image_present=args.missing not in ("image", "both"),
            audio_present=args.missing not in ("audio", "both"),
        )
        value = product.video(item, with_sound=True)
        sound = value.pop("sound", None)
        report.update(input={"family": family, "pair_index": args.which, "missing": args.missing}, **value)
        if sound is not None:
            report["sound"] = {
                "condition_program": [asdict(event) for event in sound["program"]],
                "eos": sound["eos"],
                "capped": sound["capped"],
                "codes": len(sound["codes"]),
                "samples": len(sound["waveform"]),
                "sample_rate": 8000,
            }
            if args.output_wav and sound["eos"] and not sound["capped"]:
                report["waveform"] = save_wave(args.output_wav, sound["waveform"])
    else:
        program = (
            tuple(Event(**row) for row in json.loads(args.request.read_bytes()))
            if args.request
            else (Event(220, 200, 100), Event(440, 200))
        )
        chunks = list(product.sound_chunks(program))
        samples = torch.cat([row["samples"] for row in chunks])
        report.update(
            requested_program=[asdict(event) for event in program],
            samples=len(samples),
            sample_rate=8000,
            chunks=len(chunks),
            internal_first_chunk_ready_seconds=chunks[0]["ready_seconds"],
            whole_code_generation_seconds=chunks[0]["whole_code_generation_seconds"],
            timing_scope="all autoregressive codes complete before internal chunk decoding; excludes CLI WAV writing and playback",
        )
        if args.output_wav:
            report["waveform"] = save_wave(args.output_wav, samples)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
