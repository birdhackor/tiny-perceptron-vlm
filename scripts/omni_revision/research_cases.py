"""成熟模型的小量原創材料：合成英文口語、RGB移時與短語音提示。"""

import argparse
import array
import hashlib
import json
import math
import random
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

PROMPTS = (
    ("Name this color: red. Answer with one word.", "red"),
    ("Name this color: blue. Answer with one word.", "blue"),
    ("Name this color: green. Answer with one word.", "green"),
    ("Name this color: yellow. Answer with one word.", "yellow"),
    ("Name this color: black. Answer with one word.", "black"),
    ("Name this color: white. Answer with one word.", "white"),
    ("Name this color: orange. Answer with one word.", "orange"),
    ("Name this color: purple. Answer with one word.", "purple"),
    ("What is two plus three? Say only the number.", "five"),
    ("What is four plus two? Say only the number.", "six"),
    ("What is one plus seven? Say only the number.", "eight"),
    ("What is three plus six? Say only the number.", "nine"),
    ("What is five minus three? Say only the number.", "two"),
    ("What is seven minus four? Say only the number.", "three"),
    ("What is nine minus five? Say only the number.", "four"),
    ("What is eight minus seven? Say only the number.", "one"),
    ("Is a square a shape? Answer yes or no.", "yes"),
    ("Is a circle a shape? Answer yes or no.", "yes"),
    ("Is blue a color? Answer yes or no.", "yes"),
    ("Is seven a number? Answer yes or no.", "yes"),
    ("Is red a number? Answer yes or no.", "no"),
    ("Is a square a color? Answer yes or no.", "no"),
    ("Is blue a number? Answer yes or no.", "no"),
    ("Is three a color? Answer yes or no.", "no"),
)
SPEECH_WORDS = ("red", "blue", "yes", "no", "one", "two", "hello", "goodbye")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcm(path, values, sample_rate=16000):
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(array.array("h", (round(max(-1, min(1, value)) * 32767) for value in values)).tobytes())


def build(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    (root / "audio").mkdir()
    (root / "video").mkdir()
    audio = []
    for index, (text, gold) in enumerate(PROMPTS):
        words = root / "audio" / f"{index:02}.txt"
        words.write_text(text + "\n")
        path = root / "audio" / f"{index:02}.wav"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"flite=textfile={words.resolve()}:voice=slt",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        with wave.open(str(path), "rb") as source:
            duration = source.getnframes() / source.getframerate()
            if not 0 < duration <= 8 or source.getsampwidth() != 2:
                raise ValueError("合成英文材料須完整且不超過8秒")
        audio.append(
            {
                "id": f"speech-pair-{index:02}",
                "transcript": text,
                "gold": gold,
                "audio": path.relative_to(root).as_posix(),
                "audio_sha256": sha(path),
                "duration_seconds": duration,
                "sample_rate": 16000,
                "source": "project-authored question synthesized by built-in FFmpeg/libflite slt; not a human speaker or naturalness judgment",
            }
        )
    patterns = list(range(1, 255))
    random.Random(20261029 + 4).shuffle(patterns)
    videos = []
    for index, pattern in enumerate(patterns[:24]):
        folder = root / "video" / f"{index:02}"
        folder.mkdir()
        bits = [(pattern >> (7 - i)) & 1 for i in range(8)]
        frames = []
        for frame, bit in enumerate(bits):
            path = folder / f"frame-{frame}.png"
            Image.new("RGB", (112, 112), (220, 20, 20) if bit == 0 else (20, 20, 220)).save(path)
            frames.append({"path": path.relative_to(root).as_posix(), "sha256": sha(path), "timestamp_ms": frame * 500})
        pair = []
        for color, label in ((0, "red"), (1, "blue")):
            position = next(i for i, value in enumerate(bits) if value == color)
            center = position * 0.5 + 0.25
            path = folder / f"audio-{color}.wav"
            begin, length = round((center - 0.04) * 16000), 1280
            values = [0.0] * 64000
            for t in range(length):
                values[begin + t] = 0.6 * math.sin(2 * math.pi * 440 * t / 16000)
            pcm(path, values)
            pair.append(
                {
                    "audio": path.relative_to(root).as_posix(),
                    "sha256": sha(path),
                    "beep_center_seconds": center,
                    "gold": label,
                }
            )
        videos.append(
            {
                "id": f"rgb-time-pair-{index:02}",
                "abstract_pattern": pattern,
                "frames": frames,
                "pair": pair,
                "fps": 2.0,
                "duration_seconds": 4,
                "question": "What color is visible when the beep happens? Answer only red or blue.",
                "lineage": "Same abstract eight-segment patterns also exist in micro-A's exhaustive254 universe; pixels/color world and full-clip audio are different. No independent-union n or architecture-causal comparison is claimed.",
            }
        )
    result = {
        "schema": "omni-mature-research-cases-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "language": "English primary; Chinese Kev results separate",
        "audio_text_pairs": audio,
        "av_shift_pairs": videos,
        "speech_output_prompts": [
            {"id": f"short-speech-{i}", "text": f"Say only the word {word}.", "gold_word": word}
            for i, word in enumerate(SPEECH_WORDS)
        ],
        "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "source_sha256": sha(__file__),
        "model_outputs_used": False,
        "human_listening_test": False,
    }
    (root / "cases.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.output)
    print(
        json.dumps(
            {
                "audio_text_pairs": len(value["audio_text_pairs"]),
                "av_shift_pairs": len(value["av_shift_pairs"]),
                "speech_prompts": len(value["speech_output_prompts"]),
                "cases_sha256": sha(args.output / "cases.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
