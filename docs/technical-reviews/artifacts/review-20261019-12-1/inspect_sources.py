"""以 curl 讀取原始來源，只保存短摘錄與取得收據。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag == "img" and not self.hidden:
            self.parts.append(dict(attrs).get("alt", ""))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


SOURCES = [
    ("sinusoids", "https://www.dsprelated.com/freebooks/mdft/Sinusoids.html",
     "A sinusoid is any function", "Why Sinusoids are Important",
     "cycles per second"),
    ("spectrum", "https://www.dsprelated.com/freebooks/sasp/Spectrum_Analysis_Sinusoids.html",
     "Sinusoidal components are fundamental", "Because spectral peaks are so relatively dominant",
     "Sinusoidal components are fundamental building blocks of sound."),
    ("sampling", "https://manual.audacityteam.org/man/digital_audio.html",
     "Digital recording works differently", "Sample formats",
     "it samples the waveform at evenly-spaced timepoints"),
    ("hearing", "https://openstax.org/books/college-physics-2e/pages/17-6-hearing",
     "Pitch perception is directly related", "The ear is remarkably sensitive",
     "Pitch perception is directly related to frequency"),
    ("padding", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pad_sequence.html",
     "Pad a list of variable length", "Parameters",
     "stacks a list of Tensors along a new dimension"),
    ("sin", "https://docs.pytorch.org/docs/2.14/generated/torch.sin.html",
     "Returns a new tensor with the sine", "Parameters",
     "where each value in this input tensor is in radians."),
    ("arange", "https://docs.pytorch.org/docs/2.14/generated/torch.arange.html",
     "Returns a 1-D tensor", "Parameters",
     "Returns a 1-D tensor"),
    ("tolist", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.tolist.html",
     "Returns the tensor as a", "Examples",
     "Returns the tensor as a (nested) list."),
    ("round", "https://docs.python.org/3.13/library/functions.html#round",
     "round (", "class set",
     "rounding is done toward the even choice"),
]


def main():
    receipts = []
    for identifier, url, start, end, excerpt in SOURCES:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "40", url]
        result = subprocess.run(command, capture_output=True, check=False)
        receipt = {
            "id": identifier, "url": url, "command_argv": command,
            "accessed_at": datetime.now(UTC).isoformat(), "exit_code": result.returncode,
            "stderr": result.stderr.decode("utf-8", errors="replace"),
        }
        if result.returncode == 0:
            parser = Text()
            parser.feed(result.stdout.decode("utf-8", errors="replace"))
            content = " ".join(" ".join(parser.parts).split())
            receipt["response_sha256"] = hashlib.sha256(result.stdout).hexdigest()
            receipt["response_bytes"] = len(result.stdout)
            receipt["excerpt"] = excerpt if excerpt in content else None
            begin = content.find(start)
            finish = content.find(end, begin + len(start)) if begin >= 0 else -1
            inspection = content[begin:finish] if begin >= 0 and finish > begin else content[max(0, begin):max(0, begin) + 5000]
            print(json.dumps({"id": identifier, "start_found": begin >= 0,
                              "inspection_text": inspection[:5000]}, ensure_ascii=False))
        receipts.append(receipt)
    destination = Path(__file__).with_name("source-receipts.json")
    destination.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
