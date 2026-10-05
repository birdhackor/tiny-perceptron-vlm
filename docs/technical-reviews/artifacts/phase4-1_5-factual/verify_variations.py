"""Independent CPU audit: handwritten transition inventory and bounded changes."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import platform
import sys
import torch

root = Path(__file__).resolve().parent
torch.set_default_device("cpu")
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version, "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu", "platform": platform.platform(),
    "threads": str(torch.get_num_threads()),
}
(root / "variations-environment.json").write_text(json.dumps(environment, indent=2) + "\n")

# Literal inventory read directly from the twelve printed characters, including punctuation.
pairs = [
    ("貓", "看"), ("看", "狗"), ("狗", "。"), ("。", "狗"),
    ("狗", "吃"), ("吃", "魚"), ("魚", "。"), ("。", "貓"),
    ("貓", "看"), ("看", "鳥"), ("鳥", "。"),
]
golden = Counter(pairs)
golden_chars = ["。", "吃", "狗", "看", "貓", "魚", "鳥"]
original = root / "original" / "fence-1.py"
scope = {"__name__": "__main__"}
exec(compile(original.read_bytes(), str(original), "exec"), scope)
text, chars, counts, probability = [scope[k] for k in ("text", "chars", "counts", "probability")]
assert len(text) == 12 and len(pairs) == 11 and len(chars) == 7
assert chars == golden_chars
for i, current in enumerate(chars):
    for j, answer in enumerate(chars):
        assert counts[i, j].item() == golden[(current, answer)] + 1
assert counts.sum().item() == 49 + 11 == 60
assert counts.shape == (7, 7) and counts.sum(dim=1, keepdim=True).shape == (7, 1)
assert counts.sum(dim=1).tolist() == [9, 8, 9, 9, 9, 8, 8]
assert abs(probability.sum().item() - 7) < 1e-6
assert abs(probability[chars.index("貓"), chars.index("看")].item() - float(Fraction(3, 9))) < 1e-6
assert probability.requires_grad is False
print("manually inventoried transitions:", sorted((a + "→" + b, n) for (a, b), n in golden.items()))
print("characters=12, transitions=11, vocabulary=7; all smoothed cells total=60")
print("per-row denominators:", counts.sum(dim=1).tolist())
print("denominator shape:", list(counts.sum(dim=1, keepdim=True).shape))
print("row probability sums:", probability.sum(dim=1).tolist(), "matrix probability sum:", probability.sum().item())

# Run the prescribed exercise using the actual original source, changing only the initializer.
exercise_code = original.read_text().replace(
    "counts = torch.ones(len(chars), len(chars))",
    "counts = torch.full((len(chars), len(chars)), 0.1)",
)
(root / "exercise-original-with-0_1.py").write_text(exercise_code)
exercise = {"__name__": "__main__"}
exec(compile(exercise_code, "exercise-original-with-0_1.py", "exec"), exercise)
cat, look = chars.index("貓"), chars.index("看")
denom_01 = exercise["counts"][cat].sum().item()
p_01 = exercise["probability"][cat, look].item()
assert abs(denom_01 - 2.7) < 1e-6
assert abs(p_01 - float(Fraction(21, 27))) < 1e-6
assert p_01 > probability[cat, look].item()
print("alpha=0.1 cat denominator:", denom_01, "p(look|cat):", p_01, "exact:", str(Fraction(21, 27)))

# A denominator-axis negative control: dropping keepdim broadcasts a (7,) vector across columns.
wrong = counts / counts.sum(dim=1)
assert not torch.allclose(wrong.sum(dim=1), torch.ones(7))
print("without keepdim, row sums:", wrong.sum(dim=1).tolist(), "normalization check:", False)

# Appending a final cat adds period→cat but no cat→anything; outgoing counts, not raw occurrence counts.
end_code = original.read_text().replace('text = "貓看狗。狗吃魚。貓看鳥。"', 'text = "貓看狗。狗吃魚。貓看鳥。貓"')
(root / "variation-terminal-cat.py").write_text(end_code)
endpoint = {"__name__": "__main__"}
exec(compile(end_code, "variation-terminal-cat.py", "exec"), endpoint)
assert endpoint["text"].count("貓") == 3
assert torch.equal(endpoint["counts"][cat], counts[cat])
assert endpoint["counts"][chars.index("。"), cat].item() == counts[chars.index("。"), cat].item() + 1
print("appended terminal cat: raw cats=3, cat outgoing observations=2, cat denominator=9")

# Context-free comparisons on precisely specified denominators, for the opening sentence only.
next_marginal = Fraction(sum(b == "看" for _, b in pairs), len(pairs))
all_char_marginal = Fraction(text.count("看"), len(text))
assert Fraction(3, 9) > next_marginal and Fraction(3, 9) > all_char_marginal
print("context-free look proportions:", str(next_marginal), "over the 11 next-position targets;", str(all_char_marginal), "over all 12 chars")
print("PASS: original, manual transition inventory, alpha exercise, denominator-axis control, endpoint change")
