"""Reader's optional two-cell exercise, not a repository benchmark.

Chosen drawing: each strip is 4 by 8 pixels, inside an 8 by 8 cell.
Move the red strip to a second, initially black cell; leave blue fixed.
"""
import json

red = [1.0, 0.0, 0.0]
blue = [0.0, 0.0, 1.0]
black = [0.0, 0.0, 0.0]

before = [
    [red[:] for _ in range(32)] + [blue[:] for _ in range(32)],
    [black[:] for _ in range(64)],
]
after = [
    [black[:] for _ in range(32)] + [blue[:] for _ in range(32)],
    [red[:] for _ in range(32)] + [black[:] for _ in range(32)],
]

def means(cells):
    return [[sum(pixel[channel] for pixel in cell) / len(cell)
             for channel in range(3)] for cell in cells]

before_means = means(before)
after_means = means(after)
print(json.dumps({
    "scope": "reader-created paper-exercise arithmetic, chosen 4x8 strips",
    "before_cell_RGB_means": before_means,
    "after_cell_RGB_means": after_means,
    "cell_summaries_still_equal": before_means == after_means,
}, ensure_ascii=False, indent=2))
