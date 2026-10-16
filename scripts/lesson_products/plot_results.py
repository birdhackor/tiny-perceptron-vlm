"""由最後報告畫配對機制曲線；不讀資料、不重算或挑選測試結果。"""

import argparse
import hashlib
import json
from pathlib import Path


def plot(report_path, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_path = Path(report_path)
    report = json.loads(report_path.read_bytes())
    plt.rcParams["svg.hashsalt"] = "tiny-perceptron-looped-mechanism-v1"
    plt.rcParams.update({"font.size": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    figure, axes = plt.subplots(2, 1, figsize=(5, 6.8), sharex=True, layout="constrained")
    for mode, label, color in (
        ("fixed3", "fixed terminal depth 3", "#175a97"),
        ("odd", "sampled terminal depth 1/3/5", "#b34813"),
    ):
        rows = report["mechanisms"][mode]["evaluation"]
        values = [rows[str(depth)]["boxes"] for depth in range(1, 7)]
        axes[0].plot(range(1, 7), [x["cross_entropy"] for x in values], marker="o", label=label, color=color)
        axes[1].plot(range(1, 7), [100 * x["accuracy"] for x in values], marker="o", color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cross-entropy\n(log scale; lower is better)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Inference loops")
    axes[1].set_xticks(range(1, 7))
    axes[1].set_ylim(0, 102)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="best", frameon=False, fontsize=10.5)
    figure.suptitle("Same initialization and 1,500 updates\nHeldout box tracking", fontsize=13)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Date": None, "Creator": "Tiny Perceptron; plot_results.py"})
    plt.close(figure)
    return {
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "figure_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(plot(options.report, options.output)))


if __name__ == "__main__":
    main()
