"""python -m tiny_perceptron.lesson_products --help"""

import argparse
import json
from pathlib import Path

import torch

from .delivery import download_product, serve
from .inference import EFFORT, Product, auto_solve


def main():
    parser = argparse.ArgumentParser(description="有限世界B/C成品；不使用外部預訓練權重")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("solve", "judge", "auto", "serve"):
        sub = commands.add_parser(name)
        sub.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
        if name in ("solve", "auto", "serve"):
            sub.add_argument("--b", type=Path, default=Path("outputs/lesson-products/b"))
        if name in ("judge", "auto", "serve"):
            sub.add_argument("--c", type=Path, default=Path("outputs/lesson-products/c"))
        if name != "serve":
            sub.add_argument("--state", required=True)
        if name == "solve":
            sub.add_argument("--loops", type=int, default=3, choices=range(1, 7))
        elif name == "judge":
            sub.add_argument("--question", choices=("choice", "info", "rubric"), default="choice")
            sub.add_argument("--candidates", help="逗號分隔的候選意義ID；省略時使用該question完整集合")
        elif name == "auto":
            sub.add_argument("--effort", choices=(*EFFORT, "auto"), default="auto")
            sub.add_argument("--max-loops", type=int, default=5)
        else:
            sub.add_argument("--port", type=int, default=8765)
    sub = commands.add_parser("download")
    sub.add_argument("--product", choices=("b", "c"), required=True)
    sub.add_argument("--destination", type=Path)
    sub.add_argument("--release", type=Path, default=Path(__file__).with_name("release.json"))
    options = parser.parse_args()
    torch.set_num_threads(2)
    if options.command == "download":
        release = json.loads(options.release.read_bytes())
        entry = release["products"][options.product]
        result = download_product(
            release["repo"],
            release["revision"],
            entry["prefix"],
            entry["manifest_sha256"],
            options.destination or Path("outputs/lesson-products") / options.product,
            options.product,
        )
    elif options.command == "serve":
        serve(options.b, options.c, options.port, options.device)
        return
    elif options.command == "solve":
        result = Product(options.b, "b", options.device).solve(options.state, options.loops)
    elif options.command == "judge":
        result = Product(options.c, "c", options.device).judge(
            options.state, options.question, options.candidates.split(",") if options.candidates else None
        )
    else:
        result = auto_solve(
            Product(options.b, "b", options.device),
            Product(options.c, "c", options.device),
            options.state,
            options.effort,
            options.max_loops,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
