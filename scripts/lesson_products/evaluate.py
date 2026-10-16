"""先freeze開發設定，再另次命令執行最後test；不提供以test選參數的模式。"""

import argparse
import json
from pathlib import Path

from tiny_perceptron.lesson_products.evaluation import evaluate_frozen, select_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "test"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--fixed3", type=Path, required=True)
    parser.add_argument("--odd135", type=Path, required=True)
    parser.add_argument("--settings-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    options = parser.parse_args()
    mechanism = {"fixed3": options.fixed3, "odd": options.odd135}
    if options.stage == "freeze":
        result = select_settings(options.data, options.b, options.c, options.settings_freeze, options.device, mechanism)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if options.output is None:
            parser.error("最後test需要另立--output原始結果，不覆寫既有檔案")
        result = evaluate_frozen(
            options.settings_freeze, options.data, options.b, options.c, options.output, mechanism, options.device
        )
        print(
            json.dumps(
                {"gates": result["gates"], "output": str(options.output), "elapsed_seconds": result["elapsed_seconds"]},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
