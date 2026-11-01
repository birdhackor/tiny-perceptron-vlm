"""讀取本機 bundle 與 JSON 請求，輸出決策建議及拒答原因。"""

import argparse
import json
import sys
from pathlib import Path

import torch

from .service import Action, DecisionEngine, Device


def predict_request(engine, request):
    if type(request) is not dict or set(request) - {"devices", "target", "kind", "language", "candidates"}:
        raise ValueError("請求必須是物件，且不能含未知欄位")
    if not {"devices", "target"} <= set(request) or type(request["devices"]) is not list:
        raise ValueError("請求須提供 devices 清單及 target")
    devices = []
    for item in request["devices"]:
        if type(item) is not dict or set(item) != {"id", "task", "permission", "ready"}:
            raise ValueError("每個裝置須明列 id/task/permission/ready，未知欄位請用 null")
        devices.append(Device(**item))
    kind = request.get("kind", "choice")
    if type(kind) is not str or kind not in {"choice", "info", "rubric", "all"}:
        raise ValueError("kind 僅支援 choice/info/rubric/all")
    if "candidates" in request and kind not in {"choice", "all"}:
        raise ValueError("candidates 只適用於動作選擇")
    candidates = None
    if "candidates" in request:
        if type(request["candidates"]) is not list:
            raise ValueError("candidates 必須是清單")
        candidates = []
        for item in request["candidates"]:
            if type(item) is not dict or set(item) != {"id", "device", "action"}:
                raise ValueError("候選須明列 id/device/action")
            candidates.append(Action(**item))
    session = engine.session(devices, request.get("language", "zh"))
    target = request["target"]
    results = {}
    if kind in {"choice", "all"}:
        results["choice"] = session.choose(target, candidates).to_dict()
    if kind in {"info", "all"}:
        results["info"] = session.information(target).to_dict()
    if kind in {"rubric", "all"}:
        results["rubric"] = session.grade(target).to_dict()
    return {"target": target, "results": results}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="已驗收 bundle 目錄")
    parser.add_argument("--request", required=True, help="JSON 請求檔案")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads 須大於0")
    torch.set_num_threads(args.threads)
    try:
        engine = DecisionEngine(args.bundle, args.device)
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = predict_request(engine, request)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"無法完成判斷：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
