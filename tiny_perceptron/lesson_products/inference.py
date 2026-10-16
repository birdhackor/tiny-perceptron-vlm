"""三個有限入口：B解題、C機率判斷，以及保留手動檔位的規則Auto。"""

import hashlib
import json
import re
import time
from pathlib import Path

import torch

from .data import CANDIDATES, MEANINGS
from .model import CandidateScorer, Config, LoopedSolver, Tokenizer, padded, validate_candidates

EFFORT = {"low": 1, "medium": 3, "high": 5}


def parse_state(state):
    """驗證公開有限語法及值域；不計算答案，也不替C判斷資訊是否充分。"""
    if not isinstance(state, str) or len(state) > 256:
        raise ValueError("state需為至多256字的公開有限語法")
    boxes = re.fullmatch(r"(?:兩盒追蹤|盒子狀態);([AP])=([0-7?]);([BQ])=([0-7?]);事件=(.*);查詢=([ABPQ?])", state)
    if boxes:
        a, _, b, _, events, query = boxes.groups()
        if (a, b) not in (("A", "B"), ("P", "Q")) or query not in (a, b, "?"):
            raise ValueError("兩盒名稱及查詢需一致")
        parts = [] if events == "?" else events.split(",")
        if events != "?" and (
            not 1 <= len(parts) <= 6 or any(p != "交換" and re.fullmatch(rf"[{a}{b}]設[0-7]", p) is None for p in parts)
        ):
            raise ValueError("事件需為1–6個盒子覆寫或交換")
        return {"task": "boxes", "event_count": len(parts), "stress_length": len(parts) > 4}
    if re.fullmatch(r"(?:三數加法|整數加總);x=[0-6?];y=[0-6?];z=[0-6?]", state):
        return {"task": "sum"}
    if re.fullmatch(r"(?:卡片進度|查卡片目前進度);申請編號=(?:[0-9]{6}|\?)", state):
        return {"task": "card"}
    if state in ("地址填寫規則", "我想瞭解地址怎麼填"):
        return {"task": "address"}
    if re.fullmatch(r"(?:App重設密碼|App密碼忘記了);裝置=(?:手機|平板|電腦|\?)", state):
        return {"task": "app"}
    if state in ("天氣預測", "幫我預報明天天氣"):
        return {"task": "none"}
    raise ValueError("超出公開有限語法；這個示範不接受任意聊天文字")


def check_manifest(directory, product):
    directory = Path(directory)
    manifest = json.loads((directory / "artifact-manifest.json").read_bytes())
    if manifest.get("product") != product or manifest.get("version") != 1:
        raise ValueError("成品manifest種類或版本不同")
    required = {
        "model.pt",
        "config.json",
        "tokenizer.json",
        "calibration.json" if product == "c" else "auto-policy.json",
    }
    if not required <= set(manifest["files"]):
        raise ValueError("manifest缺少權重／設定／tokenizer")
    for name in ("calibration.json", "auto-policy.json"):
        if (directory / name).exists() and name not in manifest["files"]:
            raise ValueError("存在未被manifest綁定的推論設定：" + name)
    for name, expected in manifest["files"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("manifest檔名需在成品目錄內")
        path = directory / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("成品檔案指紋不符：" + name)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError("成品檔案大小不符：" + name)
    return manifest


class Product:
    def __init__(self, directory, product, device="cpu", verify=True):
        if product not in ("b", "c"):
            raise ValueError("需指定獨立的B或C成品")
        self.directory, self.product = Path(directory), product
        self.manifest = check_manifest(directory, product) if verify else None
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.load(self.directory / "tokenizer.json")
        self.config = Config(**json.loads((self.directory / "config.json").read_bytes()))
        if self.config.vocab_size != len(self.tokenizer):
            raise ValueError("tokenizer與模型維度不符")
        cls = LoopedSolver if product == "b" else CandidateScorer
        self.model = cls(self.config).to(self.device).eval()
        self.model.load_state_dict(torch.load(self.directory / "model.pt", map_location=self.device, weights_only=True))
        path = self.directory / "calibration.json"
        self.calibration = (
            json.loads(path.read_bytes()) if path.is_file() else {"temperatures": {q: 1.0 for q in CANDIDATES}}
        )
        path = self.directory / "auto-policy.json"
        self.auto_policy = json.loads(path.read_bytes()) if path.is_file() else None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def solve(self, state, loops=3):
        if self.product != "b":
            raise ValueError("solve需載入B權重")
        parsed = parse_state(state)
        if parsed["task"] not in ("boxes", "sum") or "?" in state:
            raise ValueError("B只解資訊完整的兩盒／三數有限題目")
        self.synchronize()
        started = time.perf_counter()
        tokens = padded([self.tokenizer.encode(state)], self.config.max_length, self.device)
        probabilities = self.model(tokens, loops=loops)[0].softmax(-1)
        predicted = int(probabilities.argmax())
        self.synchronize()
        return {
            "answer": predicted,
            "loops": loops,
            "seconds": time.perf_counter() - started,
            "stress_length": parsed.get("stress_length", False),
        }

    @torch.inference_mode()
    def judge(self, state, question="choice", candidates=None):
        if self.product != "c":
            raise ValueError("judge需載入C權重")
        parse_state(state)
        candidates = list(CANDIDATES.get(question, ())) if candidates is None else list(candidates)
        validate_candidates(question, candidates)
        temperature = self.calibration["temperatures"][question]
        if not 0 < temperature <= 100:
            raise ValueError("不合法校準溫度")
        self.synchronize()
        started = time.perf_counter()
        tokens = padded(
            [self.tokenizer.encode(state, question, c) for c in candidates], self.config.max_length, self.device
        )
        scores = self.model(tokens).float() / temperature
        probabilities = scores.softmax(-1).cpu().tolist()
        # 相同分數用意義ID排序，避免argmax在平手時偷偷依清單位置選答案。
        chosen = min(range(len(candidates)), key=lambda i: (-probabilities[i], candidates[i]))
        n = len(candidates)
        self.synchronize()
        return {
            "question": question,
            "selected": candidates[chosen],
            "probabilities": {c: p for c, p in zip(candidates, probabilities, strict=True)},
            "meanings": {c: MEANINGS[c] for c in candidates},
            "temperature": temperature,
            "concentration": (n * max(probabilities) - 1) / (n - 1),
            "concentration_scope": "候選間集中程度；不是答對機率。候選集合改變時需重新計算。",
            "seconds": time.perf_counter() - started,
        }


def rule_depth(policy, task, event_count=0):
    if policy.get("version") != 1:
        raise ValueError("需要凍結的Auto規則版本")
    if task == "sum":
        depth = policy["sum_depth"]
    else:
        depth = 1 if event_count <= policy["low_max_events"] else 3 if event_count <= policy["medium_max_events"] else 5
    if depth not in (1, 3, 5):
        raise ValueError("Auto規則只能選1／3／5輪")
    return depth


def auto_solve(b, c, state, effort="auto", max_loops=5):
    if effort not in (*EFFORT, "auto") or type(max_loops) is not int or not 1 <= max_loops <= 6:
        raise ValueError("effort需為auto/low/medium/high，硬上限需為1–6")
    parsed = parse_state(state)
    b.synchronize()
    c.synchronize()
    started = time.perf_counter()
    choice = c.judge(state, "choice")
    info = c.judge(state, "info")
    result = {"effort": effort, "max_loops": max_loops, "choice": choice, "info": info, "b_loops": 0}
    if choice["selected"] not in ("boxes", "sum"):
        result["status"] = "unsupported"
    elif info["selected"] != "yes":
        result["status"] = "clarification_needed"
    else:
        if effort == "auto" and b.auto_policy is None:
            raise ValueError("Auto需先載入policy集合選定並凍結的規則")
        requested = (
            EFFORT[effort]
            if effort != "auto"
            else rule_depth(b.auto_policy, choice["selected"], parsed.get("event_count", 0))
        )
        actual = min(requested, max_loops)
        result.update(requested_loops=requested, b_loops=actual)
        try:
            result["solution"] = b.solve(state, loops=actual)
            result["status"] = "budget_limited" if actual < requested else "answered"
        except ValueError as error:
            result.update(status="input_contract_rejected", b_loops=0, reason=str(error))
    result["seconds"] = time.perf_counter() - started
    result["c_seconds"] = choice["seconds"] + info["seconds"]
    return result
